"""
NV PreComp Warp - Aligns closed-model character render to source actor pose via tracked-point similarity transform.

Designed as the alignment layer for the "rough pre-composite + low-denoise VACE refinement" workflow:
  1. Generate target character via Kling/Seedance (scene-matched, identity-correct, but pose-imperfect)
  2. Track torso anchors on source actor (NV_PointPicker -> NV_CoTrackerTrajectoriesJSON) -> source_traj [T, N, 2]
  3. Track SAME anchors on render too (NV_PointPicker -> NV_CoTrackerTrajectoriesJSON) -> render_traj [T, N, 2]
     The render is itself a moving video — a static frame-0 reference cannot align a moving render to a moving source.
  4. NV_PreCompWarp solves per-frame 4-DOF similarity transform mapping source[t] -> render[t]
  5. Outputs warped masked render ready for paste over source for VACE V2V refinement (fill_mode='none')

Architecture decisions (validated by 2026-04-30 multi-AI debate + research synthesis):
  - 4-DOF SIMILARITY transform (tx, ty, uniform scale, rotation) - preserves aspect, no shear
    Citation: Umeyama 1991 "Least-squares estimation of transformation parameters between two point patterns"
    Implementation: cv2.estimateAffinePartial2D (RANSAC-robust)
  - Savitzky-Golay smoothing on parameters (zero-phase, preserves motion peaks; offline pipeline)
    Smooth scale in log-space; rotation after np.unwrap to handle the -pi/+pi discontinuity
  - Erode-warp-feather edge pipeline:
    1. Erode mask 1-2 px before warp (kills bg color contamination from render edges)
    2. Premultiplied RGB+alpha warp (prevents bg color bleed into edges via interpolation)
    3. INTER_CUBIC for RGB (sharpness), INTER_LINEAR for mask
    4. Gaussian feather 2-4 px post-warp (gives VACE smooth gradient to blend at low denoise)
  - Failure detection: RMSE residual after fit > threshold fraction of torso diagonal
    Fallback: translation-only fit (drop scale+rotation) + flag for downstream crop_expand_px inflation
  - Anchors: torso-centric (head/neck, L/R shoulders, L/R hips) - most stable across motion blur
"""

import json
from typing import Optional, Tuple

import numpy as np
import torch
import cv2
from scipy.signal import savgol_filter


class NV_PreCompWarp:
    """Align closed-model character render to source actor pose via similarity-transform pre-comp."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "render_video": ("IMAGE", {
                    "tooltip": "The closed-model output (Kling/Seedance) showing the target character. "
                               "Should be approximately scene-matched to source. Will be warped per-frame to align."
                }),
                "render_mask": ("MASK", {
                    "tooltip": "SAM3 silhouette mask of the target character extracted from the render. "
                               "Background pixels = 0; character pixels = 1."
                }),
                "source_trajectories": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "JSON of CoTracker3-tracked anchor trajectories on the source actor. "
                               "Format: shape [T, N, 2] where T=frames, N=points, last dim=(x, y) in pixels. "
                               "Either a JSON array directly, or {'shape': [T, N, 2], 'data': [...], 'names': [...]}."
                }),
                "render_trajectories": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "JSON of CoTracker3-tracked anchor trajectories on the render (Kling/Seedance) video, "
                               "in SAME ORDER and SAME COUNT as source_trajectories. "
                               "Format: shape [T, N, 2] where T=frames, N=points, last dim=(x, y) in pixels. "
                               "Either a JSON array directly, or {'shape': [T, N, 2], 'data': [...], 'names': [...]}. "
                               "The render must be tracked too — a static frame-0 reference cannot align a moving render to a moving source."
                }),
                "smooth_window": ("INT", {
                    "default": 11, "min": 3, "max": 51, "step": 2,
                    "tooltip": "Savitzky-Golay smoothing window (must be odd). Larger = smoother, less responsive. "
                               "11-15 typical for 24fps action shots. Auto-clamped to T-1 if larger than the sequence."
                }),
                "smooth_polyorder": ("INT", {
                    "default": 2, "min": 1, "max": 5,
                    "tooltip": "Savitzky-Golay polynomial order. 2 preserves motion peaks well; 3-4 for very smooth."
                }),
                "edge_erode_px": ("INT", {
                    "default": 2, "min": 0, "max": 8,
                    "tooltip": "Erode mask before warp to kill background-color contamination from render edges. "
                               "1-2 px standard. Higher (4-6) if render bg is high-contrast (green-screen artifacts)."
                }),
                "edge_feather_px": ("FLOAT", {
                    "default": 3.0, "min": 0.0, "max": 16.0, "step": 0.5,
                    "tooltip": "Gaussian feather radius after warp. 2-4 px gives VACE smooth gradient to blend. "
                               "0 = hard edge."
                }),
                "rmse_threshold_frac": ("FLOAT", {
                    "default": 0.10, "min": 0.01, "max": 0.5, "step": 0.01,
                    "tooltip": "Per-frame failure threshold: RMSE residual after fit > this fraction of torso diagonal "
                               "triggers fallback. 0.08-0.12 typical."
                }),
                "fallback_mode": (["smoothed_interp", "translation_only", "hold_last_good", "bypass_pass_through"], {
                    "default": "smoothed_interp",
                    "tooltip": "What to do on high-residual frames AFTER smoothing has already filled them via interpolation.\n"
                               "  smoothed_interp (default, recommended): trust the smoother+interpolation result, no override. "
                               "Best for action shots where rotation/scale matter and the interpolation is plausible.\n"
                               "  translation_only: zero scale/rotation, keep only tx/ty. Conservative; forces pure translation. "
                               "Wrong choice for shots where the character rotates or changes scale during failure stretches.\n"
                               "  hold_last_good: step-hold the full smoothed transform of the last good frame across the failure gap. "
                               "Good for 'mostly static pose with periodic tracker dropouts'.\n"
                               "  bypass_pass_through: no warp at all on failure frames; output raw render. Most conservative."
                }),
                "visibility_aware_solve": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "When ON: per-frame solver uses CoTracker3 visibility scores to weight (or filter) "
                               "anchors. Bad anchors at frame t no longer poison that frame's fit. Dispatch by "
                               "active-anchor count: >=3 → weighted Umeyama; 2 → exact 2-point with baseline+residual "
                               "checks; 1 → translation-only relative to prev frame's transform; 0 → existing fallback. "
                               "Continuous weighting eliminates anchor-set crossing-threshold jitter that hard-filter "
                               "+ smoother can leave as 1-3px sustained centroid bias. When OFF: legacy cv2 RANSAC "
                               "with all anchors at every frame (back-compat path). Default ON — strict improvement."
                }),
                "quality_floor": ("FLOAT", {
                    "default": 0.20, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Hard floor for the visibility-aware solve. Anchors with quality (vis_src * vis_render) "
                               "below this are dropped from the per-frame fit. Default 0.20 — kills clearly dead "
                               "anchors only; legitimate borderline anchors still contribute via continuous weighting."
                }),
                "quality_gamma": ("FLOAT", {
                    "default": 2.0, "min": 0.5, "max": 4.0, "step": 0.1,
                    "tooltip": "Weight-shaping exponent for visibility-aware solver. weights = ((q - q_floor) / "
                               "(1 - q_floor)) ** gamma. gamma=1 is linear; gamma>1 sharpens (high-quality anchors "
                               "dominate); gamma<1 softens. Default 2.0 — moderate sharpening favors confident anchors."
                }),
                "min_baseline_px": ("FLOAT", {
                    "default": 24.0, "min": 4.0, "max": 200.0, "step": 1.0,
                    "tooltip": "Minimum anchor separation (pixels) for the 2-anchor exact-similarity case. "
                               "Two-point similarity is mathematically valid but unstable at small baselines — small "
                               "tracker noise on close anchors amplifies into large rotation/scale jitter. "
                               "Default 24 px (multi-AI consensus). Sparse-shot mode (N<=6) ALSO requires "
                               "baseline >= sparse_shot_span_frac × frame-0 anchor spread."
                }),
                "sparse_shot_span_frac": ("FLOAT", {
                    "default": 0.10, "min": 0.0, "max": 0.5, "step": 0.01,
                    "tooltip": "For sparse shots (nominal anchor count N <= 6), 2-anchor case requires baseline >= "
                               "this fraction of the frame-0 object span (max pairwise distance). Set 0 to disable. "
                               "Default 0.10 — losing 2 of 5 anchors is a support collapse; tightening prevents "
                               "shaky fits from accidentally close survivors."
                }),
                "short_gap_max": ("INT", {
                    "default": 3, "min": 0, "max": 10,
                    "tooltip": "Hybrid Strategy E: per-anchor visibility gaps <= this length are bridged by linear "
                               "interpolation (motion blur flicker); longer gaps are hard-filtered (true occlusion). "
                               "Default 3 frames (~125ms at 24fps). Set 0 to disable interpolation entirely "
                               "(pure hard-filter). Reuses CoTracker's per-anchor visibility from trajectory JSON."
                }),
                "verbose_debug": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Print per-frame diagnostics (transform params, residuals, fallback frames)."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "STRING")
    RETURN_NAMES = ("warped_render", "warped_mask", "warped_render_full", "alignment_info")
    FUNCTION = "execute"
    CATEGORY = "NV_Utils/Inpaint"
    DESCRIPTION = (
        "Per-frame similarity transform (4-DOF: translation + uniform scale + rotation) from tracked source "
        "and render trajectories. Warps the masked render at frame t to align with the source actor at frame t. "
        "Designed for the pre-composite + low-denoise VACE refinement workflow."
    )

    # =========================================================================
    # Main execute
    # =========================================================================

    def execute(self, render_video, render_mask, source_trajectories, render_trajectories,
                smooth_window, smooth_polyorder, edge_erode_px, edge_feather_px,
                rmse_threshold_frac, fallback_mode,
                visibility_aware_solve, quality_floor, quality_gamma,
                min_baseline_px, sparse_shot_span_frac, short_gap_max,
                verbose_debug):

        # --- Step 1: Parse inputs ---
        # _parse_trajectories now returns (positions, visibility_or_None). Visibility is present
        # when the input JSON came from NV_CoTrackerTrajectoriesJSON (which emits a `visibility`
        # field in the dict form). Raw [T,N,2] arrays don't carry visibility — visibility_aware
        # mode falls back to all-anchors-equal-weight in that case (effectively legacy behavior).
        source_traj, source_vis = self._parse_trajectories(source_trajectories)  # [T_src, N, 2], [T_src, N] | None
        render_traj, render_vis = self._parse_trajectories(render_trajectories)  # [T_ren, N, 2], [T_ren, N] | None

        T_render, H, W, C = render_video.shape
        T_track, N, _ = source_traj.shape
        T_render_track, N_render, _ = render_traj.shape
        T_mask = render_mask.shape[0]

        if N != N_render:
            raise ValueError(
                f"[NV_PreCompWarp] Anchor count mismatch: source has {N} points, "
                f"render has {N_render}. Must be equal and in same order."
            )
        if N < 2:
            raise ValueError(f"[NV_PreCompWarp] Need at least 2 anchors for similarity fit, got {N}.")
        if T_render < 1:
            raise ValueError("[NV_PreCompWarp] render_video has zero frames")
        if T_mask < 1:
            raise ValueError("[NV_PreCompWarp] render_mask has zero frames")
        if T_track < 1:
            raise ValueError("[NV_PreCompWarp] source_trajectories has zero frames")
        if T_render_track < 1:
            raise ValueError("[NV_PreCompWarp] render_trajectories has zero frames")

        if T_track != T_render or T_render_track != T_render:
            print(f"[NV_PreCompWarp] WARNING: source T={T_track}, render_traj T={T_render_track}, "
                  f"render_video T={T_render}. Output truncated to min(T) = "
                  f"{min(T_render, T_track, T_render_track)}.")
        T = min(T_render, T_track, T_render_track)

        # Use render's frame-0 anchor spread as the torso scale reference for RMSE threshold.
        # (Per-frame torso scale would also work but adds noise; frame-0 is a stable reference.)
        torso_diag = self._compute_torso_diag(render_traj[0])
        if verbose_debug:
            print(f"[NV_PreCompWarp] T={T}, N={N} anchors, torso_diag={torso_diag:.1f}px")

        # --- Step 2: Solve per-frame similarity transforms ---
        # When visibility_aware_solve is ON: dispatch by per-frame active anchor count, weighted
        # Umeyama for >=3 active, exact 2-point for 2 active w/ baseline check, translation-only
        # relative to prev M for 1 active, fail otherwise. Continuous weighting eliminates the
        # anchor-set crossing-threshold jitter that pure hard-filter + smoother leaves as 1-3 px
        # sustained centroid bias (multi-AI debate R4 Codex argument).
        # When OFF: legacy cv2 RANSAC with all anchors at every frame (back-compat path).
        # All emitted M's use cv2.WARP_INVERSE_MAP convention (multi-AI review R1 CRIT #1).
        params = np.zeros((T, 4), dtype=np.float64)  # [tx, ty, log_s, theta]
        residuals = np.zeros(T, dtype=np.float64)
        fit_ok = np.zeros(T, dtype=bool)
        # Per-frame active anchor count + branch label, surfaced in alignment_info for diagnostics
        active_count_per_frame = np.zeros(T, dtype=np.int64)
        branch_per_frame = ["unset"] * T  # "weighted_umeyama" / "two_point" / "translation_only" / "fail" / "legacy_ransac"

        # Combine source + render visibility into a single per-frame per-anchor quality signal.
        # If either trajectory lacks visibility (e.g., raw [T,N,2] JSON), fall back to all-equal.
        # Also apply hybrid Strategy E: short-gap visibility flicker gets bumped to 0.5 (low-trust
        # but active); long gaps stay low and get hard-filtered by quality_floor.
        if visibility_aware_solve and source_vis is not None and render_vis is not None:
            # Truncate to T frames + slice to N anchors (defensive — trajectory shapes already validated)
            src_v = source_vis[:T, :N]
            ren_v = render_vis[:T, :N]
            src_v = self._interpolate_visibility_short_gaps(src_v, int(short_gap_max))
            ren_v = self._interpolate_visibility_short_gaps(ren_v, int(short_gap_max))
            quality_per_frame = np.clip(src_v * ren_v, 0.0, 1.0)  # [T, N]
        elif visibility_aware_solve:
            # Visibility-aware mode requested but trajectories don't carry visibility — degrade
            # to all-anchors-equal-weight (effectively gives weighted Umeyama with uniform weights,
            # which still has the smooth-weighting property but no per-anchor quality signal).
            print("[NV_PreCompWarp] visibility_aware_solve=ON but trajectories lack visibility "
                  "field; using uniform weights (each anchor weight=1.0).")
            quality_per_frame = np.ones((T, N), dtype=np.float64)
        else:
            quality_per_frame = None  # legacy mode signal

        # Sparse-shot baseline floor: for shots where the nominal anchor count is small, the
        # 2-anchor case requires a baseline larger than just min_baseline_px to avoid catastrophic
        # support collapse. Computed from the frame-0 anchor spread on render side (stable scale).
        if N <= 6 and sparse_shot_span_frac > 0.0:
            sparse_baseline_min = max(float(min_baseline_px), float(sparse_shot_span_frac) * torso_diag)
        else:
            sparse_baseline_min = float(min_baseline_px)

        # Effective stricter residual gate for 2-anchor case (multi-AI consensus: 0.75x normal)
        rmse_threshold_px_2pt = 0.75 * (rmse_threshold_frac * torso_diag)

        # Track previous fit_ok index for 1-anchor translation-only fallback
        last_good_idx_for_oneanchor = -1

        for t in range(T):
            from_pts_full = source_traj[t]      # [N, 2]
            to_pts_full = render_traj[t]         # [N, 2]

            if not visibility_aware_solve:
                # Legacy cv2 RANSAC path
                tx, ty, log_s, theta, residual, ok = self._solve_similarity(from_pts_full, to_pts_full)
                params[t] = [tx, ty, log_s, theta]
                residuals[t] = residual
                fit_ok[t] = ok
                active_count_per_frame[t] = N if ok else 0
                branch_per_frame[t] = "legacy_ransac"
                if ok:
                    last_good_idx_for_oneanchor = t
                continue

            # Visibility-aware dispatch
            q_t = quality_per_frame[t]
            active_mask = q_t >= float(quality_floor)
            active_idx = np.where(active_mask)[0]
            n_active = int(active_idx.size)
            active_count_per_frame[t] = n_active

            if n_active >= 3:
                # Weighted Umeyama on active anchors
                src_active = from_pts_full[active_idx]
                dst_active = to_pts_full[active_idx]
                q_active = q_t[active_idx]
                # Weight shaping: ((q - q_floor) / (1 - q_floor))^gamma. q_floor frames get
                # weight 0; q=1 frames get weight 1. gamma>1 sharpens dominance of strong anchors.
                denom = max(1.0 - float(quality_floor), 1e-6)
                w = np.power(np.clip((q_active - float(quality_floor)) / denom, 0.0, 1.0),
                             float(quality_gamma))
                tx, ty, log_s, theta, residual, ok = self._weighted_umeyama_2d(src_active, dst_active, w)
                params[t] = [tx, ty, log_s, theta]
                residuals[t] = residual
                fit_ok[t] = ok
                branch_per_frame[t] = "weighted_umeyama"
                if ok:
                    last_good_idx_for_oneanchor = t

            elif n_active == 2:
                # Exact 2-point similarity with baseline + stricter residual gates
                src_active = from_pts_full[active_idx]
                dst_active = to_pts_full[active_idx]
                baseline = float(np.linalg.norm(src_active[0] - src_active[1]))
                if baseline >= sparse_baseline_min:
                    tx, ty, log_s, theta, residual, ok = self._solve_2point_similarity(src_active, dst_active)
                    if ok:
                        # Stricter residual gate for 2-anchor: residual is 0 by construction on
                        # these 2 points, so verify fit against the FULL set's good anchors at
                        # this frame to catch "the 2 surviving anchors agreed but everything else
                        # is wildly off" — we promote ok=False if so.
                        # Compute residual against any other anchors with q > 0.05 (low bar):
                        check_mask = (q_t > 0.05) & ~active_mask
                        if check_mask.any():
                            cs = from_pts_full[check_mask]
                            cd = to_pts_full[check_mask]
                            scale = np.exp(log_s)
                            cos_t, sin_t = np.cos(theta) * scale, np.sin(theta) * scale
                            M_check = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float64)
                            transformed = (M_check @ cs.T).T + np.array([tx, ty])
                            check_err = np.linalg.norm(transformed - cd, axis=1)
                            check_rmse = float(np.sqrt(np.mean(check_err ** 2)))
                            if check_rmse > rmse_threshold_px_2pt:
                                ok = False
                                residual = check_rmse
                        params[t] = [tx, ty, log_s, theta]
                        residuals[t] = residual
                        fit_ok[t] = ok
                        branch_per_frame[t] = "two_point" if ok else "two_point_failed_check"
                        if ok:
                            last_good_idx_for_oneanchor = t
                    else:
                        params[t] = [0.0, 0.0, 0.0, 0.0]
                        residuals[t] = np.inf
                        fit_ok[t] = False
                        branch_per_frame[t] = "two_point_solve_failed"
                else:
                    # Baseline too small — refuse to fit
                    params[t] = [0.0, 0.0, 0.0, 0.0]
                    residuals[t] = np.inf
                    fit_ok[t] = False
                    branch_per_frame[t] = "two_point_baseline_short"

            elif n_active == 1:
                # Translation-only relative to previous good frame's transform.
                # Lock scale + rotation to prev frame; only update tx, ty so the single active
                # anchor maps correctly under the locked rotation/scale.
                if last_good_idx_for_oneanchor >= 0:
                    tx_p, ty_p, log_s_p, theta_p = params[last_good_idx_for_oneanchor]
                    scale_p = float(np.exp(log_s_p))
                    cos_p, sin_p = np.cos(theta_p) * scale_p, np.sin(theta_p) * scale_p
                    R_scaled = np.array([[cos_p, -sin_p], [sin_p, cos_p]], dtype=np.float64)
                    src_pt = from_pts_full[active_idx[0]]
                    dst_pt = to_pts_full[active_idx[0]]
                    new_t = dst_pt - R_scaled @ src_pt
                    params[t] = [float(new_t[0]), float(new_t[1]), float(log_s_p), float(theta_p)]
                    residuals[t] = 0.0  # exact fit on the 1 anchor; degraded by construction
                    fit_ok[t] = True  # marked OK to feed downstream smoother + interpolation
                    branch_per_frame[t] = "translation_only"
                    # Don't update last_good_idx_for_oneanchor — translation-only is degraded;
                    # we want to propagate the LAST true 3+/2 anchor solve, not chain translations.
                else:
                    # No prior good frame — give up
                    params[t] = [0.0, 0.0, 0.0, 0.0]
                    residuals[t] = np.inf
                    fit_ok[t] = False
                    branch_per_frame[t] = "translation_only_no_prev"

            else:  # n_active == 0
                params[t] = [0.0, 0.0, 0.0, 0.0]
                residuals[t] = np.inf
                fit_ok[t] = False
                branch_per_frame[t] = "zero_active"

        # NOTE: np.unwrap deliberately NOT applied here on the raw series. Multi-AI review R2
        # (Gemini CRIT): unwrapping cyclic noise from failed-tracker frames as if it were real
        # >180° rotations would create permanent phase shifts. With Step 3's interpolation,
        # those phantom rotations would then be linearly bridged across gaps producing 360°
        # "spin" artifacts in the warped output. We unwrap only over good_indices in Step 3.

        # --- Step 3: Detect failure frames + INTERPOLATE failure params from good neighbors ---
        # We do NOT stamp translation_only zeros here — that would inject log_s=0, theta=0 into
        # the param series and the smoother would propagate those zeros into neighboring good
        # frames (observed: applied_failure_count 91/121 vs raw 65/121). Instead, linearly
        # interpolate each param column over good frames so the smoother sees a continuous
        # plausible series. The user's `fallback_mode` is then applied POST-smoothing as a
        # safety override (Step 5). Multi-AI review R1 HIGH #2 + R2 follow-up.
        rmse_threshold_px = rmse_threshold_frac * torso_diag
        failure_mask = (~fit_ok) | (residuals > rmse_threshold_px)
        failure_count = int(failure_mask.sum())
        if verbose_debug:
            print(f"[NV_PreCompWarp] RMSE threshold: {rmse_threshold_px:.2f}px | "
                  f"failure frames: {failure_count}/{T}")

        good_indices = np.where(~failure_mask)[0]
        if len(good_indices) >= 2:
            # Unwrap rotation ONLY over good frames before interpolating, so cyclic
            # noise from failed-tracker frames doesn't get treated as real rotations.
            params[good_indices, 3] = np.unwrap(params[good_indices, 3])

            # Linear interpolation per parameter column from good frames onto all frames.
            # np.interp clamps to endpoint values for out-of-range queries (no extrapolation).
            x_all = np.arange(T, dtype=np.float64)
            x_good = good_indices.astype(np.float64)
            for col in range(4):
                params[:, col] = np.interp(x_all, x_good, params[good_indices, col])
        elif len(good_indices) == 1:
            # Only one good frame — propagate its params to all frames as best-effort
            params[:] = params[good_indices[0]]
        else:
            # Zero good frames — every frame failed RMSE. Raw params are garbage from failed
            # solves. Degrade to identity transform per-frame so the warp is a no-op rather
            # than applying nonsense matrices. Multi-AI review R2 MED (Codex). User should
            # switch to bypass_pass_through fallback_mode or re-place anchors.
            print(f"[NV_PreCompWarp] WARNING: ALL {T} frames failed RMSE threshold "
                  f"({rmse_threshold_px:.2f}px). Falling back to identity transform "
                  f"(no warp applied). Consider bypass_pass_through or different anchors.")
            params[:] = 0.0  # tx=ty=log_s=theta=0 -> identity transform

        # --- Step 4: Savitzky-Golay smoothing per parameter (on sanitized series) ---
        eff_window = self._safe_savgol_window(smooth_window, T, smooth_polyorder)
        if eff_window is not None:
            for i in range(4):
                params[:, i] = savgol_filter(params[:, i], eff_window, smooth_polyorder)
            if verbose_debug:
                print(f"[NV_PreCompWarp] Smoothed with savgol window={eff_window}, polyorder={smooth_polyorder}")
        else:
            if verbose_debug:
                print(f"[NV_PreCompWarp] Sequence too short for smoothing (T={T}); using raw per-frame params")

        # --- Step 5: Apply user fallback_mode (POST-smoothing override) ---
        # After interpolation+smoothing, params for failure frames already contain a plausible
        # estimate from neighboring good frames. fallback_mode is an explicit user override:
        #   smoothed_interp (default): NO OVERRIDE. Trust the smoother+interpolation result.
        #                              Best for action shots where rotation/scale matter and
        #                              translation-only would discard valuable transform info.
        #   translation_only: zero scale/rotation, keep tx/ty. Conservative — forces pure
        #                     translation. WRONG for shots where character rotates/scales
        #                     during failure stretches (residuals balloon to 200+px).
        #   hold_last_good:   step-hold semantics — copy last good frame's smoothed params
        #                     onto each failure frame. Good for "mostly static pose with
        #                     periodic tracker dropouts". Multi-AI R2 MED (Codex): was a no-op
        #                     pre-fix.
        #   bypass_pass_through: handled at warp time (explicit copy of raw render).
        if fallback_mode == "smoothed_interp":
            pass  # no override — smoothed-interpolated params used as-is
        elif fallback_mode == "translation_only":
            for t in range(T):
                if failure_mask[t]:
                    params[t, 2] = 0.0  # log_s = 0 -> scale = 1
                    params[t, 3] = 0.0  # theta = 0
        elif fallback_mode == "hold_last_good":
            last_good_idx = None
            for t in range(T):
                if not failure_mask[t]:
                    last_good_idx = t
                    continue
                if last_good_idx is not None:
                    params[t] = params[last_good_idx].copy()
                # else: leave smoothed-from-interpolation as best-effort
        # bypass_pass_through: applied at warp time

        # --- Step 5b: Re-evaluate residuals against the SMOOTHED+fallback transform actually applied ---
        # Reason: smoothing can pull the transform away from the optimal raw fit; a frame that passed raw
        # RMSE could become noticeably worse after smoothing. Codex review flag #3.
        applied_residuals = np.zeros(T, dtype=np.float64)
        for t in range(T):
            tx_a, ty_a, log_s_a, theta_a = params[t]
            scale_a = float(np.exp(log_s_a))
            cos_a = float(np.cos(theta_a)) * scale_a
            sin_a = float(np.sin(theta_a)) * scale_a
            M_a = np.array([
                [cos_a, -sin_a, float(tx_a)],
                [sin_a,  cos_a, float(ty_a)],
            ], dtype=np.float64)
            from_2d = source_traj[t].astype(np.float64)
            to_2d = render_traj[t].astype(np.float64)
            ones = np.ones((from_2d.shape[0], 1), dtype=np.float64)
            from_h = np.hstack([from_2d, ones])
            transformed = (M_a @ from_h.T).T
            applied_residuals[t] = float(np.sqrt(np.mean(np.linalg.norm(transformed - to_2d, axis=1) ** 2)))

        # --- Step 6: Apply per-frame warp + edge pipeline ---
        # Output tensors sized to T (the truncated min length) — not T_render — to avoid silent zero tail
        warped_rgb = np.zeros((T, H, W, C), dtype=np.float32)
        warped_mask_out = np.zeros((T, H, W), dtype=np.float32)
        # Full-frame warp output (no masking applied) — useful for diagnostic visualization
        # of where the entire render lands in source coords, and for compositing workflows
        # that want the unmasked warped render as a base layer. Uses BORDER_REFLECT so the
        # edges look natural rather than abrupt black bands when the warp shifts.
        warped_full = np.zeros((T, H, W, C), dtype=np.float32)

        render_np = render_video.cpu().numpy()  # [T_render, H, W, C] in [0, 1]
        mask_np = render_mask.cpu().numpy()     # [T_mask, H, W] in [0, 1]

        erode_kernel = None
        if edge_erode_px > 0:
            ek = 2 * edge_erode_px + 1
            erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ek, ek))

        for t in range(T):
            t_mask = min(t, T_mask - 1)
            t_img = min(t, T_render - 1)
            mask_t = mask_np[t_mask].astype(np.float32)
            img_t = render_np[t_img].astype(np.float32)

            # bypass_pass_through on failure frames: explicit copy of original render+mask, no warp
            # (Identity warp would still resample with INTER_CUBIC and apply edge erode/feather, which
            # is NOT a true pass-through; user expectation is "if alignment fails, give me the raw render".)
            if fallback_mode == "bypass_pass_through" and failure_mask[t]:
                # warped_rgb (the masked output) must match the standard path's behavior of
                # zero-ing the background outside the mask. Otherwise non-bypass frames have
                # black BG and bypass frames flash to the render's actual BG = visible strobe.
                # Multi-AI review R3 LOW/MED (Gemini).
                bypass_zero_mask = mask_t[..., None] <= 1e-6
                warped_rgb[t] = np.where(bypass_zero_mask, 0.0, img_t)
                warped_mask_out[t] = mask_t
                warped_full[t] = img_t  # full-frame output also passes through unwarped
                continue

            tx, ty, log_s, theta = params[t]
            scale = float(np.exp(log_s))
            cos_t = float(np.cos(theta)) * scale
            sin_t = float(np.sin(theta)) * scale
            M = np.array([
                [cos_t, -sin_t, float(tx)],
                [sin_t,  cos_t, float(ty)],
            ], dtype=np.float32)

            if erode_kernel is not None:
                mask_t = cv2.erode(mask_t, erode_kernel)

            # Pre-warp feather (Multi-AI review HIGH #3, Gemini): blurring alpha post-warp
            # while leaving premult RGB sharp causes the unpremult divide (rgb / feathered_alpha)
            # to blow out edge pixels. Soften the input mask BEFORE premult so warp resamples
            # an already-soft edge and unpremult sees consistent values.
            if edge_feather_px > 0.0:
                k = max(3, int(edge_feather_px * 2) | 1)  # odd kernel
                mask_t = cv2.GaussianBlur(mask_t, (k, k), edge_feather_px)

            # Premultiplied RGB warp prevents bg-color bleed at mask edges.
            # CRITICAL: pass cv2.WARP_INVERSE_MAP — M maps source_pos -> render_pos (inverse-map
            # convention: "for each output pixel at source_pos, read input at M @ source_pos =
            # render_pos"). Without this flag cv2 treats M as a forward src->dst transform and
            # inverts it internally before sampling, applying the inverse of the intended warp
            # — i.e., the geometry is mirrored. Multi-AI review R1 CRIT #1 (Codex + Gemini consensus).
            #
            # Both warps use INTER_CUBIC: mismatched modes (cubic premult / linear alpha)
            # produced subtle dark fringing because cubic's edge overshoot rings divided by
            # smooth linear alpha left harsh structural edges. Multi-AI review R2 MED (Gemini).
            premult = img_t * mask_t[..., None]
            warped_premult = cv2.warpAffine(
                premult, M, (W, H),
                flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
            )
            warped_alpha = cv2.warpAffine(
                mask_t, M, (W, H),
                flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
            )
            # Cubic can produce slight negative overshoots on hard edges; clip alpha to [0, 1].
            warped_alpha = np.clip(warped_alpha, 0.0, 1.0)

            # Unpremultiply to standard RGB+alpha for downstream consumers
            alpha_safe = np.maximum(warped_alpha, 1e-6)
            rgb_unpremult = warped_premult / alpha_safe[..., None]
            # Zero-out RGB where alpha is essentially 0 (avoids garbage from divide)
            zero_mask = warped_alpha[..., None] <= 1e-6
            rgb_unpremult = np.where(zero_mask, 0.0, rgb_unpremult)

            warped_rgb[t] = rgb_unpremult
            warped_mask_out[t] = warped_alpha

            # Full-frame warp output: warp the entire render frame (no premult, no mask).
            # Same M and WARP_INVERSE_MAP. BORDER_REPLICATE (industry standard for motion
            # compensation) smears the edge pixel outward — visually fades into the comp
            # without drawing the eye, unlike BORDER_REFLECT which mirrors a localized scene
            # patch that looks artificial. Multi-AI review R3 (Gemini recommendation).
            warped_full_t = cv2.warpAffine(
                img_t, M, (W, H),
                flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REPLICATE
            )
            warped_full[t] = np.clip(warped_full_t, 0.0, 1.0)

        # --- Step 7: Build alignment_info JSON for downstream nodes ---
        # Report BOTH raw-fit residuals (pre-smoothing) and applied residuals (post-smoothing+fallback)
        # so downstream tools / debug viewers can distinguish "fit failed" from "smoothing distorted".
        applied_failure_mask = applied_residuals > rmse_threshold_px

        # Smoother health metric: ratio of applied/raw residuals.
        # Healthy: applied ~= raw (ratio ~1.0) — smoother preserves the per-frame fit quality
        # Sick:    applied >> raw (ratio >> 1.0) — smoother is destroying good fits
        # Computed only over GOOD raw frames (where raw_residuals < threshold), since failure
        # frames have garbage raw residuals that make the ratio meaningless.
        good_raw_mask = ~failure_mask & (residuals < np.inf)
        if good_raw_mask.any():
            raw_good = residuals[good_raw_mask]
            applied_good = applied_residuals[good_raw_mask]
            mean_ratio = float(applied_good.mean() / max(raw_good.mean(), 1e-6))
            median_ratio = float(np.median(applied_good) / max(np.median(raw_good), 1e-6))
            if median_ratio < 1.5:
                health = "HEALTHY (smoother preserves per-frame fit quality)"
            elif median_ratio < 3.0:
                health = "OK (smoother adds some drift, within tolerance)"
            elif median_ratio < 6.0:
                health = "DEGRADED (smoother is meaningfully harming fits — consider lower polyorder/window)"
            else:
                health = "BROKEN (smoother is destroying fits — check for fallback-poison or noisy input)"
        else:
            mean_ratio = float("inf")
            median_ratio = float("inf")
            health = "UNDEFINED (no good frames to evaluate)"

        # Branch counts for visibility-aware diagnostics
        from collections import Counter
        branch_counter = Counter(branch_per_frame)
        active_count_stats = {
            "min": int(active_count_per_frame.min()),
            "p50": int(np.percentile(active_count_per_frame, 50)),
            "mean": float(active_count_per_frame.mean()),
            "max": int(active_count_per_frame.max()),
            "frames_with_zero_active": int((active_count_per_frame == 0).sum()),
            "frames_with_one_active": int((active_count_per_frame == 1).sum()),
            "frames_with_two_active": int((active_count_per_frame == 2).sum()),
            "frames_with_3plus_active": int((active_count_per_frame >= 3).sum()),
        }

        info = {
            "T": int(T),
            "N_anchors": int(N),
            "torso_diag_px": float(torso_diag),
            "rmse_threshold_px": float(rmse_threshold_px),
            "rmse_threshold_frac": float(rmse_threshold_frac),
            "raw_failure_frame_indices": failure_mask.nonzero()[0].tolist(),
            "raw_failure_count": failure_count,
            "applied_failure_frame_indices": applied_failure_mask.nonzero()[0].tolist(),
            "applied_failure_count": int(applied_failure_mask.sum()),
            "smoother_health": health,
            "smoother_health_metrics": {
                "mean_ratio_applied_over_raw": mean_ratio,
                "median_ratio_applied_over_raw": median_ratio,
                "note": "Computed over good-raw-fit frames only. Ratio ~1.0 = healthy; >>1.0 = smoother is corrupting fits."
            },
            "visibility_aware_solve": bool(visibility_aware_solve),
            "solver_branches": dict(branch_counter),
            "active_anchor_count_stats": active_count_stats,
            "quality_floor": float(quality_floor) if visibility_aware_solve else None,
            "quality_gamma": float(quality_gamma) if visibility_aware_solve else None,
            "min_baseline_px_used": float(sparse_baseline_min) if visibility_aware_solve else None,
            "short_gap_max": int(short_gap_max) if visibility_aware_solve else None,
            "fallback_mode_used": fallback_mode,
            "smooth_window": int(eff_window) if eff_window is not None else None,
            "smooth_polyorder": int(smooth_polyorder),
            "edge_erode_px": int(edge_erode_px),
            "edge_feather_px": float(edge_feather_px),
            "params_layout": "[T, 4] = [tx, ty, log_scale, theta_unwrapped_radians]",
            "residuals_note": "RMSE computed over ALL anchors (not just RANSAC inliers); strict gate.",
            "per_frame_raw_residuals_px": (residuals.tolist() if verbose_debug else
                                           "(set verbose_debug=True to include)"),
            "per_frame_applied_residuals_px": (applied_residuals.tolist() if verbose_debug else
                                               "(set verbose_debug=True to include)"),
        }
        info_str = json.dumps(info, indent=2)

        warped_rgb_tensor = torch.from_numpy(warped_rgb).clamp(0.0, 1.0)
        warped_mask_tensor = torch.from_numpy(warped_mask_out).clamp(0.0, 1.0)
        warped_full_tensor = torch.from_numpy(warped_full).clamp(0.0, 1.0)

        if verbose_debug:
            print(f"[NV_PreCompWarp] Done. warped_render={tuple(warped_rgb_tensor.shape)}, "
                  f"warped_mask={tuple(warped_mask_tensor.shape)}, "
                  f"warped_render_full={tuple(warped_full_tensor.shape)}")

        return (warped_rgb_tensor, warped_mask_tensor, warped_full_tensor, info_str)

    # =========================================================================
    # Helpers
    # =========================================================================

    @staticmethod
    def _parse_trajectories(s: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Parse trajectories JSON. Accepts:
        - Raw JSON array of shape [T, N, 2]
        - Dict with {'data': [[[x,y], ...], ...], optional 'names': [...], optional 'visibility': [[v,...]]}
        Returns (positions [T, N, 2], visibility [T, N] or None).
        Visibility is only present when input is the dict form from NV_CoTrackerTrajectoriesJSON.
        """
        if not s.strip():
            raise ValueError("[NV_PreCompWarp] trajectories input is empty")
        try:
            obj = json.loads(s)
        except json.JSONDecodeError as e:
            raise ValueError(f"[NV_PreCompWarp] trajectories not valid JSON: {e}")

        vis = None
        if isinstance(obj, dict) and "data" in obj:
            arr = np.asarray(obj["data"], dtype=np.float64)
            if "visibility" in obj and obj["visibility"] is not None:
                try:
                    vis = np.asarray(obj["visibility"], dtype=np.float64)
                    if vis.ndim != 2:
                        # Defensive: if shape is unexpected, drop it rather than crash
                        vis = None
                except (TypeError, ValueError):
                    vis = None
        else:
            arr = np.asarray(obj, dtype=np.float64)

        if arr.ndim != 3 or arr.shape[-1] != 2:
            raise ValueError(
                f"[NV_PreCompWarp] trajectories must have shape [T, N, 2], got {arr.shape}"
            )

        # Validate visibility shape matches positions
        if vis is not None and vis.shape != (arr.shape[0], arr.shape[1]):
            print(f"[NV_PreCompWarp] WARNING: visibility shape {vis.shape} doesn't match positions "
                  f"{arr.shape[:2]}; dropping visibility (falling back to equal weights).")
            vis = None

        return arr, vis

    @staticmethod
    def _check_geometry_ok(pts: np.ndarray, min_spread_frac: float = 0.02) -> bool:
        """Sanity-check anchor geometry before fitting. Rejects:
        - All points nearly collinear (rank-deficient → unstable similarity fit)
        - All points within a tiny region (< min_spread_frac of pairwise diagonal)
        Returns False if anchors are geometrically degenerate.
        """
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        if pts.shape[0] < 2:
            return False

        # Spread test: max pairwise distance must be non-trivial
        diffs = pts[:, None, :] - pts[None, :, :]
        dists = np.linalg.norm(diffs, axis=-1)
        spread = float(np.max(dists))
        if spread < 1.0:  # all anchors within 1 pixel of each other
            return False

        # Collinearity test only meaningful for N >= 3.
        # 2 anchors are inherently collinear (any 2 points lie on a line) but a 4-DOF
        # similarity transform is still exactly determined from 2 points (4 unknowns,
        # 2 points x 2 coords = 4 equations). So skip collinearity rejection when N == 2.
        if pts.shape[0] >= 3:
            centered = pts - pts.mean(axis=0, keepdims=True)
            try:
                _, sv, _ = np.linalg.svd(centered, full_matrices=False)
            except np.linalg.LinAlgError:
                return False
            if sv.size < 2 or sv[0] < 1e-6:
                return False
            ratio = sv[1] / sv[0]
            if ratio < 1e-3:  # near-collinear
                return False
        return True

    @staticmethod
    def _solve_similarity(from_pts: np.ndarray, to_pts: np.ndarray) -> Tuple[float, float, float, float, float, bool]:
        """Solve 4-DOF similarity transform (Umeyama 1991) mapping from_pts -> to_pts.

        Returns (tx, ty, log_s, theta_rad, rms_residual_px, ok).
        ok=False if the fit failed (degenerate points, all collinear, off-frame outliers, etc.)
        """
        # Geometry sanity check before fitting (catches near-collinear / collapsed anchors that
        # cv2.estimateAffinePartial2D may silently return a numerically-plausible-but-degenerate
        # matrix for). Codex review flag #5.
        if not NV_PreCompWarp._check_geometry_ok(from_pts) or not NV_PreCompWarp._check_geometry_ok(to_pts):
            return 0.0, 0.0, 0.0, 0.0, np.inf, False

        src = np.asarray(from_pts, dtype=np.float32).reshape(-1, 1, 2)
        dst = np.asarray(to_pts, dtype=np.float32).reshape(-1, 1, 2)

        M, inliers = cv2.estimateAffinePartial2D(
            src, dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
        )
        if M is None:
            return 0.0, 0.0, 0.0, 0.0, np.inf, False

        # Decompose 2x3 similarity: M = [[s*cos, -s*sin, tx], [s*sin, s*cos, ty]]
        a, b = float(M[0, 0]), float(M[0, 1])
        tx, ty = float(M[0, 2]), float(M[1, 2])
        scale = float(np.sqrt(a * a + b * b))
        if scale < 1e-6:
            return 0.0, 0.0, 0.0, 0.0, np.inf, False
        theta = float(np.arctan2(M[1, 0], M[0, 0]))
        log_s = float(np.log(scale))

        # Compute residual: apply M to from_pts and compare to to_pts.
        # Note: residual is computed over ALL anchors (not just RANSAC inliers) — this is a strict gate.
        # Frames where one anchor is grossly mistracked will be flagged for fallback even if 4/5 inliers fit well.
        from_2d = np.asarray(from_pts, dtype=np.float64).reshape(-1, 2)
        to_2d = np.asarray(to_pts, dtype=np.float64).reshape(-1, 2)
        ones = np.ones((from_2d.shape[0], 1), dtype=np.float64)
        from_h = np.hstack([from_2d, ones])
        transformed = (M.astype(np.float64) @ from_h.T).T  # [N, 2]
        per_pt_err = np.linalg.norm(transformed - to_2d, axis=1)
        rms = float(np.sqrt(np.mean(per_pt_err ** 2)))

        return tx, ty, log_s, theta, rms, True

    @staticmethod
    def _compute_torso_diag(pts: np.ndarray) -> float:
        """Compute torso scale via max pairwise distance among anchors. More stable than bbox-diagonal
        when anchor distribution is non-rectangular (e.g., 6 torso joints)."""
        pts = np.asarray(pts, dtype=np.float64)
        if pts.shape[0] < 2:
            return 1.0
        diffs = pts[:, None, :] - pts[None, :, :]
        dists = np.linalg.norm(diffs, axis=-1)
        return float(np.max(dists))

    @staticmethod
    def _safe_savgol_window(requested: int, T: int, polyorder: int) -> Optional[int]:
        """Return the largest valid odd window <= min(requested, T) that satisfies the polynomial
        order constraint, or None if smoothing is infeasible for this sequence.

        savgol requires: window > polyorder, window <= sequence length, and window odd.
        """
        if T < polyorder + 2:
            return None
        win = min(int(requested), T)
        if win % 2 == 0:
            win -= 1
        if win < polyorder + 2:
            return None
        return win

    @staticmethod
    def _weighted_umeyama_2d(src: np.ndarray, dst: np.ndarray,
                              weights: np.ndarray) -> Tuple[float, float, float, float, float, bool]:
        """Weighted Umeyama (1991) 2D similarity solve. Returns (tx, ty, log_s, theta, weighted_rms, ok).

        Solves y = s*R*x + t with weighted least squares. Continuous weighting eliminates the
        anchor-set crossing-threshold jitter that hard filtering can leave (multi-AI debate R4
        Codex argument). Implementation follows the standard Umeyama algorithm with weighted
        centroids/covariance/scale.
        """
        src = np.asarray(src, dtype=np.float64).reshape(-1, 2)
        dst = np.asarray(dst, dtype=np.float64).reshape(-1, 2)
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        N = src.shape[0]

        if N < 2 or src.shape[0] != dst.shape[0] or w.shape[0] != N:
            return 0.0, 0.0, 0.0, 0.0, np.inf, False

        w_sum = float(w.sum())
        if w_sum < 1e-9:
            return 0.0, 0.0, 0.0, 0.0, np.inf, False

        # Geometry sanity check on the (unweighted) source spread — reject if all anchors
        # collapsed or near-collinear with N >= 3
        if not NV_PreCompWarp._check_geometry_ok(src):
            return 0.0, 0.0, 0.0, 0.0, np.inf, False

        # Weighted centroids
        src_mean = (w[:, None] * src).sum(axis=0) / w_sum
        dst_mean = (w[:, None] * dst).sum(axis=0) / w_sum

        # Centered points
        src_c = src - src_mean
        dst_c = dst - dst_mean

        # Weighted source variance (denominator for scale). If too small the solve is ill-posed.
        sigma_src_sq = float((w * (src_c ** 2).sum(axis=1)).sum() / w_sum)
        if sigma_src_sq < 1e-9:
            return 0.0, 0.0, 0.0, 0.0, np.inf, False

        # Weighted cross-covariance H = (1/W) sum_i w_i * dst_c_i * src_c_i^T  (2x2)
        H = (w[:, None] * dst_c).T @ src_c / w_sum
        try:
            U, S, Vt = np.linalg.svd(H)
        except np.linalg.LinAlgError:
            return 0.0, 0.0, 0.0, 0.0, np.inf, False

        # Reflection-resistant proper rotation (Umeyama eq. 39-43): D corrects for det(UV^T) = -1
        d = np.sign(np.linalg.det(U @ Vt))
        D = np.eye(2)
        D[1, 1] = d
        R = U @ D @ Vt
        # Scale s = trace(D * Sigma) / sigma_src_sq
        s = float((S[0] + d * S[1]) / sigma_src_sq)
        if not np.isfinite(s) or s < 1e-6:
            return 0.0, 0.0, 0.0, 0.0, np.inf, False

        # Translation: t = dst_mean - s*R*src_mean
        t_vec = dst_mean - s * (R @ src_mean)
        tx, ty = float(t_vec[0]), float(t_vec[1])

        # Decompose rotation -> theta
        theta = float(np.arctan2(R[1, 0], R[0, 0]))
        log_s = float(np.log(s))

        # Weighted RMS residual
        transformed = (s * (src @ R.T)) + t_vec
        per_pt_err_sq = ((transformed - dst) ** 2).sum(axis=1)
        weighted_mse = float((w * per_pt_err_sq).sum() / w_sum)
        rms = float(np.sqrt(max(weighted_mse, 0.0)))

        return tx, ty, log_s, theta, rms, True

    @staticmethod
    def _solve_2point_similarity(src: np.ndarray,
                                  dst: np.ndarray) -> Tuple[float, float, float, float, float, bool]:
        """Exact 2-point similarity solve. With exactly 2 point pairs, the 4-DOF similarity is
        uniquely determined. Returns (tx, ty, log_s, theta, residual=0, ok).

        Caller is responsible for the baseline check — this routine assumes the 2 points are
        sufficiently separated.
        """
        src = np.asarray(src, dtype=np.float64).reshape(-1, 2)
        dst = np.asarray(dst, dtype=np.float64).reshape(-1, 2)
        if src.shape[0] != 2 or dst.shape[0] != 2:
            return 0.0, 0.0, 0.0, 0.0, np.inf, False

        src_diff = src[1] - src[0]
        dst_diff = dst[1] - dst[0]
        src_len = float(np.linalg.norm(src_diff))
        dst_len = float(np.linalg.norm(dst_diff))
        if src_len < 1e-6 or dst_len < 1e-6:
            return 0.0, 0.0, 0.0, 0.0, np.inf, False

        s = dst_len / src_len
        log_s = float(np.log(max(s, 1e-12)))
        theta = float(np.arctan2(dst_diff[1], dst_diff[0]) - np.arctan2(src_diff[1], src_diff[0]))
        # Wrap theta to (-pi, pi]
        theta = float((theta + np.pi) % (2 * np.pi) - np.pi)

        cos_t, sin_t = np.cos(theta) * s, np.sin(theta) * s
        R_scaled = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float64)
        t_vec = dst[0] - R_scaled @ src[0]
        tx, ty = float(t_vec[0]), float(t_vec[1])

        # Exact fit: residual is mathematically zero on these 2 points. Step 5b will recompute
        # against the full anchor set if needed.
        return tx, ty, log_s, theta, 0.0, True

    @staticmethod
    def _interpolate_visibility_short_gaps(vis: np.ndarray, max_gap: int) -> np.ndarray:
        """Hybrid Strategy E: per-anchor, identify visibility gaps (consecutive frames where
        vis < small threshold) and force them to a "trustworthy" value if gap length <= max_gap
        (motion blur flicker), else mark them as fully invisible (true occlusion).

        We don't change the actual numbers — we set short-gap frames to a neutral 0.5 score
        (interpolated, low-trust but usable) and leave long-gap frames at their raw (low) vis.
        Returns a NEW [T, N] array of "effective" visibility. The interp at the trajectory
        level is already done in NV_CoTrackerTrajectoriesJSON; this routine only modulates the
        per-anchor weight signal so the solver weights short-gap-interpolated frames lower
        than full-vis frames (preventing a brief flicker from completely zero-weighting an
        otherwise-good anchor).
        """
        vis = np.asarray(vis, dtype=np.float64).copy()
        T, N = vis.shape
        if max_gap <= 0:
            return vis  # disabled — return as-is

        # Threshold below which we consider a frame's visibility "low"
        LOW = 0.30  # conservative — don't flag mildly-low as gap

        for n in range(N):
            t = 0
            while t < T:
                if vis[t, n] < LOW:
                    gap_start = t
                    while t < T and vis[t, n] < LOW:
                        t += 1
                    gap_len = t - gap_start
                    if gap_len <= max_gap:
                        # Short gap: interpolate to a low-trust 0.5 (still active but downweighted)
                        vis[gap_start:t, n] = np.maximum(vis[gap_start:t, n], 0.5)
                    # else: long gap, leave as-is (will be hard-filtered by quality_floor)
                else:
                    t += 1
        return vis


NODE_CLASS_MAPPINGS = {
    "NV_PreCompWarp": NV_PreCompWarp,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_PreCompWarp": "NV PreComp Warp",
}
