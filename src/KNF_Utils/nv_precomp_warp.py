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
                rmse_threshold_frac, fallback_mode, verbose_debug):

        # --- Step 1: Parse inputs ---
        source_traj = self._parse_trajectories(source_trajectories)  # [T_src, N, 2]
        render_traj = self._parse_trajectories(render_trajectories)  # [T_ren, N, 2]

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
        # We solve M mapping source_pos -> render_pos (cv2.estimateAffinePartial2D src->dst).
        # That M is the INVERSE-MAP for cv2.warpAffine: at output pixel = source_pos, read from
        # render at M @ source_pos = render_pos. So when we call cv2.warpAffine below, we MUST
        # pass cv2.WARP_INVERSE_MAP, otherwise cv2 treats M as a forward transform and inverts
        # it internally — applying the inverse of what we want (geometrically backward warp).
        # Multi-AI review CRIT #1 (Codex + Gemini consensus 2026-04-30).
        params = np.zeros((T, 4), dtype=np.float64)  # [tx, ty, log_s, theta]
        residuals = np.zeros(T, dtype=np.float64)
        fit_ok = np.zeros(T, dtype=bool)

        for t in range(T):
            from_pts = source_traj[t]      # source actor anchors at frame t
            to_pts = render_traj[t]         # render character anchors at frame t (per-frame, was static t=0)
            tx, ty, log_s, theta, residual, ok = self._solve_similarity(from_pts, to_pts)
            params[t] = [tx, ty, log_s, theta]
            residuals[t] = residual
            fit_ok[t] = ok

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
    def _parse_trajectories(s: str) -> np.ndarray:
        """Parse source_trajectories JSON. Accepts:
        - Raw JSON array of shape [T, N, 2]
        - Dict with {'data': [[[x,y], ...], ...], optional 'names': [...]}
        Returns numpy [T, N, 2].
        """
        if not s.strip():
            raise ValueError("[NV_PreCompWarp] source_trajectories is empty")
        try:
            obj = json.loads(s)
        except json.JSONDecodeError as e:
            raise ValueError(f"[NV_PreCompWarp] source_trajectories not valid JSON: {e}")

        if isinstance(obj, dict) and "data" in obj:
            arr = np.asarray(obj["data"], dtype=np.float64)
        else:
            arr = np.asarray(obj, dtype=np.float64)

        if arr.ndim != 3 or arr.shape[-1] != 2:
            raise ValueError(
                f"[NV_PreCompWarp] source_trajectories must have shape [T, N, 2], got {arr.shape}"
            )
        return arr

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


NODE_CLASS_MAPPINGS = {
    "NV_PreCompWarp": NV_PreCompWarp,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_PreCompWarp": "NV PreComp Warp",
}
