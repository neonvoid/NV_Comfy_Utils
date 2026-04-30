"""
NV CoTracker Trajectories - Run CoTracker3 on a video with NV_PointPicker anchors and emit
JSON trajectories in the format that NV_PreCompWarp consumes.

This is a small bridge node between:
  - NV_PointPicker (user clicks N anchors on a keyframe of the source video, outputs [{x,y,t},...] JSON)
  - NV_CoTrackerBridge has its own CoTracker3 invocation but its OUTPUTS are warped images, not raw
    trajectories. This node exposes the trajectories themselves as JSON for downstream nodes that
    need positional info (NV_PreCompWarp, future skeleton retargeters, debug overlays).

Reuses the cached CoTracker3 model singleton from cotracker_bridge to avoid double-loading.

Output format (matches NV_PreCompWarp.source_trajectories input):
{
    "shape": [T, N, 2],
    "data": [[[x_t0_p0, y_t0_p0], [x_t0_p1, y_t0_p1], ...], [[x_t1_p0, y_t1_p0], ...], ...],
    "names": ["anchor_0", "anchor_1", ..., "anchor_{N-1}"],
    "visibility": [[v_t0_p0, v_t0_p1, ...], ...],   # optional, mean per-frame per-point
    "fps": null,                                      # not tracked here; downstream consumers may ignore
    "interpolated_invisible_frames": [t0, t3, ...]    # frames where any point was below visibility threshold
}
"""

import json
from typing import List, Tuple

import cv2
import numpy as np
import torch

# Reuse the model singleton from cotracker_bridge (avoids double-load of the 200MB+ checkpoint).
# `_get_cotracker_model` is module-private by convention; importing it explicitly is fine.
from .cotracker_bridge import _get_cotracker_model


def _interpolate_invisible_xy(
    positions: List[Tuple[float, float]],
    visibility: List[float],
    threshold: float = 0.5,
) -> List[Tuple[float, float]]:
    """Interpolate (x, y) positions across frames where visibility falls below threshold.
    If all frames are invisible, returns the input unchanged. Edge frames clamp to nearest visible."""
    n = len(positions)
    if n == 0:
        return positions
    visible = [v >= threshold for v in visibility]
    if not any(visible):
        return positions

    out = list(positions)
    # Forward fill from first visible
    first_visible = next(i for i, v in enumerate(visible) if v)
    for i in range(first_visible):
        out[i] = positions[first_visible]
    # Backward fill from last visible
    last_visible = max(i for i, v in enumerate(visible) if v)
    for i in range(last_visible + 1, n):
        out[i] = positions[last_visible]
    # Linear interpolate gaps between visible frames
    i = first_visible
    while i < last_visible:
        if visible[i + 1]:
            i += 1
            continue
        gap_start = i
        j = i + 1
        while j <= last_visible and not visible[j]:
            j += 1
        # Linear interp from positions[gap_start] to positions[j]
        x0, y0 = positions[gap_start]
        x1, y1 = positions[j]
        steps = j - gap_start
        for k in range(1, steps):
            alpha = k / steps
            out[gap_start + k] = (
                x0 + alpha * (x1 - x0),
                y0 + alpha * (y1 - y0),
            )
        i = j
    return out


class NV_CoTrackerTrajectoriesJSON:
    """Run CoTracker3 on a video with seed points and emit JSON trajectories for downstream nodes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "The video to track points across (e.g., the source actor's footage). "
                               "Shape [B, H, W, C] in [0, 1]."
                }),
                "tracking_points": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "JSON array of seed points from NV_PointPicker. Format: [{\"x\":..,\"y\":..,\"t\":..}, ...]. "
                               "Each point's `t` field anchors which frame the point was clicked on. "
                               "CoTracker3 supports per-query t natively — different anchors can come from different keyframes. "
                               "STRICT: any input dict missing 'x' or 'y' raises ValueError (no silent drop) — fix the upstream JSON."
                }),
                "min_visibility": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Visibility threshold. Frames where a point's predicted visibility falls below this "
                               "are linearly interpolated from neighboring visible frames. 0.0 = no interpolation, "
                               "use raw predictions even if invisible."
                }),
                "names_csv": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Optional comma-separated names for anchors (in same order as tracking_points). "
                               "E.g., 'head,L_shoulder,R_shoulder,L_hip,R_hip'. If blank, auto-generates 'anchor_0'..'anchor_{N-1}'."
                }),
                "build_debug_overlay": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Render an IMAGE batch showing each anchor as a colored dot per frame "
                               "(filled = tracked above min_visibility, hollow = below threshold / interpolated). "
                               "Anchor index labels + per-frame visible count + max frame-to-frame jump in header. "
                               "Wire to NV_Preview_Animation or VHS_VideoCombine to *see* whether CoTracker is actually tracking."
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("trajectories_json", "info", "debug_overlay")
    FUNCTION = "execute"
    CATEGORY = "NV_Utils/Tracking"
    DESCRIPTION = (
        "Run CoTracker3 on the input video using anchor points from NV_PointPicker. "
        "Emits JSON-formatted [T, N, 2] trajectories for NV_PreCompWarp and other position-consuming nodes. "
        "Reuses the cached CoTracker3 model singleton."
    )

    def execute(self, image, tracking_points, min_visibility, names_csv, build_debug_overlay):
        # --- Step 1: Parse seed points ---
        if not tracking_points or not tracking_points.strip():
            raise ValueError("[NV_CoTrackerTrajectoriesJSON] tracking_points is empty. "
                             "Connect NV_PointPicker output.")
        try:
            parsed = json.loads(tracking_points)
        except json.JSONDecodeError as e:
            raise ValueError(f"[NV_CoTrackerTrajectoriesJSON] tracking_points not valid JSON: {e}")
        if not isinstance(parsed, list) or len(parsed) == 0:
            raise ValueError(f"[NV_CoTrackerTrajectoriesJSON] tracking_points must be a non-empty list, "
                             f"got {type(parsed).__name__}")

        T_video = image.shape[0]
        H, W = image.shape[1], image.shape[2]
        if T_video < 2:
            raise ValueError(
                f"[NV_CoTrackerTrajectoriesJSON] Need at least 2 frames for tracking, got T={T_video}. "
                f"CoTracker3 cannot produce meaningful trajectories on single-frame input."
            )

        # Build query list (x, y, t) — same parsing pattern as cotracker_bridge for consistency.
        # Codex review flag: count and report skipped malformed points instead of silently dropping.
        query_list = []  # list of (x, y, t)
        skipped_count = 0
        for p in parsed:
            if not (isinstance(p, dict) and "x" in p and "y" in p):
                skipped_count += 1
                continue
            try:
                t = max(0, min(int(p.get("t", 0)), T_video - 1))
            except (TypeError, ValueError):
                t = 0
            try:
                qx = float(p["x"])
                qy = float(p["y"])
            except (TypeError, ValueError) as e:
                raise ValueError(f"[NV_CoTrackerTrajectoriesJSON] Bad x/y in tracking_points: {e}")
            query_list.append((qx, qy, t))

        if skipped_count > 0:
            raise ValueError(
                f"[NV_CoTrackerTrajectoriesJSON] {skipped_count}/{len(parsed)} input points missing "
                f"'x' or 'y' keys. Refusing silent drop — fix the upstream NV_PointPicker JSON."
            )
        if not query_list:
            raise ValueError("[NV_CoTrackerTrajectoriesJSON] No valid points found in tracking_points "
                             "(need at least one with 'x' and 'y' keys).")
        N = len(query_list)
        print(f"[NV_CoTrackerTrajectoriesJSON] {N} anchors over {T_video} frames")

        # --- Step 2: Resolve names ---
        if names_csv.strip():
            # Sanitize: strip whitespace, replace any empty entries (from trailing/repeated commas)
            # with the auto-generated default. Codex review flag.
            raw_names = [s.strip() for s in names_csv.split(",")]
            names = [n if n else f"anchor_{i}" for i, n in enumerate(raw_names)]
            if len(names) != N:
                print(f"[NV_CoTrackerTrajectoriesJSON] WARNING: names_csv has {len(names)} entries "
                      f"but {N} anchors. Padding/truncating to match.")
                if len(names) < N:
                    names = names + [f"anchor_{i}" for i in range(len(names), N)]
                else:
                    names = names[:N]
        else:
            names = [f"anchor_{i}" for i in range(N)]

        # --- Step 3: Run CoTracker3 ---
        # queries shape: (1, N, 3) = [[t, x, y], ...]   <-- NOTE: t comes first, NOT (x, y, t)
        queries = torch.tensor(
            [[[float(t), qx, qy] for qx, qy, t in query_list]],
            dtype=torch.float32,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = _get_cotracker_model()
        all_tracks = None
        all_vis = None
        try:
            model = model.to(device)
            # CRITICAL: CoTracker3 expects video in [0, 255] float range (per the official
            # PyTorch Hub examples: `read_video_from_path` returns uint8 [0,255], then `.float()`
            # is called WITHOUT normalization). ComfyUI IMAGE is [0, 1] float — without scaling,
            # the model sees almost-black frames on every input and visibility collapses to ~0
            # across all anchors regardless of how visually clear the actual image is.
            video_dev = image.permute(0, 3, 1, 2).unsqueeze(0).to(device) * 255.0  # [1, T, C, H, W] in [0, 255]
            queries_dev = queries.to(device)
            with torch.no_grad():
                pred_tracks, pred_visibility = model(video_dev, queries=queries_dev)
            # pred_tracks: [1, T, N, 2], pred_visibility: [1, T, N] or [1, T, N, 1]
            # Move to CPU INSIDE try block, then explicitly del GPU refs so empty_cache
            # actually frees the VRAM (Gemini review flag — local GPU tensors otherwise outlive
            # the empty_cache call due to Python GC ordering).
            all_tracks = pred_tracks[0].cpu()
            # CoTracker3 may return pred_visibility as a bool tensor in some versions —
            # cast to float at the boundary so downstream `.item()`, JSON serialization,
            # and visibility statistics behave as numeric values (not True/False that format
            # as "0.00" and confuse the user-facing reporting). Multi-AI review MED #4.
            all_vis = pred_visibility[0].cpu().float()
            # Diagnostic: print model output stats so input/output mismatches are visible at runtime.
            # Run once, then can be removed when stable.
            print(f"[NV_CoTrackerTrajectoriesJSON] DEBUG model outputs: "
                  f"tracks shape={tuple(all_tracks.shape)} dtype={all_tracks.dtype} "
                  f"range=[{all_tracks.min().item():.1f}, {all_tracks.max().item():.1f}] | "
                  f"visibility shape={tuple(all_vis.shape)} dtype={all_vis.dtype} "
                  f"range=[{all_vis.float().min().item():.3f}, {all_vis.float().max().item():.3f}] "
                  f"mean={all_vis.float().mean().item():.3f}")
            del video_dev, queries_dev, pred_tracks, pred_visibility
        finally:
            try:
                model.cpu()
            except Exception:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if all_tracks is None or all_vis is None:
            # Should be unreachable if the try block completed; defensive.
            raise RuntimeError("[NV_CoTrackerTrajectoriesJSON] CoTracker3 inference produced no output.")
        if all_vis.dim() > 2:
            all_vis = all_vis.squeeze(-1)

        # --- Step 4: Build trajectory data + visibility, with optional invisible-frame interpolation ---
        # Per-point: linearly interpolate (x, y) where visibility < min_visibility
        per_point_positions = []  # length N, each [(x, y), ...] of length T
        per_point_vis = []        # length N, each [v, ...] of length T
        interpolated_frames = set()

        for qi in range(N):
            positions_qi = [(all_tracks[t, qi, 0].item(), all_tracks[t, qi, 1].item()) for t in range(T_video)]
            vis_qi = [all_vis[t, qi].item() for t in range(T_video)]
            if min_visibility > 0:
                interp = _interpolate_invisible_xy(positions_qi, vis_qi, threshold=min_visibility)
                for t in range(T_video):
                    if vis_qi[t] < min_visibility:
                        interpolated_frames.add(t)
                per_point_positions.append(interp)
            else:
                per_point_positions.append(positions_qi)
            per_point_vis.append(vis_qi)

        # Reshape to [T, N, 2] and [T, N]
        traj_data = [[per_point_positions[qi][t] for qi in range(N)] for t in range(T_video)]
        vis_data = [[per_point_vis[qi][t] for qi in range(N)] for t in range(T_video)]

        # --- Step 5: Pack JSON ---
        out = {
            "shape": [int(T_video), int(N), 2],
            "data": traj_data,
            "names": names,
            "visibility": vis_data,
            "interpolated_invisible_frames": sorted(interpolated_frames),
            "min_visibility_threshold": float(min_visibility),
        }
        out_str = json.dumps(out)

        # --- Step 6: Diagnostic stats ---
        # Per-anchor visibility stats
        per_anchor_stats = []
        for qi in range(N):
            v_arr = np.asarray(per_point_vis[qi], dtype=np.float64)
            visible_count = int(np.sum(v_arr >= min_visibility))
            per_anchor_stats.append({
                "name": names[qi],
                "vis_avg": float(v_arr.mean()),
                "vis_min": float(v_arr.min()),
                "vis_max": float(v_arr.max()),
                "visible_frames": visible_count,
                "visible_pct": (visible_count / T_video) * 100.0,
            })

        # Frame-to-frame anchor displacement (max across anchors per frame transition).
        # Sudden jumps = CoTracker losing track. p99/max indicate how violent the worst jumps are.
        max_jumps_per_transition = []
        worst_jump_frame = -1
        worst_jump_value = 0.0
        for t in range(1, T_video):
            jumps_at_t = []
            for qi in range(N):
                x0, y0 = per_point_positions[qi][t - 1]
                x1, y1 = per_point_positions[qi][t]
                jumps_at_t.append(float(np.hypot(x1 - x0, y1 - y0)))
            mx = max(jumps_at_t) if jumps_at_t else 0.0
            max_jumps_per_transition.append(mx)
            if mx > worst_jump_value:
                worst_jump_value = mx
                worst_jump_frame = t
        if max_jumps_per_transition:
            jumps_arr = np.asarray(max_jumps_per_transition)
            jump_p50 = float(np.percentile(jumps_arr, 50))
            jump_p95 = float(np.percentile(jumps_arr, 95))
            jump_p99 = float(np.percentile(jumps_arr, 99))
            jump_max = float(jumps_arr.max())
        else:
            jump_p50 = jump_p95 = jump_p99 = jump_max = 0.0

        # Per-frame visible counts -> "frames with all visible", "frames with <2 visible" (warp can't fit)
        per_frame_visible = []
        for t in range(T_video):
            vc = sum(1 for qi in range(N) if per_point_vis[qi][t] >= min_visibility)
            per_frame_visible.append(vc)
        all_visible = sum(1 for vc in per_frame_visible if vc == N)
        half_visible = sum(1 for vc in per_frame_visible if vc >= max(2, N // 2))
        below_two = sum(1 for vc in per_frame_visible if vc < 2)

        # --- Step 6b: Per-anchor verdicts (DROP / MARGINAL / OK / STRONG) ---
        # Used both in the per-anchor table and in actionable suggestions below.
        def _classify_anchor(pct):
            if pct < 30: return "DROP"
            if pct < 60: return "MARGINAL"
            if pct < 90: return "OK"
            return "STRONG"
        for s in per_anchor_stats:
            s["verdict"] = _classify_anchor(s["visible_pct"])

        # --- Step 6c: Find consecutive runs where visible_count < min threshold ---
        # These are the ranges where the warp can't fit similarity (need >=2 anchors).
        # We also surface ranges with only 1 anchor visible — warp can technically fit but
        # is vulnerable to drift since there's no rotation/scale lever arm.
        def _find_runs(per_frame_count, predicate):
            """Return list of (start, end) inclusive frame ranges where predicate(count) is True."""
            runs, start = [], None
            for t in range(len(per_frame_count)):
                if predicate(per_frame_count[t]):
                    if start is None: start = t
                else:
                    if start is not None:
                        runs.append((start, t - 1))
                        start = None
            if start is not None:
                runs.append((start, len(per_frame_count) - 1))
            return runs

        no_fit_runs = _find_runs(per_frame_visible, lambda c: c < 2)            # warp fails entirely
        thin_runs = _find_runs(per_frame_visible, lambda c: c == 1)             # warp drift-prone

        # For each problem range, find the frame within it (or just outside) most likely to
        # benefit from a multi-keyframe seed: highest visible_count, then highest sum of
        # raw visibility scores. If all frames in the range have 0 visibility, fall back to
        # the closest frame outside the range with at least 1 visible anchor.
        def _best_seed_frame(start, end):
            best_t, best_score = start, -1.0
            for t in range(start, end + 1):
                vis_sum = sum(per_point_vis[qi][t] for qi in range(N))
                count = per_frame_visible[t]
                # Score: prioritize visible_count, break ties by sum of vis scores
                score = count * 1000 + vis_sum
                if score > best_score:
                    best_score = score
                    best_t = t
            # If best frame in range has 0 visibility, look just outside the range
            if per_frame_visible[best_t] == 0:
                for offset in range(1, T_video):
                    for cand in (start - offset, end + offset):
                        if 0 <= cand < T_video and per_frame_visible[cand] >= 1:
                            return cand, "outside_range"
                return best_t, "all_invisible"
            return best_t, "within_range"

        # --- Step 6d: Overall quality verdict ---
        n_strong = sum(1 for s in per_anchor_stats if s["verdict"] == "STRONG")
        n_ok = sum(1 for s in per_anchor_stats if s["verdict"] == "OK")
        n_drop = sum(1 for s in per_anchor_stats if s["verdict"] == "DROP")
        no_fit_pct = 100 * below_two / T_video
        if n_drop == 0 and no_fit_pct < 5 and jump_p99 < 30:
            quality = "GOOD"
        elif n_drop <= max(1, N // 4) and no_fit_pct < 25 and jump_p99 < 60:
            quality = "FAIR"
        else:
            quality = "POOR"

        # --- Build info string with verdicts + actionable suggestions ---
        info_lines = [
            f"[NV_CoTrackerTrajectoriesJSON]",
            f"  T={T_video} frames, N={N} anchors, {H}x{W} px",
            f"",
            f"  ┌─ TRACKING QUALITY: {quality} ─────────────────────────",
            f"  │ Strong anchors (>=90% visible):     {n_strong}/{N}",
            f"  │ OK or better (>=60% visible):       {n_strong + n_ok}/{N}",
            f"  │ Frames warp cannot fit (<2 visible): {below_two}/{T_video} ({no_fit_pct:.0f}%)",
            f"  │ Worst tracker jump: {jump_max:.0f}px at frame {worst_jump_frame}",
            f"  └─",
            f"",
            f"  Per-anchor breakdown (verdict | vis_avg | vis_min | visible%):",
        ]
        for s in per_anchor_stats:
            info_lines.append(
                f"    {s['name']:<14} {s['verdict']:<9} avg={s['vis_avg']:.2f}  "
                f"min={s['vis_min']:.2f}  visible={s['visible_frames']}/{T_video} ({s['visible_pct']:.0f}%)"
            )
        info_lines.extend([
            f"",
            f"  Frame-to-frame anchor jump (max across {N} anchors per transition):",
            f"    p50={jump_p50:.1f}px  p95={jump_p95:.1f}px  p99={jump_p99:.1f}px  "
            f"max={jump_max:.1f}px (at frame {worst_jump_frame})",
            f"",
            f"  Visibility threshold: {min_visibility:.2f}",
            f"    All {N} anchors visible: {all_visible}/{T_video} ({100*all_visible/T_video:.0f}%)",
            f"    >={max(2, N // 2)} anchors visible: {half_visible}/{T_video} ({100*half_visible/T_video:.0f}%)",
            f"    <2 anchors visible (warp cannot fit): {below_two}/{T_video} ({100*below_two/T_video:.0f}%)",
        ])

        # --- Actionable suggestions section ---
        suggestions = []

        # 1. Per-anchor: re-place DROP/MARGINAL anchors
        drop_anchors = [s for s in per_anchor_stats if s["verdict"] == "DROP"]
        for s in drop_anchors:
            suggestions.append(
                f"DROP {s['name']} (only {s['visible_pct']:.0f}% visible) — re-place on a more "
                f"contrasted feature. Try: edge of nostril, mouth corner, eyebrow tip, "
                f"clothing seam, or a tattoo edge — NOT smooth interior skin/fabric."
            )
        marginal_anchors = [s for s in per_anchor_stats if s["verdict"] == "MARGINAL"]
        if len(marginal_anchors) >= 2:
            names_str = ", ".join(s["name"] for s in marginal_anchors)
            suggestions.append(
                f"MARGINAL anchors: {names_str}. Multiple sub-60% anchors compound noise. "
                f"Consider replacing the lowest one on a different feature."
            )

        # 2. Problem ranges: <2 visible (warp fails)
        for (start, end) in no_fit_runs:
            length = end - start + 1
            seed_t, seed_kind = _best_seed_frame(start, end)
            if seed_kind == "all_invisible":
                suggestions.append(
                    f"Frames {start}-{end} ({length}f) have ZERO anchors visible — tracker "
                    f"completely lost. Multi-keyframe seeding won't help; this stretch may need "
                    f"`bypass_pass_through` fallback or a different shot."
                )
            elif seed_kind == "outside_range":
                suggestions.append(
                    f"Frames {start}-{end} ({length}f) have <2 anchors visible — warp can't fit. "
                    f"Try seeding a multi-keyframe anchor at frame {seed_t} (closest frame "
                    f"outside range with visible features)."
                )
            else:
                seed_count = per_frame_visible[seed_t]
                suggestions.append(
                    f"Frames {start}-{end} ({length}f) have <2 anchors visible — warp can't fit "
                    f"similarity here. Recommended: change frame_index to {seed_t} (best within "
                    f"range, {seed_count}/{N} anchors visible) and click a NEW feature visible "
                    f"there. CoTracker will treat it as a multi-keyframe seed."
                )

        # 3. Thin ranges: only 1 anchor visible (warp works but drift-prone)
        if thin_runs and not no_fit_runs:  # only mention if no worse problems exist
            longest = max(thin_runs, key=lambda r: r[1] - r[0])
            tlen = longest[1] - longest[0] + 1
            if tlen >= 5:
                seed_t, _ = _best_seed_frame(longest[0], longest[1])
                suggestions.append(
                    f"Frames {longest[0]}-{longest[1]} ({tlen}f) have only 1 anchor visible — "
                    f"warp fits but is drift-prone (no rotation/scale lever arm). Consider "
                    f"adding a 2nd anchor at frame {seed_t} on a different feature for stability."
                )

        # 4. Tracker jump warning
        if jump_max > 50 and jump_p95 > 25:
            suggestions.append(
                f"Violent tracker jumps detected (max {jump_max:.0f}px at frame "
                f"{worst_jump_frame}, p95 {jump_p95:.1f}px). Likely motion blur breaking anchor "
                f"lock. Watch the debug_overlay near frame {worst_jump_frame} to diagnose."
            )

        # 5. Universal hint when adding multi-keyframe seeds
        if no_fit_runs or (thin_runs and len(thin_runs) > 1):
            suggestions.append(
                f"REMINDER: when adding multi-keyframe seeds, click a feature that is NOT "
                f"already tracked. CoTracker treats each (x,y,t) as an INDEPENDENT track — "
                f"re-clicking the same logical feature creates duplicate tracks, not stronger ones."
            )

        if suggestions:
            info_lines.append("")
            info_lines.append("⚠ ACTIONABLE SUGGESTIONS:")
            for i, sug in enumerate(suggestions, 1):
                # Hanging-indent wrap for readability
                info_lines.append(f"  {i}. {sug}")
        else:
            info_lines.append("")
            info_lines.append("✓ No actionable issues detected — tracking quality looks healthy.")

        info_str = "\n".join(info_lines)
        print(info_str)

        # --- Step 7: Build debug overlay (optional) ---
        if build_debug_overlay:
            overlay = self._render_debug_overlay(
                image, per_point_positions, per_point_vis,
                names, float(min_visibility), max_jumps_per_transition,
            )
        else:
            # Return a 1-frame placeholder to satisfy the IMAGE output type
            overlay = torch.zeros((1, H, W, image.shape[3]), dtype=torch.float32)

        return (out_str, info_str, overlay)

    @staticmethod
    def _render_debug_overlay(
        image: torch.Tensor,
        per_point_positions: List[List[Tuple[float, float]]],
        per_point_vis: List[List[float]],
        names: List[str],
        min_visibility: float,
        max_jumps_per_transition: List[float],
    ) -> torch.Tensor:
        """Render the input video with anchor positions overlaid as colored dots.

        - Filled circle: visibility >= min_visibility (raw CoTracker output trusted)
        - Hollow circle: visibility < min_visibility (linearly interpolated, tracker had no signal here)
        - Per-anchor distinct color (HSV cycle)
        - Anchor index labels next to each dot
        - Header bar with frame number, visible count, and frame-to-frame max jump
        """
        T, H, W, C = image.shape
        N = len(per_point_positions)

        # Per-anchor distinct colors via HSV cycle (BGR for cv2)
        bgr_colors = []
        for qi in range(N):
            hue_180 = int((qi * 180.0 / max(1, N)) % 180.0)  # cv2 uses 0-180 hue range
            hsv = np.array([[[hue_180, 255, 255]]], dtype=np.uint8)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
            bgr_colors.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))

        img_np = image.detach().cpu().numpy()  # [T, H, W, C] in [0, 1]
        overlay_frames = np.empty((T, H, W, C), dtype=np.float32)

        # Sizing scales with frame size so dots/text remain readable across resolutions
        dot_radius = max(4, min(H, W) // 120)
        ring_thickness = max(1, dot_radius // 4)
        font_scale = max(0.3, min(H, W) / 1500.0)
        font_thickness = max(1, int(font_scale * 2))
        header_h = max(20, int(min(H, W) * 0.04))

        for t in range(T):
            # Convert to BGR uint8 for cv2 drawing
            frame_rgb = (np.clip(img_np[t], 0.0, 1.0) * 255.0).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            visible_count = 0
            for qi in range(N):
                x_f, y_f = per_point_positions[qi][t]
                vis = per_point_vis[qi][t]
                color = bgr_colors[qi]

                xi, yi = int(round(x_f)), int(round(y_f))
                if not (0 <= xi < W and 0 <= yi < H):
                    continue  # off-frame — don't draw

                if vis >= min_visibility:
                    visible_count += 1
                    # Filled circle with thin black outline for contrast
                    cv2.circle(frame_bgr, (xi, yi), dot_radius, color, -1, lineType=cv2.LINE_AA)
                    cv2.circle(frame_bgr, (xi, yi), dot_radius, (0, 0, 0), 1, lineType=cv2.LINE_AA)
                else:
                    # Hollow ring for interpolated/invisible
                    cv2.circle(frame_bgr, (xi, yi), dot_radius, color, ring_thickness, lineType=cv2.LINE_AA)
                    cv2.line(frame_bgr,
                             (xi - dot_radius, yi - dot_radius),
                             (xi + dot_radius, yi + dot_radius),
                             color, ring_thickness, lineType=cv2.LINE_AA)

                # Anchor index label
                label = str(qi)
                cv2.putText(frame_bgr, label,
                            (xi + dot_radius + 2, yi - dot_radius - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                            (0, 0, 0), font_thickness + 1, cv2.LINE_AA)
                cv2.putText(frame_bgr, label,
                            (xi + dot_radius + 2, yi - dot_radius - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                            color, font_thickness, cv2.LINE_AA)

            # Header bar
            jump_str = ""
            if t > 0 and (t - 1) < len(max_jumps_per_transition):
                jump_str = f"  jump={max_jumps_per_transition[t - 1]:.1f}px"
            header = f"t={t}/{T - 1}  visible={visible_count}/{N}{jump_str}"
            cv2.rectangle(frame_bgr, (0, 0), (W, header_h), (0, 0, 0), -1)
            cv2.putText(frame_bgr, header, (8, int(header_h * 0.72)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.1,
                        (255, 255, 255), font_thickness, cv2.LINE_AA)

            # Convert back to RGB float [0,1]
            frame_rgb_out = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            overlay_frames[t] = frame_rgb_out.astype(np.float32) / 255.0

        return torch.from_numpy(overlay_frames)


NODE_CLASS_MAPPINGS = {
    "NV_CoTrackerTrajectoriesJSON": NV_CoTrackerTrajectoriesJSON,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_CoTrackerTrajectoriesJSON": "NV CoTracker Trajectories (JSON)",
}
