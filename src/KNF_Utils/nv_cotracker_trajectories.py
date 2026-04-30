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
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("trajectories_json", "info")
    FUNCTION = "execute"
    CATEGORY = "NV_Utils/Tracking"
    DESCRIPTION = (
        "Run CoTracker3 on the input video using anchor points from NV_PointPicker. "
        "Emits JSON-formatted [T, N, 2] trajectories for NV_PreCompWarp and other position-consuming nodes. "
        "Reuses the cached CoTracker3 model singleton."
    )

    def execute(self, image, tracking_points, min_visibility, names_csv):
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
            video_dev = image.permute(0, 3, 1, 2).unsqueeze(0).to(device)  # [1, T, C, H, W]
            queries_dev = queries.to(device)
            with torch.no_grad():
                pred_tracks, pred_visibility = model(video_dev, queries=queries_dev)
            # pred_tracks: [1, T, N, 2], pred_visibility: [1, T, N] or [1, T, N, 1]
            # Move to CPU INSIDE try block, then explicitly del GPU refs so empty_cache
            # actually frees the VRAM (Gemini review flag — local GPU tensors otherwise outlive
            # the empty_cache call due to Python GC ordering).
            all_tracks = pred_tracks[0].cpu()
            all_vis = pred_visibility[0].cpu()
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

        info_lines = [
            f"[NV_CoTrackerTrajectoriesJSON]",
            f"  T={T_video} frames, N={N} anchors, {H}x{W} px",
            f"  Anchor names: {', '.join(names)}",
            f"  Frames with at least one invisible anchor: {len(interpolated_frames)}/{T_video}",
            f"  Min visibility per anchor: " + ", ".join(
                f"{names[qi]}={min(per_point_vis[qi]):.2f}" for qi in range(N)
            ),
        ]
        info_str = "\n".join(info_lines)
        print(info_str)

        return (out_str, info_str)


NODE_CLASS_MAPPINGS = {
    "NV_CoTrackerTrajectoriesJSON": NV_CoTrackerTrajectoriesJSON,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_CoTrackerTrajectoriesJSON": "NV CoTracker Trajectories (JSON)",
}
