"""
NV Point-Driven BBox — drive a crop bbox from a CoTracker feature trajectory
instead of from per-frame mask centroids.

Root cause this node attacks:
  NV_MaskTrackingBBox derives the bbox center from mask pixel centroids.
  SAM3 (or any per-frame segmenter) produces a stochastic silhouette — the
  centroid inherits that stochasticity. No amount of one_euro / Kalman
  smoothing downstream recovers all the lost information; filters either lag
  real motion or pass through residual noise. That noise becomes crop-window
  jitter, which then propagates through the whole VACE inpaint pipeline.

What this node does instead:
  Runs CoTracker3 on the FULL IMAGE (not the crop) using a user-picked face
  feature point. CoTracker produces a sub-pixel-stable trajectory. That
  trajectory drives the bbox CENTER. The mask is consulted only for SIZE
  (via a lock_largest / lock_mean / lock_first selector) — never for position.

Drop-in replacement for NV_MaskTrackingBBox's bbox_mask output in the crop
geometry path. The crop trajectory becomes as stable as the feature tracker,
independent of mask quality.

Workflow:
  VHS_LoadVideo ──► NV_PointDrivenBBox ──► NV_InpaintCrop2 (as bounding_box_mask)
                         ▲
                         │
  NV_PointPicker (on frame 0 of full image, face center)

CoTrackerBridge (on the cropped image) stays in the pipeline as a sub-pixel
polish on top of this — PointDrivenBBox does whole-pixel stabilization,
CoTrackerBridge cleans up the integer-quantization residual inside the crop.
"""

import json

import torch
import torch.nn.functional as F
import comfy.model_management

from .cotracker_bridge import _get_cotracker_model, _interpolate_invisible
from .bbox_ops import extract_bboxes, build_bbox_masks, print_bbox_trajectory_debug
from .mask_ops import mask_smooth


LOG_PREFIX = "[NV_PointDrivenBBox]"


class NV_PointDrivenBBox:
    """Drive a crop bbox via a CoTracker3 point trajectory on the full image."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Full-resolution video [B, H, W, C]. CoTracker3 will track "
                               "your query point across this video."
                }),
                "mask": ("MASK", {
                    "tooltip": "Per-frame mask [B, H, W]. Used ONLY for size derivation "
                               "(lock_largest / lock_mean / lock_first). Mask position is "
                               "NOT used — that's the whole point of this node."
                }),
                "tracking_points": ("STRING", {
                    "forceInput": True,
                    "tooltip": "JSON array of {x, y} from NV_PointPicker on frame 0 of the "
                               "FULL-resolution image. Pick a rigid facial feature (nose bridge, "
                               "inner eye corner). CoTracker will track this point across all frames."
                }),
                "bbox_expand_pct": ("FLOAT", {
                    "default": 0.35, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Fractional expansion of the reference bbox around the tracked "
                               "center. 0.35 = 35% larger than the mask's silhouette."
                }),
            },
            "optional": {
                "size_mode": (["lock_largest", "lock_mean", "lock_first"], {
                    "default": "lock_largest",
                    "tooltip": "How to derive the reference bbox size from the mask sequence. "
                               "lock_largest: max observed bbox dims (recommended — covers all poses). "
                               "lock_mean: median dims (tighter, less padding). "
                               "lock_first: frame 0 dims only (riskiest for changing poses)."
                }),
                "downsample_for_tracking": ("INT", {
                    "default": 2, "min": 1, "max": 4, "step": 1,
                    "tooltip": "Downsample factor for CoTracker inference. 1 = full res (slow, best accuracy). "
                               "2 = half res (4x faster, still sub-pixel accurate in upscaled coords). "
                               "Higher = faster but risks losing the feature."
                }),
                "output_feather": ("INT", {
                    "default": 0, "min": 0, "max": 64, "step": 1,
                    "tooltip": "Post-build feather radius on the output bbox mask. 0 = sharp rectangles."
                }),
                "clamp_to_image": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Shift bbox when it would extend past image bounds (preserves size). "
                               "If False, bbox is clipped instead (size shrinks at edges)."
                }),
                "min_visibility": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "CoTracker visibility threshold for trusting a frame's position. "
                               "Frames below threshold are interpolated from neighbors."
                }),
                "verbose_debug": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Print per-frame bbox trajectory stats: frame-to-frame motion, "
                               "spike frames (unusually large moves), tracker-vs-mask-centroid "
                               "divergence, cumulative displacement. Useful for diagnosing "
                               "whether residual crop jitter is from the tracker or elsewhere."
                }),
            }
        }

    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("bbox_mask", "info")
    FUNCTION = "execute"
    CATEGORY = "NV_Utils/Mask"
    DESCRIPTION = (
        "Build a crop bbox sequence driven by a CoTracker3 feature trajectory on "
        "the full image, not by mask centroids. Mask is used for size only; position "
        "is sub-pixel stable from the tracker. Drop-in replacement for "
        "NV_MaskTrackingBBox's bbox_mask in crop geometry paths."
    )

    def execute(self, image, mask, tracking_points, bbox_expand_pct,
                size_mode="lock_largest", downsample_for_tracking=2,
                output_feather=0, clamp_to_image=True, min_visibility=0.5,
                verbose_debug=False):
        device = comfy.model_management.get_torch_device()
        intermediate = comfy.model_management.intermediate_device()
        info_lines = []

        # ── Validate shapes ────────────────────────────────────────────────────
        if image.dim() != 4:
            raise ValueError(f"image must be [B, H, W, C], got {list(image.shape)}")
        if mask.dim() != 3:
            raise ValueError(f"mask must be [B, H, W], got {list(mask.shape)}")

        B, H, W, C = image.shape
        if mask.shape[0] != B:
            raise ValueError(
                f"image/mask batch mismatch: image has {B} frames, mask has {mask.shape[0]}"
            )
        if mask.shape[1] != H or mask.shape[2] != W:
            raise ValueError(
                f"image/mask spatial mismatch: image is {H}x{W}, mask is "
                f"{mask.shape[1]}x{mask.shape[2]}"
            )

        # ── Parse query point ──────────────────────────────────────────────────
        try:
            parsed = json.loads(tracking_points) if tracking_points else []
            if not isinstance(parsed, list) or not parsed:
                raise ValueError("must be a non-empty JSON array of {x, y}")
            pt = parsed[0]
            if not isinstance(pt, dict) or "x" not in pt or "y" not in pt:
                raise ValueError("each point must be {x: <float>, y: <float>}")
            qx, qy = float(pt["x"]), float(pt["y"])
        except Exception as e:
            raise ValueError(f"Failed to parse tracking_points: {e}")

        if not (0 <= qx < W and 0 <= qy < H):
            raise ValueError(
                f"Query point ({qx:.1f}, {qy:.1f}) is outside image bounds {W}x{H}. "
                f"Pick on the FULL-resolution image, not a crop."
            )

        info_lines.append(f"Query point frame 0: ({qx:.1f}, {qy:.1f}) on {W}x{H} image")

        # ── Single-frame fast path ─────────────────────────────────────────────
        if B <= 1:
            info_lines.append("Single frame — no tracking needed.")
            positions = [(qx, qy)]
        else:
            # ── Step 1: Downsample video for CoTracker if requested ────────────
            ds = max(1, int(downsample_for_tracking))
            if ds > 1:
                ds_H, ds_W = H // ds, W // ds
                img_nchw = image.permute(0, 3, 1, 2)  # [B, C, H, W]
                track_nchw = F.interpolate(img_nchw, size=(ds_H, ds_W), mode='area')
                qx_t, qy_t = qx / ds, qy / ds
                info_lines.append(f"Downsample for tracking: {ds}x → {ds_W}x{ds_H}")
            else:
                track_nchw = image.permute(0, 3, 1, 2)
                qx_t, qy_t = qx, qy
                info_lines.append("No downsample — tracking at full resolution")

            # ── Step 2: Run CoTracker3 ─────────────────────────────────────────
            model = None
            try:
                print(f"{LOG_PREFIX} Running CoTracker3 on {B} frames...")
                model = _get_cotracker_model()
                model = model.to(device)

                # CoTracker expects [1, T, C, H, W] and queries [1, N, 3]=[[t, x, y],...]
                video_ct = track_nchw.unsqueeze(0).to(device)
                queries_ct = torch.tensor(
                    [[[0.0, qx_t, qy_t]]], dtype=torch.float32, device=device
                )

                with torch.no_grad():
                    pred_tracks, pred_visibility = model(video_ct, queries=queries_ct)
                # pred_tracks: [1, B, 1, 2]
                # pred_visibility: [1, B, 1] or [1, B, 1, 1]

            except Exception as e:
                raise RuntimeError(f"CoTracker3 inference failed: {e}")
            finally:
                if model is not None:
                    model.cpu()
                torch.cuda.empty_cache()

            # ── Step 3: Extract trajectory + interpolate invisible frames ──────
            tracks = pred_tracks[0, :, 0].cpu()       # [B, 2]
            vis = pred_visibility[0, :, 0]
            if vis.dim() > 1:
                vis = vis.squeeze(-1)
            vis = vis.cpu()

            positions_t = [(tracks[b, 0].item(), tracks[b, 1].item()) for b in range(B)]
            visibility = [float(vis[b].item()) for b in range(B)]

            positions_t = _interpolate_invisible(
                positions_t, visibility, threshold=min_visibility
            )
            if positions_t is None:
                raise RuntimeError(
                    "All frames invisible to CoTracker — cannot build bbox trajectory. "
                    "Try a different tracking point (more prominent facial feature), "
                    "or reduce downsample_for_tracking."
                )

            # ── Step 4: Scale positions back to full-image coords ──────────────
            if ds > 1:
                positions = [(p[0] * ds, p[1] * ds) for p in positions_t]
            else:
                positions = positions_t

            visible_count = sum(1 for v in visibility if v >= min_visibility)
            info_lines.append(
                f"CoTracker: {visible_count}/{B} frames visible "
                f"(threshold={min_visibility:.2f})"
            )

        # ── Step 5: Derive reference bbox size from mask ───────────────────────
        x1s_m, y1s_m, x2s_m, y2s_m, present_m = extract_bboxes(mask, info_lines=None)
        widths = [max(1, x2 - x1) for x1, x2 in zip(x1s_m, x2s_m)]
        heights = [max(1, y2 - y1) for y1, y2 in zip(y1s_m, y2s_m)]

        if not widths or not heights:
            # Fallback: mask is all zeros — use a quarter of the image
            ref_w = W / 4.0
            ref_h = H / 4.0
            info_lines.append(f"Empty mask — fallback size: {ref_w:.0f}x{ref_h:.0f}")
        elif size_mode == "lock_largest":
            ref_w = float(max(widths))
            ref_h = float(max(heights))
        elif size_mode == "lock_first":
            ref_w = float(widths[0])
            ref_h = float(heights[0])
        elif size_mode == "lock_mean":
            sw = sorted(widths)
            sh = sorted(heights)
            ref_w = float(sw[len(sw) // 2])  # median for robustness
            ref_h = float(sh[len(sh) // 2])
        else:
            ref_w = float(max(widths))
            ref_h = float(max(heights))

        info_lines.append(f"Reference size ({size_mode}): {ref_w:.0f}x{ref_h:.0f}px")

        # ── Optional: verbose trajectory debug ─────────────────────────────────
        if verbose_debug and B > 1:
            mask_centroids = [
                ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                for x1, x2, y1, y2 in zip(x1s_m, x2s_m, y1s_m, y2s_m)
            ]
            print_bbox_trajectory_debug(
                positions=positions,
                compare_positions=mask_centroids,
                compare_label="Tracker ↔ mask-centroid divergence "
                              "(large is USUALLY fine — tracker follows a rigid feature, "
                              "centroid shifts with silhouette shape)",
                visibility=visibility,
                min_visibility=min_visibility,
                log_prefix=LOG_PREFIX,
            )

        # ── Step 6: Build tight per-frame bboxes at tracker positions ──────────
        x1s = []
        y1s = []
        x2s = []
        y2s = []
        for b in range(B):
            cx, cy = positions[b]
            x1 = cx - ref_w / 2.0
            y1 = cy - ref_h / 2.0
            x2 = cx + ref_w / 2.0
            y2 = cy + ref_h / 2.0

            if clamp_to_image:
                # Shift to stay in bounds, preserving size where possible
                if x1 < 0:
                    shift = -x1
                    x1 += shift
                    x2 += shift
                if y1 < 0:
                    shift = -y1
                    y1 += shift
                    y2 += shift
                if x2 > W:
                    shift = x2 - W
                    x1 -= shift
                    x2 -= shift
                if y2 > H:
                    shift = y2 - H
                    y1 -= shift
                    y2 -= shift
                # Final hard clamp (fires only if subject is near a corner
                # AND ref size > half image — rare for face refinement)
                x1 = max(0.0, x1)
                y1 = max(0.0, y1)
                x2 = min(float(W), x2)
                y2 = min(float(H), y2)

            x1s.append(x1)
            y1s.append(y1)
            x2s.append(x2)
            y2s.append(y2)

        # ── Step 7: Build the bbox mask via shared helper (applies expansion) ──
        bbox_mask = build_bbox_masks(
            x1s, y1s, x2s, y2s,
            padding=bbox_expand_pct,
            H=H, W=W,
            info_lines=info_lines,
        )

        # ── Step 8: Optional output feather ────────────────────────────────────
        if output_feather > 0:
            bbox_mask = mask_smooth(bbox_mask, output_feather)
            info_lines.append(f"Output feather: {output_feather}px")

        # ── Summary ────────────────────────────────────────────────────────────
        info_lines.append(
            f"Output: {B} frames, bbox={ref_w:.0f}x{ref_h:.0f}+{bbox_expand_pct:.0%} "
            f"expansion, clamp={'on' if clamp_to_image else 'off'}"
        )
        info = "\n".join(info_lines)
        print(f"{LOG_PREFIX} {info}")

        return (bbox_mask.to(intermediate), info)


# ── Registration ────────────────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    "NV_PointDrivenBBox": NV_PointDrivenBBox,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_PointDrivenBBox": "NV Point-Driven BBox",
}
