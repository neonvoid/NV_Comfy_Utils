"""
NV Mask Tracking BBox — Per-frame bounding box extraction with temporal smoothing.

Converts per-frame segmentation masks (e.g. from SAM3) into temporally-smooth
rectangular bounding box masks suitable for InpaintCrop's bounding_box_mask input.

Unlike masks_to_bboxes() in vace_control_video_prep.py, this node:
  - Does NOT snap to VAE stride boundaries (eliminates 8px quantization jitter)
  - Offers multiple smoothing algorithms including One-Euro filter (velocity-adaptive)
  - Supports "lock" modes that fix bbox dimensions while tracking position
  - Outputs both the bbox mask and a pass-through of the input tight mask

Smoothing modes:
  kalman_rts  - 8D Kalman + RTS backward smoother. Globally optimal, no causal lag.
  one_euro    - Velocity-adaptive: heavy smoothing when still, light when moving.
                Best for subjects with variable speed. Forward-pass only (causal).
  ema         - Bidirectional Exponential Moving Average. Simple single-parameter
                control via alpha. Zero lag from forward+backward averaging.
  gaussian    - Temporal Gaussian blur on coordinates. Smooth but lags fast motion.
  median      - Median filter then Gaussian smooth (from vace_control_video_prep).
                Best for removing outlier frames.
  lock_largest - Lock bbox WIDTH and HEIGHT to the largest frame's values.
                 Position (center) still smoothed with One-Euro. Gives constant crop
                 dimensions while tracking.
  lock_first  - Lock bbox WIDTH and HEIGHT to the first frame's values.
                 Position (center) still smoothed with One-Euro.
  none        - Raw per-frame bboxes with padding only. No temporal smoothing.

smooth_strength (0-1) lerps between raw and smoothed for any mode.

Note: Anomaly detection (occlusion/tracking-loss rejection) lives in
NV_InpaintCrop v2, not here. This node focuses purely on smoothing.
InpaintCrop is where the crop canvas is determined, so anomaly detection
there ensures a smooth, non-jarring canvas for inpaint workflows.
"""

import torch

from .mask_ops import mask_erode_dilate, mask_blur
from .bbox_ops import (
    extract_bboxes, smooth_coordinates, build_bbox_masks,
)


# =============================================================================
# Node Class
# =============================================================================

class NV_MaskTrackingBBox:
    """Extract per-frame bounding boxes from masks with temporal smoothing.

    Designed for video inpainting pipelines where SAM3/segmentation masks need
    to become temporally-stable bounding box masks for InpaintCrop.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK", {
                    "tooltip": "Per-frame segmentation masks [B, H, W]. "
                               "Each frame's non-zero region defines the tight subject boundary."
                }),
                "bbox_expand_pct": ("FLOAT", {
                    "default": 0.15, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Expand bbox by this fraction of its dimensions "
                               "(0.15 = 15% on each side). Applied after smoothing. "
                               "Ensures the crop includes context around the subject."
                }),
                "smooth_mode": ([
                    "kalman_rts", "one_euro", "ema", "gaussian", "median",
                    "lock_largest", "lock_first", "none"
                ], {
                    "default": "kalman_rts",
                    "tooltip": "kalman_rts: 8D Kalman + RTS backward smoother — globally optimal, "
                               "handles missing frames, no causal lag (RECOMMENDED). "
                               "one_euro: velocity-adaptive, forward-only (slight lag). "
                               "ema: bidirectional exponential moving average — simple, one param (alpha). "
                               "gaussian: temporal Gaussian blur on coordinates. "
                               "median: median filter + Gaussian (removes outlier frames). "
                               "lock_largest: fix bbox size to largest frame, smooth position. "
                               "lock_first: fix bbox size to first frame, smooth position. "
                               "none: raw per-frame bboxes, no smoothing."
                }),
                "smooth_window": ("INT", {
                    "default": 5, "min": 1, "max": 31, "step": 2,
                    "tooltip": "Window size for gaussian/median smoothing modes "
                               "(odd recommended). Ignored for one_euro, lock, and none modes."
                }),
            },
            "optional": {
                # Deprecated name (backward compat for old workflows)
                "padding": ("FLOAT", {
                    "default": 0.15, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "DEPRECATED — use bbox_expand_pct"
                }),
                "min_cutoff": ("FLOAT", {
                    "default": 0.05, "min": 0.001, "max": 10.0, "step": 0.01,
                    "tooltip": "One-Euro min_cutoff: minimum cutoff frequency. "
                               "Lower = more aggressive jitter removal when subject is still. "
                               "Used by one_euro, lock_largest, and lock_first modes. "
                               "0.01-0.05 = heavy smoothing, 0.1-1.0 = light smoothing."
                }),
                "beta": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 10.0, "step": 0.1,
                    "tooltip": "One-Euro beta: speed coefficient. "
                               "Higher = faster adaptation when subject moves quickly. "
                               "Used by one_euro, lock_largest, and lock_first modes. "
                               "0.0 = no speed adaptation, 0.5-1.0 = moderate, 2.0+ = aggressive."
                }),
                "ema_alpha": ("FLOAT", {
                    "default": 0.3, "min": 0.01, "max": 1.0, "step": 0.05,
                    "tooltip": "EMA smoothing factor. Lower = more smoothing (more memory of past). "
                               "0.1 = heavy smoothing, 0.3 = moderate (default), 0.5 = light, 0.9 = minimal. "
                               "Only used by 'ema' smooth_mode."
                }),
                "smooth_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Lerp between raw bbox (0.0) and fully smoothed (1.0). "
                               "Works with ALL smoothing modes. Use to dial in partial smoothing "
                               "without changing algorithm parameters. 0.5 = half the correction."
                }),
                "threshold": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Binarize input mask at 0.5 before bbox extraction. "
                               "Enable if mask has soft edges from resizing that inflate the bbox."
                }),
                "output_erode": ("INT", {
                    "default": 0, "min": -64, "max": 64, "step": 1,
                    "tooltip": "Erode (negative) or dilate (positive) the output bbox masks. "
                               "Applied after bbox construction. 0 = no change."
                }),
                "output_feather": ("INT", {
                    "default": 0, "min": 0, "max": 64, "step": 1,
                    "tooltip": "Feather (Gaussian blur) the output bbox mask edges. "
                               "0 = hard rectangular edges. 8-16 = soft transition."
                }),
                "kalman_q_pos": ("FLOAT", {
                    "default": 4.0, "min": 0.1, "max": 100.0, "step": 0.5,
                    "tooltip": "Kalman process noise for position acceleration. "
                               "Higher = trust measurements more (less smooth). "
                               "Lower = smoother trajectory (may lag real motion). "
                               "Default 4.0 tuned for SAM3 at 8fps."
                }),
                "kalman_q_dim": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 50.0, "step": 0.5,
                    "tooltip": "Kalman process noise for dimension (w/h) acceleration. "
                               "Lower than q_pos because bbox size changes slowly. "
                               "Default 1.0."
                }),
                "kalman_r_pos": ("FLOAT", {
                    "default": 9.0, "min": 0.1, "max": 200.0, "step": 1.0,
                    "tooltip": "Kalman measurement noise for center position. "
                               "Higher = smoother (treats measurements as noisier). "
                               "Default 9.0 = 3px standard deviation from SAM3 mask edges."
                }),
                "kalman_r_dim": ("FLOAT", {
                    "default": 25.0, "min": 0.1, "max": 400.0, "step": 1.0,
                    "tooltip": "Kalman measurement noise for width/height. "
                               "Higher = more stable bbox dimensions. "
                               "Default 25.0 = 5px standard deviation."
                }),
            },
        }

    RETURN_TYPES = ("MASK", "MASK", "STRING")
    RETURN_NAMES = ("bbox_mask", "tight_mask", "info")
    FUNCTION = "execute"
    CATEGORY = "NV_Utils/mask"
    DESCRIPTION = (
        "Extract per-frame bounding boxes from segmentation masks with temporal smoothing. "
        "Designed to sit between SAM3 and InpaintCrop: converts irregular per-frame masks "
        "into stable rectangular bbox masks. One-Euro filter provides velocity-adaptive "
        "smoothing (heavy when still, light when moving). Lock modes fix crop dimensions. "
        "Anomaly detection (occlusion rejection) is handled by InpaintCrop v2 downstream."
    )

    def execute(self, mask, bbox_expand_pct, smooth_mode, smooth_window,
                # Deprecated name (backward compat)
                padding=0.15,
                min_cutoff=0.05, beta=0.7,
                ema_alpha=0.3, smooth_strength=1.0,
                threshold=False, output_erode=0, output_feather=0,
                kalman_q_pos=4.0, kalman_q_dim=1.0,
                kalman_r_pos=9.0, kalman_r_dim=25.0):

        # Resolve deprecated param name
        from .mask_processing_config import resolve_deprecated
        bbox_expand_pct = resolve_deprecated(bbox_expand_pct, 0.15, padding, 0.15)

        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        B, H, W = mask.shape
        info_lines = [
            f"[NV_MaskTrackingBBox] {B} frames, {W}x{H}px | "
            f"mode={smooth_mode} | padding={bbox_expand_pct:.0%}"
        ]

        # --- Threshold ---
        work_mask = mask.clone()
        if threshold:
            work_mask = (work_mask > 0.5).float()
            info_lines.append("  Threshold: applied at 0.5")

        # --- Extract per-frame bboxes ---
        x1s, y1s, x2s, y2s, present = extract_bboxes(work_mask, info_lines)

        # All-empty check
        if not any(present):
            info_lines.append("  No mask content in any frame — returning zero masks")
            info = "\n".join(info_lines)
            print(info)
            tight_out = (mask > 0.5).float() if threshold else mask.clone()
            return (torch.zeros_like(mask), tight_out, info)

        # --- Smooth ---
        x1s, y1s, x2s, y2s = smooth_coordinates(
            x1s, y1s, x2s, y2s,
            smooth_mode, smooth_window,
            min_cutoff, beta,
            info_lines,
            present=present,
            q_pos=kalman_q_pos, q_dim=kalman_q_dim,
            r_pos=kalman_r_pos, r_dim=kalman_r_dim,
            ema_alpha=ema_alpha, smooth_strength=smooth_strength
        )

        # --- Build output bbox masks ---
        bbox_mask = build_bbox_masks(x1s, y1s, x2s, y2s, bbox_expand_pct, H, W, info_lines)

        # --- Optional post-processing ---
        if output_erode != 0:
            bbox_mask = mask_erode_dilate(bbox_mask, output_erode)
            info_lines.append(f"  Output erode: {output_erode}px")

        if output_feather > 0:
            bbox_mask = mask_blur(bbox_mask, output_feather)
            info_lines.append(f"  Output feather: {output_feather}px")

        bbox_mask = bbox_mask.clamp(0.0, 1.0)

        # --- Tight mask pass-through ---
        tight_out = (mask > 0.5).float() if threshold else mask.clone()

        info = "\n".join(info_lines)
        print(info)

        return (bbox_mask, tight_out, info)


# =============================================================================
# Node Registration
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "NV_MaskTrackingBBox": NV_MaskTrackingBBox,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_MaskTrackingBBox": "NV Mask Tracking BBox",
}
