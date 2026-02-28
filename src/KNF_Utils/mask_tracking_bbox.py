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
  one_euro    - Velocity-adaptive: heavy smoothing when still, light when moving.
                Best for subjects with variable speed. Forward-pass only (causal).
  gaussian    - Temporal Gaussian blur on coordinates. Smooth but lags fast motion.
  median      - Median filter then Gaussian smooth (from vace_control_video_prep).
                Best for removing outlier frames.
  lock_largest - Lock bbox WIDTH and HEIGHT to the largest frame's values.
                 Position (center) still smoothed with One-Euro. Gives constant crop
                 dimensions while tracking.
  lock_first  - Lock bbox WIDTH and HEIGHT to the first frame's values.
                 Position (center) still smoothed with One-Euro.
  none        - Raw per-frame bboxes with padding only. No temporal smoothing.
"""

import math
import torch

from .inpaint_crop import gaussian_smooth_1d, median_filter_1d, mask_erode_dilate, mask_blur


# =============================================================================
# One-Euro Filter (batch sequence variant)
# =============================================================================

def one_euro_smooth_1d(values: list, min_cutoff: float = 0.05,
                       beta: float = 0.7, d_cutoff: float = 1.0) -> list:
    """Apply One-Euro filter to a 1D sequence of values.

    Forward-pass only (causal). The slight lag is absorbed by bbox padding.

    The One-Euro filter adapts its smoothing strength based on signal velocity:
    slow/still → heavy smoothing (jitter removal), fast motion → light smoothing
    (responsive tracking). This makes it ideal for tracking bounding boxes where
    the subject alternates between still poses and rapid movement.

    Args:
        values: Raw coordinate values, one per frame.
        min_cutoff: Minimum cutoff frequency. Lower = more smoothing when still.
        beta: Speed coefficient. Higher = faster adaptation to velocity changes.
        d_cutoff: Derivative cutoff frequency. Usually 1.0.

    Returns:
        Filtered values as list, same length as input.
    """
    if len(values) <= 1:
        return list(values)

    def _smoothing_factor(cutoff: float) -> float:
        # t_e = 1.0 (uniform frame spacing), so r = 2*pi*cutoff
        r = 2.0 * math.pi * cutoff
        return r / (r + 1.0)

    result = [values[0]]
    dx_prev = 0.0

    for i in range(1, len(values)):
        x = values[i]
        x_prev = result[i - 1]

        # Derivative estimation (t_e = 1 frame, so dx/dt = dx)
        a_d = _smoothing_factor(d_cutoff)
        dx = x - x_prev
        dx_hat = a_d * dx + (1.0 - a_d) * dx_prev

        # Adaptive cutoff: faster motion → higher cutoff → less smoothing
        cutoff = min_cutoff + beta * abs(dx_hat)
        a = _smoothing_factor(cutoff)

        # Exponential smoothing
        x_hat = a * x + (1.0 - a) * x_prev

        result.append(x_hat)
        dx_prev = dx_hat

    return result


# =============================================================================
# Bbox Extraction
# =============================================================================

def extract_bboxes(mask: torch.Tensor, info_lines: list) -> tuple:
    """Extract per-frame tight bounding boxes from masks.

    Returns (x1s, y1s, x2s, y2s, present) where:
        - x1s/y1s: top-left corners (float)
        - x2s/y2s: bottom-right corners (float, exclusive)
        - present: which frames had non-empty masks

    Empty frames are forward/backward filled so smoothing has valid data everywhere.
    """
    B, H, W = mask.shape

    x1s, y1s, x2s, y2s = [], [], [], []
    present = []

    for b in range(B):
        ys, xs = torch.where(mask[b] > 0.5)
        if len(xs) == 0:
            present.append(False)
            x1s.append(0.0)
            y1s.append(0.0)
            x2s.append(0.0)
            y2s.append(0.0)
        else:
            present.append(True)
            x1s.append(float(xs.min().item()))
            y1s.append(float(ys.min().item()))
            x2s.append(float(xs.max().item() + 1))
            y2s.append(float(ys.max().item() + 1))

    # Forward/backward fill empty frames
    coords = [x1s, y1s, x2s, y2s]
    for c in coords:
        last_valid = None
        for i in range(B):
            if present[i]:
                last_valid = c[i]
            elif last_valid is not None:
                c[i] = last_valid
        last_valid = None
        for i in range(B - 1, -1, -1):
            if present[i]:
                last_valid = c[i]
            elif last_valid is not None:
                c[i] = last_valid

    num_present = sum(present)
    info_lines.append(f"  BBox extraction: {num_present}/{B} frames with mask content")

    return x1s, y1s, x2s, y2s, present


# =============================================================================
# Smoothing Dispatch
# =============================================================================

def smooth_coordinates(x1s, y1s, x2s, y2s, smooth_mode, smooth_window,
                       min_cutoff, beta, info_lines):
    """Apply temporal smoothing to bbox coordinate sequences.

    Args:
        x1s, y1s, x2s, y2s: Per-frame corner coordinates.
        smooth_mode: One of "one_euro", "gaussian", "median",
                     "lock_largest", "lock_first", "none".
        smooth_window: Window size for gaussian/median modes.
        min_cutoff: One-Euro min_cutoff (used by one_euro and lock modes).
        beta: One-Euro beta (used by one_euro and lock modes).
        info_lines: Diagnostic output accumulator.

    Returns:
        Smoothed (x1s, y1s, x2s, y2s).
    """
    B = len(x1s)

    if smooth_mode == "none" or B <= 1:
        info_lines.append("  Smoothing: none")
        return x1s, y1s, x2s, y2s

    if smooth_mode == "one_euro":
        x1s = one_euro_smooth_1d(x1s, min_cutoff, beta)
        y1s = one_euro_smooth_1d(y1s, min_cutoff, beta)
        x2s = one_euro_smooth_1d(x2s, min_cutoff, beta)
        y2s = one_euro_smooth_1d(y2s, min_cutoff, beta)
        info_lines.append(
            f"  Smoothing: one_euro (min_cutoff={min_cutoff}, beta={beta})"
        )

    elif smooth_mode == "gaussian":
        win = min(smooth_window, B)
        x1s = gaussian_smooth_1d(x1s, win)
        y1s = gaussian_smooth_1d(y1s, win)
        x2s = gaussian_smooth_1d(x2s, win)
        y2s = gaussian_smooth_1d(y2s, win)
        info_lines.append(f"  Smoothing: gaussian (window={win})")

    elif smooth_mode == "median":
        win = min(smooth_window, B)
        x1s = gaussian_smooth_1d(median_filter_1d(x1s, win), win)
        y1s = gaussian_smooth_1d(median_filter_1d(y1s, win), win)
        x2s = gaussian_smooth_1d(median_filter_1d(x2s, win), win)
        y2s = gaussian_smooth_1d(median_filter_1d(y2s, win), win)
        info_lines.append(f"  Smoothing: median + gaussian (window={win})")

    elif smooth_mode in ("lock_largest", "lock_first"):
        # Convert corners to center + size
        cxs = [(x1 + x2) / 2.0 for x1, x2 in zip(x1s, x2s)]
        cys = [(y1 + y2) / 2.0 for y1, y2 in zip(y1s, y2s)]
        ws = [x2 - x1 for x1, x2 in zip(x1s, x2s)]
        hs = [y2 - y1 for y1, y2 in zip(y1s, y2s)]

        # Lock dimensions to reference
        if smooth_mode == "lock_largest":
            ref_w = max(ws)
            ref_h = max(hs)
            info_lines.append(
                f"  Lock: largest frame dims ({ref_w:.0f}x{ref_h:.0f}px)"
            )
        else:  # lock_first
            ref_w = ws[0]
            ref_h = hs[0]
            info_lines.append(
                f"  Lock: first frame dims ({ref_w:.0f}x{ref_h:.0f}px)"
            )

        # Smooth center position with One-Euro
        cxs = one_euro_smooth_1d(cxs, min_cutoff, beta)
        cys = one_euro_smooth_1d(cys, min_cutoff, beta)
        info_lines.append(
            f"  Position smoothing: one_euro "
            f"(min_cutoff={min_cutoff}, beta={beta})"
        )

        # Convert back to corners with locked dimensions
        x1s = [cx - ref_w / 2.0 for cx in cxs]
        y1s = [cy - ref_h / 2.0 for cy in cys]
        x2s = [cx + ref_w / 2.0 for cx in cxs]
        y2s = [cy + ref_h / 2.0 for cy in cys]

    return x1s, y1s, x2s, y2s


# =============================================================================
# Bbox Mask Builder
# =============================================================================

def build_bbox_masks(x1s, y1s, x2s, y2s, padding, H, W, info_lines):
    """Build [B, H, W] binary bbox masks from smoothed coordinates.

    No VAE stride snapping — uses int(round()) for sub-pixel rounding.
    InpaintCrop handles its own sizing downstream.
    """
    B = len(x1s)
    result = torch.zeros(B, H, W)
    bbox_sizes = []

    for b in range(B):
        bx1, by1, bx2, by2 = x1s[b], y1s[b], x2s[b], y2s[b]
        bw = bx2 - bx1
        bh = by2 - by1

        if bw < 1.0 or bh < 1.0:
            continue

        # Apply padding (percentage of bbox dimensions)
        pad_x = bw * padding
        pad_y = bh * padding
        bx1 -= pad_x
        by1 -= pad_y
        bx2 += pad_x
        by2 += pad_y

        # Clamp to image bounds (integer rounding, NO VAE stride snapping)
        bx1 = max(0, int(round(bx1)))
        by1 = max(0, int(round(by1)))
        bx2 = min(W, int(round(bx2)))
        by2 = min(H, int(round(by2)))

        if bx2 > bx1 and by2 > by1:
            result[b, by1:by2, bx1:bx2] = 1.0
            bbox_sizes.append((bx2 - bx1, by2 - by1))

    if bbox_sizes:
        avg_w = sum(s[0] for s in bbox_sizes) / len(bbox_sizes)
        avg_h = sum(s[1] for s in bbox_sizes) / len(bbox_sizes)
        min_w = min(s[0] for s in bbox_sizes)
        max_w = max(s[0] for s in bbox_sizes)
        min_h = min(s[1] for s in bbox_sizes)
        max_h = max(s[1] for s in bbox_sizes)
        info_lines.append(
            f"  BBox size: avg {avg_w:.0f}x{avg_h:.0f}px, "
            f"range W=[{min_w},{max_w}] H=[{min_h},{max_h}] "
            f"(padding={padding:.0%})"
        )

    return result


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
                "padding": ("FLOAT", {
                    "default": 0.15, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Expand bbox by this fraction of its dimensions "
                               "(0.15 = 15% on each side). Applied after smoothing. "
                               "Ensures the crop includes context around the subject."
                }),
                "smooth_mode": ([
                    "one_euro", "gaussian", "median",
                    "lock_largest", "lock_first", "none"
                ], {
                    "default": "one_euro",
                    "tooltip": "one_euro: velocity-adaptive smoothing (best default). "
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
        "smoothing (heavy when still, light when moving). Lock modes fix crop dimensions."
    )

    def execute(self, mask, padding, smooth_mode, smooth_window,
                min_cutoff=0.05, beta=0.7, threshold=False,
                output_erode=0, output_feather=0):
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        B, H, W = mask.shape
        info_lines = [
            f"[NV_MaskTrackingBBox] {B} frames, {W}x{H}px | "
            f"mode={smooth_mode} | padding={padding:.0%}"
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
            info_lines
        )

        # --- Build output bbox masks ---
        bbox_mask = build_bbox_masks(x1s, y1s, x2s, y2s, padding, H, W, info_lines)

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
