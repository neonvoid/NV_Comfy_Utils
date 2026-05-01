"""
Shared bounding box operations — extraction, smoothing, fill, anomaly detection.

Single source of truth for bbox coordinate operations used across:
  - NV_MaskTrackingBBox
  - NV_InpaintCrop2
  - NV_VaceControlVideoPrep
  - NV_TemporalMaskStabilizer

Smoothing functions live here because they only operate on bbox coordinate sequences.
"""

import math
import torch
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter


# WAN 2.1 VAE spatial compression factor — used by VAE-grid alignment helpers below.
VAE_STRIDE = 8


def snap_to_vae_grid(x, y, w, h, spatial_h_px, spatial_w_px, stride=VAE_STRIDE):
    """Snap pixel-space crop coordinates to VAE grid boundaries.

    Origin is floor-snapped, dimensions are ceil-snapped, then clamped to image bounds.
    Both origin and final dimensions are guaranteed to be multiples of stride.
    Returns (x, y, w, h) as ints, all multiples of stride.

    Args:
        x, y, w, h: pixel-space crop rectangle. May be float (e.g. from auto-resolution
                    helpers) — dimensions are ceil-snapped to preserve "fully cover the
                    requested region" semantics; origin is floor-snapped.
        spatial_h_px, spatial_w_px: the source image dimensions in pixels (may not be
                    multiples of stride — function clamps so the snapped crop never
                    extends past the largest stride-aligned origin).
        stride: VAE spatial stride. Defaults to VAE_STRIDE (8 for WAN 2.1).
    """
    if stride <= 0:
        raise ValueError(f"snap_to_vae_grid: stride must be positive (got {stride})")
    if spatial_w_px < stride or spatial_h_px < stride:
        raise ValueError(
            f"snap_to_vae_grid: image too small for stride {stride} "
            f"(got spatial={spatial_w_px}x{spatial_h_px})"
        )

    # Origin: floor-snap. math.floor handles negative floats correctly (int() truncates
    # toward zero, which would mis-snap negative inputs — defensive).
    x = (math.floor(float(x)) // stride) * stride
    y = (math.floor(float(y)) // stride) * stride

    # Dimensions: ceil-snap. Floats from auto-resolution helpers (e.g. 8.1) must round
    # UP to the next stride multiple (16), not get truncated to 8.
    w = max(stride, math.ceil(float(w) / stride) * stride)
    h = max(stride, math.ceil(float(h) / stride) * stride)

    # Clamp origin to the largest stride-aligned position that still leaves room for
    # at least one stride. Old code used `spatial_w_px - stride` directly, which yielded
    # non-aligned origins when spatial_w_px % stride != 0 (e.g. 1025 → 1017, not /8).
    max_x = ((spatial_w_px - stride) // stride) * stride
    max_y = ((spatial_h_px - stride) // stride) * stride
    x = max(0, min(x, max_x))
    y = max(0, min(y, max_y))

    # Clamp dimensions: floor-snap remaining width/height. Non-stride-multiple image dims
    # mean the largest representable crop is the largest stride multiple ≤ available room.
    if x + w > spatial_w_px:
        w = ((spatial_w_px - x) // stride) * stride
    if y + h > spatial_h_px:
        h = ((spatial_h_px - y) // stride) * stride
    return int(x), int(y), int(w), int(h)


# =============================================================================
# Forward/Backward Fill (single implementation for all callers)
# =============================================================================

def forward_backward_fill(values, present):
    """Fill missing entries in a list using nearest valid neighbors.

    Args:
        values: list of values (one per frame). Missing frames have placeholder values.
        present: list of bool — True if frame has valid data, False if placeholder.

    Returns:
        values (modified in place) with missing frames filled from nearest neighbors.
    """
    B = len(values)
    last_valid = None
    for i in range(B):
        if present[i]:
            last_valid = values[i]
        elif last_valid is not None:
            values[i] = last_valid
    last_valid = None
    for i in range(B - 1, -1, -1):
        if present[i]:
            last_valid = values[i]
        elif last_valid is not None:
            values[i] = last_valid
    return values


# =============================================================================
# Bbox Extraction
# =============================================================================

def find_bbox(mask):
    """Find bounding box of non-zero mask region. Returns (x, y, w, h) or None if empty.

    Works on 2D [H, W] or 3D [B, H, W] masks (uses first frame if batched).
    """
    if mask.dim() == 3:
        mask_2d = mask[0]
    else:
        mask_2d = mask

    non_zero = torch.nonzero(mask_2d > 0.01)

    if non_zero.numel() == 0:
        return None

    y_min = non_zero[:, 0].min().item()
    y_max = non_zero[:, 0].max().item()
    x_min = non_zero[:, 1].min().item()
    x_max = non_zero[:, 1].max().item()

    return (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)


def extract_bboxes(mask, info_lines=None):
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
    for c in [x1s, y1s, x2s, y2s]:
        forward_backward_fill(c, present)

    num_present = sum(present)
    if info_lines is not None:
        info_lines.append(f"  BBox extraction: {num_present}/{B} frames with mask content")

    return x1s, y1s, x2s, y2s, present


def compute_union_bbox(masks, padding_frac=0.1):
    """Compute the union bounding box across all frames of a mask batch.

    Args:
        masks: [B, H, W] float tensor
        padding_frac: fractional padding to add around the union bbox

    Returns:
        (x, y, w, h) tuple or None if all masks are empty
    """
    B, H, W = masks.shape
    union_mask = (masks > 0.01).any(dim=0)  # [H, W]
    non_zero = torch.nonzero(union_mask)

    if non_zero.numel() == 0:
        return None

    y_min = non_zero[:, 0].min().item()
    y_max = non_zero[:, 0].max().item()
    x_min = non_zero[:, 1].min().item()
    x_max = non_zero[:, 1].max().item()

    bw = x_max - x_min + 1
    bh = y_max - y_min + 1

    pad_x = int(bw * padding_frac)
    pad_y = int(bh * padding_frac)

    x = max(0, x_min - pad_x)
    y = max(0, y_min - pad_y)
    x2 = min(W, x_max + 1 + pad_x)
    y2 = min(H, y_max + 1 + pad_y)

    return (x, y, x2 - x, y2 - y)


# =============================================================================
# Bbox Mask Builder
# =============================================================================

def build_bbox_masks(x1s, y1s, x2s, y2s, padding, H, W, info_lines=None,
                     vae_stride=0):
    """Build [B, H, W] binary bbox masks from smoothed coordinates.

    Args:
        vae_stride: if >0, snap boundaries to this stride. 0 = no snapping (sub-pixel round).
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

        if vae_stride > 0:
            bx1 = max(0, (int(bx1) // vae_stride) * vae_stride)
            by1 = max(0, (int(by1) // vae_stride) * vae_stride)
            bx2 = min(W, ((int(bx2) + vae_stride - 1) // vae_stride) * vae_stride)
            by2 = min(H, ((int(by2) + vae_stride - 1) // vae_stride) * vae_stride)
        else:
            bx1 = max(0, int(round(bx1)))
            by1 = max(0, int(round(by1)))
            bx2 = min(W, int(round(bx2)))
            by2 = min(H, int(round(by2)))

        if bx2 > bx1 and by2 > by1:
            result[b, by1:by2, bx1:bx2] = 1.0
            bbox_sizes.append((bx2 - bx1, by2 - by1))

    if bbox_sizes and info_lines is not None:
        avg_w = sum(s[0] for s in bbox_sizes) / len(bbox_sizes)
        avg_h = sum(s[1] for s in bbox_sizes) / len(bbox_sizes)
        min_w = min(s[0] for s in bbox_sizes)
        max_w = max(s[0] for s in bbox_sizes)
        min_h = min(s[1] for s in bbox_sizes)
        max_h = max(s[1] for s in bbox_sizes)
        snap_str = f", aligned to {vae_stride}px" if vae_stride > 0 else ""
        info_lines.append(
            f"  BBox size: avg {avg_w:.0f}x{avg_h:.0f}px, "
            f"range W=[{min_w},{max_w}] H=[{min_h},{max_h}] "
            f"(padding={padding:.0%}{snap_str})"
        )

    return result


# =============================================================================
# Anomaly Detection
# =============================================================================

def detect_bbox_anomalies(bboxes, threshold, info_lines):
    """Detect and reject anomalous bounding boxes (occlusion, tracking loss).

    Computes per-frame area and center-displacement relative to stable statistics.
    Anomalous frames are set to None and forward/backward filled from neighbors.

    Args:
        bboxes: List of (x, y, w, h) tuples or None per frame.
        threshold: Sensitivity (lower = stricter). 0.0 disables detection.
        info_lines: Diagnostic output accumulator.

    Returns:
        New list of (x, y, w, h) tuples with anomalous frames filled from neighbors.
    """
    if threshold <= 0.0:
        return bboxes

    valid = [(i, b) for i, b in enumerate(bboxes) if b is not None]

    if len(valid) < 3:
        return bboxes

    areas = []
    centers = []
    for _, (x, y, w, h) in valid:
        areas.append(max(float(w * h), 1.0))
        centers.append((x + w / 2.0, y + h / 2.0))

    sorted_areas = sorted(areas)
    median_area = sorted_areas[len(sorted_areas) // 2]

    if median_area < 1.0:
        return bboxes

    flagged = []
    scores = {}

    for idx in range(len(valid)):
        frame_idx = valid[idx][0]
        area = areas[idx]

        area_delta = abs(area - median_area) / median_area

        if idx > 0:
            dx = centers[idx][0] - centers[idx - 1][0]
            dy = centers[idx][1] - centers[idx - 1][1]
            center_jump = math.sqrt(dx * dx + dy * dy) / math.sqrt(median_area)
        else:
            center_jump = 0.0

        score = max(area_delta, center_jump)
        scores[frame_idx] = (area_delta, center_jump, score)

        if score > threshold:
            flagged.append(frame_idx)

    if not flagged:
        info_lines.append(f"Anomaly detection: clean (threshold={threshold:.2f})")
        return bboxes

    info_lines.append(
        f"Anomaly detection: {len(flagged)} frame(s) rejected "
        f"(threshold={threshold:.2f})"
    )

    # Compact run reporting
    runs = []
    run_start = flagged[0]
    run_end = flagged[0]
    for f in flagged[1:]:
        if f == run_end + 1:
            run_end = f
        else:
            runs.append((run_start, run_end))
            run_start = f
            run_end = f
    runs.append((run_start, run_end))

    for start, end in runs:
        if start == end:
            ad, cj, sc = scores[start]
            info_lines.append(
                f"  frame {start}: area_delta={ad:.2f}, "
                f"center_jump={cj:.2f}, score={sc:.2f}"
            )
        else:
            info_lines.append(f"  frames {start}-{end} ({end - start + 1} frames)")

    result = list(bboxes)
    for f in flagged:
        result[f] = None

    # Forward/backward fill None entries
    B = len(result)
    last_valid = None
    for i in range(B):
        if result[i] is not None:
            last_valid = result[i]
        elif last_valid is not None:
            result[i] = last_valid
    last_valid = None
    for i in range(B - 1, -1, -1):
        if result[i] is not None:
            last_valid = result[i]
        elif last_valid is not None:
            result[i] = last_valid

    remaining = sum(1 for b in result if b is not None)
    info_lines.append(
        f"  {remaining}/{B} frames remain after rejection + fill"
    )

    return result


# =============================================================================
# 1D Smoothing Functions
# =============================================================================

def gaussian_smooth_1d(values, window_size):
    """Apply Gaussian smoothing to 1D sequence."""
    if len(values) <= 1:
        return values
    sigma = window_size / 4.0
    smoothed = gaussian_filter1d(values, sigma, mode='nearest')
    return smoothed.tolist()


def median_filter_1d(values, window_size):
    """Apply median filter to 1D sequence."""
    if len(values) <= 1:
        return values
    filtered = median_filter(np.array(values), size=window_size, mode='nearest')
    return filtered.tolist()


def one_euro_smooth_1d(values, min_cutoff=0.05, beta=0.7, d_cutoff=1.0):
    """Apply One-Euro filter to a 1D sequence of values.

    Forward-pass only (causal). Adapts smoothing based on signal velocity:
    slow/still -> heavy smoothing, fast motion -> light smoothing.
    """
    if len(values) <= 1:
        return list(values)

    def _smoothing_factor(cutoff):
        r = 2.0 * math.pi * cutoff
        return r / (r + 1.0)

    result = [values[0]]
    dx_prev = 0.0

    for i in range(1, len(values)):
        x = values[i]
        x_prev = result[i - 1]

        a_d = _smoothing_factor(d_cutoff)
        dx = x - x_prev
        dx_hat = a_d * dx + (1.0 - a_d) * dx_prev

        cutoff = min_cutoff + beta * abs(dx_hat)
        a = _smoothing_factor(cutoff)

        x_hat = a * x + (1.0 - a) * x_prev

        result.append(x_hat)
        dx_prev = dx_hat

    return result


def ema_smooth_1d(values, alpha=0.3, bidirectional=True):
    """Apply Exponential Moving Average to a 1D sequence.

    Bidirectional mode runs forward + backward passes and averages for zero lag.
    """
    if len(values) <= 1:
        return list(values)

    fwd = [values[0]]
    for i in range(1, len(values)):
        fwd.append(alpha * values[i] + (1.0 - alpha) * fwd[i - 1])

    if not bidirectional:
        return fwd

    bwd = [0.0] * len(values)
    bwd[-1] = values[-1]
    for i in range(len(values) - 2, -1, -1):
        bwd[i] = alpha * values[i] + (1.0 - alpha) * bwd[i + 1]

    return [(f + b) / 2.0 for f, b in zip(fwd, bwd)]


# =============================================================================
# Smoothing Dispatch
# =============================================================================

def smooth_coordinates(x1s, y1s, x2s, y2s, smooth_mode, smooth_window,
                       min_cutoff, beta, info_lines,
                       present=None, q_pos=4.0, q_dim=1.0,
                       r_pos=9.0, r_dim=25.0,
                       ema_alpha=0.3, smooth_strength=1.0):
    """Apply temporal smoothing to bbox coordinate sequences.

    Supports: kalman_rts, one_euro, ema, gaussian, median, lock_largest, lock_first, none.
    """
    from .kalman_rts_smoother import kalman_rts_smooth

    B = len(x1s)

    if smooth_mode == "none" or B <= 1:
        info_lines.append("  Smoothing: none")
        return x1s, y1s, x2s, y2s

    # Save raw values for smooth_strength lerp
    raw_x1s, raw_y1s = list(x1s), list(y1s)
    raw_x2s, raw_y2s = list(x2s), list(y2s)

    if smooth_mode == "kalman_rts":
        if B < 10:
            info_lines.append(
                f"  Smoothing: kalman_rts -> fallback to one_euro ({B} frames < 10)"
            )
            smooth_mode = "one_euro"
        else:
            if present is None:
                present = [True] * B
            x1s, y1s, x2s, y2s = kalman_rts_smooth(
                x1s, y1s, x2s, y2s, present,
                q_pos=q_pos, q_dim=q_dim, r_pos=r_pos, r_dim=r_dim
            )
            info_lines.append(
                f"  Smoothing: kalman_rts (q_pos={q_pos}, q_dim={q_dim}, "
                f"r_pos={r_pos}, r_dim={r_dim})"
            )
            if smooth_strength < 1.0:
                s = smooth_strength
                x1s = [r + s * (sm - r) for r, sm in zip(raw_x1s, x1s)]
                y1s = [r + s * (sm - r) for r, sm in zip(raw_y1s, y1s)]
                x2s = [r + s * (sm - r) for r, sm in zip(raw_x2s, x2s)]
                y2s = [r + s * (sm - r) for r, sm in zip(raw_y2s, y2s)]
                info_lines.append(f"  Strength: {smooth_strength:.0%} (lerp raw->smoothed)")
            return x1s, y1s, x2s, y2s

    if smooth_mode == "one_euro":
        x1s = one_euro_smooth_1d(x1s, min_cutoff, beta)
        y1s = one_euro_smooth_1d(y1s, min_cutoff, beta)
        x2s = one_euro_smooth_1d(x2s, min_cutoff, beta)
        y2s = one_euro_smooth_1d(y2s, min_cutoff, beta)
        info_lines.append(
            f"  Smoothing: one_euro (min_cutoff={min_cutoff}, beta={beta})"
        )

    elif smooth_mode == "ema":
        x1s = ema_smooth_1d(x1s, ema_alpha, bidirectional=True)
        y1s = ema_smooth_1d(y1s, ema_alpha, bidirectional=True)
        x2s = ema_smooth_1d(x2s, ema_alpha, bidirectional=True)
        y2s = ema_smooth_1d(y2s, ema_alpha, bidirectional=True)
        info_lines.append(f"  Smoothing: ema (alpha={ema_alpha}, bidirectional)")

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
        cxs = [(x1 + x2) / 2.0 for x1, x2 in zip(x1s, x2s)]
        cys = [(y1 + y2) / 2.0 for y1, y2 in zip(y1s, y2s)]
        ws = [x2 - x1 for x1, x2 in zip(x1s, x2s)]
        hs = [y2 - y1 for y1, y2 in zip(y1s, y2s)]

        if smooth_mode == "lock_largest":
            ref_w = max(ws)
            ref_h = max(hs)
            info_lines.append(f"  Lock: largest frame dims ({ref_w:.0f}x{ref_h:.0f}px)")
        else:
            ref_w = ws[0]
            ref_h = hs[0]
            info_lines.append(f"  Lock: first frame dims ({ref_w:.0f}x{ref_h:.0f}px)")

        cxs = one_euro_smooth_1d(cxs, min_cutoff, beta)
        cys = one_euro_smooth_1d(cys, min_cutoff, beta)
        info_lines.append(
            f"  Position smoothing: one_euro (min_cutoff={min_cutoff}, beta={beta})"
        )

        x1s = [cx - ref_w / 2.0 for cx in cxs]
        y1s = [cy - ref_h / 2.0 for cy in cys]
        x2s = [cx + ref_w / 2.0 for cx in cxs]
        y2s = [cy + ref_h / 2.0 for cy in cys]

    # Apply smooth_strength lerp
    if smooth_strength < 1.0:
        s = smooth_strength
        x1s = [r + s * (sm - r) for r, sm in zip(raw_x1s, x1s)]
        y1s = [r + s * (sm - r) for r, sm in zip(raw_y1s, y1s)]
        x2s = [r + s * (sm - r) for r, sm in zip(raw_x2s, x2s)]
        y2s = [r + s * (sm - r) for r, sm in zip(raw_y2s, y2s)]
        info_lines.append(f"  Strength: {smooth_strength:.0%} (lerp raw->smoothed)")

    return x1s, y1s, x2s, y2s


# =============================================================================
# Expand-Crop-Trim (shared canvas expansion for content stabilization)
# =============================================================================

def expand_crop_trim(canvas, ctc_x, ctc_y, ctc_w, ctc_h,
                     target_w, target_h, dx, dy,
                     resize_algorithm, device):
    """Expand a crop region from canvas, apply translation, trim back to target size.

    Used by both InpaintCrop2 centroid stabilization and CoTrackerBridge.
    Returns (warped_image_nchw, trim_left, trim_top, expanded_w, expanded_h)
    or None if displacement is zero.
    """
    from .mask_ops import rescale_image

    canvas_h, canvas_w = canvas.shape[1], canvas.shape[2]
    sx = target_w / ctc_w
    sy = target_h / ctc_h

    # Available margins in target space
    avail_l = ctc_x * sx
    avail_r = (canvas_w - ctc_x - ctc_w) * sx
    avail_t = ctc_y * sy
    avail_b = (canvas_h - ctc_y - ctc_h) * sy

    # Clamp displacement
    dx = max(-avail_l, min(avail_r, dx))
    dy = max(-avail_t, min(avail_b, dy))
    clamped = (dx != dx or dy != dy)  # will be False, just for API

    # Per-side margins
    need_l = (int(math.ceil(abs(dx))) + 1) if dx < 0 else 1
    need_r = (int(math.ceil(abs(dx))) + 1) if dx > 0 else 1
    need_t = (int(math.ceil(abs(dy))) + 1) if dy < 0 else 1
    need_b = (int(math.ceil(abs(dy))) + 1) if dy > 0 else 1

    mc_l = int(math.ceil(need_l / sx)) + 1
    mc_r = int(math.ceil(need_r / sx)) + 1
    mc_t = int(math.ceil(need_t / sy)) + 1
    mc_b = int(math.ceil(need_b / sy)) + 1

    ex = max(0, ctc_x - mc_l)
    ey = max(0, ctc_y - mc_t)
    ex2 = min(canvas_w, ctc_x + ctc_w + mc_r)
    ey2 = min(canvas_h, ctc_y + ctc_h + mc_b)

    actual_l_c = ctc_x - ex
    actual_t_c = ctc_y - ey
    total_c_w = ex2 - ex
    total_c_h = ey2 - ey
    expanded_w = int(round(total_c_w * sx))
    expanded_h = int(round(total_c_h * sy))
    trim_left = int(round(actual_l_c * sx))
    trim_top = int(round(actual_t_c * sy))

    exp_img = canvas[:, ey:ey2, ex:ex2, :]
    exp_img = rescale_image(exp_img, expanded_w, expanded_h, resize_algorithm)

    return exp_img, trim_left, trim_top, expanded_w, expanded_h, dx, dy


# =============================================================================
# Trajectory Debug (shared between bbox-driving nodes)
# =============================================================================

def print_bbox_trajectory_debug(positions, compare_positions=None, compare_label="",
                                visibility=None, min_visibility=0.5, log_prefix="[debug]",
                                top_spikes=10):
    """Print diagnostic stats about a bbox-center trajectory.

    Shared across NV_MaskTrackingBBox, NV_PointDrivenBBox, and any other node
    that drives a crop bbox from a per-frame (x, y) trajectory. Provides A/B
    parity when comparing different bbox sources.

    Blocks printed:
      1. Per-frame motion distribution (mean/median/max/std/p95/p99 of |Δ|)
      2. Spike frames where |Δ| > mean + 2σ (top N by magnitude)
      3. Optional divergence block — if compare_positions is provided,
         prints mean/max distance between the two trajectories at each frame.
         Useful for:
           - NV_PointDrivenBBox: tracker vs mask-centroid (are they consistent?)
           - NV_MaskTrackingBBox: smoothed vs raw centroid (how much work is the filter doing?)
      4. Cumulative displacement (start, end, arc-length, arc/line ratio)

    Args:
        positions: list of (x, y) floats per frame — primary trajectory
        compare_positions: optional list of (x, y) — second trajectory for divergence
        compare_label: section heading for the divergence block (e.g. "Smoothed ↔ Raw")
        visibility: optional list of floats [0,1] per frame — flags interpolated frames
        min_visibility: threshold below which a frame is marked interpolated
        log_prefix: line prefix for console output (e.g. "[NV_MaskTrackingBBox]")
        top_spikes: how many spike frames to print (default 10)
    """
    import math as _math

    B = len(positions)
    if B < 2:
        print(f"{log_prefix} [debug] Single frame — no trajectory to analyze.")
        return

    # ── Frame-to-frame delta magnitudes ────────────────────────────────────────
    deltas = []
    for i in range(1, B):
        dx = positions[i][0] - positions[i - 1][0]
        dy = positions[i][1] - positions[i - 1][1]
        mag = _math.sqrt(dx * dx + dy * dy)
        deltas.append((i, dx, dy, mag))

    mags = [d[3] for d in deltas]
    mean_mag = sum(mags) / len(mags)
    var_mag = sum((m - mean_mag) ** 2 for m in mags) / len(mags)
    std_mag = _math.sqrt(var_mag)
    max_mag = max(mags)
    sorted_mags = sorted(mags)
    median_mag = sorted_mags[len(sorted_mags) // 2]
    p95_mag = sorted_mags[min(len(sorted_mags) - 1, int(0.95 * len(sorted_mags)))]
    p99_mag = sorted_mags[min(len(sorted_mags) - 1, int(0.99 * len(sorted_mags)))]

    print(f"\n{log_prefix} === VERBOSE DEBUG: BBox Trajectory ({B} frames) ===")
    print(f"{log_prefix} Frame-to-frame Δ magnitude (bbox center):")
    print(f"{log_prefix}   mean   = {mean_mag:.3f} px")
    print(f"{log_prefix}   median = {median_mag:.3f} px")
    print(f"{log_prefix}   max    = {max_mag:.3f} px")
    print(f"{log_prefix}   std    = {std_mag:.3f} px")
    print(f"{log_prefix}   p95    = {p95_mag:.3f} px")
    print(f"{log_prefix}   p99    = {p99_mag:.3f} px")

    # ── Spike frames (|Δ| > mean + 2σ) ─────────────────────────────────────────
    spike_threshold = mean_mag + 2.0 * std_mag
    spikes = [d for d in deltas if d[3] > spike_threshold]
    spikes_sorted = sorted(spikes, key=lambda d: -d[3])[:top_spikes]
    print(f"{log_prefix} Spike frames (|Δ| > {spike_threshold:.2f} = mean + 2σ): "
          f"{len(spikes)} total, showing top {len(spikes_sorted)}")
    for (i, dx, dy, mag) in spikes_sorted:
        vis_tag = ""
        if visibility is not None:
            v = visibility[i] if i < len(visibility) else 1.0
            if v < min_visibility:
                vis_tag = f" [INTERPOLATED: vis={v:.2f}]"
        print(f"{log_prefix}   Frame {i - 1}→{i}: Δ=({dx:+.2f}, {dy:+.2f}) "
              f"|Δ|={mag:.2f} px{vis_tag}")

    # ── Optional: divergence between two trajectories ──────────────────────────
    if compare_positions is not None and len(compare_positions) == B:
        divergences = []
        for b in range(B):
            px, py = positions[b]
            cx, cy = compare_positions[b]
            dx = px - cx
            dy = py - cy
            divergences.append(_math.sqrt(dx * dx + dy * dy))

        mean_div = sum(divergences) / len(divergences)
        max_div = max(divergences)
        max_div_frame = divergences.index(max_div)
        label = compare_label or "Trajectory divergence"
        print(f"{log_prefix} {label}:")
        print(f"{log_prefix}   mean = {mean_div:.2f} px")
        print(f"{log_prefix}   max  = {max_div:.2f} px (frame {max_div_frame})")

    # ── Cumulative displacement ────────────────────────────────────────────────
    start_x, start_y = positions[0]
    end_x, end_y = positions[-1]
    total_dx = end_x - start_x
    total_dy = end_y - start_y
    straight_line = _math.sqrt(total_dx * total_dx + total_dy * total_dy)
    arc_length = sum(mags)
    print(f"{log_prefix} Cumulative displacement:")
    print(f"{log_prefix}   start   = ({start_x:.1f}, {start_y:.1f})")
    print(f"{log_prefix}   end     = ({end_x:.1f}, {end_y:.1f})")
    print(f"{log_prefix}   straight-line: {straight_line:.1f} px")
    print(f"{log_prefix}   arc length:    {arc_length:.1f} px")
    ratio = arc_length / max(straight_line, 1e-6)
    print(f"{log_prefix}   arc/line ratio: {ratio:.2f}x "
          f"(high ratio = lots of wobble relative to net motion)")
    print(f"{log_prefix} === END DEBUG ===\n")
