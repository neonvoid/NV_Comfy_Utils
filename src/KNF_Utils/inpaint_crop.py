"""
NV Inpaint Crop v2 - Clean, focused inpainting crop node

Crops an image around a masked region for inpainting, with optional bounding box control.

Key features:
- Two mask outputs: original (for tight stitching) and processed (for diffusion)
- bounding_box_mask for explicit crop region control
- Advanced mask processing using scipy grey morphology
- Stitch uses original mask for minimal changes to the image
"""

import math
import time
import torch
import comfy.model_management
import comfy.utils
import nodes
import numpy as np
import scipy.ndimage
from scipy.ndimage import gaussian_filter1d, median_filter
import torchvision.transforms.v2 as T
import torch.nn.functional as TF


# =============================================================================
# Content Stabilization (within-crop pixel locking)
# =============================================================================

def _compute_mask_centroids(masks):
    """Compute weighted centroid per frame from cropped masks [B, H, W]."""
    B, H, W = masks.shape
    cx_list, cy_list, valid = [], [], []
    ys = torch.arange(H, device=masks.device, dtype=masks.dtype)
    xs = torch.arange(W, device=masks.device, dtype=masks.dtype)

    for b in range(B):
        m = masks[b]
        total = m.sum()
        if total < 1.0:
            valid.append(False)
            cx_list.append(W / 2.0)
            cy_list.append(H / 2.0)
        else:
            cy = float((m.sum(dim=1) * ys).sum() / total)
            cx = float((m.sum(dim=0) * xs).sum() / total)
            valid.append(True)
            cx_list.append(cx)
            cy_list.append(cy)

    # Forward/backward fill invalid frames
    last = None
    for i in range(B):
        if valid[i]:
            last = (cx_list[i], cy_list[i])
        elif last is not None:
            cx_list[i], cy_list[i] = last
    last = None
    for i in range(B - 1, -1, -1):
        if valid[i]:
            last = (cx_list[i], cy_list[i])
        elif last is not None:
            cx_list[i], cy_list[i] = last

    return cx_list, cy_list, valid


def _smooth_centroid_trajectory(cx_list, cy_list, valid):
    """Smooth centroid trajectory using Kalman+RTS (>=10 frames) or One-Euro."""
    B = len(cx_list)
    if B <= 2:
        return cx_list, cy_list

    if B >= 10:
        from .kalman_rts_smoother import kalman_rts_smooth
        # Fake-bbox trick: x1=cx, x2=cx+1 — only position matters
        x2s = [cx + 1.0 for cx in cx_list]
        y2s = [cy + 1.0 for cy in cy_list]
        sx1, sy1, _, _ = kalman_rts_smooth(
            cx_list, cy_list, x2s, y2s, valid,
            q_pos=4.0, q_dim=0.01, r_pos=9.0, r_dim=0.01
        )
        return sx1, sy1

    # Short sequences: One-Euro (lazy import to avoid circular dependency)
    from .mask_tracking_bbox import one_euro_smooth_1d
    return one_euro_smooth_1d(cx_list, 0.05, 0.7), one_euro_smooth_1d(cy_list, 0.05, 0.7)


def _build_translation_grid(dx, dy, H, W, device):
    """Build affine_grid for pure translation. dx/dy in pixels."""
    norm_dx = 2.0 * dx / W
    norm_dy = 2.0 * dy / H
    theta = torch.tensor([
        [1.0, 0.0, norm_dx],
        [0.0, 1.0, norm_dy]
    ], device=device, dtype=torch.float32).unsqueeze(0)
    return TF.affine_grid(theta, (1, 1, H, W), align_corners=False)


def _stabilize_content_centroid(images, masks_orig, masks_proc, stitcher=None,
                                target_w=None, target_h=None, resize_algorithm='lanczos',
                                strength=1.0):
    """Centroid-lock stabilization: translate each frame to lock mask centroid.

    Uses expand-crop-trim to avoid edge spillage: expands each frame from
    its stored canvas by the max displacement margin, warps at the expanded
    resolution, then center-crops back to target size.  All output pixels
    come from real image content — no padding artifacts.

    Args:
        images: [B, H, W, C] cropped frames at target resolution
        masks_orig: [B, H, W] original masks
        masks_proc: [B, H, W] processed masks
        stitcher: STITCHER dict with canvas_image and crop coordinates
        target_w, target_h: target output dimensions
        resize_algorithm: algorithm for rescaling expanded crops
        strength: 0.0-2.0, controls how aggressively centroids are locked.
            0.0 = no correction.
            0.5 = smoothed centroids only (conservative).
            1.0 = raw centroids (full lock, default).
            >1.0 = overcorrect (compensate for centroid bias).

    Returns:
        (warped_images, warped_masks_orig, warped_masks_proc, warp_data)
    """
    B, H, W, C = images.shape
    device = images.device

    cx_raw, cy_raw, valid = _compute_mask_centroids(masks_orig)
    cx_smooth, cy_smooth = _smooth_centroid_trajectory(cx_raw, cy_raw, valid)

    # Reference = temporal median of smoothed centroids (stable anchor)
    ref_cx = float(sorted(cx_smooth)[B // 2])
    ref_cy = float(sorted(cy_smooth)[B // 2])

    # Effective centroids: interpolate between smoothed and raw based on strength.
    # strength=0 → smoothed (conservative), strength=1 → raw (full lock),
    # strength>1 → extrapolate past raw (overcorrect).
    cx_eff = [cx_smooth[b] + (cx_raw[b] - cx_smooth[b]) * strength for b in range(B)]
    cy_eff = [cy_smooth[b] + (cy_raw[b] - cy_smooth[b]) * strength for b in range(B)]

    # Compute max displacement to determine expansion margin
    max_dx = max(abs(cx - ref_cx) for cx in cx_eff)
    max_dy = max(abs(cy - ref_cy) for cy in cy_eff)
    margin = int(math.ceil(max(max_dx, max_dy))) + 1

    # Determine target dimensions
    tw = target_w if target_w is not None else W
    th = target_h if target_h is not None else H

    # Can we do expand-crop-trim? Need stitcher with canvas data.
    use_expansion = (margin > 0 and stitcher is not None
                     and len(stitcher.get('canvas_image', [])) == B)

    warp_data = []
    warped_imgs, warped_mo, warped_mp = [], [], []
    n_clamped = 0

    for b in range(B):
        dx = cx_eff[b] - ref_cx
        dy = cy_eff[b] - ref_cy

        if use_expansion:
            # Expand: re-crop a larger region from the stored canvas
            canvas = stitcher['canvas_image'][b].to(device)
            if canvas.dim() == 3:
                canvas = canvas.unsqueeze(0)  # [1, H_c, W_c, C]
            ctc_x = stitcher['cropped_to_canvas_x'][b]
            ctc_y = stitcher['cropped_to_canvas_y'][b]
            ctc_w = stitcher['cropped_to_canvas_w'][b]
            ctc_h = stitcher['cropped_to_canvas_h'][b]
            canvas_h, canvas_w = canvas.shape[1], canvas.shape[2]

            # Scale factors: canvas-space → target-space
            sx = tw / ctc_w
            sy = th / ctc_h

            # Available canvas margin per side (in target-space pixels)
            avail_l = ctc_x * sx
            avail_r = (canvas_w - ctc_x - ctc_w) * sx
            avail_t = ctc_y * sy
            avail_b = (canvas_h - ctc_y - ctc_h) * sy

            # Clamp displacement to available canvas margins so we never
            # need content beyond what the canvas provides.
            # dx > 0 → grid samples rightward → needs right-side content
            dx_orig, dy_orig = dx, dy
            dx = max(-avail_l, min(avail_r, dx))
            dy = max(-avail_t, min(avail_b, dy))
            if dx != dx_orig or dy != dy_orig:
                n_clamped += 1

            # Per-side margins: just enough for the clamped displacement + 1px safety
            need_l = (int(math.ceil(abs(dx))) + 1) if dx < 0 else 1
            need_r = (int(math.ceil(abs(dx))) + 1) if dx > 0 else 1
            need_t = (int(math.ceil(abs(dy))) + 1) if dy < 0 else 1
            need_b = (int(math.ceil(abs(dy))) + 1) if dy > 0 else 1

            # Canvas-space margins per side (proportional to target margins)
            mc_l = int(math.ceil(need_l / sx)) + 1
            mc_r = int(math.ceil(need_r / sx)) + 1
            mc_t = int(math.ceil(need_t / sy)) + 1
            mc_b = int(math.ceil(need_b / sy)) + 1

            # Expanded crop region in canvas space (safety-clamped)
            ex = max(0, ctc_x - mc_l)
            ey = max(0, ctc_y - mc_t)
            ex2 = min(canvas_w, ctc_x + ctc_w + mc_r)
            ey2 = min(canvas_h, ctc_y + ctc_h + mc_b)

            # Actual canvas margins obtained
            actual_l_c = ctc_x - ex
            actual_t_c = ctc_y - ey

            # Target-space dimensions of expanded image (proportional resize)
            total_c_w = ex2 - ex
            total_c_h = ey2 - ey
            expanded_w_b = int(round(total_c_w * sx))
            expanded_h_b = int(round(total_c_h * sy))

            # Trim offsets: where the crop content starts in the expanded image
            trim_left = int(round(actual_l_c * sx))
            trim_top = int(round(actual_t_c * sy))

            # Crop from canvas and resize (proportional — no spatial distortion)
            exp_img = canvas[:, ey:ey2, ex:ex2, :]
            exp_img = rescale_image(exp_img, expanded_w_b, expanded_h_b, resize_algorithm)

            # Masks: pad at target resolution with per-side margins
            pad_right = max(0, expanded_w_b - trim_left - tw)
            pad_bottom = max(0, expanded_h_b - trim_top - th)
            mo_padded = TF.pad(masks_orig[b].unsqueeze(0).unsqueeze(0),
                               (trim_left, pad_right, trim_top, pad_bottom),
                               mode='constant', value=0).squeeze(0).squeeze(0)
            mp_padded = TF.pad(masks_proc[b].unsqueeze(0).unsqueeze(0),
                               (trim_left, pad_right, trim_top, pad_bottom),
                               mode='constant', value=0).squeeze(0).squeeze(0)

            img_nchw = exp_img.permute(0, 3, 1, 2)  # [1, C, eH, eW]
            mo_4d = mo_padded.unsqueeze(0).unsqueeze(0)  # [1, 1, eH, eW]
            mp_4d = mp_padded.unsqueeze(0).unsqueeze(0)

            grid = _build_translation_grid(dx, dy, expanded_h_b, expanded_w_b, device)

            wi = TF.grid_sample(img_nchw, grid, mode='bilinear',
                                padding_mode='zeros', align_corners=False)
            wmo = TF.grid_sample(mo_4d, grid, mode='bilinear',
                                 padding_mode='zeros', align_corners=False)
            wmp = TF.grid_sample(mp_4d, grid, mode='bilinear',
                                 padding_mode='zeros', align_corners=False)

            # Trim back to target resolution at computed offsets
            wi = wi[:, :, trim_top:trim_top+th, trim_left:trim_left+tw]
            wmo = wmo[:, :, trim_top:trim_top+th, trim_left:trim_left+tw]
            wmp = wmp[:, :, trim_top:trim_top+th, trim_left:trim_left+tw]
        else:
            # No expansion available — pass through without stabilization
            wi = images[b:b+1].permute(0, 3, 1, 2)
            wmo = masks_orig[b:b+1].unsqueeze(1)
            wmp = masks_proc[b:b+1].unsqueeze(1)
            dx, dy = 0.0, 0.0

        warp_data.append({"dx": dx, "dy": dy})
        warped_imgs.append(wi.permute(0, 2, 3, 1).squeeze(0))
        warped_mo.append(wmo.squeeze(0).squeeze(0))
        warped_mp.append(wmp.squeeze(0).squeeze(0))

    max_disp = max(max(abs(d["dx"]) for d in warp_data), max(abs(d["dy"]) for d in warp_data))
    mode_str = "expand-crop-trim" if use_expansion else "no-stitcher"
    clamp_str = f", {n_clamped} edge-clamped" if n_clamped > 0 else ""
    print(f"[ContentStabilize/centroid] {B} frames, strength={strength:.2f}, "
          f"ref=({ref_cx:.1f},{ref_cy:.1f}), max_disp={max_disp:.1f}px, "
          f"margin={margin}px ({mode_str}{clamp_str})")

    return (torch.stack(warped_imgs), torch.stack(warped_mo), torch.stack(warped_mp), warp_data)


# Module-level RAFT model cache
_raft_model = None


def _get_raft_model(device):
    """Lazy-load and cache RAFT-small model."""
    global _raft_model
    if _raft_model is None:
        from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
        _raft_model = raft_small(weights=Raft_Small_Weights.DEFAULT)
        _raft_model.eval()
        print("[ContentStabilize/flow] Loaded RAFT-small model")
    return _raft_model.to(device)


def _warp_with_flow(img_nchw, flow, padding_mode='border'):
    """Warp image using optical flow field.

    Args:
        img_nchw: [1, C, H, W]
        flow: [1, 2, H, W] or [2, H, W] — (dx, dy) pixel displacement
    """
    if flow.dim() == 3:
        flow = flow.unsqueeze(0)
    _, _, H, W = img_nchw.shape
    device = img_nchw.device

    y = torch.arange(H, device=device, dtype=torch.float32)
    x = torch.arange(W, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing='ij')

    xx_warped = xx + flow[0, 0]
    yy_warped = yy + flow[0, 1]

    # Normalize to [-1, 1]
    xx_norm = 2.0 * xx_warped / (W - 1) - 1.0
    yy_norm = 2.0 * yy_warped / (H - 1) - 1.0
    grid = torch.stack([xx_norm, yy_norm], dim=-1).unsqueeze(0)

    return TF.grid_sample(img_nchw, grid, mode='bilinear', padding_mode=padding_mode, align_corners=True)


def _stabilize_content_flow(images, masks_orig, masks_proc):
    """Optical flow stabilization: RAFT-based alignment to reference frame.

    Args:
        images: [B, H, W, C] cropped frames
        masks_orig: [B, H, W]
        masks_proc: [B, H, W]

    Returns:
        (warped_images, warped_masks_orig, warped_masks_proc, warp_data)
    """
    B, H, W, C = images.shape
    device = images.device
    ref_idx = B // 2

    # Convert to NCHW [0,255] for RAFT
    imgs_nchw = images.permute(0, 3, 1, 2) * 255.0  # [B, C, H, W]

    # Pad to multiple of 8 if needed
    pad_h = (8 - H % 8) % 8
    pad_w = (8 - W % 8) % 8
    if pad_h > 0 or pad_w > 0:
        imgs_nchw = TF.pad(imgs_nchw, (0, pad_w, 0, pad_h), mode='replicate')

    model = _get_raft_model(device)

    # Compute pairwise forward flows
    pairwise_flows = []
    with torch.no_grad():
        for i in range(B - 1):
            flow_list = model(imgs_nchw[i:i+1], imgs_nchw[i+1:i+1+1])
            flow = flow_list[-1]  # [1, 2, H_pad, W_pad]
            # Crop back to original size
            if pad_h > 0 or pad_w > 0:
                flow = flow[:, :, :H, :W]
            pairwise_flows.append(flow.squeeze(0))  # [2, H, W]

    # Accumulate flow to reference frame
    accumulated = []
    for i in range(B):
        if i == ref_idx:
            accumulated.append(torch.zeros(2, H, W, device=device))
        elif i < ref_idx:
            # Chain forward: frame i → i+1 → ... → ref
            acc = torch.zeros(2, H, W, device=device)
            for j in range(i, ref_idx):
                acc = acc + pairwise_flows[j]
            accumulated.append(acc)
        else:
            # Chain backward: frame i → i-1 → ... → ref (negate forward flows)
            acc = torch.zeros(2, H, W, device=device)
            for j in range(ref_idx, i):
                acc = acc - pairwise_flows[j]
            accumulated.append(acc)

    # Warp all frames to reference
    warp_data = []
    warped_imgs, warped_mo, warped_mp = [], [], []

    for b in range(B):
        flow = accumulated[b]
        warp_data.append({"flow": flow.cpu()})  # Store on CPU for stitcher

        # Warp image (negate flow: forward flow points src→ref, but grid_sample
        # needs backward mapping ref→src, so we sample from src at (x - flow_x))
        neg_flow = -flow
        img_nchw_b = images[b:b+1].permute(0, 3, 1, 2)
        wi = _warp_with_flow(img_nchw_b, neg_flow, padding_mode='reflection')
        warped_imgs.append(wi.permute(0, 2, 3, 1).squeeze(0))

        # Warp masks
        mo_4d = masks_orig[b:b+1].unsqueeze(1)
        wmo = _warp_with_flow(mo_4d, neg_flow, padding_mode='zeros')
        warped_mo.append(wmo.squeeze(0).squeeze(0))

        mp_4d = masks_proc[b:b+1].unsqueeze(1)
        wmp = _warp_with_flow(mp_4d, neg_flow, padding_mode='zeros')
        warped_mp.append(wmp.squeeze(0).squeeze(0))

    max_flow = max(f.abs().max().item() for f in accumulated)
    print(f"[ContentStabilize/flow] {B} frames, ref={ref_idx}, max_flow={max_flow:.1f}px")

    return (torch.stack(warped_imgs), torch.stack(warped_mo), torch.stack(warped_mp), warp_data)


# =============================================================================
# WAN Resolution Presets
# =============================================================================

WAN_PRESETS = {
    "WAN_480p": {"pixels": 399_360, "divisor": 16, "min_side": 256, "max_side": 1024},
    "WAN_720p": {"pixels": 921_600, "divisor": 16, "min_side": 384, "max_side": 1536},
}


def compute_auto_resolution(bbox_aspect, preset_name, padding_multiple):
    """
    Compute optimal target resolution from bbox aspect ratio and WAN preset.

    Given a pixel budget and aspect ratio, solves:
        width * height = target_pixels
        width / height = aspect
    => width = sqrt(pixels * aspect), height = sqrt(pixels / aspect)

    Then snaps to the stricter of the preset's divisor or the padding_multiple.
    """
    config = WAN_PRESETS[preset_name]
    pixels = config["pixels"]
    divisor = max(config["divisor"], padding_multiple) if padding_multiple > 0 else config["divisor"]
    min_side = config["min_side"]
    max_side = config["max_side"]

    # Clamp extreme aspect ratios (beyond 3:1 or 1:3 is rarely useful for diffusion)
    bbox_aspect = max(1/3, min(3.0, bbox_aspect))

    # Compute raw dimensions
    raw_w = math.sqrt(pixels * bbox_aspect)
    raw_h = math.sqrt(pixels / bbox_aspect)

    # Snap to divisor
    w = round(raw_w / divisor) * divisor
    h = round(raw_h / divisor) * divisor

    # Clamp to min/max
    w = max(min_side, min(max_side, w))
    h = max(min_side, min(max_side, h))

    # Re-snap after clamp (clamp might have broken divisibility)
    w = round(w / divisor) * divisor
    h = round(h / divisor) * divisor

    return int(w), int(h)


# =============================================================================
# Utility Functions
# =============================================================================

def gaussian_smooth_1d(values, window_size):
    """Apply Gaussian smoothing to 1D sequence for crop stabilization."""
    if len(values) <= 1:
        return values
    sigma = window_size / 4.0
    smoothed = gaussian_filter1d(values, sigma, mode='nearest')
    return smoothed.tolist()


def median_filter_1d(values, window_size):
    """Apply median filter to 1D sequence for crop stabilization."""
    if len(values) <= 1:
        return values
    filtered = median_filter(np.array(values), size=window_size, mode='nearest')
    return filtered.tolist()


def rescale_image(samples, width, height, algorithm='bicubic'):
    """Resize image tensor [B, H, W, C] using GPU."""
    algorithm_map = {
        'nearest': 'nearest',
        'bilinear': 'bilinear',
        'bicubic': 'bicubic',
        'lanczos': 'bicubic',
        'area': 'area',
    }
    mode = algorithm_map.get(algorithm.lower(), 'bicubic')

    # [B, H, W, C] -> [B, C, H, W]
    samples = samples.movedim(-1, 1)

    samples = torch.nn.functional.interpolate(
        samples,
        size=(height, width),
        mode=mode,
        align_corners=False if mode in ['bilinear', 'bicubic'] else None
    )

    # [B, C, H, W] -> [B, H, W, C]
    return samples.movedim(1, -1)


def rescale_mask(samples, width, height, algorithm='bilinear'):
    """Resize mask tensor [B, H, W] using GPU."""
    algorithm_map = {
        'nearest': 'nearest',
        'bilinear': 'bilinear',
        'bicubic': 'bicubic',
        'lanczos': 'bicubic',
        'area': 'area',
    }
    mode = algorithm_map.get(algorithm.lower(), 'bilinear')

    # [B, H, W] -> [B, 1, H, W]
    samples = samples.unsqueeze(1)

    samples = torch.nn.functional.interpolate(
        samples,
        size=(height, width),
        mode=mode,
        align_corners=False if mode in ['bilinear', 'bicubic'] else None
    )

    # [B, 1, H, W] -> [B, H, W]
    return samples.squeeze(1)


# =============================================================================
# Mask Processing Functions (using scipy grey morphology)
# =============================================================================

def mask_erode_dilate(mask, amount):
    """
    Erode (negative) or dilate (positive) mask using scipy grey morphology.
    Grey operations preserve grayscale gradients unlike binary operations.
    """
    if amount == 0:
        return mask

    device = mask.device
    results = []

    for m in mask:
        m_np = m.cpu().numpy()
        if amount < 0:
            # Erosion (shrink mask)
            m_np = scipy.ndimage.grey_erosion(m_np, size=(-amount, -amount))
        else:
            # Dilation (expand mask)
            m_np = scipy.ndimage.grey_dilation(m_np, size=(amount, amount))
        results.append(torch.from_numpy(m_np).to(device))

    return torch.stack(results, dim=0)


def mask_fill_holes(mask, size):
    """
    Fill holes in mask using grey closing (dilation followed by erosion).
    Better than binary fill for soft/gradient masks.
    """
    if size == 0:
        return mask

    device = mask.device
    results = []

    for m in mask:
        m_np = m.cpu().numpy()
        m_np = scipy.ndimage.grey_closing(m_np, size=(size, size))
        results.append(torch.from_numpy(m_np).to(device))

    return torch.stack(results, dim=0)


def mask_remove_noise(mask, size):
    """
    Remove isolated pixels/noise using grey opening (erosion followed by dilation).
    Eliminates small specks while preserving larger regions.
    """
    if size == 0:
        return mask

    device = mask.device
    results = []

    for m in mask:
        m_np = m.cpu().numpy()
        m_np = scipy.ndimage.grey_opening(m_np, size=(size, size))
        results.append(torch.from_numpy(m_np).to(device))

    return torch.stack(results, dim=0)


def mask_smooth(mask, amount):
    """
    Smooth mask edges by binarizing then blurring.
    Creates cleaner, crisper edges than direct blur.
    """
    if amount == 0:
        return mask

    if amount % 2 == 0:
        amount += 1

    # Binarize first (threshold at 0.5)
    binary = mask > 0.5

    # Then blur
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)

    smoothed = T.functional.gaussian_blur(binary.unsqueeze(1).float(), amount).squeeze(1)

    return smoothed


def mask_blur(mask, amount):
    """
    Direct Gaussian blur on mask (preserves gradients).
    Used for blend feathering.
    """
    if amount == 0:
        return mask

    if amount % 2 == 0:
        amount += 1

    if mask.dim() == 2:
        mask = mask.unsqueeze(0)

    blurred = T.functional.gaussian_blur(mask.unsqueeze(1), amount).squeeze(1)

    return blurred


def detect_bbox_anomalies(bboxes, threshold, info_lines):
    """Detect and reject anomalous bounding boxes (occlusion, tracking loss).

    Computes per-frame area and center-displacement relative to stable statistics.
    Anomalous frames are set to None and forward/backward filled from neighbors,
    so downstream smoothing never sees the bad data.

    Metrics:
        area_delta  = |area[t] - median_area| / median_area
        center_jump = euclidean(center[t], center[t-1]) / sqrt(median_area)
    A frame is flagged if max(area_delta, center_jump) > threshold.

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

    # Compute per-frame area and center
    areas = []
    centers = []
    for _, (x, y, w, h) in valid:
        areas.append(max(float(w * h), 1.0))
        centers.append((x + w / 2.0, y + h / 2.0))

    # Stable reference: median area
    sorted_areas = sorted(areas)
    median_area = sorted_areas[len(sorted_areas) // 2]

    if median_area < 1.0:
        return bboxes

    # Score each frame
    flagged = []
    scores = {}  # frame_idx -> (area_delta, center_jump, score)

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

    # Report
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

    # Reject: set flagged frames to None
    result = list(bboxes)
    for f in flagged:
        result[f] = None

    # Forward/backward fill None entries from nearest valid neighbor
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


def find_bbox(mask):
    """Find bounding box of non-zero mask region. Returns (x, y, w, h) or None if empty."""
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


def pad_to_multiple(value, multiple):
    """Round up value to nearest multiple."""
    if multiple <= 0:
        return value
    return int(math.ceil(value / multiple) * multiple)


# =============================================================================
# Core Crop Function
# =============================================================================

def crop_for_inpaint(image, original_mask, processed_mask, bbox_x, bbox_y, bbox_w, bbox_h,
                     target_w, target_h, padding_multiple, resize_algorithm):
    """
    Crop image and masks around bounding box, adjusting for target aspect ratio.

    Returns:
        canvas_image: Expanded image (may be larger than original if bbox doesn't fit)
        cto_x, cto_y, cto_w, cto_h: Where original image sits in canvas
        cropped_image: The cropped and resized region
        cropped_mask_original: Original mask cropped and resized (for tight stitching)
        cropped_mask_processed: Processed mask cropped and resized (for diffusion)
        ctc_x, ctc_y, ctc_w, ctc_h: Where crop region sits in canvas
    """
    B, img_h, img_w, C = image.shape
    device = image.device

    # Ensure target dimensions are multiples of padding
    if padding_multiple > 0:
        target_w = pad_to_multiple(target_w, padding_multiple)
        target_h = pad_to_multiple(target_h, padding_multiple)

    # Calculate target aspect ratio
    target_aspect = target_w / target_h
    bbox_aspect = bbox_w / bbox_h

    # Grow bbox to match target aspect ratio, centered on bbox center.
    # Use float center to avoid ±1px oscillation from integer division.
    bbox_cx = bbox_x + bbox_w / 2.0
    bbox_cy = bbox_y + bbox_h / 2.0

    if bbox_aspect < target_aspect:
        # Need wider - grow width
        new_w = int(round(bbox_h * target_aspect))
        new_h = bbox_h
    else:
        # Need taller - grow height
        new_w = bbox_w
        new_h = int(round(bbox_w / target_aspect))

    new_x = int(round(bbox_cx - new_w / 2.0))
    new_y = int(round(bbox_cy - new_h / 2.0))

    # Try to keep within image bounds by shifting
    if new_x < 0 and new_x + new_w <= img_w:
        new_x = 0
    elif new_x + new_w > img_w and new_x >= 0:
        new_x = img_w - new_w

    if new_y < 0 and new_y + new_h <= img_h:
        new_y = 0
    elif new_y + new_h > img_h and new_y >= 0:
        new_y = img_h - new_h

    # Calculate padding needed if bbox still exceeds image
    pad_left = max(0, -new_x)
    pad_right = max(0, new_x + new_w - img_w)
    pad_top = max(0, -new_y)
    pad_bottom = max(0, new_y + new_h - img_h)

    # Create canvas (expanded image)
    canvas_w = img_w + pad_left + pad_right
    canvas_h = img_h + pad_top + pad_bottom

    canvas_image = torch.zeros((B, canvas_h, canvas_w, C), device=device, dtype=image.dtype)
    canvas_mask_orig = torch.zeros((B, canvas_h, canvas_w), device=device, dtype=original_mask.dtype)
    canvas_mask_proc = torch.zeros((B, canvas_h, canvas_w), device=device, dtype=processed_mask.dtype)

    # Place original image and masks in canvas
    canvas_image[:, pad_top:pad_top+img_h, pad_left:pad_left+img_w, :] = image
    canvas_mask_orig[:, pad_top:pad_top+img_h, pad_left:pad_left+img_w] = original_mask
    canvas_mask_proc[:, pad_top:pad_top+img_h, pad_left:pad_left+img_w] = processed_mask

    # Fill edges by replicating border pixels
    if pad_top > 0:
        canvas_image[:, :pad_top, pad_left:pad_left+img_w, :] = image[:, 0:1, :, :].expand(-1, pad_top, -1, -1)
    if pad_bottom > 0:
        canvas_image[:, -pad_bottom:, pad_left:pad_left+img_w, :] = image[:, -1:, :, :].expand(-1, pad_bottom, -1, -1)
    if pad_left > 0:
        canvas_image[:, :, :pad_left, :] = canvas_image[:, :, pad_left:pad_left+1, :].expand(-1, -1, pad_left, -1)
    if pad_right > 0:
        canvas_image[:, :, -pad_right:, :] = canvas_image[:, :, -pad_right-1:-pad_right, :].expand(-1, -1, pad_right, -1)

    # Canvas-to-original coordinates
    cto_x, cto_y = pad_left, pad_top
    cto_w, cto_h = img_w, img_h

    # Crop coordinates in canvas space
    ctc_x = new_x + pad_left
    ctc_y = new_y + pad_top
    ctc_w, ctc_h = new_w, new_h

    # Crop from canvas
    cropped_image = canvas_image[:, ctc_y:ctc_y+ctc_h, ctc_x:ctc_x+ctc_w, :]
    cropped_mask_orig = canvas_mask_orig[:, ctc_y:ctc_y+ctc_h, ctc_x:ctc_x+ctc_w]
    cropped_mask_proc = canvas_mask_proc[:, ctc_y:ctc_y+ctc_h, ctc_x:ctc_x+ctc_w]

    # Resize to target dimensions
    if target_w != ctc_w or target_h != ctc_h:
        cropped_image = rescale_image(cropped_image, target_w, target_h, resize_algorithm)
        cropped_mask_orig = rescale_mask(cropped_mask_orig, target_w, target_h, resize_algorithm)
        cropped_mask_proc = rescale_mask(cropped_mask_proc, target_w, target_h, resize_algorithm)

    return (canvas_image, cto_x, cto_y, cto_w, cto_h,
            cropped_image, cropped_mask_orig, cropped_mask_proc,
            ctc_x, ctc_y, ctc_w, ctc_h)


# =============================================================================
# Main Node Class
# =============================================================================

class NV_InpaintCrop:
    """
    Crops an image around a masked region for inpainting.

    Outputs TWO masks:
    - cropped_mask: Original mask (unprocessed) - use for tight stitching
    - cropped_mask_processed: With all processing ops - use for diffusion inpainting

    The stitcher uses the original mask for blending, minimizing changes to the image.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),

                "target_mode": (["manual", "auto"], {
                    "default": "manual",
                    "tooltip": "manual: use target_width/height below. auto: compute optimal resolution from bbox aspect ratio and preset."
                }),
                "target_width": ("INT", {
                    "default": 512, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 8,
                    "tooltip": "Output width for cropped region (manual mode). Ignored when target_mode=auto."
                }),
                "target_height": ("INT", {
                    "default": 512, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 8,
                    "tooltip": "Output height for cropped region (manual mode). Ignored when target_mode=auto."
                }),
                "auto_preset": (list(WAN_PRESETS.keys()), {
                    "default": "WAN_480p",
                    "tooltip": "Resolution preset for auto mode. WAN_480p: ~400k pixels (832x480 scale). WAN_720p: ~920k pixels (1280x720 scale)."
                }),
                "padding_multiple": (["0", "8", "16", "32", "64"], {
                    "default": "32",
                    "tooltip": "Round output dimensions to multiples of this value. Use 8 for most models, 32 for latent-space operations, 64 for maximum compatibility."
                }),

                # Mask processing (applied to processed mask only)
                "mask_erode_dilate": ("INT", {
                    "default": 0, "min": -128, "max": 128, "step": 1,
                    "tooltip": "Shrink (negative) or expand (positive) the mask using grey morphology. "
                               "Recommended: -8 to -16 for tighter face boundaries, +8 to +16 for object removal safety margin, "
                               "+32 to +64 for large area inpainting. Uses scipy grey_erosion/dilation to preserve gradient edges."
                }),
                "mask_fill_holes": ("INT", {
                    "default": 0, "min": 0, "max": 128, "step": 1,
                    "tooltip": "Fill gaps/holes in mask using morphological closing (dilate then erode). "
                               "Recommended: 8-16 for small gaps between strokes, 32-64 for medium holes, 64+ for large interior gaps. "
                               "Useful for masks with unwanted holes from segmentation."
                }),
                "mask_remove_noise": ("INT", {
                    "default": 0, "min": 0, "max": 32, "step": 1,
                    "tooltip": "Remove isolated pixels/specks using morphological opening (erode then dilate). "
                               "Recommended: 2-4 for tiny specks, 8-16 for larger noise clusters. "
                               "Keeps main mask regions intact while eliminating stray pixels."
                }),
                "mask_smooth": ("INT", {
                    "default": 0, "min": 0, "max": 127, "step": 1,
                    "tooltip": "Smooth jagged mask edges by binarizing (threshold 0.5) then Gaussian blurring. "
                               "Recommended: 3-9 for subtle smoothing, 15-31 for noticeable softening. "
                               "Creates cleaner edges than direct blur. Value must be odd (auto-adjusted if even)."
                }),

                # Blend settings (for stitching)
                "stitch_source": (["tight", "processed", "bbox"], {
                    "default": "tight",
                    "tooltip": "Which mask to use as the base for stitch blending. "
                               "tight: original unprocessed mask (minimal changes to image). "
                               "processed: mask after erode/dilate/fill/smooth (matches diffusion mask). "
                               "bbox: full crop region (blends in the entire re-generated area)."
                }),
                "mask_blend_pixels": ("INT", {
                    "default": 16, "min": 0, "max": 64, "step": 1,
                    "tooltip": "Feather the stitch mask edges for seamless stitching (dilate + blur). "
                               "Recommended: 8-16 for subtle blending, 24-32 for visible seam hiding, 48-64 for aggressive blending."
                }),

                "resize_algorithm": (["bicubic", "bilinear", "nearest", "area"], {
                    "default": "bicubic",
                    "tooltip": "Interpolation for resizing. bicubic: best quality (smooth), bilinear: fast/good, "
                               "nearest: preserves hard edges (pixel art), area: best for downscaling."
                }),
            },
            "optional": {
                "mask_config": ("MASK_PROCESSING_CONFIG", {
                    "tooltip": "Optional shared config from NV_MaskProcessingConfig. "
                               "When connected, overrides this node's local mask processing widgets "
                               "(erode_dilate, fill_holes, remove_noise, smooth, blend_pixels)."
                }),
                "bounding_box_mask": ("MASK", {
                    "tooltip": "Optional mask defining minimum crop area. Crop region will encompass this entire mask. "
                               "Use to ensure specific areas are included even if main mask is smaller. "
                               "Main mask must be fully contained within this bounding box mask."
                }),

                # Video stabilization
                "stabilize_crop": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable temporal stabilization for video batches. Smooths crop position/size across frames to reduce jitter."
                }),
                "stabilization_mode": (["smooth", "lock_first", "lock_largest", "median"], {
                    "default": "smooth",
                    "tooltip": "smooth: Gaussian filter on bbox coords (gentle motion). lock_first: Use first frame's size for all. "
                               "lock_largest: Use largest bbox size for all (prevents clipping). median: Median filter (removes outliers)."
                }),
                "smooth_window": ("INT", {
                    "default": 5, "min": 3, "max": 21, "step": 2,
                    "tooltip": "Window size for temporal smoothing. Larger = more stable but less responsive. "
                               "3-5: subtle stabilization, 7-11: moderate smoothing, 13-21: heavy stabilization."
                }),
                "anomaly_threshold": ("FLOAT", {
                    "default": 1.5, "min": 0.0, "max": 10.0, "step": 0.1,
                    "tooltip": "Reject frames where bbox jumps exceed this threshold "
                               "(occlusion, tracking loss, someone walking in front). "
                               "Rejected frames are filled from neighbors before stabilization. "
                               "Ensures a smooth, non-jarring crop canvas for inpaint workflows. "
                               "0.0 = disabled. 1.0 = strict. 1.5 = moderate (recommended). 3.0 = lenient."
                }),
                "content_stabilize": (["off", "centroid", "optical_flow"], {
                    "default": "off",
                    "tooltip": "Stabilize content WITHIN the crop to pixel-lock the subject for denoise. "
                               "off: no content warp. "
                               "centroid: translate to lock mask centroid (fast, translation only). "
                               "optical_flow: RAFT-based alignment (slower, handles rotation + deformation)."
                }),
                "stabilize_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Content stabilization strength (centroid mode). "
                               "0.0 = no correction. "
                               "0.5 = conservative (smoothed centroids only, less jitter in corrections). "
                               "1.0 = full lock (raw centroids, maximum stabilization). "
                               "1.0-2.0 = overcorrect (can compensate for systematic centroid bias)."
                }),
            }
        }

    RETURN_TYPES = ("STITCHER", "IMAGE", "MASK", "MASK", "STRING")
    RETURN_NAMES = ("stitcher", "cropped_image", "cropped_mask", "cropped_mask_processed", "info")
    FUNCTION = "crop"
    CATEGORY = "NV_Utils/Inpaint"
    DESCRIPTION = "Crops image for inpainting. Outputs both original mask (for stitching) and processed mask (for diffusion). stitch_source selects which mask drives the blend: tight (original), processed, or bbox (full region). Auto mode computes optimal resolution from bbox."

    def crop(self, image, mask, target_mode, target_width, target_height, auto_preset,
             padding_multiple, mask_erode_dilate, mask_fill_holes, mask_remove_noise,
             mask_smooth, stitch_source, mask_blend_pixels, resize_algorithm,
             mask_config=None, bounding_box_mask=None, stabilize_crop=False,
             stabilization_mode="smooth", smooth_window=5, anomaly_threshold=1.5,
             content_stabilize="off", stabilize_strength=1.0):

        # Apply shared config override if connected
        from .mask_processing_config import apply_mask_config
        vals = apply_mask_config(mask_config,
            mask_erode_dilate=mask_erode_dilate,
            mask_fill_holes=mask_fill_holes,
            mask_remove_noise=mask_remove_noise,
            mask_smooth=mask_smooth,
            mask_blend_pixels=mask_blend_pixels,
        )
        mask_erode_dilate = vals["mask_erode_dilate"]
        mask_fill_holes = vals["mask_fill_holes"]
        mask_remove_noise = vals["mask_remove_noise"]
        mask_smooth = vals["mask_smooth"]
        mask_blend_pixels = vals["mask_blend_pixels"]

        padding_multiple = int(padding_multiple)
        device = comfy.model_management.get_torch_device()

        # Move to GPU
        image = image.to(device)
        mask = mask.to(device)
        if bounding_box_mask is not None:
            bounding_box_mask = bounding_box_mask.to(device)

        batch_size = image.shape[0]

        # Handle batch dimension mismatches
        if mask.shape[0] == 1 and batch_size > 1:
            mask = mask.expand(batch_size, -1, -1).clone()
        if bounding_box_mask is not None and bounding_box_mask.shape[0] == 1 and batch_size > 1:
            bounding_box_mask = bounding_box_mask.expand(batch_size, -1, -1).clone()

        # Validate dimensions
        assert image.shape[1] == mask.shape[1] and image.shape[2] == mask.shape[2], \
            f"Image and mask dimensions must match: image {image.shape}, mask {mask.shape}"

        info_lines = []

        # Auto resolution: compute optimal target from bbox aspect ratio
        if target_mode == "auto":
            bbox_union = self._compute_bbox_union(
                bounding_box_mask if bounding_box_mask is not None else mask,
                batch_size
            )
            if bbox_union is not None:
                _, _, union_w, union_h = bbox_union
                bbox_aspect = union_w / union_h
                target_width, target_height = compute_auto_resolution(
                    bbox_aspect, auto_preset, padding_multiple
                )
                info_lines.append(f"Auto resolution: {target_width}x{target_height}")
                info_lines.append(f"  bbox union: {union_w}x{union_h} (aspect {bbox_aspect:.3f})")
                info_lines.append(f"  preset: {auto_preset} ({WAN_PRESETS[auto_preset]['pixels']:,} px budget)")
            else:
                info_lines.append(f"Auto resolution: no bbox found, falling back to {target_width}x{target_height}")

        print(f"[NV_InpaintCrop] Processing {batch_size} frame(s), target {target_width}x{target_height}")

        # Initialize stitcher
        stitcher = {
            'resize_algorithm': resize_algorithm,
            'blend_pixels': mask_blend_pixels,
            'canvas_to_orig_x': [],
            'canvas_to_orig_y': [],
            'canvas_to_orig_w': [],
            'canvas_to_orig_h': [],
            'canvas_image': [],
            'cropped_to_canvas_x': [],
            'cropped_to_canvas_y': [],
            'cropped_to_canvas_w': [],
            'cropped_to_canvas_h': [],
            'cropped_mask_for_blend': [],
            'skipped_indices': [],
            'original_frames': [],
            'total_frames': batch_size,
        }

        result_images = []
        result_masks_original = []
        result_masks_processed = []

        # First pass: collect bounding boxes for anomaly detection / stabilization
        needs_bbox_pass = batch_size > 1 and (stabilize_crop or anomaly_threshold > 0.0)
        if needs_bbox_pass:
            raw_bboxes = []
            for b in range(batch_size):
                bbox_source = bounding_box_mask[b] if bounding_box_mask is not None else mask[b]
                bbox = find_bbox(bbox_source)
                raw_bboxes.append(bbox)

            # Anomaly detection: reject occlusion/tracking-loss frames before smoothing
            if anomaly_threshold > 0.0:
                raw_bboxes = detect_bbox_anomalies(raw_bboxes, anomaly_threshold, info_lines)

            # Temporal stabilization
            if stabilize_crop:
                raw_bboxes = self._stabilize_bboxes(raw_bboxes, stabilization_mode, smooth_window)
        else:
            raw_bboxes = None

        # Process each frame
        for b in range(batch_size):
            one_image = image[b:b+1]
            one_mask = mask[b:b+1]
            one_bbox_mask = bounding_box_mask[b:b+1] if bounding_box_mask is not None else None

            # Check for empty mask
            if torch.count_nonzero(one_mask) == 0:
                print(f"[NV_InpaintCrop] Frame {b}: Empty mask - skipping")
                stitcher['skipped_indices'].append(b)
                stitcher['original_frames'].append(one_image.squeeze(0).to(comfy.model_management.intermediate_device()))
                continue

            # Keep original mask unmodified
            original_mask = one_mask.clone()

            # Process mask for diffusion (apply all operations)
            processed_mask = one_mask.clone()

            if mask_fill_holes > 0:
                processed_mask = mask_fill_holes_fn(processed_mask, mask_fill_holes)
            if mask_remove_noise > 0:
                processed_mask = mask_remove_noise_fn(processed_mask, mask_remove_noise)
            if mask_erode_dilate != 0:
                processed_mask = mask_erode_dilate_fn(processed_mask, mask_erode_dilate)
            if mask_smooth > 0:
                processed_mask = mask_smooth_fn(processed_mask, mask_smooth)

            # Determine crop bounding box
            if raw_bboxes is not None and raw_bboxes[b] is not None:
                bbox = raw_bboxes[b]
            elif one_bbox_mask is not None:
                bbox = find_bbox(one_bbox_mask)
                if bbox is None:
                    bbox = find_bbox(processed_mask)
            else:
                bbox = find_bbox(processed_mask)

            if bbox is None:
                print(f"[NV_InpaintCrop] Frame {b}: No bbox found - skipping")
                stitcher['skipped_indices'].append(b)
                stitcher['original_frames'].append(one_image.squeeze(0).to(comfy.model_management.intermediate_device()))
                continue

            bbox_x, bbox_y, bbox_w, bbox_h = bbox
            print(f"[NV_InpaintCrop] Frame {b}: bbox ({bbox_x}, {bbox_y}, {bbox_w}x{bbox_h})")

            # Perform crop
            (canvas_image, cto_x, cto_y, cto_w, cto_h,
             cropped_image, cropped_mask_orig, cropped_mask_proc,
             ctc_x, ctc_y, ctc_w, ctc_h) = crop_for_inpaint(
                one_image, original_mask, processed_mask,
                bbox_x, bbox_y, bbox_w, bbox_h,
                target_width, target_height, padding_multiple, resize_algorithm
            )

            # Create blend mask from selected source
            if stitch_source == "bbox":
                # Full crop region — blend in everything
                blend_mask = torch.ones_like(cropped_mask_orig)
            elif stitch_source == "processed":
                blend_mask = cropped_mask_proc.clone()
            else:
                # "tight" — original unprocessed mask (default)
                blend_mask = cropped_mask_orig.clone()

            if mask_blend_pixels > 0 and stitch_source != "bbox":
                blend_mask = mask_erode_dilate_fn(blend_mask, mask_blend_pixels)
                blend_mask = mask_blur(blend_mask, mask_blend_pixels)

            # Store in stitcher
            intermediate = comfy.model_management.intermediate_device()
            stitcher['canvas_to_orig_x'].append(cto_x)
            stitcher['canvas_to_orig_y'].append(cto_y)
            stitcher['canvas_to_orig_w'].append(cto_w)
            stitcher['canvas_to_orig_h'].append(cto_h)
            stitcher['canvas_image'].append(canvas_image.squeeze(0).to(intermediate))
            stitcher['cropped_to_canvas_x'].append(ctc_x)
            stitcher['cropped_to_canvas_y'].append(ctc_y)
            stitcher['cropped_to_canvas_w'].append(ctc_w)
            stitcher['cropped_to_canvas_h'].append(ctc_h)
            stitcher['cropped_mask_for_blend'].append(blend_mask.squeeze(0).to(intermediate))

            result_images.append(cropped_image.squeeze(0).to(intermediate))
            result_masks_original.append(cropped_mask_orig.squeeze(0).to(intermediate))
            result_masks_processed.append(cropped_mask_proc.squeeze(0).to(intermediate))

        # Handle all-skipped case
        if len(result_images) == 0:
            print(f"[NV_InpaintCrop] All frames skipped - returning original")
            info_lines.append("All frames skipped - no mask content found")
            empty_mask = torch.zeros((batch_size, image.shape[1], image.shape[2]),
                                     device=comfy.model_management.intermediate_device())
            return (stitcher,
                    image.to(comfy.model_management.intermediate_device()),
                    empty_mask,
                    empty_mask,
                    "\n".join(info_lines))

        result_images = torch.stack(result_images, dim=0)
        result_masks_original = torch.stack(result_masks_original, dim=0)
        result_masks_processed = torch.stack(result_masks_processed, dim=0)

        # Content stabilization: warp pixels within crop to lock subject
        if content_stabilize != "off" and result_images.shape[0] > 1:
            if content_stabilize == "centroid":
                result_images, result_masks_original, result_masks_processed, warp_data = \
                    _stabilize_content_centroid(result_images, result_masks_original, result_masks_processed,
                                               stitcher=stitcher, target_w=target_width, target_h=target_height,
                                               resize_algorithm=resize_algorithm,
                                               strength=stabilize_strength)
            else:
                result_images, result_masks_original, result_masks_processed, warp_data = \
                    _stabilize_content_flow(result_images, result_masks_original, result_masks_processed)
            stitcher['content_warp_mode'] = content_stabilize
            stitcher['content_warp_data'] = warp_data
            info_lines.append(f"Content stabilization: {content_stabilize} ({len(warp_data)} frames)")

        out_h, out_w = result_images.shape[1], result_images.shape[2]
        info_lines.append(f"Output: {result_images.shape[0]} frames @ {out_w}x{out_h}")
        print(f"[NV_InpaintCrop] Output: {result_images.shape[0]} frames, {out_w}x{out_h}")

        return (stitcher, result_images, result_masks_original, result_masks_processed,
                "\n".join(info_lines))

    def _compute_bbox_union(self, mask_source, batch_size):
        """
        Compute the union bounding box across all frames.

        Returns (x, y, w, h) of the envelope containing all per-frame bboxes,
        or None if no frames have content.
        """
        union_x_min, union_y_min = float('inf'), float('inf')
        union_x_max, union_y_max = 0, 0
        found_any = False

        for b in range(batch_size):
            bbox = find_bbox(mask_source[b])
            if bbox is not None:
                bx, by, bw, bh = bbox
                union_x_min = min(union_x_min, bx)
                union_y_min = min(union_y_min, by)
                union_x_max = max(union_x_max, bx + bw)
                union_y_max = max(union_y_max, by + bh)
                found_any = True

        if not found_any:
            return None

        return (int(union_x_min), int(union_y_min),
                int(union_x_max - union_x_min), int(union_y_max - union_y_min))

    def _stabilize_bboxes(self, bboxes, mode, window_size):
        """Apply temporal stabilization to bounding box sequence.

        Smooths bbox CENTER and dimensions independently, then derives
        corner positions from the smoothed center.  This prevents the
        ±1px A-B-A-B oscillation caused by integer-division centering
        when x/y/w/h are smoothed and rounded separately.
        """
        valid_indices = [i for i, b in enumerate(bboxes) if b is not None]

        if len(valid_indices) <= 1:
            return bboxes

        xs = [bboxes[i][0] for i in valid_indices]
        ys = [bboxes[i][1] for i in valid_indices]
        ws = [bboxes[i][2] for i in valid_indices]
        hs = [bboxes[i][3] for i in valid_indices]

        # Convert corner+size → center+size (float)
        cxs = [x + w / 2.0 for x, w in zip(xs, ws)]
        cys = [y + h / 2.0 for y, h in zip(ys, hs)]

        if mode == "smooth":
            cxs = list(gaussian_smooth_1d(cxs, window_size))
            cys = list(gaussian_smooth_1d(cys, window_size))
            ws = [int(round(v)) for v in gaussian_smooth_1d(ws, window_size)]
            hs = [int(round(v)) for v in gaussian_smooth_1d(hs, window_size)]

        elif mode == "lock_first":
            ref_w, ref_h = ws[0], hs[0]
            # Centers already computed above; just lock dimensions
            ws = [ref_w] * len(valid_indices)
            hs = [ref_h] * len(valid_indices)

        elif mode == "lock_largest":
            max_w = max(ws)
            max_h = max(hs)
            ws = [max_w] * len(valid_indices)
            hs = [max_h] * len(valid_indices)

        elif mode == "median":
            cxs = list(median_filter_1d(cxs, window_size))
            cys = list(median_filter_1d(cys, window_size))
            ws = [int(round(v)) for v in median_filter_1d(ws, window_size)]
            hs = [int(round(v)) for v in median_filter_1d(hs, window_size)]

        # Derive corner from smoothed center — round position LAST
        xs = [int(round(cx - w / 2.0)) for cx, w in zip(cxs, ws)]
        ys = [int(round(cy - h / 2.0)) for cy, h in zip(cys, hs)]

        result = list(bboxes)
        for idx, i in enumerate(valid_indices):
            result[i] = (xs[idx], ys[idx], ws[idx], hs[idx])

        return result


# Alias functions to avoid name collision with parameters
mask_erode_dilate_fn = mask_erode_dilate
mask_fill_holes_fn = mask_fill_holes
mask_remove_noise_fn = mask_remove_noise
mask_smooth_fn = mask_smooth


# =============================================================================
# Node Registration
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "NV_InpaintCrop2": NV_InpaintCrop,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_InpaintCrop2": "NV Inpaint Crop v2",
}
