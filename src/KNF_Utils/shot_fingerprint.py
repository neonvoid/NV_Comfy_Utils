"""Shot fingerprint — structured "what kind of shot is this" classifier.

Computed from input plate (NOT result), so still emits even if render fails.
Lets the agent compare across renders within the same regime: a recommendation
that worked on dark-BG walking shots is irrelevant to bright-BG static shots.

All numpy-style scalars come back as Python floats so json.dumps doesn't
choke on tensor or numpy types.
"""

import torch

from .shot_telemetry_types import (
    BG_LUMINANCE_BUCKETS,
    MASK_OCCUPANCY_BUCKETS,
    MOTION_CLASS_BUCKETS,
)


def _bucket(value, buckets):
    """Return label for first threshold >= value. buckets is [(thresh, label)]."""
    for thresh, label in buckets:
        if value < thresh:
            return label
    return buckets[-1][1]


def _luminance(image_bhwc):
    """Per-frame luminance [B, H, W] from RGB IMAGE [B, H, W, 3]."""
    weights = torch.tensor([0.2126, 0.7152, 0.0722], device=image_bhwc.device, dtype=image_bhwc.dtype)
    return (image_bhwc * weights).sum(dim=-1)


def _normalize_mask(mask, target_shape_bhw):
    """Bring MASK to image-space [B, H, W] via nearest-neighbor resize +
    batch broadcast. Centralizes the shape contract so every consumer
    operates on the same normalized tensor.
    """
    if mask is None:
        return None
    m = mask.float()
    if m.dim() == 2:
        m = m.unsqueeze(0)
    target_B, target_H, target_W = target_shape_bhw
    if m.shape[-2:] != (target_H, target_W):
        m = torch.nn.functional.interpolate(
            m.unsqueeze(1), size=(target_H, target_W), mode="nearest"
        ).squeeze(1)
    if m.shape[0] == 1 and target_B > 1:
        m = m.expand(target_B, -1, -1)
    return m


def _bg_stats(image_bhwc, mask_bhw_norm):
    """Mean + std of luminance in the NON-mask area. Mask must already be
    normalized to image-space via _normalize_mask().
    """
    lum = _luminance(image_bhwc)
    if mask_bhw_norm is None:
        return float(lum.mean()), float(lum.std(unbiased=False))
    bg_keep = (mask_bhw_norm < 0.5).float()
    if bg_keep.sum() < 1:
        return float(lum.mean()), float(lum.std(unbiased=False))
    n = bg_keep.sum().clamp(min=1.0)
    mean = ((lum * bg_keep).sum() / n).item()
    var = (((lum - mean) ** 2 * bg_keep).sum() / n).item()
    return mean, var ** 0.5


def _mask_stats(mask_bhw_norm):
    """Mean + std of per-frame mask occupancy. Mask must be normalized to
    image-space first. Uses biased std so single-frame batches don't return NaN.
    """
    if mask_bhw_norm is None:
        return None, None
    binary = (mask_bhw_norm > 0.5).float()
    H, W = binary.shape[-2:]
    total_pixels = H * W
    if total_pixels <= 0:
        return None, None
    per_frame = binary.flatten(start_dim=1).sum(dim=1) / total_pixels  # [B]
    mean = float(per_frame.mean())
    # unbiased=False so B=1 returns 0.0 instead of NaN — non-finite floats
    # poison the JSONL record (json.dumps default emits NaN, strict readers reject).
    std = float(per_frame.std(unbiased=False)) if per_frame.numel() > 0 else 0.0
    return mean, std


def _motion_from_stitcher(stitcher):
    """Mean per-frame bbox-center displacement (pixels) from stitcher trajectory.
    Returns (motion_px, error_str). motion_px is None when no signal can be
    computed; error_str is a short reason ("missing key X", "ragged lengths")
    so the caller can surface schema bugs rather than silently bucketing
    'static'.
    """
    if stitcher is None:
        return None, "no_stitcher"
    if not isinstance(stitcher, dict):
        return None, f"stitcher_type={type(stitcher).__name__}"
    cx = stitcher.get('cropped_to_canvas_x')
    cy = stitcher.get('cropped_to_canvas_y')
    cw = stitcher.get('cropped_to_canvas_w')
    ch = stitcher.get('cropped_to_canvas_h')
    if cx is None or cy is None or cw is None or ch is None:
        return None, "missing_bbox_keys"
    try:
        n = len(cx)
        if not (len(cy) == len(cw) == len(ch) == n):
            return None, "ragged_bbox_arrays"
        if n < 2:
            return None, None  # single-frame is genuine, not an error
        centers = [(cx[i] + cw[i] / 2.0, cy[i] + ch[i] / 2.0) for i in range(n)]
        deltas = [
            ((centers[i][0] - centers[i - 1][0]) ** 2 + (centers[i][1] - centers[i - 1][1]) ** 2) ** 0.5
            for i in range(1, n)
        ]
        if not deltas:
            return None, None
        return sum(deltas) / len(deltas), None
    except (TypeError, IndexError) as e:
        return None, f"{type(e).__name__}: {e}"


def _motion_from_mask_centroid(mask_bhw):
    """Fallback: centroid trajectory if no stitcher. Coarse but good enough
    to bucket static / head_tilt / walking / fast.
    """
    if mask_bhw is None or mask_bhw.dim() != 3 or mask_bhw.shape[0] < 2:
        return None
    binary = (mask_bhw > 0.5).float()
    H, W = binary.shape[-2:]
    yy, xx = torch.meshgrid(
        torch.arange(H, device=binary.device, dtype=binary.dtype),
        torch.arange(W, device=binary.device, dtype=binary.dtype),
        indexing="ij",
    )
    deltas = []
    prev = None
    for i in range(binary.shape[0]):
        m = binary[i]
        total = m.sum()
        if total < 1:
            continue
        cy = (m * yy).sum().item() / total.item()
        cx = (m * xx).sum().item() / total.item()
        if prev is not None:
            dx, dy = cx - prev[0], cy - prev[1]
            deltas.append((dx * dx + dy * dy) ** 0.5)
        prev = (cx, cy)
    if not deltas:
        return None
    return sum(deltas) / len(deltas)


def compute_fingerprint(images, mask=None, stitcher=None):
    """Compute regime descriptor from the input plate.

    Args:
        images: IMAGE tensor [B, H, W, 3], values in [0, 1].
        mask:   MASK tensor [B, H, W] or [H, W], values in [0, 1]. Optional.
        stitcher: dict from NV_InpaintCrop2 / similar. Optional.

    Returns dict with the schema documented in shot_telemetry_types.py.
    """
    fp = {}
    try:
        if not torch.is_tensor(images) or images.dim() != 4:
            fp["error"] = f"expected IMAGE [B,H,W,3], got {type(images).__name__}"
            return fp
        B, H, W, C = images.shape
        fp["resolution"] = [int(H), int(W)]
        fp["frame_count"] = int(B)

        # Normalize mask to image space ONCE — _bg_stats and _mask_stats both
        # need image-space resolution, otherwise occupancy fractions can exceed
        # 1.0 when mask is at a different resolution and poison bucket assignment.
        mask_norm = _normalize_mask(mask, (B, H, W)) if mask is not None else None

        # BG luminance
        bg_mean, bg_std = _bg_stats(images, mask_norm)
        fp["bg_luminance_mean"] = round(bg_mean, 4)
        fp["bg_luminance_std"] = round(bg_std, 4)
        fp["bg_luminance_bucket"] = _bucket(bg_mean, BG_LUMINANCE_BUCKETS)

        # Mask occupancy
        occ_mean, occ_std = _mask_stats(mask_norm)
        if occ_mean is not None:
            fp["mask_occupancy_mean"] = round(occ_mean, 4)
            fp["mask_area_std"] = round(occ_std, 4) if occ_std is not None else None
            fp["mask_occupancy_bucket"] = _bucket(occ_mean, MASK_OCCUPANCY_BUCKETS)

        # Motion
        motion_px, motion_err = _motion_from_stitcher(stitcher)
        motion_source = "stitcher_bbox"
        if motion_px is None and stitcher is not None and motion_err and motion_err != "no_stitcher":
            # Schema-level stitcher failure — surface it instead of silently
            # falling through to mask centroid (which would mask the upstream bug).
            fp["motion_error"] = motion_err
        if motion_px is None:
            motion_px = _motion_from_mask_centroid(mask_norm)
            motion_source = "mask_centroid"
        if motion_px is not None:
            fp["motion_displacement_mean_px"] = round(motion_px, 3)
            fp["motion_class"] = _bucket(motion_px, MOTION_CLASS_BUCKETS)
            fp["motion_source"] = motion_source

        # Aspect ratio bucket
        if H > W * 1.1:
            fp["aspect_ratio"] = "vertical"
        elif W > H * 1.1:
            fp["aspect_ratio"] = "wide"
        else:
            fp["aspect_ratio"] = "square"

        # Composite regime tags — what the agent slices on
        tags = []
        if "bg_luminance_bucket" in fp:
            tags.append(fp["bg_luminance_bucket"] + "_bg")
        if "mask_occupancy_bucket" in fp:
            tags.append(fp["mask_occupancy_bucket"] + "_mask")
        if "motion_class" in fp:
            tags.append(fp["motion_class"])
        tags.append(fp["aspect_ratio"])
        fp["regime_tags"] = tags

    except Exception as e:
        fp["compute_error"] = f"{type(e).__name__}: {e}"
    return fp
