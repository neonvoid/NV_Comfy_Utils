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


def _bg_stats(image_bhwc, mask_bhw):
    """Mean + std of luminance in the NON-mask area, averaged over frames."""
    lum = _luminance(image_bhwc)              # [B, H, W]
    if mask_bhw is None:
        bg_lum = lum                          # whole frame is "BG" if no mask
    else:
        if mask_bhw.dim() == 2:
            mask_bhw = mask_bhw.unsqueeze(0)
        # broadcast mask to match lum batch size if needed
        if mask_bhw.shape[0] == 1 and lum.shape[0] > 1:
            mask_bhw = mask_bhw.expand(lum.shape[0], -1, -1)
        # Resize mask if spatial dims differ (cheap nearest)
        if mask_bhw.shape[-2:] != lum.shape[-2:]:
            mask_bhw = torch.nn.functional.interpolate(
                mask_bhw.unsqueeze(1).float(),
                size=lum.shape[-2:],
                mode="nearest",
            ).squeeze(1)
        bg_keep = (mask_bhw < 0.5).float()    # 1 where NOT mask
        if bg_keep.sum() < 1:
            return float(lum.mean()), float(lum.std())
        bg_lum = lum * bg_keep
        # mean over kept pixels only
        sum_lum = bg_lum.sum()
        n = bg_keep.sum().clamp(min=1.0)
        mean = (sum_lum / n).item()
        # variance over kept pixels
        diff_sq = ((lum - mean) ** 2) * bg_keep
        var = (diff_sq.sum() / n).item()
        return mean, var ** 0.5
    return float(bg_lum.mean()), float(bg_lum.std())


def _mask_stats(mask_bhw, total_pixels):
    """Mean + std of mask occupancy fraction across frames."""
    if mask_bhw is None or total_pixels <= 0:
        return None, None
    if mask_bhw.dim() == 2:
        mask_bhw = mask_bhw.unsqueeze(0)
    binary = (mask_bhw > 0.5).float()
    per_frame = binary.flatten(start_dim=1).sum(dim=1) / total_pixels  # [B]
    return float(per_frame.mean()), float(per_frame.std())


def _motion_from_stitcher(stitcher):
    """Mean per-frame bbox-center displacement (pixels) from stitcher trajectory.
    Returns None if stitcher missing or trajectory too short.
    """
    if stitcher is None:
        return None
    try:
        cx = stitcher.get('cropped_to_canvas_x')
        cy = stitcher.get('cropped_to_canvas_y')
        cw = stitcher.get('cropped_to_canvas_w')
        ch = stitcher.get('cropped_to_canvas_h')
        if cx is None or cy is None or cw is None or ch is None:
            return None
        n = len(cx)
        if n < 2:
            return None
        # bbox center per frame
        centers = [(cx[i] + cw[i] / 2.0, cy[i] + ch[i] / 2.0) for i in range(n)]
        deltas = [
            ((centers[i][0] - centers[i - 1][0]) ** 2 + (centers[i][1] - centers[i - 1][1]) ** 2) ** 0.5
            for i in range(1, n)
        ]
        if not deltas:
            return None
        return sum(deltas) / len(deltas)
    except Exception:
        return None


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

        # BG luminance
        bg_mean, bg_std = _bg_stats(images, mask)
        fp["bg_luminance_mean"] = round(bg_mean, 4)
        fp["bg_luminance_std"] = round(bg_std, 4)
        fp["bg_luminance_bucket"] = _bucket(bg_mean, BG_LUMINANCE_BUCKETS)

        # Mask occupancy
        occ_mean, occ_std = _mask_stats(mask, H * W)
        if occ_mean is not None:
            fp["mask_occupancy_mean"] = round(occ_mean, 4)
            fp["mask_area_std"] = round(occ_std, 4) if occ_std is not None else None
            fp["mask_occupancy_bucket"] = _bucket(occ_mean, MASK_OCCUPANCY_BUCKETS)

        # Motion
        motion_px = _motion_from_stitcher(stitcher)
        motion_source = "stitcher_bbox"
        if motion_px is None:
            motion_px = _motion_from_mask_centroid(mask)
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
