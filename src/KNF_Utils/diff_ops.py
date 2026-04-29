"""Image-diff metric computation — extracted from image_diff_analyzer.py.

Pure functions returning structured dicts of FLOAT scalars. No vis output.
Splits stats into interior-kept vs boundary-kept zones; the boundary zone
catches seam-localized failures that mean-over-everything would hide.

Imported by both NV_ImageDiffAnalyzer (legacy viz wrapper) and NV_ShotMeasure.
"""

import torch
import torch.nn.functional as F


def _percentile(t, q):
    """Single-percentile via sort. q in [0, 1]. Tensors only, returns float."""
    if t.numel() == 0:
        return 0.0
    flat = t.flatten()
    k = max(0, min(flat.numel() - 1, int(round(q * (flat.numel() - 1)))))
    return float(flat.sort().values[k])


def _compute_boundary_mask(mask_binary_bhw, width):
    """Bool tensor: True for kept pixels within `width` px of mask edge.
    mask_binary: [B, H, W] bool, True = generated (excluded).
    """
    m_float = mask_binary_bhw.float().unsqueeze(1)
    k = 2 * width + 1
    kernel = torch.ones(1, 1, k, k, device=m_float.device)
    dilated = F.conv2d(m_float, kernel, padding=width) > 0
    dilated = dilated.squeeze(1)
    return dilated & (~mask_binary_bhw)


def _zone_stats(abs_diff_bhwc, zone_mask_bhw):
    """Per-channel mean + percentiles for the kept-zone pixels. Returns dict."""
    out = {}
    if zone_mask_bhw.sum() == 0:
        return {"count": 0}
    mean_per_channel = []
    p95_per_channel = []
    for c in range(abs_diff_bhwc.shape[-1]):
        ch = abs_diff_bhwc[..., c][zone_mask_bhw]
        if ch.numel() == 0:
            mean_per_channel.append(0.0)
            p95_per_channel.append(0.0)
        else:
            mean_per_channel.append(float(ch.mean()))
            p95_per_channel.append(_percentile(ch, 0.95))
    out["count"] = int(zone_mask_bhw.sum())
    # Aggregate across channels — agent only needs one number per zone
    out["mean"] = sum(mean_per_channel) / max(1, len(mean_per_channel))
    out["p95"]  = max(p95_per_channel) if p95_per_channel else 0.0
    return out


def compute_diff_metrics(image_a, image_b, mask=None, mask_threshold=0.5, boundary_width=16):
    """Compute scalar diff metrics for one render's source vs result.

    Args:
        image_a: source IMAGE [B, H, W, 3] (the input plate / pre-render)
        image_b: result IMAGE [B, H, W, 3] (post-render output)
        mask:    MASK [B, H, W] or [H, W]; mask=1 = generated region (excluded
                 from diff). If None, all pixels are "kept".
        mask_threshold: pixels with mask < threshold are "kept" (measured).
        boundary_width: pixels around mask edge that count as "boundary kept".

    Returns dict with the percentile-heavy schema agreed on by the panel.
    """
    out = {}
    try:
        if image_a.shape != image_b.shape:
            # Resize B to match A — same as legacy node behavior
            b = image_b.permute(0, 3, 1, 2)
            b = F.interpolate(b, size=(image_a.shape[1], image_a.shape[2]),
                              mode="bilinear", align_corners=False)
            image_b = b.permute(0, 2, 3, 1)

        abs_diff = (image_b.float() - image_a.float()).abs()  # [B, H, W, 3]

        # Global stats (all pixels)
        global_flat = abs_diff.mean(dim=-1)  # collapse channels for percentile
        out["global_mean"] = round(float(global_flat.mean()), 6)
        out["global_p95"]  = round(_percentile(global_flat, 0.95), 6)

        # Mask-aware zone stats
        if mask is not None:
            m = mask.float()
            if m.dim() == 2:
                m = m.unsqueeze(0)
            if m.shape[1:] != image_a.shape[1:3]:
                m = F.interpolate(m.unsqueeze(1), size=(image_a.shape[1], image_a.shape[2]),
                                  mode="bilinear", align_corners=False).squeeze(1)
            if m.shape[0] == 1 and abs_diff.shape[0] > 1:
                m = m.expand(abs_diff.shape[0], -1, -1)

            mask_binary = (m >= mask_threshold)        # True = generated (excluded)
            interior = (~mask_binary)                  # all kept pixels
            boundary = _compute_boundary_mask(mask_binary, boundary_width)

            i_stats = _zone_stats(abs_diff, interior)
            b_stats = _zone_stats(abs_diff, boundary)
            out["interior_mean"] = round(i_stats.get("mean", 0.0), 6)
            out["interior_p95"]  = round(i_stats.get("p95", 0.0), 6)
            out["boundary_mean"] = round(b_stats.get("mean", 0.0), 6)
            out["boundary_p95"]  = round(b_stats.get("p95", 0.0), 6)
            out["interior_pixel_count"] = i_stats.get("count", 0)
            out["boundary_pixel_count"] = b_stats.get("count", 0)

            # Boundary-to-interior ratio: high = seam-localized failure
            if out["interior_mean"] > 1e-9:
                out["boundary_to_interior_ratio"] = round(out["boundary_mean"] / out["interior_mean"], 3)
            else:
                out["boundary_to_interior_ratio"] = None
            out["mask_occupancy"] = round(float(mask_binary.float().mean()), 4)
        else:
            out["mask_present"] = False
    except Exception as e:
        out["compute_error"] = f"{type(e).__name__}: {e}"
    return out
