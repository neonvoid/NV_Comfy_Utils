"""
Guided Filter — edge-aware mask refinement using image structure (He et al. 2013).

Two use cases:
  1. Stitch-time blend mask refinement: snap blend boundary to real image edges
  2. SAM3 mask cleanup: align noisy segmentation edges to source frame gradients

Provides gray (luminance) and color (full 3x3 RGB covariance) variants,
plus a fast variant that solves coefficients at reduced resolution for 1080p.
"""

import torch
import torch.nn.functional as F


# =============================================================================
# Box filter (O(1) per pixel via avg_pool2d)
# =============================================================================

def box_filter(x, radius):
    """Fast local mean using avg_pool2d with reflect padding.

    Clamps radius to avoid reflect-padding crash on tiny inputs
    (PyTorch requires each padded dim > padding size).
    """
    if radius <= 0:
        return x
    _, _, h, w = x.shape
    r = min(radius, h - 1, w - 1)
    if r <= 0:
        return x
    kernel = 2 * r + 1
    x = F.pad(x, (r, r, r, r), mode="reflect")
    return F.avg_pool2d(x, kernel_size=kernel, stride=1)


# =============================================================================
# Gray guided filter (single-channel guide)
# =============================================================================

def _guided_filter_gray_coefficients(guide, src, radius, eps):
    """Compute linear coefficients a, b for grayscale guide. guide/src: [B,1,H,W]."""
    mean_i = box_filter(guide, radius)
    mean_p = box_filter(src, radius)
    corr_ii = box_filter(guide * guide, radius)
    corr_ip = box_filter(guide * src, radius)

    var_i = corr_ii - mean_i * mean_i
    cov_ip = corr_ip - mean_i * mean_p

    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    return a, b


def _guided_filter_gray_apply(guide, a, b, radius):
    """Apply averaged coefficients to produce filtered output."""
    mean_a = box_filter(a, radius)
    mean_b = box_filter(b, radius)
    return mean_a * guide + mean_b


# =============================================================================
# Color guided filter (RGB 3x3 covariance solve via Cramer's rule)
# =============================================================================

def _guided_filter_color_coefficients(guide, src, radius, eps):
    """Compute linear coefficients a [B,3,H,W], b [B,1,H,W] for RGB guide.

    Uses explicit Cramer's rule (determinant + cofactors) instead of
    torch.linalg.solve to avoid materializing per-pixel 3x3 matrices.
    This is faster on GPU for large images.
    """
    ir, ig, ib = guide[:, 0:1], guide[:, 1:2], guide[:, 2:3]

    mean_i = box_filter(guide, radius)  # [B, 3, H, W]
    mean_p = box_filter(src, radius)    # [B, 1, H, W]
    mean_ir, mean_ig, mean_ib = mean_i[:, 0:1], mean_i[:, 1:2], mean_i[:, 2:3]

    # Covariance matrix Sigma (symmetric 3x3) + eps*I on diagonal
    var_rr = box_filter(ir * ir, radius) - mean_ir * mean_ir + eps
    var_rg = box_filter(ir * ig, radius) - mean_ir * mean_ig
    var_rb = box_filter(ir * ib, radius) - mean_ir * mean_ib
    var_gg = box_filter(ig * ig, radius) - mean_ig * mean_ig + eps
    var_gb = box_filter(ig * ib, radius) - mean_ig * mean_ib
    var_bb = box_filter(ib * ib, radius) - mean_ib * mean_ib + eps

    # Cross-covariance with mask
    cov_rp = box_filter(ir * src, radius) - mean_ir * mean_p
    cov_gp = box_filter(ig * src, radius) - mean_ig * mean_p
    cov_bp = box_filter(ib * src, radius) - mean_ib * mean_p

    # Cramer's rule: a = Sigma^-1 @ cov via cofactor expansion
    det = (
        var_rr * (var_gg * var_bb - var_gb * var_gb)
        - var_rg * (var_rg * var_bb - var_rb * var_gb)
        + var_rb * (var_rg * var_gb - var_rb * var_gg)
    ).clamp_min(1e-12)

    inv_rr = (var_gg * var_bb - var_gb * var_gb) / det
    inv_rg = (var_rb * var_gb - var_rg * var_bb) / det
    inv_rb = (var_rg * var_gb - var_rb * var_gg) / det
    inv_gg = (var_rr * var_bb - var_rb * var_rb) / det
    inv_gb = (var_rb * var_rg - var_rr * var_gb) / det
    inv_bb = (var_rr * var_gg - var_rg * var_rg) / det

    a_r = inv_rr * cov_rp + inv_rg * cov_gp + inv_rb * cov_bp
    a_g = inv_rg * cov_rp + inv_gg * cov_gp + inv_gb * cov_bp
    a_b = inv_rb * cov_rp + inv_gb * cov_gp + inv_bb * cov_bp

    a = torch.cat([a_r, a_g, a_b], dim=1)  # [B, 3, H, W]
    b = mean_p - (a_r * mean_ir + a_g * mean_ig + a_b * mean_ib)
    return a, b


def _guided_filter_color_apply(guide, a, b, radius):
    """Apply averaged color coefficients."""
    mean_a = box_filter(a, radius)
    mean_b = box_filter(b, radius)
    return torch.sum(mean_a * guide, dim=1, keepdim=True) + mean_b


# =============================================================================
# Public API
# =============================================================================

def guided_filter_color(guide, src, radius, eps):
    """Full color guided filter with 3x3 covariance solve per pixel.

    Args:
        guide: [B, 3, H, W] RGB guide image
        src: [B, 1, H, W] mask to refine
        radius: window radius (int)
        eps: regularization (float), lower = sharper edge tracking
    Returns:
        [B, 1, H, W] refined mask
    """
    a, b = _guided_filter_color_coefficients(guide, src, radius, eps)
    return _guided_filter_color_apply(guide, a, b, radius)


def guided_filter_fast(guide, src, radius, eps, subsample=4):
    """Fast guided filter — solves at reduced resolution, applies at full res.

    Useful for 1080p+ images. Falls back to full resolution when subsample <= 1
    or image is already small enough.

    Args:
        guide: [B, C, H, W] with C=1 or C=3
        src: [B, 1, H, W]
        radius: window radius at full resolution
        eps: regularization
        subsample: downscale factor for coefficient solve (default 4)
    Returns:
        [B, 1, H, W]
    """
    is_color = guide.shape[1] == 3
    subsample = max(1, int(subsample))

    # Fall back to full res if subsample disabled or image already small
    _, _, h, w = guide.shape
    small_h = max(1, h // subsample)
    small_w = max(1, w // subsample)

    if subsample <= 1 or (small_h == h and small_w == w):
        if is_color:
            return guided_filter_color(guide, src, radius, eps)
        a, b = _guided_filter_gray_coefficients(guide, src, radius, eps)
        return _guided_filter_gray_apply(guide, a, b, radius)

    # Downsample, solve coefficients at low res
    guide_small = F.interpolate(guide, size=(small_h, small_w), mode="bilinear", align_corners=False)
    src_small = F.interpolate(src, size=(small_h, small_w), mode="bilinear", align_corners=False)
    radius_small = max(1, round(radius / subsample))

    if is_color:
        a_small, b_small = _guided_filter_color_coefficients(guide_small, src_small, radius_small, eps)
    else:
        a_small, b_small = _guided_filter_gray_coefficients(guide_small, src_small, radius_small, eps)

    # Upsample coefficients, apply at full res
    a = F.interpolate(a_small, size=(h, w), mode="bilinear", align_corners=False)
    b = F.interpolate(b_small, size=(h, w), mode="bilinear", align_corners=False)

    if is_color:
        return torch.sum(a * guide, dim=1, keepdim=True) + b
    return a * guide + b


def refine_mask(mask, guide_image, radius=8, eps=0.001, strength=0.7, mode="color"):
    """Refine a soft mask using image-guided edge alignment.

    High-level API that handles tensor layout conversion, auto-selects fast
    variant for large images, and applies strength-based safety clamping.

    Args:
        mask: [B, H, W] soft mask in [0, 1]
        guide_image: [B, H, W, C] reference image (ComfyUI IMAGE format)
        radius: guided filter window radius
        eps: regularization (lower = sharper edge tracking, higher = smoother)
        strength: lerp between original (0.0) and refined (1.0) mask. Default 0.7
                  prevents the guided filter from eroding valid mask coverage.
        mode: 'color' (full RGB covariance) or 'gray' (luminance only)
    Returns:
        [B, H, W] refined mask
    """
    mask = mask.clamp(0.0, 1.0)

    # Shape validation
    if mask.dim() != 3:
        raise ValueError(f"mask must be [B, H, W], got shape {list(mask.shape)}")
    if guide_image.dim() != 4:
        raise ValueError(f"guide_image must be [B, H, W, C], got shape {list(guide_image.shape)}")
    if mask.shape[0] != guide_image.shape[0]:
        raise ValueError(f"Batch mismatch: mask batch={mask.shape[0]}, guide batch={guide_image.shape[0]}")
    if mask.shape[1] != guide_image.shape[1] or mask.shape[2] != guide_image.shape[2]:
        raise ValueError(f"Spatial mismatch: mask={list(mask.shape[1:])}, guide={list(guide_image.shape[1:3])}")

    if mode not in ("color", "gray"):
        raise ValueError(f"mode must be 'color' or 'gray', got '{mode}'")

    src = mask.unsqueeze(1)  # [B, 1, H, W]
    guide = guide_image.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]

    if mode == "gray":
        if guide.shape[1] >= 3:
            weights = guide.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
            guide_used = torch.sum(guide[:, :3] * weights, dim=1, keepdim=True)
        else:
            guide_used = guide[:, :1]
    else:
        # color mode
        if guide.shape[1] == 1:
            guide_used = guide.repeat(1, 3, 1, 1)
        else:
            guide_used = guide[:, :3]

    # Auto-select fast variant for HD+ images
    _, _, h, w = src.shape
    use_fast = h * w >= 1280 * 720

    if use_fast:
        refined = guided_filter_fast(guide_used, src, radius, eps, subsample=4)
    else:
        if guide_used.shape[1] == 3:
            refined = guided_filter_color(guide_used, src, radius, eps)
        else:
            a, b = _guided_filter_gray_coefficients(guide_used, src, radius, eps)
            refined = _guided_filter_gray_apply(guide_used, a, b, radius)

    refined = refined.squeeze(1).clamp(0.0, 1.0)

    # Safety clamp: lerp between original and refined to prevent mask erosion
    strength = float(max(0.0, min(1.0, strength)))
    final = torch.lerp(mask, refined, strength)
    return final.clamp(0.0, 1.0)
