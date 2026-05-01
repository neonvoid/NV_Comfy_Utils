"""
NV Boundary Color Match — Reinhard color transfer at stitch boundaries.

Samples color statistics from strips on both sides of the stitch boundary,
then applies a Reinhard-style transfer (match mean/std in Lab color space)
with gradient falloff into the interior. Fixes the color/tone shift component
of VAE roundtrip seam artifacts.

Fast (<1ms per frame), no model inference, pure signal processing.
"""

import torch
import torch.nn.functional as F


def _rgb_to_lab(rgb):
    """Convert RGB [0,1] to CIE Lab. Input/output: [B, 3, H, W]."""
    # RGB → linear RGB (sRGB gamma removal)
    linear = torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)

    # Linear RGB → XYZ (D65 illuminant)
    r, g, b = linear[:, 0:1], linear[:, 1:2], linear[:, 2:3]
    x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b

    # Normalize by D65 white point
    x = x / 0.95047
    z = z / 1.08883

    # XYZ → Lab
    epsilon = 0.008856
    kappa = 903.3

    def f(t):
        return torch.where(t > epsilon, t.clamp(min=1e-8).pow(1.0 / 3.0), (kappa * t + 16.0) / 116.0)

    fx, fy, fz = f(x), f(y), f(z)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b_ch = 200.0 * (fy - fz)

    return torch.cat([L, a, b_ch], dim=1)


def _lab_to_rgb(lab):
    """Convert CIE Lab to RGB [0,1]. Input/output: [B, 3, H, W]."""
    L, a, b_ch = lab[:, 0:1], lab[:, 1:2], lab[:, 2:3]

    # Lab → XYZ
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b_ch / 200.0

    epsilon = 0.008856
    kappa = 903.3

    x = torch.where(fx ** 3 > epsilon, fx ** 3, (116.0 * fx - 16.0) / kappa)
    y = torch.where(L > kappa * epsilon, fy ** 3, L / kappa)
    z = torch.where(fz ** 3 > epsilon, fz ** 3, (116.0 * fz - 16.0) / kappa)

    # Denormalize by D65 white point
    x = x * 0.95047
    z = z * 1.08883

    # XYZ → linear RGB
    r = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z
    g = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z
    b = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z

    linear = torch.cat([r, g, b], dim=1).clamp(0.0, 1.0)

    # Linear RGB → sRGB
    return torch.where(linear <= 0.0031308, 12.92 * linear, 1.055 * linear.clamp(min=1e-8).pow(1.0 / 2.4) - 0.055)


def _erode_mask(mask, pixels):
    """Erode a [B, 1, H, W] binary mask by `pixels` using max pooling of inverted mask."""
    if pixels <= 0:
        return mask
    inverted = 1.0 - mask
    eroded_inv = F.max_pool2d(inverted, kernel_size=2 * pixels + 1, stride=1, padding=pixels)
    return 1.0 - eroded_inv


def _dilate_mask(mask, pixels):
    """Dilate a [B, 1, H, W] binary mask by `pixels` using max pooling."""
    if pixels <= 0:
        return mask
    return F.max_pool2d(mask, kernel_size=2 * pixels + 1, stride=1, padding=pixels)


def _make_falloff_mask(mask, falloff_radius):
    """Create a gradient mask that's 1.0 at the boundary and fades to 0.0 inside the mask region.

    Uses the mask boundary as the origin, with Gaussian-like falloff into the interior.
    """
    if falloff_radius <= 0:
        # Binary: just the boundary strip
        return mask

    # Iterative box blur approximation for distance-based falloff
    # Start from the boundary mask and progressively blur
    kernel_size = falloff_radius * 2 + 1
    sigma = falloff_radius / 3.0
    coords = torch.arange(kernel_size, dtype=torch.float32, device=mask.device) - falloff_radius
    g = torch.exp(-0.5 * (coords / max(sigma, 0.1)) ** 2)
    g = g / g.sum()

    # Separable Gaussian blur
    kh = g.view(1, 1, 1, -1)
    kv = g.view(1, 1, -1, 1)

    blurred = F.pad(mask, (falloff_radius, falloff_radius, 0, 0), mode='reflect')
    blurred = F.conv2d(blurred, kh)
    blurred = F.pad(blurred, (0, 0, falloff_radius, falloff_radius), mode='reflect')
    blurred = F.conv2d(blurred, kv)

    return blurred.clamp(0.0, 1.0)



# Standalone NV_BoundaryColorMatch node deleted 2026-05-01 (see node_notes/cleanup/
# 2026-05-01_post_process_node_cleanup.md). Helper functions above (_rgb_to_lab,
# _lab_to_rgb) are still imported by crop_color_fix.py for V2 lab_lowfreq pipeline.
