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


class NV_BoundaryColorMatch:
    """Reinhard color transfer at stitch boundaries for seam color correction.

    Samples color statistics from strips on both sides of the stitch boundary,
    then applies mean/std matching in Lab color space with gradient falloff.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Stitched image with potential color seam."
                }),
                "mask": ("MASK", {
                    "tooltip": "Stitch mask (1.0 = inpainted region, 0.0 = original). "
                               "Boundary is derived from the mask edge."
                }),
                "strip_width": ("INT", {
                    "default": 32, "min": 4, "max": 128, "step": 4,
                    "tooltip": "Width of sampling strip on each side of boundary (pixels). "
                               "Wider = more stable statistics, narrower = more local matching."
                }),
                "strength": ("FLOAT", {
                    "default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Color correction strength. 1.0 = full Reinhard transfer. "
                               "0.0 = no correction."
                }),
                "falloff_radius": ("INT", {
                    "default": 48, "min": 0, "max": 256, "step": 4,
                    "tooltip": "Gradient falloff from boundary into interior (pixels). "
                               "Larger = more gradual transition. 0 = correction only at boundary."
                }),
                "temporal_smooth": ("FLOAT", {
                    "default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Temporal smoothing of color statistics across batch. "
                               "0.0 = per-frame independent (may flicker). "
                               "1.0 = use batch-wide statistics (most stable). "
                               "0.8 = blend 80% batch stats + 20% per-frame (recommended)."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "NV_Utils/Inpaint"
    DESCRIPTION = (
        "Reinhard color transfer at stitch boundaries. Matches color/tone "
        "across the seam in Lab space with gradient falloff. Place after "
        "InpaintStitch2 to fix color shift from VAE roundtrip mismatch."
    )

    def execute(self, image, mask, strip_width=32, strength=0.8, falloff_radius=48, temporal_smooth=0.8):
        if strength <= 0.0:
            return (image,)

        device = image.device

        # IMAGE: [B, H, W, C] → [B, C, H, W]
        img = image.permute(0, 3, 1, 2).to(device).float()
        B, C, H, W = img.shape

        # Mask: [B, H, W] → [B, 1, H, W]
        m = mask.to(device).float()
        if m.dim() == 2:
            m = m.unsqueeze(0)
        if m.dim() == 3:
            m = m.unsqueeze(1)

        # Resize mask if needed
        if m.shape[2:] != (H, W):
            m = F.interpolate(m, size=(H, W), mode='bilinear', align_corners=False)

        # Broadcast mask batch
        if m.shape[0] == 1 and B > 1:
            m = m.expand(B, -1, -1, -1)

        # Binarize mask
        binary_mask = (m > 0.5).float()

        # Create inner and outer strips
        inner_edge = binary_mask - _erode_mask(binary_mask, strip_width)  # Strip just inside boundary
        outer_edge = _dilate_mask(binary_mask, strip_width) - binary_mask  # Strip just outside boundary

        # Check we have enough pixels to compute stats
        inner_count = inner_edge.sum(dim=(2, 3), keepdim=True).clamp(min=1)
        outer_count = outer_edge.sum(dim=(2, 3), keepdim=True).clamp(min=1)

        if inner_count.min() < 10 or outer_count.min() < 10:
            print("[NV_BoundaryColorMatch] Warning: too few pixels in boundary strips, skipping correction")
            return (image,)

        # Convert to Lab
        lab = _rgb_to_lab(img[:, :3])  # Only first 3 channels

        # Compute per-frame, per-channel mean and std in each strip
        # inner_edge is [B, 1, H, W], lab is [B, 3, H, W]
        inner_mask_3c = inner_edge.expand(-1, 3, -1, -1)
        outer_mask_3c = outer_edge.expand(-1, 3, -1, -1)

        inner_sum = (lab * inner_mask_3c).sum(dim=(2, 3), keepdim=True)
        inner_mean = inner_sum / inner_count

        outer_sum = (lab * outer_mask_3c).sum(dim=(2, 3), keepdim=True)
        outer_mean = outer_sum / outer_count

        inner_var = ((lab - inner_mean) ** 2 * inner_mask_3c).sum(dim=(2, 3), keepdim=True) / inner_count
        outer_var = ((lab - outer_mean) ** 2 * outer_mask_3c).sum(dim=(2, 3), keepdim=True) / outer_count

        inner_std = inner_var.clamp(min=1e-6).sqrt()
        outer_std = outer_var.clamp(min=1e-6).sqrt()

        # Temporal smoothing: blend per-frame stats toward batch-wide stats to prevent flicker.
        # At temporal_smooth=1.0, all frames use identical batch-wide stats (most stable).
        # At temporal_smooth=0.0, each frame uses its own stats (may flicker).
        if temporal_smooth > 0.0 and B > 1:
            # Compute batch-wide weighted stats (weighted by pixel count per frame)
            batch_inner_count = inner_count.sum(dim=0, keepdim=True).clamp(min=1)
            batch_inner_mean = (inner_sum.sum(dim=0, keepdim=True)) / batch_inner_count
            batch_inner_var = (((lab - batch_inner_mean) ** 2 * inner_mask_3c).sum(dim=(0, 2, 3), keepdim=True)) / batch_inner_count
            batch_inner_std = batch_inner_var.clamp(min=1e-6).sqrt()

            batch_outer_count = outer_count.sum(dim=0, keepdim=True).clamp(min=1)
            batch_outer_mean = (outer_sum.sum(dim=0, keepdim=True)) / batch_outer_count
            batch_outer_var = (((lab - batch_outer_mean) ** 2 * outer_mask_3c).sum(dim=(0, 2, 3), keepdim=True)) / batch_outer_count
            batch_outer_std = batch_outer_var.clamp(min=1e-6).sqrt()

            # Lerp per-frame stats toward batch stats
            ts = temporal_smooth
            inner_mean = (1.0 - ts) * inner_mean + ts * batch_inner_mean
            inner_std = (1.0 - ts) * inner_std + ts * batch_inner_std
            outer_mean = (1.0 - ts) * outer_mean + ts * batch_outer_mean
            outer_std = (1.0 - ts) * outer_std + ts * batch_outer_std

        # Reinhard transfer: shift inner region to match outer statistics
        corrected_lab = (lab - inner_mean) * (outer_std / inner_std) + outer_mean

        # Create application mask: correction applies inside the mask region with falloff
        application_mask = _make_falloff_mask(binary_mask, falloff_radius)
        application_mask = application_mask * binary_mask  # Only apply inside the inpainted region

        # Blend correction with original
        correction_strength = application_mask * strength  # [B, 1, H, W]
        final_lab = correction_strength * corrected_lab + (1.0 - correction_strength) * lab

        # Convert back to RGB
        final_rgb = _lab_to_rgb(final_lab).clamp(0.0, 1.0)

        # Handle alpha channel if present (pass through)
        if C > 3:
            final_rgb = torch.cat([final_rgb, img[:, 3:]], dim=1)

        # Back to [B, H, W, C]
        result = final_rgb.permute(0, 2, 3, 1)

        print(f"[NV_BoundaryColorMatch] Corrected {B} frames, "
              f"strip_width={strip_width}, strength={strength:.2f}, falloff={falloff_radius}, "
              f"temporal_smooth={temporal_smooth:.2f}")

        return (result,)


NODE_CLASS_MAPPINGS = {
    "NV_BoundaryColorMatch": NV_BoundaryColorMatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_BoundaryColorMatch": "NV Boundary Color Match",
}
