"""
NV Multi-Band Blend Stitch — Laplacian pyramid blending for seam-free stitching.

Decomposes stitched and original images into frequency bands (Laplacian pyramid),
then blends each band with a progressively wider mask. Low frequencies blend broadly
(catches color/tone shift), high frequencies blend narrowly (preserves edge detail).

Standard technique from panorama stitching (Burt & Adelson 1983), adapted for
video inpaint stitch seam repair. Operates on IMAGE tensors after InpaintStitch2.
"""

import torch
import torch.nn.functional as F


def _gaussian_downsample(img, kernel):
    """Blur then 2x downsample. img: [B, C, H, W], kernel: [1, 1, K, K]."""
    C = img.shape[1]
    # Expand kernel to match channels (grouped conv)
    k = kernel.expand(C, -1, -1, -1)
    pad = kernel.shape[-1] // 2
    padded = F.pad(img, (pad, pad, pad, pad), mode='reflect')
    blurred = F.conv2d(padded, k, groups=C)
    return blurred[:, :, ::2, ::2]


def _gaussian_upsample(img, kernel, target_h, target_w):
    """2x upsample then blur. img: [B, C, H, W], kernel: [1, 1, K, K]."""
    up = F.interpolate(img, size=(target_h, target_w), mode='bilinear', align_corners=False)
    C = up.shape[1]
    k = kernel.expand(C, -1, -1, -1)
    pad = kernel.shape[-1] // 2
    padded = F.pad(up, (pad, pad, pad, pad), mode='reflect')
    return F.conv2d(padded, k, groups=C)


def _make_gaussian_kernel_2d(size=5):
    """Create a 2D Gaussian kernel for pyramid operations."""
    sigma = size / 6.0
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel = g.outer(g)
    kernel = kernel / kernel.sum()
    return kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, K, K]


def build_gaussian_pyramid(img, num_levels, kernel):
    """Build Gaussian pyramid. Returns list of [B, C, H, W] tensors."""
    pyramid = [img]
    current = img
    for _ in range(num_levels - 1):
        current = _gaussian_downsample(current, kernel)
        pyramid.append(current)
    return pyramid


def build_laplacian_pyramid(img, num_levels, kernel):
    """Build Laplacian pyramid. Returns list of [B, C, H, W] difference images + base."""
    gauss = build_gaussian_pyramid(img, num_levels, kernel)
    laplacian = []
    for i in range(len(gauss) - 1):
        h, w = gauss[i].shape[2], gauss[i].shape[3]
        expanded = _gaussian_upsample(gauss[i + 1], kernel, h, w)
        laplacian.append(gauss[i] - expanded)
    laplacian.append(gauss[-1])  # Base level (lowest frequency)
    return laplacian


def collapse_laplacian_pyramid(pyramid, kernel):
    """Reconstruct image from Laplacian pyramid."""
    img = pyramid[-1]
    for i in range(len(pyramid) - 2, -1, -1):
        h, w = pyramid[i].shape[2], pyramid[i].shape[3]
        img = _gaussian_upsample(img, kernel, h, w) + pyramid[i]
    return img


def build_mask_pyramid(mask, num_levels, kernel):
    """Build Gaussian pyramid of the blend mask (progressively blurred at each level)."""
    pyramid = [mask]
    current = mask
    for _ in range(num_levels - 1):
        current = _gaussian_downsample(current, kernel)
        pyramid.append(current)
    return pyramid


def multiband_blend(stitched, original, mask, num_levels=5, kernel_size=5):
    """Perform multi-band blending using Laplacian pyramids.

    Args:
        stitched: [B, C, H, W] — the stitched image (inpainted composite).
        original: [B, C, H, W] — the original unmodified image.
        mask: [B, 1, H, W] — blend mask (1.0 = use stitched, 0.0 = use original).
        num_levels: Number of pyramid levels.
        kernel_size: Gaussian kernel size for pyramid operations.

    Returns:
        [B, C, H, W] — blended result.
    """
    device = stitched.device
    kernel = _make_gaussian_kernel_2d(kernel_size).to(device)

    # Clamp pyramid levels to avoid too-small images
    min_dim = min(stitched.shape[2], stitched.shape[3])
    max_levels = max(1, int(torch.tensor(float(min_dim)).log2().item()) - 2)
    num_levels = min(num_levels, max_levels)

    # Build Laplacian pyramids for both images
    lap_stitched = build_laplacian_pyramid(stitched, num_levels, kernel)
    lap_original = build_laplacian_pyramid(original, num_levels, kernel)

    # Build Gaussian pyramid for mask
    mask_pyr = build_mask_pyramid(mask, num_levels, kernel)

    # Blend each level
    blended_pyr = []
    for i in range(num_levels):
        m = mask_pyr[i]
        blended_pyr.append(m * lap_stitched[i] + (1.0 - m) * lap_original[i])

    # Collapse
    return collapse_laplacian_pyramid(blended_pyr, kernel)


class NV_MultiBandBlendStitch:
    """Laplacian pyramid multi-band blending for stitch seam repair.

    Blends low frequencies broadly (catches color/tone shift from VAE roundtrips)
    and high frequencies narrowly (preserves crisp edge detail). Place after
    InpaintStitch2 to reduce visible seam artifacts without model inference.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stitched_image": ("IMAGE", {
                    "tooltip": "Output of InpaintStitch2 (the composited result with seam)."
                }),
                "original_image": ("IMAGE", {
                    "tooltip": "Original unmodified frames (same batch size as stitched)."
                }),
                "blend_mask": ("MASK", {
                    "tooltip": "Stitch mask defining the inpainted region. 1.0 = inpainted, "
                               "0.0 = original. From bbox_mask, InpaintStitch2 mask, etc."
                }),
                "num_levels": ("INT", {
                    "default": 5, "min": 2, "max": 8, "step": 1,
                    "tooltip": "Pyramid levels. 5 = good default for 720p-1080p. "
                               "More levels = broader low-frequency blending."
                }),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Blend strength. 1.0 = full multi-band blend. "
                               "0.0 = original stitch (no change). Lerp between."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "NV_Utils/Inpaint"
    DESCRIPTION = (
        "Laplacian pyramid multi-band blending for stitch seam repair. "
        "Blends low frequencies broadly and high frequencies narrowly. "
        "Place after InpaintStitch2 to reduce VAE roundtrip seam artifacts."
    )

    def execute(self, stitched_image, original_image, blend_mask, num_levels=5, strength=1.0):
        if strength <= 0.0:
            return (stitched_image,)

        device = stitched_image.device

        # IMAGE is [B, H, W, C] — convert to [B, C, H, W] for conv operations
        stitched = stitched_image.permute(0, 3, 1, 2).to(device)
        original = original_image.permute(0, 3, 1, 2).to(device)

        # Handle batch size mismatch (broadcast single original to match stitched)
        if original.shape[0] == 1 and stitched.shape[0] > 1:
            original = original.expand(stitched.shape[0], -1, -1, -1)

        # Handle resolution mismatch
        if original.shape[2:] != stitched.shape[2:]:
            original = F.interpolate(original, size=stitched.shape[2:], mode='bilinear', align_corners=False)

        # Mask: [B, H, W] → [B, 1, H, W]
        mask = blend_mask.to(device)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)

        # Handle mask resolution mismatch
        if mask.shape[2:] != stitched.shape[2:]:
            mask = F.interpolate(mask, size=stitched.shape[2:], mode='bilinear', align_corners=False)

        # Handle mask batch mismatch
        if mask.shape[0] == 1 and stitched.shape[0] > 1:
            mask = mask.expand(stitched.shape[0], -1, -1, -1)

        result = multiband_blend(stitched, original, mask, num_levels=num_levels)
        result = result.clamp(0.0, 1.0)

        # Lerp with original stitched output if strength < 1
        if strength < 1.0:
            result = strength * result + (1.0 - strength) * stitched

        # Back to [B, H, W, C]
        result = result.permute(0, 2, 3, 1)

        print(f"[NV_MultiBandBlendStitch] Blended {result.shape[0]} frames, "
              f"{num_levels} levels, strength={strength:.2f}")

        return (result,)


NODE_CLASS_MAPPINGS = {
    "NV_MultiBandBlendStitch": NV_MultiBandBlendStitch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_MultiBandBlendStitch": "NV Multi-Band Blend Stitch",
}
