"""NV Image Masked Blur — gaussian-blur the masked region of an image.

Primary use case: bypass Seedance 2.0's real-person gate selectively. Use
SAM3 / MaskTrackingBBox to mask just the face + body areas of a reference
video or image, then blur ONLY those areas. Background, environment, and
non-person content stay sharp. Gate sees blurred face features (passes),
the model still sees full motion/scene/composition cues (preserves).

Works on single images and batched video frames in one shot. Mask is
broadcast across frames if it's single-frame [1,H,W] vs the image's [B,H,W,C].

Outputs the same IMAGE shape as input.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _make_gaussian_kernel_1d(radius: int, sigma: float, device, dtype) -> torch.Tensor:
    """Build a 1D normalized gaussian kernel."""
    if sigma <= 0:
        # Sigma auto-derived from radius (matches OpenCV's default formula)
        sigma = 0.3 * (radius - 1) + 0.8
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    k = k / k.sum()
    return k


def _gaussian_blur_BCHW(image: torch.Tensor, radius: int, sigma: float) -> torch.Tensor:
    """Separable 2D gaussian blur on [B, C, H, W] tensor. Reflection-padded."""
    if radius < 1:
        return image
    B, C, H, W = image.shape
    k1d = _make_gaussian_kernel_1d(radius, sigma, image.device, image.dtype)
    k_h = k1d.view(1, 1, 1, -1).expand(C, 1, 1, -1).contiguous()  # [C,1,1,K]
    k_v = k1d.view(1, 1, -1, 1).expand(C, 1, -1, 1).contiguous()  # [C,1,K,1]
    pad_h = (0, 0, radius, radius)  # left, right, top, bottom for 4d
    # F.pad uses last-to-first axis order: (W_left, W_right, H_top, H_bottom)
    # Horizontal blur first
    x = F.pad(image, (radius, radius, 0, 0), mode="reflect")
    x = F.conv2d(x, k_h, padding=0, groups=C)
    # Vertical blur
    x = F.pad(x, (0, 0, radius, radius), mode="reflect")
    x = F.conv2d(x, k_v, padding=0, groups=C)
    return x


def _feather_mask_BHW(mask: torch.Tensor, feather_px: int) -> torch.Tensor:
    """Gaussian-blur a [B, H, W] mask to soften its edges. Returns [B, H, W]."""
    if feather_px < 1:
        return mask
    B, H, W = mask.shape
    # Add channel dim: [B, 1, H, W]
    m = mask.unsqueeze(1)
    sigma = 0.3 * (feather_px - 1) + 0.8
    k1d = _make_gaussian_kernel_1d(feather_px, sigma, mask.device, mask.dtype)
    k_h = k1d.view(1, 1, 1, -1).contiguous()
    k_v = k1d.view(1, 1, -1, 1).contiguous()
    m = F.pad(m, (feather_px, feather_px, 0, 0), mode="reflect")
    m = F.conv2d(m, k_h, padding=0, groups=1)
    m = F.pad(m, (0, 0, feather_px, feather_px), mode="reflect")
    m = F.conv2d(m, k_v, padding=0, groups=1)
    return m.squeeze(1)


class NV_ImageMaskedBlur:
    """Gaussian-blur an image (or batched video frames) inside a mask region only.

    Inverse mode (invert_mask=True) blurs the OUTSIDE of the mask, e.g. for
    "blur the background, keep the subject sharp" — opposite of the typical
    privacy-blur use case.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image or batched video frames [B,H,W,C]."}),
                "mask": ("MASK", {
                    "tooltip": (
                        "Mask defining where to blur. White (1.0) = blurred, black (0.0) = sharp. "
                        "Single-frame mask [1,H,W] broadcasts across video batch."
                    ),
                }),
                "blur_radius": ("INT", {
                    "default": 25, "min": 0, "max": 300, "step": 1,
                    "tooltip": (
                        "Gaussian blur kernel radius in pixels. 0 = no blur. Larger = more "
                        "destructive. For Seedance gate-bypass on a 720p face, 25-50 is typical."
                    ),
                }),
                "blur_sigma": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 100.0, "step": 0.5,
                    "tooltip": (
                        "Sigma of the gaussian. 0 = auto-derived from radius (OpenCV formula). "
                        "Override only if you want non-default heavy/light tail."
                    ),
                }),
                "mask_feather_px": ("INT", {
                    "default": 8, "min": 0, "max": 100, "step": 1,
                    "tooltip": (
                        "Soften the mask edge so the blur fades smoothly. 0 = hard edge "
                        "(visible boundary line). 8-16 is typical for natural transitions."
                    ),
                }),
                "invert_mask": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "False (default) = blur INSIDE the mask (privacy/face-blur use case). "
                        "True = blur OUTSIDE the mask (background blur use case)."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "blur"
    CATEGORY = "NV_Utils/image"
    DESCRIPTION = (
        "Selectively gaussian-blur the masked region of an image (or batched video frames). "
        "Built for Seedance real-person gate bypass: mask faces+body via SAM3, blur only those, "
        "keep background sharp. Also handles inverse (background-blur) for portrait depth effects."
    )

    def blur(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        blur_radius: int,
        blur_sigma: float,
        mask_feather_px: int,
        invert_mask: bool,
    ):
        # --- shape normalize ---
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4 or image.shape[-1] < 3:
            raise ValueError(
                f"[NV_ImageMaskedBlur] image must be [B,H,W,C>=3], got {tuple(image.shape)}"
            )
        # Take RGB only; preserve alpha if present
        if image.shape[-1] > 3:
            rgb = image[..., :3]
            alpha = image[..., 3:]
        else:
            rgb = image
            alpha = None

        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.ndim != 3:
            raise ValueError(
                f"[NV_ImageMaskedBlur] mask must be [H,W] or [B,H,W], got {tuple(mask.shape)}"
            )

        B_img, H, W, _ = rgb.shape
        # Broadcast single-frame mask across video batch
        if mask.shape[0] == 1 and B_img > 1:
            mask = mask.expand(B_img, -1, -1)
        elif mask.shape[0] != B_img:
            raise ValueError(
                f"[NV_ImageMaskedBlur] mask batch={mask.shape[0]} doesn't match image batch={B_img}. "
                f"Mask should be [1,H,W] (single, broadcast) or [{B_img},H,W]."
            )
        if mask.shape[1:] != (H, W):
            raise ValueError(
                f"[NV_ImageMaskedBlur] mask spatial={tuple(mask.shape[1:])} doesn't match "
                f"image spatial=({H},{W})."
            )

        # Align device + dtype to image
        mask = mask.to(device=rgb.device, dtype=rgb.dtype).clamp(0.0, 1.0)
        if invert_mask:
            mask = 1.0 - mask
        if mask_feather_px > 0:
            mask = _feather_mask_BHW(mask, mask_feather_px).clamp(0.0, 1.0)

        # If blur_radius is 0, output is just the input (no work to do)
        if blur_radius < 1:
            print("[NV_ImageMaskedBlur] blur_radius=0, passthrough")
            return (image.clamp(0.0, 1.0).to(dtype=torch.float32),)

        # Convert to BCHW for conv
        rgb_bchw = rgb.permute(0, 3, 1, 2).contiguous()
        blurred_bchw = _gaussian_blur_BCHW(rgb_bchw, blur_radius, blur_sigma)
        blurred = blurred_bchw.permute(0, 2, 3, 1).contiguous()

        # Alpha-blend: out = original * (1 - mask) + blurred * mask
        m = mask.unsqueeze(-1)
        out_rgb = rgb * (1.0 - m) + blurred * m

        if alpha is not None:
            result = torch.cat([out_rgb, alpha], dim=-1)
        else:
            result = out_rgb

        result = result.clamp(0.0, 1.0).to(dtype=torch.float32)
        print(
            f"[NV_ImageMaskedBlur] {B_img} frame(s) {H}×{W}, "
            f"blur_radius={blur_radius} sigma={blur_sigma:.1f} "
            f"feather={mask_feather_px}px invert={invert_mask}"
        )
        return (result,)


NODE_CLASS_MAPPINGS = {
    "NV_ImageMaskedBlur": NV_ImageMaskedBlur,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_ImageMaskedBlur": "NV Image Masked Blur",
}
