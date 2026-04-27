"""
NV Static VACE Mask - Temporally invariant rectangular mask for VACE conditioning.

Architectural fix for residual head jitter in face-swap pipelines. Instead of
feeding VACE a temporally-varying silhouette mask (which the model interprets as
geometry motion via the 64-channel pixel-shuffle pack), feed a fixed rectangle
that's identical for every frame in the sequence. The face still moves naturally
in the final output because:

  1. The crop window tracks the head upstream (PointDrivenBBox + OptimizeCropTrajectory)
  2. CoTrackerBridge stabilizes face position within the crop
  3. The face is approximately stationary in crop coordinates by the time VACE sees it
  4. The static rectangle just demarcates "where VACE may paint"
  5. Position info still flows via input RGB pixels, reference frames, and attention
  6. Final InpaintStitch2 uses the SAM3 silhouette to composite only face pixels back

The rectangle is quantized to the VAE stride (8px) so its boundary lands exactly
on VAE block boundaries. The 64-channel packed VACE control mask becomes
bit-identical across all frames in the sequence — zero temporal variance, no
moving boundary for the model to interpret as head motion.

Architecture validated via 3-round adversarial multi-AI debate (Codex + Gemini)
on 2026-04-26. Both AIs converged on this solution after independently challenging
each other's blind spots:
  - Codex caught: silhouette must be removed from RGB pixels too, not just mask
  - Gemini caught: dynamic-tracked rectangle still snaps at modulo-8 thresholds
The mutual fix: static (non-tracking) 8px-aligned rectangle, raw RGB to VACE.

Usage: place between your bbox/crop pipeline and NV_VaceControlVideoPrep. Output
this node's mask into VaceControlVideoPrep.mask, leave the silhouette routing to
NV_InpaintStitch2 unchanged.
"""

import torch


def _quantize_to_stride(value: int, stride: int = 8) -> int:
    """Round value down to nearest multiple of stride (always returns >= stride)."""
    return max(stride, (value // stride) * stride)


def _build_centered_rect_mask(
    height: int,
    width: int,
    rect_height: int,
    rect_width: int,
    center_offset_x: int = 0,
    center_offset_y: int = 0,
) -> torch.Tensor:
    """Build a [H, W] mask with a centered rectangle of 1.0, rest 0.0.

    Rectangle dimensions are quantized to VAE stride (8px) so the boundary
    lands exactly on VAE block boundaries.
    """
    mask = torch.zeros((height, width), dtype=torch.float32)

    cy = height // 2 + center_offset_y
    cx = width // 2 + center_offset_x
    half_h = rect_height // 2
    half_w = rect_width // 2

    y0 = max(0, cy - half_h)
    y1 = min(height, cy + half_h)
    x0 = max(0, cx - half_w)
    x1 = min(width, cx + half_w)

    # Snap to 8px grid so the resulting boundary is VAE-aligned
    y0 = (y0 // 8) * 8
    y1 = ((y1 + 7) // 8) * 8
    y1 = min(height, y1)
    x0 = (x0 // 8) * 8
    x1 = ((x1 + 7) // 8) * 8
    x1 = min(width, x1)

    mask[y0:y1, x0:x1] = 1.0
    return mask


class NV_StaticVaceMask:
    """Output a temporally invariant rectangular mask for VACE conditioning.

    Removes silhouette-driven boundary jitter by giving VACE a fixed rectangular
    region of interest that's identical for every frame in the sequence. The 64ch
    pixel-shuffle pack inside WanVaceToVideo becomes bit-identical across frames.

    Wire this node's output into NV_VaceControlVideoPrep.mask in place of the
    upstream stabilized silhouette. Keep silhouette routing to InpaintStitch2
    composite unchanged.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_mask": ("MASK", {
                    "tooltip": "Reference mask for length/dimensions only. Output mask shape "
                               "matches this. Pass any [T, H, W] mask sequence — typically the "
                               "stabilized silhouette so dimensions/length auto-match.",
                }),
                "rect_size_pct": ("FLOAT", {
                    "default": 0.75, "min": 0.25, "max": 1.0, "step": 0.05,
                    "tooltip": "Rectangle size as fraction of crop dimensions. 0.75 = rectangle "
                               "is 75%% of crop width/height, centered. Larger = more freedom for "
                               "model to paint, but more BG inside the rectangle gets regenerated "
                               "(then composited away by silhouette stitch). 0.6-0.85 typical.",
                }),
                "aspect_mode": (["square", "match_crop"], {
                    "default": "match_crop",
                    "tooltip": "square: rectangle is square (min of W, H). "
                               "match_crop: rectangle preserves crop aspect ratio.",
                }),
            },
            "optional": {
                "center_offset_x": ("INT", {
                    "default": 0, "min": -256, "max": 256, "step": 8,
                    "tooltip": "Horizontal offset of rectangle center from crop center, in pixels. "
                               "Quantized to 8px (VAE stride). 0 = centered.",
                }),
                "center_offset_y": ("INT", {
                    "default": 0, "min": -256, "max": 256, "step": 8,
                    "tooltip": "Vertical offset of rectangle center from crop center, in pixels. "
                               "Quantized to 8px (VAE stride). 0 = centered.",
                }),
                "vae_stride": ("INT", {
                    "default": 8, "min": 4, "max": 32, "step": 4,
                    "tooltip": "VAE spatial stride (8 for WAN). Rectangle dimensions and offsets "
                               "snap to this grid. Leave at 8 for WAN/VACE.",
                }),
            },
        }

    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("mask", "info")
    FUNCTION = "execute"
    CATEGORY = "NV_Utils/VACE"
    DESCRIPTION = (
        "Static rectangular mask for VACE conditioning — eliminates silhouette-driven "
        "head jitter by giving the model a temporally invariant ROI. Use as drop-in "
        "replacement for the silhouette mask going into NV_VaceControlVideoPrep. The "
        "real silhouette continues to drive the final InpaintStitch2 composite."
    )

    def execute(
        self,
        reference_mask,
        rect_size_pct,
        aspect_mode,
        center_offset_x=0,
        center_offset_y=0,
        vae_stride=8,
    ):
        if reference_mask.ndim == 2:
            reference_mask = reference_mask.unsqueeze(0)

        T, H, W = reference_mask.shape

        # Compute rectangle dimensions
        if aspect_mode == "square":
            base_dim = min(H, W)
            rect_h = int(base_dim * rect_size_pct)
            rect_w = rect_h
        else:  # match_crop
            rect_h = int(H * rect_size_pct)
            rect_w = int(W * rect_size_pct)

        # Snap dimensions to VAE stride
        rect_h = _quantize_to_stride(rect_h, vae_stride)
        rect_w = _quantize_to_stride(rect_w, vae_stride)

        # Snap offsets to VAE stride
        offset_x = (center_offset_x // vae_stride) * vae_stride
        offset_y = (center_offset_y // vae_stride) * vae_stride

        # Build a single static rectangular mask
        single_frame_mask = _build_centered_rect_mask(
            H, W, rect_h, rect_w,
            center_offset_x=offset_x,
            center_offset_y=offset_y,
        )

        # Replicate across temporal dimension on the same device as input
        device = reference_mask.device
        dtype = reference_mask.dtype
        out_mask = single_frame_mask.unsqueeze(0).expand(T, -1, -1).to(device=device, dtype=dtype).contiguous()

        # Diagnostic info
        cy = H // 2 + offset_y
        cx = W // 2 + offset_x
        info_lines = [
            "[NV_StaticVaceMask]",
            f"  Crop: {W}x{H}, length={T} frames",
            f"  Rectangle: {rect_w}x{rect_h} ({rect_w/W*100:.0f}%x{rect_h/H*100:.0f}% of crop)",
            f"  Center: ({cx}, {cy}) — offset ({offset_x}, {offset_y})",
            f"  VAE-aligned: {vae_stride}px stride — boundary at "
            f"x=[{(cx - rect_w//2)}:{(cx + rect_w//2)}], y=[{(cy - rect_h//2)}:{(cy + rect_h//2)}]",
            f"  Temporal variance: ZERO (mask identical across all {T} frames)",
        ]
        info = "\n".join(info_lines)
        print(info)

        return (out_mask, info)


NODE_CLASS_MAPPINGS = {
    "NV_StaticVaceMask": NV_StaticVaceMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_StaticVaceMask": "NV Static VACE Mask",
}
