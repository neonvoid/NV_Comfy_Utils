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

Implementation review (multi-AI 2026-04-27) added:
  - Temporal union bounding mode (auto-fits rectangle to actual subject motion)
  - 0.9 hard cap on rect_size_pct (prevents inactive-context collapse)
  - Consistent stride handling throughout

DOWNSTREAM CONFIG REQUIREMENTS:
For this node to deliver its architectural benefit, NV_VaceControlVideoPrep
must NOT apply morphology that reintroduces variance. Set in MaskProcessingConfig:
  - cleanup_fill_holes: 0
  - vace_erosion_blocks: 0.0
  - vace_feather_blocks: 0.0
  - vace_halo_px: 0
And in NV_VaceControlVideoPrep:
  - threshold: True (cheap insurance for binary preservation)
  - mask_shape: as_is (the input is already a clean rectangle)
"""

import torch


def _snap_down(value: int, stride: int) -> int:
    """Floor-snap value to nearest multiple of stride."""
    return (value // stride) * stride


def _snap_up(value: int, stride: int) -> int:
    """Ceiling-snap value to nearest multiple of stride."""
    return ((value + stride - 1) // stride) * stride


def _build_static_rect_mask(
    height: int,
    width: int,
    cy: int,
    cx: int,
    rect_height: int,
    rect_width: int,
    vae_stride: int,
) -> tuple:
    """Build a [H, W] mask with a centered rectangle, all aligned to vae_stride.

    Returns (mask, realized_y0, realized_y1, realized_x0, realized_x1) so callers
    can report the actual painted bounds if clipping occurred.
    """
    half_h = rect_height // 2
    half_w = rect_width // 2

    y0 = max(0, cy - half_h)
    y1 = min(height, cy + half_h)
    x0 = max(0, cx - half_w)
    x1 = min(width, cx + half_w)

    # Snap boundaries to vae_stride grid (parameterized, not hardcoded)
    y0 = _snap_down(y0, vae_stride)
    y1 = min(height, _snap_up(y1, vae_stride))
    x0 = _snap_down(x0, vae_stride)
    x1 = min(width, _snap_up(x1, vae_stride))

    mask = torch.zeros((height, width), dtype=torch.float32)
    mask[y0:y1, x0:x1] = 1.0
    return mask, y0, y1, x0, x1


class NV_StaticVaceMask:
    """Output a temporally invariant rectangular mask for VACE conditioning.

    Removes silhouette-driven boundary jitter by giving VACE a fixed rectangular
    region of interest that's identical for every frame in the sequence. The 64ch
    pixel-shuffle pack inside WanVaceToVideo becomes bit-identical across frames.

    Wire this node's output into NV_VaceControlVideoPrep.mask in place of the
    upstream stabilized silhouette. Keep silhouette routing to InpaintStitch2
    composite unchanged.

    Two modes:
      - temporal_union: auto-derive rectangle from union of reference_mask across
        all frames (fire-and-forget, recommended)
      - manual: use rect_size_pct + offsets (legacy, full control)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_mask": ("MASK", {
                    "tooltip": "In temporal_union mode: used to compute the auto-rectangle. "
                               "In manual mode: used only for shape/length. Pass any [T, H, W] "
                               "mask sequence — typically the stabilized silhouette.",
                }),
                "mode": (["temporal_union", "manual"], {
                    "default": "temporal_union",
                    "tooltip": "temporal_union: auto-derive rectangle from union of all-frames mask "
                               "(recommended — fire-and-forget). "
                               "manual: use rect_size_pct + center offsets directly.",
                }),
            },
            "optional": {
                "padding_pct": ("FLOAT", {
                    "default": 0.1, "min": 0.0, "max": 0.5, "step": 0.05,
                    "tooltip": "Padding around temporal union bounds (temporal_union mode only). "
                               "0.1 = 10%% expansion. Snapped to vae_stride.",
                }),
                "aspect_mode": (["square", "match_subject", "match_crop"], {
                    "default": "square",
                    "tooltip": "square: rectangle is square (max of subject H, W). "
                               "match_subject: preserves subject bbox aspect. "
                               "match_crop: preserves crop aspect ratio.",
                }),
                "rect_size_pct": ("FLOAT", {
                    "default": 0.75, "min": 0.25, "max": 0.9, "step": 0.05,
                    "tooltip": "Rectangle size as fraction of crop dimensions (manual mode only). "
                               "Capped at 0.9 to preserve inactive context. 0.6-0.85 typical.",
                }),
                "center_offset_x": ("INT", {
                    "default": 0, "min": -512, "max": 512, "step": 8,
                    "tooltip": "Horizontal offset from default center, in pixels. Snapped to vae_stride.",
                }),
                "center_offset_y": ("INT", {
                    "default": 0, "min": -512, "max": 512, "step": 8,
                    "tooltip": "Vertical offset from default center, in pixels. Snapped to vae_stride.",
                }),
                "vae_stride": ("INT", {
                    "default": 8, "min": 4, "max": 32, "step": 4,
                    "tooltip": "VAE spatial stride (8 for WAN/VACE). All boundaries snap to this grid.",
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
        "real silhouette continues to drive the final InpaintStitch2 composite. "
        "Recommended: mode=temporal_union for fire-and-forget. "
        "REQUIRED downstream: cleanup_fill_holes=0, vace_erosion_blocks=0, "
        "vace_feather_blocks=0, threshold=True."
    )

    def execute(
        self,
        reference_mask,
        mode,
        padding_pct=0.1,
        aspect_mode="square",
        rect_size_pct=0.75,
        center_offset_x=0,
        center_offset_y=0,
        vae_stride=8,
    ):
        if reference_mask.ndim == 2:
            reference_mask = reference_mask.unsqueeze(0)

        T, H, W = reference_mask.shape

        # Snap offsets to vae_stride grid
        offset_x = _snap_down(center_offset_x, vae_stride) if center_offset_x >= 0 else -_snap_down(-center_offset_x, vae_stride)
        offset_y = _snap_down(center_offset_y, vae_stride) if center_offset_y >= 0 else -_snap_down(-center_offset_y, vae_stride)

        # Compute rectangle dimensions and center based on mode
        if mode == "temporal_union":
            # Auto-derive rectangle from temporal union of subject mask
            union_mask = reference_mask.max(dim=0).values > 0.01
            y_indices, x_indices = torch.where(union_mask)

            if y_indices.numel() == 0:
                # Empty mask — fall back to centered 60% rectangle
                cy = H // 2 + offset_y
                cx = W // 2 + offset_x
                rect_h = int(H * 0.6)
                rect_w = int(W * 0.6)
                fallback_note = " (FALLBACK: empty reference_mask, used 60% centered)"
            else:
                y_min = y_indices.min().item()
                y_max = y_indices.max().item()
                x_min = x_indices.min().item()
                x_max = x_indices.max().item()

                # Center on temporal union bounds
                cy = (y_min + y_max) // 2 + offset_y
                cx = (x_min + x_max) // 2 + offset_x

                # Subject bounds with padding
                subj_h = (y_max - y_min) + 1
                subj_w = (x_max - x_min) + 1
                pad_h = int(subj_h * padding_pct)
                pad_w = int(subj_w * padding_pct)

                if aspect_mode == "square":
                    base = max(subj_h, subj_w) + 2 * max(pad_h, pad_w)
                    rect_h = base
                    rect_w = base
                elif aspect_mode == "match_subject":
                    rect_h = subj_h + 2 * pad_h
                    rect_w = subj_w + 2 * pad_w
                else:  # match_crop
                    rect_h = int(H * 0.75)
                    rect_w = int(W * 0.75)

                fallback_note = ""
        else:  # manual
            cy = H // 2 + offset_y
            cx = W // 2 + offset_x

            # Cap rect_size_pct at 0.9 to preserve inactive context
            effective_pct = min(0.9, rect_size_pct)

            if aspect_mode in ("square", "match_subject"):
                base = int(min(H, W) * effective_pct)
                rect_h = base
                rect_w = base
            else:  # match_crop
                rect_h = int(H * effective_pct)
                rect_w = int(W * effective_pct)

            fallback_note = ""

        # Hard cap at 90% of crop dimensions to guarantee inactive context border
        rect_h = min(rect_h, int(H * 0.9))
        rect_w = min(rect_w, int(W * 0.9))

        # Snap rectangle dimensions to vae_stride
        rect_h = max(vae_stride, _snap_down(rect_h, vae_stride))
        rect_w = max(vae_stride, _snap_down(rect_w, vae_stride))

        # Build the mask
        single_frame_mask, y0, y1, x0, x1 = _build_static_rect_mask(
            H, W, cy, cx, rect_h, rect_w, vae_stride,
        )

        # Replicate across temporal dimension
        device = reference_mask.device
        dtype = reference_mask.dtype
        out_mask = (
            single_frame_mask.unsqueeze(0)
            .expand(T, -1, -1)
            .to(device=device, dtype=dtype)
            .contiguous()
        )

        realized_w = x1 - x0
        realized_h = y1 - y0
        info_lines = [
            f"[NV_StaticVaceMask] mode={mode}{fallback_note}",
            f"  Crop: {W}x{H}, length={T} frames",
            f"  Realized rect: {realized_w}x{realized_h} ({realized_w/W*100:.0f}%x{realized_h/H*100:.0f}% of crop)",
            f"  Bounds: x=[{x0}:{x1}] y=[{y0}:{y1}], center=({cx}, {cy})",
            f"  VAE stride: {vae_stride}px (all boundaries on this grid)",
            f"  Temporal variance: ZERO (mask identical across all {T} frames)",
            "  REMINDER: set cleanup_fill_holes=0, vace_erosion_blocks=0, vace_feather_blocks=0,",
            "            threshold=True downstream to preserve invariance.",
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
