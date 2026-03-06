"""
NV Latent Inpaint Crop — 5D-safe latent crop for WAN video inpainting.

Latent-space equivalent of NV_InpaintCrop2. Crops a spatial region from a
video latent [B,C,T,H,W] and packages stitching metadata for NV_LatentInpaintStitch.

Key differences from pixel-space InpaintCrop:
- Union bbox (single static crop for all frames) — per-frame variation via noise_mask
- No canvas expansion — subject is always in-bounds (Stage 1 output)
- Optional target resize — crop can be scaled to a target resolution for KSampler,
  then NV_LatentInpaintStitch resizes back before pasting
- Uses ellipsis indexing for 5D safety (built-in LatentCrop is broken for 5D)

Pipeline:
    [Stage 1 Latent] → NV_LatentInpaintCrop → cropped_latent → KSampler
                                             → stitcher ────────────────→ NV_LatentInpaintStitch
                                             → cropped_mask → SetLatentNoiseMask ↗
"""

import torch
import torch.nn.functional as F
import comfy.model_management
import comfy.utils

from .latent_constants import LATENT_SAFE_KEYS
from .inpaint_crop import (
    mask_erode_dilate as _mask_erode_dilate,
    mask_fill_holes as _mask_fill_holes,
    mask_remove_noise as _mask_remove_noise,
    mask_smooth as _mask_smooth,
    mask_blur as _mask_blur,
    rescale_mask as _rescale_mask,
    compute_auto_resolution,
    WAN_PRESETS,
)

VAE_STRIDE = 8  # WAN 2.1 spatial compression factor


# =============================================================================
# Helper Functions
# =============================================================================

def compute_union_bbox_from_mask(bbox_mask, spatial_h_px, spatial_w_px):
    """Compute pixel-space union bbox from [B,H,W] or [H,W] bbox_mask.

    Max-reduces across all frames to get the envelope covering the subject's
    full trajectory. Handles resolution mismatch between mask and latent pixel space.

    Returns (x, y, w, h) as floats in pixel space, or None if mask is empty.
    """
    if bbox_mask.dim() == 2:
        m = bbox_mask
    else:
        # [B, H, W] — union across all frames
        m = bbox_mask.max(dim=0).values

    non_zero = torch.nonzero(m > 0.01)
    if non_zero.numel() == 0:
        return None

    y_min = non_zero[:, -2].min().item()
    y_max = non_zero[:, -2].max().item()
    x_min = non_zero[:, -1].min().item()
    x_max = non_zero[:, -1].max().item()

    # Scale if mask resolution differs from latent pixel space
    mask_h, mask_w = m.shape[-2], m.shape[-1]
    if mask_h != spatial_h_px or mask_w != spatial_w_px:
        scale_y = spatial_h_px / mask_h
        scale_x = spatial_w_px / mask_w
        x_min = x_min * scale_x
        x_max = x_max * scale_x
        y_min = y_min * scale_y
        y_max = y_max * scale_y

    return (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)


def snap_to_vae_grid(x, y, w, h, spatial_h_px, spatial_w_px, stride=VAE_STRIDE):
    """Snap crop coordinates to VAE grid boundaries.

    Origin is floor-snapped, dimensions are ceil-snapped, then clamped to image bounds.
    Returns (x, y, w, h) as ints, all multiples of stride.
    """
    x = (int(x) // stride) * stride
    y = (int(y) // stride) * stride
    w = max(stride, ((int(w) + stride - 1) // stride) * stride)
    h = max(stride, ((int(h) + stride - 1) // stride) * stride)

    # Clamp origin first, then clamp dimensions to stay in bounds
    x = max(0, min(x, spatial_w_px - stride))
    y = max(0, min(y, spatial_h_px - stride))
    if x + w > spatial_w_px:
        w = ((spatial_w_px - x) // stride) * stride
        w = max(stride, w)
    if y + h > spatial_h_px:
        h = ((spatial_h_px - y) // stride) * stride
        h = max(stride, h)

    return x, y, w, h


# =============================================================================
# Node Class
# =============================================================================

class NV_LatentInpaintCrop:
    """5D-safe latent crop for WAN video inpainting with STITCHER output.

    Crops a static spatial region (bbox union across all frames) from a video latent
    and packages the original latent + coordinates into a LATENT_STITCHER for
    NV_LatentInpaintStitch to paste back after denoising.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "padding": ("INT", {
                    "default": 0, "min": 0, "max": 512, "step": 8,
                    "tooltip": "Extra padding around bbox union in pixels (snapped to VAE 8px grid). "
                               "Provides surrounding context for the denoiser."
                }),

                # Aspect ratio control
                "crop_aspect": (["off", "auto", "manual"], {
                    "default": "off",
                    "tooltip": "off: crop = raw bbox + padding (no aspect constraint). "
                               "auto: grow crop to match WAN preset aspect ratio. "
                               "manual: grow crop to match target_width/target_height aspect ratio. "
                               "No resize — only the spatial extent changes."
                }),
                "auto_preset": (list(WAN_PRESETS.keys()), {
                    "default": "WAN_480p",
                    "tooltip": "Resolution preset for auto mode. "
                               "WAN_480p: ~400k pixels. WAN_720p: ~920k pixels."
                }),
                "target_width": ("INT", {
                    "default": 512, "min": 64, "max": 8192, "step": 8,
                    "tooltip": "Target width for manual aspect/resize mode. "
                               "Used by crop_aspect for aspect ratio only, "
                               "and by target_mode for actual resize dimensions."
                }),
                "target_height": ("INT", {
                    "default": 512, "min": 64, "max": 8192, "step": 8,
                    "tooltip": "Target height for manual aspect/resize mode. "
                               "Used by crop_aspect for aspect ratio only, "
                               "and by target_mode for actual resize dimensions."
                }),

                # Target resolution resize
                "target_mode": (["off", "auto", "manual"], {
                    "default": "off",
                    "tooltip": "off: native latent res (current behavior). "
                               "auto: resize crop to WAN preset resolution (uses auto_preset). "
                               "manual: resize crop to target_width x target_height. "
                               "NV_LatentInpaintStitch resizes back automatically. "
                               "Best combined with crop_aspect=auto (same preset) to match aspect before resize."
                }),
                "resize_method": (["bicubic", "bilinear", "nearest-exact", "area", "bislerp", "lanczos"], {
                    "default": "bislerp",
                    "tooltip": "Interpolation method for resizing latent. "
                               "bislerp: best quality for latents (slerp-based). "
                               "bicubic: good general quality. bilinear: fast."
                }),

                # Mask processing (applied to processed mask only)
                "mask_erode_dilate": ("INT", {
                    "default": 0, "min": -128, "max": 128, "step": 1,
                    "tooltip": "Shrink (negative) or expand (positive) the subject mask using grey morphology. "
                               "Applied to the processed mask output only; original mask stays untouched for stitching."
                }),
                "mask_fill_holes": ("INT", {
                    "default": 0, "min": 0, "max": 128, "step": 1,
                    "tooltip": "Fill gaps/holes in mask using morphological closing (dilate then erode). "
                               "Useful for masks with unwanted holes from segmentation."
                }),
                "mask_remove_noise": ("INT", {
                    "default": 0, "min": 0, "max": 32, "step": 1,
                    "tooltip": "Remove isolated pixels/specks using morphological opening (erode then dilate). "
                               "Keeps main mask regions intact while eliminating stray pixels."
                }),
                "mask_smooth": ("INT", {
                    "default": 0, "min": 0, "max": 127, "step": 1,
                    "tooltip": "Smooth jagged mask edges by binarizing (threshold 0.5) then Gaussian blurring. "
                               "Creates cleaner edges than direct blur. Value must be odd (auto-adjusted if even)."
                }),

                # Blend settings (for stitching)
                "mask_blend_pixels": ("INT", {
                    "default": 16, "min": 0, "max": 64, "step": 1,
                    "tooltip": "Feather the original mask edges for seamless stitching (dilate + blur). "
                               "Stored in stitcher for automatic use by NV_LatentInpaintStitch. "
                               "0 = hard paste (no blend mask pre-computed)."
                }),
            },
            "optional": {
                "mask_config": ("MASK_PROCESSING_CONFIG", {
                    "tooltip": "Optional shared config from NV_MaskProcessingConfig. "
                               "When connected, overrides this node's local mask processing widgets "
                               "(erode_dilate, fill_holes, remove_noise, smooth, blend_pixels)."
                }),
                "bbox_mask": ("MASK", {
                    "tooltip": "Bounding box mask [B,H,W] from MaskTrackingBBox. "
                               "Union across ALL frames determines crop region. "
                               "When provided, manual x/y/width/height are ignored."
                }),
                "subject_mask": ("MASK", {
                    "tooltip": "Subject silhouette mask [B,H,W]. Cropped to the same region "
                               "and output as cropped_mask for SetLatentNoiseMask. "
                               "If omitted, cropped_mask is all-ones (refine entire crop)."
                }),
                "x": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8,
                       "tooltip": "Manual crop left edge in pixels. Ignored when bbox_mask is connected."}),
                "y": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8,
                       "tooltip": "Manual crop top edge in pixels. Ignored when bbox_mask is connected."}),
                "width": ("INT", {"default": 512, "min": 8, "max": 8192, "step": 8,
                           "tooltip": "Manual crop width in pixels. Ignored when bbox_mask is connected."}),
                "height": ("INT", {"default": 512, "min": 8, "max": 8192, "step": 8,
                            "tooltip": "Manual crop height in pixels. Ignored when bbox_mask is connected."}),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT_STITCHER", "MASK", "MASK", "STRING")
    RETURN_NAMES = ("latent", "stitcher", "cropped_mask", "cropped_mask_processed", "info")
    FUNCTION = "execute"
    CATEGORY = "NV_Utils/Inpaint"
    DESCRIPTION = (
        "5D-safe latent crop for WAN video inpainting. "
        "Crops a spatial region (bbox union across all frames) from a video latent "
        "and packages coordinates into a LATENT_STITCHER for paste-back after denoising. "
        "Built-in LatentCrop is broken for 5D — this uses ellipsis indexing."
    )

    def execute(self, latent, padding=0,
                crop_aspect="off", auto_preset="WAN_480p",
                target_width=512, target_height=512,
                target_mode="off", resize_method="bislerp",
                mask_erode_dilate=0, mask_fill_holes=0,
                mask_remove_noise=0, mask_smooth=0,
                mask_blend_pixels=16,
                mask_config=None, bbox_mask=None, subject_mask=None,
                x=0, y=0, width=512, height=512):

        # Apply shared config override if connected
        from .mask_processing_config import apply_mask_config
        vals = apply_mask_config(mask_config,
            mask_erode_dilate=mask_erode_dilate,
            mask_fill_holes=mask_fill_holes,
            mask_remove_noise=mask_remove_noise,
            mask_smooth=mask_smooth,
            mask_blend_pixels=mask_blend_pixels,
        )
        mask_erode_dilate = vals["mask_erode_dilate"]
        mask_fill_holes = vals["mask_fill_holes"]
        mask_remove_noise = vals["mask_remove_noise"]
        mask_smooth = vals["mask_smooth"]
        mask_blend_pixels = vals["mask_blend_pixels"]

        samples = latent["samples"]
        info_lines = []

        # --- Validate 5D ---
        if samples.ndim != 5:
            raise ValueError(
                f"[NV_LatentInpaintCrop] Expected 5D video latent [B,C,T,H,W], "
                f"got {samples.ndim}D: {list(samples.shape)}"
            )

        B, C, T, H_l, W_l = samples.shape
        spatial_h_px = H_l * VAE_STRIDE
        spatial_w_px = W_l * VAE_STRIDE
        info_lines.append(
            f"Input: {list(samples.shape)}, pixel space {spatial_w_px}x{spatial_h_px}"
        )

        # --- Determine crop region ---
        if bbox_mask is not None:
            result = compute_union_bbox_from_mask(bbox_mask, spatial_h_px, spatial_w_px)
            if result is None:
                raise ValueError(
                    "[NV_LatentInpaintCrop] bbox_mask has no non-zero pixels."
                )
            bx, by, bw, bh = result
            num_frames = bbox_mask.shape[0] if bbox_mask.dim() == 3 else 1
            info_lines.append(
                f"BBox union ({num_frames} frames): "
                f"({bx:.1f},{by:.1f}) {bw:.1f}x{bh:.1f}px"
            )
        else:
            bx, by, bw, bh = float(x), float(y), float(width), float(height)
            info_lines.append(f"Manual crop: ({bx},{by}) {bw}x{bh}px")

        # Save raw bbox aspect before padding/growth for auto resolution computation.
        # This matches pixel InpaintCrop which uses raw bbox union aspect.
        raw_bbox_aspect = bw / bh

        # --- Apply padding ---
        if padding > 0:
            bx = max(0.0, bx - padding)
            by = max(0.0, by - padding)
            bw = min(float(spatial_w_px) - bx, bw + 2 * padding)
            bh = min(float(spatial_h_px) - by, bh + 2 * padding)
            info_lines.append(f"Padding: +{padding}px each side")

        # --- Aspect ratio growth ---
        if crop_aspect != "off":
            if crop_aspect == "auto":
                tw, th = compute_auto_resolution(raw_bbox_aspect, auto_preset, 0)
            else:  # manual
                tw, th = target_width, target_height

            target_aspect = tw / th
            bbox_aspect = bw / bh

            # Grow bbox in one dimension to match target aspect
            if bbox_aspect < target_aspect:
                # Need wider
                new_w = bh * target_aspect
                bx = bx - (new_w - bw) / 2.0
                bw = new_w
            else:
                # Need taller
                new_h = bw / target_aspect
                by = by - (new_h - bh) / 2.0
                bh = new_h

            # Shift to keep in bounds (no canvas expansion in latent space)
            if bx < 0:
                bx = 0.0
            elif bx + bw > spatial_w_px:
                bx = float(spatial_w_px) - bw
            if by < 0:
                by = 0.0
            elif by + bh > spatial_h_px:
                by = float(spatial_h_px) - bh

            # Hard clamp if still out of bounds (grown bbox larger than frame)
            bx = max(0.0, bx)
            by = max(0.0, by)
            bw = min(bw, float(spatial_w_px) - bx)
            bh = min(bh, float(spatial_h_px) - by)

            info_lines.append(
                f"Aspect growth ({crop_aspect}): target {tw}x{th} "
                f"(ar={target_aspect:.3f}) -> bbox {bw:.0f}x{bh:.0f}px"
            )

        # --- Snap to VAE grid ---
        cx_px, cy_px, cw_px, ch_px = snap_to_vae_grid(
            bx, by, bw, bh, spatial_h_px, spatial_w_px
        )
        cx_l = cx_px // VAE_STRIDE
        cy_l = cy_px // VAE_STRIDE
        cw_l = cw_px // VAE_STRIDE
        ch_l = ch_px // VAE_STRIDE

        info_lines.append(
            f"Crop: pixel ({cx_px},{cy_px}) {cw_px}x{ch_px} | "
            f"latent ({cx_l},{cy_l}) {cw_l}x{ch_l}"
        )

        # --- Crop latent (5D-safe ellipsis) ---
        cropped_samples = samples[..., cy_l:cy_l + ch_l, cx_l:cx_l + cw_l].clone()
        info_lines.append(f"Cropped shape: {list(cropped_samples.shape)}")

        # --- Compute target resize dimensions ---
        target_h_latent = None
        target_w_latent = None

        if target_mode != "off":
            if target_mode == "auto":
                tw_px, th_px = compute_auto_resolution(raw_bbox_aspect, auto_preset, 0)
            else:  # manual
                tw_px, th_px = target_width, target_height

            # Snap target to VAE grid
            tw_px = max(VAE_STRIDE, ((tw_px + VAE_STRIDE - 1) // VAE_STRIDE) * VAE_STRIDE)
            th_px = max(VAE_STRIDE, ((th_px + VAE_STRIDE - 1) // VAE_STRIDE) * VAE_STRIDE)

            target_w_latent = tw_px // VAE_STRIDE
            target_h_latent = th_px // VAE_STRIDE

            info_lines.append(
                f"Target resize ({target_mode}): "
                f"crop {cw_px}x{ch_px}px -> target {tw_px}x{th_px}px | "
                f"latent {cw_l}x{ch_l} -> {target_w_latent}x{target_h_latent} "
                f"({resize_method})"
            )

        # --- Resize cropped latent to target dimensions ---
        if target_h_latent is not None and target_w_latent is not None:
            if target_h_latent != ch_l or target_w_latent != cw_l:
                cropped_samples = comfy.utils.common_upscale(
                    cropped_samples, target_w_latent, target_h_latent,
                    resize_method, "disabled"
                )
                info_lines.append(f"Resized latent: {list(cropped_samples.shape)}")

        # --- Build output latent dict (clean, safe-key pattern) ---
        out_latent = {"samples": cropped_samples}
        safe_keys_snapshot = {}
        for key in LATENT_SAFE_KEYS:
            if key in latent:
                out_latent[key] = latent[key]
                safe_keys_snapshot[key] = latent[key]

        # Crop noise_mask if present (and resize to target if active)
        if "noise_mask" in latent:
            nm = latent["noise_mask"]
            cropped_nm = nm[..., cy_l:cy_l + ch_l, cx_l:cx_l + cw_l].clone()

            if target_h_latent is not None and target_w_latent is not None:
                if target_h_latent != ch_l or target_w_latent != cw_l:
                    cropped_nm = comfy.utils.common_upscale(
                        cropped_nm, target_w_latent, target_h_latent,
                        "bilinear", "disabled"
                    )

            out_latent["noise_mask"] = cropped_nm
            info_lines.append(
                f"noise_mask: cropped{' + resized' if target_h_latent is not None else ''}"
            )

        # --- Crop subject_mask (pixel space) ---
        if subject_mask is not None:
            sm = subject_mask
            if sm.dim() == 2:
                sm = sm.unsqueeze(0)

            # Resize to pixel space if resolution differs
            sm_h, sm_w = sm.shape[-2], sm.shape[-1]
            if sm_h != spatial_h_px or sm_w != spatial_w_px:
                sm = F.interpolate(
                    sm.unsqueeze(1).float(),
                    size=(spatial_h_px, spatial_w_px),
                    mode='bilinear', align_corners=False
                ).squeeze(1)

            cropped_mask = sm[..., cy_px:cy_px + ch_px, cx_px:cx_px + cw_px].clone()
            info_lines.append(
                f"subject_mask cropped: {list(sm.shape)} -> {list(cropped_mask.shape)}"
            )
        else:
            # Default: all-ones (refine entire crop region)
            num_mask_frames = T if samples.ndim == 5 else 1
            cropped_mask = torch.ones(num_mask_frames, ch_px, cw_px)

        # --- Process mask for diffusion (apply all operations) ---
        # Skip processing when subject_mask is None (all-ones mask — nothing to clean up,
        # and negative erode_dilate would unexpectedly shrink the all-ones mask).
        cropped_mask_processed = cropped_mask.clone()
        if subject_mask is not None:
            mask_ops = []

            if mask_fill_holes > 0:
                cropped_mask_processed = _mask_fill_holes(cropped_mask_processed, mask_fill_holes)
                mask_ops.append(f"fill_holes={mask_fill_holes}")
            if mask_remove_noise > 0:
                cropped_mask_processed = _mask_remove_noise(cropped_mask_processed, mask_remove_noise)
                mask_ops.append(f"remove_noise={mask_remove_noise}")
            if mask_erode_dilate != 0:
                cropped_mask_processed = _mask_erode_dilate(cropped_mask_processed, mask_erode_dilate)
                mask_ops.append(f"erode_dilate={mask_erode_dilate}")
            if mask_smooth > 0:
                cropped_mask_processed = _mask_smooth(cropped_mask_processed, mask_smooth)
                mask_ops.append(f"smooth={mask_smooth}")

            if mask_ops:
                info_lines.append(f"Mask processing: {', '.join(mask_ops)}")

        # --- Pre-compute blend mask for stitching ---
        # MUST happen BEFORE mask resize: blend mask stays at original crop pixel
        # resolution because the stitch pastes at original crop coordinates.
        blend_mask_for_stitch = None
        if mask_blend_pixels > 0 and subject_mask is not None:
            blend_mask_for_stitch = cropped_mask.max(dim=0, keepdim=True).values
            blend_mask_for_stitch = _mask_erode_dilate(blend_mask_for_stitch, mask_blend_pixels)
            blend_mask_for_stitch = _mask_blur(blend_mask_for_stitch, mask_blend_pixels)
            info_lines.append(f"Blend mask: union + dilate+blur {mask_blend_pixels}px")

        # --- Resize output masks to target pixel dimensions ---
        # After blend mask (which needs original crop res), resize masks to match
        # the target resolution that KSampler / SetLatentNoiseMask will see.
        if target_h_latent is not None and target_w_latent is not None:
            target_mask_h = target_h_latent * VAE_STRIDE
            target_mask_w = target_w_latent * VAE_STRIDE
            if target_mask_h != ch_px or target_mask_w != cw_px:
                cropped_mask = _rescale_mask(cropped_mask, target_mask_w, target_mask_h, 'bilinear')
                cropped_mask_processed = _rescale_mask(
                    cropped_mask_processed, target_mask_w, target_mask_h, 'bilinear'
                )
                info_lines.append(
                    f"Masks resized: {cw_px}x{ch_px} -> {target_mask_w}x{target_mask_h}px"
                )

        # --- Build LATENT_STITCHER ---
        intermediate = comfy.model_management.intermediate_device()
        stitcher = {
            'original_samples': samples.to(intermediate),
            'crop_x_latent': cx_l,
            'crop_y_latent': cy_l,
            'crop_w_latent': cw_l,
            'crop_h_latent': ch_l,
            'crop_x_pixel': cx_px,
            'crop_y_pixel': cy_px,
            'crop_w_pixel': cw_px,
            'crop_h_pixel': ch_px,
            'safe_keys': safe_keys_snapshot,
            'target_w_latent': target_w_latent,
            'target_h_latent': target_h_latent,
            'resize_method': resize_method if target_h_latent is not None else None,
        }
        if blend_mask_for_stitch is not None:
            stitcher['blend_mask'] = blend_mask_for_stitch.to(intermediate)

        info = "\n".join(info_lines)
        print(f"[NV_LatentInpaintCrop] {info}")

        return (out_latent, stitcher, cropped_mask, cropped_mask_processed, info)


# =============================================================================
# Node Registration
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "NV_LatentInpaintCrop": NV_LatentInpaintCrop,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_LatentInpaintCrop": "NV Latent Inpaint Crop",
}
