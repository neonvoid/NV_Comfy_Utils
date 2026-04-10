"""
NV Texture Harmonize — Match generated crop's texture quality to surrounding context.

When an AI-generated crop is composited back into the original frame, it often
looks "too clean" — sharper, less noisy, more AI-perfect than the surrounding
footage. This node measures texture characteristics (sharpness, grain, micro-contrast)
from the original context pixels and adjusts the generated region to match.

Content-agnostic: adapts regardless of whether the source is clean studio footage,
noisy handheld video, compressed web video, or already AI-generated. If both sides
are equally clean, the node does almost nothing.

Algorithm stack (all pure PyTorch, ~30ms per frame):
  1. Laplacian Pyramid variance matching — sharpness + micro-contrast per frequency band
  2. Grain synthesis — per-channel noise profile matching via residual analysis

Place between CropColorFix and InpaintStitch.
"""

import torch
import torch.nn.functional as F

from .multiband_blend_stitch import (
    _make_gaussian_kernel_2d,
    build_laplacian_pyramid,
    collapse_laplacian_pyramid,
)


def _gaussian_blur_sep(tensor, sigma):
    """Separable Gaussian blur with reflect padding. tensor: [B, C, H, W]."""
    if sigma <= 0:
        return tensor
    k = int(sigma * 3) * 2 + 1
    k = max(k, 3)
    max_k = min(tensor.shape[2], tensor.shape[3]) * 2 - 1
    if k > max_k:
        k = max(max_k // 2 * 2 + 1, 3)
    x = torch.arange(k, dtype=torch.float32, device=tensor.device) - (k - 1) / 2
    gauss_1d = torch.exp(-0.5 * (x / max(sigma, 1e-6)) ** 2)
    gauss_1d = gauss_1d / gauss_1d.sum()
    C = tensor.shape[1]
    kh = gauss_1d.view(1, 1, 1, -1).expand(C, -1, -1, -1)
    padded = F.pad(tensor, (k // 2, k // 2, 0, 0), mode="reflect")
    out = F.conv2d(padded, kh, groups=C)
    kv = gauss_1d.view(1, 1, -1, 1).expand(C, -1, -1, -1)
    padded = F.pad(out, (0, 0, k // 2, k // 2), mode="reflect")
    return F.conv2d(padded, kv, groups=C)


def _masked_std(tensor, mask_4d, min_pixels=100):
    """Compute per-channel std over masked region. Returns [B, C, 1, 1] or None.

    Uses soft mask weights (not binarized) for smooth stats at coarse pyramid levels.
    Checks min pixel count per-frame (not batch average) to avoid hiding bad frames.
    """
    weights = mask_4d.clamp(0, 1)
    count = weights.sum(dim=(2, 3), keepdim=True).clamp(min=1)
    # Per-frame check: reject if ANY frame has too few pixels
    if count.min().item() < min_pixels:
        return None
    mean = (tensor * weights).sum(dim=(2, 3), keepdim=True) / count
    var = ((tensor - mean) ** 2 * weights).sum(dim=(2, 3), keepdim=True) / count
    return var.sqrt().clamp(min=1e-6)


def _masked_mad(tensor, mask_4d, min_pixels=100):
    """Compute per-channel MAD (Median Absolute Deviation) over masked region.

    Returns [B, C, 1, 1] scaled by 1.4826 to be std-equivalent for normal data,
    or None if any frame has too few masked pixels.

    USED ONLY FOR THE GRAIN STAGE, NOT THE SHARPNESS PYRAMID. Grain is the bulk
    distribution of high-freq residual after blur — most pixels have small noise,
    edges are outliers that pollute the measurement. MAD captures the bulk and
    ignores edge contamination. The sharpness pyramid wants the OPPOSITE — band
    energy IS dominated by edges, so std (which is sensitive to those outliers)
    is the right tool there.

    Implementation: binarized masks (>= 0.5). The grain stage operates at the
    full crop resolution so binarization is exact for hard masks; soft masks
    only matter at coarse pyramid levels which use std (not this function).
    """
    B, C, H, W = tensor.shape
    binary_mask = mask_4d.clamp(0, 1) >= 0.5  # [B, 1, H, W] bool
    count = binary_mask.sum(dim=(2, 3))  # [B, 1]
    if count.min().item() < min_pixels:
        return None

    result = torch.empty(B, C, 1, 1, device=tensor.device, dtype=tensor.dtype)
    for b in range(B):
        m = binary_mask[b, 0]  # [H, W]
        for c in range(C):
            values = tensor[b, c][m]  # [N] masked pixels
            median = torch.quantile(values, 0.5)
            abs_dev = (values - median).abs()
            mad = torch.quantile(abs_dev, 0.5)
            result[b, c, 0, 0] = mad * 1.4826

    return result.clamp(min=1e-6)


def _downsample_mask_to(mask_4d, target_h, target_w):
    """Downsample [B, 1, H, W] mask to target spatial dims."""
    if mask_4d.shape[2] == target_h and mask_4d.shape[3] == target_w:
        return mask_4d
    return F.interpolate(mask_4d, size=(target_h, target_w), mode="bilinear", align_corners=False)


def _build_full_frame_context_from_stitcher(stitcher, device, dtype):
    """Reconstruct the full-frame context per batch frame from a STITCHER dict.

    Returns:
        full_frames [B, C, frame_h, frame_w]: original frames at native resolution
        ctx_masks   [B, 1, frame_h, frame_w]: 1.0 = full-frame context (excluding inpaint area)
        gen_masks   [B, 1, frame_h, frame_w]: 1.0 = inpaint area (in native frame coords)
        ctc_dims    list of (ctc_w, ctc_h) tuples per frame — native crop bbox dims

    All frames in a stitcher batch share the same canvas (frame) dimensions, so
    we stack them into a single tensor. Per-frame bbox positions still vary.
    """
    canvas_images = stitcher['canvas_image']
    cropped_blend_masks = stitcher['cropped_mask_for_blend']
    ctc_x = stitcher['cropped_to_canvas_x']
    ctc_y = stitcher['cropped_to_canvas_y']
    ctc_w_list = stitcher['cropped_to_canvas_w']
    ctc_h_list = stitcher['cropped_to_canvas_h']

    n = len(canvas_images)
    if n == 0:
        return None, None, None, None

    # All canvas frames in a NV_InpaintCrop2 stitcher share the same native dims
    # (canvas_image == original full frame, with cto_x/y == 0, cto_w/h == frame dims)
    first = canvas_images[0]
    frame_h, frame_w = first.shape[0], first.shape[1]

    full_frames = torch.empty(n, frame_h, frame_w, first.shape[2], device=device, dtype=dtype)
    inpaint_masks = torch.zeros(n, 1, frame_h, frame_w, device=device, dtype=dtype)
    ctc_dims = []

    for b in range(n):
        # Move full frame to the working device + dtype
        full_frames[b] = canvas_images[b].to(device=device, dtype=dtype)

        # The blend mask is in crop processing space (e.g., 832×832).
        # Resize back to native crop bbox dims, then paste at the bbox position.
        blend = cropped_blend_masks[b].to(device=device, dtype=dtype)  # [crop_h, crop_w]
        if blend.dim() == 2:
            blend = blend.unsqueeze(0).unsqueeze(0)  # [1, 1, crop_h, crop_w]
        elif blend.dim() == 3:
            blend = blend.unsqueeze(1)  # [1, 1, crop_h, crop_w]

        cw = int(ctc_w_list[b])
        ch = int(ctc_h_list[b])
        cx = int(ctc_x[b])
        cy = int(ctc_y[b])
        ctc_dims.append((cw, ch))

        # Resize blend mask to native bbox dims
        if blend.shape[2] != ch or blend.shape[3] != cw:
            blend_native = F.interpolate(blend, size=(ch, cw), mode="bilinear", align_corners=False)
        else:
            blend_native = blend

        # Clip to frame bounds (defensive — bbox should already be in-bounds from cropper)
        x0 = max(0, cx)
        y0 = max(0, cy)
        x1 = min(frame_w, cx + cw)
        y1 = min(frame_h, cy + ch)
        bw = x1 - x0
        bh = y1 - y0
        if bw <= 0 or bh <= 0:
            continue
        sx = x0 - cx
        sy = y0 - cy
        inpaint_masks[b, 0, y0:y1, x0:x1] = blend_native[0, 0, sy:sy+bh, sx:sx+bw]

    # Permute to NCHW
    full_frames_nchw = full_frames.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]

    # Context = inverse of inpaint area (binarize at 0.5 to avoid soft-edge contamination)
    ctx_masks = (inpaint_masks < 0.5).to(dtype)
    gen_masks = (inpaint_masks >= 0.5).to(dtype)

    return full_frames_nchw, ctx_masks, gen_masks, ctc_dims


class NV_TextureHarmonize:
    """Match generated crop's texture quality to surrounding context."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "generated_crop": ("IMAGE", {
                    "tooltip": "Generated crop after CropColorFix (post color-correction, pre-stitch)."
                }),
                "original_crop": ("IMAGE", {
                    "tooltip": "Original cropped source from InpaintCrop (unmodified source crop)."
                }),
                "mask": ("MASK", {
                    "tooltip": "Crop-space mask. 1 = generated region, 0 = original context."
                }),
            },
            "optional": {
                "sharpness_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Sharpness/micro-contrast matching strength. "
                               "1.0 = full match. 0.0 = disabled."
                }),
                "grain_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Grain/noise matching strength. "
                               "1.0 = full match. 0.0 = disabled."
                }),
                "pyramid_levels": ("INT", {
                    "default": 4, "min": 2, "max": 6, "step": 1,
                    "tooltip": "Laplacian pyramid levels. 4 = good for 832x832."
                }),
                "grain_blur_sigma": ("FLOAT", {
                    "default": 1.5, "min": 0.5, "max": 5.0, "step": 0.5,
                    "tooltip": "Blur sigma for grain extraction. "
                               "Larger = coarser grain estimate."
                }),
                "skip_base_levels": ("INT", {
                    "default": 1, "min": 0, "max": 3, "step": 1,
                    "tooltip": "Skip this many low-frequency (base) pyramid levels. "
                               "1 = skip base residual (color already handled by CropColorFix). "
                               "Pyramid: level 0 = finest detail, last = base color."
                }),
                "temporal_seed": ("INT", {
                    "default": 0, "min": 0,
                    "tooltip": "Base seed for grain synthesis. Use frame number for temporal variation."
                }),
                "min_context_pixels": ("INT", {
                    "default": 1000, "min": 100, "max": 50000, "step": 100,
                    "tooltip": "Minimum context pixels needed for statistics. Below this, skip matching."
                }),
                "context_scope": (["ring_only", "whole_crop", "full_frame"], {
                    "default": "ring_only",
                    "tooltip": "Where to measure original-footage texture stats. "
                               "ring_only = only the mask=0 donut around the inpaint area (smallest sample, most local). "
                               "whole_crop = entire original_crop including the masked region (14× more samples, "
                               "same lighting). "
                               "full_frame = use the entire original full frame from stitcher (largest sample, "
                               "matches the actual stitch target since the inpaint will be composited back into "
                               "the full frame). REQUIRES stitcher input — falls back to whole_crop if missing."
                }),
                "stitcher": ("STITCHER", {
                    "tooltip": "Optional STITCHER from NV_InpaintCrop2. Required for context_scope=full_frame. "
                               "Provides the original full-frame data so we can measure texture statistics "
                               "from the actual full frame (which is what the inpaint will be stitched back into) "
                               "instead of just the crop window."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("harmonized_crop", "info")
    FUNCTION = "execute"
    CATEGORY = "NV_Utils/Inpaint"
    DESCRIPTION = (
        "Matches generated crop's texture quality (sharpness, grain, micro-contrast) "
        "to the surrounding original context. Content-agnostic — adapts to clean studio "
        "footage, noisy handheld, compressed web video, or prior AI generations. "
        "Place between CropColorFix and InpaintStitch."
    )

    def execute(self, generated_crop, original_crop, mask,
                sharpness_strength=1.0, grain_strength=1.0,
                pyramid_levels=4, grain_blur_sigma=1.5,
                skip_base_levels=1, temporal_seed=0, min_context_pixels=1000,
                context_scope="ring_only", stitcher=None):
        TAG = "[NV_TextureHarmonize]"
        device = generated_crop.device
        B, H, W, C = generated_crop.shape

        # Resolve full_frame mode — falls back to whole_crop if stitcher missing or wrong size
        effective_scope = context_scope
        if context_scope == "full_frame":
            if stitcher is None:
                print(f"{TAG} WARNING: context_scope=full_frame requires stitcher input — falling back to whole_crop")
                effective_scope = "whole_crop"
            else:
                stitcher_n = len(stitcher.get('canvas_image', []))
                if stitcher_n != B:
                    print(f"{TAG} WARNING: stitcher has {stitcher_n} frames but generated_crop has {B} — "
                          f"falling back to whole_crop. (Skipped frames cause this; route the texture node "
                          f"on post-skip outputs only.)")
                    effective_scope = "whole_crop"

        info_lines = [f"{TAG} {B} frames, {H}x{W}, sharpness={sharpness_strength}, "
                      f"grain={grain_strength}, scope={effective_scope} "
                      f"(sharpness=std/soft-mask, grain=mad/binary-mask)"]

        if original_crop.shape[0] != B:
            raise ValueError(f"{TAG} Frame count mismatch: generated={B}, original={original_crop.shape[0]}")

        # --- Prepare mask [B, H, W] → [B, 1, H, W] ---
        m = mask.float().to(device)
        if m.dim() == 2:
            m = m.unsqueeze(0)
        if m.shape[0] == 1 and B > 1:
            m = m.expand(B, -1, -1)
        if m.shape[1:] != (H, W):
            m = F.interpolate(m.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False).squeeze(1)

        ctx_4d = (m < 0.5).float().unsqueeze(1)   # [B, 1, H, W] context (mask=0 ring)
        gen_4d = (m >= 0.5).float().unsqueeze(1)   # [B, 1, H, W] generated (mask=1 region)

        # Measurement masks: where we sample stats from. Application masks (gen_4d/ctx_4d
        # above) are unchanged — we still only modify the inpaint region.
        if effective_scope == "whole_crop":
            meas_ctx_4d = torch.ones_like(ctx_4d)
            meas_gen_4d = torch.ones_like(gen_4d)
        else:  # ring_only or full_frame (full_frame uses these only as fallback for non-stitcher data)
            meas_ctx_4d = ctx_4d
            meas_gen_4d = gen_4d

        # Convert crop tensors to NCHW
        gen = generated_crop.permute(0, 3, 1, 2).float()   # [B, C, H, W]
        orig = original_crop.permute(0, 3, 1, 2).float()
        result = gen.clone()

        # =================================================================
        # Full-frame data path (only when effective_scope == "full_frame")
        # =================================================================
        # We pull the full original frames from the stitcher, build a per-frame
        # context mask in native coords, and use these as the MEASUREMENT source.
        # The generated_crop / original_crop are still the APPLICATION targets —
        # we only change WHERE we measure stats from.
        ff_full_frames = None  # [B, C, frame_h, frame_w]
        ff_ctx_4d = None       # [B, 1, frame_h, frame_w] full-frame context mask
        ff_ctc_dims = None     # list of (ctc_w, ctc_h)
        if effective_scope == "full_frame":
            ff_full_frames, ff_ctx_4d, _, ff_ctc_dims = _build_full_frame_context_from_stitcher(
                stitcher, device, gen.dtype
            )
            if ff_full_frames is None:
                print(f"{TAG} WARNING: stitcher full-frame reconstruction failed — falling back to whole_crop")
                effective_scope = "whole_crop"
                meas_ctx_4d = torch.ones_like(ctx_4d)
                meas_gen_4d = torch.ones_like(gen_4d)

        # Check we have enough measurement support (use full-frame mask if active)
        if effective_scope == "full_frame":
            avg_ctx_px = ff_ctx_4d.sum(dim=(2, 3)).mean().item()
        else:
            avg_ctx_px = meas_ctx_4d.sum(dim=(2, 3)).mean().item()
        if avg_ctx_px < min_context_pixels:
            info_lines.append(f"  Skipped: {int(avg_ctx_px)} measurement pixels < {min_context_pixels}")
            info = "\n".join(info_lines)
            print(info)
            return (generated_crop, info)

        # =================================================================
        # Stage 1: Laplacian Pyramid Sharpness/Micro-Contrast Matching
        # =================================================================
        if sharpness_strength > 0:
            kernel = _make_gaussian_kernel_2d(5).to(device)
            gen_pyr = build_laplacian_pyramid(result, pyramid_levels, kernel)

            # Build the REFERENCE pyramid: in full_frame mode, resize the full frame
            # to crop dims first so the pyramid bands align with the crop's processing
            # resolution. The same resize kernel is used consistently, so band-energy
            # ratios remain meaningful even though absolute spectral content differs.
            if effective_scope == "full_frame":
                ref_source = F.interpolate(ff_full_frames, size=(H, W), mode="bilinear", align_corners=False)
                ref_ctx_mask = F.interpolate(ff_ctx_4d, size=(H, W), mode="bilinear", align_corners=False)
                ref_pyr = build_laplacian_pyramid(ref_source, pyramid_levels, kernel)
            else:
                ref_pyr = build_laplacian_pyramid(orig, pyramid_levels, kernel)
                ref_ctx_mask = None  # use the per-level downsampled crop-space mask below

            # Pyramid layout: [0]=finest detail, ..., [N-1]=coarse detail, [N]=base residual
            # Process from finest (0) up to but excluding base levels at the end.
            # skip_base_levels=1 means skip the base residual [N] (color — CropColorFix handles it)
            num_detail_levels = len(gen_pyr) - 1  # exclude base residual
            process_up_to = num_detail_levels - skip_base_levels

            ratios = []
            for lvl in range(len(gen_pyr)):
                if lvl >= process_up_to:
                    ratios.append("-")
                    continue

                lh, lw = gen_pyr[lvl].shape[2], gen_pyr[lvl].shape[3]
                lvl_gen = _downsample_mask_to(gen_4d, lh, lw)  # application mask (mask=1 only)
                lvl_meas_gen = _downsample_mask_to(meas_gen_4d, lh, lw)

                # Reference (ctx) measurement mask — full-frame or crop-space
                if effective_scope == "full_frame":
                    lvl_meas_ctx = _downsample_mask_to(ref_ctx_mask, lh, lw)
                else:
                    lvl_meas_ctx = _downsample_mask_to(meas_ctx_4d, lh, lw)

                # Sharpness uses std (not MAD) — the pyramid band energy IS the
                # outliers (edges). MAD would discard exactly what we want to measure.
                ctx_std = _masked_std(ref_pyr[lvl], lvl_meas_ctx)
                gen_std = _masked_std(gen_pyr[lvl], lvl_meas_gen)

                if ctx_std is None or gen_std is None:
                    ratios.append("-")
                    continue

                # Per-channel ratio clamped conservatively to avoid ringing/fringing
                ratio = (ctx_std / gen_std).clamp(0.25, 3.0)

                # Temporal stability: average ratios across batch to prevent flicker
                if B > 1:
                    ratio = ratio.mean(dim=0, keepdim=True).expand(B, -1, -1, -1)

                effective = (1.0 + sharpness_strength * (ratio - 1.0)).clamp(min=0.1)

                # Apply only to generated region
                gen_region = gen_pyr[lvl] * effective * lvl_gen
                ctx_region = gen_pyr[lvl] * (1.0 - lvl_gen)
                gen_pyr[lvl] = ctx_region + gen_region

                ratios.append(f"{ratio.mean().item():.2f}")

            result = collapse_laplacian_pyramid(gen_pyr, kernel)
            info_lines.append(f"  Sharpness: levels={pyramid_levels}, skip_base={skip_base_levels}, "
                              f"ratios=[{', '.join(str(r) for r in ratios)}]")

        # =================================================================
        # Stage 2: Grain Synthesis (Per-Channel Noise Matching)
        # =================================================================
        if grain_strength > 0:
            # Extract grain from context: high-freq residual = original - blur(original)
            # In full_frame mode we measure on the NATIVE-RESOLUTION full frame, which
            # is the most representative sample of the camera/codec grain pipeline.
            # The grain residual is dominated by sensor noise + sharpening halos which
            # are roughly stationary across the frame — so the lighting-locality concern
            # that applies to the sharpness pyramid is much weaker here.
            if effective_scope == "full_frame":
                grain_source = ff_full_frames
                grain_ctx_mask = ff_ctx_4d
            else:
                grain_source = orig
                grain_ctx_mask = meas_ctx_4d

            grain_blur_src = _gaussian_blur_sep(grain_source, grain_blur_sigma)
            ctx_grain = grain_source - grain_blur_src

            # Generated grain is always measured on the crop (it's where we apply the noise)
            gen_blur = _gaussian_blur_sep(result, grain_blur_sigma)
            gen_grain = result - gen_blur

            grain_added = []
            generator = torch.Generator(device=device)

            for c in range(min(C, 3)):
                # Measure context grain std (per channel)
                ctx_ch = ctx_grain[:, c:c+1, :, :]
                # Grain uses MAD — the noise residual's BULK is the signal,
                # edges in the residual are texture pollution we want to ignore.
                ctx_std = _masked_mad(ctx_ch, grain_ctx_mask)
                if ctx_std is None:
                    grain_added.append("skip")
                    continue
                # Average across batch for temporal stability
                ctx_val = ctx_std.mean().item()

                # Measure generated grain std
                gen_ch = gen_grain[:, c:c+1, :, :]
                gen_std = _masked_mad(gen_ch, meas_gen_4d)
                gen_val = gen_std.mean().item() if gen_std is not None else 0.0

                # Additional grain needed = difference (only add, don't subtract)
                additional = max(0.0, ctx_val - gen_val)
                if additional < 1e-5:
                    grain_added.append(f"{ctx_val:.4f}={gen_val:.4f}")
                    continue

                # Per-frame deterministic noise — unique seed per frame to avoid
                # fixed-overlay look and grain reset between batch chunks
                noise = torch.empty(B, 1, H, W, device=device)
                for b in range(B):
                    generator.manual_seed(temporal_seed + b * 97 + c * 13)
                    noise[b:b+1] = torch.randn(1, 1, H, W, device=device, generator=generator)
                noise = noise * additional * grain_strength

                # Apply only to generated region
                result[:, c:c+1, :, :] = result[:, c:c+1, :, :] + noise * gen_4d
                grain_added.append(f"+{additional:.4f}")

            info_lines.append(f"  Grain: sigma={grain_blur_sigma}, channels=[{', '.join(grain_added)}]")

        # Clamp and convert back to BHWC
        result = result.clamp(0.0, 1.0).permute(0, 2, 3, 1)
        info = "\n".join(info_lines)
        print(info)

        return (result, info)


NODE_CLASS_MAPPINGS = {
    "NV_TextureHarmonize": NV_TextureHarmonize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_TextureHarmonize": "NV Texture Harmonize",
}
