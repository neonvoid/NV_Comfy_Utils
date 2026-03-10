"""
NV Crop Color Fix V2 — Multipass diffusion drift correction before stitching.

When KSampler runs at denoise < 1.0, it modifies the ENTIRE crop — not just
the masked region. Non-masked pixels come back color-shifted (~7/255 typical).
This creates a visible seam when stitched onto the original frame.

V2 multipass pipeline:
  Step 1: Low-frequency Lab-space color correction (eroded reference, preserves HF detail)
  Step 2: Boundary-local residual correction (distance-weighted, guided-filter regularized)
  Step 3: Frequency-aware composite (multiband Laplacian pyramid OR Gaussian alpha)

Place between VAE Decode and InpaintStitch2:
  InpaintCrop2 → VAE Encode → KSampler → VAE Decode
      → [NV_CropColorFix] → InpaintStitch2
"""

import torch
import torch.nn.functional as F

from .boundary_color_match import _rgb_to_lab, _lab_to_rgb
from .multiband_blend_stitch import multiband_blend


def _erode_mask_2d(mask, pixels):
    """Erode a [B, H, W] mask by `pixels` using max_pool2d on inverted mask."""
    if pixels <= 0:
        return mask
    inverted = 1.0 - mask.unsqueeze(1)  # [B, 1, H, W]
    eroded_inv = F.max_pool2d(inverted, kernel_size=2 * pixels + 1, stride=1, padding=pixels)
    return (1.0 - eroded_inv).squeeze(1)  # [B, H, W]


def _dilate_mask_2d(mask, pixels):
    """Dilate a [B, H, W] mask by `pixels` using max_pool2d."""
    if pixels <= 0:
        return mask
    return F.max_pool2d(mask.unsqueeze(1), kernel_size=2 * pixels + 1, stride=1, padding=pixels).squeeze(1)


def _gaussian_blur_reflect(tensor, sigma):
    """Apply separable Gaussian blur with reflect padding. tensor: [B, C, H, W] or [B, 1, H, W]."""
    if sigma <= 0:
        return tensor
    k = int(sigma * 3) * 2 + 1
    x = torch.arange(k, dtype=torch.float32, device=tensor.device) - (k - 1) / 2
    gauss_1d = torch.exp(-0.5 * (x / max(sigma, 1e-6)) ** 2)
    gauss_1d = gauss_1d / gauss_1d.sum()
    C = tensor.shape[1]
    # Horizontal pass
    kh = gauss_1d.view(1, 1, 1, -1).expand(C, -1, -1, -1)
    padded = F.pad(tensor, (k // 2, k // 2, 0, 0), mode="reflect")
    out = F.conv2d(padded, kh, groups=C)
    # Vertical pass
    kv = gauss_1d.view(1, 1, -1, 1).expand(C, -1, -1, -1)
    padded = F.pad(out, (0, 0, k // 2, k // 2), mode="reflect")
    return F.conv2d(padded, kv, groups=C)


class NV_CropColorFix:
    """Multipass color correction and composite for generated crops before stitching.

    V2 pipeline: LF Lab correction → boundary-local residual → frequency-aware composite.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_crop": ("IMAGE", {
                    "tooltip": "Original cropped image from InpaintCrop2 (the source crop before diffusion)."
                }),
                "generated_crop": ("IMAGE", {
                    "tooltip": "Generated crop from VAE Decode (after KSampler). Same resolution as original_crop."
                }),
                "mask": ("MASK", {
                    "tooltip": "Crop-space mask from InpaintCrop2 (cropped_mask or cropped_mask_processed). "
                               "mask=1 = generated region, mask=0 = kept region."
                }),
                "color_correction": (["lab_lowfreq", "lab_full", "reinhard", "mean_only", "none"], {
                    "default": "lab_lowfreq",
                    "tooltip": "lab_lowfreq: match mean+std on low-frequency component in Lab space (best — preserves texture). "
                               "lab_full: match mean+std on full image in Lab space (good for uniform drift). "
                               "reinhard: match mean+std per RGB channel (V1 legacy). "
                               "mean_only: match mean only in RGB (simplest). "
                               "none: skip color correction, only do composite."
                }),
                "composite_mode": (["multiband", "gaussian", "hard"], {
                    "default": "multiband",
                    "tooltip": "multiband: Laplacian pyramid blend — wide LF transition, tight HF (best seam hiding). "
                               "gaussian: Gaussian-blurred alpha mask (V1 legacy, simpler). "
                               "hard: binary mask paste, no feathering."
                }),
                "blend_pixels": ("INT", {
                    "default": 8, "min": 0, "max": 64, "step": 1,
                    "tooltip": "Controls transition width. For gaussian: blur sigma. For multiband: mask expansion before pyramid."
                }),
                "blend_expansion": ("INT", {
                    "default": 0, "min": 0, "max": 64, "step": 1,
                    "tooltip": "Dilate the mask outward by N pixels BEFORE compositing. "
                               "Pushes the blend zone further into original pixels. "
                               "0 = no expansion. 16-32 = wider gradient reaching into original."
                }),
            },
            "optional": {
                "ref_threshold": ("FLOAT", {
                    "default": 0.01, "min": 0.0, "max": 0.5, "step": 0.01,
                    "tooltip": "Pixels with mask value below this are used as color reference. "
                               "Default 0.01 = only truly untouched pixels."
                }),
                "ref_erosion": ("INT", {
                    "default": 8, "min": 0, "max": 32, "step": 1,
                    "tooltip": "Erode the reference region inward by N pixels to exclude boundary-contaminated pixels "
                               "from color statistics. 8 = 1 VAE block (recommended)."
                }),
                "lf_sigma": ("FLOAT", {
                    "default": 10.0, "min": 1.0, "max": 50.0, "step": 1.0,
                    "tooltip": "Gaussian sigma for low-frequency decomposition (lab_lowfreq mode only). "
                               "Larger = broader low-frequency component. 10 = good default for 512x512 crops."
                }),
                "boundary_correction": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable Step 2: boundary-local residual correction. Measures residual in a narrow ring "
                               "outside the mask and propagates correction inward with distance decay."
                }),
                "boundary_ring": ("INT", {
                    "default": 12, "min": 4, "max": 32, "step": 1,
                    "tooltip": "Width of the exterior ring for boundary residual measurement (pixels)."
                }),
                "boundary_decay": ("FLOAT", {
                    "default": 0.15, "min": 0.01, "max": 0.5, "step": 0.01,
                    "tooltip": "Exponential decay rate for boundary correction propagation inward. "
                               "Smaller = correction reaches further. 0.15 = ~10px effective radius."
                }),
                "temporal_smooth": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 0.95, "step": 0.05,
                    "tooltip": "EMA smoothing for correction values across frames (0=off, 0.3=moderate, 0.8=heavy). "
                               "Prevents frame-to-frame jitter in the color correction."
                }),
                "multiband_levels": ("INT", {
                    "default": 4, "min": 2, "max": 6, "step": 1,
                    "tooltip": "Pyramid levels for multiband composite mode. 4 = good default for 512x512 crops."
                }),
                "min_ref_pixels": ("INT", {
                    "default": 100, "min": 10, "max": 10000, "step": 10,
                    "tooltip": "Minimum non-masked pixels required for reliable statistics. "
                               "If fewer, falls back to mean_only or skips correction."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("corrected_crop", "correction_info",)
    FUNCTION = "execute"
    CATEGORY = "NV_Utils/Inpaint"
    DESCRIPTION = (
        "V2 multipass diffusion drift fix. Step 1: Lab-space low-frequency color correction "
        "(preserves texture detail). Step 2: Boundary-local residual correction. "
        "Step 3: Frequency-aware composite (multiband Laplacian pyramid). "
        "Place between VAE Decode and InpaintStitch2."
    )

    def execute(self, original_crop, generated_crop, mask, color_correction,
                composite_mode, blend_pixels, blend_expansion=0,
                ref_threshold=0.01, ref_erosion=8, lf_sigma=10.0,
                boundary_correction=True, boundary_ring=12, boundary_decay=0.15,
                temporal_smooth=0.0, multiband_levels=4, min_ref_pixels=100):
        device = original_crop.device

        # --- Batch / spatial alignment ---
        B_orig = original_crop.shape[0]
        B_gen = generated_crop.shape[0]
        B = min(B_orig, B_gen)
        H, W, C = original_crop.shape[1], original_crop.shape[2], original_crop.shape[3]

        gen = generated_crop[:B].float()
        orig = original_crop[:B].float()
        if gen.shape[1:3] != orig.shape[1:3]:
            gen = gen.permute(0, 3, 1, 2)
            gen = F.interpolate(gen, size=(H, W), mode="bilinear", align_corners=False)
            gen = gen.permute(0, 2, 3, 1)

        # Prepare mask [B, H, W]
        m = mask.float()
        if m.dim() == 2:
            m = m.unsqueeze(0)
        if m.shape[1:] != (H, W):
            m = F.interpolate(m.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False).squeeze(1)
        if m.shape[0] > B:
            m = m[:B]
        elif m.shape[0] == 1 and B > 1:
            m = m.expand(B, -1, -1)

        info_lines = [f"NV_CropColorFix V2: {B} frames, {H}x{W}, correction={color_correction}, composite={composite_mode}"]
        if B_orig != B_gen:
            info_lines.append(f"  Batch mismatch: original={B_orig}, generated={B_gen}, using {B}")

        # =====================================================================
        # STEP 1: Color correction
        # =====================================================================
        corrected = gen.clone()

        if color_correction != "none":
            # Build eroded reference mask for clean stats (exclude boundary pixels)
            ref_base = (m < ref_threshold).float()  # [B, H, W] — 1.0 = reference pixel
            if ref_erosion > 0:
                ref_region = _erode_mask_2d(ref_base, ref_erosion)
            else:
                ref_region = ref_base

            use_lab = color_correction in ("lab_lowfreq", "lab_full")
            use_lf_decomp = color_correction == "lab_lowfreq"

            # Convert to NCHW for Lab conversion and blur ops
            gen_nchw = gen.permute(0, 3, 1, 2)  # [B, C, H, W]
            orig_nchw = orig.permute(0, 3, 1, 2)

            if use_lab:
                gen_work = _rgb_to_lab(gen_nchw[:, :3])   # [B, 3, H, W] in Lab
                orig_work = _rgb_to_lab(orig_nchw[:, :3])
                work_channels = 3
            else:
                gen_work = gen_nchw[:, :3]
                orig_work = orig_nchw[:, :3]
                work_channels = 3

            if use_lf_decomp:
                # Decompose into low/high frequency
                gen_low = _gaussian_blur_reflect(gen_work, lf_sigma)
                gen_high = gen_work - gen_low
                orig_low = _gaussian_blur_reflect(orig_work, lf_sigma)
                # We'll correct gen_low to match orig_low, then recombine with gen_high
                work_src = gen_low
                work_tgt = orig_low
            else:
                work_src = gen_work
                work_tgt = orig_work
                gen_high = None

            # EMA state for temporal smoothing
            ema_shift = torch.zeros(work_channels, device=device)
            ema_scale = torch.ones(work_channels, device=device)
            ema_initialized = False

            total_shift = torch.zeros(work_channels, device=device)
            total_frames_corrected = 0

            corrected_work = work_src.clone()  # [B, 3, H, W]

            for b in range(B):
                ref_mask_b = ref_region[b] > 0.5  # [H, W] boolean

                n_ref = ref_mask_b.sum().item()
                if n_ref < min_ref_pixels:
                    info_lines.append(f"  Frame {b}: {int(n_ref)} ref pixels < {min_ref_pixels}, skipped")
                    continue

                for c in range(work_channels):
                    src_vals = work_src[b, c][ref_mask_b]
                    tgt_vals = work_tgt[b, c][ref_mask_b]

                    mu_src = src_vals.mean()
                    mu_tgt = tgt_vals.mean()
                    shift = mu_tgt - mu_src

                    if color_correction in ("reinhard", "lab_lowfreq", "lab_full"):
                        std_src = src_vals.std().clamp(min=1e-6)
                        std_tgt = tgt_vals.std().clamp(min=1e-6)
                        scale = std_tgt / std_src
                    else:
                        scale = torch.tensor(1.0, device=device)

                    # Temporal EMA smoothing
                    if temporal_smooth > 0:
                        if ema_initialized:
                            shift = temporal_smooth * ema_shift[c] + (1 - temporal_smooth) * shift
                            scale = temporal_smooth * ema_scale[c] + (1 - temporal_smooth) * scale
                        ema_shift[c] = shift
                        ema_scale[c] = scale

                    # Apply Reinhard correction to entire frame
                    corrected_work[b, c] = (work_src[b, c] - mu_src) * scale + mu_tgt

                    total_shift[c] += shift.abs()

                ema_initialized = True
                total_frames_corrected += 1

            # Recombine LF + HF if using frequency decomposition
            if use_lf_decomp:
                corrected_work = corrected_work + gen_high

            # Convert back from Lab to RGB if needed
            if use_lab:
                corrected_nchw = _lab_to_rgb(corrected_work).clamp(0, 1)
                # Preserve alpha if present
                if C > 3:
                    corrected_nchw = torch.cat([corrected_nchw, gen_nchw[:, 3:]], dim=1)
            else:
                corrected_nchw = corrected_work.clamp(0, 1)
                if C > 3:
                    corrected_nchw = torch.cat([corrected_nchw, gen_nchw[:, 3:]], dim=1)

            corrected = corrected_nchw.permute(0, 2, 3, 1)  # back to [B, H, W, C]

            if total_frames_corrected > 0:
                avg_shift = total_shift / total_frames_corrected
                space_label = "Lab" if use_lab else "RGB"
                lf_label = " (LF only)" if use_lf_decomp else ""
                info_lines.append(
                    f"  Step 1 ({space_label}{lf_label}): avg correction ch0={avg_shift[0].item():.3f}, "
                    f"ch1={avg_shift[1].item():.3f}, ch2={avg_shift[2].item():.3f} "
                    f"(eroded ref by {ref_erosion}px, {total_frames_corrected}/{B} frames)"
                )
            else:
                info_lines.append("  Step 1: WARNING — no frames had enough reference pixels for correction")

        corrected = corrected.clamp(0, 1)

        # =====================================================================
        # STEP 2: Boundary-local residual correction
        # =====================================================================
        if boundary_correction and color_correction != "none":
            # Build boundary ring on the exterior (original) side
            mask_binary = (m > 0.5).float()  # [B, H, W]
            outer_ring = _dilate_mask_2d(mask_binary, boundary_ring) - mask_binary  # [B, H, W]
            outer_ring = outer_ring.clamp(0, 1)

            # Build distance-based decay field from mask edge into interior
            # Use dilated-eroded difference as boundary proximity
            inner_band = mask_binary - _erode_mask_2d(mask_binary, boundary_ring)
            inner_band = inner_band.clamp(0, 1)

            # Measure low-frequency residual in the exterior ring
            ring_residual_sum = torch.zeros(B, 3, device=device)
            ring_pixel_count = torch.zeros(B, 1, device=device)

            for b in range(B):
                ring_mask = outer_ring[b] > 0.5  # [H, W]
                n_ring = ring_mask.sum().item()
                if n_ring < 10:
                    continue
                ring_pixel_count[b] = n_ring
                for c in range(min(C, 3)):
                    orig_ring = orig[b, :, :, c][ring_mask]
                    corr_ring = corrected[b, :, :, c][ring_mask]
                    ring_residual_sum[b, c] = (orig_ring - corr_ring).mean()

            # Propagate correction into interior with distance decay
            # Use the inner_band as a spatial weight (1 at boundary, fading inward)
            if ring_pixel_count.sum() > 0:
                # Smooth the inner band to create gradient falloff
                inner_weight = _gaussian_blur_reflect(
                    inner_band.unsqueeze(1), boundary_ring * boundary_decay * 10
                ).squeeze(1).clamp(0, 1)  # [B, H, W]

                # Apply residual correction weighted by inner_weight
                for b in range(B):
                    if ring_pixel_count[b] < 10:
                        continue
                    for c in range(min(C, 3)):
                        correction = ring_residual_sum[b, c] * inner_weight[b]  # [H, W]
                        corrected[b, :, :, c] = corrected[b, :, :, c] + correction

                corrected = corrected.clamp(0, 1)
                avg_boundary_residual = ring_residual_sum.abs().mean().item()
                info_lines.append(
                    f"  Step 2 (boundary): ring={boundary_ring}px, avg residual={avg_boundary_residual*255:.2f}/255"
                )
            else:
                info_lines.append("  Step 2 (boundary): no exterior ring pixels, skipped")

        # =====================================================================
        # STEP 3: Composite — replace non-masked pixels with originals
        # =====================================================================
        # Build the composite mask
        comp_mask = m.clone()

        # Dilate mask outward to push blend zone into original pixels
        if blend_expansion > 0:
            comp_mask = _dilate_mask_2d(comp_mask, blend_expansion)

        if composite_mode == "multiband":
            # Laplacian pyramid blend — wide LF transition, tight HF transition
            # Convert to NCHW for multiband_blend
            corr_nchw = corrected.permute(0, 3, 1, 2)  # [B, C, H, W]
            orig_nchw = orig.permute(0, 3, 1, 2)        # [B, C, H, W]
            mask_nchw = comp_mask.unsqueeze(1)            # [B, 1, H, W]

            # Soften mask edges before pyramid (avoid hard transitions in HF bands)
            if blend_pixels > 0:
                mask_nchw = _gaussian_blur_reflect(mask_nchw, blend_pixels / 3.0)

            blended_nchw = multiband_blend(corr_nchw, orig_nchw, mask_nchw, num_levels=multiband_levels)
            output = blended_nchw.clamp(0, 1).permute(0, 2, 3, 1)  # [B, H, W, C]
            info_lines.append(f"  Step 3 (multiband): {multiband_levels} levels, expansion={blend_expansion}")

        elif composite_mode == "gaussian":
            # V1 Gaussian alpha blend with reflect padding
            feathered = comp_mask
            if blend_pixels > 0:
                feathered = _gaussian_blur_reflect(
                    feathered.unsqueeze(1), blend_pixels / 3.0
                ).squeeze(1).clamp(0, 1)

            alpha = feathered.unsqueeze(-1)  # [B, H, W, 1]
            output = orig * (1.0 - alpha) + corrected * alpha
            output = output.clamp(0, 1)
            info_lines.append(f"  Step 3 (gaussian): sigma={blend_pixels / 3.0:.1f}, expansion={blend_expansion}")

        elif composite_mode == "hard":
            hard_mask = (comp_mask > 0.5).float().unsqueeze(-1)
            output = orig * (1.0 - hard_mask) + corrected * hard_mask
            output = output.clamp(0, 1)
            info_lines.append(f"  Step 3 (hard): expansion={blend_expansion}")

        # --- Final residual measurement ---
        ref_zone = m < ref_threshold
        if ref_zone.sum() > 0:
            residual = (output[..., :3] - orig[..., :3]).abs()
            ref_residual = residual[ref_zone.unsqueeze(-1).expand_as(residual)].mean().item()
            info_lines.append(f"  Final residual in ref zone: {ref_residual * 255:.2f}/255")
        else:
            info_lines.append("  No reference zone pixels to measure residual")

        info_text = "\n".join(info_lines)
        return (output, info_text,)


NODE_CLASS_MAPPINGS = {
    "NV_CropColorFix": NV_CropColorFix,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_CropColorFix": "NV Crop Color Fix",
}
