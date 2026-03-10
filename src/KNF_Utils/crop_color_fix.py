"""
NV Crop Color Fix — Eliminate diffusion color drift before stitching.

When KSampler runs at denoise < 1.0, it modifies the ENTIRE crop — not just
the masked region. Non-masked pixels come back color-shifted (~7/255 typical).
This creates a visible seam when stitched onto the original frame.

This node fixes the problem in two steps:
  1. Color-correct the generated crop to match the original crop's statistics
     (per-channel Reinhard transfer using non-masked pixels as reference)
  2. Composite: replace non-masked pixels with originals, feathered blend at edge

Place between VAE Decode and InpaintStitch2:
  InpaintCrop2 → VAE Encode → KSampler → VAE Decode
      → [NV_CropColorFix] → InpaintStitch2
"""

import torch
import torch.nn.functional as F


class NV_CropColorFix:
    """Color-correct and composite a generated crop before stitching."""

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
                "color_correction": (["reinhard", "mean_only", "none"], {
                    "default": "reinhard",
                    "tooltip": "reinhard: match mean + std per channel (best for systematic shift). "
                               "mean_only: match mean only (simpler, less overcorrection risk). "
                               "none: skip color correction, only do pixel composite."
                }),
                "blend_pixels": ("INT", {
                    "default": 8, "min": 0, "max": 64, "step": 1,
                    "tooltip": "Gaussian blur sigma for the feather gradient. "
                               "Controls how soft the transition is (higher = smoother gradient)."
                }),
                "blend_expansion": ("INT", {
                    "default": 0, "min": 0, "max": 64, "step": 1,
                    "tooltip": "Dilate the mask outward by N pixels BEFORE blurring. "
                               "Pushes the blend zone further into original pixels. "
                               "0 = no expansion (blur only). 16-32 = wider gradient reaching into original."
                }),
            },
            "optional": {
                "ref_threshold": ("FLOAT", {
                    "default": 0.01, "min": 0.0, "max": 0.5, "step": 0.01,
                    "tooltip": "Pixels with mask value below this are used as color reference. "
                               "Default 0.01 = only truly untouched pixels."
                }),
                "temporal_smooth": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 0.95, "step": 0.05,
                    "tooltip": "EMA smoothing for correction values across frames (0=off, 0.3=moderate, 0.8=heavy). "
                               "Prevents frame-to-frame jitter in the color correction."
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
        "Fix diffusion color drift in cropped inpainting output. "
        "Color-corrects the generated crop to match the original, then composites "
        "non-masked pixels back from the original. Place between VAE Decode and InpaintStitch2."
    )

    def execute(self, original_crop, generated_crop, mask, color_correction,
                blend_pixels, blend_expansion=0, ref_threshold=0.01, temporal_smooth=0.0, min_ref_pixels=100):
        device = original_crop.device

        # Use the minimum batch size — generated_crop may have fewer frames
        # (e.g., VAE decode after trim_latent removes reference frames)
        B_orig = original_crop.shape[0]
        B_gen = generated_crop.shape[0]
        B = min(B_orig, B_gen)
        H, W, C = original_crop.shape[1], original_crop.shape[2], original_crop.shape[3]

        # Truncate to matching batch size and ensure spatial match
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
        # Truncate or expand mask to match B
        if m.shape[0] > B:
            m = m[:B]
        elif m.shape[0] == 1 and B > 1:
            m = m.expand(B, -1, -1)

        # --- Step 1: Color correction ---
        corrected = gen.clone()
        info_lines = [f"NV_CropColorFix: {B} frames, {H}x{W}, method={color_correction}"]
        if B_orig != B_gen:
            info_lines.append(f"  Batch mismatch: original_crop={B_orig}, generated_crop={B_gen}, using {B}")

        if color_correction != "none":
            # Track EMA state for temporal smoothing
            ema_shift = torch.zeros(C, device=device)
            ema_scale = torch.ones(C, device=device)
            ema_initialized = False

            total_shift = torch.zeros(C, device=device)
            total_frames_corrected = 0

            for b in range(B):
                ref_mask = m[b] < ref_threshold  # [H, W] — True = reference pixel

                n_ref = ref_mask.sum().item()
                if n_ref < min_ref_pixels:
                    info_lines.append(f"  Frame {b}: {int(n_ref)} ref pixels < {min_ref_pixels}, skipped")
                    continue

                frame_orig = orig[b]  # [H, W, C]
                frame_gen = corrected[b]

                for c in range(C):
                    orig_vals = frame_orig[..., c][ref_mask]
                    gen_vals = frame_gen[..., c][ref_mask]

                    mu_orig = orig_vals.mean()
                    mu_gen = gen_vals.mean()
                    shift = mu_orig - mu_gen

                    if color_correction == "reinhard":
                        std_orig = orig_vals.std().clamp(min=1e-6)
                        std_gen = gen_vals.std().clamp(min=1e-6)
                        scale = std_orig / std_gen
                    else:
                        scale = torch.tensor(1.0, device=device)

                    # Temporal EMA smoothing
                    if temporal_smooth > 0:
                        if ema_initialized:
                            shift = temporal_smooth * ema_shift[c] + (1 - temporal_smooth) * shift
                            scale = temporal_smooth * ema_scale[c] + (1 - temporal_smooth) * scale
                        ema_shift[c] = shift
                        ema_scale[c] = scale

                    # Apply correction to entire frame (including generated region)
                    corrected[b, ..., c] = (frame_gen[..., c] - mu_gen) * scale + mu_orig

                    total_shift[c] += (mu_orig - mu_gen).abs()

                ema_initialized = True
                total_frames_corrected += 1

            if total_frames_corrected > 0:
                avg_shift = total_shift / total_frames_corrected
                info_lines.append(
                    f"  Avg correction: R={avg_shift[0].item()*255:.1f}/255, "
                    f"G={avg_shift[1].item()*255:.1f}/255, B={avg_shift[2].item()*255:.1f}/255"
                )
            else:
                info_lines.append("  WARNING: no frames had enough reference pixels for correction")

        corrected = corrected.clamp(0, 1)

        # --- Step 2: Pixel composite — replace non-masked pixels with originals ---
        # Build feathered mask for compositing
        feathered = m  # default: use mask as-is

        # Dilate mask outward to push blend zone into original pixels
        if blend_expansion > 0:
            dk = blend_expansion * 2 + 1
            dilate_kernel = torch.ones(1, 1, dk, dk, device=device)
            expanded = F.conv2d(m.unsqueeze(1), dilate_kernel, padding=blend_expansion)
            feathered = (expanded > 0).float().squeeze(1)  # [B, H, W] — binary dilated
        else:
            feathered = m

        if blend_pixels > 0:
            # Gaussian blur for soft transition
            k = blend_pixels * 2 + 1
            sigma = blend_pixels / 3.0
            x = torch.arange(k, dtype=torch.float32, device=device) - (k - 1) / 2
            gauss_1d = torch.exp(-0.5 * (x / max(sigma, 1e-6)) ** 2)
            gauss_1d = gauss_1d / gauss_1d.sum()
            feathered = feathered.unsqueeze(1)  # [B, 1, H, W]
            feathered = F.conv2d(feathered, gauss_1d.view(1, 1, -1, 1), padding=(k // 2, 0))
            feathered = F.conv2d(feathered, gauss_1d.view(1, 1, 1, -1), padding=(0, k // 2))
            feathered = feathered.squeeze(1).clamp(0, 1)  # [B, H, W]
        else:
            feathered = m

        # Composite: feathered_mask=0 → original, feathered_mask=1 → corrected (generated)
        alpha = feathered.unsqueeze(-1)  # [B, H, W, 1]
        output = orig * (1.0 - alpha) + corrected * alpha
        output = output.clamp(0, 1)

        # Stats for the output
        # Measure residual diff in reference zone after correction
        ref_zone = m < ref_threshold
        if ref_zone.sum() > 0:
            residual = (output[..., :3] - orig[..., :3]).abs()
            ref_residual = residual[ref_zone.unsqueeze(-1).expand_as(residual)].mean().item()
            info_lines.append(f"  Residual in ref zone after fix: {ref_residual*255:.2f}/255")
        else:
            info_lines.append("  No reference zone pixels to measure residual")

        info_lines.append(f"  Blend pixels: {blend_pixels}, Expansion: {blend_expansion}, Temporal smooth: {temporal_smooth}")
        info_text = "\n".join(info_lines)

        return (output, info_text,)


NODE_CLASS_MAPPINGS = {
    "NV_CropColorFix": NV_CropColorFix,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_CropColorFix": "NV Crop Color Fix",
}
