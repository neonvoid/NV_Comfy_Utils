"""
NV Low-Freq Recompose — principled fix for the "contaminated ref zone" problem.

Architectural root cause (from 2026-04-24 multi-AI debate):
  CropColorFix assumes "mask<threshold pixels are pristine originals." That
  assumption breaks once we dilate the VACE mask for face-boundary stability
  (vace_input_grow_px > 0 makes the entire crop VACE-modified). Reference
  statistics become contaminated → color correction undershoots → visible
  color mismatch between AI face and original plate in the final stitched
  output. Especially painful on bright/neutral backgrounds where the tonal
  delta is not masked by dark-BG luminance.

Solution — provenance-aware recomposition:
  Skip color "correction" entirely. Instead, reconstruct the output as:

      output = low_freq(original_crop) + high_freq(generated_crop)

  inside the tight mask. The low-frequency field (illumination, local shadow,
  base skin tone) comes from the ACTUAL ORIGINAL PLATE — there's no inference,
  no ref zone, no contamination. The high-frequency detail (pores, beard,
  eye sharpness, identity) comes from VACE's generated crop.

  For face refinement specifically this is architecturally correct:
    - We WANT VACE to refine detail (pores, sharpness, identity)
    - We DON'T want VACE to change the shot's tone/illumination
    - Therefore: AI for detail, original for plate-consistent tone

  This eliminates CropColorFix's need to "infer the target color" from
  contaminated statistics.

Replaces NV_CropColorFix in the pipeline for this use case. Does not depend
on ref_zone / ref_erosion / boundary_ring / ref_threshold. Works identically
regardless of vace_input_grow_px or mask dilation (there's no ref zone
assumption to break).

References:
  Multi-AI debate 2026-04-24 — Codex #1 pick, Gemini #2 pick, both converged
  on the paired orig-as-truth formulation.

  Burt & Adelson, "The Laplacian Pyramid as a Compact Image Code" (1983) —
  foundational frequency decomposition; we use a simpler Gaussian-subtract
  pair for speed and locality.
"""

import torch
import torch.nn.functional as F

from .boundary_color_match import _rgb_to_lab, _lab_to_rgb


LOG_PREFIX = "[NV_LowFreqRecompose]"


# =============================================================================
# Gaussian blur (separable, for low-freq extraction)
# =============================================================================

def _gaussian_kernel_1d(sigma, truncate=4.0):
    """Build a 1D Gaussian kernel with radius determined by sigma."""
    if sigma <= 0:
        return None
    radius = max(1, int(round(sigma * truncate)))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    k = k / k.sum()
    return k, radius


def _gaussian_blur_2d(x, sigma):
    """Separable Gaussian blur on [B, C, H, W]. Reflection padding to avoid
    edge darkening at image boundaries."""
    if sigma <= 0:
        return x
    k, radius = _gaussian_kernel_1d(sigma)
    k = k.to(device=x.device, dtype=x.dtype)
    C = x.shape[1]
    # Horizontal pass
    kernel_h = k.view(1, 1, 1, -1).repeat(C, 1, 1, 1)
    x_p = F.pad(x, (radius, radius, 0, 0), mode='reflect')
    x = F.conv2d(x_p, kernel_h, groups=C)
    # Vertical pass
    kernel_v = k.view(1, 1, -1, 1).repeat(C, 1, 1, 1)
    x_p = F.pad(x, (0, 0, radius, radius), mode='reflect')
    x = F.conv2d(x_p, kernel_v, groups=C)
    return x


# =============================================================================
# Bidirectional temporal EMA (zero-phase smoothing across frames)
# =============================================================================

def _temporal_ema_bidir(x, alpha):
    """Forward + backward EMA pass across batch dim 0 (time).

    alpha in [0, 1]:
      0.0 = no smoothing (passthrough)
      0.3 = moderate (matches CropColorFix default)
      0.5 = heavy smoothing
      0.8+ = extreme, content becomes temporally blurry

    Bidirectional pass produces zero phase lag (no trailing delay from a purely
    causal filter). Applied to the final recomposed output to dampen per-frame
    texture variance from both the plate's natural temporal noise (grain,
    rolling shutter) and VACE's intrinsic per-frame generation variance.
    """
    if alpha <= 0 or x.shape[0] < 2:
        return x
    one_minus = 1.0 - alpha

    fwd = x.clone()
    for t in range(1, x.shape[0]):
        fwd[t] = alpha * fwd[t - 1] + one_minus * x[t]

    bwd = fwd.clone()
    for t in range(x.shape[0] - 2, -1, -1):
        bwd[t] = alpha * bwd[t + 1] + one_minus * fwd[t]

    return bwd


# =============================================================================
# Mask feather (for soft blend at tight-mask boundary)
# =============================================================================

def _feather_mask(mask, falloff_px):
    """Feather a binary mask by gaussian blur with sigma ~= falloff_px/2.
    Input/output: [B, 1, H, W] in [0, 1]."""
    if falloff_px <= 0:
        return mask
    sigma = max(0.5, falloff_px / 2.0)
    return _gaussian_blur_2d(mask, sigma).clamp(0.0, 1.0)


# =============================================================================
# Core recomposition
# =============================================================================

def _recompose_lf(orig_bchw, gen_bchw, sigma_px, recompose_strength, color_space):
    """Apply low-frequency recomposition: out = LP(orig) + HP(gen).

    Args:
        orig_bchw, gen_bchw: [B, 3, H, W] in [0, 1] RGB
        sigma_px: gaussian sigma for low-freq extraction
        recompose_strength: 0..1 lerp between generated (0) and recomposed (1)
        color_space: 'lab' (perceptual, recommended) or 'rgb' (faster, simpler)

    Returns:
        [B, 3, H, W] in [0, 1] RGB, fully recomposed (no mask blending yet).
    """
    if color_space == 'lab':
        orig = _rgb_to_lab(orig_bchw)
        gen = _rgb_to_lab(gen_bchw)
    else:
        orig = orig_bchw
        gen = gen_bchw

    lp_orig = _gaussian_blur_2d(orig, sigma_px)
    lp_gen = _gaussian_blur_2d(gen, sigma_px)
    hp_gen = gen - lp_gen

    # The core algorithm: original's low-freq + generated's high-freq
    recomposed = lp_orig + hp_gen

    # Optional strength lerp (1.0 = full recompose, 0.0 = no change)
    if recompose_strength < 1.0:
        recomposed = gen + (recomposed - gen) * recompose_strength

    if color_space == 'lab':
        recomposed = _lab_to_rgb(recomposed)

    return recomposed.clamp(0.0, 1.0)


# =============================================================================
# Node
# =============================================================================

class NV_LowFreqRecompose:
    """Recompose the generated face as (original_low_freq + generated_high_freq).

    Use this INSTEAD of NV_CropColorFix when vace_input_grow_px > 0 makes the
    CropColorFix ref zone unreliable. Also works as a general-purpose alternative
    to CropColorFix — it's simpler, has fewer failure modes, and doesn't rely
    on any "pristine pixel" assumption.

    Pipeline placement:
      VAE Decode → NV_LowFreqRecompose → NV_TextureHarmonize → NV_InpaintStitch2

    Trade-off vs CropColorFix:
      - CropColorFix can preserve VACE's intended illumination changes (if you
        wanted them). Recompose does NOT — it always restores original tone.
      - For face refinement (our case), we want VACE for detail only, plate
        for tone. Recompose encodes that preference architecturally.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_crop": ("IMAGE", {
                    "tooltip": "Pristine cropped image from BEFORE VACE (plate reference). "
                               "Provides the low-frequency color/tone field."
                }),
                "generated_crop": ("IMAGE", {
                    "tooltip": "Post-VAE-decode AI output. Provides the high-frequency detail "
                               "(pores, beard, eye sharpness, identity)."
                }),
                "mask": ("MASK", {
                    "tooltip": "Tight subject mask. Recomposition is applied only inside this "
                               "mask with a soft edge falloff. Outside the mask, original pixels "
                               "pass through unchanged."
                }),
                "lf_sigma_px": ("FLOAT", {
                    "default": 32.0, "min": 1.0, "max": 256.0, "step": 1.0,
                    "tooltip": "Gaussian sigma for the low-frequency split (pixels). Higher = "
                               "more frequencies are treated as 'tone' and taken from original; "
                               "lower = more frequencies are treated as 'detail' and kept from "
                               "generated. Default 32 is a good starting point for 512x512 faces; "
                               "raise to 48-64 if VACE drifted coarser structural lighting; lower "
                               "to 16-24 if VACE's finer color shifts must also be preserved."
                }),
            },
            "optional": {
                "recompose_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "How much recomposition to apply. 1.0 = full (LP(orig) + HP(gen)); "
                               "0.0 = no change (passthrough of generated). Use <1.0 to partially "
                               "preserve VACE's illumination changes."
                }),
                "edge_falloff_px": ("INT", {
                    "default": 8, "min": 0, "max": 64, "step": 1,
                    "tooltip": "Soft-feather the mask edge by this many pixels before blending. "
                               "Avoids a hard transition from recomposed (inside mask) to original "
                               "(outside). Default 8 matches the typical processed-mask feather."
                }),
                "color_space": (["lab", "rgb"], {
                    "default": "lab",
                    "tooltip": "lab: perceptually uniform recomposition (recommended). "
                               "rgb: faster but may shift hue at large color deltas."
                }),
                "preserve_luminance": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If True, only replace low-freq chroma (a, b channels in Lab), "
                               "keeping generated's L (luminance). Use if VACE's lighting is "
                               "correct but skin tone/hue is off. lab color_space only."
                }),
                "temporal_smoothing": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 0.9, "step": 0.05,
                    "tooltip": "Bidirectional EMA across frames applied to the recomposed output. "
                               "Dampens per-frame temporal variance from both the plate's natural "
                               "noise (grain, rolling shutter) and VACE's intrinsic per-frame "
                               "generation variance. 0.0 = off (may flicker), 0.3 = moderate "
                               "(default, matches CropColorFix), 0.5 = heavy (clean but slightly "
                               "softer detail over time). Zero phase lag — no trailing delay."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("recomposed_crop",)
    FUNCTION = "execute"
    CATEGORY = "NV_Utils/Color"
    DESCRIPTION = (
        "Principled alternative to NV_CropColorFix: reconstructs the face as "
        "original's low-frequency (color/tone) + generated's high-frequency (detail). "
        "Eliminates the contaminated-reference-zone failure mode when vace_input_grow_px > 0."
    )

    def execute(self, original_crop, generated_crop, mask,
                lf_sigma_px,
                recompose_strength=1.0, edge_falloff_px=8,
                color_space="lab", preserve_luminance=False,
                temporal_smoothing=0.3):
        info_lines = []

        # Shape validation
        if original_crop.dim() != 4 or generated_crop.dim() != 4:
            raise ValueError(
                f"original_crop and generated_crop must be [B, H, W, 3]; "
                f"got {list(original_crop.shape)} / {list(generated_crop.shape)}"
            )
        if original_crop.shape != generated_crop.shape:
            raise ValueError(
                f"original_crop ({list(original_crop.shape)}) and generated_crop "
                f"({list(generated_crop.shape)}) must match shape."
            )
        B, H, W, C = original_crop.shape
        if C not in (3, 4):
            raise ValueError(f"Expected 3 or 4 channels, got {C}")
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        if mask.shape[0] != B or mask.shape[1] != H or mask.shape[2] != W:
            raise ValueError(
                f"mask shape {list(mask.shape)} must be [B={B}, H={H}, W={W}]"
            )

        device = original_crop.device
        dtype = original_crop.dtype

        # Convert to BCHW for convolutions (keep only RGB channels)
        orig_bchw = original_crop[..., :3].permute(0, 3, 1, 2).contiguous().float()
        gen_bchw = generated_crop[..., :3].permute(0, 3, 1, 2).contiguous().float()
        mask_bchw = mask.unsqueeze(1).float()  # [B, 1, H, W]

        # Feather the mask for soft edge blending
        soft_mask = _feather_mask(mask_bchw, edge_falloff_px)

        # Recompose (always in Lab if color_space=='lab', regardless of luma-preserve)
        if preserve_luminance and color_space == "lab":
            # Recompose chroma only: L = gen_L, a = LP(orig_a) + HP(gen_a), same for b
            orig_lab = _rgb_to_lab(orig_bchw)
            gen_lab = _rgb_to_lab(gen_bchw)
            lp_orig = _gaussian_blur_2d(orig_lab, lf_sigma_px)
            lp_gen = _gaussian_blur_2d(gen_lab, lf_sigma_px)
            hp_gen = gen_lab - lp_gen
            recomposed_lab = gen_lab.clone()
            # Only replace low-freq a, b (chroma)
            recomposed_lab[:, 1:3] = lp_orig[:, 1:3] + hp_gen[:, 1:3]
            # Strength lerp
            if recompose_strength < 1.0:
                recomposed_lab[:, 1:3] = (
                    gen_lab[:, 1:3] + (recomposed_lab[:, 1:3] - gen_lab[:, 1:3]) * recompose_strength
                )
            recomposed = _lab_to_rgb(recomposed_lab).clamp(0.0, 1.0)
            info_lines.append(
                f"luma-preserved (L=gen, a/b recomposed at sigma={lf_sigma_px:.0f}px)"
            )
        else:
            recomposed = _recompose_lf(
                orig_bchw, gen_bchw,
                sigma_px=lf_sigma_px,
                recompose_strength=recompose_strength,
                color_space=color_space,
            )
            info_lines.append(
                f"full recomposition ({color_space}, sigma={lf_sigma_px:.0f}px, "
                f"strength={recompose_strength:.2f})"
            )

        # Apply bidirectional temporal EMA to dampen per-frame variance before
        # blending with original. Only smooths the RECOMPOSED content — the
        # original pixels outside the mask already have natural plate temporal
        # behavior and should pass through unchanged.
        if temporal_smoothing > 0 and recomposed.shape[0] >= 2:
            recomposed = _temporal_ema_bidir(recomposed, float(temporal_smoothing))
            info_lines.append(f"temporal EMA: alpha={temporal_smoothing:.2f} bidir")

        # Blend recomposed (inside mask) with original (outside mask)
        # output = soft_mask * recomposed + (1 - soft_mask) * original_crop
        output_bchw = soft_mask * recomposed + (1.0 - soft_mask) * orig_bchw

        # Measure the shift magnitude for diagnostic output
        with torch.no_grad():
            mask_region = soft_mask > 0.5
            if mask_region.any():
                delta = (output_bchw - gen_bchw) * mask_region.float()
                mean_shift = delta.abs().mean().item() * 255.0
                max_shift = delta.abs().max().item() * 255.0
                info_lines.append(
                    f"In-mask shift from generated: mean={mean_shift:.2f}/255 max={max_shift:.2f}/255 "
                    f"(how much color was pulled back toward original)"
                )

        # BCHW → BHWC
        output = output_bchw.permute(0, 2, 3, 1).contiguous()
        output = output.to(device=device, dtype=dtype)

        # Pass through alpha if the input had 4 channels
        if C == 4:
            output = torch.cat([output, original_crop[..., 3:4]], dim=-1)

        info = f"{LOG_PREFIX} " + " | ".join(info_lines)
        print(info)
        return (output,)


# ── Registration ────────────────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    "NV_LowFreqRecompose": NV_LowFreqRecompose,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_LowFreqRecompose": "NV Low-Freq Recompose (color fix alt)",
}
