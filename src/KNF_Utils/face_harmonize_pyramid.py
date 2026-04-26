"""
NV Face Harmonize Pyramid — unified post-decode spatial harmonizer.

Replaces NV_CropColorFix + NV_LowFreqRecompose + NV_TextureHarmonize for
face-refinement workflows.

Architecture: masked-normalized-convolution (MNC) Laplacian pyramid in
**linear RGB**, band-grouped into base/mid/top regimes:
    - Base bands (coarsest):      tone replacement from plate (LFR approach)
    - Mid bands:                  preserve gen structure + boundary ring lock
    - Top bands (finest):         texture/grain stat matching (TH approach)

Design principles (from 2026-04-24 multi-AI consensus + 2026-04-24 review):
  1. MNC is the default blur primitive — no unmasked Gaussians near mask edges
  2. Laplacian pyramid decomposition, not single Gaussian
  3. Linear RGB ONLY — Lab was removed after review. LP+HP sums must be in
     additive photon space; Lab is perceptually non-linear and produces
     non-additive band swaps (hue-twist artifacts).
  4. Spatial logic only — no temporal EMA. Pair with NV_FrameTemporalStabilize.
  5. Single `match_strength` driver + targeted overrides
  6. Content-aware mask-size gating with SMOOTH transitions (no hard threshold
     pop-in as mask_occupancy drifts across frames).

Key corrections vs. the initial MVP (2026-04-24 review response):
  - Lab color_space and preserve_luminance REMOVED (principle-3 drift)
  - Boundary ring is computed ONCE at the finest scale and carried down through
    the mask pyramid — keeps world-space seam width consistent across levels
  - mask_occupancy threshold is now a smooth sigmoid-style lerp in [0.6, 0.9]
  - _match_band_stats has a flat-band fallback (switches to mean-only when
    gen_spread is near zero) so near-constant bands can't hide failures
  - Batch chunking in execute() prevents VRAM blowup for long T

2026-04-25 update — base-band tone source decoupled from boundary-jitter:
  When the silhouette mask jitters frame-to-frame (head rotation, SAM3 flicker,
  rotation-misaligned consensus output), the MNC formula blur(orig*m)/blur(m)
  reads a different "global face color" each frame because the boundary
  transiently sweeps across plate pixels OUTSIDE the intended face region —
  chin shadow, neck shadow, hair, collar. The base-band tone target then drifts
  toward those values, producing dark blotches in the harmonized output.

  Fix: a separate, more-aggressively-eroded CORE mask is used as the support
  weight for the BASE-BAND plate pyramid only. Mid + top bands keep the
  standard mask (they need boundary signal — mid for ring lock, top for
  texture stats). Param: core_erode_px (default 8 px on top of mask_erode_px).

Reused primitives:
  - low_freq_recompose._erode_mask_2d (min-pool erosion)
  - low_freq_recompose._feather_mask (edge feather)
  - multiband_blend_stitch._make_gaussian_kernel_2d (2D Gaussian kernel)
"""

import math

import torch
import torch.nn.functional as F

from .low_freq_recompose import _erode_mask_2d, _feather_mask
from .multiband_blend_stitch import _make_gaussian_kernel_2d


LOG_PREFIX = "[NV_FaceHarmonizePyramid]"

# Smooth occupancy-to-mode transition: below OCC_REPLACE use structural swap,
# above OCC_STATISTICAL use CCF-style statistical match, between lerp smoothly.
OCC_REPLACE = 0.60
OCC_STATISTICAL = 0.90

# Flat-band detection: spreads below this fraction of the mean |value| are
# treated as near-constant bands — scale matching falls back to mean-only.
FLAT_BAND_REL_TOL = 1e-3


# =============================================================================
# sRGB ↔ linear RGB — pyramid math lives in linear light space
# =============================================================================

def _srgb_to_linear(x):
    """sRGB [0,1] → linear RGB. Standard gamma curve."""
    return torch.where(
        x <= 0.04045,
        x / 12.92,
        ((x.clamp_min(0.0) + 0.055) / 1.055) ** 2.4,
    )


def _linear_to_srgb(x):
    """Linear RGB → sRGB [0,1]."""
    x = x.clamp_min(0.0)
    return torch.where(
        x <= 0.0031308,
        12.92 * x,
        1.055 * x.clamp_min(1e-8).pow(1.0 / 2.4) - 0.055,
    ).clamp(0.0, 1.0)


# =============================================================================
# Masked normalized convolution for pyramid construction
# =============================================================================

def _mnc_blur_and_downsample(x_bchw, mask_bchw, kernel, support_tau):
    """MNC-blur then ×2 downsample both x and mask.

    Returns (x_down, mask_down) at half resolution. The blurred mask itself
    becomes the support weight for the next pyramid level.
    """
    pad = kernel.shape[-1] // 2
    C = x_bchw.shape[1]
    k_x = kernel.expand(C, -1, -1, -1)
    k_m = kernel  # [1, 1, K, K], single-channel

    xm = x_bchw * mask_bchw
    xm_pad = F.pad(xm, (pad, pad, pad, pad), mode="reflect")
    blur_xm = F.conv2d(xm_pad, k_x, groups=C)

    m_pad = F.pad(mask_bchw, (pad, pad, pad, pad), mode="reflect")
    blur_m = F.conv2d(m_pad, k_m)

    x_pad = F.pad(x_bchw, (pad, pad, pad, pad), mode="reflect")
    blur_x = F.conv2d(x_pad, k_x, groups=C)

    safe_m = blur_m.clamp_min(support_tau)
    masked_ratio = blur_xm / safe_m
    confidence = (blur_m / support_tau).clamp(0.0, 1.0)
    x_smooth = confidence * masked_ratio + (1.0 - confidence) * blur_x

    x_down = x_smooth[:, :, ::2, ::2]
    m_down = blur_m[:, :, ::2, ::2]
    return x_down, m_down


def _downsample_mask_only(mask_bchw, kernel):
    """Blur + ×2 decimate a single-channel mask (no MNC — just a matched
    downsample used to carry auxiliary mask-shaped tensors like the boundary
    ring through the pyramid with the same response as the companion mask)."""
    pad = kernel.shape[-1] // 2
    k_m = kernel
    m_pad = F.pad(mask_bchw, (pad, pad, pad, pad), mode="reflect")
    blur_m = F.conv2d(m_pad, k_m)
    return blur_m[:, :, ::2, ::2]


def _gaussian_upsample(x_bchw, kernel, target_h, target_w):
    """Bilinear ×2 upsample then Gaussian smooth. Unmasked — only used to
    reconstruct the next-finer Gaussian level from the coarser one."""
    up = F.interpolate(x_bchw, size=(target_h, target_w), mode="bilinear", align_corners=False)
    C = up.shape[1]
    k = kernel.expand(C, -1, -1, -1)
    pad = kernel.shape[-1] // 2
    up_pad = F.pad(up, (pad, pad, pad, pad), mode="reflect")
    return F.conv2d(up_pad, k, groups=C)


def _build_masked_laplacian_pyramid(x_bchw, mask_bchw, num_levels, kernel, support_tau):
    """Build a masked Laplacian pyramid + companion mask pyramid."""
    gauss_x = [x_bchw]
    gauss_m = [mask_bchw]
    x_cur, m_cur = x_bchw, mask_bchw
    for _ in range(num_levels - 1):
        x_cur, m_cur = _mnc_blur_and_downsample(x_cur, m_cur, kernel, support_tau)
        gauss_x.append(x_cur)
        gauss_m.append(m_cur)

    lap = []
    for i in range(num_levels - 1):
        h, w = gauss_x[i].shape[2], gauss_x[i].shape[3]
        expanded = _gaussian_upsample(gauss_x[i + 1], kernel, h, w)
        lap.append(gauss_x[i] - expanded)
    lap.append(gauss_x[-1])
    return lap, gauss_m


def _collapse_pyramid(pyramid, kernel):
    """Collapse Laplacian pyramid to full-res image."""
    img = pyramid[-1]
    for i in range(len(pyramid) - 2, -1, -1):
        h, w = pyramid[i].shape[2], pyramid[i].shape[3]
        img = _gaussian_upsample(img, kernel, h, w) + pyramid[i]
    return img


# =============================================================================
# Boundary ring (computed at finest scale, downsampled through pyramid)
# =============================================================================

def _compute_boundary_ring(mask_bchw, width_px):
    """Ring mask: 1 at mask boundary, 0 inside or far outside. Soft (0..1).

    Computed at the FINEST scale. Downstream, carry through the pyramid via
    _downsample_mask_only so the ring has consistent world-space width.
    """
    if width_px <= 0:
        return torch.zeros_like(mask_bchw)
    r = max(1, int(round(width_px)))
    dilated = F.max_pool2d(mask_bchw, kernel_size=2 * r + 1, stride=1, padding=r)
    eroded = -F.max_pool2d(-mask_bchw, kernel_size=2 * r + 1, stride=1, padding=r)
    return (dilated - eroded).clamp(0.0, 1.0)


def _build_ring_pyramid(ring_bchw, num_levels, kernel):
    """Carry the finest-scale ring through the pyramid with matched blur+decimate.

    This produces a per-level ring that covers the same world-space region at
    every pyramid scale — unlike re-computing a ring at each level's local
    resolution, which gives inconsistent seam widths across bands.
    """
    rings = [ring_bchw]
    cur = ring_bchw
    for _ in range(num_levels - 1):
        cur = _downsample_mask_only(cur, kernel)
        rings.append(cur)
    return rings


# =============================================================================
# Band statistic matching
# =============================================================================

def _masked_mean(t, m):
    """Per-channel masked mean. t: [B, C, H, W]. m: [B, 1, H, W]. Returns [B, C, 1, 1]."""
    count = m.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
    return (t * m).sum(dim=(2, 3), keepdim=True) / count


def _masked_std(t, m, mean=None):
    if mean is None:
        mean = _masked_mean(t, m)
    count = m.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
    var = (((t - mean) ** 2) * m).sum(dim=(2, 3), keepdim=True) / count
    return var.clamp_min(1e-12).sqrt()


def _masked_mean_abs_dev(t, m, mean=None):
    """Per-channel masked mean-absolute-deviation (fast MAD proxy).

    True MAD needs a median (expensive on GPU). Mean-AD is a robust spread
    estimator scaled by 1.2533 to be std-equivalent for Gaussian data.
    """
    if mean is None:
        mean = _masked_mean(t, m)
    count = m.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
    mad = ((t - mean).abs() * m).sum(dim=(2, 3), keepdim=True) / count
    return mad.clamp_min(1e-12) * 1.2533


def _match_band_stats(gen_band, orig_band, mask_band, use_robust_spread,
                      scale_clamp=(0.7, 1.43)):
    """Re-scale gen_band so its in-mask (mean, spread) matches orig_band's.

    Flat-band fallback: when gen_spread is within FLAT_BAND_REL_TOL of the
    band's mean magnitude, the scale is ill-defined and would amplify noise.
    In that case we fall back to mean-only matching (no scale).
    """
    gen_mean = _masked_mean(gen_band, mask_band)
    orig_mean = _masked_mean(orig_band, mask_band)
    if use_robust_spread:
        gen_spread = _masked_mean_abs_dev(gen_band, mask_band, gen_mean)
        orig_spread = _masked_mean_abs_dev(orig_band, mask_band, orig_mean)
    else:
        gen_spread = _masked_std(gen_band, mask_band, gen_mean)
        orig_spread = _masked_std(orig_band, mask_band, orig_mean)

    # Flat-band detection: compare gen_spread to band magnitude, not its mean
    # (which can be ~0 for Laplacian residuals). Use reference magnitude.
    band_magnitude = gen_band.abs().mean(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    flat = gen_spread < (FLAT_BAND_REL_TOL * band_magnitude)

    raw_scale = orig_spread / gen_spread
    scale = raw_scale.clamp(scale_clamp[0], scale_clamp[1])
    scale = torch.where(flat, torch.ones_like(scale), scale)
    return (gen_band - gen_mean) * scale + orig_mean


# =============================================================================
# Band group planner
# =============================================================================

def _plan_pyramid_levels(H, W, user_choice):
    """Decide pyramid depth. auto ~ log2(min_dim / 32), clamped to [3, 6]."""
    if user_choice == "auto":
        min_dim = min(H, W)
        levels = max(3, min(6, int(math.log2(max(min_dim, 64) / 32.0)) + 3))
        return levels
    try:
        return max(3, min(6, int(user_choice)))
    except (TypeError, ValueError):
        return 5


def _assign_band_groups(num_levels):
    """Partition level indices into top / mid / base groups.

    Convention: level 0 is finest Laplacian residual, num_levels-1 is base
    Gaussian. top = finest 1-2 levels; base = coarsest 1 level; mid = between.

    Narrowed base to single coarsest level (was 2 levels) on 2026-04-24 after
    real-footage testing showed dual base-band replacement at sigma~32 + sigma~64
    over-injects plate shadows into the gen face when mask_occupancy is low and
    tone_mode=replace is active. LFR v4's single-sigma replacement (~sigma=32)
    didn't have this issue. Narrowing to coarsest-only matches LFR's footprint
    while preserving the multi-band stat-matching benefits at top.
    """
    if num_levels <= 3:
        top = [0]
        base = [num_levels - 1]
        mid = [i for i in range(num_levels) if i not in top and i not in base]
    elif num_levels == 4:
        top = [0]
        base = [3]
        mid = [1, 2]
    else:  # 5, 6
        top = [0, 1]
        base = [num_levels - 1]
        mid = [i for i in range(num_levels) if i not in top and i not in base]
    return top, mid, base


# =============================================================================
# Band processors
# =============================================================================

def _occupancy_blend_weight(mask_occupancy):
    """Smooth [0, 1] lerp: 0 at occ <= OCC_REPLACE, 1 at occ >= OCC_STATISTICAL.

    Cubic smoothstep across the transition band prevents hard threshold
    pop-in when mask_occupancy drifts frame-to-frame (e.g., subject walks
    closer, grows from 70% → 80% of crop).
    """
    t = (float(mask_occupancy) - OCC_REPLACE) / max(OCC_STATISTICAL - OCC_REPLACE, 1e-6)
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)  # smoothstep


def _process_base_band(gen_band, orig_band, mask_band, mask_occupancy,
                       tone_mode, match_strength):
    """Base-band tone replacement.

    Content-aware routing with SMOOTH transition:
      - low occupancy (<60%): LFR structural swap (take orig band)
      - high occupancy (>90%): CCF statistical match (mean+std match gen to orig)
      - in between: smoothstep lerp between the two approaches

    User can override via tone_mode (replace | statistical | blend | auto).
    """
    if tone_mode == "statistical":
        candidate = _match_band_stats(gen_band, orig_band, mask_band, use_robust_spread=False)
    elif tone_mode == "replace":
        candidate = orig_band
    elif tone_mode == "blend":
        candidate = 0.5 * gen_band + 0.5 * orig_band
    else:  # "auto" — content-gated with smooth lerp
        w = _occupancy_blend_weight(mask_occupancy)
        replace_candidate = orig_band
        stat_candidate = _match_band_stats(gen_band, orig_band, mask_band, use_robust_spread=False)
        candidate = replace_candidate + (stat_candidate - replace_candidate) * w

    return gen_band + (candidate - gen_band) * match_strength


def _process_mid_band(gen_band, orig_band, ring_band, edge_protection):
    """Mid-band: preserve gen structure, apply boundary ring lock toward orig.

    ring_band is the finest-scale ring already downsampled through the mask
    pyramid, so it represents the same world-space region at this level.
    """
    if edge_protection <= 0:
        return gen_band
    pull_strength = edge_protection
    target = gen_band + (orig_band - gen_band) * pull_strength
    return gen_band + (target - gen_band) * ring_band


def _process_top_band(gen_band, orig_band, mask_band,
                      level_index, num_top_levels,
                      texture_strength, grain_strength):
    """Top-band: stat-match gen to orig.

    Finest level (0): grain residual — mean-AD spread (robust to edges), grain_strength.
    Second finest (1, if present): structural HF — std spread (edges ARE signal), texture_strength.
    """
    if num_top_levels >= 2 and level_index == 0:
        matched = _match_band_stats(gen_band, orig_band, mask_band, use_robust_spread=True)
        strength = grain_strength
    else:
        matched = _match_band_stats(gen_band, orig_band, mask_band, use_robust_spread=False)
        strength = texture_strength

    strength = max(0.0, min(2.0, strength))
    if strength <= 0.0:
        return gen_band
    return gen_band + (matched - gen_band) * strength


# =============================================================================
# Single-chunk processor (everything below goes through this for a batch slice)
# =============================================================================

def _process_batch_chunk(
    orig_bchw, gen_bchw, mask_bchw, mask_occupancy, *,
    match_strength, tone_mode,
    texture_strength, grain_strength,
    edge_protection,
    pyramid_levels, mask_erode_px, core_erode_px, support_tau, ring_px_finest,
    kernel,
):
    """Harmonize one batch chunk end-to-end. Returns a [B, 3, H, W] linear-RGB
    tensor (pre-sRGB conversion, pre-feather-blend). The caller handles the
    final sRGB conversion and feathered composite with original.
    """
    B, C, H, W = orig_bchw.shape

    # Decomposition-support mask (used for mid + top bands and the gen pyramid)
    mask_support = _erode_mask_2d(mask_bchw, mask_erode_px) if mask_erode_px > 0 else mask_bchw

    # Core-support mask: additional erosion for the BASE-BAND tone source. Decouples
    # the MNC normalizer denominator from the jittery silhouette boundary so the
    # global face-tone target stays stable across frames where the silhouette mask
    # transiently sweeps across boundary plate pixels (chin shadow, neck shadow,
    # hair, collar). Without this, blur(orig*m)/blur(m) at coarsest band reads a
    # different "global face color" each time the boundary moves a few pixels.
    #
    # Fallback is computed PER-FRAME, not per-chunk: a chunk-global sum can hide
    # near-empty masks behind larger ones in the same window, leaving a few bad
    # frames going through with degenerate stats. Per-frame torch.where ensures
    # each item in the batch independently picks its own support.
    if core_erode_px > 0:
        core_support_eroded = _erode_mask_2d(mask_support, core_erode_px)
        ms_sum = mask_support.sum(dim=(1, 2, 3), keepdim=True).clamp_min(1.0)
        cs_sum = core_support_eroded.sum(dim=(1, 2, 3), keepdim=True)
        # Per-frame fallback: where core eroded below 5% of mask_support's mass,
        # use mask_support directly. torch.where broadcasts the [B,1,1,1] gate.
        use_fallback = cs_sum < (0.05 * ms_sum)
        core_support = torch.where(use_fallback, mask_support, core_support_eroded)
    else:
        core_support = mask_support

    # Finest-scale ring, then pyramid-downsampled
    num_levels = _plan_pyramid_levels(H, W, pyramid_levels)
    ring_finest = _compute_boundary_ring(mask_bchw, ring_px_finest) if edge_protection > 0 else None
    ring_pyramid = _build_ring_pyramid(ring_finest, num_levels, kernel) if ring_finest is not None else None

    # Pyramids in linear RGB
    orig_lin = _srgb_to_linear(orig_bchw)
    gen_lin = _srgb_to_linear(gen_bchw)

    # Standard pyramid (mask_support) — used for mid + top bands.
    lap_orig, _ = _build_masked_laplacian_pyramid(orig_lin, mask_support, num_levels, kernel, support_tau)
    lap_gen, gauss_m = _build_masked_laplacian_pyramid(gen_lin, mask_support, num_levels, kernel, support_tau)

    # Core pyramid for the PLATE only (core_support) — used at base bands so the
    # tone-replacement source and stat-matching mask are jitter-immune. Built only
    # if core_erode_px actually narrowed the mask; otherwise reuses standard.
    if core_erode_px > 0 and not torch.equal(core_support, mask_support):
        lap_orig_core, gauss_m_core = _build_masked_laplacian_pyramid(
            orig_lin, core_support, num_levels, kernel, support_tau
        )
    else:
        lap_orig_core, gauss_m_core = lap_orig, gauss_m

    # Release tensors we no longer need for pyramid build
    del orig_lin, gen_lin

    top_levels, mid_levels, base_levels = _assign_band_groups(num_levels)

    pyr_out = [None] * num_levels

    # Base (tone replacement) — uses core pyramid so boundary jitter does not
    # contaminate the global face-tone target.
    for lvl in base_levels:
        pyr_out[lvl] = _process_base_band(
            lap_gen[lvl], lap_orig_core[lvl], gauss_m_core[lvl],
            mask_occupancy, tone_mode, match_strength,
        )

    # Mid (preserve gen + boundary ring lock) — uses standard pyramid so the
    # boundary signal is preserved (we WANT plate structure at the seam here).
    for lvl in mid_levels:
        ring_lvl = ring_pyramid[lvl] if ring_pyramid is not None else torch.zeros_like(lap_gen[lvl][:, :1])
        pyr_out[lvl] = _process_mid_band(lap_gen[lvl], lap_orig[lvl], ring_lvl, edge_protection)

    # Top (stat matching) — uses standard pyramid; texture/grain stats benefit
    # from full-mask coverage, and the [0.7, 1.43] scale clamp guards outliers.
    num_top = len(top_levels)
    for idx_in_top, lvl in enumerate(top_levels):
        pyr_out[lvl] = _process_top_band(
            lap_gen[lvl], lap_orig[lvl], gauss_m[lvl],
            level_index=idx_in_top, num_top_levels=num_top,
            texture_strength=texture_strength, grain_strength=grain_strength,
        )

    # Release source pyramids now that pyr_out is built
    del lap_orig, lap_gen, gauss_m, lap_orig_core, gauss_m_core

    result_lin = _collapse_pyramid(pyr_out, kernel)
    return result_lin, num_levels, (top_levels, mid_levels, base_levels)


# =============================================================================
# Node
# =============================================================================

class NV_FaceHarmonizePyramid:
    """Unified post-decode spatial harmonizer in linear RGB.

    Replaces NV_CropColorFix + NV_LowFreqRecompose + NV_TextureHarmonize for
    face-refinement workflows.

    Pipeline placement:
        VAE Decode → NV_FaceHarmonizePyramid → [NV_FrameTemporalStabilize] → NV_InpaintStitch2
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_crop": ("IMAGE", {
                    "tooltip": "Pristine cropped image from BEFORE VACE (plate reference). "
                               "Provides base-band tone / illumination and top-band texture stats."
                }),
                "generated_crop": ("IMAGE", {
                    "tooltip": "Post-VAE-decode AI output. Provides mid-band structure / identity."
                }),
                "mask": ("MASK", {
                    "tooltip": "Tight subject mask. Harmonization is applied inside this mask with "
                               "a soft edge falloff. Outside the mask, original pixels pass through."
                }),
                "match_strength": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Master intensity. Drives base-band tone replacement. "
                               "0 = passthrough of generated; 1 = full plate-anchored recomposition. "
                               "Default lowered to 0.5 (was 0.75) on 2026-04-24 after real-footage "
                               "testing — pure 0.75 replace pulled plate shadows into gen face. "
                               "Raise toward 1.0 only if tone match is undershooting."
                }),
            },
            "optional": {
                "tone_mode": (["auto", "replace", "blend", "statistical"], {
                    "default": "auto",
                    "tooltip": "Base-band policy. auto: smooth content-gated lerp between replace "
                               "(small mask) and statistical (large mask). replace: LFR structural "
                               "swap. blend: 50/50 gradient. statistical: CCF-style mean+std match."
                }),
                "texture_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Top-band sharpness matching strength (std-based). "
                               "Raise if face looks too clean vs plate; lower if it looks gritty."
                }),
                "grain_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Finest-band grain matching strength (MAD-proxy, robust to edges). "
                               "Reduces AI 'VAE fizz' by matching sensor noise of the plate."
                }),
                "edge_protection": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Boundary ring lock strength for mid bands. Ring is computed at the "
                               "finest pyramid scale (edge_protection * 5%% of short side) and "
                               "carried down through the mask pyramid so seam width is consistent "
                               "in world-space. 0 = off (pure gen structure everywhere in mid bands)."
                }),
                "edge_falloff_px": ("INT", {
                    "default": 8, "min": 0, "max": 64, "step": 1,
                    "tooltip": "Final soft-feather width at the mask boundary before compositing "
                               "harmonized pixels back over the original. Matches typical stitch "
                               "feather (8 px)."
                }),
                "pyramid_levels": (["auto", "3", "4", "5", "6"], {
                    "default": "auto",
                    "tooltip": "Laplacian pyramid depth. auto scales with crop size "
                               "(~5 for 512×512, ~6 for 1080p-ish crops)."
                }),
                "mask_erode_px": ("INT", {
                    "default": 2, "min": 0, "max": 16, "step": 1,
                    "tooltip": "Erosion radius for the decomposition-support mask. Keeps MNC "
                               "numerator/denominator away from boundary pixels whose value could "
                               "bleed into the band stats. 2 px default."
                }),
                "core_erode_px": ("INT", {
                    "default": 8, "min": 0, "max": 32, "step": 1,
                    "tooltip": "Additional erosion on top of mask_erode_px to derive a jitter-immune "
                               "CORE mask used only for BASE-BAND tone replacement. Decouples the "
                               "global face-tone target from silhouette boundary jitter (head nods, "
                               "rotation, SAM3 flicker). Without this, plate pixels just outside the "
                               "intended face region (chin/neck shadow, hair, collar) drift into the "
                               "MNC numerator when the mask boundary moves, producing dark blotches "
                               "in the harmonized output. 0 = legacy behavior. 8 px default ≈ 1.5%% "
                               "of a 512-px crop. Auto-falls-back to mask_support if erosion empties "
                               "the core."
                }),
                "support_tau": ("FLOAT", {
                    "default": 0.001, "min": 0.0, "max": 0.1, "step": 0.0005,
                    "tooltip": "MNC fallback threshold. Where blurred-mask support drops below this, "
                               "masked ratio is blended toward unmasked blur."
                }),
                "chunk_size": ("INT", {
                    "default": 16, "min": 1, "max": 256, "step": 1,
                    "tooltip": "Batch chunk size. Spatial ops are per-frame independent, so we "
                               "process chunks of this many frames at a time to bound VRAM for long "
                               "clips. 16 is safe for 1080p on 24GB VRAM; lower if you OOM."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("harmonized_crop", "info")
    FUNCTION = "execute"
    CATEGORY = "NV_Utils/Color"
    DESCRIPTION = (
        "EXPERIMENTAL — multi-AI calibration review (2026-04-25) found this node "
        "regresses color quality vs the proven CCF + TH combo on dynamic-lighting "
        "face shots. Use NV_CropColorFix + NV_TextureHarmonize for production face "
        "refinement; this node loses per-component decoupling that makes the 3-node "
        "stack tunable. Kept in tree as experimental architecture. "
        "Mechanism: unified spatial harmonizer via masked-normalized-convolution "
        "Laplacian pyramid in linear RGB, single match_strength driver + targeted "
        "overrides. No temporal logic — pair with NV_FrameTemporalStabilize."
    )

    def execute(self, original_crop, generated_crop, mask, match_strength,
                tone_mode="auto",
                texture_strength=1.0, grain_strength=1.0,
                edge_protection=0.5, edge_falloff_px=8,
                pyramid_levels="auto",
                mask_erode_px=2, core_erode_px=8, support_tau=0.001,
                chunk_size=16):
        info_lines = []

        # Shape validation
        if original_crop.dim() != 4 or generated_crop.dim() != 4:
            raise ValueError(
                f"original_crop and generated_crop must be [B, H, W, C]; "
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
        out_dtype = original_crop.dtype

        # BHWC → BCHW (float32 for numeric stability)
        orig_bchw_full = original_crop[..., :3].permute(0, 3, 1, 2).contiguous().float()
        gen_bchw_full = generated_crop[..., :3].permute(0, 3, 1, 2).contiguous().float()
        mask_bchw_full = mask.unsqueeze(1).float()

        # Mask support (eroded) — compute ONCE for full batch; derives mask_occupancy
        # as a clip-wide scalar (stable across chunks, avoids per-chunk drift).
        mask_support_full = (
            _erode_mask_2d(mask_bchw_full, mask_erode_px) if mask_erode_px > 0 else mask_bchw_full
        )
        mask_occupancy = float(mask_support_full.mean().item())
        # Save memory — we only needed mask_support to compute occupancy;
        # _process_batch_chunk will recompute per chunk.
        del mask_support_full

        # Finest-scale ring width, used for all chunks
        short_side_finest = min(H, W)
        ring_ratio = 0.05  # 5% of short side at max edge_protection
        ring_px_finest = max(1.0, ring_ratio * short_side_finest * edge_protection)

        # Kernel built once, reused across chunks
        kernel = _make_gaussian_kernel_2d(5).to(device=device, dtype=torch.float32)

        # Band plan is fixed by (H, W, pyramid_levels) — compute once for diag
        num_levels_preview = _plan_pyramid_levels(H, W, pyramid_levels)
        top_preview, mid_preview, base_preview = _assign_band_groups(num_levels_preview)
        info_lines.append(
            f"levels={num_levels_preview} | top={top_preview} mid={mid_preview} base={base_preview} | "
            f"mask_occupancy={mask_occupancy:.2f} (smooth gate) | ring_px_finest={ring_px_finest:.1f} | "
            f"mask_erode={mask_erode_px}px core_erode={core_erode_px}px (base-band)"
        )

        # Output accumulator (pre-allocated; avoids a list-and-cat spike for large B)
        out_bchw_full = torch.empty_like(orig_bchw_full)

        chunk_size = max(1, int(chunk_size))
        for start in range(0, B, chunk_size):
            end = min(start + chunk_size, B)
            orig_chunk = orig_bchw_full[start:end]
            gen_chunk = gen_bchw_full[start:end]
            mask_chunk = mask_bchw_full[start:end]

            result_lin, _, _ = _process_batch_chunk(
                orig_chunk, gen_chunk, mask_chunk, mask_occupancy,
                match_strength=match_strength, tone_mode=tone_mode,
                texture_strength=texture_strength, grain_strength=grain_strength,
                edge_protection=edge_protection,
                pyramid_levels=pyramid_levels,
                mask_erode_px=mask_erode_px,
                core_erode_px=core_erode_px,
                support_tau=support_tau,
                ring_px_finest=ring_px_finest,
                kernel=kernel,
            )

            # Back to sRGB, feathered composite with original plate
            result_rgb = _linear_to_srgb(result_lin).clamp(0.0, 1.0)
            soft_mask_chunk = _feather_mask(mask_chunk, edge_falloff_px)
            out_bchw_full[start:end] = (
                soft_mask_chunk * result_rgb + (1.0 - soft_mask_chunk) * orig_chunk
            )

            # Release chunk-local intermediates
            del result_lin, result_rgb, soft_mask_chunk

        # Diagnostic: shift magnitude vs generated (computed on full batch)
        with torch.no_grad():
            soft_mask_full = _feather_mask(mask_bchw_full, edge_falloff_px)
            inside = (soft_mask_full > 0.5)
            if inside.any():
                delta = (out_bchw_full - gen_bchw_full) * inside.float()
                mean_shift = delta.abs().mean().item() * 255.0
                max_shift = delta.abs().max().item() * 255.0
                info_lines.append(
                    f"in-mask shift from gen: mean={mean_shift:.2f}/255 max={max_shift:.2f}/255"
                )
            del soft_mask_full

        info_lines.append(f"chunks={math.ceil(B / chunk_size)} of {chunk_size} frames")

        # BCHW → BHWC, restore dtype, re-attach alpha from original plate
        output = out_bchw_full.permute(0, 2, 3, 1).contiguous().to(device=device, dtype=out_dtype)
        if C == 4:
            output = torch.cat([output, original_crop[..., 3:4]], dim=-1)

        info = f"{LOG_PREFIX} " + " | ".join(info_lines)
        print(info)
        return (output, info)


# ── Registration ─────────────────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    "NV_FaceHarmonizePyramid": NV_FaceHarmonizePyramid,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_FaceHarmonizePyramid": "NV Face Harmonize Pyramid (unified)",
}
