"""
NV Crop Color Fix V2 — Multipass diffusion drift correction before stitching.

When KSampler runs at denoise < 1.0, it modifies the ENTIRE crop — not just
the masked region. Non-masked pixels come back color-shifted (~7/255 typical).
This creates a visible seam when stitched onto the original frame.

V2 multipass pipeline:
  Step 1: Low-frequency Lab-space color correction (eroded reference, preserves HF detail)
  Step 2: Boundary-local residual correction (distance-weighted, guided-filter regularized)
  Step 3: Frequency-aware composite (multiband Laplacian pyramid OR Gaussian alpha)

Motion-adaptive features (Phase 1B + Phase 2):
  - Per-frame velocity estimation via mask centroid displacement
  - Velocity-scaled blend width (wider on fast-motion frames)
  - Temporal EMA of blend weight maps with centroid-based crop-space alignment
  - Hard boundary lock to prevent correction bleeding into untouched pixels

Place between VAE Decode and InpaintStitch2:
  InpaintCrop2 → VAE Encode → KSampler → VAE Decode
      → [NV_CropColorFix] → InpaintStitch2
"""

import gc

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
    # Clamp kernel so padding < spatial dim (reflect padding requirement)
    max_k = min(tensor.shape[2], tensor.shape[3]) * 2 - 1
    if k > max_k:
        k = max(max_k // 2 * 2 + 1, 1)  # keep odd
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


def _compute_mask_centroids_and_velocity(mask):
    """Compute per-frame mask centroids and inter-frame velocity.

    Args:
        mask: [B, H, W] float tensor (1.0 = generated region).

    Returns:
        centroids: [B, 2] tensor of (y, x) centroids per frame.
        velocity: [B] tensor of L2 centroid displacement (0 for frame 0).
        has_mask: [B] bool tensor — True where frame had real mask pixels.
    """
    B, H, W = mask.shape
    device = mask.device
    default_centroid = torch.tensor([H / 2.0, W / 2.0], device=device)
    centroids = torch.zeros(B, 2, device=device)
    has_mask = torch.zeros(B, dtype=torch.bool, device=device)

    for b in range(B):
        indices = torch.nonzero(mask[b] > 0.5, as_tuple=False)  # [N, 2] — (y, x)
        if indices.numel() == 0:
            # No mask pixels: use previous centroid or default
            centroids[b] = centroids[b - 1] if b > 0 else default_centroid
        else:
            centroids[b] = indices.float().mean(dim=0)
            has_mask[b] = True

    velocity = torch.zeros(B, device=device)
    for b in range(1, B):
        # Only compute velocity when both frames have real masks
        # (avoids spike from default center → real mask position)
        if has_mask[b] and has_mask[b - 1]:
            velocity[b] = torch.norm(centroids[b] - centroids[b - 1], p=2)

    return centroids, velocity, has_mask


def _translate_mask_2d(mask_2d, dx, dy):
    """Translate a [H, W] mask by (dx, dy) pixels using affine_grid + grid_sample.

    Used for crop-space alignment of temporal EMA blend weights.
    """
    inp = mask_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    _, _, H, W = inp.shape

    # Normalized translation for grid_sample (align_corners=True)
    tx = 0.0 if W <= 1 else (-2.0 * dx / (W - 1))
    ty = 0.0 if H <= 1 else (-2.0 * dy / (H - 1))

    theta = inp.new_tensor([[[1.0, 0.0, tx], [0.0, 1.0, ty]]])  # [1, 2, 3]
    grid = F.affine_grid(theta, inp.size(), align_corners=True)
    out = F.grid_sample(inp, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return out[0, 0]  # [H, W]


class NV_CropColorFix:
    """Multipass color correction and composite for generated crops before stitching.

    V2 pipeline: LF Lab correction → boundary-local residual → frequency-aware composite.
    Motion-adaptive: velocity-scaled blend width + temporal weight EMA + boundary lock.
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
                    "tooltip": "Controls how far boundary correction propagates inward (scales blur sigma). "
                               "Larger = correction reaches further. 0.15 = ~10px effective radius."
                }),
                "temporal_smooth": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 0.95, "step": 0.05,
                    "tooltip": "EMA smoothing for blend weight maps across frames (0=off, 0.3=moderate, 0.8=heavy). "
                               "Uses centroid-aligned crop-space translation for temporal coherence. "
                               "Also smooths color correction values when > 0."
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
                "motion_sensitivity": ("FLOAT", {
                    "default": 1.5, "min": 0.0, "max": 5.0, "step": 0.1,
                    "tooltip": "Scale factor for velocity-adaptive blend width. 0=off (static width), "
                               "1.5=default, higher=wider blend on fast-motion frames. "
                               "Blend width = blend_pixels + K * velocity, clamped to 3x max."
                }),
                "boundary_lock": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Force pristine source pixels outside the blend zone after compositing. "
                               "Fixes color correction bleeding into untouched areas (0.9/255 leak)."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("corrected_crop", "correction_info",)
    FUNCTION = "execute"
    CATEGORY = "NV_Utils/Inpaint"
    DESCRIPTION = (
        "V2 multipass diffusion drift fix with motion-adaptive blending. "
        "Step 1: Lab-space low-frequency color correction (preserves texture detail). "
        "Step 2: Boundary-local residual correction. "
        "Step 3: Frequency-aware composite (multiband Laplacian pyramid) with "
        "velocity-adaptive blend width and temporal weight smoothing. "
        "Place between VAE Decode and InpaintStitch2."
    )

    def execute(self, original_crop, generated_crop, mask, color_correction,
                composite_mode, blend_pixels, blend_expansion=0,
                ref_threshold=0.01, ref_erosion=8, lf_sigma=10.0,
                boundary_correction=True, boundary_ring=12, boundary_decay=0.15,
                temporal_smooth=0.0, multiband_levels=4, min_ref_pixels=100,
                motion_sensitivity=1.5, boundary_lock=True):
        device = original_crop.device
        TAG = "[NV_CropColorFix]"

        # --- Batch / spatial alignment ---
        B_orig = original_crop.shape[0]
        B_gen = generated_crop.shape[0]
        if B_orig != B_gen:
            raise ValueError(
                f"{TAG} Frame count mismatch! original_crop has {B_orig} frames but "
                f"generated_crop has {B_gen} frames. If using tail prepend from "
                f"VaceControlVideoPrep, ensure tail_trim frames are stripped from the "
                f"generated output BEFORE feeding into CropColorFix. "
                f"Use ImageFromBatch(batch_index=tail_trim, length={B_orig})."
            )
        B = B_orig
        H, W, C = original_crop.shape[1], original_crop.shape[2], original_crop.shape[3]
        print(f"{TAG} Starting: {B} frames, {H}x{W}, correction={color_correction}, composite={composite_mode}")

        gen = generated_crop[:B].float()
        orig = original_crop[:B].float()
        if gen.shape[1:3] != orig.shape[1:3]:
            gen = gen.permute(0, 3, 1, 2)
            gen = F.interpolate(gen, size=(H, W), mode="bilinear", align_corners=False)
            gen = gen.permute(0, 2, 3, 1)

        # Prepare mask [B, H, W]
        m = mask.float().to(device)
        if m.dim() == 2:
            m = m.unsqueeze(0)
        if m.shape[1:] != (H, W):
            m = F.interpolate(m.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False).squeeze(1)
        if m.shape[0] > B:
            m = m[:B]
        elif m.shape[0] < B:
            # Repeat last mask frame to fill batch (handles both 1→B and partial batches)
            pad = m[-1:].expand(B - m.shape[0], -1, -1)
            m = torch.cat([m, pad], dim=0)

        info_lines = [f"NV_CropColorFix V2: {B} frames, {H}x{W}, correction={color_correction}, composite={composite_mode}"]
        if B_orig != B_gen:
            info_lines.append(f"  Batch mismatch: original={B_orig}, generated={B_gen}, using {B}")

        # =================================================================
        # Motion analysis (Phase 1B + Phase 2)
        # =================================================================
        print(f"{TAG} Analyzing motion...")
        centroids, velocity, has_mask = _compute_mask_centroids_and_velocity(m)
        if B > 1:
            fast_count = (velocity > 5.0).sum().item()
            info_lines.append(
                f"  Motion: velocity min={velocity.min().item():.1f}, "
                f"mean={velocity.mean().item():.1f}, max={velocity.max().item():.1f} px/frame "
                f"({int(fast_count)} fast frames >5px)"
            )
        else:
            info_lines.append("  Motion: single frame (velocity=0)")

        # =====================================================================
        # STEP 1: Color correction
        # =====================================================================
        print(f"{TAG} Step 1: Color correction ({color_correction})...")
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

                    # Apply Reinhard correction using (potentially EMA-smoothed) shift and scale
                    corrected_work[b, c] = (work_src[b, c] - mu_src) * scale + (mu_src + shift)

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
        print(f"{TAG} Step 2: Boundary correction (enabled={boundary_correction})...")
        if boundary_correction and color_correction != "none":
            # Build boundary ring on the exterior (original) side
            mask_binary = (m > 0.5).float()  # [B, H, W]
            outer_ring = _dilate_mask_2d(mask_binary, boundary_ring) - mask_binary  # [B, H, W]
            outer_ring = outer_ring.clamp(0, 1)

            # Build distance-based decay field from mask edge into interior
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
        print(f"{TAG} Step 3: Composite ({composite_mode}, blend={blend_pixels}px, expansion={blend_expansion}px)...")
        # Build the composite mask
        comp_mask = m.clone()

        # Dilate mask outward to push blend zone into original pixels
        if blend_expansion > 0:
            comp_mask = _dilate_mask_2d(comp_mask, blend_expansion)

        # --- Phase 2: Temporal EMA on blend weight map (centroid-aligned) ---
        if temporal_smooth > 0 and B > 1:
            print(f"{TAG}   Phase 2: Temporal EMA (smooth={temporal_smooth})...")
            for b in range(1, B):
                # Skip EMA on mask appearance/disappearance transitions to avoid ghosting
                if not (has_mask[b] and has_mask[b - 1]):
                    continue

                dy = (centroids[b, 0] - centroids[b - 1, 0]).item()
                dx = (centroids[b, 1] - centroids[b - 1, 1]).item()

                # Translate previous weight map to align with current frame's crop-space
                # dx/dy = how centroid moved; shift prev mask in same direction to align
                prev_aligned = _translate_mask_2d(comp_mask[b - 1], dx, dy).clamp(0, 1)

                # Base alpha derived from temporal_smooth slider (higher slider = more smoothing = lower alpha)
                # Motion-adaptive: fast motion → less smoothing (alpha pushed toward 1.0)
                base_alpha = 1.0 - temporal_smooth  # slider 0.3 → alpha 0.7, slider 0.8 → alpha 0.2
                motion_boost = 0.2 * min(velocity[b].item() / 20.0, 1.0)
                ema_alpha = min(base_alpha + motion_boost, 1.0)
                comp_mask[b] = (ema_alpha * comp_mask[b] + (1.0 - ema_alpha) * prev_aligned).clamp(0, 1)

            info_lines.append(
                f"  Phase 2: temporal EMA on blend weights (centroid-aligned, decay adaptive)"
            )

        # --- Phase 1B: Velocity-adaptive blend width ---
        print(f"{TAG}   Phase 1B: Velocity-adaptive blend (K={motion_sensitivity})...")
        use_per_frame = motion_sensitivity > 0 and B > 1
        blend_base = float(blend_pixels)
        # Ensure adaptive mode has a nonzero floor so velocity can actually widen the blend
        blend_min = max(blend_base, 3.0) if use_per_frame else blend_base
        blend_max = max(blend_base * 3.0, 9.0) if use_per_frame else blend_base * 3.0

        if use_per_frame:
            effective_blend = (blend_base + motion_sensitivity * velocity).clamp(blend_min, blend_max)
            info_lines.append(
                f"  Phase 1B: adaptive blend min={effective_blend.min().item():.1f}, "
                f"mean={effective_blend.mean().item():.1f}, max={effective_blend.max().item():.1f} "
                f"(K={motion_sensitivity:.1f})"
            )
        else:
            effective_blend = torch.full((B,), blend_base, device=device)
            if motion_sensitivity == 0:
                info_lines.append("  Phase 1B: disabled (motion_sensitivity=0)")

        # --- Composite ---
        if composite_mode == "multiband":
            corr_nchw = corrected.permute(0, 3, 1, 2)  # [B, C, H, W]
            orig_nchw = orig.permute(0, 3, 1, 2)        # [B, C, H, W]

            if use_per_frame:
                # Per-frame composite: each frame gets velocity-adapted blend width
                out_frames = []
                log_interval = max(B // 10, 1)
                for b in range(B):
                    if b % log_interval == 0:
                        print(f"{TAG}   Multiband blend: frame {b}/{B}...")
                    mask_b = comp_mask[b:b + 1].unsqueeze(1)  # [1, 1, H, W]
                    sigma_b = effective_blend[b].item() / 3.0
                    if sigma_b > 0:
                        mask_b = _gaussian_blur_reflect(mask_b, sigma_b).clamp(0, 1)
                    blended_b = multiband_blend(
                        corr_nchw[b:b + 1], orig_nchw[b:b + 1], mask_b, num_levels=multiband_levels
                    )
                    out_frames.append(blended_b)
                # Free NCHW intermediates before cat to reduce peak memory
                del corr_nchw, orig_nchw
                blended_nchw = torch.cat(out_frames, dim=0)
                del out_frames
                output = blended_nchw.clamp(0, 1).permute(0, 2, 3, 1)
                del blended_nchw
                info_lines.append(
                    f"  Step 3 (multiband): {multiband_levels} levels, adaptive width, expansion={blend_expansion}"
                )
            else:
                # Batch mode: all frames use same blend width (fast path)
                mask_nchw = comp_mask.unsqueeze(1)  # [B, 1, H, W]
                if blend_pixels > 0:
                    mask_nchw = _gaussian_blur_reflect(mask_nchw, blend_pixels / 3.0)
                blended_nchw = multiband_blend(corr_nchw, orig_nchw, mask_nchw, num_levels=multiband_levels)
                del corr_nchw, orig_nchw
                output = blended_nchw.clamp(0, 1).permute(0, 2, 3, 1)  # [B, H, W, C]
                del blended_nchw
                info_lines.append(f"  Step 3 (multiband): {multiband_levels} levels, expansion={blend_expansion}")

        elif composite_mode == "gaussian":
            if use_per_frame:
                # Per-frame Gaussian feathering with adaptive sigma
                feathered = torch.zeros_like(comp_mask)
                for b in range(B):
                    sigma_b = effective_blend[b].item() / 3.0
                    if sigma_b > 0:
                        feathered[b:b + 1] = _gaussian_blur_reflect(
                            comp_mask[b:b + 1].unsqueeze(1), sigma_b
                        ).squeeze(1).clamp(0, 1)
                    else:
                        feathered[b] = comp_mask[b]
                alpha = feathered.unsqueeze(-1)  # [B, H, W, 1]
                output = orig * (1.0 - alpha) + corrected * alpha
                output = output.clamp(0, 1)
                info_lines.append(f"  Step 3 (gaussian): adaptive sigma, expansion={blend_expansion}")
            else:
                # Batch mode
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

        # =====================================================================
        # Phase 1A: Hard boundary lock — force pristine source outside blend zone
        # =====================================================================
        print(f"{TAG} Phase 1A: Boundary lock (enabled={boundary_lock})...")
        if boundary_lock:
            # Erode the inverse mask inward by blend transition width + safety margin
            # so the lock doesn't truncate the blending transition zone
            # In adaptive mode, use max effective_blend to cover widest frame's transition
            if use_per_frame:
                lock_margin = max(int(effective_blend.max().item()) + blend_expansion, 2)
            else:
                lock_margin = max(blend_pixels + blend_expansion, 2)
            inv_mask = (1.0 - comp_mask).clamp(0, 1)
            eroded_inv = _erode_mask_2d(inv_mask, lock_margin)
            lock_zone = (eroded_inv > 0.5).unsqueeze(-1)  # [B, H, W, 1]
            output = torch.where(lock_zone, orig, output)
            info_lines.append(f"  Phase 1A: boundary lock applied ({lock_margin}px eroded keep zone)")

        output = output.clamp(0, 1)

        # --- Final residual measurement ---
        ref_zone = m < ref_threshold
        if ref_zone.sum() > 0:
            residual = (output[..., :3] - orig[..., :3]).abs()
            ref_residual = residual[ref_zone.unsqueeze(-1).expand_as(residual)].mean().item()
            info_lines.append(f"  Final residual in ref zone: {ref_residual * 255:.2f}/255")
        else:
            info_lines.append("  No reference zone pixels to measure residual")

        # Free large intermediates before returning
        del corrected, comp_mask
        gc.collect()
        torch.cuda.empty_cache()

        info_text = "\n".join(info_lines)
        print(f"{TAG} Done. {B} frames processed.")
        return (output, info_text,)


NODE_CLASS_MAPPINGS = {
    "NV_CropColorFix": NV_CropColorFix,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_CropColorFix": "NV Crop Color Fix",
}
