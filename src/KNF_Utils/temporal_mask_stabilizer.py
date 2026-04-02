"""
NV Temporal Mask Stabilizer — motion-aware temporal mask cleanup for video sequences.

Core path (when guide_image connected):
  1. Joint bilateral temporal filtering gated by source-frame color similarity
  2. Pop detection against an EMA prior, with geodesic reconstruction repair

Fallback path (no guide_image):
  - Temporal median consensus with reduced window

Optional bbox_mask crops to the region of interest before processing.
Optional MASK_PROCESSING_CONFIG applies spatial cleanup after stabilization.
Optional SDF smoothing is retained as a legacy stage and is disabled by default.

Input:
  - MASK [B, H, W] : per-frame masks (e.g. from SAM3)
  - IMAGE [B, H, W, C] : source video frames for motion-aware temporal gating
Output:
  - MASK [B, H, W] : temporally stabilized masks
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt, gaussian_filter, gaussian_filter1d

from .bbox_ops import compute_union_bbox
from .mask_ops import mask_erode_dilate, mask_fill_holes, mask_remove_noise, mask_smooth, mask_connected_components_gate

LOG_PREFIX = "[NV_TemporalMaskStabilizer]"


# =============================================================================
# Logit-space helpers
# =============================================================================

def mask_to_logits(mask, eps=1e-4):
    """Convert probabilities in [0, 1] to logits safely."""
    eps = float(max(1e-8, min(1e-2, eps)))
    return torch.logit(mask.clamp(eps, 1.0 - eps))


def logits_to_mask(logits):
    """Convert logits back to probabilities."""
    return torch.sigmoid(logits)


def _mask_lerp(a, b, weight, use_logit_space=False, logit_eps=1e-4):
    """Interpolate masks either in probability space or logit space."""
    weight = float(min(max(weight, 0.0), 1.0))
    if not use_logit_space:
        return torch.lerp(a, b, weight)
    a_l = mask_to_logits(a, eps=logit_eps)
    b_l = mask_to_logits(b, eps=logit_eps)
    return logits_to_mask(torch.lerp(a_l, b_l, weight))


# =============================================================================
# Stage 1a: Joint Bilateral Temporal Filter (GPU, motion-aware)
# =============================================================================

def _temporal_validity(frames, dt, device, dtype):
    """Validity mask for torch.roll-based temporal shifts (prevents wraparound)."""
    valid = torch.ones(frames, device=device, dtype=dtype)
    if dt > 0:
        valid[:dt] = 0
    elif dt < 0:
        valid[frames + dt:] = 0
    return valid.view(frames, 1, 1)


def joint_bilateral_temporal_filter(masks, guide_image, radius=3, sigma_color=0.1, sigma_time=1.5,
                                    use_logit_space=False, logit_eps=1e-4):
    """Motion-aware temporal smoothing using image color similarity as a gate.

    For each temporal offset dt:
      weight = exp(-dt^2 / (2*sigma_time^2)) * exp(-color_diff_sq / (2*sigma_color^2))

    Neighbor frames with large RGB change at a pixel contribute very little, which
    prevents moving-object shrinkage compared with naive temporal median / averaging.
    """
    if masks.dim() != 3:
        raise ValueError(f"masks must be [B, H, W], got shape {list(masks.shape)}")
    if guide_image.dim() != 4:
        raise ValueError(f"guide_image must be [B, H, W, C], got shape {list(guide_image.shape)}")
    if masks.shape[0] != guide_image.shape[0]:
        raise ValueError(f"Batch mismatch: mask batch={masks.shape[0]}, guide batch={guide_image.shape[0]}")
    if masks.shape[1] != guide_image.shape[1] or masks.shape[2] != guide_image.shape[2]:
        raise ValueError(f"Spatial mismatch: mask={list(masks.shape[1:])}, guide={list(guide_image.shape[1:3])}")

    frames = masks.shape[0]
    if frames <= 1 or radius <= 0:
        return masks.clone()

    sigma_color = max(float(sigma_color), 1e-6)
    sigma_time = max(float(sigma_time), 1e-6)

    guide = guide_image[..., :3].to(device=masks.device, dtype=torch.float32).contiguous()
    src_prob = masks.to(dtype=torch.float32).clamp(0.0, 1.0)
    src = mask_to_logits(src_prob, eps=logit_eps) if use_logit_space else src_prob

    numerator = src.clone()
    denominator = torch.ones_like(src)

    for dt in range(-radius, radius + 1):
        if dt == 0:
            continue

        shifted_mask = torch.roll(src, shifts=dt, dims=0)
        shifted_guide = torch.roll(guide, shifts=dt, dims=0)

        valid = _temporal_validity(frames, dt, src.device, src.dtype)
        color_diff_sq = torch.sum((guide - shifted_guide) ** 2, dim=-1)

        color_weight = torch.exp(-color_diff_sq / (2.0 * sigma_color * sigma_color))
        time_weight = math.exp(-(dt * dt) / (2.0 * sigma_time * sigma_time))
        weight = color_weight * valid * time_weight

        numerator = numerator + weight * shifted_mask
        denominator = denominator + weight

    filtered = numerator / denominator.clamp_min(1e-6)
    if use_logit_space:
        return logits_to_mask(filtered)
    return filtered.clamp(0.0, 1.0)


# =============================================================================
# Stage 1b: Fallback Temporal Median (no guide image)
# =============================================================================

def compute_temporal_consensus(masks, window_radius=3):
    """Fallback temporal median consensus when no guide image is available."""
    if masks.dim() != 3:
        raise ValueError(f"masks must be [B, H, W], got shape {list(masks.shape)}")

    frames = masks.shape[0]
    if frames <= 1 or window_radius <= 0:
        return masks.clone()

    result = []
    for t in range(frames):
        lo = max(0, t - window_radius)
        hi = min(frames, t + window_radius + 1)
        result.append(torch.median(masks[lo:hi], dim=0).values)
    return torch.stack(result, dim=0)


# =============================================================================
# Stage 2: Pop Detection + Geodesic Reconstruction
# =============================================================================

def _mask_iou(a, b, threshold=0.5):
    """Binary IoU for two single-frame masks [H, W]."""
    a_bin = a > threshold
    b_bin = b > threshold
    inter = torch.logical_and(a_bin, b_bin).sum().float()
    union = torch.logical_or(a_bin, b_bin).sum().float()
    if union.item() == 0:
        return 1.0
    return float((inter / union).item())


def _area_ratio_change(current, prior, threshold=0.5):
    """Relative area change between two single-frame masks [H, W]."""
    current_area = (current > threshold).float().mean()
    prior_area = (prior > threshold).float().mean()
    denom = prior_area.clamp_min(1e-6)
    return float((current_area - prior_area).abs().div(denom).item())


def geodesic_reconstruct(seed, support, iterations=10):
    """Geodesic reconstruction by iterative dilation clipped to the support mask.

    Uses F.max_pool2d for GPU-native dilation — no scipy/numpy.
    """
    seed_bin = seed.bool()
    support_bin = support.bool()

    if not torch.any(seed_bin) or not torch.any(support_bin):
        return seed_bin & support_bin

    current = seed_bin.unsqueeze(0).unsqueeze(0).float()
    support_4d = support_bin.unsqueeze(0).unsqueeze(0)

    for _ in range(max(1, int(iterations))):
        dilated = F.max_pool2d(current, kernel_size=3, stride=1, padding=1) > 0.5
        updated = torch.logical_and(dilated, support_4d)
        if torch.equal(updated, current > 0.5):
            break
        current = updated.float()

    return (current > 0.5).squeeze(0).squeeze(0)


def repair_pop_frame(current, prior, iterations=10, use_logit_space=False, logit_eps=1e-4):
    """Repair a popped frame by blending toward the prior and reconstructing support."""
    current = current.clamp(0.0, 1.0)
    prior = prior.clamp(0.0, 1.0)

    if use_logit_space:
        cur_l = mask_to_logits(current, eps=logit_eps)
        pri_l = mask_to_logits(prior, eps=logit_eps)
        blended = logits_to_mask(0.7 * pri_l + 0.3 * cur_l)
    else:
        blended = 0.7 * prior + 0.3 * current
    del current  # use blended from here
    support = blended > 0.30
    prior_bin = prior > 0.50

    overlap_seed = torch.logical_and(support, prior_bin)
    if torch.any(overlap_seed):
        connected_support = geodesic_reconstruct(overlap_seed, support, iterations=iterations)
    else:
        connected_support = support

    seed_source = torch.maximum(blended, prior)
    high_conf_seed = torch.logical_and(seed_source > 0.75, connected_support)

    if torch.any(high_conf_seed):
        reconstructed = geodesic_reconstruct(high_conf_seed, connected_support, iterations=iterations)
    else:
        reconstructed = connected_support

    repaired = blended * reconstructed.float()
    return repaired.clamp(0.0, 1.0)


def apply_pop_detection_and_repair(masks, enabled=True, area_thresh=0.20, iou_thresh=0.65, ema_alpha=0.25,
                                   info_lines=None, use_logit_space=False, logit_eps=1e-4):
    """Detect temporal pops against an EMA prior and repair them via geodesic reconstruction."""
    if info_lines is None:
        info_lines = []

    if not enabled or masks.shape[0] <= 1:
        if not enabled:
            info_lines.append("[Pop] Pop detection disabled.")
        return masks.clone(), 0

    result = masks.clone()
    prior = result[0].clone()
    repaired_count = 0

    stable_alpha = float(min(max(ema_alpha, 0.01), 0.99))
    pop_alpha = 0.10

    for t in range(result.shape[0]):
        current = result[t]
        area_change = _area_ratio_change(current, prior)
        iou = _mask_iou(current, prior)

        is_pop = area_change > float(area_thresh) or iou < float(iou_thresh)
        if is_pop:
            repaired = repair_pop_frame(current, prior, iterations=10,
                                       use_logit_space=use_logit_space, logit_eps=logit_eps)
            result[t] = repaired
            prior = _mask_lerp(prior, repaired, pop_alpha, use_logit_space=use_logit_space, logit_eps=logit_eps)
            repaired_count += 1
            info_lines.append(f"[Pop] Frame {t}: repaired (area_change={area_change:.3f}, iou={iou:.3f})")
        else:
            prior = _mask_lerp(prior, current, stable_alpha, use_logit_space=use_logit_space, logit_eps=logit_eps)

    if repaired_count == 0:
        info_lines.append("[Pop] No pops detected.")
    else:
        info_lines.append(f"[Pop] Repaired {repaired_count} frame(s).")

    return result, repaired_count


# =============================================================================
# Legacy SDF Smoothing (disabled by default)
# =============================================================================

def mask_to_sdf(mask_np, narrow_band=0):
    """Convert a binary-ish mask to a signed distance field."""
    binary = (mask_np > 0.5).astype(np.float64)
    dist_outside = distance_transform_edt(1.0 - binary)
    dist_inside = distance_transform_edt(binary)
    sdf = dist_outside - dist_inside
    if narrow_band > 0:
        sdf = np.clip(sdf, -narrow_band, narrow_band)
    return sdf.astype(np.float32)


def sdf_to_mask(sdf):
    """Convert SDF back to a near-binary soft mask."""
    return 1.0 / (1.0 + np.exp(sdf * 5.0))


def temporal_sdf_smooth(masks, sigma_temporal=1.0, sigma_spatial=0.5, narrow_band=64):
    """Legacy SDF temporal smoothing stage (CPU, scipy)."""
    frames, height, width = masks.shape
    masks_np = masks.detach().cpu().numpy()

    sdf_volume = np.zeros((frames, height, width), dtype=np.float32)
    for t in range(frames):
        sdf_volume[t] = mask_to_sdf(masks_np[t], narrow_band=narrow_band)

    if sigma_temporal > 0 and frames > 1:
        sdf_volume = gaussian_filter1d(sdf_volume, sigma=sigma_temporal, axis=0, mode="nearest")

    if sigma_spatial > 0:
        for t in range(frames):
            sdf_volume[t] = gaussian_filter(sdf_volume[t], sigma=sigma_spatial, mode="nearest")

    result = np.zeros((frames, height, width), dtype=np.float32)
    for t in range(frames):
        result[t] = sdf_to_mask(sdf_volume[t])

    return torch.from_numpy(result).to(device=masks.device, dtype=torch.float32)


# =============================================================================
# Crop Helpers
# =============================================================================

def crop_tensors(masks, images, bbox):
    """Crop masks [B,H,W] and images [B,H,W,C] to a bbox."""
    x, y, w, h = bbox
    cropped_masks = masks[:, y:y + h, x:x + w].clone()
    cropped_images = None
    if images is not None:
        cropped_images = images[:, y:y + h, x:x + w, :].clone()
    return cropped_masks, cropped_images


def paste_masks(result_full, cropped_result, bbox):
    """Paste cropped masks back into a full-frame tensor."""
    x, y, w, h = bbox
    result_full[:, y:y + h, x:x + w] = cropped_result
    return result_full


# =============================================================================
# Spatial Cleanup
# =============================================================================

def apply_spatial_cleanup(masks, erode_dilate=0, fill_holes=0, remove_noise=0, smooth=0):
    """Apply per-frame spatial cleanup after temporal stabilization."""
    result = masks
    if fill_holes > 0:
        result = mask_fill_holes(result, fill_holes)
    if remove_noise > 0:
        result = mask_remove_noise(result, remove_noise)
    if erode_dilate != 0:
        result = mask_erode_dilate(result, erode_dilate)
    if smooth > 0:
        result = mask_smooth(result, smooth)
    return result


# =============================================================================
# Full Pipeline
# =============================================================================

def run_stabilization_pipeline(masks, guide_image=None, consensus_window=3, sigma_color=0.1, sigma_time=1.5,
                               pop_detection=True, pop_area_thresh=0.20, pop_iou_thresh=0.65,
                               enable_sdf=False, sdf_sigma_temporal=1.0, sdf_sigma_spatial=0.5, sdf_narrow_band=64,
                               output_mode="binary", info_lines=None,
                               use_logit_space=False, logit_eps=1e-4,
                               cc_gate_enable=False, cc_gate_min_area=300, cc_gate_dilate=15, cc_gate_blur=5):
    """Run the full temporal mask stabilization pipeline."""
    if info_lines is None:
        info_lines = []

    if masks.dim() != 3:
        raise ValueError(f"mask must be [B, H, W], got shape {list(masks.shape)}")

    frames, height, width = masks.shape
    info_lines.append(f"[Input] {frames} frames at {height}x{width}")
    work = masks.to(dtype=torch.float32).clamp(0.0, 1.0)

    if frames > 1:
        if guide_image is not None:
            info_lines.append(f"[Stage 1] Joint bilateral temporal filter (radius={consensus_window}, "
                              f"sigma_color={sigma_color:.3f}, sigma_time={sigma_time:.3f})")
            print(f"{LOG_PREFIX} Stage 1: joint bilateral temporal filter "
                  f"(radius={consensus_window}, sigma_color={sigma_color}, sigma_time={sigma_time})")
            work = joint_bilateral_temporal_filter(work, guide_image, radius=consensus_window,
                                                   sigma_color=sigma_color, sigma_time=sigma_time,
                                                   use_logit_space=use_logit_space, logit_eps=logit_eps)
        else:
            info_lines.append(f"[Stage 1] Fallback temporal median (radius={consensus_window})")
            print(f"{LOG_PREFIX} Stage 1: fallback temporal median (radius={consensus_window})")
            work = compute_temporal_consensus(work, window_radius=consensus_window)
    else:
        info_lines.append("[Stage 1] Single frame, temporal filtering skipped.")

    work, _ = apply_pop_detection_and_repair(work, enabled=pop_detection, area_thresh=pop_area_thresh,
                                             iou_thresh=pop_iou_thresh, info_lines=info_lines,
                                             use_logit_space=use_logit_space, logit_eps=logit_eps)

    if enable_sdf:
        info_lines.append(f"[Stage 3] Legacy SDF smoothing (sigma_t={sdf_sigma_temporal:.2f}, "
                          f"sigma_s={sdf_sigma_spatial:.2f})")
        print(f"{LOG_PREFIX} Stage 3: legacy SDF smoothing (sigma_t={sdf_sigma_temporal}, sigma_s={sdf_sigma_spatial})")
        work = temporal_sdf_smooth(work, sigma_temporal=sdf_sigma_temporal,
                                   sigma_spatial=sdf_sigma_spatial, narrow_band=sdf_narrow_band)

    # --- Stage 4: Connected components gating ---
    if cc_gate_enable:
        info_lines.append(f"[Stage 4] CC gating (min_area={cc_gate_min_area}, dilate={cc_gate_dilate}, blur={cc_gate_blur})")
        print(f"{LOG_PREFIX} Stage 4: CC gating (min_area={cc_gate_min_area}, dilate={cc_gate_dilate})")
        work = mask_connected_components_gate(work, min_area=cc_gate_min_area,
                                              dilate_radius=cc_gate_dilate, blur_kernel=cc_gate_blur)

    if output_mode == "binary":
        work = (work > 0.5).float()
        info_lines.append("[Output] Binary mask at threshold 0.5.")
    else:
        work = work.clamp(0.0, 1.0)
        info_lines.append("[Output] Soft mask preserved.")

    return work


# =============================================================================
# ComfyUI Node
# =============================================================================

class NV_TemporalMaskStabilizer:
    """Motion-aware temporal stabilization for video masks."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK", {
                    "tooltip": "Per-frame masks [B, H, W]. Connect SAM3 or any mask sequence."
                }),
                "consensus_window": ("INT", {
                    "default": 3, "min": 1, "max": 10, "step": 1,
                    "tooltip": "Temporal window radius in frames. 3 = recommended (7-frame window). "
                               "Larger windows increase smoothing but can hurt motion tracking."
                }),
                "output_mode": (["binary", "soft"], {
                    "default": "binary",
                    "tooltip": "binary: threshold to clean solid masks. soft: keep soft probabilities."
                }),
            },
            "optional": {
                "guide_image": ("IMAGE", {
                    "tooltip": "Source video frames [B, H, W, C] for motion-aware temporal filtering. "
                               "STRONGLY RECOMMENDED — without this, falls back to temporal median "
                               "which shrinks masks on moving objects."
                }),
                "bbox_mask": ("MASK", {
                    "tooltip": "Optional bbox/ROI mask. Stabilization runs only inside union crop."
                }),
                "mask_config": ("MASK_PROCESSING_CONFIG", {
                    "tooltip": "Shared mask-processing config bus. Overrides local cleanup widgets."
                }),
                "sigma_color": ("FLOAT", {
                    "default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01,
                    "tooltip": "Color-similarity sigma for bilateral filter. Lower = stricter motion rejection."
                }),
                "sigma_time": ("FLOAT", {
                    "default": 1.5, "min": 0.1, "max": 5.0, "step": 0.1,
                    "tooltip": "Temporal decay sigma in frames. Lower = prioritize nearby frames."
                }),
                "pop_detection": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Detect sudden mask pops and repair with geodesic reconstruction."
                }),
                "pop_area_thresh": ("FLOAT", {
                    "default": 0.20, "min": 0.05, "max": 0.5, "step": 0.01,
                    "tooltip": "Pop trigger: relative area change threshold."
                }),
                "pop_iou_thresh": ("FLOAT", {
                    "default": 0.65, "min": 0.3, "max": 0.95, "step": 0.01,
                    "tooltip": "Pop trigger: IoU with EMA prior below which to repair."
                }),
                "use_logit_space": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Operate in logit space instead of probability space. Prevents saturation blocking "
                               "near 0/1 values. Experimental — may need threshold retuning."
                }),
                "cc_gate_enable": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable connected-components gating after stabilization. Kills isolated noise specks "
                               "while preserving soft edges in valid regions. (CorridorKey-style clean_matte)"
                }),
                "cc_gate_min_area": ("INT", {
                    "default": 300, "min": 10, "max": 10000, "step": 10,
                    "tooltip": "Minimum component area in pixels to keep. Smaller components are killed."
                }),
                "crop_expand_px": ("INT", {
                    "default": 0, "min": -128, "max": 128, "step": 1,
                    "tooltip": "Post-stabilization erosion (<0) or dilation (>0). Overridden by mask_config."
                }),
                "cleanup_fill_holes": ("INT", {
                    "default": 0, "min": 0, "max": 128, "step": 1,
                    "tooltip": "Post-stabilization hole filling. Overridden by mask_config."
                }),
                "cleanup_remove_noise": ("INT", {
                    "default": 0, "min": 0, "max": 32, "step": 1,
                    "tooltip": "Post-stabilization noise removal. Overridden by mask_config."
                }),
                "cleanup_smooth": ("INT", {
                    "default": 0, "min": 0, "max": 127, "step": 1,
                    "tooltip": "Post-stabilization edge smoothing. Overridden by mask_config."
                }),
                # Deprecated names (backward compat for old workflows)
                "mask_erode_dilate": ("INT", {
                    "default": 0, "min": -128, "max": 128, "step": 1,
                    "tooltip": "DEPRECATED — use crop_expand_px"
                }),
                "mask_fill_holes": ("INT", {
                    "default": 0, "min": 0, "max": 128, "step": 1,
                    "tooltip": "DEPRECATED — use cleanup_fill_holes"
                }),
                "mask_remove_noise": ("INT", {
                    "default": 0, "min": 0, "max": 32, "step": 1,
                    "tooltip": "DEPRECATED — use cleanup_remove_noise"
                }),
                "mask_smooth": ("INT", {
                    "default": 0, "min": 0, "max": 127, "step": 1,
                    "tooltip": "DEPRECATED — use cleanup_smooth"
                }),
                "enable_sdf": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable legacy SDF smoothing after bilateral + pop pipeline. Off by default."
                }),
                "sdf_sigma_temporal": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1,
                    "tooltip": "Legacy SDF temporal sigma. Only used when enable_sdf is on."
                }),
                "sdf_sigma_spatial": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 5.0, "step": 0.1,
                    "tooltip": "Legacy SDF spatial sigma. Only used when enable_sdf is on."
                }),
                "crop_padding": ("FLOAT", {
                    "default": 0.15, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Padding around bbox crop. Only used when bbox_mask is connected."
                }),
            },
        }

    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("mask", "info")
    FUNCTION = "execute"
    CATEGORY = "NV_Utils/Mask"
    DESCRIPTION = (
        "Motion-aware temporal mask stabilization. Uses joint bilateral temporal filtering "
        "with source video frames to prevent mask shrinkage on moving objects, then repairs "
        "sudden pops using an EMA prior and geodesic reconstruction. "
        "Falls back to temporal median if guide_image is not connected."
    )

    def execute(self, mask, consensus_window, output_mode,
                guide_image=None, bbox_mask=None, mask_config=None,
                sigma_color=0.1, sigma_time=1.5,
                pop_detection=True, pop_area_thresh=0.20, pop_iou_thresh=0.65,
                use_logit_space=False, cc_gate_enable=False, cc_gate_min_area=300,
                crop_expand_px=0, cleanup_fill_holes=0, cleanup_remove_noise=0, cleanup_smooth=0,
                # Deprecated names (backward compat)
                mask_erode_dilate=0, mask_fill_holes=0, mask_remove_noise=0, mask_smooth=0,
                enable_sdf=False, sdf_sigma_temporal=1.0, sdf_sigma_spatial=0.5, crop_padding=0.15):

        # Resolve deprecated param names
        from .mask_processing_config import resolve_deprecated, apply_mask_config
        crop_expand_px = resolve_deprecated(crop_expand_px, 0, mask_erode_dilate, 0)
        cleanup_fill_holes = resolve_deprecated(cleanup_fill_holes, 0, mask_fill_holes, 0)
        cleanup_remove_noise = resolve_deprecated(cleanup_remove_noise, 0, mask_remove_noise, 0)
        cleanup_smooth = resolve_deprecated(cleanup_smooth, 0, mask_smooth, 0)

        if mask.dim() != 3:
            raise ValueError(f"mask must be [B, H, W], got shape {list(mask.shape)}")

        info_lines = []
        frames, height, width = mask.shape
        print(f"{LOG_PREFIX} Starting: {frames} frames, {height}x{width}")

        if guide_image is not None:
            if guide_image.dim() != 4:
                raise ValueError(f"guide_image must be [B, H, W, C], got shape {list(guide_image.shape)}")
            if guide_image.shape[0] != frames:
                raise ValueError(f"guide_image batch must match mask: guide={guide_image.shape[0]}, mask={frames}")
            if guide_image.shape[1] != height or guide_image.shape[2] != width:
                raise ValueError(f"guide_image spatial must match mask: guide={list(guide_image.shape[1:3])}, "
                                 f"mask={[height, width]}")
            guide_image = guide_image.to(device=mask.device, dtype=torch.float32).clamp(0.0, 1.0)

        mask = mask.to(dtype=torch.float32).clamp(0.0, 1.0)

        vals = apply_mask_config(mask_config, crop_expand_px=crop_expand_px, cleanup_fill_holes=cleanup_fill_holes,
                                 cleanup_remove_noise=cleanup_remove_noise, cleanup_smooth=cleanup_smooth)
        post_erode_dilate = vals["crop_expand_px"]
        post_fill_holes = vals["cleanup_fill_holes"]
        post_remove_noise = vals["cleanup_remove_noise"]
        post_smooth = vals["cleanup_smooth"]
        has_spatial_cleanup = (post_erode_dilate != 0 or post_fill_holes > 0
                               or post_remove_noise > 0 or post_smooth > 0)

        use_crop = bbox_mask is not None
        crop_bbox = None
        work_masks = mask
        work_images = guide_image

        if use_crop:
            crop_bbox = compute_union_bbox(bbox_mask, padding_frac=crop_padding)
            if crop_bbox is not None:
                x, y, w, h = crop_bbox
                info_lines.append(f"[Crop] Using bbox crop ({x},{y}) {w}x{h} with padding={crop_padding:.0%}.")
                work_masks, work_images = crop_tensors(mask, guide_image, crop_bbox)
            else:
                info_lines.append("[Crop] bbox_mask is empty. Processing full frame.")
                use_crop = False

        stabilized = run_stabilization_pipeline(
            masks=work_masks, guide_image=work_images, consensus_window=consensus_window,
            sigma_color=sigma_color, sigma_time=sigma_time,
            pop_detection=pop_detection, pop_area_thresh=pop_area_thresh, pop_iou_thresh=pop_iou_thresh,
            enable_sdf=enable_sdf, sdf_sigma_temporal=sdf_sigma_temporal, sdf_sigma_spatial=sdf_sigma_spatial,
            sdf_narrow_band=64, output_mode=output_mode, info_lines=info_lines,
            use_logit_space=use_logit_space, logit_eps=1e-4,
            cc_gate_enable=cc_gate_enable, cc_gate_min_area=cc_gate_min_area,
        )

        if use_crop and crop_bbox is not None:
            result = torch.zeros_like(mask)
            result = paste_masks(result, stabilized, crop_bbox)
            info_lines.append("[Crop] Pasted stabilized crop back into full-frame mask.")
        else:
            result = stabilized

        if has_spatial_cleanup:
            cleanup_parts = []
            if post_fill_holes > 0:
                cleanup_parts.append(f"fill_holes={post_fill_holes}")
            if post_remove_noise > 0:
                cleanup_parts.append(f"remove_noise={post_remove_noise}")
            if post_erode_dilate != 0:
                cleanup_parts.append(f"erode_dilate={post_erode_dilate}")
            if post_smooth > 0:
                cleanup_parts.append(f"smooth={post_smooth}")
            info_lines.append(f"[Spatial] Post-cleanup: {', '.join(cleanup_parts)}")
            print(f"{LOG_PREFIX} Applying spatial cleanup: {', '.join(cleanup_parts)}")
            result = apply_spatial_cleanup(result, erode_dilate=post_erode_dilate, fill_holes=post_fill_holes,
                                           remove_noise=post_remove_noise, smooth=post_smooth)
            if output_mode == "binary":
                result = (result > 0.5).float()

        info = "\n".join(info_lines)
        print(f"{LOG_PREFIX} Done. {frames} frames processed.")
        return (result.clamp(0.0, 1.0), info)


NODE_CLASS_MAPPINGS = {
    "NV_TemporalMaskStabilizer": NV_TemporalMaskStabilizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_TemporalMaskStabilizer": "NV Temporal Mask Stabilizer",
}
