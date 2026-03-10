"""
NV Temporal Mask Stabilizer — Temporal mask cleaning for video sequences.

Fixes mask "pops" (sudden shape changes between frames) using a 2-stage pipeline:

  Stage 1: Temporal Consensus — temporal median of neighboring frames
  Stage 2: SDF Temporal Smoothing — signed distance field smoothing (continuous boundary)

Optional bbox_mask input crops to the mask region before processing (faster + higher detail).
Optional MASK_PROCESSING_CONFIG applies spatial cleanup (erode/dilate, fill_holes, etc.) after stabilization.

Input:  MASK [B, H, W] — per-frame segmentation masks (e.g. from SAM3)
Output: MASK [B, H, W] — temporally stabilized masks
"""

import torch
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, gaussian_filter1d

from .bbox_ops import compute_union_bbox
from .mask_ops import mask_erode_dilate, mask_fill_holes, mask_remove_noise, mask_smooth


# =============================================================================
# Stage 1: Temporal Consensus (no-flow median)
# =============================================================================

def compute_temporal_consensus(masks, window_size=5):
    """Temporal median consensus — simple and effective temporal smoothing.

    For each frame, takes the median across a temporal window of neighboring frames.
    This eliminates outlier frames ("pops") without requiring optical flow.

    Args:
        masks: [B, H, W] float tensor
        window_size: radius of temporal window

    Returns:
        consensus: [B, H, W] float tensor
    """
    B, H, W = masks.shape
    if B <= 1:
        return masks.clone()

    masks_np = masks.cpu().numpy()
    consensus = np.zeros_like(masks_np)

    for t in range(B):
        lo = max(0, t - window_size)
        hi = min(B, t + window_size + 1)
        consensus[t] = np.median(masks_np[lo:hi], axis=0)

    return torch.from_numpy(consensus).to(masks.device)


# =============================================================================
# Stage 2: SDF Temporal Smoothing
# =============================================================================

def mask_to_sdf(mask_np, narrow_band=0):
    """Convert a binary mask to a signed distance function.

    Positive outside, negative inside. The zero-crossing is the mask boundary.

    Args:
        mask_np: [H, W] numpy array, values in [0, 1]
        narrow_band: if >0, clamp SDF to [-band, +band] for efficiency

    Returns:
        sdf: [H, W] numpy float32 array
    """
    binary = (mask_np > 0.5).astype(np.float64)

    # Distance from outside to boundary
    dist_outside = distance_transform_edt(1.0 - binary)
    # Distance from inside to boundary
    dist_inside = distance_transform_edt(binary)

    sdf = dist_outside - dist_inside  # positive outside, negative inside

    if narrow_band > 0:
        sdf = np.clip(sdf, -narrow_band, narrow_band)

    return sdf.astype(np.float32)


def sdf_to_mask(sdf):
    """Convert SDF back to mask. Zero-crossing becomes the boundary.

    Uses a steep sigmoid so the output is near-binary (solid white inside,
    solid black outside) with only a ~1px transition at the boundary.
    The steepness (scale=5.0) means pixels >1px inside are already >0.99.
    """
    # Steep sigmoid: scale=5.0 means 1px from boundary -> 0.993 or 0.007
    # This prevents the soft gradient artifacts that a gentle sigmoid creates
    return 1.0 / (1.0 + np.exp(sdf * 5.0))


def temporal_sdf_smooth(masks, sigma_temporal=2.0, sigma_spatial=1.0, narrow_band=64):
    """Smooth masks via SDF representation — the core quality stage.

    1. Convert each frame's mask to a signed distance function
    2. Apply Gaussian smoothing along temporal axis in SDF space
    3. Optional light spatial smoothing to clean SDF noise
    4. Convert back to masks via sigmoid at zero-crossing

    Smoothing in SDF space is mathematically superior to smoothing binary masks:
    - SDFs are continuous and smooth by nature
    - Temporal filtering interpolates boundary positions, not binary values
    - Topology changes (splits, merges) are handled gracefully
    - No sampling artifacts from contour extraction

    Args:
        masks: [B, H, W] float tensor
        sigma_temporal: Gaussian sigma along time axis (frames). Higher = more smoothing.
        sigma_spatial: Gaussian sigma for per-frame SDF cleanup. 0 = skip.
        narrow_band: SDF clamping distance (pixels). Limits computation to boundary region.

    Returns:
        smoothed: [B, H, W] float tensor
    """
    B, H, W = masks.shape
    masks_np = masks.cpu().numpy()

    # Step 1: Convert all frames to SDF
    sdf_volume = np.zeros((B, H, W), dtype=np.float32)
    for t in range(B):
        sdf_volume[t] = mask_to_sdf(masks_np[t], narrow_band=narrow_band)

    # Step 2: Temporal Gaussian smoothing in SDF space
    if sigma_temporal > 0 and B > 1:
        sdf_volume = gaussian_filter1d(sdf_volume, sigma=sigma_temporal, axis=0, mode="nearest")

    # Step 3: Optional light spatial smoothing
    if sigma_spatial > 0:
        for t in range(B):
            sdf_volume[t] = gaussian_filter(sdf_volume[t], sigma=sigma_spatial, mode="nearest")

    # Step 4: Convert back to masks via sigmoid at zero-crossing
    result = np.zeros((B, H, W), dtype=np.float32)
    for t in range(B):
        result[t] = sdf_to_mask(sdf_volume[t])

    return torch.from_numpy(result).to(masks.device)


# =============================================================================
# Bbox Crop/Paste Helpers
# =============================================================================

def crop_tensors(masks, images, bbox):
    """Crop mask and image tensors to a bounding box region."""
    x, y, w, h = bbox
    cropped_masks = masks[:, y:y + h, x:x + w].clone()
    cropped_images = None
    if images is not None:
        cropped_images = images[:, y:y + h, x:x + w, :].clone()
    return cropped_masks, cropped_images


def paste_masks(result_full, cropped_result, bbox):
    """Paste cropped stabilized masks back into full-frame masks."""
    x, y, w, h = bbox
    result_full[:, y:y + h, x:x + w] = cropped_result
    return result_full


# =============================================================================
# Post-Stabilization Spatial Cleanup (from mask config)
# =============================================================================

def apply_spatial_cleanup(masks, erode_dilate=0, fill_holes=0,
                          remove_noise=0, smooth=0):
    """Apply per-frame spatial mask operations after temporal stabilization."""
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

def run_stabilization_pipeline(masks,
                               consensus_window=5, sdf_sigma_temporal=2.0,
                               sdf_sigma_spatial=1.0, sdf_narrow_band=64,
                               enable_sdf=True,
                               output_mode="binary",
                               info_lines=None):
    """Run the temporal mask stabilization pipeline.

    Args:
        masks: [B, H, W] float tensor — input masks
        consensus_window: frames on each side for temporal consensus
        sdf_sigma_temporal: temporal smoothing in SDF space
        sdf_sigma_spatial: spatial SDF cleanup
        sdf_narrow_band: SDF clamping distance in pixels
        enable_sdf: run SDF smoothing stage
        output_mode: "binary" (threshold at 0.5) or "soft" (keep gradients)
        info_lines: list to append diagnostic strings to

    Returns:
        result: [B, H, W] float tensor — stabilized masks
    """
    if info_lines is None:
        info_lines = []

    B, H, W = masks.shape
    info_lines.append(f"[TemporalStabilizer] Input: {B} frames, {H}x{W}")

    # --- Stage 1: Temporal Consensus ---
    if B > 1:
        print(f"[TemporalStabilizer] Stage 1: Temporal median consensus (window={consensus_window})...")
        info_lines.append(f"[Stage 1] Temporal median consensus (window={consensus_window})...")
        consensus = compute_temporal_consensus(masks, window_size=consensus_window)
    else:
        info_lines.append("[Stage 1] Single frame, skipping temporal consensus.")
        consensus = masks.clone()

    # --- Stage 2: SDF Temporal Smoothing ---
    if enable_sdf and B > 1:
        print(f"[TemporalStabilizer] Stage 2: SDF temporal smoothing (sigma_t={sdf_sigma_temporal}, sigma_s={sdf_sigma_spatial})...")
        info_lines.append(f"[Stage 2] SDF temporal smoothing (sigma_t={sdf_sigma_temporal}, sigma_s={sdf_sigma_spatial})...")
        result = temporal_sdf_smooth(consensus, sigma_temporal=sdf_sigma_temporal,
                                     sigma_spatial=sdf_sigma_spatial, narrow_band=sdf_narrow_band)
    elif enable_sdf:
        # Single frame — only spatial SDF smoothing
        info_lines.append("[Stage 2] SDF spatial smoothing (single frame)...")
        result = temporal_sdf_smooth(consensus, sigma_temporal=0, sigma_spatial=sdf_sigma_spatial,
                                     narrow_band=sdf_narrow_band)
    else:
        result = consensus

    # --- Final Output ---
    if output_mode == "binary":
        result = (result > 0.5).float()
        info_lines.append("[Output] Binarized at 0.5 threshold (clean binary masks).")
    else:
        result = result.clamp(0, 1)
        info_lines.append("[Output] Soft output (gradients preserved).")

    info_lines.append(f"[TemporalStabilizer] Done. {len(info_lines)} diagnostic lines.")
    return result


# =============================================================================
# ComfyUI Node
# =============================================================================

class NV_TemporalMaskStabilizer:
    """Temporal mask stabilization for video sequences.

    Eliminates mask "pops" (sudden shape changes) using temporal median consensus
    and signed distance field smoothing.

    Optional bbox_mask crops to the region of interest before processing — faster and
    higher quality since SDF operates on a smaller, more detailed region.

    Optional MASK_PROCESSING_CONFIG applies spatial cleanup after stabilization using
    the same shared settings as InpaintCrop, LatentInpaintCrop, etc.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK", {
                    "tooltip": "Per-frame segmentation masks [B, H, W]. Connect SAM3 or other segmentation output."
                }),
                "consensus_window": ("INT", {
                    "default": 5, "min": 1, "max": 20, "step": 1,
                    "tooltip": "Temporal consensus window radius (frames on each side). Larger = more temporal context but slower."
                }),
                "sdf_sigma_temporal": ("FLOAT", {
                    "default": 1.5, "min": 0.0, "max": 10.0, "step": 0.1,
                    "tooltip": "SDF temporal smoothing strength (Gaussian sigma in frames). Higher = smoother boundaries over time but more shape loss. 0 = skip."
                }),
                "sdf_sigma_spatial": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 5.0, "step": 0.1,
                    "tooltip": "SDF spatial cleanup (Gaussian sigma in pixels). Removes boundary noise. 0 = skip. Keep low to preserve shape."
                }),
                "output_mode": (["binary", "soft"], {
                    "default": "binary",
                    "tooltip": "binary = clean solid masks (threshold at 0.5). soft = preserve gradients (for blend masks)."
                }),
                "crop_padding": ("FLOAT", {
                    "default": 0.15, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Padding around crop region as fraction of bbox size. Only used when bbox_mask is connected."
                }),
            },
            "optional": {
                "bbox_mask": ("MASK", {
                    "tooltip": "Bounding box mask from MaskTrackingBBox. When connected, stabilization runs only on the "
                              "cropped region (faster, higher detail). Result is pasted back to full frame."
                }),
                "mask_config": ("MASK_PROCESSING_CONFIG", {
                    "tooltip": "Shared mask processing config from NV_MaskProcessingConfig. Applies spatial cleanup "
                              "(erode/dilate, fill_holes, remove_noise, smooth) after temporal stabilization."
                }),
                "mask_erode_dilate": ("INT", {
                    "default": 0, "min": -128, "max": 128, "step": 1,
                    "tooltip": "Post-stabilization erosion (<0) or dilation (>0). Overridden by mask_config if connected."
                }),
                "mask_fill_holes": ("INT", {
                    "default": 0, "min": 0, "max": 128, "step": 1,
                    "tooltip": "Post-stabilization hole filling (grey closing). Overridden by mask_config if connected."
                }),
                "mask_remove_noise": ("INT", {
                    "default": 0, "min": 0, "max": 32, "step": 1,
                    "tooltip": "Post-stabilization noise removal (grey opening). Overridden by mask_config if connected."
                }),
                "mask_smooth": ("INT", {
                    "default": 0, "min": 0, "max": 127, "step": 1,
                    "tooltip": "Post-stabilization edge smoothing (binarize + blur). Overridden by mask_config if connected."
                }),
                "enable_sdf": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable SDF temporal smoothing. This is the core quality stage — smooths boundaries in continuous distance space."
                }),
            },
        }

    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("mask", "info")
    FUNCTION = "execute"
    CATEGORY = "NV_Utils/Mask"
    DESCRIPTION = (
        "Temporal mask stabilization. Fixes mask pops using temporal median consensus "
        "and SDF boundary smoothing. "
        "Connect bbox_mask to crop first (faster). Connect mask_config for shared spatial cleanup."
    )

    def execute(self, mask, consensus_window, sdf_sigma_temporal, sdf_sigma_spatial,
                output_mode="binary", crop_padding=0.15,
                bbox_mask=None, mask_config=None,
                mask_erode_dilate=0, mask_fill_holes=0, mask_remove_noise=0, mask_smooth=0,
                enable_sdf=True):

        info_lines = []
        B = mask.shape[0]
        print(f"[NV_TemporalMaskStabilizer] Starting: {B} frames, {mask.shape[1]}x{mask.shape[2]}")

        # --- Resolve mask config (config bus overrides local widgets) ---
        from .mask_processing_config import apply_mask_config
        vals = apply_mask_config(mask_config,
            mask_erode_dilate=mask_erode_dilate,
            mask_fill_holes=mask_fill_holes,
            mask_remove_noise=mask_remove_noise,
            mask_smooth=mask_smooth,
        )
        post_erode_dilate = vals["mask_erode_dilate"]
        post_fill_holes = vals["mask_fill_holes"]
        post_remove_noise = vals["mask_remove_noise"]
        post_smooth = vals["mask_smooth"]

        has_spatial_cleanup = (post_erode_dilate != 0 or post_fill_holes > 0
                               or post_remove_noise > 0 or post_smooth > 0)

        # --- Bbox crop mode ---
        use_crop = bbox_mask is not None
        crop_bbox = None

        if use_crop:
            crop_bbox = compute_union_bbox(bbox_mask, padding_frac=crop_padding)
            if crop_bbox is not None:
                x, y, w, h = crop_bbox
                info_lines.append(f"[Crop] Cropping to bbox region: ({x},{y}) {w}x{h} (padding={crop_padding:.0%})")
                work_masks, _ = crop_tensors(mask, None, crop_bbox)
            else:
                info_lines.append("[Crop] bbox_mask is empty — processing full frame.")
                use_crop = False
                work_masks = mask
        else:
            work_masks = mask

        # --- Run temporal stabilization pipeline on work region ---
        stabilized = run_stabilization_pipeline(
            masks=work_masks,
            consensus_window=consensus_window,
            sdf_sigma_temporal=sdf_sigma_temporal,
            sdf_sigma_spatial=sdf_sigma_spatial,
            sdf_narrow_band=64,
            enable_sdf=enable_sdf,
            output_mode=output_mode,
            info_lines=info_lines,
        )

        # --- Paste back if cropped ---
        if use_crop and crop_bbox is not None:
            # Start with zeros (black outside crop region)
            result = torch.zeros_like(mask)
            result = paste_masks(result, stabilized, crop_bbox)
            info_lines.append(f"[Crop] Pasted stabilized region back to full {mask.shape[1]}x{mask.shape[2]} frame.")
        else:
            result = stabilized

        # --- Post-stabilization spatial cleanup ---
        if has_spatial_cleanup:
            print(f"[NV_TemporalMaskStabilizer] Applying spatial cleanup...")
            cleanup_parts = []
            if post_fill_holes > 0:
                cleanup_parts.append(f"fill_holes={post_fill_holes}")
            if post_remove_noise > 0:
                cleanup_parts.append(f"remove_noise={post_remove_noise}")
            if post_erode_dilate != 0:
                cleanup_parts.append(f"erode_dilate={post_erode_dilate}")
            if post_smooth > 0:
                cleanup_parts.append(f"smooth={post_smooth}")
            info_lines.append(f"[Spatial] Applying post-stabilization cleanup: {', '.join(cleanup_parts)}")

            result = apply_spatial_cleanup(result,
                erode_dilate=post_erode_dilate,
                fill_holes=post_fill_holes,
                remove_noise=post_remove_noise,
                smooth=post_smooth,
            )

            # Re-binarize after spatial cleanup if binary mode
            if output_mode == "binary":
                result = (result > 0.5).float()

        info = "\n".join(info_lines)
        print(f"[NV_TemporalMaskStabilizer] Done. {B} frames processed.")

        return (result, info)


NODE_CLASS_MAPPINGS = {
    "NV_TemporalMaskStabilizer": NV_TemporalMaskStabilizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_TemporalMaskStabilizer": "NV Temporal Mask Stabilizer",
}
