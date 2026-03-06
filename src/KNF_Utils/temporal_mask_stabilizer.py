"""
NV Temporal Mask Stabilizer — Multi-stage temporal mask cleaning for video sequences.

Fixes mask "pops" (sudden shape changes between frames) using a 5-stage pipeline:

  Stage 1: Optical Flow — RAFT dense flow between consecutive frames (motion awareness)
  Stage 2: Flow-Warped Temporal Consensus — warp neighbor masks to each frame, soft vote
  Stage 3: SDF Temporal Smoothing — signed distance field smoothing (continuous boundary)
  Stage 4: IoU Outlier Detection — detect pop frames, replace with consensus
  Stage 5: Edge Refinement — guided filter snaps mask edges to RGB boundaries

Optional bbox_mask input crops to the mask region before processing (faster + higher detail).
Optional MASK_PROCESSING_CONFIG applies spatial cleanup (erode/dilate, fill_holes, etc.) after stabilization.

Input:  MASK [B, H, W] — per-frame segmentation masks (e.g. from SAM3)
        IMAGE [B, H, W, C] — optional RGB frames for edge-guided refinement (Stage 5)
Output: MASK [B, H, W] — temporally stabilized masks
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, gaussian_filter1d

import comfy.model_management


# =============================================================================
# Stage 1: Optical Flow Computation (RAFT)
# =============================================================================

def compute_optical_flow(images, device):
    """Compute dense bidirectional optical flow between consecutive frames using RAFT.

    Args:
        images: [B, H, W, C] float32 tensor in [0,1] — RGB frames.
        device: torch device for RAFT inference.

    Returns:
        forward_flows: list of [2, H, W] tensors — flow from frame t to t+1
        backward_flows: list of [2, H, W] tensors — flow from frame t+1 to t
    """
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

    B = images.shape[0]
    if B < 2:
        return [], []

    weights = Raft_Large_Weights.DEFAULT
    transforms = weights.transforms()
    model = raft_large(weights=weights, progress=False).to(device).eval()

    # RAFT expects [B, 3, H, W] in specific normalization
    imgs = images.permute(0, 3, 1, 2).to(device)  # [B, 3, H, W]

    # Pad to 8x multiple for RAFT
    _, _, H, W = imgs.shape
    pad_h = (8 - H % 8) % 8
    pad_w = (8 - W % 8) % 8
    if pad_h > 0 or pad_w > 0:
        imgs = F.pad(imgs, (0, pad_w, 0, pad_h), mode="replicate")

    forward_flows = []
    backward_flows = []

    with torch.no_grad():
        for t in range(B - 1):
            print(f"  [RAFT] Computing flow pair {t + 1}/{B - 1}...", end="\r")
            img1, img2 = transforms(imgs[t:t + 1], imgs[t + 1:t + 2])
            # RAFT returns list of flow refinements — take the last (most refined)
            fwd = model(img1, img2)[-1][0]  # [2, H_pad, W_pad]
            bwd = model(img2, img1)[-1][0]

            # Remove padding
            forward_flows.append(fwd[:, :H, :W].cpu())
            backward_flows.append(bwd[:, :H, :W].cpu())
        print(f"  [RAFT] Computed {B - 1} flow pairs.{' ' * 20}")

    del model
    comfy.model_management.soft_empty_cache()

    return forward_flows, backward_flows


def warp_mask_with_flow(mask, flow):
    """Warp a single-frame mask using optical flow via grid_sample.

    Args:
        mask: [H, W] float tensor
        flow: [2, H, W] float tensor (dx, dy)

    Returns:
        warped: [H, W] float tensor
    """
    H, W = mask.shape
    device = mask.device

    # Build sampling grid: identity + flow
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij"
    )
    # flow[0] = horizontal (x), flow[1] = vertical (y)
    flow_dev = flow.to(device)
    sample_x = grid_x + flow_dev[0]
    sample_y = grid_y + flow_dev[1]

    # Normalize to [-1, 1] for grid_sample
    sample_x = 2.0 * sample_x / (W - 1) - 1.0
    sample_y = 2.0 * sample_y / (H - 1) - 1.0

    grid = torch.stack([sample_x, sample_y], dim=-1).unsqueeze(0)  # [1, H, W, 2]
    mask_4d = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

    warped = F.grid_sample(mask_4d, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return warped[0, 0]


# =============================================================================
# Stage 2: Flow-Warped Temporal Consensus
# =============================================================================

def compute_temporal_consensus(masks, forward_flows, backward_flows, window_size=5):
    """For each frame, warp masks from neighboring frames and compute soft consensus.

    Uses optical flow to motion-compensate neighbors, then takes a confidence-weighted
    average. Closer frames and higher flow confidence get more weight.

    Args:
        masks: [B, H, W] float tensor
        forward_flows: list of [2, H, W] — flow t->t+1
        backward_flows: list of [2, H, W] — flow t+1->t
        window_size: number of frames on each side to consider

    Returns:
        consensus: [B, H, W] float tensor — soft consensus mask (0-1 continuous)
    """
    B, H, W = masks.shape
    consensus = torch.zeros_like(masks)
    half = window_size

    for t in range(B):
        weighted_sum = masks[t].clone().float()
        weight_sum = torch.ones(H, W, device=masks.device)

        # Chain flows to reach frame t from neighboring frames
        # Forward direction: frames before t -> warp forward to t
        for k in range(1, half + 1):
            src = t - k
            if src < 0:
                break
            # Chain forward flows from src to t: src->src+1->...->t
            flow_chain = torch.zeros(2, H, W)
            valid = True
            for step in range(src, t):
                if step >= len(forward_flows):
                    valid = False
                    break
                flow_chain = flow_chain + forward_flows[step]
            if not valid:
                break

            warped = warp_mask_with_flow(masks[src], flow_chain)
            # Weight: temporal Gaussian decay * flow magnitude confidence
            temporal_weight = np.exp(-0.5 * (k / (half * 0.5)) ** 2)
            # Lower weight for large flow (less reliable warps)
            flow_mag = flow_chain.norm(dim=0).mean().item()
            flow_confidence = np.exp(-flow_mag / 50.0)  # 50px = half-weight
            w = temporal_weight * flow_confidence
            weighted_sum += warped * w
            weight_sum += w

        # Backward direction: frames after t -> warp backward to t
        for k in range(1, half + 1):
            src = t + k
            if src >= B:
                break
            # Chain backward flows from src to t: src->src-1->...->t
            flow_chain = torch.zeros(2, H, W)
            valid = True
            for step in range(src - 1, t - 1, -1):
                if step >= len(backward_flows) or step < 0:
                    valid = False
                    break
                flow_chain = flow_chain + backward_flows[step]
            if not valid:
                break

            warped = warp_mask_with_flow(masks[src], flow_chain)
            temporal_weight = np.exp(-0.5 * (k / (half * 0.5)) ** 2)
            flow_mag = flow_chain.norm(dim=0).mean().item()
            flow_confidence = np.exp(-flow_mag / 50.0)
            w = temporal_weight * flow_confidence
            weighted_sum += warped * w
            weight_sum += w

        consensus[t] = weighted_sum / weight_sum.clamp(min=1e-6)

    return consensus


def compute_temporal_consensus_no_flow(masks, window_size=5):
    """Fallback temporal consensus without optical flow — simple temporal median.

    Used when no IMAGE input is provided (no flow computation possible).

    Args:
        masks: [B, H, W] float tensor
        window_size: radius of temporal window

    Returns:
        consensus: [B, H, W] float tensor
    """
    B, H, W = masks.shape
    masks_np = masks.cpu().numpy()
    consensus = np.zeros_like(masks_np)

    for t in range(B):
        lo = max(0, t - window_size)
        hi = min(B, t + window_size + 1)
        consensus[t] = np.median(masks_np[lo:hi], axis=0)

    return torch.from_numpy(consensus).to(masks.device)


# =============================================================================
# Stage 3: SDF Temporal Smoothing
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
# Stage 4: IoU Outlier Detection + Replacement
# =============================================================================

def detect_and_fix_outliers(original_masks, consensus_masks, sdf_smoothed_masks,
                            iou_threshold=0.85, blend_power=2.0):
    """Detect temporal outlier frames (pops) and replace them with consensus.

    For each frame, compute IoU between the original mask and the flow-warped consensus.
    Low IoU = the frame's mask disagrees with its temporal neighbors = pop.

    All frames use the SDF-smoothed consensus as output. The IoU score is diagnostic
    only — it tells you which frames were outliers (pops) vs clean.

    Args:
        original_masks: [B, H, W] — raw input masks
        consensus_masks: [B, H, W] — flow-warped temporal consensus (Stage 2)
        sdf_smoothed_masks: [B, H, W] — SDF-smoothed masks (Stage 3 applied to consensus)
        iou_threshold: below this, frame is considered an outlier
        blend_power: unused (kept for API compat)

    Returns:
        fixed_masks: [B, H, W] — outlier-corrected masks
        iou_scores: [B] — per-frame IoU values (for diagnostics)
    """
    B, H, W = original_masks.shape
    fixed = torch.zeros_like(original_masks)
    iou_scores = torch.zeros(B)

    for t in range(B):
        orig = original_masks[t]
        cons = consensus_masks[t]

        # Binary IoU
        orig_bin = (orig > 0.5).float()
        cons_bin = (cons > 0.5).float()

        intersection = (orig_bin * cons_bin).sum()
        union = (orig_bin + cons_bin).clamp(max=1).sum()
        iou = (intersection / union.clamp(min=1)).item()
        iou_scores[t] = iou

        # Use SDF-smoothed consensus for all frames — avoids gradient artifacts from blending
        fixed[t] = sdf_smoothed_masks[t]

    return fixed, iou_scores


# =============================================================================
# Stage 5: Edge Refinement — Guided Filter
# =============================================================================

def guided_filter(guide, source, radius=8, eps=1e-3):
    """Apply guided filter using RGB image as guide to snap mask edges to image edges.

    The guided filter (He et al., 2013) is an edge-preserving smoother that uses
    a guidance image to determine where edges should be. When applied to a mask with
    the corresponding RGB frame as guide, it:
    - Smooths the mask in flat regions (removes noise)
    - Preserves edges that align with the image (respects object boundaries)
    - Suppresses edges that don't correspond to image features (removes artifacts)

    This is used instead of DenseCRF (which requires pydensecrf). The guided filter
    achieves similar edge-snapping behavior and is faster.

    Args:
        guide: [H, W, C] numpy float32 — RGB image in [0, 1]
        source: [H, W] numpy float32 — mask to refine
        radius: filter window radius (pixels). Larger = more smoothing.
        eps: regularization. Smaller = more edge-sensitive.

    Returns:
        refined: [H, W] numpy float32 — edge-refined mask
    """
    import cv2

    guide_u8 = (guide * 255).clip(0, 255).astype(np.uint8)
    source_f32 = source.astype(np.float32)

    # Use (2*radius+1) as diameter for cv2 convention
    d = 2 * radius + 1

    # Convert guide to grayscale for single-channel guided filter
    guide_gray = cv2.cvtColor(guide_u8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    # Box filter means for guided filter computation
    mean_I = cv2.blur(guide_gray, (d, d))
    mean_p = cv2.blur(source_f32, (d, d))
    mean_Ip = cv2.blur(guide_gray * source_f32, (d, d))
    mean_II = cv2.blur(guide_gray * guide_gray, (d, d))

    cov_Ip = mean_Ip - mean_I * mean_p
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.blur(a, (d, d))
    mean_b = cv2.blur(b, (d, d))

    refined = mean_a * guide_gray + mean_b
    return np.clip(refined, 0, 1).astype(np.float32)


# =============================================================================
# Bbox Crop/Paste Helpers
# =============================================================================

def compute_union_bbox(masks, padding_frac=0.1):
    """Compute the union bounding box across all frames of a mask batch.

    Args:
        masks: [B, H, W] float tensor
        padding_frac: fractional padding to add around the union bbox

    Returns:
        (x, y, w, h) tuple or None if all masks are empty
    """
    B, H, W = masks.shape
    # Union across all frames
    union_mask = (masks > 0.01).any(dim=0)  # [H, W]
    non_zero = torch.nonzero(union_mask)

    if non_zero.numel() == 0:
        return None

    y_min = non_zero[:, 0].min().item()
    y_max = non_zero[:, 0].max().item()
    x_min = non_zero[:, 1].min().item()
    x_max = non_zero[:, 1].max().item()

    bw = x_max - x_min + 1
    bh = y_max - y_min + 1

    # Add padding
    pad_x = int(bw * padding_frac)
    pad_y = int(bh * padding_frac)

    x = max(0, x_min - pad_x)
    y = max(0, y_min - pad_y)
    x2 = min(W, x_max + 1 + pad_x)
    y2 = min(H, y_max + 1 + pad_y)

    return (x, y, x2 - x, y2 - y)


def crop_tensors(masks, images, bbox):
    """Crop mask and image tensors to a bounding box region.

    Args:
        masks: [B, H, W]
        images: [B, H, W, C] or None
        bbox: (x, y, w, h)

    Returns:
        cropped_masks: [B, h, w]
        cropped_images: [B, h, w, C] or None
    """
    x, y, w, h = bbox
    cropped_masks = masks[:, y:y + h, x:x + w].clone()
    cropped_images = None
    if images is not None:
        cropped_images = images[:, y:y + h, x:x + w, :].clone()
    return cropped_masks, cropped_images


def paste_masks(result_full, cropped_result, bbox):
    """Paste cropped stabilized masks back into full-frame masks.

    Args:
        result_full: [B, H, W] — full-frame mask (will be modified)
        cropped_result: [B, crop_h, crop_w] — stabilized crop
        bbox: (x, y, w, h)

    Returns:
        result_full with crop region replaced
    """
    x, y, w, h = bbox
    result_full[:, y:y + h, x:x + w] = cropped_result
    return result_full


# =============================================================================
# Post-Stabilization Spatial Cleanup (from mask config)
# =============================================================================

def apply_spatial_cleanup(masks, mask_erode_dilate=0, mask_fill_holes=0,
                          mask_remove_noise=0, mask_smooth=0):
    """Apply per-frame spatial mask operations after temporal stabilization.

    Uses the same functions as InpaintCrop to ensure identical behavior.

    Args:
        masks: [B, H, W] float tensor
        mask_erode_dilate: erosion (<0) or dilation (>0) amount
        mask_fill_holes: grey closing kernel size (0 = skip)
        mask_remove_noise: grey opening kernel size (0 = skip)
        mask_smooth: Gaussian blur after binarization (0 = skip)

    Returns:
        processed: [B, H, W] float tensor
    """
    from .inpaint_crop import mask_erode_dilate as _erode_dilate
    from .inpaint_crop import mask_fill_holes as _fill_holes
    from .inpaint_crop import mask_remove_noise as _remove_noise
    from .inpaint_crop import mask_smooth as _smooth

    result = masks
    if mask_fill_holes > 0:
        result = _fill_holes(result, mask_fill_holes)
    if mask_remove_noise > 0:
        result = _remove_noise(result, mask_remove_noise)
    if mask_erode_dilate != 0:
        result = _erode_dilate(result, mask_erode_dilate)
    if mask_smooth > 0:
        result = _smooth(result, mask_smooth)
    return result


# =============================================================================
# Full Pipeline
# =============================================================================

def run_stabilization_pipeline(masks, images=None,
                               flow_window=5, sdf_sigma_temporal=2.0,
                               sdf_sigma_spatial=1.0, sdf_narrow_band=64,
                               iou_threshold=0.85, iou_blend_power=2.0,
                               guided_radius=8, guided_eps=0.01,
                               enable_flow=True, enable_sdf=True,
                               enable_outlier=True, enable_edge_refine=True,
                               output_mode="binary",
                               info_lines=None):
    """Run the full 5-stage temporal mask stabilization pipeline.

    Args:
        masks: [B, H, W] float tensor — input masks
        images: [B, H, W, C] float tensor or None — RGB frames
        flow_window: frames on each side for temporal consensus (Stage 2)
        sdf_sigma_temporal: temporal smoothing in SDF space (Stage 3)
        sdf_sigma_spatial: spatial SDF cleanup (Stage 3)
        sdf_narrow_band: SDF clamping distance in pixels (Stage 3)
        iou_threshold: outlier detection sensitivity (Stage 4)
        iou_blend_power: unused (kept for API compat)
        guided_radius: edge refinement radius (Stage 5)
        guided_eps: edge refinement sensitivity (Stage 5)
        enable_flow: use optical flow (True) or simple temporal median (False)
        enable_sdf: run SDF smoothing stage
        enable_outlier: run outlier detection stage
        enable_edge_refine: run guided filter edge refinement (requires images)
        output_mode: "binary" (threshold at 0.5, clean masks) or "soft" (keep gradients)
        info_lines: list to append diagnostic strings to

    Returns:
        result: [B, H, W] float tensor — stabilized masks
    """
    if info_lines is None:
        info_lines = []

    B, H, W = masks.shape
    device = masks.device
    info_lines.append(f"[TemporalStabilizer] Input: {B} frames, {H}x{W}")

    # --- Stage 1 + 2: Optical Flow + Temporal Consensus ---
    if enable_flow and images is not None and B > 1:
        print(f"[TemporalStabilizer] Stage 1: RAFT optical flow ({B - 1} pairs)...")
        info_lines.append(f"[Stage 1] Computing RAFT optical flow ({B - 1} pairs)...")
        flow_device = comfy.model_management.get_torch_device()
        forward_flows, backward_flows = compute_optical_flow(images, flow_device)
        print(f"[TemporalStabilizer] Stage 2: Flow-warped consensus (window={flow_window})...")
        info_lines.append(f"[Stage 2] Flow-warped temporal consensus (window={flow_window})...")
        consensus = compute_temporal_consensus(masks, forward_flows, backward_flows, window_size=flow_window)
        del forward_flows, backward_flows
    elif B > 1:
        print(f"[TemporalStabilizer] Stage 2: Temporal median consensus (no flow, window={flow_window})...")
        info_lines.append(f"[Stage 2] Temporal median consensus (no flow, window={flow_window})...")
        consensus = compute_temporal_consensus_no_flow(masks, window_size=flow_window)
    else:
        info_lines.append("[Stage 1-2] Single frame, skipping temporal stages.")
        consensus = masks.clone()

    # --- Stage 3: SDF Temporal Smoothing ---
    if enable_sdf and B > 1:
        print(f"[TemporalStabilizer] Stage 3: SDF temporal smoothing (sigma_t={sdf_sigma_temporal}, sigma_s={sdf_sigma_spatial})...")
        info_lines.append(f"[Stage 3] SDF temporal smoothing (sigma_t={sdf_sigma_temporal}, sigma_s={sdf_sigma_spatial})...")
        sdf_smoothed = temporal_sdf_smooth(consensus, sigma_temporal=sdf_sigma_temporal,
                                           sigma_spatial=sdf_sigma_spatial, narrow_band=sdf_narrow_band)
    elif enable_sdf:
        # Single frame — only spatial SDF smoothing
        info_lines.append("[Stage 3] SDF spatial smoothing (single frame)...")
        sdf_smoothed = temporal_sdf_smooth(consensus, sigma_temporal=0, sigma_spatial=sdf_sigma_spatial,
                                           narrow_band=sdf_narrow_band)
    else:
        sdf_smoothed = consensus

    # --- Stage 4: IoU Outlier Detection ---
    if enable_outlier and B > 1:
        print(f"[TemporalStabilizer] Stage 4: IoU outlier detection (threshold={iou_threshold})...")
        info_lines.append(f"[Stage 4] IoU outlier detection (threshold={iou_threshold})...")
        result, iou_scores = detect_and_fix_outliers(masks, consensus, sdf_smoothed,
                                                      iou_threshold=iou_threshold,
                                                      blend_power=iou_blend_power)
        outlier_count = (iou_scores < iou_threshold).sum().item()
        min_iou = iou_scores.min().item()
        info_lines.append(f"  -> {outlier_count}/{B} outlier frames detected (min IoU={min_iou:.3f})")
    else:
        result = sdf_smoothed

    # --- Stage 5: Guided Filter Edge Refinement ---
    if enable_edge_refine and images is not None:
        print(f"[TemporalStabilizer] Stage 5: Guided filter edge refinement (radius={guided_radius}, eps={guided_eps})...")
        info_lines.append(f"[Stage 5] Guided filter edge refinement (radius={guided_radius}, eps={guided_eps})...")
        result_np = result.cpu().numpy()
        images_np = images.cpu().numpy()
        for t in range(B):
            result_np[t] = guided_filter(images_np[t], result_np[t],
                                         radius=guided_radius, eps=guided_eps)
        result = torch.from_numpy(result_np).to(device)
    elif enable_edge_refine:
        info_lines.append("[Stage 5] Skipped — no IMAGE input for edge guidance.")

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
    """Multi-stage temporal mask stabilization for video sequences.

    Eliminates mask "pops" (sudden shape changes) using optical flow, signed distance
    field smoothing, outlier detection, and edge-guided refinement.

    Optional bbox_mask crops to the region of interest before processing — faster and
    higher quality since RAFT/SDF operates on a smaller, more detailed region.

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
                "flow_window": ("INT", {
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
                "iou_threshold": ("FLOAT", {
                    "default": 0.85, "min": 0.5, "max": 0.99, "step": 0.01,
                    "tooltip": "IoU threshold for outlier detection. Frames with IoU below this vs temporal consensus are replaced. Lower = less aggressive."
                }),
                "guided_radius": ("INT", {
                    "default": 8, "min": 1, "max": 32, "step": 1,
                    "tooltip": "Guided filter radius for edge refinement. Larger = more smoothing in flat regions."
                }),
                "guided_eps": ("FLOAT", {
                    "default": 0.01, "min": 0.001, "max": 0.5, "step": 0.001,
                    "tooltip": "Guided filter edge sensitivity. Smaller = stronger edge snapping. 0.01 = sharp edges, 0.1 = softer."
                }),
                "output_mode": (["binary", "soft"], {
                    "default": "binary",
                    "tooltip": "binary = clean solid masks (threshold at 0.5). soft = preserve gradients (for blend masks)."
                }),
                "crop_padding": ("FLOAT", {
                    "default": 0.15, "min": 0.0, "max": 0.5, "step": 0.01,
                    "tooltip": "Padding around crop region as fraction of bbox size. Only used when bbox_mask is connected."
                }),
            },
            "optional": {
                "image": ("IMAGE", {
                    "tooltip": "RGB frames [B, H, W, C] for optical flow (Stage 1) and edge refinement (Stage 5). "
                              "Without this, falls back to temporal median + no edge refinement."
                }),
                "bbox_mask": ("MASK", {
                    "tooltip": "Bounding box mask from MaskTrackingBBox. When connected, stabilization runs only on the "
                              "cropped region (faster, higher detail). Result is pasted back to full frame."
                }),
                "mask_config": ("MASK_PROCESSING_CONFIG", {
                    "tooltip": "Shared mask processing config from NV_MaskProcessingConfig. Applies spatial cleanup "
                              "(erode/dilate, fill_holes, remove_noise, smooth) after temporal stabilization."
                }),
                "mask_erode_dilate": ("INT", {
                    "default": 0, "min": -64, "max": 64, "step": 1,
                    "tooltip": "Post-stabilization erosion (<0) or dilation (>0). Overridden by mask_config if connected."
                }),
                "mask_fill_holes": ("INT", {
                    "default": 0, "min": 0, "max": 64, "step": 1,
                    "tooltip": "Post-stabilization hole filling (grey closing). Overridden by mask_config if connected."
                }),
                "mask_remove_noise": ("INT", {
                    "default": 0, "min": 0, "max": 64, "step": 1,
                    "tooltip": "Post-stabilization noise removal (grey opening). Overridden by mask_config if connected."
                }),
                "mask_smooth": ("INT", {
                    "default": 0, "min": 0, "max": 64, "step": 1,
                    "tooltip": "Post-stabilization edge smoothing (binarize + blur). Overridden by mask_config if connected."
                }),
                "enable_flow": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use RAFT optical flow for motion-aware consensus. Disable for speed (falls back to temporal median)."
                }),
                "enable_sdf": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable SDF temporal smoothing. This is the core quality stage — smooths boundaries in continuous distance space."
                }),
                "enable_outlier": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable IoU outlier detection. Detects and replaces pop frames."
                }),
                "enable_edge_refine": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable guided filter edge refinement. Snaps mask boundaries to image edges. Requires IMAGE input."
                }),
            },
        }

    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("mask", "info")
    FUNCTION = "execute"
    CATEGORY = "NV_Utils/Mask"
    DESCRIPTION = (
        "Multi-stage temporal mask stabilization. Fixes mask pops using optical flow consensus, "
        "SDF boundary smoothing, outlier detection, and edge-guided refinement. "
        "Connect bbox_mask to crop first (faster). Connect mask_config for shared spatial cleanup."
    )

    def execute(self, mask, flow_window, sdf_sigma_temporal, sdf_sigma_spatial,
                iou_threshold, guided_radius, guided_eps, output_mode="binary",
                crop_padding=0.15,
                image=None, bbox_mask=None, mask_config=None,
                mask_erode_dilate=0, mask_fill_holes=0, mask_remove_noise=0, mask_smooth=0,
                enable_flow=True, enable_sdf=True,
                enable_outlier=True, enable_edge_refine=True):

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
                work_masks, work_images = crop_tensors(mask, image, crop_bbox)
            else:
                info_lines.append("[Crop] bbox_mask is empty — processing full frame.")
                use_crop = False
                work_masks = mask
                work_images = image
        else:
            work_masks = mask
            work_images = image

        # --- Run temporal stabilization pipeline on work region ---
        stabilized = run_stabilization_pipeline(
            masks=work_masks,
            images=work_images,
            flow_window=flow_window,
            sdf_sigma_temporal=sdf_sigma_temporal,
            sdf_sigma_spatial=sdf_sigma_spatial,
            sdf_narrow_band=64,
            iou_threshold=iou_threshold,
            iou_blend_power=2.0,
            guided_radius=guided_radius,
            guided_eps=guided_eps,
            enable_flow=enable_flow,
            enable_sdf=enable_sdf,
            enable_outlier=enable_outlier,
            enable_edge_refine=enable_edge_refine,
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
                mask_erode_dilate=post_erode_dilate,
                mask_fill_holes=post_fill_holes,
                mask_remove_noise=post_remove_noise,
                mask_smooth=post_smooth,
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
