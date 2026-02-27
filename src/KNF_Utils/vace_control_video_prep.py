"""
NV VACE Control Video Prep — Single-node VACE control video + mask preparation.

Replaces the multi-node pipeline of: Image To Mask → InvertMask → VaceMaskPrep → ImageRemoveAlpha
                                      Image To Mask → VaceMaskPrep → WanVaceToVideo

The problem with separate nodes: two VaceMaskPrep instances process different masks
(original vs inverted), producing non-complementary results. Erosion on the inverted
mask EXPANDS the grey zone, while erosion on the original mask SHRINKS the VACE zone,
creating an N-pixel gap where grey leaks into the "preserve" channel → visible halo.

This node processes the mask ONCE, then uses that same result for both:
  1. Compositing the control video (fill masked area with VACE-neutral color)
  2. Output VACE mask for WanVaceToVideo's control_masks input

Guarantees perfect pixel-level alignment between grey zone and VACE mask boundary.

Mask shape modes:
  - "as_is": Use the input mask directly (tight segmentation mask).
  - "bbox": Convert to per-frame bounding boxes with temporal smoothing. Produces
    cleaner VACE results because: (a) rectangular boundaries align with the VAE's
    8x8 spatial blocks, (b) the mask perimeter is shorter with no irregular notches,
    (c) VACE was trained with bbox masks as a first-class augmentation mode.

Fill modes:
  - "soft": Alpha-composite using the feathered mask. Both VAE channels (inactive/reactive)
    are smooth, but the inactive signal has quadratic attenuation in the transition zone
    [(real-0.5)*(1-m)² instead of (real-0.5)*(1-m)] due to double-masking.
  - "none": No fill — pass through real content. The VACE formula alone handles the
    transition, producing the mathematically ideal linear attenuation. Old face content
    appears in the reactive channel but is attenuated by the mask and overridden by LoRA/prompt.
    Best for face replacement with LoRA where you want the cleanest transition.

The correct VACE neutral value is 0.5 in [0,1] space. WanVaceToVideo centers by
subtracting 0.5, so fill=0.5 → centered=0.0 → both inactive and reactive get pure 0.5
(true neutral). The original #999999 (0.6) produces a slight brightness bias.
"""

import torch
import numpy as np
from scipy.ndimage import distance_transform_edt

from .inpaint_crop import mask_erode_dilate, mask_blur, gaussian_smooth_1d, median_filter_1d


def compute_inscribed_radius(mask: torch.Tensor) -> tuple[list[float], int]:
    """Compute inscribed radius per frame via EDT. Returns (radii_list, min_radius_frame_idx)."""
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)

    radii = []
    for b in range(mask.shape[0]):
        mask_bin = (mask[b] > 0.5).cpu().numpy().astype(np.uint8)
        if np.any(mask_bin):
            dist = distance_transform_edt(mask_bin)
            radii.append(float(np.max(dist)))
        else:
            radii.append(0.0)

    min_idx = int(np.argmin(radii)) if radii else 0
    return radii, min_idx


def _snap_down(val: int, stride: int) -> int:
    """Snap value down to nearest multiple of stride."""
    return (val // stride) * stride


def _snap_up(val: int, stride: int) -> int:
    """Snap value up to nearest multiple of stride."""
    return ((val + stride - 1) // stride) * stride


def masks_to_bboxes(mask: torch.Tensor, padding: float, smooth_window: int,
                    vae_stride: int, info_lines: list) -> torch.Tensor:
    """Convert per-frame masks to temporally-smoothed, padded, VAE-aligned bounding box masks.

    Args:
        mask: [B, H, W] binary/soft mask tensor
        padding: Fractional expansion of bbox dimensions (0.15 = 15%)
        smooth_window: Temporal smoothing window size (0 = disabled)
        vae_stride: VAE spatial stride for boundary alignment
        info_lines: Mutable list for diagnostic output

    Returns:
        [B, H, W] tensor of rectangular masks (binary, 0 or 1)
    """
    B, H, W = mask.shape

    # --- Step 1: Per-frame tight bounding boxes ---
    x1s, y1s, x2s, y2s = [], [], [], []
    present = []

    for b in range(B):
        ys, xs = torch.where(mask[b] > 0.5)
        if len(xs) == 0:
            # Empty frame — mark as absent, fill with placeholder
            present.append(False)
            x1s.append(0.0)
            y1s.append(0.0)
            x2s.append(0.0)
            y2s.append(0.0)
        else:
            present.append(True)
            x1s.append(float(xs.min().item()))
            y1s.append(float(ys.min().item()))
            x2s.append(float(xs.max().item() + 1))
            y2s.append(float(ys.max().item() + 1))

    # --- Step 2: Forward/backward fill empty frames ---
    # So smoothing has valid data everywhere
    coords = [x1s, y1s, x2s, y2s]
    for c in coords:
        # Forward fill
        last_valid = None
        for i in range(B):
            if present[i]:
                last_valid = c[i]
            elif last_valid is not None:
                c[i] = last_valid
        # Backward fill (for leading empty frames)
        last_valid = None
        for i in range(B - 1, -1, -1):
            if present[i]:
                last_valid = c[i]
            elif last_valid is not None:
                c[i] = last_valid

    num_present = sum(present)
    if num_present == 0:
        info_lines.append("  BBox: no mask content in any frame — returning zero masks")
        return torch.zeros_like(mask)

    info_lines.append(f"  BBox: {num_present}/{B} frames with mask content")

    # --- Step 3: Temporal smoothing ---
    if smooth_window > 1 and B > 1:
        # Median filter first (removes outlier frames), then Gaussian smooth
        x1s = median_filter_1d(x1s, min(smooth_window, B))
        y1s = median_filter_1d(y1s, min(smooth_window, B))
        x2s = median_filter_1d(x2s, min(smooth_window, B))
        y2s = median_filter_1d(y2s, min(smooth_window, B))

        x1s = gaussian_smooth_1d(x1s, min(smooth_window, B))
        y1s = gaussian_smooth_1d(y1s, min(smooth_window, B))
        x2s = gaussian_smooth_1d(x2s, min(smooth_window, B))
        y2s = gaussian_smooth_1d(y2s, min(smooth_window, B))

        info_lines.append(f"  BBox smoothing: median + gaussian, window={smooth_window}")

    # --- Step 4: Padding + VAE stride alignment ---
    result = torch.zeros_like(mask)

    bbox_sizes = []
    for b in range(B):
        bx1, by1, bx2, by2 = x1s[b], y1s[b], x2s[b], y2s[b]
        bw = bx2 - bx1
        bh = by2 - by1

        if bw < 1 or bh < 1:
            continue

        # Apply padding (percentage of bbox dimensions)
        pad_x = bw * padding
        pad_y = bh * padding
        bx1 -= pad_x
        by1 -= pad_y
        bx2 += pad_x
        by2 += pad_y

        # Snap to VAE stride boundaries (outward)
        bx1 = max(0, _snap_down(int(bx1), vae_stride))
        by1 = max(0, _snap_down(int(by1), vae_stride))
        bx2 = min(W, _snap_up(int(bx2), vae_stride))
        by2 = min(H, _snap_up(int(by2), vae_stride))

        result[b, by1:by2, bx1:bx2] = 1.0
        bbox_sizes.append((bx2 - bx1, by2 - by1))

    if bbox_sizes:
        avg_w = sum(s[0] for s in bbox_sizes) / len(bbox_sizes)
        avg_h = sum(s[1] for s in bbox_sizes) / len(bbox_sizes)
        info_lines.append(
            f"  BBox size (avg): {avg_w:.0f}x{avg_h:.0f}px "
            f"(padding={padding:.0%}, aligned to {vae_stride}px)"
        )

    return result


class NV_VaceControlVideoPrep:
    """Prepare both control_video and control_masks for WanVaceToVideo in a single node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "mask_shape": (["as_is", "bbox"], {
                    "default": "as_is",
                    "tooltip": "as_is: use input mask directly (tight segmentation). "
                               "bbox: convert to per-frame bounding boxes with temporal smoothing. "
                               "Box masks produce cleaner VACE results because boundaries align with "
                               "VAE 8x8 blocks and match VACE training distribution."
                }),
                "mode": (["auto", "manual"], {
                    "default": "auto",
                    "tooltip": "auto: analyze mask geometry and derive optimal erosion/feather, "
                               "using widget values as targets (capped by mask safety limits). "
                               "manual: use erosion_blocks and feather_blocks directly."
                }),
                "erosion_blocks": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 4.0, "step": 0.25,
                    "tooltip": "Erode mask inward by this many VAE blocks. "
                               "In auto mode, this is the target value capped by mask geometry. "
                               "0.5 blocks = 4px for WAN."
                }),
                "feather_blocks": ("FLOAT", {
                    "default": 2.0, "min": 0.0, "max": 8.0, "step": 0.25,
                    "tooltip": "Feather mask edge over this many VAE blocks. "
                               "2.0 blocks = 16px = 2 full VAE encoding blocks = 1 VACE attention patch. "
                               "In auto mode, this is the target value capped by mask geometry."
                }),
                "fill_mode": (["soft", "none"], {
                    "default": "soft",
                    "tooltip": "soft: alpha-composite fill using feathered mask. Smooth channels, "
                               "mild double-attenuation in transition zone. "
                               "none: no fill, real content everywhere. Mathematically ideal "
                               "inactive channel. Best for LoRA face replacement."
                }),
            },
            "optional": {
                "bbox_padding": ("FLOAT", {
                    "default": 0.15, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "BBox padding as fraction of bbox dimensions (0.15 = 15%). "
                               "Only used when mask_shape='bbox'. VACE training uses 10-20%."
                }),
                "bbox_smooth_frames": ("INT", {
                    "default": 5, "min": 0, "max": 31, "step": 2,
                    "tooltip": "Temporal smoothing window for bbox coordinates (odd values recommended). "
                               "Applies median filter then Gaussian smooth to eliminate jitter. "
                               "0 = disabled. Only used when mask_shape='bbox'."
                }),
                "fill_value": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Fill color for masked area (uniform across RGB). "
                               "0.5 = true VACE neutral (WanVaceToVideo centers by -0.5). "
                               "Only used when fill_mode='soft'."
                }),
                "threshold": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Binarize mask at 0.5 before processing. Enable if mask has "
                               "intermediate values from resizing that should be hard edges."
                }),
                "vae_stride": ("INT", {
                    "default": 8, "min": 4, "max": 32, "step": 4,
                    "tooltip": "VAE spatial stride in pixels (8 for WAN)."
                }),
                "stitch_source": (["tight", "bbox"], {
                    "default": "tight",
                    "tooltip": "Source mask for pixel-space stitching. "
                               "'tight': use original input mask (precise subject boundary, but may "
                               "cut into VACE content at a boundary VACE wasn't aware of). "
                               "'bbox': use the VACE bbox mask eroded inward — hides bbox-edge seam "
                               "artifacts since the blend happens in VACE's clean interior zone. "
                               "Trade-off: pastes everything VACE generated inside the bbox."
                }),
                "stitch_erosion": ("INT", {
                    "default": 0, "min": -32, "max": 32, "step": 1,
                    "tooltip": "Erode (negative) or dilate (positive) the stitch mask for "
                               "pixel-space compositing. Independent of the VACE mask erosion. "
                               "Controls the pixel blend boundary, not the VACE conditioning zone."
                }),
                "stitch_feather": ("INT", {
                    "default": 8, "min": 0, "max": 64, "step": 1,
                    "tooltip": "Feather the stitch mask edges for seamless pixel compositing. "
                               "This controls the pixel-space blend, not the VACE conditioning transition. "
                               "8-16px = subtle, 24-32px = visible softening, 0 = hard edge."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "MASK", "STRING")
    RETURN_NAMES = ("control_video", "control_masks", "stitch_mask", "tight_mask", "info")
    FUNCTION = "execute"
    CATEGORY = "NV_Utils/VACE"
    DESCRIPTION = (
        "Single-node VACE control video + mask preparation with triple-mask output. "
        "control_masks = VACE conditioning mask (bbox, eroded, feathered). "
        "stitch_mask = compositing mask (source selectable: 'tight' for precise subject boundary, "
        "'bbox' for eroded-inward bbox that hides VACE edge seams). "
        "tight_mask = always the original tight mask with stitch erosion/feather applied (backup). "
        "Use stitch_mask with NV_VaceVideoStitch for pixel compositing."
    )

    def execute(self, image, mask, mask_shape, mode, erosion_blocks, feather_blocks, fill_mode,
                bbox_padding=0.15, bbox_smooth_frames=5,
                fill_value=0.5, threshold=False, vae_stride=8,
                stitch_source="tight", stitch_erosion=0, stitch_feather=8):
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        info_lines = [
            f"[NV_VaceControlVideoPrep] shape={mask_shape} | mode={mode} | "
            f"fill={fill_mode} | vae_stride={vae_stride}px"
        ]

        # --- Edge case: trivial masks ---
        mask_min = mask.min().item()
        mask_max = mask.max().item()

        if mask_max < 0.01:
            info_lines.append("  Mask is all zeros — passing through unchanged")
            info = "\n".join(info_lines)
            print(info)
            return (image, mask, mask, mask, info)

        if mask_min > 0.99:
            fill = torch.full_like(image, fill_value) if fill_mode == "soft" else image
            info_lines.append("  Mask is all ones — full fill applied")
            info = "\n".join(info_lines)
            print(info)
            return (fill, mask, mask, mask, info)

        result_mask = mask.clone()

        # --- Step 1: Threshold ---
        if threshold:
            result_mask = (result_mask > 0.5).float()
            info_lines.append("  Threshold: applied at 0.5")

        # --- Step 2: BBox conversion (before erosion/feather) ---
        if mask_shape == "bbox":
            result_mask = masks_to_bboxes(
                result_mask, bbox_padding, bbox_smooth_frames,
                vae_stride, info_lines
            )

            # Re-check for trivial after bbox conversion
            if result_mask.max().item() < 0.01:
                info = "\n".join(info_lines)
                print(info)
                return (image, result_mask, mask, mask, info)

        # --- Step 3: Analyze mask geometry ---
        radii, min_frame = compute_inscribed_radius(result_mask)
        min_radius = radii[min_frame] if radii else 0.0
        radius_blocks = min_radius / vae_stride

        info_lines.append(
            f"  Inscribed radius: {min_radius:.1f}px ({radius_blocks:.1f} VAE blocks)"
            f" — min at frame {min_frame}/{len(radii)}"
        )

        # --- Step 4: Compute erosion/feather pixel values ---
        if mode == "auto":
            # Use widget values as TARGETS, capped by mask geometry safety limits
            erosion_target = erosion_blocks * vae_stride
            erosion_max_safe = max(1.0, min_radius / 3.0)
            erosion_px = int(round(min(erosion_target, erosion_max_safe)))

            feather_target = feather_blocks * vae_stride
            feather_max_safe = max(float(vae_stride), min_radius / 2.0)
            feather_px = int(round(min(feather_target, feather_max_safe)))

            info_lines.append(
                f"  Erosion (auto): {erosion_px}px "
                f"— target: {erosion_target:.0f}px, max safe: {erosion_max_safe:.0f}px"
            )
            info_lines.append(
                f"  Feather (auto): {feather_px}px "
                f"— target: {feather_target:.0f}px, max safe: {feather_max_safe:.0f}px"
            )

            if min_radius < vae_stride:
                info_lines.append(
                    f"  WARNING: Mask thinner than 1 VAE block ({min_radius:.1f}px < {vae_stride}px). "
                    f"Auto values are heavily constrained."
                )

        else:  # manual
            erosion_px = int(round(erosion_blocks * vae_stride))
            feather_px = int(round(feather_blocks * vae_stride))
            info_lines.append(
                f"  Erosion (manual): {erosion_blocks} blocks = {erosion_px}px"
            )
            info_lines.append(
                f"  Feather (manual): {feather_blocks} blocks = {feather_px}px"
            )

        # --- Step 5: Erode ---
        if erosion_px > 0:
            result_mask = mask_erode_dilate(result_mask, -erosion_px)

        # --- Step 6: Blur (feather) ---
        if feather_px > 0:
            result_mask = mask_blur(result_mask, feather_px)
            effective_kernel = feather_px if feather_px % 2 == 1 else feather_px + 1
            info_lines.append(f"  Blur kernel: {effective_kernel}px (odd)")

        # --- Step 7: Clamp ---
        result_mask = result_mask.clamp(0.0, 1.0)

        # --- Step 8: Composite control video ---
        # Expand mask for broadcasting: [B, H, W] → [B, H, W, 1] for IMAGE [B, H, W, C]
        mask_expanded = result_mask.unsqueeze(-1)

        if fill_mode == "soft":
            # Alpha composite: real * (1 - mask) + fill * mask
            control_video = image * (1.0 - mask_expanded) + fill_value * mask_expanded
            info_lines.append(
                f"  Fill: soft composite with value={fill_value:.2f} "
                f"(VACE neutral={'yes' if abs(fill_value - 0.5) < 0.01 else 'NO — 0.5 is neutral'})"
            )
        else:  # none
            control_video = image
            info_lines.append("  Fill: none — real content passed through")

        # Transition summary
        transition_px = erosion_px + feather_px
        transition_blocks = transition_px / vae_stride
        transition_patches = transition_px / (vae_stride * 2)  # VACE patch stride (1,2,2)
        info_lines.append(
            f"  Transition zone: ~{transition_px}px "
            f"({transition_blocks:.1f} VAE blocks, {transition_patches:.1f} VACE attention patches)"
        )

        if fill_mode == "soft" and feather_px > 0:
            info_lines.append(
                "  Note: soft fill + feathered mask creates mild double-attenuation in "
                "transition zone. Switch to fill_mode='none' for linear attenuation."
            )

        # --- Step 9: Build tight mask (always from original input mask) ---
        tight_mask = mask.clone()
        if threshold:
            tight_mask = (tight_mask > 0.5).float()

        # Apply stitch erosion/feather to tight mask
        tight_processed = tight_mask.clone()
        if stitch_erosion != 0:
            tight_processed = mask_erode_dilate(tight_processed, stitch_erosion)
        if stitch_feather > 0:
            tight_processed = mask_blur(tight_processed, stitch_feather)
        tight_processed = tight_processed.clamp(0.0, 1.0)

        # --- Step 10: Build stitch mask (source depends on stitch_source param) ---
        if stitch_source == "bbox" and mask_shape == "bbox":
            # Derive stitch mask from the bbox (result_mask before erosion/feather for VACE).
            # Re-run bbox computation on the pre-erosion mask to get clean binary boxes,
            # then apply stitch_erosion/feather independently.
            # result_mask at this point has VACE erosion+feather applied, so we rebuild
            # from the bbox step output. We can use the binary bbox by thresholding result_mask
            # before VACE feathering was applied — but we already modified result_mask.
            # Instead, recompute bboxes from the original mask.
            bbox_binary = masks_to_bboxes(
                (mask > 0.5).float() if threshold else mask,
                bbox_padding, bbox_smooth_frames, vae_stride, []  # discard info
            )
            stitch_mask = bbox_binary.clone()

            # Erode inward to cut inside the bbox seam
            default_bbox_erosion = vae_stride  # 1 VAE block = 8px default inset
            effective_erosion = stitch_erosion if stitch_erosion != 0 else -default_bbox_erosion
            if effective_erosion != 0:
                stitch_mask = mask_erode_dilate(stitch_mask, effective_erosion)

            if stitch_feather > 0:
                stitch_mask = mask_blur(stitch_mask, stitch_feather)

            stitch_mask = stitch_mask.clamp(0.0, 1.0)

            info_lines.append(
                f"  Stitch mask: source=bbox, erosion={effective_erosion}px "
                f"({'default 1 VAE block inset' if stitch_erosion == 0 else 'manual'}),"
                f" feather={stitch_feather}px"
            )
        else:
            # Use tight mask (original behavior)
            stitch_mask = tight_processed
            if stitch_source == "bbox" and mask_shape != "bbox":
                info_lines.append(
                    "  Stitch source: 'bbox' requested but mask_shape is not 'bbox' "
                    "— falling back to tight mask."
                )
            info_lines.append(
                f"  Stitch mask: source=tight, erosion={stitch_erosion}px, "
                f"feather={stitch_feather}px"
            )

        if stitch_feather == 0:
            info_lines.append(
                "  WARNING: stitch_feather=0 produces hard edges — visible seams likely."
            )

        info = "\n".join(info_lines)
        print(info)

        return (control_video, result_mask, stitch_mask, tight_processed, info)


NODE_CLASS_MAPPINGS = {
    "NV_VaceControlVideoPrep": NV_VaceControlVideoPrep,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_VaceControlVideoPrep": "NV VACE Control Video Prep",
}
