"""
NV Inpaint Crop v2 - Clean, focused inpainting crop node

Crops an image around a masked region for inpainting, with optional bounding box control.

Key features:
- Two mask outputs: original (for tight stitching) and processed (for diffusion)
- bounding_box_mask for explicit crop region control
- Advanced mask processing using scipy grey morphology
- Stitch uses original mask for minimal changes to the image
"""

import math
import torch
import comfy.model_management
import comfy.utils
import nodes

from .mask_ops import (
    mask_erode_dilate as _op_erode_dilate,
    mask_fill_holes as _op_fill_holes,
    mask_remove_noise as _op_remove_noise,
    mask_smooth as _op_smooth,
    mask_blur,
    rescale_image, rescale_mask,
)
from .bbox_ops import find_bbox, detect_bbox_anomalies


# =============================================================================
# WAN Resolution Presets
# =============================================================================

WAN_PRESETS = {
    "WAN_480p": {"pixels": 399_360, "divisor": 16, "min_side": 256, "max_side": 1024},
    "WAN_720p": {"pixels": 921_600, "divisor": 16, "min_side": 384, "max_side": 1536},
}


def compute_auto_resolution(bbox_aspect, preset_name, padding_multiple):
    """
    Compute optimal target resolution from bbox aspect ratio and WAN preset.

    Given a pixel budget and aspect ratio, solves:
        width * height = target_pixels
        width / height = aspect
    => width = sqrt(pixels * aspect), height = sqrt(pixels / aspect)

    Then snaps to the stricter of the preset's divisor or the padding_multiple.
    """
    config = WAN_PRESETS[preset_name]
    pixels = config["pixels"]
    divisor = max(config["divisor"], padding_multiple) if padding_multiple > 0 else config["divisor"]
    min_side = config["min_side"]
    max_side = config["max_side"]

    # Clamp extreme aspect ratios (beyond 3:1 or 1:3 is rarely useful for diffusion)
    bbox_aspect = max(1/3, min(3.0, bbox_aspect))

    # Compute raw dimensions
    raw_w = math.sqrt(pixels * bbox_aspect)
    raw_h = math.sqrt(pixels / bbox_aspect)

    # Snap to divisor
    w = round(raw_w / divisor) * divisor
    h = round(raw_h / divisor) * divisor

    # Clamp to min/max
    w = max(min_side, min(max_side, w))
    h = max(min_side, min(max_side, h))

    # Re-snap after clamp (clamp might have broken divisibility)
    w = round(w / divisor) * divisor
    h = round(h / divisor) * divisor

    return int(w), int(h)


# =============================================================================
# Utility Functions
# =============================================================================

def pad_to_multiple(value, multiple):
    """Round up value to nearest multiple."""
    if multiple <= 0:
        return value
    return int(math.ceil(value / multiple) * multiple)


# =============================================================================
# Core Crop Function
# =============================================================================

def crop_for_inpaint(image, original_mask, processed_mask, bbox_x, bbox_y, bbox_w, bbox_h,
                     target_w, target_h, padding_multiple, resize_algorithm):
    """
    Crop image and masks around bounding box, adjusting for target aspect ratio.

    Returns:
        canvas_image: Expanded image (may be larger than original if bbox doesn't fit)
        cto_x, cto_y, cto_w, cto_h: Where original image sits in canvas
        cropped_image: The cropped and resized region
        cropped_mask_original: Original mask cropped and resized (for tight stitching)
        cropped_mask_processed: Processed mask cropped and resized (for diffusion)
        ctc_x, ctc_y, ctc_w, ctc_h: Where crop region sits in canvas
    """
    B, img_h, img_w, C = image.shape
    device = image.device

    # Ensure target dimensions are multiples of padding
    if padding_multiple > 0:
        target_w = pad_to_multiple(target_w, padding_multiple)
        target_h = pad_to_multiple(target_h, padding_multiple)

    # Calculate target aspect ratio
    target_aspect = target_w / target_h
    bbox_aspect = bbox_w / bbox_h

    # Grow bbox to match target aspect ratio, centered on bbox center.
    # Use float center to avoid ±1px oscillation from integer division.
    bbox_cx = bbox_x + bbox_w / 2.0
    bbox_cy = bbox_y + bbox_h / 2.0

    if bbox_aspect < target_aspect:
        # Need wider - grow width
        new_w = int(round(bbox_h * target_aspect))
        new_h = bbox_h
    else:
        # Need taller - grow height
        new_w = bbox_w
        new_h = int(round(bbox_w / target_aspect))

    new_x = int(round(bbox_cx - new_w / 2.0))
    new_y = int(round(bbox_cy - new_h / 2.0))

    # Try to keep within image bounds by shifting
    if new_x < 0 and new_x + new_w <= img_w:
        new_x = 0
    elif new_x + new_w > img_w and new_x >= 0:
        new_x = img_w - new_w

    if new_y < 0 and new_y + new_h <= img_h:
        new_y = 0
    elif new_y + new_h > img_h and new_y >= 0:
        new_y = img_h - new_h

    # Calculate padding needed if bbox still exceeds image
    pad_left = max(0, -new_x)
    pad_right = max(0, new_x + new_w - img_w)
    pad_top = max(0, -new_y)
    pad_bottom = max(0, new_y + new_h - img_h)

    # Create canvas (expanded image)
    canvas_w = img_w + pad_left + pad_right
    canvas_h = img_h + pad_top + pad_bottom

    canvas_image = torch.zeros((B, canvas_h, canvas_w, C), device=device, dtype=image.dtype)
    canvas_mask_orig = torch.zeros((B, canvas_h, canvas_w), device=device, dtype=original_mask.dtype)
    canvas_mask_proc = torch.zeros((B, canvas_h, canvas_w), device=device, dtype=processed_mask.dtype)

    # Place original image and masks in canvas
    canvas_image[:, pad_top:pad_top+img_h, pad_left:pad_left+img_w, :] = image
    canvas_mask_orig[:, pad_top:pad_top+img_h, pad_left:pad_left+img_w] = original_mask
    canvas_mask_proc[:, pad_top:pad_top+img_h, pad_left:pad_left+img_w] = processed_mask

    # Fill edges by replicating border pixels
    if pad_top > 0:
        canvas_image[:, :pad_top, pad_left:pad_left+img_w, :] = image[:, 0:1, :, :].expand(-1, pad_top, -1, -1)
    if pad_bottom > 0:
        canvas_image[:, -pad_bottom:, pad_left:pad_left+img_w, :] = image[:, -1:, :, :].expand(-1, pad_bottom, -1, -1)
    if pad_left > 0:
        canvas_image[:, :, :pad_left, :] = canvas_image[:, :, pad_left:pad_left+1, :].expand(-1, -1, pad_left, -1)
    if pad_right > 0:
        canvas_image[:, :, -pad_right:, :] = canvas_image[:, :, -pad_right-1:-pad_right, :].expand(-1, -1, pad_right, -1)

    # Canvas-to-original coordinates
    cto_x, cto_y = pad_left, pad_top
    cto_w, cto_h = img_w, img_h

    # Crop coordinates in canvas space
    ctc_x = new_x + pad_left
    ctc_y = new_y + pad_top
    ctc_w, ctc_h = new_w, new_h

    # Crop from canvas
    cropped_image = canvas_image[:, ctc_y:ctc_y+ctc_h, ctc_x:ctc_x+ctc_w, :]
    cropped_mask_orig = canvas_mask_orig[:, ctc_y:ctc_y+ctc_h, ctc_x:ctc_x+ctc_w]
    cropped_mask_proc = canvas_mask_proc[:, ctc_y:ctc_y+ctc_h, ctc_x:ctc_x+ctc_w]

    # Resize to target dimensions
    if target_w != ctc_w or target_h != ctc_h:
        cropped_image = rescale_image(cropped_image, target_w, target_h, resize_algorithm)
        cropped_mask_orig = rescale_mask(cropped_mask_orig, target_w, target_h, resize_algorithm)
        cropped_mask_proc = rescale_mask(cropped_mask_proc, target_w, target_h, resize_algorithm)

    return (canvas_image, cto_x, cto_y, cto_w, cto_h,
            cropped_image, cropped_mask_orig, cropped_mask_proc,
            ctc_x, ctc_y, ctc_w, ctc_h)


# =============================================================================
# Main Node Class
# =============================================================================

class NV_InpaintCrop:
    """
    Crops an image around a masked region for inpainting.

    Outputs TWO masks:
    - cropped_mask: Original mask (unprocessed) - use for tight stitching
    - cropped_mask_processed: With all processing ops - use for diffusion inpainting

    The stitcher uses the original mask for blending, minimizing changes to the image.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),

                "target_mode": (["manual", "auto"], {
                    "default": "manual",
                    "tooltip": "manual: use target_width/height below. auto: compute optimal resolution from bbox aspect ratio and preset."
                }),
                "target_width": ("INT", {
                    "default": 512, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 8,
                    "tooltip": "Output width for cropped region (manual mode). Ignored when target_mode=auto."
                }),
                "target_height": ("INT", {
                    "default": 512, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 8,
                    "tooltip": "Output height for cropped region (manual mode). Ignored when target_mode=auto."
                }),
                "auto_preset": (list(WAN_PRESETS.keys()), {
                    "default": "WAN_480p",
                    "tooltip": "Resolution preset for auto mode. WAN_480p: ~400k pixels (832x480 scale). WAN_720p: ~920k pixels (1280x720 scale)."
                }),
                "padding_multiple": (["0", "8", "16", "32", "64"], {
                    "default": "32",
                    "tooltip": "Round output dimensions to multiples of this value. Use 8 for most models, 32 for latent-space operations, 64 for maximum compatibility."
                }),

                # Mask processing (applied to processed mask only)
                "mask_erode_dilate": ("INT", {
                    "default": 0, "min": -128, "max": 128, "step": 1,
                    "tooltip": "Shrink (negative) or expand (positive) the mask using grey morphology. "
                               "Recommended: -8 to -16 for tighter face boundaries, +8 to +16 for object removal safety margin, "
                               "+32 to +64 for large area inpainting. Uses scipy grey_erosion/dilation to preserve gradient edges."
                }),
                "mask_fill_holes": ("INT", {
                    "default": 0, "min": 0, "max": 128, "step": 1,
                    "tooltip": "Fill gaps/holes in mask using morphological closing (dilate then erode). "
                               "Recommended: 8-16 for small gaps between strokes, 32-64 for medium holes, 64+ for large interior gaps. "
                               "Useful for masks with unwanted holes from segmentation."
                }),
                "mask_remove_noise": ("INT", {
                    "default": 0, "min": 0, "max": 32, "step": 1,
                    "tooltip": "Remove isolated pixels/specks using morphological opening (erode then dilate). "
                               "Recommended: 2-4 for tiny specks, 8-16 for larger noise clusters. "
                               "Keeps main mask regions intact while eliminating stray pixels."
                }),
                "mask_smooth": ("INT", {
                    "default": 0, "min": 0, "max": 127, "step": 1,
                    "tooltip": "Smooth jagged mask edges by binarizing (threshold 0.5) then Gaussian blurring. "
                               "Recommended: 3-9 for subtle smoothing, 15-31 for noticeable softening. "
                               "Creates cleaner edges than direct blur. Value must be odd (auto-adjusted if even)."
                }),

                # Blend settings (for stitching)
                "stitch_source": (["tight", "processed", "bbox"], {
                    "default": "tight",
                    "tooltip": "Which mask to use as the base for stitch blending. "
                               "tight: original unprocessed mask (minimal changes to image). "
                               "processed: mask after erode/dilate/fill/smooth (matches diffusion mask). "
                               "bbox: full crop region (blends in the entire re-generated area)."
                }),
                "mask_blend_pixels": ("INT", {
                    "default": 16, "min": 0, "max": 64, "step": 1,
                    "tooltip": "Feather the stitch mask edges for seamless stitching (dilate + blur). "
                               "Recommended: 8-16 for subtle blending, 24-32 for visible seam hiding, 48-64 for aggressive blending."
                }),

                "resize_algorithm": (["bicubic", "bilinear", "nearest", "area"], {
                    "default": "bicubic",
                    "tooltip": "Interpolation for resizing. bicubic: best quality (smooth), bilinear: fast/good, "
                               "nearest: preserves hard edges (pixel art), area: best for downscaling."
                }),
            },
            "optional": {
                "mask_config": ("MASK_PROCESSING_CONFIG", {
                    "tooltip": "Optional shared config from NV_MaskProcessingConfig. "
                               "When connected, overrides this node's local mask processing widgets "
                               "(erode_dilate, fill_holes, remove_noise, smooth, blend_pixels)."
                }),
                "bounding_box_mask": ("MASK", {
                    "tooltip": "Optional mask defining minimum crop area. Crop region will encompass this entire mask. "
                               "Use to ensure specific areas are included even if main mask is smaller. "
                               "Main mask must be fully contained within this bounding box mask."
                }),
                "anomaly_threshold": ("FLOAT", {
                    "default": 1.5, "min": 0.0, "max": 10.0, "step": 0.1,
                    "tooltip": "Reject frames where bbox jumps exceed this threshold "
                               "(occlusion, tracking loss, someone walking in front). "
                               "Rejected frames are filled from neighbors. "
                               "0.0 = disabled. 1.0 = strict. 1.5 = moderate (recommended). 3.0 = lenient."
                }),
            }
        }

    RETURN_TYPES = ("STITCHER", "IMAGE", "MASK", "MASK", "STRING")
    RETURN_NAMES = ("stitcher", "cropped_image", "cropped_mask", "cropped_mask_processed", "info")
    FUNCTION = "crop"
    CATEGORY = "NV_Utils/Inpaint"
    DESCRIPTION = "Crops image for inpainting. Outputs both original mask (for stitching) and processed mask (for diffusion). stitch_source selects which mask drives the blend: tight (original), processed, or bbox (full region). Auto mode computes optimal resolution from bbox."

    def crop(self, image, mask, target_mode, target_width, target_height, auto_preset,
             padding_multiple, mask_erode_dilate=0, mask_fill_holes=0, mask_remove_noise=0,
             mask_smooth=0, stitch_source="tight", mask_blend_pixels=16, resize_algorithm="bicubic",
             mask_config=None, bounding_box_mask=None, anomaly_threshold=1.5):

        # Apply shared config override if connected
        from .mask_processing_config import apply_mask_config
        vals = apply_mask_config(mask_config,
            mask_erode_dilate=mask_erode_dilate,
            mask_fill_holes=mask_fill_holes,
            mask_remove_noise=mask_remove_noise,
            mask_smooth=mask_smooth,
            mask_blend_pixels=mask_blend_pixels,
        )
        # Use _v suffix locals to avoid shadowing imported functions
        erode_dilate_v = vals["mask_erode_dilate"]
        fill_holes_v = vals["mask_fill_holes"]
        remove_noise_v = vals["mask_remove_noise"]
        smooth_v = vals["mask_smooth"]
        blend_pixels_v = vals["mask_blend_pixels"]

        padding_multiple = int(padding_multiple)
        device = comfy.model_management.get_torch_device()

        # Move to GPU
        image = image.to(device)
        mask = mask.to(device)
        if bounding_box_mask is not None:
            bounding_box_mask = bounding_box_mask.to(device)

        batch_size = image.shape[0]

        # Handle batch dimension mismatches
        if mask.shape[0] == 1 and batch_size > 1:
            mask = mask.expand(batch_size, -1, -1).clone()
        if bounding_box_mask is not None and bounding_box_mask.shape[0] == 1 and batch_size > 1:
            bounding_box_mask = bounding_box_mask.expand(batch_size, -1, -1).clone()

        # Validate dimensions
        assert image.shape[1] == mask.shape[1] and image.shape[2] == mask.shape[2], \
            f"Image and mask dimensions must match: image {image.shape}, mask {mask.shape}"

        info_lines = []

        # Auto resolution: compute optimal target from bbox aspect ratio
        if target_mode == "auto":
            bbox_union = self._compute_bbox_union(
                bounding_box_mask if bounding_box_mask is not None else mask,
                batch_size
            )
            if bbox_union is not None:
                _, _, union_w, union_h = bbox_union
                bbox_aspect = union_w / union_h
                target_width, target_height = compute_auto_resolution(
                    bbox_aspect, auto_preset, padding_multiple
                )
                info_lines.append(f"Auto resolution: {target_width}x{target_height}")
                info_lines.append(f"  bbox union: {union_w}x{union_h} (aspect {bbox_aspect:.3f})")
                info_lines.append(f"  preset: {auto_preset} ({WAN_PRESETS[auto_preset]['pixels']:,} px budget)")
            else:
                info_lines.append(f"Auto resolution: no bbox found, falling back to {target_width}x{target_height}")

        print(f"[NV_InpaintCrop] Processing {batch_size} frame(s), target {target_width}x{target_height}")

        # Initialize stitcher
        stitcher = {
            'resize_algorithm': resize_algorithm,
            'blend_pixels': blend_pixels_v,
            'crop_target_w': target_width,
            'crop_target_h': target_height,
            'canvas_to_orig_x': [],
            'canvas_to_orig_y': [],
            'canvas_to_orig_w': [],
            'canvas_to_orig_h': [],
            'canvas_image': [],
            'cropped_to_canvas_x': [],
            'cropped_to_canvas_y': [],
            'cropped_to_canvas_w': [],
            'cropped_to_canvas_h': [],
            'cropped_mask_for_blend': [],
            'skipped_indices': [],
            'original_frames': [],
            'total_frames': batch_size,
        }

        result_images = []
        result_masks_original = []
        result_masks_processed = []

        # First pass: collect bounding boxes for anomaly detection
        needs_bbox_pass = batch_size > 1 and anomaly_threshold > 0.0
        if needs_bbox_pass:
            raw_bboxes = []
            for b in range(batch_size):
                bbox_source = bounding_box_mask[b] if bounding_box_mask is not None else mask[b]
                bbox = find_bbox(bbox_source)
                raw_bboxes.append(bbox)

            # Anomaly detection: reject occlusion/tracking-loss frames before cropping
            if anomaly_threshold > 0.0:
                raw_bboxes = detect_bbox_anomalies(raw_bboxes, anomaly_threshold, info_lines)
        else:
            raw_bboxes = None

        # Process each frame
        for b in range(batch_size):
            one_image = image[b:b+1]
            one_mask = mask[b:b+1]
            one_bbox_mask = bounding_box_mask[b:b+1] if bounding_box_mask is not None else None

            # Check for empty mask
            if torch.count_nonzero(one_mask) == 0:
                print(f"[NV_InpaintCrop] Frame {b}: Empty mask - skipping")
                stitcher['skipped_indices'].append(b)
                stitcher['original_frames'].append(one_image.squeeze(0).to(comfy.model_management.intermediate_device()))
                continue

            # Keep original mask unmodified
            original_mask = one_mask.clone()

            # Process mask for diffusion (apply all operations)
            processed_mask = one_mask.clone()

            if fill_holes_v > 0:
                processed_mask = _op_fill_holes(processed_mask, fill_holes_v)
            if remove_noise_v > 0:
                processed_mask = _op_remove_noise(processed_mask, remove_noise_v)
            if erode_dilate_v != 0:
                processed_mask = _op_erode_dilate(processed_mask, erode_dilate_v)
            if smooth_v > 0:
                processed_mask = _op_smooth(processed_mask, smooth_v)

            # Determine crop bounding box
            if raw_bboxes is not None and raw_bboxes[b] is not None:
                bbox = raw_bboxes[b]
            elif one_bbox_mask is not None:
                bbox = find_bbox(one_bbox_mask)
                if bbox is None:
                    bbox = find_bbox(processed_mask)
            else:
                bbox = find_bbox(processed_mask)

            if bbox is None:
                print(f"[NV_InpaintCrop] Frame {b}: No bbox found - skipping")
                stitcher['skipped_indices'].append(b)
                stitcher['original_frames'].append(one_image.squeeze(0).to(comfy.model_management.intermediate_device()))
                continue

            bbox_x, bbox_y, bbox_w, bbox_h = bbox
            print(f"[NV_InpaintCrop] Frame {b}: bbox ({bbox_x}, {bbox_y}, {bbox_w}x{bbox_h})")

            # Perform crop
            (canvas_image, cto_x, cto_y, cto_w, cto_h,
             cropped_image, cropped_mask_orig, cropped_mask_proc,
             ctc_x, ctc_y, ctc_w, ctc_h) = crop_for_inpaint(
                one_image, original_mask, processed_mask,
                bbox_x, bbox_y, bbox_w, bbox_h,
                target_width, target_height, padding_multiple, resize_algorithm
            )

            # Create blend mask from selected source
            if stitch_source == "bbox":
                # Full crop region — blend in everything
                blend_mask = torch.ones_like(cropped_mask_orig)
            elif stitch_source == "processed":
                blend_mask = cropped_mask_proc.clone()
            else:
                # "tight" — original unprocessed mask (default)
                blend_mask = cropped_mask_orig.clone()

            if blend_pixels_v > 0:
                if stitch_source == "bbox":
                    # Bbox is all-ones — erode INWARD from edges to create a border, then blur
                    blend_mask = _op_erode_dilate(blend_mask, -blend_pixels_v)
                else:
                    # Tight/processed — dilate OUTWARD to extend blend beyond mask edge
                    blend_mask = _op_erode_dilate(blend_mask, blend_pixels_v)
                blend_mask = mask_blur(blend_mask, blend_pixels_v)

            # Store in stitcher
            intermediate = comfy.model_management.intermediate_device()
            stitcher['canvas_to_orig_x'].append(cto_x)
            stitcher['canvas_to_orig_y'].append(cto_y)
            stitcher['canvas_to_orig_w'].append(cto_w)
            stitcher['canvas_to_orig_h'].append(cto_h)
            stitcher['canvas_image'].append(canvas_image.squeeze(0).to(intermediate))
            stitcher['cropped_to_canvas_x'].append(ctc_x)
            stitcher['cropped_to_canvas_y'].append(ctc_y)
            stitcher['cropped_to_canvas_w'].append(ctc_w)
            stitcher['cropped_to_canvas_h'].append(ctc_h)
            stitcher['cropped_mask_for_blend'].append(blend_mask.squeeze(0).to(intermediate))

            result_images.append(cropped_image.squeeze(0).to(intermediate))
            result_masks_original.append(cropped_mask_orig.squeeze(0).to(intermediate))
            result_masks_processed.append(cropped_mask_proc.squeeze(0).to(intermediate))

        # Handle all-skipped case
        if len(result_images) == 0:
            print(f"[NV_InpaintCrop] All frames skipped - returning original")
            info_lines.append("All frames skipped - no mask content found")
            empty_mask = torch.zeros((batch_size, image.shape[1], image.shape[2]),
                                     device=comfy.model_management.intermediate_device())
            return (stitcher,
                    image.to(comfy.model_management.intermediate_device()),
                    empty_mask,
                    empty_mask,
                    "\n".join(info_lines))

        result_images = torch.stack(result_images, dim=0)
        result_masks_original = torch.stack(result_masks_original, dim=0)
        result_masks_processed = torch.stack(result_masks_processed, dim=0)

        out_h, out_w = result_images.shape[1], result_images.shape[2]
        info_lines.append(f"Output: {result_images.shape[0]} frames @ {out_w}x{out_h}")
        print(f"[NV_InpaintCrop] Output: {result_images.shape[0]} frames, {out_w}x{out_h}")

        return (stitcher, result_images, result_masks_original, result_masks_processed,
                "\n".join(info_lines))

    def _compute_bbox_union(self, mask_source, batch_size):
        """
        Compute the union bounding box across all frames.

        Returns (x, y, w, h) of the envelope containing all per-frame bboxes,
        or None if no frames have content.
        """
        union_x_min, union_y_min = float('inf'), float('inf')
        union_x_max, union_y_max = 0, 0
        found_any = False

        for b in range(batch_size):
            bbox = find_bbox(mask_source[b])
            if bbox is not None:
                bx, by, bw, bh = bbox
                union_x_min = min(union_x_min, bx)
                union_y_min = min(union_y_min, by)
                union_x_max = max(union_x_max, bx + bw)
                union_y_max = max(union_y_max, by + bh)
                found_any = True

        if not found_any:
            return None

        return (int(union_x_min), int(union_y_min),
                int(union_x_max - union_x_min), int(union_y_max - union_y_min))


# =============================================================================
# Node Registration
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "NV_InpaintCrop2": NV_InpaintCrop,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_InpaintCrop2": "NV Inpaint Crop v2",
}
