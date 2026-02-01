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
import time
import torch
import comfy.model_management
import comfy.utils
import nodes
import numpy as np
import scipy.ndimage
from scipy.ndimage import gaussian_filter1d, median_filter
import torchvision.transforms.v2 as T


# =============================================================================
# Utility Functions
# =============================================================================

def gaussian_smooth_1d(values, window_size):
    """Apply Gaussian smoothing to 1D sequence for crop stabilization."""
    if len(values) <= 1:
        return values
    sigma = window_size / 4.0
    smoothed = gaussian_filter1d(values, sigma, mode='nearest')
    return smoothed.tolist()


def median_filter_1d(values, window_size):
    """Apply median filter to 1D sequence for crop stabilization."""
    if len(values) <= 1:
        return values
    filtered = median_filter(np.array(values), size=window_size, mode='nearest')
    return filtered.tolist()


def rescale_image(samples, width, height, algorithm='bicubic'):
    """Resize image tensor [B, H, W, C] using GPU."""
    algorithm_map = {
        'nearest': 'nearest',
        'bilinear': 'bilinear',
        'bicubic': 'bicubic',
        'lanczos': 'bicubic',
        'area': 'area',
    }
    mode = algorithm_map.get(algorithm.lower(), 'bicubic')

    # [B, H, W, C] -> [B, C, H, W]
    samples = samples.movedim(-1, 1)

    samples = torch.nn.functional.interpolate(
        samples,
        size=(height, width),
        mode=mode,
        align_corners=False if mode in ['bilinear', 'bicubic'] else None
    )

    # [B, C, H, W] -> [B, H, W, C]
    return samples.movedim(1, -1)


def rescale_mask(samples, width, height, algorithm='bilinear'):
    """Resize mask tensor [B, H, W] using GPU."""
    algorithm_map = {
        'nearest': 'nearest',
        'bilinear': 'bilinear',
        'bicubic': 'bicubic',
        'lanczos': 'bicubic',
        'area': 'area',
    }
    mode = algorithm_map.get(algorithm.lower(), 'bilinear')

    # [B, H, W] -> [B, 1, H, W]
    samples = samples.unsqueeze(1)

    samples = torch.nn.functional.interpolate(
        samples,
        size=(height, width),
        mode=mode,
        align_corners=False if mode in ['bilinear', 'bicubic'] else None
    )

    # [B, 1, H, W] -> [B, H, W]
    return samples.squeeze(1)


# =============================================================================
# Mask Processing Functions (using scipy grey morphology)
# =============================================================================

def mask_erode_dilate(mask, amount):
    """
    Erode (negative) or dilate (positive) mask using scipy grey morphology.
    Grey operations preserve grayscale gradients unlike binary operations.
    """
    if amount == 0:
        return mask

    device = mask.device
    results = []

    for m in mask:
        m_np = m.cpu().numpy()
        if amount < 0:
            # Erosion (shrink mask)
            m_np = scipy.ndimage.grey_erosion(m_np, size=(-amount, -amount))
        else:
            # Dilation (expand mask)
            m_np = scipy.ndimage.grey_dilation(m_np, size=(amount, amount))
        results.append(torch.from_numpy(m_np).to(device))

    return torch.stack(results, dim=0)


def mask_fill_holes(mask, size):
    """
    Fill holes in mask using grey closing (dilation followed by erosion).
    Better than binary fill for soft/gradient masks.
    """
    if size == 0:
        return mask

    device = mask.device
    results = []

    for m in mask:
        m_np = m.cpu().numpy()
        m_np = scipy.ndimage.grey_closing(m_np, size=(size, size))
        results.append(torch.from_numpy(m_np).to(device))

    return torch.stack(results, dim=0)


def mask_remove_noise(mask, size):
    """
    Remove isolated pixels/noise using grey opening (erosion followed by dilation).
    Eliminates small specks while preserving larger regions.
    """
    if size == 0:
        return mask

    device = mask.device
    results = []

    for m in mask:
        m_np = m.cpu().numpy()
        m_np = scipy.ndimage.grey_opening(m_np, size=(size, size))
        results.append(torch.from_numpy(m_np).to(device))

    return torch.stack(results, dim=0)


def mask_smooth(mask, amount):
    """
    Smooth mask edges by binarizing then blurring.
    Creates cleaner, crisper edges than direct blur.
    """
    if amount == 0:
        return mask

    if amount % 2 == 0:
        amount += 1

    # Binarize first (threshold at 0.5)
    binary = mask > 0.5

    # Then blur
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)

    smoothed = T.functional.gaussian_blur(binary.unsqueeze(1).float(), amount).squeeze(1)

    return smoothed


def mask_blur(mask, amount):
    """
    Direct Gaussian blur on mask (preserves gradients).
    Used for blend feathering.
    """
    if amount == 0:
        return mask

    if amount % 2 == 0:
        amount += 1

    if mask.dim() == 2:
        mask = mask.unsqueeze(0)

    blurred = T.functional.gaussian_blur(mask.unsqueeze(1), amount).squeeze(1)

    return blurred


def find_bbox(mask):
    """Find bounding box of non-zero mask region. Returns (x, y, w, h) or None if empty."""
    if mask.dim() == 3:
        mask_2d = mask[0]
    else:
        mask_2d = mask

    non_zero = torch.nonzero(mask_2d > 0.01)

    if non_zero.numel() == 0:
        return None

    y_min = non_zero[:, 0].min().item()
    y_max = non_zero[:, 0].max().item()
    x_min = non_zero[:, 1].min().item()
    x_max = non_zero[:, 1].max().item()

    return (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)


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

    # Grow bbox to match target aspect ratio
    if bbox_aspect < target_aspect:
        # Need wider - grow width
        new_w = int(bbox_h * target_aspect)
        new_h = bbox_h
        new_x = bbox_x - (new_w - bbox_w) // 2
        new_y = bbox_y
    else:
        # Need taller - grow height
        new_w = bbox_w
        new_h = int(bbox_w / target_aspect)
        new_x = bbox_x
        new_y = bbox_y - (new_h - bbox_h) // 2

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

                "target_width": ("INT", {
                    "default": 512, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 8,
                    "tooltip": "Output width for cropped region. Should match your model's expected input size (512 for SD1.5, 1024 for SDXL)."
                }),
                "target_height": ("INT", {
                    "default": 512, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 8,
                    "tooltip": "Output height for cropped region. Should match your model's expected input size (512 for SD1.5, 1024 for SDXL)."
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
                "mask_blend_pixels": ("INT", {
                    "default": 16, "min": 0, "max": 64, "step": 1,
                    "tooltip": "Feather the ORIGINAL mask edges for seamless stitching (dilate + blur). "
                               "Recommended: 8-16 for subtle blending, 24-32 for visible seam hiding, 48-64 for aggressive blending. "
                               "Applied to original mask, not processed mask, for tight stitch boundaries."
                }),

                "resize_algorithm": (["bicubic", "bilinear", "nearest", "area"], {
                    "default": "bicubic",
                    "tooltip": "Interpolation for resizing. bicubic: best quality (smooth), bilinear: fast/good, "
                               "nearest: preserves hard edges (pixel art), area: best for downscaling."
                }),
            },
            "optional": {
                "bounding_box_mask": ("MASK", {
                    "tooltip": "Optional mask defining minimum crop area. Crop region will encompass this entire mask. "
                               "Use to ensure specific areas are included even if main mask is smaller. "
                               "Main mask must be fully contained within this bounding box mask."
                }),

                # Video stabilization
                "stabilize_crop": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable temporal stabilization for video batches. Smooths crop position/size across frames to reduce jitter."
                }),
                "stabilization_mode": (["smooth", "lock_first", "lock_largest", "median"], {
                    "default": "smooth",
                    "tooltip": "smooth: Gaussian filter on bbox coords (gentle motion). lock_first: Use first frame's size for all. "
                               "lock_largest: Use largest bbox size for all (prevents clipping). median: Median filter (removes outliers)."
                }),
                "smooth_window": ("INT", {
                    "default": 5, "min": 3, "max": 21, "step": 2,
                    "tooltip": "Window size for temporal smoothing. Larger = more stable but less responsive. "
                               "3-5: subtle stabilization, 7-11: moderate smoothing, 13-21: heavy stabilization."
                }),
            }
        }

    RETURN_TYPES = ("STITCHER", "IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("stitcher", "cropped_image", "cropped_mask", "cropped_mask_processed")
    FUNCTION = "crop"
    CATEGORY = "NV_Utils/Inpaint"
    DESCRIPTION = "Crops image for inpainting. Outputs both original mask (for stitching) and processed mask (for diffusion)."

    def crop(self, image, mask, target_width, target_height, padding_multiple,
             mask_erode_dilate, mask_fill_holes, mask_remove_noise, mask_smooth,
             mask_blend_pixels, resize_algorithm,
             bounding_box_mask=None, stabilize_crop=False, stabilization_mode="smooth", smooth_window=5):

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

        print(f"[NV_InpaintCrop] Processing {batch_size} frame(s), target {target_width}x{target_height}")

        # Initialize stitcher
        stitcher = {
            'resize_algorithm': resize_algorithm,
            'blend_pixels': mask_blend_pixels,
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

        # First pass: collect bounding boxes for stabilization
        if stabilize_crop and batch_size > 1:
            raw_bboxes = []
            for b in range(batch_size):
                bbox_source = bounding_box_mask[b] if bounding_box_mask is not None else mask[b]
                bbox = find_bbox(bbox_source)
                raw_bboxes.append(bbox)

            raw_bboxes = self._stabilize_bboxes(raw_bboxes, stabilization_mode, smooth_window)
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

            if mask_fill_holes > 0:
                processed_mask = mask_fill_holes_fn(processed_mask, mask_fill_holes)
            if mask_remove_noise > 0:
                processed_mask = mask_remove_noise_fn(processed_mask, mask_remove_noise)
            if mask_erode_dilate != 0:
                processed_mask = mask_erode_dilate_fn(processed_mask, mask_erode_dilate)
            if mask_smooth > 0:
                processed_mask = mask_smooth_fn(processed_mask, mask_smooth)

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

            # Create blend mask from ORIGINAL mask (for tight stitching)
            blend_mask = cropped_mask_orig.clone()
            if mask_blend_pixels > 0:
                blend_mask = mask_erode_dilate_fn(blend_mask, mask_blend_pixels)
                blend_mask = mask_blur(blend_mask, mask_blend_pixels)

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
            empty_mask = torch.zeros((batch_size, image.shape[1], image.shape[2]),
                                     device=comfy.model_management.intermediate_device())
            return (stitcher,
                    image.to(comfy.model_management.intermediate_device()),
                    empty_mask,
                    empty_mask)

        result_images = torch.stack(result_images, dim=0)
        result_masks_original = torch.stack(result_masks_original, dim=0)
        result_masks_processed = torch.stack(result_masks_processed, dim=0)

        print(f"[NV_InpaintCrop] Output: {result_images.shape[0]} frames, {result_images.shape[2]}x{result_images.shape[1]}")

        return (stitcher, result_images, result_masks_original, result_masks_processed)

    def _stabilize_bboxes(self, bboxes, mode, window_size):
        """Apply temporal stabilization to bounding box sequence."""
        valid_indices = [i for i, b in enumerate(bboxes) if b is not None]

        if len(valid_indices) <= 1:
            return bboxes

        xs = [bboxes[i][0] for i in valid_indices]
        ys = [bboxes[i][1] for i in valid_indices]
        ws = [bboxes[i][2] for i in valid_indices]
        hs = [bboxes[i][3] for i in valid_indices]

        if mode == "smooth":
            xs = [int(round(v)) for v in gaussian_smooth_1d(xs, window_size)]
            ys = [int(round(v)) for v in gaussian_smooth_1d(ys, window_size)]
            ws = [int(round(v)) for v in gaussian_smooth_1d(ws, window_size)]
            hs = [int(round(v)) for v in gaussian_smooth_1d(hs, window_size)]

        elif mode == "lock_first":
            ref_w, ref_h = ws[0], hs[0]
            for idx in range(len(valid_indices)):
                center_x = xs[idx] + ws[idx] // 2
                center_y = ys[idx] + hs[idx] // 2
                ws[idx] = ref_w
                hs[idx] = ref_h
                xs[idx] = center_x - ref_w // 2
                ys[idx] = center_y - ref_h // 2

        elif mode == "lock_largest":
            max_w = max(ws)
            max_h = max(hs)
            for idx in range(len(valid_indices)):
                center_x = xs[idx] + ws[idx] // 2
                center_y = ys[idx] + hs[idx] // 2
                ws[idx] = max_w
                hs[idx] = max_h
                xs[idx] = center_x - max_w // 2
                ys[idx] = center_y - max_h // 2

        elif mode == "median":
            xs = [int(round(v)) for v in median_filter_1d(xs, window_size)]
            ys = [int(round(v)) for v in median_filter_1d(ys, window_size)]
            ws = [int(round(v)) for v in median_filter_1d(ws, window_size)]
            hs = [int(round(v)) for v in median_filter_1d(hs, window_size)]

        result = list(bboxes)
        for idx, i in enumerate(valid_indices):
            result[i] = (xs[idx], ys[idx], ws[idx], hs[idx])

        return result


# Alias functions to avoid name collision with parameters
mask_erode_dilate_fn = mask_erode_dilate
mask_fill_holes_fn = mask_fill_holes
mask_remove_noise_fn = mask_remove_noise
mask_smooth_fn = mask_smooth


# =============================================================================
# Node Registration
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "NV_InpaintCrop2": NV_InpaintCrop,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_InpaintCrop2": "NV Inpaint Crop v2",
}
