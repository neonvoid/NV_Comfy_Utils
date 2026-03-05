"""
NV Mask Pipeline Viz - Debug visualization of all mask processing stages.

Shows a 3x2 contact sheet with colored mask overlays at each stage of the
MaskTrackingBBox -> InpaintCrop v2 -> VACE Control Video Prep pipeline.
Helps tune mask parameters by making spatial relationships visible.
"""

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .inpaint_crop import (
    mask_erode_dilate as _mask_erode_dilate,
    mask_fill_holes as _mask_fill_holes,
    mask_remove_noise as _mask_remove_noise,
    mask_smooth as _mask_smooth,
    mask_blur as _mask_blur,
    rescale_mask,
)


# Panel color assignments (RGB) — chosen for visual distinctness
PANEL_CONFIGS = [
    ("original",  "1. Original Mask",  (60, 120, 255)),   # blue
    ("bbox",      "2. BBox Mask",      (255, 220, 60)),    # yellow
    ("processed", "3. Processed Mask", (60, 220, 100)),    # green
    ("blend",     "4. Blend Mask",     (255, 80, 80)),     # red
    ("vace",      "5. VACE Cond Mask", (220, 80, 255)),    # magenta
    ("stitch",    "6. Stitch Mask",    (80, 220, 255)),    # cyan
]


def _get_font(size):
    """Load font with fallback chain."""
    for name in ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except (OSError, IOError):
            continue
    try:
        return ImageFont.load_default(size)
    except TypeError:
        return ImageFont.load_default()


def _overlay_mask_soft(img_np, mask_np, color, alpha):
    """Composite a soft mask as colored overlay. Preserves mask gradients
    (unlike binary threshold) so feathered edges are visible."""
    h, w = img_np.shape[:2]
    mh, mw = mask_np.shape

    if mh != h or mw != w:
        mask_t = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0)
        mask_t = torch.nn.functional.interpolate(mask_t, size=(h, w), mode='bilinear', align_corners=False)
        mask_np = mask_t.squeeze().numpy()

    mask_np = np.clip(mask_np, 0.0, 1.0)

    overlay = img_np.astype(np.float32)
    for c in range(3):
        overlay[:, :, c] = overlay[:, :, c] * (1.0 - alpha * mask_np) + color[c] * (alpha * mask_np)

    return np.clip(overlay, 0, 255).astype(np.uint8)


def _draw_label(img_np, text, font):
    """Draw a text label with dark background in the top-left corner."""
    pil_img = Image.fromarray(img_np)
    draw = ImageDraw.Draw(pil_img)

    pad = 6
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Dark background rectangle
    draw.rectangle(
        [pad, pad, pad + text_w + 8, pad + text_h + 8],
        fill=(0, 0, 0, 180),
    )
    # White text
    draw.text((pad + 4, pad + 4), text, fill=(255, 255, 255), font=font)

    return np.array(pil_img)


class NV_MaskPipelineViz:
    """Visualize all mask processing stages as a 3x2 colored overlay contact sheet."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Source video frames [B, H, W, C].",
                }),
                "original_mask": ("MASK", {
                    "tooltip": "Raw segmentation mask [B, H, W].",
                }),
                "frame_index": ("INT", {
                    "default": 0, "min": 0, "max": 99999, "step": 1,
                    "tooltip": "Which frame to visualize.",
                }),
                "overlay_alpha": ("FLOAT", {
                    "default": 0.4, "min": 0.1, "max": 0.9, "step": 0.05,
                    "tooltip": "Mask overlay opacity.",
                }),
                # InpaintCrop mask processing params
                "mask_erode_dilate": ("INT", {
                    "default": 4, "min": -128, "max": 128, "step": 1,
                    "tooltip": "Mirror of InpaintCrop. Negative=shrink, positive=expand.",
                }),
                "mask_fill_holes": ("INT", {
                    "default": 0, "min": 0, "max": 128, "step": 1,
                    "tooltip": "Mirror of InpaintCrop. Grey closing to fill gaps.",
                }),
                "mask_remove_noise": ("INT", {
                    "default": 0, "min": 0, "max": 32, "step": 1,
                    "tooltip": "Mirror of InpaintCrop. Grey opening to remove specks.",
                }),
                "mask_smooth": ("INT", {
                    "default": 0, "min": 0, "max": 127, "step": 1,
                    "tooltip": "Mirror of InpaintCrop. Binarize + blur edges.",
                }),
                "mask_blend_pixels": ("INT", {
                    "default": 16, "min": 0, "max": 64, "step": 1,
                    "tooltip": "Mirror of InpaintCrop. Dilate + blur for stitching.",
                }),
                # VACE Control Video Prep params
                "erosion_blocks": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 4.0, "step": 0.25,
                    "tooltip": "Mirror of VACE Prep. Erode inward in VAE block units.",
                }),
                "feather_blocks": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 8.0, "step": 0.25,
                    "tooltip": "Mirror of VACE Prep. Feather in VAE block units.",
                }),
                "vae_stride": ("INT", {
                    "default": 8, "min": 4, "max": 32, "step": 4,
                    "tooltip": "VAE spatial stride. 8 for WAN.",
                }),
                "stitch_erosion": ("INT", {
                    "default": 0, "min": -32, "max": 32, "step": 1,
                    "tooltip": "Mirror of VACE Prep. Erode/dilate stitch mask (pixels).",
                }),
                "stitch_feather": ("INT", {
                    "default": 4, "min": 0, "max": 64, "step": 1,
                    "tooltip": "Mirror of VACE Prep. Feather stitch mask (pixels).",
                }),
            },
            "optional": {
                "mask_config": ("MASK_PROCESSING_CONFIG", {
                    "tooltip": "Optional shared config from NV_MaskProcessingConfig. "
                               "When connected, overrides all mask processing widgets on this node."
                }),
                "bbox_mask": ("MASK", {
                    "tooltip": "Optional bounding box mask from MaskTrackingBBox.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("visualization",)
    FUNCTION = "execute"
    CATEGORY = "NV_Utils/Debug"
    DESCRIPTION = (
        "Debug visualization: 3x2 contact sheet showing all 6 mask processing "
        "stages as colored overlays. Mirror your InpaintCrop + VACE Prep "
        "parameters here to see what each stage looks like spatially."
    )

    def execute(self, image, original_mask, frame_index, overlay_alpha,
                mask_erode_dilate, mask_fill_holes, mask_remove_noise,
                mask_smooth, mask_blend_pixels,
                erosion_blocks, feather_blocks, vae_stride,
                stitch_erosion, stitch_feather,
                mask_config=None, bbox_mask=None):

        # Apply shared config override if connected
        from .mask_processing_config import apply_mask_config, apply_vace_mask_config
        vals = apply_mask_config(mask_config,
            mask_erode_dilate=mask_erode_dilate,
            mask_fill_holes=mask_fill_holes,
            mask_remove_noise=mask_remove_noise,
            mask_smooth=mask_smooth,
            mask_blend_pixels=mask_blend_pixels,
        )
        mask_erode_dilate = vals["mask_erode_dilate"]
        mask_fill_holes = vals["mask_fill_holes"]
        mask_remove_noise = vals["mask_remove_noise"]
        mask_smooth = vals["mask_smooth"]
        mask_blend_pixels = vals["mask_blend_pixels"]
        vace_vals = apply_vace_mask_config(mask_config,
            erosion_blocks=erosion_blocks,
            feather_blocks=feather_blocks,
            stitch_erosion=stitch_erosion,
            stitch_feather=stitch_feather,
        )
        erosion_blocks = vace_vals["erosion_blocks"]
        feather_blocks = vace_vals["feather_blocks"]
        stitch_erosion = vace_vals["stitch_erosion"]
        stitch_feather = vace_vals["stitch_feather"]

        B = image.shape[0]
        fi = min(frame_index, B - 1)

        # Extract single frame
        frame = image[fi]  # [H, W, C]
        H, W = frame.shape[0], frame.shape[1]

        # Extract and validate mask — ensure [1, H, W] for processing functions
        if original_mask.dim() == 2:
            original_mask = original_mask.unsqueeze(0)
        mask_fi = min(fi, original_mask.shape[0] - 1)
        mask_frame = original_mask[mask_fi:mask_fi + 1]  # [1, H, W]

        # Resize mask if resolution mismatch
        if mask_frame.shape[1] != H or mask_frame.shape[2] != W:
            mask_frame = rescale_mask(mask_frame, W, H)

        # Handle optional bbox_mask
        has_bbox = bbox_mask is not None
        bbox_frame = None
        if has_bbox:
            if bbox_mask.dim() == 2:
                bbox_mask = bbox_mask.unsqueeze(0)
            bbox_fi = min(fi, bbox_mask.shape[0] - 1)
            bbox_frame = bbox_mask[bbox_fi:bbox_fi + 1]
            if bbox_frame.shape[1] != H or bbox_frame.shape[2] != W:
                bbox_frame = rescale_mask(bbox_frame, W, H)

        # --- Compute 6 mask stages ---
        masks = {}

        # 1. Original — as-is
        masks["original"] = mask_frame.clone()

        # 2. BBox — from optional input
        masks["bbox"] = bbox_frame.clone() if has_bbox else None

        # 3. Processed — InpaintCrop pipeline (same order as inpaint_crop.py execute)
        processed = mask_frame.clone()
        if mask_fill_holes > 0:
            processed = _mask_fill_holes(processed, mask_fill_holes)
        if mask_remove_noise > 0:
            processed = _mask_remove_noise(processed, mask_remove_noise)
        if mask_erode_dilate != 0:
            processed = _mask_erode_dilate(processed, mask_erode_dilate)
        if mask_smooth > 0:
            processed = _mask_smooth(processed, mask_smooth)
        masks["processed"] = processed

        # 4. Blend — dilate + blur of ORIGINAL mask (not processed)
        blend = mask_frame.clone()
        if mask_blend_pixels > 0:
            blend = _mask_erode_dilate(blend, mask_blend_pixels)  # positive = dilate
            blend = _mask_blur(blend, mask_blend_pixels)
        masks["blend"] = blend.clamp(0.0, 1.0)

        # 5. VACE conditioning — erode inward + feather
        vace = mask_frame.clone()
        erosion_px = int(round(erosion_blocks * vae_stride))
        feather_px = int(round(feather_blocks * vae_stride))
        if erosion_px > 0:
            vace = _mask_erode_dilate(vace, -erosion_px)  # negative = erode
        if feather_px > 0:
            vace = _mask_blur(vace, feather_px)
        masks["vace"] = vace.clamp(0.0, 1.0)

        # 6. Stitch — erosion + feather (pixels, independent from VACE blocks)
        stitch = mask_frame.clone()
        if stitch_erosion != 0:
            stitch = _mask_erode_dilate(stitch, stitch_erosion)
        if stitch_feather > 0:
            stitch = _mask_blur(stitch, stitch_feather)
        masks["stitch"] = stitch.clamp(0.0, 1.0)

        # --- Build overlay panels ---
        frame_np = (frame.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        font = _get_font(max(16, min(H // 20, 28)))

        panels = []
        for key, label, color in PANEL_CONFIGS:
            panel_img = frame_np.copy()

            if masks[key] is not None:
                mask_np = masks[key][0].cpu().numpy()  # [H, W]
                panel_img = _overlay_mask_soft(panel_img, mask_np, color, overlay_alpha)
            else:
                label += " (none)"

            panel_img = _draw_label(panel_img, label, font)
            panels.append(panel_img)

        # --- Assemble 3x2 grid ---
        row1 = np.concatenate([panels[0], panels[1]], axis=1)
        row2 = np.concatenate([panels[2], panels[3]], axis=1)
        row3 = np.concatenate([panels[4], panels[5]], axis=1)
        contact_sheet = np.concatenate([row1, row2, row3], axis=0)

        # --- Convert to IMAGE tensor ---
        result = torch.from_numpy(contact_sheet.astype(np.float32) / 255.0)
        result = result.unsqueeze(0)  # [1, 3*H, 2*W, 3]

        return (result,)


NODE_CLASS_MAPPINGS = {
    "NV_MaskPipelineViz": NV_MaskPipelineViz,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_MaskPipelineViz": "NV Mask Pipeline Viz",
}
