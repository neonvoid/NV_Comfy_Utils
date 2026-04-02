"""
NV Mask Pipeline Viz - Debug visualization of all mask processing stages.

Modes:
  grid:  3x2 contact sheet of all 6 stages for a single frame (original behavior)
  batch: 6 separate images as a batch — click through in ComfyUI's image preview
  video: All frames for a single selected stage — scrub through to check temporal consistency

Accepts optional STITCHER from NV_InpaintCrop to preview cropped image + mask
at the resolution the diffusion model actually sees.
"""

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .mask_ops import (
    mask_erode_dilate as _mask_erode_dilate,
    mask_fill_holes as _mask_fill_holes,
    mask_remove_noise as _mask_remove_noise,
    mask_smooth as _mask_smooth,
    mask_blur as _mask_blur,
    rescale_mask,
)


# Stage definitions: (key, label, color_rgb)
STAGE_CONFIGS = [
    ("original",  "1. Original Mask",  (60, 120, 255)),   # blue
    ("bbox",      "2. BBox Mask",      (255, 220, 60)),    # yellow
    ("processed", "3. Processed Mask", (60, 220, 100)),    # green
    ("blend",     "4. Blend Mask",     (255, 80, 80)),     # red
    ("vace",      "5. VACE Cond Mask", (220, 80, 255)),    # magenta
    ("stitch",    "6. Stitch Mask",    (80, 220, 255)),    # cyan
]

STAGE_KEYS = [s[0] for s in STAGE_CONFIGS]
STAGE_LABELS = {s[0]: s[1] for s in STAGE_CONFIGS}
STAGE_COLORS = {s[0]: s[2] for s in STAGE_CONFIGS}


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

    draw.rectangle(
        [pad, pad, pad + text_w + 8, pad + text_h + 8],
        fill=(0, 0, 0, 180),
    )
    draw.text((pad + 4, pad + 4), text, fill=(255, 255, 255), font=font)

    return np.array(pil_img)


def _compute_masks_for_frame(mask_frame, bbox_frame,
                             mask_erode_dilate, mask_fill_holes, mask_remove_noise,
                             mask_smooth, mask_blend_pixels,
                             erosion_blocks, feather_blocks, vae_stride,
                             stitch_erosion, stitch_feather):
    """Compute all 6 mask stages for a single frame. Returns dict of [1,H,W] tensors (or None for bbox)."""
    masks = {}

    # 1. Original
    masks["original"] = mask_frame.clone()

    # 2. BBox
    masks["bbox"] = bbox_frame.clone() if bbox_frame is not None else None

    # 3. Processed — InpaintCrop pipeline order
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

    # 4. Blend — dilate + blur of ORIGINAL (not processed)
    blend = mask_frame.clone()
    if mask_blend_pixels > 0:
        blend = _mask_erode_dilate(blend, mask_blend_pixels)
        blend = _mask_blur(blend, mask_blend_pixels)
    masks["blend"] = blend.clamp(0.0, 1.0)

    # 5. VACE conditioning
    vace = mask_frame.clone()
    erosion_px = int(round(erosion_blocks * vae_stride))
    feather_px = int(round(feather_blocks * vae_stride))
    if erosion_px > 0:
        vace = _mask_erode_dilate(vace, -erosion_px)
    if feather_px > 0:
        vace = _mask_blur(vace, feather_px)
    masks["vace"] = vace.clamp(0.0, 1.0)

    # 6. Stitch
    stitch = mask_frame.clone()
    if stitch_erosion != 0:
        stitch = _mask_erode_dilate(stitch, stitch_erosion)
    if stitch_feather > 0:
        stitch = _mask_blur(stitch, stitch_feather)
    masks["stitch"] = stitch.clamp(0.0, 1.0)

    return masks


def _render_overlay(frame_np, mask_tensor, stage_key, overlay_alpha, font, frame_label=None):
    """Render a single overlay panel as uint8 numpy [H,W,3]."""
    label = STAGE_LABELS[stage_key]
    color = STAGE_COLORS[stage_key]
    panel = frame_np.copy()

    if mask_tensor is not None:
        mask_np = mask_tensor[0].cpu().numpy()
        panel = _overlay_mask_soft(panel, mask_np, color, overlay_alpha)
    else:
        label += " (none)"

    if frame_label is not None:
        label = f"[{frame_label}] {label}"

    panel = _draw_label(panel, label, font)
    return panel


class NV_MaskPipelineViz:
    """Visualize mask processing stages as grid, image batch, or temporal video."""

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
                "mode": (["grid", "batch", "video"], {
                    "default": "batch",
                    "tooltip": "grid: 3x2 contact sheet (single frame, all stages). "
                               "batch: 6 separate images as clickable batch (single frame, all stages). "
                               "video: all frames for a single selected stage (temporal preview)."
                }),
                "frame_index": ("INT", {
                    "default": 0, "min": 0, "max": 99999, "step": 1,
                    "tooltip": "Which frame to visualize (grid/batch modes). Ignored in video mode.",
                }),
                "video_stage": (STAGE_KEYS, {
                    "default": "processed",
                    "tooltip": "Which mask stage to preview across all frames (video mode only).",
                }),
                "overlay_alpha": ("FLOAT", {
                    "default": 0.4, "min": 0.1, "max": 0.9, "step": 0.05,
                    "tooltip": "Mask overlay opacity.",
                }),
                # InpaintCrop mask processing params
                "crop_expand_px": ("INT", {
                    "default": 4, "min": -128, "max": 128, "step": 1,
                    "tooltip": "Mirror of InpaintCrop. Negative=shrink, positive=expand.",
                }),
                "cleanup_fill_holes": ("INT", {
                    "default": 0, "min": 0, "max": 128, "step": 1,
                    "tooltip": "Mirror of InpaintCrop. Grey closing to fill gaps.",
                }),
                "cleanup_remove_noise": ("INT", {
                    "default": 0, "min": 0, "max": 32, "step": 1,
                    "tooltip": "Mirror of InpaintCrop. Grey opening to remove specks.",
                }),
                "cleanup_smooth": ("INT", {
                    "default": 0, "min": 0, "max": 127, "step": 1,
                    "tooltip": "Mirror of InpaintCrop. Binarize + blur edges.",
                }),
                "crop_blend_feather_px": ("INT", {
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
                "vace_stitch_erosion_px": ("INT", {
                    "default": 0, "min": -32, "max": 32, "step": 1,
                    "tooltip": "Mirror of VACE Prep. Erode/dilate stitch mask (pixels).",
                }),
                "vace_stitch_feather_px": ("INT", {
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
                "cropped_image": ("IMAGE", {
                    "tooltip": "Cropped image from NV_InpaintCrop (cropped_image output). "
                               "When connected, shows mask stages on the crop the diffusion model sees. "
                               "Wire: InpaintCrop.cropped_image -> here."
                }),
                "cropped_mask": ("MASK", {
                    "tooltip": "Cropped mask from NV_InpaintCrop (cropped_mask output). "
                               "Used as the original_mask when previewing the cropped view. "
                               "Wire: InpaintCrop.cropped_mask -> here."
                }),
                # Deprecated (kept for old workflow compat — do not remove)
                "mask_erode_dilate": ("INT", {"default": 4, "min": -128, "max": 128, "step": 1,
                    "tooltip": "DEPRECATED — use crop_expand_px"}),
                "mask_fill_holes": ("INT", {"default": 0, "min": 0, "max": 128, "step": 1,
                    "tooltip": "DEPRECATED — use cleanup_fill_holes"}),
                "mask_remove_noise": ("INT", {"default": 0, "min": 0, "max": 32, "step": 1,
                    "tooltip": "DEPRECATED — use cleanup_remove_noise"}),
                "mask_smooth": ("INT", {"default": 0, "min": 0, "max": 127, "step": 1,
                    "tooltip": "DEPRECATED — use cleanup_smooth"}),
                "mask_blend_pixels": ("INT", {"default": 16, "min": 0, "max": 64, "step": 1,
                    "tooltip": "DEPRECATED — use crop_blend_feather_px"}),
                "stitch_erosion": ("INT", {"default": 0, "min": -32, "max": 32, "step": 1,
                    "tooltip": "DEPRECATED — use vace_stitch_erosion_px"}),
                "stitch_feather": ("INT", {"default": 4, "min": 0, "max": 64, "step": 1,
                    "tooltip": "DEPRECATED — use vace_stitch_feather_px"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("visualization",)
    FUNCTION = "execute"
    CATEGORY = "NV_Utils/Debug"
    DESCRIPTION = (
        "Debug visualization of mask processing stages. "
        "grid: 3x2 contact sheet. batch: 6 clickable images. video: temporal scrub for one stage. "
        "Connect cropped_image + cropped_mask from InpaintCrop to preview the crop the model sees."
    )

    def execute(self, image, original_mask, mode, frame_index, video_stage, overlay_alpha,
                crop_expand_px, cleanup_fill_holes, cleanup_remove_noise,
                cleanup_smooth, crop_blend_feather_px,
                erosion_blocks, feather_blocks, vae_stride,
                vace_stitch_erosion_px, vace_stitch_feather_px,
                mask_config=None, bbox_mask=None,
                cropped_image=None, cropped_mask=None,
                # Deprecated names (backward compat)
                mask_erode_dilate=4, mask_fill_holes=0, mask_remove_noise=0,
                mask_smooth=0, mask_blend_pixels=16,
                stitch_erosion=0, stitch_feather=4):

        # Resolve deprecated param names
        from .mask_processing_config import resolve_deprecated, apply_mask_config, apply_vace_mask_config
        crop_expand_px = resolve_deprecated(crop_expand_px, 4, mask_erode_dilate, 4)
        cleanup_fill_holes = resolve_deprecated(cleanup_fill_holes, 0, mask_fill_holes, 0)
        cleanup_remove_noise = resolve_deprecated(cleanup_remove_noise, 0, mask_remove_noise, 0)
        cleanup_smooth = resolve_deprecated(cleanup_smooth, 0, mask_smooth, 0)
        crop_blend_feather_px = resolve_deprecated(crop_blend_feather_px, 16, mask_blend_pixels, 16)
        vace_stitch_erosion_px = resolve_deprecated(vace_stitch_erosion_px, 0, stitch_erosion, 0)
        vace_stitch_feather_px = resolve_deprecated(vace_stitch_feather_px, 4, stitch_feather, 4)

        # Apply shared config override if connected
        vals = apply_mask_config(mask_config,
            crop_expand_px=crop_expand_px,
            cleanup_fill_holes=cleanup_fill_holes,
            cleanup_remove_noise=cleanup_remove_noise,
            cleanup_smooth=cleanup_smooth,
            crop_blend_feather_px=crop_blend_feather_px,
        )
        crop_expand_px = vals["crop_expand_px"]
        cleanup_fill_holes = vals["cleanup_fill_holes"]
        cleanup_remove_noise = vals["cleanup_remove_noise"]
        cleanup_smooth = vals["cleanup_smooth"]
        crop_blend_feather_px = vals["crop_blend_feather_px"]
        vace_vals = apply_vace_mask_config(mask_config,
            erosion_blocks=erosion_blocks,
            feather_blocks=feather_blocks,
            vace_stitch_erosion_px=vace_stitch_erosion_px,
            vace_stitch_feather_px=vace_stitch_feather_px,
        )
        erosion_blocks = vace_vals["erosion_blocks"]
        feather_blocks = vace_vals["feather_blocks"]
        vace_stitch_erosion_px = vace_vals["vace_stitch_erosion_px"]
        vace_stitch_feather_px = vace_vals["vace_stitch_feather_px"]

        # Common mask processing kwargs
        mask_kwargs = dict(
            mask_erode_dilate=crop_expand_px,
            mask_fill_holes=cleanup_fill_holes,
            mask_remove_noise=cleanup_remove_noise,
            mask_smooth=cleanup_smooth,
            mask_blend_pixels=crop_blend_feather_px,
            erosion_blocks=erosion_blocks,
            feather_blocks=feather_blocks,
            vae_stride=vae_stride,
            stitch_erosion=vace_stitch_erosion_px,
            stitch_feather=vace_stitch_feather_px,
        )

        # Resolve image/mask source: use cropped inputs if connected, otherwise full-frame
        if cropped_image is not None and cropped_mask is not None:
            src_image = cropped_image
            src_mask = cropped_mask if cropped_mask.dim() == 3 else cropped_mask.unsqueeze(0)
            src_bbox = None  # bbox doesn't apply to already-cropped images
        else:
            src_image = image
            src_mask = original_mask if original_mask.dim() == 3 else original_mask.unsqueeze(0)
            src_bbox = bbox_mask

        if mode == "video":
            return self._mode_video(src_image, src_mask, src_bbox, video_stage, overlay_alpha, mask_kwargs)
        else:
            return self._mode_single_frame(src_image, src_mask, src_bbox, frame_index, overlay_alpha, mask_kwargs, grid=(mode == "grid"))

    def _get_frame_data(self, src_image, src_mask, src_bbox, fi):
        """Extract and validate frame + mask + optional bbox for frame index fi."""
        B = src_image.shape[0]
        fi = min(fi, B - 1)

        frame = src_image[fi]  # [H, W, C]
        H, W = frame.shape[0], frame.shape[1]

        mask_fi = min(fi, src_mask.shape[0] - 1)
        mask_frame = src_mask[mask_fi:mask_fi + 1]  # [1, H, W]
        if mask_frame.shape[1] != H or mask_frame.shape[2] != W:
            mask_frame = rescale_mask(mask_frame, W, H)

        bbox_frame = None
        if src_bbox is not None:
            if src_bbox.dim() == 2:
                src_bbox = src_bbox.unsqueeze(0)
            bbox_fi = min(fi, src_bbox.shape[0] - 1)
            bbox_frame = src_bbox[bbox_fi:bbox_fi + 1]
            if bbox_frame.shape[1] != H or bbox_frame.shape[2] != W:
                bbox_frame = rescale_mask(bbox_frame, W, H)

        frame_np = (frame.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        return frame_np, mask_frame, bbox_frame, H, W

    def _mode_single_frame(self, src_image, src_mask, src_bbox, frame_index, overlay_alpha, mask_kwargs, grid=False):
        """Grid or batch mode: process one frame, output all 6 stages."""
        frame_np, mask_frame, bbox_frame, H, W = self._get_frame_data(src_image, src_mask, src_bbox, frame_index)
        font = _get_font(max(16, min(H // 20, 28)))

        masks = _compute_masks_for_frame(mask_frame, bbox_frame, **mask_kwargs)

        panels = []
        for key, label, color in STAGE_CONFIGS:
            panel = _render_overlay(frame_np, masks[key], key, overlay_alpha, font)
            panels.append(panel)

        if grid:
            # 3x2 contact sheet
            row1 = np.concatenate([panels[0], panels[1]], axis=1)
            row2 = np.concatenate([panels[2], panels[3]], axis=1)
            row3 = np.concatenate([panels[4], panels[5]], axis=1)
            contact_sheet = np.concatenate([row1, row2, row3], axis=0)
            result = torch.from_numpy(contact_sheet.astype(np.float32) / 255.0)
            result = result.unsqueeze(0)  # [1, 3*H, 2*W, 3]
        else:
            # Batch: 6 separate images as [6, H, W, 3]
            tensors = []
            for panel in panels:
                t = torch.from_numpy(panel.astype(np.float32) / 255.0)
                tensors.append(t)
            result = torch.stack(tensors, dim=0)  # [6, H, W, 3]

        return (result,)

    def _mode_video(self, src_image, src_mask, src_bbox, video_stage, overlay_alpha, mask_kwargs):
        """Video mode: process all frames for a single selected stage."""
        B = src_image.shape[0]
        color = STAGE_COLORS[video_stage]
        label_base = STAGE_LABELS[video_stage]

        frames_out = []
        for fi in range(B):
            frame_np, mask_frame, bbox_frame, H, W = self._get_frame_data(src_image, src_mask, src_bbox, fi)
            font = _get_font(max(16, min(H // 20, 28)))

            masks = _compute_masks_for_frame(mask_frame, bbox_frame, **mask_kwargs)
            panel = _render_overlay(frame_np, masks[video_stage], video_stage, overlay_alpha, font,
                                    frame_label=f"F{fi}")
            t = torch.from_numpy(panel.astype(np.float32) / 255.0)
            frames_out.append(t)

        result = torch.stack(frames_out, dim=0)  # [B, H, W, 3]
        return (result,)


NODE_CLASS_MAPPINGS = {
    "NV_MaskPipelineViz": NV_MaskPipelineViz,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_MaskPipelineViz": "NV Mask Pipeline Viz",
}
