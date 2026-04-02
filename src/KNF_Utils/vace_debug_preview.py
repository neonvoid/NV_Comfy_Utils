"""
NV VACE Debug Preview — visual diagnostic for VACE inpainting mask pipeline.

Shows the processed mask and grey control video side-by-side so you can SEE
what VACE is actually working with. Also outputs a cumulative expansion summary
so you can audit the total effect of all mask parameters without mental math.

Wire after NV_VaceControlVideoPrep:
  VaceControlVideoPrep.control_video  →  VaceDebugPreview.control_video
  VaceControlVideoPrep.control_masks  →  VaceDebugPreview.control_masks
  VaceControlVideoPrep.stitch_mask    →  VaceDebugPreview.stitch_mask (optional)
  source image                        →  VaceDebugPreview.image
"""

import torch
import numpy as np


def _overlay_mask(image_np, mask_np, color, alpha=0.5):
    """Soft overlay: blend a colored mask onto an image.

    Args:
        image_np: [H, W, 3] float32 in [0, 1]
        mask_np: [H, W] float32 in [0, 1]
        color: (R, G, B) tuple in [0, 1]
        alpha: overlay opacity

    Returns:
        [H, W, 3] float32 in [0, 1]
    """
    result = image_np.copy()
    for c in range(3):
        result[:, :, c] = image_np[:, :, c] * (1.0 - alpha * mask_np) + color[c] * (alpha * mask_np)
    return np.clip(result, 0.0, 1.0)


def _draw_label(image_np, text, y=10, color=(1.0, 1.0, 1.0)):
    """Draw a simple text label by burning white pixels. Crude but dependency-free."""
    # Simple 5x3 pixel font for uppercase + digits (enough for labels)
    # Skip actual font rendering — just add a colored bar with no text dependency
    H, W, _ = image_np.shape
    bar_h = min(24, H // 20)
    if bar_h < 4:
        return image_np
    result = image_np.copy()
    result[y:y + bar_h, :, :] = result[y:y + bar_h, :, :] * 0.3  # darken bar
    return result


def _resize_to_match(tensor, target_h, target_w):
    """Resize a [B, H, W, C] or [B, H, W] tensor to target dimensions."""
    if tensor.dim() == 4:
        # IMAGE [B, H, W, C] -> [B, C, H, W] for interpolate
        t = tensor.permute(0, 3, 1, 2)
        t = torch.nn.functional.interpolate(t, size=(target_h, target_w), mode='bilinear', align_corners=False)
        return t.permute(0, 2, 3, 1)
    elif tensor.dim() == 3:
        # MASK [B, H, W]
        t = tensor.unsqueeze(1)
        t = torch.nn.functional.interpolate(t, size=(target_h, target_w), mode='bilinear', align_corners=False)
        return t.squeeze(1)
    return tensor


class NV_VaceDebugPreview:
    """Visual diagnostic for VACE mask pipeline — see what VACE actually gets."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "control_video": ("IMAGE",),
                "control_masks": ("MASK",),
                "mode": (["side_by_side", "overlay", "grid"], {
                    "default": "side_by_side",
                    "tooltip": "side_by_side: [original | control_video | mask overlay] horizontal. "
                               "overlay: single frame with VACE mask (magenta) + stitch mask (cyan). "
                               "grid: 2x2 [original, control_video, VACE mask, stitch mask]."
                }),
            },
            "optional": {
                "stitch_mask": ("MASK", {
                    "tooltip": "Stitch boundary mask from VaceControlVideoPrep. Shown in cyan overlay."
                }),
                "mask_config": ("MASK_PROCESSING_CONFIG", {
                    "tooltip": "Config dict — used to compute cumulative expansion summary."
                }),
                "overlay_alpha": ("FLOAT", {
                    "default": 0.5, "min": 0.1, "max": 0.9, "step": 0.05,
                    "tooltip": "Mask overlay opacity."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("preview", "expansion_info")
    FUNCTION = "execute"
    CATEGORY = "NV_Utils/Debug"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Visual diagnostic for VACE inpainting. Shows the processed mask and grey "
        "control video so you can see what VACE is working with. Outputs cumulative "
        "expansion summary to audit total mask expansion without mental math."
    )

    def execute(self, image, control_video, control_masks, mode="side_by_side",
                stitch_mask=None, mask_config=None, overlay_alpha=0.5):

        B = image.shape[0]
        H, W = image.shape[1], image.shape[2]

        # Ensure control_video and masks match image dimensions
        if control_video.shape[1] != H or control_video.shape[2] != W:
            control_video = _resize_to_match(control_video, H, W)
        if control_masks.shape[1] != H or control_masks.shape[2] != W:
            control_masks = _resize_to_match(control_masks, H, W)
        if stitch_mask is not None and (stitch_mask.shape[1] != H or stitch_mask.shape[2] != W):
            stitch_mask = _resize_to_match(stitch_mask, H, W)

        # Match batch sizes
        ctrl_B = control_video.shape[0]
        mask_B = control_masks.shape[0]
        if ctrl_B < B:
            control_video = control_video[-1:].expand(B, -1, -1, -1)
        if mask_B < B:
            control_masks = control_masks[-1:].expand(B, -1, -1)
        if stitch_mask is not None and stitch_mask.shape[0] < B:
            stitch_mask = stitch_mask[-1:].expand(B, -1, -1)

        frames = []
        for b in range(B):
            img = image[b].cpu().numpy().astype(np.float32)
            ctrl = control_video[b].cpu().numpy().astype(np.float32)
            vace_m = control_masks[b].cpu().numpy().astype(np.float32)
            stitch_m = stitch_mask[b].cpu().numpy().astype(np.float32) if stitch_mask is not None else None

            if mode == "side_by_side":
                # [original | control_video | VACE mask overlay on original]
                overlay = _overlay_mask(img, vace_m, (0.86, 0.31, 1.0), overlay_alpha)  # magenta
                if stitch_m is not None:
                    overlay = _overlay_mask(overlay, stitch_m, (0.31, 0.86, 1.0), overlay_alpha * 0.5)  # cyan
                frame = np.concatenate([img, ctrl, overlay], axis=1)  # [H, 3W, 3]

            elif mode == "overlay":
                # Single frame with both masks overlaid
                overlay = _overlay_mask(img, vace_m, (0.86, 0.31, 1.0), overlay_alpha)
                if stitch_m is not None:
                    overlay = _overlay_mask(overlay, stitch_m, (0.31, 0.86, 1.0), overlay_alpha * 0.5)
                frame = overlay

            elif mode == "grid":
                # 2x2: [original, control_video] / [VACE mask overlay, stitch mask overlay]
                vace_overlay = _overlay_mask(img, vace_m, (0.86, 0.31, 1.0), overlay_alpha)
                if stitch_m is not None:
                    stitch_overlay = _overlay_mask(img, stitch_m, (0.31, 0.86, 1.0), overlay_alpha)
                else:
                    stitch_overlay = img.copy()
                top = np.concatenate([img, ctrl], axis=1)
                bottom = np.concatenate([vace_overlay, stitch_overlay], axis=1)
                frame = np.concatenate([top, bottom], axis=0)  # [2H, 2W, 3]
            else:
                frame = img

            frames.append(frame)

        # Stack to tensor [B, H', W', 3]
        result = torch.from_numpy(np.stack(frames, axis=0)).float().clamp(0.0, 1.0)

        # Build expansion info
        info_lines = []
        if mask_config is not None:
            grow = mask_config.get("vace_input_grow_px", mask_config.get("mask_grow", 0))
            expand = mask_config.get("crop_expand_px", mask_config.get("mask_erode_dilate", 0))
            halo = mask_config.get("vace_halo_px", 0)
            erosion_blocks = mask_config.get("vace_erosion_blocks", 0.5)
            feather_blocks = mask_config.get("vace_feather_blocks", 1.5)
            blend = mask_config.get("crop_blend_feather_px", mask_config.get("mask_blend_pixels", 16))
            stitch_ero = mask_config.get("vace_stitch_erosion_px", mask_config.get("vace_stitch_erosion", 0))
            stitch_fea = mask_config.get("vace_stitch_feather_px", mask_config.get("vace_stitch_feather", 8))
            vae_stride = 8  # WAN default

            erosion_px = erosion_blocks * vae_stride
            feather_px = feather_blocks * vae_stride

            info_lines.append(f"Input mask grow:    {grow:+d} px")
            info_lines.append(f"Crop expansion:     {expand:+d} px")
            info_lines.append(f"VACE halo:          {halo:+d} px")
            info_lines.append(f"VACE erosion:       {-erosion_px:+.0f} px (inward)")
            info_lines.append(f"VACE feather:       {feather_px:.0f} px")
            info_lines.append(f"Blend feather:      {blend:d} px")
            info_lines.append(f"Stitch erosion:     {stitch_ero:+d} px")
            info_lines.append(f"Stitch feather:     {stitch_fea:d} px")
            info_lines.append("─" * 30)
            approx_outward = grow + expand + halo - erosion_px
            info_lines.append(f"Approx outward:     ~{approx_outward:.0f} px")
        else:
            info_lines.append("Connect mask_config for expansion summary")

        expansion_info = "\n".join(info_lines)
        print(f"[NV_VaceDebugPreview] {B} frames, mode={mode}")

        return (result, expansion_info)


NODE_CLASS_MAPPINGS = {
    "NV_VaceDebugPreview": NV_VaceDebugPreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_VaceDebugPreview": "NV VACE Debug Preview",
}
