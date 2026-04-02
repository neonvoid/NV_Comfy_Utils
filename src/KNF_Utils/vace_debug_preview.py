"""
NV VACE Debug Preview — visual diagnostic for VACE inpainting mask pipeline.

Shows the processed mask and grey control video side-by-side so you can SEE
what VACE is actually working with. Also outputs a cumulative expansion summary
so you can audit the total effect of all mask parameters without mental math.

Wire after NV_VaceControlVideoPrep + NV_InpaintCrop2:
  VaceControlVideoPrep.control_video  →  VaceDebugPreview.control_video
  VaceControlVideoPrep.control_masks  →  VaceDebugPreview.control_masks
  VaceControlVideoPrep.stitch_mask    →  VaceDebugPreview.stitch_mask (optional)
  InpaintCrop2.stitcher               →  VaceDebugPreview.stitcher (optional)
  source image                        →  VaceDebugPreview.image
"""

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont


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


def _get_font(size=14):
    """Load a font with fallback chain."""
    for name in ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except (OSError, IOError):
            continue
    try:
        return ImageFont.load_default(size)
    except TypeError:
        return ImageFont.load_default()


def _draw_label(image_np, text, position="top_left"):
    """Draw a text label on an image using PIL. Returns modified copy."""
    H, W, _ = image_np.shape
    pil_img = Image.fromarray((image_np * 255).astype(np.uint8))
    draw = ImageDraw.Draw(pil_img)
    font_size = max(12, min(24, H // 20))
    font = _get_font(font_size)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = 4
    if position == "top_left":
        x, y = pad, pad
    elif position == "top_center":
        x, y = (W - tw) // 2, pad
    else:
        x, y = pad, pad
    # Dark background for readability
    draw.rectangle([x - pad, y - pad, x + tw + pad, y + th + pad], fill=(0, 0, 0, 180))
    draw.text((x, y), text, fill=(255, 255, 255), font=font)
    return np.array(pil_img).astype(np.float32) / 255.0


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
                "stitcher": ("STITCHER", {
                    "tooltip": "Stitcher from InpaintCrop2. Shows the crop blend mask (yellow) — "
                               "this is what actually controls the pixel-space composite boundary."
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
                stitch_mask=None, stitcher=None, mask_config=None, overlay_alpha=0.5):

        B = image.shape[0]
        H, W = image.shape[1], image.shape[2]

        # Ensure control_video and masks match image dimensions
        if control_video.shape[1] != H or control_video.shape[2] != W:
            control_video = _resize_to_match(control_video, H, W)
        if control_masks.shape[1] != H or control_masks.shape[2] != W:
            control_masks = _resize_to_match(control_masks, H, W)
        if stitch_mask is not None and (stitch_mask.shape[1] != H or stitch_mask.shape[2] != W):
            stitch_mask = _resize_to_match(stitch_mask, H, W)

        # Extract crop blend mask from stitcher (if connected)
        # This is the actual mask InpaintStitch2 uses for compositing
        crop_blend_masks = None
        crop_info = ""
        if stitcher is not None:
            blend_list = stitcher.get("cropped_mask_for_blend", [])
            stitch_source = stitcher.get("stitch_source", "?")
            crop_target_w = stitcher.get("crop_target_w", "?")
            crop_target_h = stitcher.get("crop_target_h", "?")
            crop_info = f"Crop: {crop_target_w}x{crop_target_h}, stitch_source={stitch_source}"
            if blend_list:
                # Blend masks are in crop space — resize to full frame for overlay
                # They're [H_crop, W_crop] tensors stored per-frame
                crop_blend_masks = []
                for bm in blend_list:
                    if bm.dim() == 2:
                        bm = bm.unsqueeze(0)  # [1, H, W]
                    bm = _resize_to_match(bm, H, W)
                    crop_blend_masks.append(bm.squeeze(0))

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
            crop_bm = None
            if crop_blend_masks is not None and b < len(crop_blend_masks):
                crop_bm = crop_blend_masks[b].cpu().numpy().astype(np.float32)

            if mode == "side_by_side":
                # [original | control_video | VACE mask | crop blend mask]
                img_labeled = _draw_label(img, "Original", "top_left")
                ctrl_labeled = _draw_label(ctrl, "VACE Control Video", "top_left")
                vace_overlay = _overlay_mask(img, vace_m, (0.86, 0.31, 1.0), overlay_alpha)
                if stitch_m is not None:
                    vace_overlay = _overlay_mask(vace_overlay, stitch_m, (0.31, 0.86, 1.0), overlay_alpha * 0.5)
                vace_labeled = _draw_label(vace_overlay, "VACE mask (magenta) + stitch (cyan)", "top_left")

                panels = [img_labeled, ctrl_labeled, vace_labeled]

                if crop_bm is not None:
                    crop_overlay = _overlay_mask(img, crop_bm, (1.0, 0.85, 0.2), overlay_alpha)  # yellow
                    crop_labeled = _draw_label(crop_overlay, "Crop Blend Mask (yellow)", "top_left")
                    panels.append(crop_labeled)

                frame = np.concatenate(panels, axis=1)

            elif mode == "overlay":
                overlay = _overlay_mask(img, vace_m, (0.86, 0.31, 1.0), overlay_alpha)
                if stitch_m is not None:
                    overlay = _overlay_mask(overlay, stitch_m, (0.31, 0.86, 1.0), overlay_alpha * 0.5)
                if crop_bm is not None:
                    overlay = _overlay_mask(overlay, crop_bm, (1.0, 0.85, 0.2), overlay_alpha * 0.3)
                frame = _draw_label(overlay, "VACE (magenta) + stitch (cyan) + crop blend (yellow)", "top_left")

            elif mode == "grid":
                # 2x2: [original, control_video] / [VACE mask, crop blend mask]
                img_labeled = _draw_label(img, "Original", "top_left")
                ctrl_labeled = _draw_label(ctrl, "VACE Control Video", "top_left")
                vace_overlay = _overlay_mask(img, vace_m, (0.86, 0.31, 1.0), overlay_alpha)
                vace_labeled = _draw_label(vace_overlay, "VACE Cond Mask", "top_left")
                if crop_bm is not None:
                    crop_overlay = _overlay_mask(img, crop_bm, (1.0, 0.85, 0.2), overlay_alpha)
                    crop_labeled = _draw_label(crop_overlay, "Crop Blend Mask", "top_left")
                elif stitch_m is not None:
                    crop_labeled = _draw_label(
                        _overlay_mask(img, stitch_m, (0.31, 0.86, 1.0), overlay_alpha),
                        "VACE Stitch Mask", "top_left")
                else:
                    crop_labeled = _draw_label(img.copy(), "No stitch data", "top_left")
                top = np.concatenate([img_labeled, ctrl_labeled], axis=1)
                bottom = np.concatenate([vace_labeled, crop_labeled], axis=1)
                frame = np.concatenate([top, bottom], axis=0)
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

            info_lines.append("── VACE Side ──")
            info_lines.append(f"Input mask grow:    {grow:+d} px")
            info_lines.append(f"VACE halo:          {halo:+d} px")
            info_lines.append(f"VACE erosion:       {-erosion_px:+.0f} px (inward)")
            info_lines.append(f"VACE feather:       {feather_px:.0f} px")
            info_lines.append(f"VACE stitch erosion:{stitch_ero:+d} px")
            info_lines.append(f"VACE stitch feather:{stitch_fea:d} px")
            info_lines.append("")
            info_lines.append("── Crop/Stitch Side ──")
            info_lines.append(f"Crop expansion:     {expand:+d} px")
            info_lines.append(f"Blend feather:      {blend:d} px")
            if crop_info:
                info_lines.append(f"{crop_info}")
            info_lines.append("")
            info_lines.append("── Net Effect ──")
            vace_outward = grow + halo - erosion_px
            stitch_outward = expand + blend
            gap = vace_outward - stitch_outward
            info_lines.append(f"VACE outward:       ~{vace_outward:.0f} px (grow + halo - erosion)")
            info_lines.append(f"Stitch outward:     ~{stitch_outward:.0f} px (expand + feather)")
            info_lines.append(f"Gap (VACE - stitch): ~{gap:.0f} px")

            # ── Diagnosis + Suggestions ──
            info_lines.append("")
            info_lines.append("── Suggestions ──")
            suggestions = []

            if gap > 8:
                # VACE extends well past stitch — grey fill visible at boundary
                fix_expand = int(expand + gap)
                suggestions.append(f"GREY VISIBLE: VACE extends {gap:.0f}px past stitch boundary")
                suggestions.append(f"  Fix A: crop_expand_px {expand:+d} → {fix_expand:+d} (expand stitch to match)")
                suggestions.append(f"  Fix B: vace_halo_px {halo} → {max(0, halo - int(gap))} (shrink VACE to match)")
                if grow > 0:
                    suggestions.append(f"  Fix C: vace_input_grow_px {grow:+d} → {max(0, grow - int(gap)):+d} (less input growth)")

            elif gap < -8:
                # Stitch extends past VACE — original/unregenerated content in blend zone
                suggestions.append(f"STITCH OVERREACH: stitch extends {-gap:.0f}px past VACE boundary")
                suggestions.append(f"  Fix A: crop_expand_px {expand:+d} → {max(-128, expand + int(gap)):+d} (shrink stitch)")
                suggestions.append(f"  Fix B: crop_blend_feather_px {blend} → {max(4, blend + int(gap))} (less feather)")
                suggestions.append(f"  Fix C: vace_halo_px {halo} → {halo + int(-gap)} (extend VACE to cover)")

            else:
                suggestions.append("OK: VACE and stitch boundaries are well-matched")

            # Check for overmasking risk (total expansion > 40px)
            total_out = max(vace_outward, stitch_outward)
            if total_out > 40:
                suggestions.append(f"")
                suggestions.append(f"OVERMASKING RISK: ~{total_out:.0f}px total outward expansion")
                suggestions.append(f"  Consider: vace_input_grow_px=0 + mask_shape=bbox if adding new object")
                suggestions.append(f"  Or reduce: total of grow({grow}) + halo({halo}) + expand({expand}) + feather({blend})")

            # Check for dark seam risk
            if erosion_blocks == 0:
                suggestions.append("")
                suggestions.append("DARK SEAM RISK: vace_erosion_blocks=0 — set to 0.5 minimum")
            if feather_blocks == 0:
                suggestions.append("")
                suggestions.append("DARK SEAM RISK: vace_feather_blocks=0 — set to 1.0 minimum")

            # Check for clipping (stitch has no expansion but VACE generates beyond mask)
            if expand == 0 and grow > 0:
                suggestions.append("")
                suggestions.append(f"CLIP RISK: vace_input_grow_px={grow} but crop_expand_px=0")
                suggestions.append(f"  Generated content may extend beyond stitch boundary")
                suggestions.append(f"  Set crop_expand_px to {min(grow, 16):+d} to cover generated area")

            for s in suggestions:
                info_lines.append(s)
        else:
            info_lines.append("Connect mask_config for expansion summary")
            if crop_info:
                info_lines.append(crop_info)

        expansion_info = "\n".join(info_lines)
        print(f"[NV_VaceDebugPreview] {B} frames, mode={mode}")

        return (result, expansion_info)


NODE_CLASS_MAPPINGS = {
    "NV_VaceDebugPreview": NV_VaceDebugPreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_VaceDebugPreview": "NV VACE Debug Preview",
}
