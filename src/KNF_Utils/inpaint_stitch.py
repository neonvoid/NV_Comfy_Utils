"""
NV Inpaint Stitch v2 — Composites inpainted crops back into the original frames.

Pairs with NV_InpaintCrop v2 (inpaint_crop.py). Takes the STITCHER dict produced
by the crop node plus the inpainted image batch, and blends each frame back into
its canvas at the stored coordinates using the stored blend mask.

Handles both stitcher formats:
  - v2 (NV_InpaintCrop v2): 'resize_algorithm' (single key)
  - v1 (comfy-inpaint-crop-fork): 'downscale_algorithm' / 'upscale_algorithm'

Frame skipping: if the crop node skipped frames (empty mask), the stitch node
reinserts the original frames at their correct positions in the output batch.
"""

import torch
import torch.nn.functional as TF
import comfy.model_management

from .inpaint_crop import rescale_image, rescale_mask
from .multiband_blend_stitch import multiband_blend


# =============================================================================
# Inverse Content Warp
# =============================================================================

def _inverse_content_warp(image, mask, warp_mode, warp_entry):
    """Apply inverse content warp to undo crop stabilization before blending.

    Args:
        image: [1, H, W, C] — resized inpainted crop.
        mask: [1, H, W, 1] — blend mask (already clamped+expanded).
        warp_mode: 'centroid' or 'optical_flow'.
        warp_entry: dict with warp data for this frame.

    Returns:
        (image, mask) with inverse warp applied.
    """
    device = image.device
    _, H, W, C = image.shape

    if warp_mode == "centroid":
        dx = -warp_entry["dx"]
        dy = -warp_entry["dy"]
        norm_dx = 2.0 * dx / W
        norm_dy = 2.0 * dy / H
        theta = torch.tensor([
            [1.0, 0.0, norm_dx],
            [0.0, 1.0, norm_dy]
        ], device=device, dtype=torch.float32).unsqueeze(0)
        grid = TF.affine_grid(theta, (1, 1, H, W), align_corners=False)

        img_nchw = image.permute(0, 3, 1, 2)
        image = TF.grid_sample(img_nchw, grid, mode='bilinear',
                               padding_mode='border', align_corners=False).permute(0, 2, 3, 1)

        mask_nchw = mask.permute(0, 3, 1, 2)
        mask = TF.grid_sample(mask_nchw, grid, mode='bilinear',
                              padding_mode='zeros', align_corners=False).permute(0, 2, 3, 1)

    elif warp_mode == "optical_flow":
        flow = warp_entry["flow"].to(device)  # Forward flow: maps src→ref, used as-is for inverse grid
        if flow.dim() == 3:
            flow = flow.unsqueeze(0)

        y = torch.arange(H, device=device, dtype=torch.float32)
        x = torch.arange(W, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(y, x, indexing='ij')

        xx_warped = xx + flow[0, 0]
        yy_warped = yy + flow[0, 1]
        xx_norm = 2.0 * xx_warped / (W - 1) - 1.0
        yy_norm = 2.0 * yy_warped / (H - 1) - 1.0
        grid = torch.stack([xx_norm, yy_norm], dim=-1).unsqueeze(0)

        img_nchw = image.permute(0, 3, 1, 2)
        image = TF.grid_sample(img_nchw, grid, mode='bilinear',
                               padding_mode='border', align_corners=True).permute(0, 2, 3, 1)

        mask_nchw = mask.permute(0, 3, 1, 2)
        mask = TF.grid_sample(mask_nchw, grid, mode='bilinear',
                              padding_mode='zeros', align_corners=True).permute(0, 2, 3, 1)

    return image, mask


# =============================================================================
# Core Stitch Function
# =============================================================================

def stitch_single_frame(canvas_image, inpainted_image, mask,
                        ctc_x, ctc_y, ctc_w, ctc_h,
                        cto_x, cto_y, cto_w, cto_h,
                        resize_algorithm,
                        warp_mode=None, warp_entry=None,
                        blend_mode="alpha", multiband_levels=5):
    """Blend one inpainted crop back into its canvas and extract the original region.

    Args:
        canvas_image: [H, W, C] or [1, H, W, C] — the padded original frame.
        inpainted_image: [1, H, W, C] — the inpainted crop at target resolution.
        mask: [H, W] or [1, H, W] — blend mask at crop resolution.
        ctc_x/y/w/h: Where the crop sits on the canvas.
        cto_x/y/w/h: Where the original image sits on the canvas.
        resize_algorithm: Interpolation method for resizing crop back to canvas scale.
        warp_mode: Optional content warp mode ('centroid' or 'optical_flow').
        warp_entry: Optional per-frame warp data dict.
        blend_mode: 'alpha' (standard), 'multiband' (Laplacian pyramid), or 'hard' (no blend).
        multiband_levels: Pyramid levels for multiband mode (default 5).

    Returns:
        [1, cto_h, cto_w, C] output image (original image region with inpainted area blended in).
    """
    device = canvas_image.device

    # Ensure canvas has batch dim
    if canvas_image.dim() == 3:
        canvas_image = canvas_image.unsqueeze(0)

    # Ensure mask has batch dim
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)

    # Inverse content warp BEFORE resize: dx/dy are in the crop's working resolution
    # (target res from InpaintCrop), so must be applied before downsizing to canvas scale.
    # Only warp the IMAGE, not the blend mask — the mask was created at InpaintCrop time
    # in original (non-stabilized) coordinates and was never forward-warped by CoTrackerBridge.
    # After inverse-warping the image back to original coordinates, the mask already aligns.
    if warp_mode is not None and warp_entry is not None:
        # Create a dummy mask (all-ones) for the warp function signature — we discard it
        dummy_mask = torch.ones(1, inpainted_image.shape[1], inpainted_image.shape[2], 1,
                                device=device, dtype=inpainted_image.dtype)
        inpainted_image, _ = _inverse_content_warp(
            inpainted_image, dummy_mask, warp_mode, warp_entry
        )

    # Resize inpainted image and mask to canvas crop size
    _, h, w, _ = inpainted_image.shape
    if ctc_w != w or ctc_h != h:
        resized_image = rescale_image(inpainted_image, ctc_w, ctc_h, resize_algorithm)
        resized_mask = rescale_mask(mask, ctc_w, ctc_h, resize_algorithm)
    else:
        resized_image = inpainted_image
        resized_mask = mask

    # Clamp mask and expand to match image channels
    resized_mask = resized_mask.clamp(0, 1).unsqueeze(-1)  # [1, H, W, 1]

    # Extract canvas region, blend, paste back
    canvas_crop = canvas_image[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w, :]

    if blend_mode == "multiband":
        # Laplacian pyramid: blend each frequency band at appropriate spatial scale.
        # Operates on [B,C,H,W] — convert from [1,H,W,C].
        inp_nchw = resized_image.permute(0, 3, 1, 2)
        cvs_nchw = canvas_crop.permute(0, 3, 1, 2)
        mask_nchw = resized_mask.permute(0, 3, 1, 2)[:, :1]  # [1, 1, H, W]
        blended_nchw = multiband_blend(inp_nchw, cvs_nchw, mask_nchw, num_levels=multiband_levels)
        blended = blended_nchw.clamp(0.0, 1.0).permute(0, 2, 3, 1)
    elif blend_mode == "hard":
        # Hard paste: no blending, just overwrite (useful for latent path comparisons)
        blended = resized_mask * resized_image + (1.0 - resized_mask) * canvas_crop
    else:
        # Standard alpha composite
        blended = resized_mask * resized_image + (1.0 - resized_mask) * canvas_crop

    canvas_image[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w, :] = blended

    # Extract the original image area from canvas
    output_image = canvas_image[:, cto_y:cto_y + cto_h, cto_x:cto_x + cto_w, :]

    return output_image


def _get_resize_algorithm(stitcher):
    """Extract resize algorithm from stitcher, handling both v1 and v2 formats."""
    # v2 format (NV_InpaintCrop v2): single 'resize_algorithm'
    if 'resize_algorithm' in stitcher:
        return stitcher['resize_algorithm']
    # v1 format (fork): separate up/downscale — just use upscale as default
    return stitcher.get('upscale_algorithm', stitcher.get('downscale_algorithm', 'bicubic'))


# =============================================================================
# Node Class
# =============================================================================

class NV_InpaintStitch:
    """Stitch inpainted crops back into original frames using STITCHER metadata."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stitcher": ("STITCHER",),
                "inpainted_image": ("IMAGE",),
                "blend_mode": (["alpha", "multiband", "hard"], {
                    "default": "alpha",
                    "tooltip": "alpha: standard feathered blend (default). "
                               "multiband: Laplacian pyramid — blends low freq broadly, "
                               "high freq narrowly (best for VAE roundtrip seams). "
                               "hard: binary mask paste, no feathering."
                }),
            },
            "optional": {
                "multiband_levels": ("INT", {
                    "default": 5, "min": 2, "max": 8, "step": 1,
                    "tooltip": "Pyramid levels for multiband mode. 5 = good for 720p-1080p. "
                               "More levels = broader low-frequency blending."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "stitch"
    CATEGORY = "NV_Utils/Inpaint"
    DESCRIPTION = (
        "Composites inpainted crops back into the original frames. "
        "Pairs with NV Inpaint Crop v2. Handles frame skipping automatically. "
        "Supports alpha, multiband (Laplacian pyramid), or hard blend modes."
    )

    def stitch(self, stitcher, inpainted_image, blend_mode="alpha", multiband_levels=5):
        device = comfy.model_management.get_torch_device()
        intermediate = comfy.model_management.intermediate_device()
        resize_algorithm = _get_resize_algorithm(stitcher)

        skipped_indices = set(stitcher.get('skipped_indices', []))
        original_frames = stitcher.get('original_frames', [])
        total_frames = stitcher.get('total_frames', inpainted_image.shape[0])

        # Content warp data (backward-compatible — absent in old stitchers)
        content_warp_mode = stitcher.get('content_warp_mode', None)
        content_warp_data = stitcher.get('content_warp_data', None)

        results = []

        if len(skipped_indices) == 0:
            # Simple path: all frames were inpainted
            batch_size = inpainted_image.shape[0]
            num_coords = len(stitcher['cropped_to_canvas_x'])
            single_stitcher = (num_coords == 1 and batch_size > 1)

            for b in range(batch_size):
                idx = 0 if single_stitcher else b
                warp_entry = content_warp_data[idx] if (content_warp_data and not single_stitcher) else None
                out = stitch_single_frame(
                    stitcher['canvas_image'][idx].to(device),
                    inpainted_image[b:b+1].to(device),
                    stitcher['cropped_mask_for_blend'][idx].to(device),
                    stitcher['cropped_to_canvas_x'][idx],
                    stitcher['cropped_to_canvas_y'][idx],
                    stitcher['cropped_to_canvas_w'][idx],
                    stitcher['cropped_to_canvas_h'][idx],
                    stitcher['canvas_to_orig_x'][idx],
                    stitcher['canvas_to_orig_y'][idx],
                    stitcher['canvas_to_orig_w'][idx],
                    stitcher['canvas_to_orig_h'][idx],
                    resize_algorithm,
                    warp_mode=content_warp_mode,
                    warp_entry=warp_entry,
                    blend_mode=blend_mode,
                    multiband_levels=multiband_levels,
                )
                results.append(out.squeeze(0).to(intermediate))
        else:
            # Reconstruct full batch with skipped frames reinserted
            inpainted_idx = 0
            original_idx = 0

            for frame_idx in range(total_frames):
                if frame_idx in skipped_indices:
                    results.append(original_frames[original_idx].to(intermediate))
                    original_idx += 1
                else:
                    warp_entry = content_warp_data[inpainted_idx] if content_warp_data else None
                    out = stitch_single_frame(
                        stitcher['canvas_image'][inpainted_idx].to(device),
                        inpainted_image[inpainted_idx:inpainted_idx+1].to(device),
                        stitcher['cropped_mask_for_blend'][inpainted_idx].to(device),
                        stitcher['cropped_to_canvas_x'][inpainted_idx],
                        stitcher['cropped_to_canvas_y'][inpainted_idx],
                        stitcher['cropped_to_canvas_w'][inpainted_idx],
                        stitcher['cropped_to_canvas_h'][inpainted_idx],
                        stitcher['canvas_to_orig_x'][inpainted_idx],
                        stitcher['canvas_to_orig_y'][inpainted_idx],
                        stitcher['canvas_to_orig_w'][inpainted_idx],
                        stitcher['canvas_to_orig_h'][inpainted_idx],
                        resize_algorithm,
                        warp_mode=content_warp_mode,
                        warp_entry=warp_entry,
                        blend_mode=blend_mode,
                        multiband_levels=multiband_levels,
                    )
                    results.append(out.squeeze(0).to(intermediate))
                    inpainted_idx += 1

        result_batch = torch.stack(results, dim=0)
        print(f"[NV_InpaintStitch] Stitched {result_batch.shape[0]} frames "
              f"({len(skipped_indices)} skipped, {inpainted_image.shape[0]} inpainted, "
              f"blend={blend_mode})")

        return (result_batch,)


# =============================================================================
# Node Registration
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "NV_InpaintStitch2": NV_InpaintStitch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_InpaintStitch2": "NV Inpaint Stitch v2",
}
