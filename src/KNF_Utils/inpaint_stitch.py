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
import comfy.model_management

from .inpaint_crop import rescale_image, rescale_mask


# =============================================================================
# Core Stitch Function
# =============================================================================

def stitch_single_frame(canvas_image, inpainted_image, mask,
                        ctc_x, ctc_y, ctc_w, ctc_h,
                        cto_x, cto_y, cto_w, cto_h,
                        resize_algorithm):
    """Blend one inpainted crop back into its canvas and extract the original region.

    Args:
        canvas_image: [H, W, C] or [1, H, W, C] — the padded original frame.
        inpainted_image: [1, H, W, C] — the inpainted crop at target resolution.
        mask: [H, W] or [1, H, W] — blend mask at crop resolution.
        ctc_x/y/w/h: Where the crop sits on the canvas.
        cto_x/y/w/h: Where the original image sits on the canvas.
        resize_algorithm: Interpolation method for resizing crop back to canvas scale.

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
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "stitch"
    CATEGORY = "NV_Utils/Inpaint"
    DESCRIPTION = (
        "Composites inpainted crops back into the original frames. "
        "Pairs with NV Inpaint Crop v2. Handles frame skipping automatically."
    )

    def stitch(self, stitcher, inpainted_image):
        device = comfy.model_management.get_torch_device()
        intermediate = comfy.model_management.intermediate_device()
        resize_algorithm = _get_resize_algorithm(stitcher)

        skipped_indices = set(stitcher.get('skipped_indices', []))
        original_frames = stitcher.get('original_frames', [])
        total_frames = stitcher.get('total_frames', inpainted_image.shape[0])

        results = []

        if len(skipped_indices) == 0:
            # Simple path: all frames were inpainted
            batch_size = inpainted_image.shape[0]
            num_coords = len(stitcher['cropped_to_canvas_x'])
            single_stitcher = (num_coords == 1 and batch_size > 1)

            for b in range(batch_size):
                idx = 0 if single_stitcher else b
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
                    )
                    results.append(out.squeeze(0).to(intermediate))
                    inpainted_idx += 1

        result_batch = torch.stack(results, dim=0)
        print(f"[NV_InpaintStitch] Stitched {result_batch.shape[0]} frames "
              f"({len(skipped_indices)} skipped, {inpainted_image.shape[0]} inpainted)")

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
