"""
NV Latent Inpaint Stitch — Pastes denoised latent crop back into the full-frame video latent.

Pairs with NV_LatentInpaintCrop. Takes the LATENT_STITCHER produced by the crop node
and the inpainted (denoised) crop, composites back using 5D-safe ellipsis indexing.

Blend modes:
  - Hard paste (default): Direct replacement. Best when SetLatentNoiseMask
    already gates what the KSampler modifies.
  - Soft blend: Connect a blend_mask (pixel-space, downscaled internally).
    Use blend_feather for smooth edges before downscale.
"""

import torch
import comfy.model_management
import comfy.utils

from .inpaint_crop import rescale_mask, mask_blur


# =============================================================================
# Core Stitch Function
# =============================================================================

def latent_stitch(original_samples, inpainted_samples, stitcher, blend_mask_latent=None):
    """Paste inpainted crop back into the original full-frame latent.

    Args:
        original_samples: [B,C,T,H_full,W_full] original full-frame latent.
        inpainted_samples: [B,C,T,H_crop,W_crop] denoised crop.
        stitcher: LATENT_STITCHER dict with crop coordinates.
        blend_mask_latent: [B, H_crop_latent, W_crop_latent] float mask at latent
                           resolution, or None for hard paste.

    Returns:
        [B,C,T,H_full,W_full] output latent tensor.
    """
    cx = stitcher['crop_x_latent']
    cy = stitcher['crop_y_latent']
    ch = stitcher['crop_h_latent']
    cw = stitcher['crop_w_latent']

    result = original_samples.clone()

    if blend_mask_latent is None:
        # Hard paste
        result[..., cy:cy + ch, cx:cx + cw] = inpainted_samples
    else:
        # Soft blend: expand mask to [B, 1, 1, H, W] for 5D broadcast over C and T
        mask_5d = blend_mask_latent[:, None, None, :, :]
        mask_5d = mask_5d.clamp(0.0, 1.0)

        orig_region = original_samples[..., cy:cy + ch, cx:cx + cw]
        blended = mask_5d * inpainted_samples + (1.0 - mask_5d) * orig_region
        result[..., cy:cy + ch, cx:cx + cw] = blended

    return result


# =============================================================================
# Node Class
# =============================================================================

class NV_LatentInpaintStitch:
    """Paste denoised latent crop back into the full-frame video latent.

    Pairs with NV_LatentInpaintCrop. Hard paste by default;
    connect blend_mask for soft edge blending.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stitcher": ("LATENT_STITCHER",),
                "inpainted_latent": ("LATENT",),
            },
            "optional": {
                "blend_mask": ("MASK", {
                    "tooltip": "Pixel-space mask [B,H,W] for soft blending at crop boundaries. "
                               "1.0 = use inpainted, 0.0 = keep original. "
                               "If not connected, performs hard paste. "
                               "Downscaled to latent resolution internally."
                }),
                "blend_feather": ("INT", {
                    "default": 0, "min": 0, "max": 256, "step": 1,
                    "tooltip": "Gaussian feather on blend_mask in pixel space "
                               "BEFORE downscaling to latent resolution. "
                               "8-32px recommended for soft edges. 0 = no feathering."
                }),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "execute"
    CATEGORY = "NV_Utils/Inpaint"
    DESCRIPTION = (
        "Paste denoised latent crop back into the full-frame video latent. "
        "Pairs with NV Latent Inpaint Crop. "
        "Hard paste by default; connect blend_mask for soft edge blending."
    )

    def execute(self, stitcher, inpainted_latent, blend_mask=None, blend_feather=0):
        device = comfy.model_management.get_torch_device()
        intermediate = comfy.model_management.intermediate_device()

        original_samples = stitcher['original_samples'].to(device)
        inpainted_samples = inpainted_latent["samples"].to(device)

        cx = stitcher['crop_x_latent']
        cy = stitcher['crop_y_latent']
        cw = stitcher['crop_w_latent']
        ch = stitcher['crop_h_latent']

        # --- Validate spatial match ---
        target_w = stitcher.get('target_w_latent')
        target_h = stitcher.get('target_h_latent')

        if target_w is not None and target_h is not None:
            expected = (target_h, target_w)
        else:
            expected = (ch, cw)

        actual = (inpainted_samples.shape[-2], inpainted_samples.shape[-1])
        if actual != expected:
            raise ValueError(
                f"[NV_LatentInpaintStitch] Spatial size mismatch: "
                f"inpainted {actual} != expected {expected}. "
                f"{'Target resize is active — did the KSampler change resolution?' if target_w else 'Did you resize the crop before stitching?'}"
            )

        # --- Validate temporal match ---
        if inpainted_samples.shape[2] != original_samples.shape[2]:
            raise ValueError(
                f"[NV_LatentInpaintStitch] Temporal mismatch: "
                f"inpainted T={inpainted_samples.shape[2]}, "
                f"original T={original_samples.shape[2]}."
            )

        # --- Resize inpainted latent back to original crop dims ---
        if target_w is not None and target_h is not None:
            if inpainted_samples.shape[-2] != ch or inpainted_samples.shape[-1] != cw:
                stitch_resize_method = stitcher.get('resize_method', 'bislerp')
                inpainted_samples = comfy.utils.common_upscale(
                    inpainted_samples, cw, ch, stitch_resize_method, "disabled"
                )
                print(
                    f"[NV_LatentInpaintStitch] Resized inpainted: "
                    f"({target_h},{target_w}) -> ({ch},{cw}) latent cells "
                    f"({stitch_resize_method})"
                )

        # --- Prepare blend mask ---
        # Priority: explicit blend_mask input > pre-computed from stitcher > hard paste
        blend_mask_latent = None
        if blend_mask is not None:
            bm = blend_mask.to(device)
            if bm.dim() == 2:
                bm = bm.unsqueeze(0)

            # Feather in pixel space before downscale
            if blend_feather > 0:
                bm = mask_blur(bm, blend_feather)

            # Downscale to latent resolution
            blend_mask_latent = rescale_mask(bm, cw, ch, 'bilinear')
        elif 'blend_mask' in stitcher:
            # Use pre-computed blend mask from NV_LatentInpaintCrop
            bm = stitcher['blend_mask'].to(device)
            if bm.dim() == 2:
                bm = bm.unsqueeze(0)

            # Downscale to latent resolution
            blend_mask_latent = rescale_mask(bm, cw, ch, 'bilinear')

        # --- Stitch ---
        result_samples = latent_stitch(
            original_samples, inpainted_samples, stitcher, blend_mask_latent
        )

        # --- Build output latent dict ---
        out_latent = {"samples": result_samples.to(intermediate)}
        for key, value in stitcher['safe_keys'].items():
            out_latent[key] = value

        if blend_mask is not None:
            mode = 'soft blend (explicit mask)'
        elif 'blend_mask' in stitcher:
            mode = 'soft blend (from stitcher)'
        else:
            mode = 'hard paste'
        resize_info = ""
        if target_w is not None and target_h is not None:
            resize_info = f", resized from {target_h}x{target_w}"
        print(
            f"[NV_LatentInpaintStitch] Stitched: crop ({cx},{cy}) "
            f"{cw}x{ch} latent cells ({mode}{resize_info})"
        )

        return (out_latent,)


# =============================================================================
# Node Registration
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "NV_LatentInpaintStitch": NV_LatentInpaintStitch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_LatentInpaintStitch": "NV Latent Inpaint Stitch",
}
