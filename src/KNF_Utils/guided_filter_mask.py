"""
NV GuidedFilterMask - Edge-aware mask refinement using image structure.

Standalone node for cleaning up SAM3 segmentation masks, tracking masks,
or any soft mask that needs pixel-accurate edge alignment with the source image.
Uses the guided filter algorithm (He et al. 2013) — zero new dependencies.

Place BEFORE NV_MaskTrackingBBox for SAM3 cleanup, or use anywhere a mask
needs edge refinement against a reference image.
"""

import torch

from .guided_filter import refine_mask


class NV_GuidedFilterMask:
    """Refines a soft mask using an edge-aware guided filter driven by the input image."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK", {
                    "tooltip": "Soft mask to refine [B, H, W]. Values in [0, 1]."
                }),
                "guide_image": ("IMAGE", {
                    "tooltip": "Reference image whose edges guide the mask refinement [B, H, W, C]. "
                               "Use the original source frame for SAM3 cleanup, or the canvas crop for stitch refinement."
                }),
                "radius": ("INT", {
                    "default": 8, "min": 1, "max": 64, "step": 1,
                    "tooltip": "Filter window radius. Larger = broader smoothing, smaller = tighter edge following. "
                               "4-8 for SAM3 cleanup, 8-16 for stitch blend refinement."
                }),
                "eps": ("FLOAT", {
                    "default": 0.001, "min": 0.0001, "max": 0.1, "step": 0.0001,
                    "tooltip": "Edge sensitivity. Lower = sharper snap to edges (risk of noise sensitivity). "
                               "Higher = smoother (risk of losing fine edges). 0.001 = good default."
                }),
                "strength": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Blend between original mask (0.0) and refined mask (1.0). "
                               "0.7 = recommended — prevents the filter from eroding valid mask coverage. "
                               "1.0 = full guided filter output (may shrink mask in low-contrast areas)."
                }),
            },
            "optional": {
                "mode": (["color", "gray"], {
                    "default": "color",
                    "tooltip": "color: full RGB covariance solve — best for color boundaries (hair vs background). "
                               "gray: luminance-only guidance — faster, sufficient when edges are brightness-defined."
                }),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("refined_mask",)
    FUNCTION = "execute"
    CATEGORY = "NV_Utils/Mask"
    DESCRIPTION = (
        "Refines a soft mask using an edge-aware guided filter (He et al. 2013). "
        "Snaps mask edges to actual image boundaries — eliminates jagged SAM3 edges, "
        "staircase artifacts, and boundary wobble. Zero new dependencies."
    )

    def execute(self, mask, guide_image, radius, eps, strength, mode="color"):
        refined = refine_mask(
            mask=mask,
            guide_image=guide_image,
            radius=radius,
            eps=eps,
            strength=strength,
            mode=mode,
        )
        return (refined.to(dtype=torch.float32),)


NODE_CLASS_MAPPINGS = {
    "NV_GuidedFilterMask": NV_GuidedFilterMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_GuidedFilterMask": "NV Guided Filter Mask",
}
