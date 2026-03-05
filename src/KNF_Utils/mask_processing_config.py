"""
NV Mask Processing Config — shared config bus for mask processing parameters.

Emits a MASK_PROCESSING_CONFIG dict that can be connected to any mask-consuming
node (InpaintCrop2, LatentInpaintCrop, MaskPipelineViz, VaceControlVideoPrep)
to ensure identical mask processing settings across the pipeline.

When connected, config values override the consuming node's local widgets.
When not connected, nodes use their own widgets as before (fully backward compatible).
"""


def apply_mask_config(mask_config, **local_values):
    """Merge mask_config over local widget values. Config wins when connected.

    Usage in consuming node:
        vals = apply_mask_config(mask_config,
            mask_erode_dilate=mask_erode_dilate,
            mask_fill_holes=mask_fill_holes,
            mask_remove_noise=mask_remove_noise,
            mask_smooth=mask_smooth,
            mask_blend_pixels=mask_blend_pixels,
        )
        mask_erode_dilate = vals["mask_erode_dilate"]
        # ...etc
    """
    if mask_config is None:
        return local_values
    result = dict(local_values)
    for key in local_values:
        if key in mask_config:
            result[key] = mask_config[key]
    return result


def apply_vace_mask_config(mask_config, **local_values):
    """Like apply_mask_config but with VACE key remapping.

    VaceControlVideoPrep uses bare names (erosion_blocks, feather_blocks, etc.)
    while the config dict uses vace_ prefix. This function handles the mapping.

    Also maps shared cleanup params (mask_fill_holes, mask_remove_noise, mask_smooth)
    directly since those names match.

    Note: mask_grow is intentionally NOT mapped from mask_erode_dilate — they have
    different semantics (mask_grow expands input mask before bbox conversion, while
    mask_erode_dilate processes the diffusion mask).
    """
    if mask_config is None:
        return local_values
    result = dict(local_values)

    # Direct matches (shared cleanup params)
    for key in ("mask_fill_holes", "mask_remove_noise", "mask_smooth"):
        if key in mask_config and key in result:
            result[key] = mask_config[key]

    # VACE key remapping: config uses vace_ prefix, node uses bare names
    remap = {
        "erosion_blocks": "vace_erosion_blocks",
        "feather_blocks": "vace_feather_blocks",
        "stitch_erosion": "vace_stitch_erosion",
        "stitch_feather": "vace_stitch_feather",
    }
    for local_key, config_key in remap.items():
        if config_key in mask_config and local_key in result:
            result[local_key] = mask_config[config_key]

    return result


MASK_CONFIG_TOOLTIP = (
    "Optional shared config from NV_MaskProcessingConfig. "
    "When connected, overrides this node's local mask processing widgets."
)


class NV_MaskProcessingConfig:
    """Emit a shared mask processing config dict for consistent settings across nodes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask_erode_dilate": ("INT", {
                    "default": 0, "min": -128, "max": 128, "step": 1,
                    "tooltip": "Shrink (negative) or expand (positive) the mask using grey morphology. "
                               "Applied to diffusion/processed mask, NOT the input mask for bbox."
                }),
                "mask_fill_holes": ("INT", {
                    "default": 0, "min": 0, "max": 128, "step": 1,
                    "tooltip": "Fill gaps/holes in mask using morphological closing (dilate then erode)."
                }),
                "mask_remove_noise": ("INT", {
                    "default": 0, "min": 0, "max": 32, "step": 1,
                    "tooltip": "Remove isolated pixels using morphological opening (erode then dilate)."
                }),
                "mask_smooth": ("INT", {
                    "default": 0, "min": 0, "max": 127, "step": 1,
                    "tooltip": "Smooth jagged edges by binarize at 0.5 then Gaussian blur."
                }),
                "mask_blend_pixels": ("INT", {
                    "default": 16, "min": 0, "max": 64, "step": 1,
                    "tooltip": "Feather stitch mask edges for seamless compositing (dilate + blur of original mask)."
                }),
            },
            "optional": {
                "vace_erosion_blocks": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 4.0, "step": 0.25,
                    "tooltip": "VACE: erode mask inward by this many VAE blocks (0.5 = 4px for WAN). "
                               "Prevents dark seam at mask/control-video boundary."
                }),
                "vace_feather_blocks": ("FLOAT", {
                    "default": 1.5, "min": 0.0, "max": 8.0, "step": 0.25,
                    "tooltip": "VACE: feather mask edge over this many VAE blocks (1.5 = 12px for WAN). "
                               "Smooth transition spanning >1 encoding block eliminates ringing."
                }),
                "vace_stitch_erosion": ("INT", {
                    "default": 0, "min": -32, "max": 32, "step": 1,
                    "tooltip": "VACE: erode/dilate the pixel-space stitch mask (independent from VACE erosion)."
                }),
                "vace_stitch_feather": ("INT", {
                    "default": 8, "min": 0, "max": 64, "step": 1,
                    "tooltip": "VACE: feather the pixel-space stitch mask edges for seamless compositing."
                }),
            },
        }

    RETURN_TYPES = ("MASK_PROCESSING_CONFIG",)
    RETURN_NAMES = ("mask_config",)
    FUNCTION = "execute"
    CATEGORY = "NV_Utils/Inpaint"
    DESCRIPTION = (
        "Shared mask processing config bus. Connect to InpaintCrop2, "
        "LatentInpaintCrop, MaskPipelineViz, or VaceControlVideoPrep "
        "to ensure identical mask settings across the pipeline. "
        "When connected, config values override the consuming node's local widgets."
    )

    def execute(self, mask_erode_dilate, mask_fill_holes, mask_remove_noise,
                mask_smooth, mask_blend_pixels,
                vace_erosion_blocks=0.5, vace_feather_blocks=1.5,
                vace_stitch_erosion=0, vace_stitch_feather=8):
        config = {
            "mask_erode_dilate": mask_erode_dilate,
            "mask_fill_holes": mask_fill_holes,
            "mask_remove_noise": mask_remove_noise,
            "mask_smooth": mask_smooth,
            "mask_blend_pixels": mask_blend_pixels,
            "vace_erosion_blocks": vace_erosion_blocks,
            "vace_feather_blocks": vace_feather_blocks,
            "vace_stitch_erosion": vace_stitch_erosion,
            "vace_stitch_feather": vace_stitch_feather,
        }
        return (config,)


NODE_CLASS_MAPPINGS = {
    "NV_MaskProcessingConfig": NV_MaskProcessingConfig,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_MaskProcessingConfig": "NV Mask Processing Config",
}
