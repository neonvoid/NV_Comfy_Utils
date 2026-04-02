"""
NV Mask Processing Config — shared config bus for mask processing parameters.

Emits a MASK_PROCESSING_CONFIG dict that can be connected to any mask-consuming
node (InpaintCrop2, LatentInpaintCrop, MaskPipelineViz, VaceControlVideoPrep)
to ensure identical mask processing settings across the pipeline.

When connected, config values override the consuming node's local widgets.
When not connected, nodes use their own widgets as before (fully backward compatible).

Parameters are organized by category:
  cleanup_*     — mask cleanup (fill, noise, smooth) — shared by all consumers
  crop_*        — crop/stitch settings — used by InpaintCrop2, LatentInpaintCrop
  vace_*        — VACE conditioning settings — used by VaceControlVideoPrep
"""


# =============================================================================
# Backward compatibility: old config key → new config key
# =============================================================================

_OLD_TO_NEW = {
    "mask_erode_dilate": "crop_expand_px",
    "mask_fill_holes": "cleanup_fill_holes",
    "mask_remove_noise": "cleanup_remove_noise",
    "mask_smooth": "cleanup_smooth",
    "mask_blend_pixels": "crop_blend_feather_px",
    "vace_stitch_erosion": "vace_stitch_erosion_px",
    "vace_stitch_feather": "vace_stitch_feather_px",
    # vace_erosion_blocks and vace_feather_blocks unchanged
}

_NEW_TO_OLD = {v: k for k, v in _OLD_TO_NEW.items()}


def resolve_deprecated(new_value, new_default, old_value, old_default):
    """If new param is at default and old param was explicitly set, use old value."""
    if new_value != new_default or old_value == old_default:
        return new_value
    return old_value


# =============================================================================
# Apply functions — merge config over local widget values
# =============================================================================

def apply_mask_config(mask_config, **local_values):
    """Merge mask_config over local widget values. Config wins when connected.

    Accepts both old and new config key names for backward compatibility.

    Usage in consuming node:
        vals = apply_mask_config(mask_config,
            crop_expand_px=crop_expand_px,
            cleanup_fill_holes=cleanup_fill_holes,
            cleanup_remove_noise=cleanup_remove_noise,
            cleanup_smooth=cleanup_smooth,
            crop_blend_feather_px=crop_blend_feather_px,
        )
        crop_expand_px = vals["crop_expand_px"]
        # ...etc
    """
    if mask_config is None:
        return local_values
    result = dict(local_values)
    for key in local_values:
        if key in mask_config:
            result[key] = mask_config[key]
        # Check old-name alias in config dict
        elif key in _NEW_TO_OLD and _NEW_TO_OLD[key] in mask_config:
            result[key] = mask_config[_NEW_TO_OLD[key]]
        # Check new-name alias (consumer still uses old name)
        elif key in _OLD_TO_NEW and _OLD_TO_NEW[key] in mask_config:
            result[key] = mask_config[_OLD_TO_NEW[key]]
    return result


def apply_vace_mask_config(mask_config, **local_values):
    """Like apply_mask_config but with VACE key remapping.

    VaceControlVideoPrep uses bare names (erosion_blocks, feather_blocks, etc.)
    while the config dict uses vace_ prefix. This function handles the mapping.

    Also maps shared cleanup params directly since those names match.

    vace_input_grow_px is now included (promoted from local-only to config bus).
    """
    if mask_config is None:
        return local_values
    result = dict(local_values)

    # Direct matches (shared cleanup params — accept both old and new names)
    cleanup_keys = {
        "cleanup_fill_holes": ["cleanup_fill_holes", "mask_fill_holes"],
        "cleanup_remove_noise": ["cleanup_remove_noise", "mask_remove_noise"],
        "cleanup_smooth": ["cleanup_smooth", "mask_smooth"],
        # Legacy names used by consumers not yet updated
        "mask_fill_holes": ["cleanup_fill_holes", "mask_fill_holes"],
        "mask_remove_noise": ["cleanup_remove_noise", "mask_remove_noise"],
        "mask_smooth": ["cleanup_smooth", "mask_smooth"],
    }
    for local_key in list(result.keys()):
        if local_key in cleanup_keys:
            for config_key in cleanup_keys[local_key]:
                if config_key in mask_config:
                    result[local_key] = mask_config[config_key]
                    break

    # VACE key remapping: config uses vace_ prefix, node uses bare names
    remap = {
        "erosion_blocks": "vace_erosion_blocks",
        "feather_blocks": "vace_feather_blocks",
        "stitch_erosion": ["vace_stitch_erosion_px", "vace_stitch_erosion"],
        "stitch_feather": ["vace_stitch_feather_px", "vace_stitch_feather"],
        "input_grow_px": ["vace_input_grow_px"],
        "halo_px": ["vace_halo_px"],
        # New-style local names (for updated consumers)
        "vace_stitch_erosion_px": ["vace_stitch_erosion_px", "vace_stitch_erosion"],
        "vace_stitch_feather_px": ["vace_stitch_feather_px", "vace_stitch_feather"],
        "vace_input_grow_px": ["vace_input_grow_px"],
        "vace_halo_px": ["vace_halo_px"],
    }
    for local_key in list(result.keys()):
        if local_key in remap:
            config_keys = remap[local_key]
            if isinstance(config_keys, str):
                config_keys = [config_keys]
            for config_key in config_keys:
                if config_key in mask_config:
                    result[local_key] = mask_config[config_key]
                    break

    return result


# =============================================================================
# Node Class
# =============================================================================

class NV_MaskProcessingConfig:
    """Emit a shared mask processing config dict for consistent settings across nodes.

    Parameters are organized by category (cleanup, crop/stitch, VACE conditioning).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # ── Cleanup ──
                "cleanup_fill_holes": ("INT", {
                    "default": 0, "min": 0, "max": 128, "step": 1,
                    "tooltip": "Fill gaps/holes in mask using morphological closing (dilate then erode). "
                               "Shared by InpaintCrop2 and VaceControlVideoPrep."
                }),
                "cleanup_remove_noise": ("INT", {
                    "default": 0, "min": 0, "max": 32, "step": 1,
                    "tooltip": "Remove isolated pixels using morphological opening (erode then dilate). "
                               "Shared by InpaintCrop2 and VaceControlVideoPrep."
                }),
                "cleanup_smooth": ("INT", {
                    "default": 0, "min": 0, "max": 127, "step": 1,
                    "tooltip": "Smooth jagged edges by binarize at 0.5 then Gaussian blur. "
                               "Shared by InpaintCrop2 and VaceControlVideoPrep."
                }),

                # ── Crop / Stitch ──
                "crop_expand_px": ("INT", {
                    "default": 0, "min": -128, "max": 128, "step": 1,
                    "tooltip": "Shrink (negative) or expand (positive) the crop/stitch mask in pixels. "
                               "Applied to the diffusion/processed mask — controls what gets denoised + stitched. "
                               "(Previously: mask_erode_dilate)"
                }),
                "crop_blend_feather_px": ("INT", {
                    "default": 16, "min": 0, "max": 64, "step": 1,
                    "tooltip": "Feather stitch mask edges for seamless compositing (dilate + blur). "
                               "Controls the pixel-space blend width at the stitch boundary. "
                               "(Previously: mask_blend_pixels)"
                }),

                # ── VACE Conditioning ──
                "vace_input_grow_px": ("INT", {
                    "default": 0, "min": -128, "max": 128, "step": 1,
                    "tooltip": "Grow (positive) or shrink (negative) the raw input mask BEFORE VACE processing. "
                               "Expands the area VACE regenerates. Use for object insertion where generated "
                               "content extends beyond the mask silhouette. 0 = mask covers the right area already. "
                               "(Previously: mask_grow on VaceControlVideoPrep, local only)"
                }),
                "vace_erosion_blocks": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 4.0, "step": 0.25,
                    "tooltip": "VACE: erode conditioning mask inward by this many VAE blocks (0.5 = 4px for WAN). "
                               "Prevents dark seam at mask/control-video boundary. NEVER set to 0."
                }),
                "vace_feather_blocks": ("FLOAT", {
                    "default": 1.5, "min": 0.0, "max": 8.0, "step": 0.25,
                    "tooltip": "VACE: feather conditioning mask edge over this many VAE blocks (1.5 = 12px for WAN). "
                               "Smooth transition spanning >1 encoding block eliminates VAE ringing. NEVER set to 0."
                }),
                "vace_halo_px": ("INT", {
                    "default": 16, "min": 0, "max": 48, "step": 4,
                    "tooltip": "Seam-Absorbing Halo: expand VACE conditioning mask OUTWARD beyond the stitch boundary. "
                               "WAN repaints this strip, so the stitch falls inside repainted content. "
                               "16px = recommended (2 VAE blocks). 0 = disabled. "
                               "(Previously: halo_pixels on VaceControlVideoPrep, local only)"
                }),
                "vace_stitch_erosion_px": ("INT", {
                    "default": 0, "min": -32, "max": 32, "step": 1,
                    "tooltip": "VACE: erode (negative) or dilate (positive) the pixel-space stitch mask. "
                               "Independent from VACE erosion — controls compositing boundary, not conditioning. "
                               "(Previously: vace_stitch_erosion)"
                }),
                "vace_stitch_feather_px": ("INT", {
                    "default": 8, "min": 0, "max": 64, "step": 1,
                    "tooltip": "VACE: feather the pixel-space stitch mask edges for seamless compositing. "
                               "8-16px = subtle, 24-32px = visible softening, 0 = hard edge. "
                               "(Previously: vace_stitch_feather)"
                }),
            },
            "optional": {
                # ── Deprecated names (backward compat for old workflows) ──
                "mask_erode_dilate": ("INT", {
                    "default": 0, "min": -128, "max": 128, "step": 1,
                    "tooltip": "DEPRECATED — use crop_expand_px"
                }),
                "mask_fill_holes": ("INT", {
                    "default": 0, "min": 0, "max": 128, "step": 1,
                    "tooltip": "DEPRECATED — use cleanup_fill_holes"
                }),
                "mask_remove_noise": ("INT", {
                    "default": 0, "min": 0, "max": 32, "step": 1,
                    "tooltip": "DEPRECATED — use cleanup_remove_noise"
                }),
                "mask_smooth": ("INT", {
                    "default": 0, "min": 0, "max": 127, "step": 1,
                    "tooltip": "DEPRECATED — use cleanup_smooth"
                }),
                "mask_blend_pixels": ("INT", {
                    "default": 16, "min": 0, "max": 64, "step": 1,
                    "tooltip": "DEPRECATED — use crop_blend_feather_px"
                }),
                "vace_stitch_erosion": ("INT", {
                    "default": 0, "min": -32, "max": 32, "step": 1,
                    "tooltip": "DEPRECATED — use vace_stitch_erosion_px"
                }),
                "vace_stitch_feather": ("INT", {
                    "default": 8, "min": 0, "max": 64, "step": 1,
                    "tooltip": "DEPRECATED — use vace_stitch_feather_px"
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
        "Parameters organized by category: cleanup_, crop_, vace_. "
        "When connected, config values override the consuming node's local widgets."
    )

    def execute(self,
                # New names (required)
                cleanup_fill_holes, cleanup_remove_noise, cleanup_smooth,
                crop_expand_px, crop_blend_feather_px,
                vace_input_grow_px, vace_erosion_blocks, vace_feather_blocks,
                vace_halo_px, vace_stitch_erosion_px, vace_stitch_feather_px,
                # Deprecated names (optional, backward compat)
                mask_erode_dilate=0, mask_fill_holes=0, mask_remove_noise=0,
                mask_smooth=0, mask_blend_pixels=16,
                vace_stitch_erosion=0, vace_stitch_feather=8):

        # Resolve deprecated: old workflow values override new defaults
        crop_expand_px = resolve_deprecated(crop_expand_px, 0, mask_erode_dilate, 0)
        cleanup_fill_holes = resolve_deprecated(cleanup_fill_holes, 0, mask_fill_holes, 0)
        cleanup_remove_noise = resolve_deprecated(cleanup_remove_noise, 0, mask_remove_noise, 0)
        cleanup_smooth = resolve_deprecated(cleanup_smooth, 0, mask_smooth, 0)
        crop_blend_feather_px = resolve_deprecated(crop_blend_feather_px, 16, mask_blend_pixels, 16)
        vace_stitch_erosion_px = resolve_deprecated(vace_stitch_erosion_px, 0, vace_stitch_erosion, 0)
        vace_stitch_feather_px = resolve_deprecated(vace_stitch_feather_px, 8, vace_stitch_feather, 8)

        config = {
            # New keys (primary)
            "cleanup_fill_holes": cleanup_fill_holes,
            "cleanup_remove_noise": cleanup_remove_noise,
            "cleanup_smooth": cleanup_smooth,
            "crop_expand_px": crop_expand_px,
            "crop_blend_feather_px": crop_blend_feather_px,
            "vace_input_grow_px": vace_input_grow_px,
            "vace_erosion_blocks": vace_erosion_blocks,
            "vace_feather_blocks": vace_feather_blocks,
            "vace_halo_px": vace_halo_px,
            "vace_stitch_erosion_px": vace_stitch_erosion_px,
            "vace_stitch_feather_px": vace_stitch_feather_px,
            # Old keys (backward compat for consumers not yet updated)
            "mask_erode_dilate": crop_expand_px,
            "mask_fill_holes": cleanup_fill_holes,
            "mask_remove_noise": cleanup_remove_noise,
            "mask_smooth": cleanup_smooth,
            "mask_blend_pixels": crop_blend_feather_px,
            "vace_stitch_erosion": vace_stitch_erosion_px,
            "vace_stitch_feather": vace_stitch_feather_px,
        }
        return (config,)


NODE_CLASS_MAPPINGS = {
    "NV_MaskProcessingConfig": NV_MaskProcessingConfig,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_MaskProcessingConfig": "NV Mask Processing Config",
}
