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
# Apply functions — merge config over local widget values
# =============================================================================

def apply_mask_config(mask_config, **local_values):
    """Merge mask_config over local widget values. Config wins when connected.

    Direct key lookup — local param names must match config dict keys exactly.

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

    # Direct matches (shared cleanup params)
    cleanup_keys = {
        "cleanup_fill_holes": ["cleanup_fill_holes"],
        "cleanup_remove_noise": ["cleanup_remove_noise"],
        "cleanup_smooth": ["cleanup_smooth"],
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
        "vace_stitch_erosion_px": ["vace_stitch_erosion_px"],
        "vace_stitch_feather_px": ["vace_stitch_feather_px"],
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
                cleanup_fill_holes, cleanup_remove_noise, cleanup_smooth,
                crop_expand_px, crop_blend_feather_px,
                vace_input_grow_px, vace_erosion_blocks, vace_feather_blocks,
                vace_halo_px, vace_stitch_erosion_px, vace_stitch_feather_px):

        config = {
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
        }
        return (config,)


# =============================================================================
# Patch Node — per-key overrides on a base config
# =============================================================================

class NV_MaskConfigPatch:
    """Patch a base MASK_PROCESSING_CONFIG with per-key overrides.

    Use case: load a saved stitcher's mask_config (via NV_LoadStitcher_V2),
    keep most values, but tweak one or two before feeding into InpaintCrop_V2
    or VaceControlVideoPrep.

    Each override widget defaults to a sentinel value (-999 for ints, -999.0
    for floats) which means "use base value unchanged". Set to a real value
    in the legitimate range to override that specific key. Sentinels are
    chosen to be outside every key's natural range.
    """

    _SENTINEL_INT = -999
    _SENTINEL_FLOAT = -999.0

    @classmethod
    def INPUT_TYPES(cls):
        # Tooltip suffix used to keep the sentinel-semantics explanation short
        # without copy-pasting it 11 times.
        _USE_BASE_INT = "  (-999 = use base unchanged)"
        _USE_BASE_FLOAT = "  (-999.0 = use base unchanged)"
        return {
            "required": {
                "base_config": ("MASK_PROCESSING_CONFIG", {
                    "tooltip": "Base config to patch. Typically wired from "
                               "NV_LoadStitcher_V2.mask_config or NV_MaskProcessingConfig."
                }),

                # Cleanup overrides
                "cleanup_fill_holes": ("INT", {
                    "default": -999, "min": -999, "max": 128, "step": 1,
                    "tooltip": "Override cleanup_fill_holes (real range 0-128)." + _USE_BASE_INT,
                }),
                "cleanup_remove_noise": ("INT", {
                    "default": -999, "min": -999, "max": 32, "step": 1,
                    "tooltip": "Override cleanup_remove_noise (real range 0-32)." + _USE_BASE_INT,
                }),
                "cleanup_smooth": ("INT", {
                    "default": -999, "min": -999, "max": 127, "step": 1,
                    "tooltip": "Override cleanup_smooth (real range 0-127)." + _USE_BASE_INT,
                }),

                # Crop / Stitch overrides
                "crop_expand_px": ("INT", {
                    "default": -999, "min": -999, "max": 128, "step": 1,
                    "tooltip": "Override crop_expand_px (real range -128 to 128)." + _USE_BASE_INT,
                }),
                "crop_blend_feather_px": ("INT", {
                    "default": -999, "min": -999, "max": 64, "step": 1,
                    "tooltip": "Override crop_blend_feather_px (real range 0-64)." + _USE_BASE_INT,
                }),

                # VACE overrides
                "vace_input_grow_px": ("INT", {
                    "default": -999, "min": -999, "max": 128, "step": 1,
                    "tooltip": "Override vace_input_grow_px (real range -128 to 128)." + _USE_BASE_INT,
                }),
                "vace_erosion_blocks": ("FLOAT", {
                    "default": -999.0, "min": -999.0, "max": 4.0, "step": 0.25,
                    "tooltip": "Override vace_erosion_blocks (real range 0.0-4.0)." + _USE_BASE_FLOAT,
                }),
                "vace_feather_blocks": ("FLOAT", {
                    "default": -999.0, "min": -999.0, "max": 8.0, "step": 0.25,
                    "tooltip": "Override vace_feather_blocks (real range 0.0-8.0)." + _USE_BASE_FLOAT,
                }),
                "vace_halo_px": ("INT", {
                    "default": -999, "min": -999, "max": 48, "step": 4,
                    "tooltip": "Override vace_halo_px (real range 0-48)." + _USE_BASE_INT,
                }),
                "vace_stitch_erosion_px": ("INT", {
                    "default": -999, "min": -999, "max": 32, "step": 1,
                    "tooltip": "Override vace_stitch_erosion_px (real range -32 to 32)." + _USE_BASE_INT,
                }),
                "vace_stitch_feather_px": ("INT", {
                    "default": -999, "min": -999, "max": 64, "step": 1,
                    "tooltip": "Override vace_stitch_feather_px (real range 0-64)." + _USE_BASE_INT,
                }),
            },
        }

    RETURN_TYPES = ("MASK_PROCESSING_CONFIG",)
    RETURN_NAMES = ("patched_config",)
    FUNCTION = "patch"
    CATEGORY = "NV_Utils/Inpaint"
    DESCRIPTION = (
        "Patch a base MASK_PROCESSING_CONFIG with per-key overrides. "
        "Each widget defaults to a sentinel (-999 / -999.0) meaning 'use "
        "base unchanged'. Set to a real value to override that specific key. "
        "Useful for loading a saved mask_config and tweaking just one or two "
        "values without rebuilding the whole config from scratch."
    )

    def patch(self,
              base_config,
              cleanup_fill_holes, cleanup_remove_noise, cleanup_smooth,
              crop_expand_px, crop_blend_feather_px,
              vace_input_grow_px, vace_erosion_blocks, vace_feather_blocks,
              vace_halo_px, vace_stitch_erosion_px, vace_stitch_feather_px):

        if not isinstance(base_config, dict):
            raise TypeError(
                f"[NV_MaskConfigPatch] base_config must be a MASK_PROCESSING_CONFIG dict, "
                f"got {type(base_config).__name__}"
            )

        result = dict(base_config)

        # (key, value, sentinel) — single source of truth for the patch loop
        overrides = (
            ("cleanup_fill_holes", cleanup_fill_holes, self._SENTINEL_INT),
            ("cleanup_remove_noise", cleanup_remove_noise, self._SENTINEL_INT),
            ("cleanup_smooth", cleanup_smooth, self._SENTINEL_INT),
            ("crop_expand_px", crop_expand_px, self._SENTINEL_INT),
            ("crop_blend_feather_px", crop_blend_feather_px, self._SENTINEL_INT),
            ("vace_input_grow_px", vace_input_grow_px, self._SENTINEL_INT),
            ("vace_erosion_blocks", vace_erosion_blocks, self._SENTINEL_FLOAT),
            ("vace_feather_blocks", vace_feather_blocks, self._SENTINEL_FLOAT),
            ("vace_halo_px", vace_halo_px, self._SENTINEL_INT),
            ("vace_stitch_erosion_px", vace_stitch_erosion_px, self._SENTINEL_INT),
            ("vace_stitch_feather_px", vace_stitch_feather_px, self._SENTINEL_INT),
        )

        applied = []
        for key, value, sentinel in overrides:
            if value != sentinel:
                old = result.get(key, "<absent>")
                result[key] = value
                applied.append(f"{key}: {old} -> {value}")

        if applied:
            print(f"[NV_MaskConfigPatch] Applied {len(applied)} override(s):")
            for line in applied:
                print(f"  {line}")
        else:
            print("[NV_MaskConfigPatch] No overrides — config passes through unchanged")

        return (result,)


NODE_CLASS_MAPPINGS = {
    "NV_MaskProcessingConfig": NV_MaskProcessingConfig,
    "NV_MaskConfigPatch": NV_MaskConfigPatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_MaskProcessingConfig": "NV Mask Processing Config",
    "NV_MaskConfigPatch": "NV Mask Config Patch",
}
