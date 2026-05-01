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

The 11 widget keys live in a single MASK_CONFIG_SCHEMA below — both
NV_MaskProcessingConfig and NV_MaskConfigPatch generate their INPUT_TYPES
from it. This eliminates the "two masters" maintenance hazard (Codex+Gemini
debate 2026-04-30, Q1 convergent verdict).

normalize_mask_config() is exported for NV_LoadStitcher_V2 to use when
loading a persisted config — fills missing keys with current schema defaults,
preserves canonical key order, and warns on unknown keys via the load_report.
"""

import copy


# =============================================================================
# MASK_CONFIG_SCHEMA — single source of truth
#
# !!! STRICT APPEND-ONLY POLICY (Codex+Gemini debate 2026-04-30, Q3 verdict) !!!
#
# ComfyUI maps `widgets_values` BY POSITION, not by name. Inserting a new key
# in the middle of this dict scrambles every saved workflow's widget values.
# RULES:
#   - Add new keys ONLY at the END of the dict.
#   - Never remove or reorder existing keys without coordinating a breaking
#     change (and updating the FROZEN_KEYS in tests/test_mask_config_schema.py).
#   - The unit test in tests/test_mask_config_schema.py freezes the current
#     key prefix; CI fails if mid-insert/reorder is attempted.
# =============================================================================

_SENTINEL_INT = -999
_SENTINEL_FLOAT = -999.0

MASK_CONFIG_SCHEMA = {
    # ── Cleanup (shared) ──
    "cleanup_fill_holes": {
        "type": "INT",
        "default": 0, "min": 0, "max": 128, "step": 1,
        "sentinel": _SENTINEL_INT,
        "tooltip": (
            "Fill gaps/holes in mask using morphological closing (dilate then erode). "
            "Shared by InpaintCrop2 and VaceControlVideoPrep."
        ),
    },
    "cleanup_remove_noise": {
        "type": "INT",
        "default": 0, "min": 0, "max": 32, "step": 1,
        "sentinel": _SENTINEL_INT,
        "tooltip": (
            "Remove isolated pixels using morphological opening (erode then dilate). "
            "Shared by InpaintCrop2 and VaceControlVideoPrep."
        ),
    },
    "cleanup_smooth": {
        "type": "INT",
        "default": 0, "min": 0, "max": 127, "step": 1,
        "sentinel": _SENTINEL_INT,
        "tooltip": (
            "Smooth jagged edges by binarize at 0.5 then Gaussian blur. "
            "Shared by InpaintCrop2 and VaceControlVideoPrep."
        ),
    },

    # ── Crop / Stitch ──
    "crop_expand_px": {
        "type": "INT",
        "default": 0, "min": -128, "max": 128, "step": 1,
        "sentinel": _SENTINEL_INT,
        "tooltip": (
            "Shrink (negative) or expand (positive) the crop/stitch mask in pixels. "
            "Applied to the diffusion/processed mask — controls what gets denoised + stitched. "
            "(Previously: mask_erode_dilate)"
        ),
    },
    "crop_blend_feather_px": {
        "type": "INT",
        "default": 16, "min": 0, "max": 64, "step": 1,
        "sentinel": _SENTINEL_INT,
        "tooltip": (
            "Feather stitch mask edges for seamless compositing (dilate + blur). "
            "Controls the pixel-space blend width at the stitch boundary. "
            "(Previously: mask_blend_pixels)"
        ),
    },

    # ── VACE Conditioning ──
    "vace_input_grow_px": {
        "type": "INT",
        "default": 0, "min": -128, "max": 128, "step": 1,
        "sentinel": _SENTINEL_INT,
        "tooltip": (
            "Grow (positive) or shrink (negative) the raw input mask BEFORE VACE processing. "
            "Expands the area VACE regenerates. Use for object insertion where generated "
            "content extends beyond the mask silhouette. 0 = mask covers the right area already. "
            "(Previously: mask_grow on VaceControlVideoPrep, local only)"
        ),
    },
    "vace_erosion_blocks": {
        "type": "FLOAT",
        "default": 0.5, "min": 0.0, "max": 4.0, "step": 0.25,
        "sentinel": _SENTINEL_FLOAT,
        "tooltip": (
            "VACE: erode conditioning mask inward by this many VAE blocks (0.5 = 4px for WAN). "
            "Prevents dark seam at mask/control-video boundary. NEVER set to 0."
        ),
    },
    "vace_feather_blocks": {
        "type": "FLOAT",
        "default": 1.5, "min": 0.0, "max": 8.0, "step": 0.25,
        "sentinel": _SENTINEL_FLOAT,
        "tooltip": (
            "VACE: feather conditioning mask edge over this many VAE blocks (1.5 = 12px for WAN). "
            "Smooth transition spanning >1 encoding block eliminates VAE ringing. NEVER set to 0."
        ),
    },
    "vace_halo_px": {
        "type": "INT",
        "default": 16, "min": 0, "max": 48, "step": 4,
        "sentinel": _SENTINEL_INT,
        "tooltip": (
            "Seam-Absorbing Halo: expand VACE conditioning mask OUTWARD beyond the stitch boundary. "
            "WAN repaints this strip, so the stitch falls inside repainted content. "
            "16px = recommended (2 VAE blocks). 0 = disabled. "
            "(Previously: halo_pixels on VaceControlVideoPrep, local only)"
        ),
    },
    "vace_stitch_erosion_px": {
        "type": "INT",
        "default": 0, "min": -32, "max": 32, "step": 1,
        "sentinel": _SENTINEL_INT,
        "tooltip": (
            "VACE: erode (negative) or dilate (positive) the pixel-space stitch mask. "
            "Independent from VACE erosion — controls compositing boundary, not conditioning. "
            "(Previously: vace_stitch_erosion)"
        ),
    },
    "vace_stitch_feather_px": {
        "type": "INT",
        "default": 8, "min": 0, "max": 64, "step": 1,
        "sentinel": _SENTINEL_INT,
        "tooltip": (
            "VACE: feather the pixel-space stitch mask edges for seamless compositing. "
            "8-16px = subtle, 24-32px = visible softening, 0 = hard edge. "
            "(Previously: vace_stitch_feather)"
        ),
    },
}


# =============================================================================
# Schema-driven INPUT_TYPES builders
# =============================================================================

def _build_config_input_types(schema):
    """Build INPUT_TYPES['required'] from schema for NV_MaskProcessingConfig.

    Each key becomes a regular widget with the schema's natural range and default.
    """
    out = {}
    for key, spec in schema.items():
        out[key] = (spec["type"], {
            "default": spec["default"],
            "min": spec["min"],
            "max": spec["max"],
            "step": spec["step"],
            "tooltip": spec["tooltip"],
        })
    return out


def _build_patch_input_types(schema):
    """Build INPUT_TYPES['required'] from schema for NV_MaskConfigPatch.

    Each key becomes a sentinel-default override widget. The widget's min is
    extended down to the sentinel value so users can keep "use base" as the
    default and explicitly type a real value to override that single key.
    """
    out = {}
    for key, spec in schema.items():
        sentinel = spec["sentinel"]
        is_int = spec["type"] == "INT"
        if is_int:
            range_str = f" Real range {spec['min']} to {spec['max']}."
            sentinel_str = f"  ({sentinel} = use base unchanged)"
        else:
            range_str = f" Real range {spec['min']:.1f} to {spec['max']:.1f}."
            sentinel_str = f"  ({sentinel} = use base unchanged)"
        out[key] = (spec["type"], {
            "default": sentinel,
            "min": sentinel,  # widget min extended to sentinel so default is selectable
            "max": spec["max"],
            "step": spec["step"],
            "tooltip": f"Override {key}.{range_str}{sentinel_str}",
        })
    return out


# =============================================================================
# Schema-driven loader normalization (Codex+Gemini debate Q6 verdict)
# =============================================================================

def normalize_mask_config(persisted_config, *, report_lines=None):
    """Normalize a loaded mask_config dict against the current schema.

    Behavior (per Codex+Gemini debate 2026-04-30 Q6 convergent verdict):
      - Deepcopies input (defensive — ComfyUI passes dicts by reference).
      - Fills missing keys with current schema defaults.
      - Produces canonical key order (schema iteration order).
      - Unknown keys: EXCLUDED from the canonical output dict (downstream
        consumers see only schema keys), but PRESERVED via the second tuple
        return value AND a warning in report_lines. This keeps consumers
        simple while retaining diagnostic evidence of forward-compat drift.

    Args:
        persisted_config: The mask_config dict from metadata.json (or None).
        report_lines: Optional list to append diagnostic strings to; the
            NV_LoadStitcher_V2 report uses this for human-debug provenance.

    Returns:
        (normalized_dict_or_None, list_of_unknown_keys)
    """
    if persisted_config is None:
        return None, []

    src = copy.deepcopy(persisted_config)
    normalized = {}
    missing = []
    for key, spec in MASK_CONFIG_SCHEMA.items():
        if key in src:
            normalized[key] = src.pop(key)
        else:
            normalized[key] = spec["default"]
            missing.append(key)

    unknown = list(src.keys())
    if report_lines is not None:
        if missing:
            report_lines.append(
                f"normalize_mask_config: filled {len(missing)} missing key(s) with "
                f"schema defaults: {missing}"
            )
        if unknown:
            report_lines.append(
                f"normalize_mask_config: WARNING: {len(unknown)} unknown key(s) "
                f"in saved config — not in current schema (likely from a newer save "
                f"or a renamed key). Excluded from canonical output. Keys: {unknown}"
            )
    return normalized, unknown


# =============================================================================
# Apply functions — merge config over local widget values (existing API)
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
# Node Classes — schema-generated INPUT_TYPES
# =============================================================================

class NV_MaskProcessingConfig:
    """Emit a shared mask processing config dict for consistent settings across nodes.

    Parameters are organized by category (cleanup, crop/stitch, VACE conditioning).
    INPUT_TYPES is generated from MASK_CONFIG_SCHEMA — adding a new param means
    adding one entry to that schema and nothing else.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": _build_config_input_types(MASK_CONFIG_SCHEMA)}

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

    def execute(self, **kwargs):
        # ComfyUI passes each widget value as a kwarg matching its INPUT_TYPES key.
        # Build the config dict in canonical schema order so downstream consumers
        # see deterministic key ordering.
        return ({key: kwargs[key] for key in MASK_CONFIG_SCHEMA},)


class NV_MaskConfigPatch:
    """Patch a base MASK_PROCESSING_CONFIG with per-key overrides.

    Use case: load a saved stitcher's mask_config (via NV_LoadStitcher_V2),
    keep most values, but tweak one or two before feeding into InpaintCrop_V2
    or VaceControlVideoPrep.

    Each override widget defaults to a sentinel value (-999 for ints, -999.0
    for floats) which means "use base value unchanged". Set to a real value
    in the legitimate range to override that specific key. Sentinels are
    chosen to be outside every key's natural range.

    Defensive: this node DEEPCOPIES the base config before applying overrides.
    Without that, mutating in place would alter the dict for any other node
    sharing the upstream wire — silent shared-state corruption (Codex+Gemini
    debate 2026-04-30, Q5 convergent verdict).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_config": ("MASK_PROCESSING_CONFIG", {
                    "tooltip": (
                        "Base config to patch. Typically wired from "
                        "NV_LoadStitcher_V2.mask_config or NV_MaskProcessingConfig."
                    ),
                }),
                **_build_patch_input_types(MASK_CONFIG_SCHEMA),
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
        "values without rebuilding the whole config from scratch. Deepcopies "
        "the base to prevent shared-reference mutation across wire branches."
    )

    def patch(self, base_config, **kwargs):
        if not isinstance(base_config, dict):
            raise TypeError(
                f"[NV_MaskConfigPatch] base_config must be a MASK_PROCESSING_CONFIG dict, "
                f"got {type(base_config).__name__}"
            )

        # DEEPCOPY before any mutation — ComfyUI dicts pass by reference between
        # nodes, so an in-place edit here would retroactively mutate the base
        # config for sibling nodes wired off the same upstream branch.
        result = copy.deepcopy(base_config)

        applied = []
        for key, spec in MASK_CONFIG_SCHEMA.items():
            if key not in kwargs:
                continue
            value = kwargs[key]
            if value != spec["sentinel"]:
                # Bounds check: the widget min was extended down to the sentinel
                # so 'use base' was a selectable default, but values strictly
                # between sentinel+1 and the natural min are NOT valid override
                # values — they only exist because the widget min was lowered.
                # Without this check, dragging the slider into the gap would
                # silently pass an out-of-bounds value to consumers.
                # (Gemini impl-review 2026-05-01, "Sentinel Gap" critical bug.)
                if not (spec["min"] <= value <= spec["max"]):
                    raise ValueError(
                        f"[NV_MaskConfigPatch] override for {key!r} = {value} is out of "
                        f"bounds [{spec['min']}, {spec['max']}]. The widget min was "
                        f"extended to the sentinel ({spec['sentinel']}) so 'use base' "
                        f"is selectable as the default, but values between sentinel+1 "
                        f"and the natural min are not valid overrides. Set the widget "
                        f"back to {spec['sentinel']} (use base) or to a value within the "
                        f"natural range [{spec['min']}, {spec['max']}]."
                    )
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
