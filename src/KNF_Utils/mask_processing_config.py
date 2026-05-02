"""
NV Mask Processing Config — bifurcated GEN / BLEND config buses.

D-189 / D-190 architecture (locked via multi-AI debate 2026-04-30, ratified +
implemented 2026-05-01):

A mask serves two semantically distinct purposes in this pipeline:

  GEN mask   — "where VACE may regenerate." Must cover the UNION of source +
               target silhouettes. Head-swap example: covers both the original
               head AND the new head's full extent.
               Consumed by: NV_VaceControlVideoPrep_V3.

  BLEND mask — "where the composite seam is hidden." Just covers the source
               silhouette at compositing time.
               Consumed by: NV_InpaintCrop3, NV_InpaintStitch3 (via stitcher),
                            NV_RebuildBlendMask_V2.

These were previously sharing a single MASK_PROCESSING_CONFIG bus → semantic
cross-talk between intents. The bifurcation makes the GEN/BLEND distinction a
first-class contract instead of a wiring trick.

The 3 retired widgets from the 2026-05-01 (pt 2) slim refactor — vace_erosion_
blocks, vace_feather_blocks, vace_halo_px — are gone from BOTH new schemas.
They were artifacts of an outdated misattribution that VACE preferred soft
feathered masks; binary-first research (paper § 3.2 arxiv 2503.07598) disproved
this. vace_input_grow_px (signed, ±128) is the single shape knob for the GEN
mask, subsuming the retired keys.

migrate_v1_to_split_configs() is exported for NV_LoadStitcher_V2 to use when
loading a persisted v1 mask_config — splits into both new buses, drops the 3
retired keys with diagnostic warning.
"""

import copy


# =============================================================================
# Shared cleanup defaults — single source of truth.
#
# Both GEN and BLEND schemas duplicate cleanup_fill_holes / cleanup_remove_noise
# / cleanup_smooth (per Q1 ruling: schemas self-contained to their domains).
# To prevent default drift across the two schemas (Codex risk #2 in plan
# review), the defaults are pulled from this dict — change once, applies to
# both schemas.
# =============================================================================

_SENTINEL_INT = -999
_SENTINEL_FLOAT = -999.0

CLEANUP_DEFAULTS = {
    "cleanup_fill_holes": {"default": 0, "min": 0, "max": 128, "step": 1},
    "cleanup_remove_noise": {"default": 0, "min": 0, "max": 32, "step": 1},
    "cleanup_smooth": {"default": 0, "min": 0, "max": 127, "step": 1},
}


def _cleanup_spec(key, tooltip):
    """Build a schema entry for a cleanup_* key from CLEANUP_DEFAULTS."""
    base = CLEANUP_DEFAULTS[key]
    return {
        "type": "INT",
        "default": base["default"],
        "min": base["min"],
        "max": base["max"],
        "step": base["step"],
        "sentinel": _SENTINEL_INT,
        "tooltip": tooltip,
    }


# =============================================================================
# MASK_GEN_SCHEMA — params for the GEN mask going to VACE conditioning.
# 4 keys.  STRICT APPEND-ONLY: ComfyUI binds widgets_values by position.
# =============================================================================

MASK_GEN_SCHEMA = {
    "cleanup_fill_holes": _cleanup_spec(
        "cleanup_fill_holes",
        "GEN-mask fill: close gaps/holes via morphological closing. "
        "Operates on the UNION mask going to VACE conditioning.",
    ),
    "cleanup_remove_noise": _cleanup_spec(
        "cleanup_remove_noise",
        "GEN-mask noise removal: erode-then-dilate to drop isolated specks "
        "before VACE conditioning sees the mask.",
    ),
    "cleanup_smooth": _cleanup_spec(
        "cleanup_smooth",
        "GEN-mask edge smoothing: binarize at 0.5 then Gaussian blur. "
        "Applied AFTER cleanup, BEFORE VACE encodes the mask.",
    ),
    "vace_input_grow_px": {
        "type": "INT",
        "default": 0, "min": -128, "max": 128, "step": 1,
        "sentinel": _SENTINEL_INT,
        "tooltip": (
            "Signed shape knob for the GEN mask. Positive = expand (cover "
            "UNION of source + target silhouettes; head-swap principle). "
            "Negative = shrink (e.g. fix over-segmented input). 0 = mask is "
            "already the right shape. Subsumes the retired vace_halo_px and "
            "erosion_blocks knobs."
        ),
    },
}


# =============================================================================
# MASK_BLEND_SCHEMA — params for the BLEND mask used at compositing.
# 7 keys (3 cleanup + 2 crop + 2 vace_stitch).
# =============================================================================

MASK_BLEND_SCHEMA = {
    "cleanup_fill_holes": _cleanup_spec(
        "cleanup_fill_holes",
        "BLEND-mask fill: close gaps/holes via morphological closing. "
        "Operates on the source-silhouette mask used for compositing.",
    ),
    "cleanup_remove_noise": _cleanup_spec(
        "cleanup_remove_noise",
        "BLEND-mask noise removal: erode-then-dilate to drop isolated "
        "specks in the compositing mask.",
    ),
    "cleanup_smooth": _cleanup_spec(
        "cleanup_smooth",
        "BLEND-mask edge smoothing: binarize at 0.5 then Gaussian blur. "
        "Applied to the mask used for the pixel-space stitch.",
    ),
    "crop_expand_px": {
        "type": "INT",
        "default": 0, "min": -128, "max": 128, "step": 1,
        "sentinel": _SENTINEL_INT,
        "tooltip": (
            "Shrink (negative) or expand (positive) the crop/stitch mask in "
            "pixels. Controls what gets denoised + stitched. (Was: "
            "mask_erode_dilate.)"
        ),
    },
    "crop_blend_feather_px": {
        "type": "INT",
        "default": 16, "min": 0, "max": 64, "step": 1,
        "sentinel": _SENTINEL_INT,
        "tooltip": (
            "Feather stitch mask edges for seamless compositing (dilate + "
            "blur). Controls pixel-space blend width at the stitch boundary. "
            "(Was: mask_blend_pixels.)"
        ),
    },
    "vace_stitch_erosion_px": {
        "type": "INT",
        "default": 0, "min": -32, "max": 32, "step": 1,
        "sentinel": _SENTINEL_INT,
        "tooltip": (
            "Erode (negative) or dilate (positive) the pixel-space stitch "
            "mask. Independent from GEN-side mask shape — controls "
            "compositing boundary, not VACE conditioning."
        ),
    },
    "vace_stitch_feather_px": {
        "type": "INT",
        "default": 8, "min": 0, "max": 64, "step": 1,
        "sentinel": _SENTINEL_INT,
        "tooltip": (
            "Feather the pixel-space stitch mask for seamless compositing. "
            "8-16px = subtle, 24-32px = visible softening, 0 = hard edge."
        ),
    },
}


# Retired keys — present in v1 MASK_PROCESSING_CONFIG saves, dropped on load
# with diagnostic warning. Subsumed by signed vace_input_grow_px in MASK_GEN.
RETIRED_V1_KEYS = ("vace_erosion_blocks", "vace_feather_blocks", "vace_halo_px")


# =============================================================================
# Schema-driven INPUT_TYPES builders (shared across both schemas).
# =============================================================================

def _build_config_input_types(schema):
    """Schema → INPUT_TYPES['required'] for a producer node."""
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
    """Schema → INPUT_TYPES['required'] for a patch node (sentinel defaults).

    Each key becomes a sentinel-default override widget. The widget's min is
    extended down to the sentinel so users can keep "use base" as the default
    and explicitly type a real value to override.
    """
    out = {}
    for key, spec in schema.items():
        sentinel = spec["sentinel"]
        is_int = spec["type"] == "INT"
        if is_int:
            range_str = f" Real range {spec['min']} to {spec['max']}."
        else:
            range_str = f" Real range {spec['min']:.1f} to {spec['max']:.1f}."
        sentinel_str = f"  ({sentinel} = use base unchanged)"
        out[key] = (spec["type"], {
            "default": sentinel,
            "min": sentinel,
            "max": spec["max"],
            "step": spec["step"],
            "tooltip": f"Override {key}.{range_str}{sentinel_str}",
        })
    return out


# =============================================================================
# Sentinel-default builders for CONSUMER nodes (Q2-c pattern).
# Consumer nodes use these so their widgets default to "use bus" when the bus
# is wired, with explicit override when the user types a real value.
# Distinct from the patch-node sentinel pattern: these widgets are the LOCAL
# fallback when no bus is wired AT ALL, not patches over a base bus.
# =============================================================================

def build_consumer_sentinel_widgets(schema):
    """Schema → INPUT_TYPES['required'] entries with sentinel defaults.

    Used by version-bumped consumer nodes (NV_VaceControlVideoPrep_V3,
    NV_InpaintCrop3, NV_RebuildBlendMask_V2). The widget's default sentinel
    means "use bus value when bus is wired; otherwise use the schema default."
    Setting the widget to a real value forces local override regardless of
    bus state.
    """
    out = {}
    for key, spec in schema.items():
        sentinel = spec["sentinel"]
        is_int = spec["type"] == "INT"
        if is_int:
            tip_range = f"Real range {spec['min']} to {spec['max']}."
        else:
            tip_range = f"Real range {spec['min']:.1f} to {spec['max']:.1f}."
        out[key] = (spec["type"], {
            "default": sentinel,
            "min": sentinel,
            "max": spec["max"],
            "step": spec["step"],
            "tooltip": (
                f"Local override for {key}. {sentinel} = defer to wired bus "
                f"or schema default ({spec['default']}) when no bus. {tip_range}"
            ),
        })
    return out


def resolve_consumer_value(local_value, bus_value, schema_default, sentinel):
    """Resolve a consumer-node value: local widget > bus > schema default.

    Consumer node calling pattern:
        cleanup_fill_holes = resolve_consumer_value(
            local_value=cleanup_fill_holes_widget,
            bus_value=(mask_config or {}).get("cleanup_fill_holes"),
            schema_default=MASK_GEN_SCHEMA["cleanup_fill_holes"]["default"],
            sentinel=MASK_GEN_SCHEMA["cleanup_fill_holes"]["sentinel"],
        )
    """
    if local_value != sentinel:
        return local_value
    if bus_value is not None:
        return bus_value
    return schema_default


def resolve_all_from_bus(schema, bus, **local_widget_values):
    """Convenience: resolve every key in schema using bus + locals.

    Example:
        vals = resolve_all_from_bus(MASK_GEN_SCHEMA, mask_config,
            cleanup_fill_holes=cleanup_fill_holes_widget,
            cleanup_remove_noise=cleanup_remove_noise_widget,
            ...)
    """
    bus = bus or {}
    out = {}
    for key, spec in schema.items():
        local = local_widget_values.get(key, spec["sentinel"])
        out[key] = resolve_consumer_value(
            local_value=local,
            bus_value=bus.get(key),
            schema_default=spec["default"],
            sentinel=spec["sentinel"],
        )
    return out


# =============================================================================
# Persistence normalizers (one per schema).
# Used by NV_LoadStitcher_V2 to fill missing keys, drop unknowns, return
# diagnostic info for the load report.
# =============================================================================

def _normalize_against_schema(persisted_config, schema, schema_name, *, report_lines=None):
    """Generic normalize: deepcopy, fill missing keys, exclude unknowns."""
    if persisted_config is None:
        return None, []

    src = copy.deepcopy(persisted_config)
    normalized = {}
    missing = []
    for key, spec in schema.items():
        if key in src:
            normalized[key] = src.pop(key)
        else:
            normalized[key] = spec["default"]
            missing.append(key)

    unknown = list(src.keys())
    if report_lines is not None:
        if missing:
            report_lines.append(
                f"normalize_{schema_name}: filled {len(missing)} missing key(s) "
                f"with schema defaults: {missing}"
            )
        if unknown:
            report_lines.append(
                f"normalize_{schema_name}: WARNING: {len(unknown)} unknown "
                f"key(s) not in current schema (renamed/newer save). "
                f"Excluded from canonical output. Keys: {unknown}"
            )
    return normalized, unknown


def normalize_gen_config(persisted_config, *, report_lines=None):
    """Normalize a loaded GEN config dict against MASK_GEN_SCHEMA."""
    return _normalize_against_schema(
        persisted_config, MASK_GEN_SCHEMA, "gen_config",
        report_lines=report_lines,
    )


def normalize_blend_config(persisted_config, *, report_lines=None):
    """Normalize a loaded BLEND config dict against MASK_BLEND_SCHEMA."""
    return _normalize_against_schema(
        persisted_config, MASK_BLEND_SCHEMA, "blend_config",
        report_lines=report_lines,
    )


# =============================================================================
# v1 → split migration (used by NV_LoadStitcher_V2 when loading old saves).
# =============================================================================

def migrate_v1_to_split_configs(v1_config, *, report_lines=None):
    """Translate a v1 MASK_PROCESSING_CONFIG dict into (gen, blend) dicts.

    Splits the monolithic v1 schema into the two new buses. The 3 retired keys
    (vace_erosion_blocks, vace_feather_blocks, vace_halo_px) are dropped with
    a diagnostic warning if report_lines is provided.

    Per Q5 ruling: auto-split with diagnostic warning. Workflows are ephemeral
    (visual breakage acceptable) but saved stitcher payloads are data-at-rest;
    silent retirement of keys would harm provenance.

    Args:
        v1_config: dict from a v1 stitcher save (or None).
        report_lines: optional list to append diagnostic messages to.

    Returns:
        (gen_dict, blend_dict, dropped_keys) — all None if v1_config is None.
    """
    if v1_config is None:
        return None, None, []

    src = copy.deepcopy(v1_config)
    dropped = []

    # Drop retired keys with explicit logging.
    for key in RETIRED_V1_KEYS:
        if key in src:
            value = src.pop(key)
            dropped.append((key, value))

    # Build GEN dict from v1 keys that map into GEN.
    gen_dict = {}
    for key in MASK_GEN_SCHEMA:
        if key in src:
            gen_dict[key] = src[key]
        else:
            gen_dict[key] = MASK_GEN_SCHEMA[key]["default"]

    # Build BLEND dict from v1 keys that map into BLEND.
    blend_dict = {}
    for key in MASK_BLEND_SCHEMA:
        if key in src:
            blend_dict[key] = src[key]
        else:
            blend_dict[key] = MASK_BLEND_SCHEMA[key]["default"]

    # Anything left in `src` was an unknown key (not in old or new schemas).
    leftover = [k for k in src if k not in MASK_GEN_SCHEMA and k not in MASK_BLEND_SCHEMA]

    if report_lines is not None:
        if dropped:
            report_lines.append(
                f"migrate_v1_to_split_configs: dropped {len(dropped)} retired "
                f"v1 key(s) — these were artifacts of pre-binary-first VACE "
                f"lore (paper § 3.2). Subsumed by signed vace_input_grow_px. "
                f"Dropped: {[(k, v) for k, v in dropped]}"
            )
        if leftover:
            report_lines.append(
                f"migrate_v1_to_split_configs: WARNING: {len(leftover)} unknown "
                f"key(s) in v1 save — not in either new schema. Discarded: "
                f"{leftover}"
            )
        report_lines.append(
            f"migrate_v1_to_split_configs: split into MASK_GEN_CONFIG "
            f"({len(gen_dict)} keys) and MASK_BLEND_CONFIG ({len(blend_dict)} "
            f"keys)."
        )

    return gen_dict, blend_dict, [k for k, _ in dropped]


# =============================================================================
# Producer nodes — emit the GEN and BLEND buses.
# =============================================================================

class NV_MaskGenConfig:
    """Emit a MASK_GEN_CONFIG bus for VACE conditioning mask params.

    Wire into NV_VaceControlVideoPrep_V3.mask_config to drive the GEN-side
    mask pipeline (cleanup + signed shape knob). 4 keys.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": _build_config_input_types(MASK_GEN_SCHEMA)}

    RETURN_TYPES = ("MASK_GEN_CONFIG",)
    RETURN_NAMES = ("mask_gen_config",)
    FUNCTION = "execute"
    CATEGORY = "NV_Utils/Inpaint"
    DESCRIPTION = (
        "GEN-side mask config bus — for the mask going INTO VACE conditioning "
        "(D-189: covers UNION of source + target silhouettes). 4 keys: 3 "
        "cleanup + 1 signed shape knob (vace_input_grow_px). Wire into "
        "NV_VaceControlVideoPrep_V3.mask_config."
    )

    def execute(self, **kwargs):
        return ({key: kwargs[key] for key in MASK_GEN_SCHEMA},)


class NV_MaskBlendConfig:
    """Emit a MASK_BLEND_CONFIG bus for compositing mask params.

    Wire into NV_InpaintCrop3 / NV_InpaintStitch3 / NV_RebuildBlendMask_V2 to
    drive the BLEND-side mask pipeline (cleanup + crop expand/feather + stitch
    erosion/feather). 7 keys.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": _build_config_input_types(MASK_BLEND_SCHEMA)}

    RETURN_TYPES = ("MASK_BLEND_CONFIG",)
    RETURN_NAMES = ("mask_blend_config",)
    FUNCTION = "execute"
    CATEGORY = "NV_Utils/Inpaint"
    DESCRIPTION = (
        "BLEND-side mask config bus — for the mask used at COMPOSITING "
        "(D-189: covers source silhouette only, hides the seam). 7 keys: 3 "
        "cleanup + 2 crop + 2 stitch. Wire into NV_InpaintCrop3 + "
        "NV_InpaintStitch3 + NV_RebuildBlendMask_V2."
    )

    def execute(self, **kwargs):
        return ({key: kwargs[key] for key in MASK_BLEND_SCHEMA},)


# =============================================================================
# Patch node — sentinel-default override for the BLEND bus.
#
# Q4 ruling: only ship the BLEND patch initially. GEN side has 4 keys with
# vace_input_grow_px being the only non-cleanup knob; late-stage Gen overrides
# are rare. YAGNI — add NV_MaskGenConfigPatch later if a real use case surfaces.
# =============================================================================

class NV_MaskBlendConfigPatch:
    """Patch a base MASK_BLEND_CONFIG with per-key overrides.

    Each override widget defaults to a sentinel (-999) which means "use base
    unchanged". Set to a real value to override that specific key. Sentinels
    are outside every key's natural range. Deepcopies the base before mutating
    to prevent shared-reference corruption (Codex+Gemini debate Q5).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_config": ("MASK_BLEND_CONFIG", {
                    "tooltip": (
                        "Base BLEND config to patch. Typically wired from "
                        "NV_LoadStitcher_V2.mask_blend_config or "
                        "NV_MaskBlendConfig."
                    ),
                }),
                **_build_patch_input_types(MASK_BLEND_SCHEMA),
            },
        }

    RETURN_TYPES = ("MASK_BLEND_CONFIG",)
    RETURN_NAMES = ("patched_config",)
    FUNCTION = "patch"
    CATEGORY = "NV_Utils/Inpaint"
    DESCRIPTION = (
        "Patch a MASK_BLEND_CONFIG with per-key overrides. Sentinel "
        "(-999) = use base unchanged; real value = override. Deepcopies "
        "to prevent shared-reference mutation across wire branches."
    )

    def patch(self, base_config, **kwargs):
        if not isinstance(base_config, dict):
            raise TypeError(
                f"[NV_MaskBlendConfigPatch] base_config must be a "
                f"MASK_BLEND_CONFIG dict, got {type(base_config).__name__}"
            )

        # DEEPCOPY before any mutation — ComfyUI dicts pass by reference between
        # nodes; in-place edit would retroactively mutate the base for sibling
        # nodes wired off the same upstream branch.
        result = copy.deepcopy(base_config)

        applied = []
        for key, spec in MASK_BLEND_SCHEMA.items():
            if key not in kwargs:
                continue
            value = kwargs[key]
            if value != spec["sentinel"]:
                # Sentinel-gap bounds check: widget min was extended to sentinel
                # so 'use base' was selectable as the default, but values
                # strictly between sentinel+1 and the natural min are NOT valid
                # overrides — they only exist because of the widget-min
                # extension. Reject them explicitly.
                if not (spec["min"] <= value <= spec["max"]):
                    raise ValueError(
                        f"[NV_MaskBlendConfigPatch] override for {key!r} = "
                        f"{value} is out of bounds [{spec['min']}, "
                        f"{spec['max']}]. Set the widget back to "
                        f"{spec['sentinel']} (use base) or to a value within "
                        f"the natural range [{spec['min']}, {spec['max']}]."
                    )
                old = result.get(key, "<absent>")
                result[key] = value
                applied.append(f"{key}: {old} -> {value}")

        if applied:
            print(f"[NV_MaskBlendConfigPatch] Applied {len(applied)} override(s):")
            for line in applied:
                print(f"  {line}")
        else:
            print("[NV_MaskBlendConfigPatch] No overrides — config passes through unchanged")

        return (result,)


NODE_CLASS_MAPPINGS = {
    "NV_MaskGenConfig": NV_MaskGenConfig,
    "NV_MaskBlendConfig": NV_MaskBlendConfig,
    "NV_MaskBlendConfigPatch": NV_MaskBlendConfigPatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_MaskGenConfig": "NV Mask GEN Config",
    "NV_MaskBlendConfig": "NV Mask BLEND Config",
    "NV_MaskBlendConfigPatch": "NV Mask BLEND Config Patch",
}
