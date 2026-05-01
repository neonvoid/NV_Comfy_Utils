"""Tests for MASK_CONFIG_SCHEMA — guards the strict append-only key-order rule.

Why this test exists:
    ComfyUI maps `widgets_values` BY POSITION, not by name. NV_MaskProcessingConfig
    and NV_MaskConfigPatch both build their INPUT_TYPES from MASK_CONFIG_SCHEMA.
    Inserting a new key mid-schema (or reordering existing keys) silently scrambles
    every saved workflow's widget values — diffusion params end up in cleanup
    slots, etc. The failure is invisible at load time and only surfaces as wrong
    pipeline output.

    This test freezes the schema's known key prefix. If a developer (or AI) edits
    MASK_CONFIG_SCHEMA in a way that violates append-only order, the test fails
    loudly with a clear remediation message.

Add a new key:
    1. Append it at the END of MASK_CONFIG_SCHEMA in mask_processing_config.py.
    2. Append it at the END of FROZEN_KEYS below.
    3. Both lists' first N entries must remain bit-identical to prior versions.

Intentional breaking change (rare):
    Coordinate with anyone holding saved workflows. Update FROZEN_KEYS to match
    the new order and document the migration in CHANGELOG.md.

(Origin: Codex+Gemini debate 2026-04-30, Q3 convergent verdict.)
"""

from src.KNF_Utils.mask_processing_config import (
    MASK_CONFIG_SCHEMA,
    normalize_mask_config,
)


# Frozen snapshot as of 2026-04-30. Append new keys at the END only — never
# insert in the middle. ComfyUI's positional widgets_values mapping depends on
# this order being stable across schema revisions.
FROZEN_KEYS = (
    "cleanup_fill_holes",
    "cleanup_remove_noise",
    "cleanup_smooth",
    "crop_expand_px",
    "crop_blend_feather_px",
    "vace_input_grow_px",
    "vace_erosion_blocks",
    "vace_feather_blocks",
    "vace_halo_px",
    "vace_stitch_erosion_px",
    "vace_stitch_feather_px",
)


def test_schema_keys_preserve_append_only_order():
    """Verify the frozen N keys are still at the START of MASK_CONFIG_SCHEMA in same order."""
    actual_keys = tuple(MASK_CONFIG_SCHEMA.keys())
    n = len(FROZEN_KEYS)
    assert len(actual_keys) >= n, (
        f"MASK_CONFIG_SCHEMA shrank from {n} keys to {len(actual_keys)}. "
        f"Removing keys breaks ComfyUI's positional widgets_values mapping in saved workflows. "
        f"If this is intentional, document the breaking change and update FROZEN_KEYS."
    )
    actual_prefix = actual_keys[:n]
    assert actual_prefix == FROZEN_KEYS, (
        f"\nMASK_CONFIG_SCHEMA key order has changed.\n"
        f"  Expected first {n} keys (FROZEN_KEYS): {FROZEN_KEYS}\n"
        f"  Actual first {n} keys:                  {actual_prefix}\n"
        f"\n"
        f"This breaks ComfyUI's positional widgets_values mapping in saved workflows.\n"
        f"New keys MUST be APPENDED at the end of MASK_CONFIG_SCHEMA, never inserted mid-list.\n"
        f"If this change is intentional, update FROZEN_KEYS in this test and document the breaking change."
    )


def test_schema_entries_have_required_metadata():
    """Each schema entry must have type, default, min, max, step, sentinel, tooltip."""
    required_metadata = {"type", "default", "min", "max", "step", "sentinel", "tooltip"}
    for key, spec in MASK_CONFIG_SCHEMA.items():
        missing = required_metadata - spec.keys()
        assert not missing, f"Schema entry {key!r} missing metadata fields: {missing}"
        assert spec["type"] in ("INT", "FLOAT"), (
            f"Schema entry {key!r} has invalid type {spec['type']!r}; expected INT or FLOAT"
        )


def test_sentinel_outside_natural_range():
    """Each key's sentinel must be strictly less than its natural min — otherwise a
    legitimate user value could collide with 'use base unchanged' semantics."""
    for key, spec in MASK_CONFIG_SCHEMA.items():
        assert spec["sentinel"] < spec["min"], (
            f"Schema entry {key!r}: sentinel ({spec['sentinel']}) must be < min ({spec['min']}) "
            f"to avoid collision with legitimate values"
        )


def test_normalize_fills_missing_keys_with_defaults():
    """A persisted config missing some keys gets filled from current schema defaults."""
    partial = {"cleanup_fill_holes": 5}
    report = []
    normalized, unknown = normalize_mask_config(partial, report_lines=report)
    assert normalized is not None
    assert normalized["cleanup_fill_holes"] == 5  # preserved
    # All other keys should be filled with their schema default
    for key, spec in MASK_CONFIG_SCHEMA.items():
        assert key in normalized, f"normalized missing schema key {key!r}"
        if key != "cleanup_fill_holes":
            assert normalized[key] == spec["default"], (
                f"key {key!r}: expected default {spec['default']!r}, got {normalized[key]!r}"
            )
    assert unknown == []
    assert any("filled" in line for line in report), "expected a 'filled missing' report line"


def test_normalize_warns_on_unknown_keys():
    """Unknown keys are excluded from canonical output and warned via report_lines."""
    config_with_extra = {"cleanup_fill_holes": 0, "totally_made_up_key": 42}
    report = []
    normalized, unknown = normalize_mask_config(config_with_extra, report_lines=report)
    assert normalized is not None
    assert "totally_made_up_key" not in normalized  # not in current schema → not in canonical output
    assert unknown == ["totally_made_up_key"]
    assert any("WARNING" in line and "totally_made_up_key" in line for line in report), (
        "expected an 'unknown key' warning in report_lines"
    )


def test_normalize_returns_canonical_key_order():
    """Normalized output keys appear in MASK_CONFIG_SCHEMA order regardless of input order."""
    # Construct input with keys in REVERSE schema order
    reversed_input = {key: spec["default"] for key, spec in reversed(list(MASK_CONFIG_SCHEMA.items()))}
    normalized, _ = normalize_mask_config(reversed_input)
    assert tuple(normalized.keys()) == tuple(MASK_CONFIG_SCHEMA.keys()), (
        "normalize_mask_config did not produce canonical (schema) key order"
    )


def test_normalize_returns_none_for_none_input():
    """None input returns None (don't synthesize a fake config when one wasn't saved)."""
    normalized, unknown = normalize_mask_config(None)
    assert normalized is None
    assert unknown == []


def test_normalize_deepcopies_input():
    """Mutating the returned dict must NOT mutate the input dict (deepcopy semantics)."""
    original = {"cleanup_fill_holes": 5}
    normalized, _ = normalize_mask_config(original)
    normalized["cleanup_fill_holes"] = 999
    assert original["cleanup_fill_holes"] == 5, (
        "normalize_mask_config did not deepcopy — mutating output corrupted input"
    )
