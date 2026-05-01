"""Standalone test runner for MASK_CONFIG_SCHEMA — guards strict append-only key order.

Usage:
    python tests/run_mask_config_schema_tests.py

Bypasses the package __init__.py chain (which requires ComfyUI runtime context)
by importing mask_processing_config.py directly via importlib.util. Mirrors the
pattern in run_v2v_tests.py / run_r2v_splitter_tests.py.

The pytest version (test_mask_config_schema.py) covers the same cases for
CI/IDE integration; this runner is the in-project execution path.
"""
import importlib.util
import os
import sys

_MODULE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "src", "KNF_Utils", "mask_processing_config.py"
)
spec = importlib.util.spec_from_file_location("mask_processing_config", _MODULE_PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

MASK_CONFIG_SCHEMA = mod.MASK_CONFIG_SCHEMA
normalize_mask_config = mod.normalize_mask_config


# Frozen snapshot as of 2026-04-30. Append new keys at the END only — never
# insert in the middle. ComfyUI's positional widgets_values mapping depends on
# this order being stable across schema revisions. (Codex+Gemini debate Q3.)
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


# --- Test infrastructure ---
passed = 0
failed = 0
errors = []


def run_test(name, fn):
    global passed, failed
    try:
        fn()
        passed += 1
        print(f"  PASS: {name}")
    except Exception as e:
        failed += 1
        errors.append((name, e))
        print(f"  FAIL: {name} - {e}")


print("Running MASK_CONFIG_SCHEMA tests...")
print()


# === Strict append-only schema-order rule ===

def test_schema_keys_preserve_append_only_order():
    actual_keys = tuple(MASK_CONFIG_SCHEMA.keys())
    n = len(FROZEN_KEYS)
    if len(actual_keys) < n:
        raise AssertionError(
            f"MASK_CONFIG_SCHEMA shrank from {n} keys to {len(actual_keys)}. "
            f"Removing keys breaks ComfyUI positional widgets_values mapping."
        )
    actual_prefix = actual_keys[:n]
    if actual_prefix != FROZEN_KEYS:
        raise AssertionError(
            f"MASK_CONFIG_SCHEMA key order changed.\n"
            f"  Expected first {n} keys: {FROZEN_KEYS}\n"
            f"  Actual first {n} keys:   {actual_prefix}\n"
            f"New keys MUST be APPENDED at the end. If intentional, update FROZEN_KEYS."
        )


run_test("schema_keys_preserve_append_only_order", test_schema_keys_preserve_append_only_order)


# === Schema entry metadata completeness ===

def test_schema_entries_have_required_metadata():
    required_metadata = {"type", "default", "min", "max", "step", "sentinel", "tooltip"}
    for key, spec in MASK_CONFIG_SCHEMA.items():
        missing = required_metadata - spec.keys()
        if missing:
            raise AssertionError(f"Schema entry {key!r} missing metadata fields: {missing}")
        if spec["type"] not in ("INT", "FLOAT"):
            raise AssertionError(
                f"Schema entry {key!r} has invalid type {spec['type']!r}; expected INT or FLOAT"
            )


run_test("schema_entries_have_required_metadata", test_schema_entries_have_required_metadata)


def test_sentinel_outside_natural_range():
    for key, spec in MASK_CONFIG_SCHEMA.items():
        if spec["sentinel"] >= spec["min"]:
            raise AssertionError(
                f"Schema entry {key!r}: sentinel ({spec['sentinel']}) must be < min ({spec['min']}) "
                f"to avoid collision with legitimate values"
            )


run_test("sentinel_outside_natural_range", test_sentinel_outside_natural_range)


# === normalize_mask_config behavior ===

def test_normalize_fills_missing_keys_with_defaults():
    partial = {"cleanup_fill_holes": 5}
    report = []
    normalized, unknown = normalize_mask_config(partial, report_lines=report)
    assert normalized is not None
    assert normalized["cleanup_fill_holes"] == 5
    for key, spec in MASK_CONFIG_SCHEMA.items():
        assert key in normalized, f"normalized missing schema key {key!r}"
        if key != "cleanup_fill_holes":
            assert normalized[key] == spec["default"], (
                f"key {key!r}: expected default {spec['default']!r}, got {normalized[key]!r}"
            )
    assert unknown == []
    assert any("filled" in line for line in report), "expected 'filled missing' report line"


run_test("normalize_fills_missing_keys_with_defaults", test_normalize_fills_missing_keys_with_defaults)


def test_normalize_warns_on_unknown_keys():
    config_with_extra = {"cleanup_fill_holes": 0, "totally_made_up_key": 42}
    report = []
    normalized, unknown = normalize_mask_config(config_with_extra, report_lines=report)
    assert normalized is not None
    assert "totally_made_up_key" not in normalized
    assert unknown == ["totally_made_up_key"]
    assert any("WARNING" in line and "totally_made_up_key" in line for line in report), (
        "expected unknown-key warning in report_lines"
    )


run_test("normalize_warns_on_unknown_keys", test_normalize_warns_on_unknown_keys)


def test_normalize_returns_canonical_key_order():
    reversed_input = {key: spec["default"] for key, spec in reversed(list(MASK_CONFIG_SCHEMA.items()))}
    normalized, _ = normalize_mask_config(reversed_input)
    if tuple(normalized.keys()) != tuple(MASK_CONFIG_SCHEMA.keys()):
        raise AssertionError(
            f"normalize_mask_config did not produce canonical (schema) key order.\n"
            f"  Expected: {tuple(MASK_CONFIG_SCHEMA.keys())}\n"
            f"  Actual:   {tuple(normalized.keys())}"
        )


run_test("normalize_returns_canonical_key_order", test_normalize_returns_canonical_key_order)


def test_normalize_returns_none_for_none_input():
    normalized, unknown = normalize_mask_config(None)
    assert normalized is None
    assert unknown == []


run_test("normalize_returns_none_for_none_input", test_normalize_returns_none_for_none_input)


def test_normalize_deepcopies_input():
    original = {"cleanup_fill_holes": 5}
    normalized, _ = normalize_mask_config(original)
    normalized["cleanup_fill_holes"] = 999
    if original["cleanup_fill_holes"] != 5:
        raise AssertionError(
            "normalize_mask_config did not deepcopy — mutating output corrupted input"
        )


run_test("normalize_deepcopies_input", test_normalize_deepcopies_input)


# === Summary ===
print()
print(f"Results: {passed} passed, {failed} failed out of {passed + failed} tests")
if errors:
    print()
    for name, e in errors:
        print(f"  FAILED {name}: {e}")
    sys.exit(1)
else:
    print("All tests passed!")
