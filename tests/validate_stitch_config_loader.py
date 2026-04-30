"""Standalone validation script for stitch_config_loader.

Run with:  python tests/validate_stitch_config_loader.py

Avoids pytest + the package's ComfyUI-runtime __init__.py by loading the
loader module via importlib.spec_from_file_location. Returns 0 on PASS, 1
on FAIL. Each check prints a OK/FAIL line.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]


def _load_module(name, file_path):
    spec = importlib.util.spec_from_file_location(name, str(file_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


loader = _load_module(
    "stitch_config_loader",
    REPO / "src" / "KNF_Utils" / "stitch_config_loader.py",
)


fails: list[str] = []


def check(label, condition, detail=""):
    if condition:
        print(f"  OK   {label}")
    else:
        print(f"  FAIL {label}  ({detail})")
        fails.append(label)


def expect_raises(label, exc_type, fn):
    try:
        fn()
        check(label, False, f"expected {exc_type.__name__}, got no exception")
    except exc_type as e:
        check(label, True)
    except Exception as e:
        check(label, False, f"got {type(e).__name__}: {e}")


def _valid_config(**overrides):
    base = {
        "schema_version": "stitch-config-1.0",
        "shot_id": "test_shot",
        "bundle_ref": "test_shot_3f.safetensors",
        "bundle_fingerprint": None,
        "exported_at": "2026-04-30T12:00:00+00:00",
        "effective": {
            "crop_params": {
                "crop_stitch_source": "processed",
                "crop_blend_feather_px": 24,
                "hybrid_falloff": 64,
                "hybrid_curve": 0.7,
            },
            "stitch_params": {
                "blend_mode": "multiband",
                "multiband_levels": 6,
                "guided_refine": True,
            },
        },
        "diff": {"crop_params": {}, "stitch_params": {}},
        "edited_keys": [],
        "pre_diffusion_keys_changed": [],
    }
    base.update(overrides)
    return base


def _stub_crop_kwargs():
    return {
        "target_mode": "manual", "target_width": 512, "target_height": 512,
        "auto_preset": "WAN_480p", "padding_multiple": "32",
        "cleanup_fill_holes": 0, "cleanup_remove_noise": 0, "cleanup_smooth": 0,
        "crop_expand_px": 0, "crop_stitch_source": "tight",
        "crop_blend_feather_px": 16, "hybrid_falloff": 48, "hybrid_curve": 0.6,
        "resize_algorithm": "bicubic", "anomaly_threshold": 1.5,
    }


def _stub_stitch_kwargs():
    return {
        "blend_mode": "alpha", "multiband_levels": 5, "guided_refine": False,
        "guided_radius": 8, "guided_eps": 0.001, "guided_strength": 0.7,
        "output_dtype": "fp16",
    }


def _write_json(tmp, name, data):
    p = tmp / name
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


# =============================================================================
# Run checks
# =============================================================================

print("=== load_editor_config — schema gate ===")

with tempfile.TemporaryDirectory() as tmp_str:
    tmp = Path(tmp_str)

    expect_raises(
        "missing file raises StitchConfigError",
        loader.StitchConfigError,
        lambda: loader.load_editor_config(str(tmp / "nope.json")),
    )

    p_bad = tmp / "bad.json"
    p_bad.write_text("{not json", encoding="utf-8")
    expect_raises(
        "invalid JSON raises StitchConfigError",
        loader.StitchConfigError,
        lambda: loader.load_editor_config(str(p_bad)),
    )

    p_arr = tmp / "arr.json"
    p_arr.write_text("[1,2,3]", encoding="utf-8")
    expect_raises(
        "top-level not dict raises StitchConfigError",
        loader.StitchConfigError,
        lambda: loader.load_editor_config(str(p_arr)),
    )

    bad = _valid_config()
    bad["schema_version"] = "stitch-config-2.0"
    p_bad_schema = _write_json(tmp, "bad_schema.json", bad)
    expect_raises(
        "unsupported schema_version raises",
        loader.StitchConfigError,
        lambda: loader.load_editor_config(str(p_bad_schema)),
    )

    # Schema gate is now strict (multi-AI 2026-04-30 MED): explicit known
    # minors only. 1.5 is NOT in the known set, so this should fail.
    # Detailed validation lives in the "Schema gate strictness" block below.
    base = _write_json(tmp, "base.json", _valid_config())
    loaded = loader.load_editor_config(str(base))
    check("stitch-config-1.0 (the canonical version) loads",
          loaded["schema_version"] == "stitch-config-1.0")

    bad_eff = _valid_config()
    bad_eff["effective"] = "not dict"
    p_be = _write_json(tmp, "be.json", bad_eff)
    expect_raises(
        "effective not dict raises",
        loader.StitchConfigError,
        lambda: loader.load_editor_config(str(p_be)),
    )

print("\n=== apply_crop_overrides ===")

cfg = _valid_config()
out, applied, unknown = loader.apply_crop_overrides(_stub_crop_kwargs(), cfg)
check("crop override: stitch_source", out["crop_stitch_source"] == "processed")
check("crop override: blend_feather", out["crop_blend_feather_px"] == 24)
check("crop override: hybrid_falloff", out["hybrid_falloff"] == 64)
check("crop override: hybrid_curve", out["hybrid_curve"] == 0.7)
check("crop unchanged: target_width preserved", out["target_width"] == 512)
check("applied list contains stitch_source", any("crop_stitch_source" in a for a in applied))
check("no unknown keys for known schema", unknown == [])

# Coercion: int -> float
cfg = _valid_config(effective={"crop_params": {"hybrid_curve": 1}, "stitch_params": {}})
out, _, _ = loader.apply_crop_overrides(_stub_crop_kwargs(), cfg)
check("int 1 coerces to float 1.0 for hybrid_curve", out["hybrid_curve"] == 1.0)

# Coercion: float with .is_integer() -> int
cfg = _valid_config(effective={"crop_params": {"crop_blend_feather_px": 32.0}, "stitch_params": {}})
out, _, _ = loader.apply_crop_overrides(_stub_crop_kwargs(), cfg)
check("32.0 coerces to int 32", out["crop_blend_feather_px"] == 32 and isinstance(out["crop_blend_feather_px"], int))

# Range violation
cfg = _valid_config(effective={"crop_params": {"crop_blend_feather_px": 9999}, "stitch_params": {}})
expect_raises(
    "out-of-range crop_blend_feather_px hard-fails",
    loader.StitchConfigError,
    lambda: loader.apply_crop_overrides(_stub_crop_kwargs(), cfg),
)

# Choice violation
cfg = _valid_config(effective={"crop_params": {"crop_stitch_source": "bogus"}, "stitch_params": {}})
expect_raises(
    "invalid stitch_source choice hard-fails",
    loader.StitchConfigError,
    lambda: loader.apply_crop_overrides(_stub_crop_kwargs(), cfg),
)

# Wrong type
cfg = _valid_config(effective={"crop_params": {"crop_blend_feather_px": "twenty"}, "stitch_params": {}})
expect_raises(
    "wrong-type crop_blend_feather_px hard-fails",
    loader.StitchConfigError,
    lambda: loader.apply_crop_overrides(_stub_crop_kwargs(), cfg),
)

# Unknown key warns, doesn't fail
cfg = _valid_config(effective={
    "crop_params": {"crop_blend_feather_px": 24, "future_param": 1},
    "stitch_params": {},
})
out, applied, unknown = loader.apply_crop_overrides(_stub_crop_kwargs(), cfg)
check("unknown key: known applied", out["crop_blend_feather_px"] == 24)
check("unknown key: surfaced in unknown list", any("future_param" in u for u in unknown))

# bool rejected for int field
cfg = _valid_config(effective={"crop_params": {"crop_blend_feather_px": True}, "stitch_params": {}})
expect_raises(
    "bool rejected for int field",
    loader.StitchConfigError,
    lambda: loader.apply_crop_overrides(_stub_crop_kwargs(), cfg),
)

print("\n=== apply_stitch_overrides ===")

cfg = _valid_config()
out, applied, _ = loader.apply_stitch_overrides(_stub_stitch_kwargs(), cfg)
check("stitch override: blend_mode", out["blend_mode"] == "multiband")
check("stitch override: multiband_levels", out["multiband_levels"] == 6)
check("stitch override: guided_refine bool", out["guided_refine"] is True)
check("stitch unchanged: guided_radius preserved", out["guided_radius"] == 8)

cfg = _valid_config(effective={"crop_params": {}, "stitch_params": {"blend_mode": "bogus"}})
expect_raises(
    "invalid blend_mode hard-fails",
    loader.StitchConfigError,
    lambda: loader.apply_stitch_overrides(_stub_stitch_kwargs(), cfg),
)

cfg = _valid_config(effective={"crop_params": {}, "stitch_params": {"guided_eps": 100.0}})
expect_raises(
    "guided_eps out-of-range hard-fails",
    loader.StitchConfigError,
    lambda: loader.apply_stitch_overrides(_stub_stitch_kwargs(), cfg),
)

print("\n=== maybe_apply_* helpers ===")

kwargs = _stub_crop_kwargs()
out = loader.maybe_apply_crop_config(kwargs, "")
check("empty path is no-op", out == kwargs)

out = loader.maybe_apply_crop_config(kwargs, "   ")
check("whitespace path is no-op", out == kwargs)

with tempfile.TemporaryDirectory() as tmp_str:
    tmp = Path(tmp_str)
    p = _write_json(tmp, "cfg.json", _valid_config())
    out = loader.maybe_apply_crop_config(_stub_crop_kwargs(), str(p))
    check("maybe_apply_crop full round-trip", out["crop_stitch_source"] == "processed")

    out = loader.maybe_apply_stitch_config(_stub_stitch_kwargs(), str(p))
    check("maybe_apply_stitch full round-trip",
          out["blend_mode"] == "multiband" and out["multiband_levels"] == 6)

print("\n=== check_canvas_drift ===")

cfg = _valid_config(bundle_fingerprint=None)
result = loader.check_canvas_drift(cfg, torch.zeros((1, 8, 8, 3)))
check("no fingerprint = drift None", result is None)

canvas = torch.randn((1, 16, 16, 3))
fp_hex = loader._hash_tensor_blake2b(canvas)
cfg = _valid_config(bundle_fingerprint={"algo": "blake2b-128", "canvas_image": fp_hex})
result = loader.check_canvas_drift(cfg, canvas)
check("fingerprint match = drift None", result is None)

canvas_old = torch.zeros((1, 16, 16, 3))
canvas_new = torch.ones((1, 16, 16, 3))
cfg = _valid_config(bundle_fingerprint={
    "algo": "blake2b-128",
    "canvas_image": loader._hash_tensor_blake2b(canvas_old),
})
result = loader.check_canvas_drift(cfg, canvas_new)
check("fingerprint mismatch = drift warning", result is not None and "DRIFT" in result)

cfg = _valid_config(bundle_fingerprint={"algo": "sha512", "canvas_image": "deadbeef"})
result = loader.check_canvas_drift(cfg, torch.zeros((1, 8, 8, 3)))
check("unknown algo = warning string",
      result is not None and "unknown" in result.lower())

print("\n=== Tightened ranges (multi-AI HIGH fix 2026-04-30) ===")

# crop_blend_feather_px: tightened to 0..64 (was 0..256). Value of 100 was
# previously accepted; should now hard-fail.
cfg = _valid_config(effective={
    "crop_params": {"crop_blend_feather_px": 100},
    "stitch_params": {},
})
expect_raises(
    "crop_blend_feather_px=100 hard-fails (tightened to 0..64)",
    loader.StitchConfigError,
    lambda: loader.apply_crop_overrides(_stub_crop_kwargs(), cfg),
)

# guided_eps: tightened to 1e-4..1e-1 (was 1e-6..10.0).
cfg = _valid_config(effective={
    "crop_params": {},
    "stitch_params": {"guided_eps": 0.5},
})
expect_raises(
    "guided_eps=0.5 hard-fails (tightened to 1e-4..1e-1)",
    loader.StitchConfigError,
    lambda: loader.apply_stitch_overrides(_stub_stitch_kwargs(), cfg),
)

# multiband_levels: tightened to 2..8 (was 2..16).
cfg = _valid_config(effective={
    "crop_params": {},
    "stitch_params": {"multiband_levels": 12},
})
expect_raises(
    "multiband_levels=12 hard-fails (tightened to 2..8)",
    loader.StitchConfigError,
    lambda: loader.apply_stitch_overrides(_stub_stitch_kwargs(), cfg),
)

# auto_preset: previously accepted any string; now validated against
# WAN_PRESETS (lazy import — fallback set is {"WAN_480p", "WAN_720p"}
# when inpaint_crop can't be imported in this isolated context).
cfg = _valid_config(effective={
    "crop_params": {"auto_preset": "WAN_GARBAGE"},
    "stitch_params": {},
})
expect_raises(
    "auto_preset='WAN_GARBAGE' hard-fails (validated against WAN_PRESETS)",
    loader.StitchConfigError,
    lambda: loader.apply_crop_overrides(_stub_crop_kwargs(), cfg),
)

# Sanity: a known preset still passes
cfg = _valid_config(effective={
    "crop_params": {"auto_preset": "WAN_480p"},
    "stitch_params": {},
})
out, _, _ = loader.apply_crop_overrides(_stub_crop_kwargs(), cfg)
check("auto_preset='WAN_480p' accepted", out["auto_preset"] == "WAN_480p")

print("\n=== Schema gate strictness (multi-AI MED fix 2026-04-30) ===")

# Previously: stitch-config-1.5 accepted via prefix. Now: explicit known
# minor list — only stitch-config-1.0.
with tempfile.TemporaryDirectory() as tmp_str:
    tmp = Path(tmp_str)
    minor = _valid_config()
    minor["schema_version"] = "stitch-config-1.5"
    p = _write_json(tmp, "minor.json", minor)
    expect_raises(
        "stitch-config-1.5 now hard-fails (explicit known minor list)",
        loader.StitchConfigError,
        lambda: loader.load_editor_config(str(p)),
    )

    base = _write_json(tmp, "base.json", _valid_config())
    loaded = loader.load_editor_config(str(base))
    check("stitch-config-1.0 still accepted", loaded["schema_version"] == "stitch-config-1.0")

print("\n=== editor_config + mask_config conflict detection (multi-AI MED fix) ===")

# detect_mask_config_conflict — None mask_config = no conflict
pre, post = loader.detect_mask_config_conflict(_valid_config(), None)
check("None mask_config -> no conflict", pre == [] and post == [])

# Empty mask_config = no conflict
pre, post = loader.detect_mask_config_conflict(_valid_config(), {})
check("Empty mask_config -> no conflict", pre == [] and post == [])

# mask_config that doesn't overlap editor's keys = no conflict
mask_cfg_safe = {"vace_input_grow_px": 80}  # not in editor's effective.crop_params
pre, post = loader.detect_mask_config_conflict(_valid_config(), mask_cfg_safe)
check("Non-overlapping mask_config -> no conflict", pre == [] and post == [])

# Overlap on a POST-diffusion key (crop_blend_feather_px is in editor's defaults)
mask_cfg_post = {"crop_blend_feather_px": 8}
pre, post = loader.detect_mask_config_conflict(_valid_config(), mask_cfg_post)
check("Post-diffusion overlap surfaced",
      pre == [] and post == ["crop_params.crop_blend_feather_px"])

# Overlap on a PRE-diffusion key (cleanup_fill_holes — must be in editor's
# effective.crop_params for the conflict to register; add it via override)
editor_cfg_pre = _valid_config(effective={
    "crop_params": {"cleanup_fill_holes": 8, "crop_blend_feather_px": 24},
    "stitch_params": {},
})
mask_cfg_pre = {"cleanup_fill_holes": 0, "crop_blend_feather_px": 16}
pre, post = loader.detect_mask_config_conflict(editor_cfg_pre, mask_cfg_pre)
check("Pre-diffusion conflict surfaced",
      pre == ["crop_params.cleanup_fill_holes"] and post == ["crop_params.crop_blend_feather_px"])

# maybe_apply_crop_config + mask_config: hard-fails on pre-diffusion conflict
with tempfile.TemporaryDirectory() as tmp_str:
    tmp = Path(tmp_str)
    p = _write_json(tmp, "cfg.json", editor_cfg_pre)
    expect_raises(
        "maybe_apply_crop_config hard-fails on pre-diffusion conflict",
        loader.StitchConfigError,
        lambda: loader.maybe_apply_crop_config(_stub_crop_kwargs(), str(p), mask_config=mask_cfg_pre),
    )

# maybe_apply_crop_config + mask_config: post-only conflict warns + proceeds
with tempfile.TemporaryDirectory() as tmp_str:
    tmp = Path(tmp_str)
    p = _write_json(tmp, "cfg.json", _valid_config())
    # Default editor cfg has crop_blend_feather_px=24; mask_config has 8.
    # Post-only overlap should not raise.
    try:
        out = loader.maybe_apply_crop_config(_stub_crop_kwargs(), str(p), mask_config={"crop_blend_feather_px": 8})
        check("post-only conflict: warn + proceed", out["crop_blend_feather_px"] == 24)
    except Exception as e:
        check("post-only conflict: warn + proceed", False, f"unexpected raise: {e}")

print("\n=== check_inpainted_drift (Step 6 sibling check) ===")

cfg = _valid_config(bundle_fingerprint=None)
result = loader.check_inpainted_drift(cfg, torch.zeros((1, 8, 8, 3)))
check("inpainted: no fingerprint = drift None", result is None)

inpainted = torch.randn((1, 8, 8, 3))
fp_hex_inp = loader._hash_tensor_blake2b(inpainted)
cfg = _valid_config(bundle_fingerprint={
    "algo": "blake2b-128",
    "inpainted_image": fp_hex_inp,
})
result = loader.check_inpainted_drift(cfg, inpainted)
check("inpainted: fingerprint match = drift None", result is None)

inp_old = torch.zeros((1, 8, 8, 3))
inp_new = torch.ones((1, 8, 8, 3))
cfg = _valid_config(bundle_fingerprint={
    "algo": "blake2b-128",
    "inpainted_image": loader._hash_tensor_blake2b(inp_old),
})
result = loader.check_inpainted_drift(cfg, inp_new)
check("inpainted: fingerprint mismatch = drift warning",
      result is not None and "DRIFT" in result and "inpainted_image" in result)

print("\n=== Drift wiring through maybe_apply_* (Step 6 wiring) ===")

# Bundle with full canvas + inpainted fingerprints, then feed CURRENT
# tensors that DIFFER. Expected: maybe_apply_* prints a drift warning to
# stdout but still returns the overridden kwargs (warning-only).
canvas_baseline = torch.randn((1, 8, 8, 3))
inp_baseline = torch.randn((1, 8, 8, 3))

cfg_with_fp = _valid_config(bundle_fingerprint={
    "algo": "blake2b-128",
    "canvas_image": loader._hash_tensor_blake2b(canvas_baseline),
    "inpainted_image": loader._hash_tensor_blake2b(inp_baseline),
})

# Submit with DIFFERENT canvas — drift should fire
canvas_drifted = torch.randn((1, 8, 8, 3))  # different content
with tempfile.TemporaryDirectory() as tmp_str:
    tmp = Path(tmp_str)
    p = _write_json(tmp, "cfg.json", cfg_with_fp)
    out = loader.maybe_apply_crop_config(
        _stub_crop_kwargs(),
        str(p),
        mask_config=None,
        canvas_image=canvas_drifted,
    )
    check("maybe_apply_crop_config returns kwargs even on drift",
          out["crop_stitch_source"] == "processed")

# Submit with MATCHING canvas — no drift warning, override still applies
with tempfile.TemporaryDirectory() as tmp_str:
    tmp = Path(tmp_str)
    p = _write_json(tmp, "cfg.json", cfg_with_fp)
    out = loader.maybe_apply_crop_config(
        _stub_crop_kwargs(),
        str(p),
        mask_config=None,
        canvas_image=canvas_baseline,
    )
    check("maybe_apply_crop_config matches fingerprint cleanly",
          out["crop_blend_feather_px"] == 24)

# Stitch side: drift on inpainted_image
inp_drifted = torch.randn((1, 8, 8, 3))
with tempfile.TemporaryDirectory() as tmp_str:
    tmp = Path(tmp_str)
    p = _write_json(tmp, "cfg.json", cfg_with_fp)
    out = loader.maybe_apply_stitch_config(
        _stub_stitch_kwargs(),
        str(p),
        inpainted_image=inp_drifted,
    )
    check("maybe_apply_stitch_config returns kwargs even on drift",
          out["blend_mode"] == "multiband")

# canvas_image=None / inpainted_image=None: drift check skipped silently
with tempfile.TemporaryDirectory() as tmp_str:
    tmp = Path(tmp_str)
    p = _write_json(tmp, "cfg.json", cfg_with_fp)
    out = loader.maybe_apply_crop_config(_stub_crop_kwargs(), str(p))
    check("maybe_apply_crop_config: None canvas_image = no error",
          out["crop_stitch_source"] == "processed")
    out = loader.maybe_apply_stitch_config(_stub_stitch_kwargs(), str(p))
    check("maybe_apply_stitch_config: None inpainted_image = no error",
          out["blend_mode"] == "multiband")

print("\n========================")
if fails:
    print(f"FAIL: {len(fails)} check(s) failed")
    for f in fails:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("PASS: all checks passed")
    sys.exit(0)
