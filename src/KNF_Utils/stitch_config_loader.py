"""Stitch config loader — applies editor-exported JSON overrides to
NV_InpaintCrop2 / NV_InpaintStitch2 widget values.

Pairs with the NV_Interactive_Masking_Suite editor's `stitch_config.py`
writer. The editor produces:

  {shot_id}_config.json:
    schema_version: "stitch-config-1.0"
    effective: { crop_params: {...}, stitch_params: {...} }   # full state
    diff:      { crop_params: {...}, stitch_params: {...} }   # human audit
    edited_keys: [...]
    pre_diffusion_keys_changed: [...]
    bundle_fingerprint: { algo, canvas_image, ..., geometry } | null
    bundle_ref: "..."
    shot_id: "..."

This module:
  - Loads + validates the JSON
  - Returns kwargs dict suitable for **kwargs splat into crop/stitch
  - Applies `effective` (deterministic replay per multi-AI synthesis)
  - Logs which keys got overridden so users can audit at runtime
  - Optionally checks the bundle_fingerprint against the live tensors and
    warns (not fails) on drift — preserves valid retunes against drifted renders

Schema versioning: this loader accepts `stitch-config-1.0`. Newer schemas
that add fields are forward-compatible (extra keys ignored). Major version
bumps must add an explicit migration.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Optional

import torch


# Schema gate (multi-AI 2026-04-30 MED): explicit known minors only.
# A future "stitch-config-1.5" that semantically redefines a key would have
# silently loaded if we matched any "stitch-config-1.*" prefix. To opt a
# new minor in, prove backward compat (additive-only changes), then add the
# string to this set.
_SUPPORTED_SCHEMAS: frozenset = frozenset({
    "stitch-config-1.0",
})


def _wan_preset_choices() -> frozenset:
    """Pull the canonical preset list from inpaint_crop. Imported lazily so
    a missing/broken inpaint_crop module doesn't crash this loader (e.g.
    when running unit tests in isolation).
    """
    try:
        from .inpaint_crop import WAN_PRESETS
        return frozenset(WAN_PRESETS.keys())
    except Exception:
        # Fallback: known presets at time of writing. Better to over-accept
        # here than reject a valid config because of a transient import error.
        return frozenset({"WAN_480p", "WAN_720p"})


# Field-level type/range guards. Hard-fail on violations so a malformed
# config can't silently produce wrong output.
#
# Ranges + choices match NV_InpaintCrop2 / NV_InpaintStitch2 INPUT_TYPES
# EXACTLY (multi-AI 2026-04-30 HIGH). Any drift here lets bad values escape
# the loader and die later as KeyError / runtime errors instead of clean
# StitchConfigError. If a node widget's range changes, update this table.
_CROP_PARAM_SPECS: dict[str, dict] = {
    "crop_stitch_source": {"type": str, "choices": {"tight", "processed", "hybrid", "bbox"}},
    "crop_blend_feather_px": {"type": int, "range": (0, 64)},
    "hybrid_falloff": {"type": int, "range": (8, 192)},
    "hybrid_curve": {"type": float, "range": (0.1, 2.0)},
    "cleanup_fill_holes": {"type": int, "range": (0, 128)},
    "cleanup_remove_noise": {"type": int, "range": (0, 32)},
    "cleanup_smooth": {"type": int, "range": (0, 127)},
    "crop_expand_px": {"type": int, "range": (-128, 128)},
    "target_mode": {"type": str, "choices": {"manual", "auto"}},
    "target_width": {"type": int, "range": (64, 16384)},
    "target_height": {"type": int, "range": (64, 16384)},
    "padding_multiple": {"type": str, "choices": {"0", "8", "16", "32", "64"}},
    "resize_algorithm": {"type": str, "choices": {"bicubic", "bilinear", "nearest", "area"}},
    "anomaly_threshold": {"type": float, "range": (0.0, 10.0)},
    # auto_preset choices resolved lazily to avoid an import cycle at module
    # load. _validate_value will read this via _wan_preset_choices() when
    # the key shows up in a config.
    "auto_preset": {"type": str, "_lazy_choices": _wan_preset_choices},
}

_STITCH_PARAM_SPECS: dict[str, dict] = {
    "blend_mode": {"type": str, "choices": {"multiband", "alpha", "hard"}},
    "multiband_levels": {"type": int, "range": (2, 8)},
    "guided_refine": {"type": bool},
    "guided_radius": {"type": int, "range": (1, 64)},
    "guided_eps": {"type": float, "range": (1e-4, 1e-1)},
    "guided_strength": {"type": float, "range": (0.0, 1.0)},
    "output_dtype": {"type": str, "choices": {"fp16", "fp32"}},
}


# =============================================================================
# Validation
# =============================================================================

class StitchConfigError(ValueError):
    """Raised when a config file is malformed, schema-incompatible, or has
    out-of-range values. Triggers a hard-fail in the loading node — better
    than silent wrong output.
    """


def _validate_value(category: str, key: str, value: Any, specs: dict) -> Any:
    """Coerce numeric types (int/float interchangeable for numeric values)
    and validate against spec. Returns the validated value.
    """
    if key not in specs:
        # Unknown key — soft-warn at the caller level (don't reject configs
        # from a future editor version that adds new params).
        return value

    spec = specs[key]
    expected_type = spec["type"]

    if expected_type is bool:
        if not isinstance(value, bool):
            raise StitchConfigError(
                f"{category}.{key}: expected bool, got {type(value).__name__}={value!r}"
            )
        return value

    if expected_type is int:
        # Accept floats with integer values (json round-trips can produce 5.0)
        if isinstance(value, bool):
            raise StitchConfigError(f"{category}.{key}: bool not allowed for int field")
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        if not isinstance(value, int):
            raise StitchConfigError(
                f"{category}.{key}: expected int, got {type(value).__name__}={value!r}"
            )

    if expected_type is float:
        if isinstance(value, bool):
            raise StitchConfigError(f"{category}.{key}: bool not allowed for float field")
        if isinstance(value, int):
            value = float(value)
        if not isinstance(value, float):
            raise StitchConfigError(
                f"{category}.{key}: expected float, got {type(value).__name__}={value!r}"
            )

    if expected_type is str and not isinstance(value, str):
        raise StitchConfigError(
            f"{category}.{key}: expected str, got {type(value).__name__}={value!r}"
        )

    if "range" in spec:
        lo, hi = spec["range"]
        if not (lo <= value <= hi):
            raise StitchConfigError(
                f"{category}.{key}: value {value} outside range [{lo}, {hi}]"
            )
    if "choices" in spec:
        if value not in spec["choices"]:
            raise StitchConfigError(
                f"{category}.{key}: value {value!r} not in {sorted(spec['choices'])}"
            )
    if "_lazy_choices" in spec:
        choices = spec["_lazy_choices"]()
        if value not in choices:
            raise StitchConfigError(
                f"{category}.{key}: value {value!r} not in {sorted(choices)}"
            )

    return value


# =============================================================================
# Loader
# =============================================================================

def load_editor_config(path: str) -> dict:
    """Load + schema-validate a stitch-config JSON. Raises StitchConfigError
    on malformed input.

    Returns the parsed dict (with `effective` + `diff` etc. as written by
    the editor). Range/type checks are deferred to `apply_*_overrides` so
    callers can apply category by category.
    """
    p = Path(path)
    if not p.exists():
        raise StitchConfigError(f"editor_config_path not found: {path}")

    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise StitchConfigError(f"editor_config_path is not valid JSON: {e}") from e

    if not isinstance(data, dict):
        raise StitchConfigError(f"editor_config_path top-level is not a dict: {type(data).__name__}")

    schema = data.get("schema_version", "")
    if not isinstance(schema, str) or schema not in _SUPPORTED_SCHEMAS:
        raise StitchConfigError(
            f"editor_config_path has unsupported schema_version={schema!r}; "
            f"expected one of {sorted(_SUPPORTED_SCHEMAS)}"
        )

    effective = data.get("effective", {})
    if not isinstance(effective, dict):
        raise StitchConfigError("editor_config_path 'effective' is not a dict")

    return data


# =============================================================================
# Override application
# =============================================================================

def apply_crop_overrides(kwargs: dict, config: dict) -> tuple[dict, list[str], list[str]]:
    """Return (overridden_kwargs, applied_keys, unknown_keys).

    `kwargs` is the bag of widget-derived values flowing into NV_InpaintCrop2.crop().
    Only keys present in the config's `effective.crop_params` are overridden.
    """
    effective = config.get("effective", {})
    crop_eff = effective.get("crop_params", {})
    if not isinstance(crop_eff, dict):
        raise StitchConfigError("config.effective.crop_params is not a dict")

    out = dict(kwargs)
    applied: list[str] = []
    unknown: list[str] = []

    for k, raw_v in crop_eff.items():
        if k not in _CROP_PARAM_SPECS:
            # Not a known crop_params key — could be from a newer editor
            # schema. Track for warning but skip override.
            unknown.append(f"crop_params.{k}")
            continue
        if k not in out:
            # Editor wrote a key the node's signature doesn't expect — skip.
            unknown.append(f"crop_params.{k}(no-widget)")
            continue
        validated = _validate_value("crop_params", k, raw_v, _CROP_PARAM_SPECS)
        out[k] = validated
        applied.append(f"crop_params.{k}={validated}")

    return out, applied, unknown


def apply_stitch_overrides(kwargs: dict, config: dict) -> tuple[dict, list[str], list[str]]:
    """Same shape as apply_crop_overrides but for stitch_params."""
    effective = config.get("effective", {})
    stitch_eff = effective.get("stitch_params", {})
    if not isinstance(stitch_eff, dict):
        raise StitchConfigError("config.effective.stitch_params is not a dict")

    out = dict(kwargs)
    applied: list[str] = []
    unknown: list[str] = []

    for k, raw_v in stitch_eff.items():
        if k not in _STITCH_PARAM_SPECS:
            unknown.append(f"stitch_params.{k}")
            continue
        if k not in out:
            unknown.append(f"stitch_params.{k}(no-widget)")
            continue
        validated = _validate_value("stitch_params", k, raw_v, _STITCH_PARAM_SPECS)
        out[k] = validated
        applied.append(f"stitch_params.{k}={validated}")

    return out, applied, unknown


# =============================================================================
# Drift check (optional, warning-only)
# =============================================================================

def _hash_tensor_blake2b(t: torch.Tensor, digest_size: int = 16) -> str:
    """Mirror NV_FixtureDumper._hash_tensor — used for drift comparisons.
    Mode and bytes must match exactly or every bundle reads as drifted.
    """
    h = hashlib.blake2b(digest_size=digest_size)
    h.update(str(t.dtype).encode("utf-8"))
    h.update(b"|")
    h.update(repr(tuple(t.shape)).encode("utf-8"))
    h.update(b"|")
    h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _check_tensor_drift(
    config: dict,
    tensor: torch.Tensor,
    fingerprint_key: str,
    label: str,
) -> Optional[str]:
    """Generic drift check: hash `tensor`, compare against config's
    `bundle_fingerprint[fingerprint_key]`. Returns a warning string on
    mismatch (or unknown algo), None on match / skip.

    Skip cases (returns None):
      - config has no `bundle_fingerprint` (pre-0.3.0 bundles)
      - fingerprint is missing the requested key
      - bundle_fingerprint is not a dict
    """
    fp = config.get("bundle_fingerprint")
    if not isinstance(fp, dict):
        return None
    if fp.get("algo") != "blake2b-128":
        return f"unknown bundle_fingerprint.algo={fp.get('algo')!r} — drift check skipped"
    expected = fp.get(fingerprint_key)
    if not isinstance(expected, str):
        return None
    actual = _hash_tensor_blake2b(tensor)
    if actual != expected:
        return (
            f"BUNDLE DRIFT: {label} fingerprint differs from editor config "
            f"({actual[:8]}... vs expected {expected[:8]}...). "
            f"Editor preview may not match production output."
        )
    return None


def check_canvas_drift(config: dict, canvas_image: torch.Tensor) -> Optional[str]:
    """Compare current canvas tensor against the config's
    `bundle_fingerprint.canvas_image`. Warning-only (multi-AI debate
    consensus): a mismatch can be intentional (user retuning against an
    updated render) so we never hard-fail.
    """
    return _check_tensor_drift(config, canvas_image, "canvas_image", "canvas_image")


def check_inpainted_drift(config: dict, inpainted_image: torch.Tensor) -> Optional[str]:
    """Like check_canvas_drift but for the KSampler output (inpainted patch).

    This is the more diagnostic check: a stale `inpainted_image` typically
    means upstream prompt/seed/model drift — exactly the case where the
    editor's preview cannot reflect production output. The `canvas_image`
    is more often genuinely stable across re-renders.
    """
    return _check_tensor_drift(config, inpainted_image, "inpainted_image", "inpainted_image")


# =============================================================================
# Public helper for nodes (load + apply both categories)
# =============================================================================

# Pre-diffusion crop_params keys — must match the editor's stitch_config.py
# PRE_DIFFUSION_CROP_KEYS. Conflict on these requires a hard-fail because the
# editor's preview is invalid relative to production output if mask_config
# silently overrides them.
_PRE_DIFFUSION_CROP_KEYS: frozenset = frozenset({
    "cleanup_fill_holes",
    "cleanup_remove_noise",
    "cleanup_smooth",
    "crop_expand_px",
})


def detect_mask_config_conflict(
    editor_config: Optional[dict],
    mask_config: Optional[dict],
) -> tuple[list[str], list[str]]:
    """Return (pre_diffusion_conflicts, post_diffusion_conflicts) — fully
    qualified key names ('crop_params.cleanup_fill_holes', ...).

    Either argument None → no conflict possible. Both present → intersect
    the editor's effective.crop_params keys with mask_config keys, then
    bucket by pre/post diffusion classification.
    """
    if not editor_config or not isinstance(mask_config, dict) or not mask_config:
        return [], []

    eff = editor_config.get("effective", {})
    crop_eff = eff.get("crop_params", {})
    if not isinstance(crop_eff, dict):
        return [], []

    overlap = set(crop_eff.keys()) & set(mask_config.keys())
    pre, post = [], []
    for k in overlap:
        full = f"crop_params.{k}"
        if k in _PRE_DIFFUSION_CROP_KEYS:
            pre.append(full)
        else:
            post.append(full)
    return sorted(pre), sorted(post)


def maybe_apply_crop_config(
    kwargs: dict,
    editor_config_path: str,
    mask_config: Optional[dict] = None,
    canvas_image: Optional[torch.Tensor] = None,
) -> dict:
    """High-level convenience for NV_InpaintCrop2.crop().

    Empty/None path → returns kwargs unchanged.
    Valid config → returns updated kwargs, prints applied + unknown lists.
    Hard-fails on malformed config or out-of-range values.

    When `mask_config` is also passed (non-None), detects overlap between
    the two override mechanisms and hard-fails on PRE-diffusion conflicts
    (multi-AI 2026-04-30 MED). Post-diffusion overlaps log a warning but
    proceed — the user has chosen to layer config buses over the editor.

    When `canvas_image` is passed, runs `check_canvas_drift()` and prints
    a warning if the live canvas tensor doesn't match the config's
    fingerprint. Warning-only — never blocks the render.
    """
    if not editor_config_path or not editor_config_path.strip():
        return kwargs
    config = load_editor_config(editor_config_path.strip())

    # Conflict detection BEFORE applying overrides — fail fast with a clear
    # message instead of producing the wrong output silently.
    pre_conflicts, post_conflicts = detect_mask_config_conflict(config, mask_config)
    if pre_conflicts:
        raise StitchConfigError(
            f"editor_config_path and mask_config both override pre-diffusion "
            f"key(s): {pre_conflicts}. mask_config silently wins on these "
            f"in NV_InpaintCrop2 — leaving both wired produces preview/runtime "
            f"divergence (the editor preview is a 'lie'). "
            f"Disconnect mask_config OR remove these keys from the editor "
            f"config / mask_config bus."
        )
    if post_conflicts:
        print(
            f"[NV_InpaintCrop2] WARNING: editor_config_path and mask_config "
            f"both set {post_conflicts}; mask_config will win for these keys. "
            f"Editor preview reflects the editor_config values; production "
            f"will use mask_config."
        )

    # Drift preflight (warning-only). Skipped silently for pre-0.3.0 bundles.
    if canvas_image is not None:
        warning = check_canvas_drift(config, canvas_image)
        if warning:
            print(f"[NV_InpaintCrop2] {warning}")

    out, applied, unknown = apply_crop_overrides(kwargs, config)
    _log_overrides(editor_config_path, "NV_InpaintCrop2", applied, unknown, config)
    return out


def maybe_apply_stitch_config(
    kwargs: dict,
    editor_config_path: str,
    inpainted_image: Optional[torch.Tensor] = None,
) -> dict:
    """High-level convenience for NV_InpaintStitch2.stitch().

    When `inpainted_image` is passed, runs `check_inpainted_drift()` and
    prints a warning to stdout if the live tensor's fingerprint doesn't
    match the config's recorded value. Warning-only — never blocks the
    render.
    """
    if not editor_config_path or not editor_config_path.strip():
        return kwargs
    config = load_editor_config(editor_config_path.strip())

    # Drift check (warning-only). Run BEFORE apply so the audit log shows
    # both messages together if both fire.
    if inpainted_image is not None:
        warning = check_inpainted_drift(config, inpainted_image)
        if warning:
            print(f"[NV_InpaintStitch2] {warning}")

    out, applied, unknown = apply_stitch_overrides(kwargs, config)
    _log_overrides(editor_config_path, "NV_InpaintStitch2", applied, unknown, config)
    return out


def _log_overrides(path: str, node_name: str, applied: list[str], unknown: list[str], config: dict) -> None:
    """Stdout log so users can audit which keys took effect on each run.
    Each line is prefixed with the node name + config filename for grep-ability.
    """
    cfg_name = os.path.basename(path)
    if applied:
        print(f"[{node_name}] editor_config={cfg_name}: applied {len(applied)} override(s):")
        for a in applied:
            print(f"  • {a}")
    if unknown:
        print(f"[{node_name}] editor_config={cfg_name}: ignored {len(unknown)} unknown key(s): {unknown}")
    pre_diff = config.get("pre_diffusion_keys_changed", [])
    if pre_diff:
        print(
            f"[{node_name}] editor_config={cfg_name}: WARNING — config edits "
            f"{len(pre_diff)} pre-diffusion param(s): {pre_diff}. "
            f"This requires a fresh KSampler render — old captured "
            f"inpainted_image won't match."
        )
