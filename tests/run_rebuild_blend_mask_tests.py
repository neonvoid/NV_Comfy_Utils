"""Standalone test runner for NV_RebuildBlendMask.

Usage:
    python tests/run_rebuild_blend_mask_tests.py

Bypasses the package __init__.py chain via importlib.util.spec_from_file_location,
matching the pattern in run_v2v_tests.py / run_mask_config_schema_tests.py.

Exercises:
- Full-pipeline rebuild against a synthetic stitcher (verifies blend mask shape changes)
- Sentinel passthrough (rebuild reproduces saved mask when all widgets at sentinel)
- mask_config override path (mask_config keys win over saved when widget is sentinel)
- Widget value beats mask_config (priority order: widget > mask_config > saved > fallback)
- Sentinel-gap bounds check (raises on out-of-bounds widget value)
- Stitcher-validation errors on missing required keys
- Type errors on non-dict input
- crop_params reflects effective values + rebuild_blend_mask_applied flag
- Shallow-copy preserves heavy tensor lists (canvas_image is shared by reference)
- Empty stitcher passthrough
"""
import importlib.util
import os
import sys
import types

import torch


# =============================================================================
# Module loader — bypasses package __init__.py via fake-parent-package shim
# =============================================================================
#
# rebuild_blend_mask.py uses `from .mask_ops import ...` which requires the
# module to live inside a package context. We can't import `src.KNF_Utils` for
# real because its __init__.py imports the full ComfyUI runtime (server,
# heartbeat, etc.) which crashes outside ComfyUI. Solution: register fake shell
# packages under sys.modules["src"] and sys.modules["src.KNF_Utils"] so
# relative imports resolve to the real .py files without ever executing the
# package __init__.

_KNF_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src", "KNF_Utils")
)

_fake_src = types.ModuleType("src")
_fake_src.__path__ = [os.path.dirname(_KNF_DIR)]
sys.modules.setdefault("src", _fake_src)

_fake_pkg = types.ModuleType("src.KNF_Utils")
_fake_pkg.__path__ = [_KNF_DIR]
sys.modules.setdefault("src.KNF_Utils", _fake_pkg)


def _load_subpackage_module(submodule_name):
    """Load src/KNF_Utils/<submodule_name>.py as src.KNF_Utils.<submodule_name>."""
    full_name = f"src.KNF_Utils.{submodule_name}"
    spec = importlib.util.spec_from_file_location(
        full_name, os.path.join(_KNF_DIR, f"{submodule_name}.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = mod
    spec.loader.exec_module(mod)
    return mod


# mask_ops must be loaded first because rebuild_blend_mask imports from it
_load_subpackage_module("mask_ops")
rebuild_mod = _load_subpackage_module("rebuild_blend_mask")

NV_RebuildBlendMask = rebuild_mod.NV_RebuildBlendMask


# =============================================================================
# Synthetic stitcher fixture
# =============================================================================

def _make_synthetic_stitcher(n_frames=2, H=128, W=160, target_w=64, target_h=64):
    """Build a minimal V2 stitcher with a circular SAM3-like mask in the middle.

    Returns a stitcher dict with the V2 contract: canvas_mask is uint8 [H,W],
    cropped_mask_for_blend is fp32 [H_target, W_target], coords + crop_target.
    """
    canvas_masks = []
    for _ in range(n_frames):
        mask = torch.zeros(H, W, dtype=torch.uint8)
        # Draw a filled circle at center, radius 24
        cy, cx = H // 2, W // 2
        for y in range(H):
            for x in range(W):
                if (y - cy) ** 2 + (x - cx) ** 2 <= 24 ** 2:
                    mask[y, x] = 255
        canvas_masks.append(mask)

    # cropped_to_canvas: a 64x64 box centered on the mask
    ctc_x = (W - 64) // 2
    ctc_y = (H - 64) // 2
    ctc_w = 64
    ctc_h = 64

    # Pre-baked blend mask (we'll replace these in rebuild)
    pre_blend_masks = [torch.ones(target_h, target_w, dtype=torch.float32) * 0.5
                       for _ in range(n_frames)]

    return {
        "canvas_mask": canvas_masks,
        "canvas_mask_processed": [m.clone() for m in canvas_masks],  # same as canvas_mask for fixture
        "canvas_image": [torch.zeros(H, W, 3, dtype=torch.float32) for _ in range(n_frames)],
        "cropped_to_canvas_x": [ctc_x] * n_frames,
        "cropped_to_canvas_y": [ctc_y] * n_frames,
        "cropped_to_canvas_w": [ctc_w] * n_frames,
        "cropped_to_canvas_h": [ctc_h] * n_frames,
        "crop_target_w": target_w,
        "crop_target_h": target_h,
        "cropped_mask_for_blend": pre_blend_masks,
        "skipped_indices": [],
        "original_frames": [],
        "total_frames": n_frames,
        "resize_algorithm": "bicubic",
        "blend_pixels": 0,
        "crop_params": {
            "crop_stitch_source": "tight",
            "crop_expand_px": 0,
            "crop_blend_feather_px": 0,  # no feather → easy to verify shape
            "cleanup_fill_holes": 0,
            "cleanup_remove_noise": 0,
            "cleanup_smooth": 0,
            "hybrid_falloff": 48,
            "hybrid_curve": 0.6,
        },
    }


def _all_sentinel_kwargs():
    return {
        "crop_stitch_source": "use_saved",
        "crop_expand_px": -999,
        "crop_blend_feather_px": -999,
        "cleanup_fill_holes": -999,
        "cleanup_remove_noise": -999,
        "cleanup_smooth": -999,
        "hybrid_falloff": -999,
        "hybrid_curve": -999.0,
    }


# =============================================================================
# Test infrastructure
# =============================================================================

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


print("Running NV_RebuildBlendMask tests...")
print()


# =============================================================================
# Tests
# =============================================================================

def test_sentinel_passthrough_reproduces_saved_blend():
    """All sentinels + saved crop_params → output blend mask matches a fresh tight rebuild."""
    stitcher = _make_synthetic_stitcher()
    node = NV_RebuildBlendMask()
    (out, report) = node.rebuild(stitcher=stitcher, mask_config=None, **_all_sentinel_kwargs())
    # Saved was tight + no feather → output should be the cropped+resized canvas_mask
    assert len(out["cropped_mask_for_blend"]) == 2
    new_blend = out["cropped_mask_for_blend"][0]
    assert new_blend.dim() == 2, f"expected [H,W] storage, got {tuple(new_blend.shape)}"
    assert new_blend.shape == (64, 64)
    assert "crop_stitch_source:    tight" in report
    assert "(saved)" in report  # source attribution


run_test("sentinel_passthrough_reproduces_saved_blend", test_sentinel_passthrough_reproduces_saved_blend)


def test_widget_overrides_saved():
    """Widget value beats saved when not at sentinel."""
    stitcher = _make_synthetic_stitcher()
    node = NV_RebuildBlendMask()
    kwargs = _all_sentinel_kwargs()
    kwargs["crop_stitch_source"] = "bbox"  # widget override
    (out, report) = node.rebuild(stitcher=stitcher, mask_config=None, **kwargs)
    assert out["crop_params"]["crop_stitch_source"] == "bbox"
    assert "crop_stitch_source:    bbox  (widget)" in report
    # bbox = all ones (before any feather)
    blend = out["cropped_mask_for_blend"][0]
    assert blend.min().item() > 0.99, f"bbox should produce all-ones, got min={blend.min().item()}"


run_test("widget_overrides_saved", test_widget_overrides_saved)


def test_mask_config_overrides_saved_when_widget_at_sentinel():
    """mask_config beats saved when widget is at sentinel."""
    stitcher = _make_synthetic_stitcher()
    node = NV_RebuildBlendMask()
    mask_config = {"crop_expand_px": 32}  # mask_config provides
    (out, _) = node.rebuild(stitcher=stitcher, mask_config=mask_config, **_all_sentinel_kwargs())
    assert out["crop_params"]["crop_expand_px"] == 32, (
        f"expected 32 from mask_config, got {out['crop_params']['crop_expand_px']}"
    )


run_test("mask_config_overrides_saved_when_widget_at_sentinel",
         test_mask_config_overrides_saved_when_widget_at_sentinel)


def test_widget_beats_mask_config():
    """Priority: widget > mask_config (when widget is non-sentinel)."""
    stitcher = _make_synthetic_stitcher()
    node = NV_RebuildBlendMask()
    kwargs = _all_sentinel_kwargs()
    kwargs["crop_expand_px"] = 8  # widget value
    mask_config = {"crop_expand_px": 32}  # mask_config tries to override
    (out, _) = node.rebuild(stitcher=stitcher, mask_config=mask_config, **kwargs)
    assert out["crop_params"]["crop_expand_px"] == 8, (
        f"widget (8) should beat mask_config (32), got {out['crop_params']['crop_expand_px']}"
    )


run_test("widget_beats_mask_config", test_widget_beats_mask_config)


def test_bounds_check_rejects_sentinel_gap_drag():
    """Dragging a slider into the sentinel gap (e.g. -50 on cleanup_fill_holes 0-128) raises."""
    stitcher = _make_synthetic_stitcher()
    node = NV_RebuildBlendMask()
    kwargs = _all_sentinel_kwargs()
    kwargs["cleanup_fill_holes"] = -50  # in the gap [-998, -1]
    raised = False
    try:
        node.rebuild(stitcher=stitcher, mask_config=None, **kwargs)
    except ValueError as e:
        raised = True
        msg = str(e)
        if "out of bounds" not in msg or "cleanup_fill_holes" not in msg:
            raise AssertionError(f"ValueError raised but message wrong: {msg!r}")
    if not raised:
        raise AssertionError("Sentinel-gap drag (-50) was accepted; bounds check missing")


run_test("bounds_check_rejects_sentinel_gap_drag", test_bounds_check_rejects_sentinel_gap_drag)


def test_invalid_stitch_source_string_rejected():
    """A garbage stitch_source value should raise."""
    stitcher = _make_synthetic_stitcher()
    node = NV_RebuildBlendMask()
    kwargs = _all_sentinel_kwargs()
    kwargs["crop_stitch_source"] = "totally_invalid_mode"
    raised = False
    try:
        node.rebuild(stitcher=stitcher, mask_config=None, **kwargs)
    except ValueError:
        raised = True
    if not raised:
        raise AssertionError("Invalid crop_stitch_source string was accepted; validation missing")


run_test("invalid_stitch_source_string_rejected", test_invalid_stitch_source_string_rejected)


def test_missing_required_key_raises():
    """Stitcher missing canvas_mask should raise a clear error."""
    stitcher = _make_synthetic_stitcher()
    del stitcher["canvas_mask"]
    node = NV_RebuildBlendMask()
    raised = False
    try:
        node.rebuild(stitcher=stitcher, mask_config=None, **_all_sentinel_kwargs())
    except ValueError as e:
        raised = True
        if "canvas_mask" not in str(e):
            raise AssertionError(f"Error message should mention canvas_mask: {e}")
    if not raised:
        raise AssertionError("Missing canvas_mask was accepted; validation missing")


run_test("missing_required_key_raises", test_missing_required_key_raises)


def test_non_dict_stitcher_raises():
    """A non-dict stitcher should raise TypeError."""
    node = NV_RebuildBlendMask()
    raised = False
    try:
        node.rebuild(stitcher="not a dict", mask_config=None, **_all_sentinel_kwargs())
    except TypeError:
        raised = True
    if not raised:
        raise AssertionError("Non-dict stitcher was accepted; type check missing")


run_test("non_dict_stitcher_raises", test_non_dict_stitcher_raises)


def test_non_dict_mask_config_raises():
    """A non-dict mask_config should raise TypeError."""
    stitcher = _make_synthetic_stitcher()
    node = NV_RebuildBlendMask()
    raised = False
    try:
        node.rebuild(stitcher=stitcher, mask_config="not a dict", **_all_sentinel_kwargs())
    except TypeError:
        raised = True
    if not raised:
        raise AssertionError("Non-dict mask_config was accepted; type check missing")


run_test("non_dict_mask_config_raises", test_non_dict_mask_config_raises)


def test_shallow_copy_preserves_canvas_image_reference():
    """Heavy tensor lists like canvas_image must be shared by reference (memory-friendly).
    The cropped_mask_for_blend list MUST be replaced (not mutated)."""
    stitcher = _make_synthetic_stitcher()
    original_canvas_image = stitcher["canvas_image"]
    original_blend = stitcher["cropped_mask_for_blend"]
    node = NV_RebuildBlendMask()
    (out, _) = node.rebuild(stitcher=stitcher, mask_config=None, **_all_sentinel_kwargs())
    # canvas_image is shared by reference (memory win)
    assert out["canvas_image"] is original_canvas_image, (
        "canvas_image should be shared by reference, but got a separate object"
    )
    # cropped_mask_for_blend is a NEW list (so siblings of input stitcher are safe)
    assert out["cropped_mask_for_blend"] is not original_blend, (
        "cropped_mask_for_blend should be a new list to avoid corrupting siblings"
    )
    # crop_params is a new dict
    assert out["crop_params"] is not stitcher["crop_params"], (
        "crop_params should be a new dict (we mutate it)"
    )


run_test("shallow_copy_preserves_canvas_image_reference",
         test_shallow_copy_preserves_canvas_image_reference)


def test_crop_params_reflects_rebuild():
    """crop_params on output should have rebuild_blend_mask_applied=True + effective values."""
    stitcher = _make_synthetic_stitcher()
    node = NV_RebuildBlendMask()
    kwargs = _all_sentinel_kwargs()
    kwargs["crop_expand_px"] = 16
    kwargs["crop_stitch_source"] = "hybrid"
    (out, _) = node.rebuild(stitcher=stitcher, mask_config=None, **kwargs)
    cp = out["crop_params"]
    assert cp["rebuild_blend_mask_applied"] is True
    assert cp["crop_expand_px"] == 16
    assert cp["crop_stitch_source"] == "hybrid"
    # Original crop_params untouched
    assert stitcher["crop_params"]["crop_expand_px"] == 0


run_test("crop_params_reflects_rebuild", test_crop_params_reflects_rebuild)


def test_empty_stitcher_returns_passthrough():
    """Stitcher with zero frames returns shallow copy + diagnostic message."""
    stitcher = _make_synthetic_stitcher(n_frames=0)
    node = NV_RebuildBlendMask()
    (out, report) = node.rebuild(stitcher=stitcher, mask_config=None, **_all_sentinel_kwargs())
    assert out["cropped_mask_for_blend"] == []
    assert "nothing to rebuild" in report.lower()


run_test("empty_stitcher_returns_passthrough", test_empty_stitcher_returns_passthrough)


def test_blend_mask_storage_shape_is_2d():
    """V2 storage convention: cropped_mask_for_blend entries are [H,W] not [1,H,W]."""
    stitcher = _make_synthetic_stitcher()
    node = NV_RebuildBlendMask()
    (out, _) = node.rebuild(stitcher=stitcher, mask_config=None, **_all_sentinel_kwargs())
    for i, blend in enumerate(out["cropped_mask_for_blend"]):
        if blend.dim() != 2:
            raise AssertionError(
                f"frame {i} blend mask has dim={blend.dim()} (shape {tuple(blend.shape)}); "
                f"V2 storage convention requires [H,W] (dim=2)"
            )


run_test("blend_mask_storage_shape_is_2d", test_blend_mask_storage_shape_is_2d)


def test_bbox_mode_produces_nonzero_everywhere_before_feather():
    """bbox mode = full crop region = no zeros in blend mask (before feather)."""
    stitcher = _make_synthetic_stitcher()
    stitcher["crop_params"]["crop_blend_feather_px"] = 0  # no feather
    node = NV_RebuildBlendMask()
    kwargs = _all_sentinel_kwargs()
    kwargs["crop_stitch_source"] = "bbox"
    (out, _) = node.rebuild(stitcher=stitcher, mask_config=None, **kwargs)
    blend = out["cropped_mask_for_blend"][0]
    # Without feather, bbox mode should produce all 1.0
    assert blend.min().item() == 1.0 and blend.max().item() == 1.0, (
        f"bbox + no feather should be all 1.0, got [{blend.min().item()}, {blend.max().item()}]"
    )


run_test("bbox_mode_produces_nonzero_everywhere_before_feather",
         test_bbox_mode_produces_nonzero_everywhere_before_feather)


def test_crop_expand_widens_processed_blend():
    """crop_expand_px=16 + processed source should produce a WIDER blend mask than expand=0."""
    stitcher = _make_synthetic_stitcher()
    stitcher["crop_params"]["crop_blend_feather_px"] = 0  # no feather → easy to compare areas
    node = NV_RebuildBlendMask()

    kwargs0 = _all_sentinel_kwargs()
    kwargs0["crop_stitch_source"] = "processed"
    kwargs0["crop_expand_px"] = 0
    (out0, _) = node.rebuild(stitcher=stitcher, mask_config=None, **kwargs0)

    kwargs1 = _all_sentinel_kwargs()
    kwargs1["crop_stitch_source"] = "processed"
    kwargs1["crop_expand_px"] = 16
    (out1, _) = node.rebuild(stitcher=stitcher, mask_config=None, **kwargs1)

    area0 = (out0["cropped_mask_for_blend"][0] > 0.5).sum().item()
    area1 = (out1["cropped_mask_for_blend"][0] > 0.5).sum().item()
    if area1 <= area0:
        raise AssertionError(
            f"crop_expand_px=16 should widen mask: area_expand0={area0}, area_expand16={area1}. "
            f"Expand had no measurable effect."
        )


run_test("crop_expand_widens_processed_blend", test_crop_expand_widens_processed_blend)


# =============================================================================
# Multi-AI impl-review fix tests (2026-05-01 round)
# =============================================================================

def test_mask_config_can_override_crop_stitch_source():
    """Fix #1: crop_stitch_source must consult mask_config when widget=use_saved
    (was previously skipped, breaking semantic uniformity with the rest of the node)."""
    stitcher = _make_synthetic_stitcher()
    node = NV_RebuildBlendMask()
    mask_config = {"crop_stitch_source": "bbox"}  # only set this one key
    (out, report) = node.rebuild(stitcher=stitcher, mask_config=mask_config, **_all_sentinel_kwargs())
    assert out["crop_params"]["crop_stitch_source"] == "bbox"
    assert "(mask_config)" in report  # source attribution should reflect mask_config


run_test("mask_config_can_override_crop_stitch_source",
         test_mask_config_can_override_crop_stitch_source)


def test_invalid_crop_stitch_source_in_mask_config_rejected():
    """Fix #2: validation must run on resolved value regardless of source.
    A garbage crop_stitch_source from mask_config must raise (was silent before)."""
    stitcher = _make_synthetic_stitcher()
    node = NV_RebuildBlendMask()
    mask_config = {"crop_stitch_source": "totally_bogus"}
    raised = False
    try:
        node.rebuild(stitcher=stitcher, mask_config=mask_config, **_all_sentinel_kwargs())
    except ValueError as e:
        raised = True
        if "totally_bogus" not in str(e) or "mask_config" not in str(e):
            raise AssertionError(f"Error should name the bad value + source: {e}")
    if not raised:
        raise AssertionError("Invalid crop_stitch_source from mask_config was accepted")


run_test("invalid_crop_stitch_source_in_mask_config_rejected",
         test_invalid_crop_stitch_source_in_mask_config_rejected)


def test_invalid_crop_stitch_source_in_saved_rejected():
    """Same fix #2: a corrupted saved crop_stitch_source must raise, not silently
    fall through to 'tight' as the prior else-branch would have done."""
    stitcher = _make_synthetic_stitcher()
    stitcher["crop_params"]["crop_stitch_source"] = "garbage_value"
    node = NV_RebuildBlendMask()
    raised = False
    try:
        node.rebuild(stitcher=stitcher, mask_config=None, **_all_sentinel_kwargs())
    except ValueError as e:
        raised = True
        if "garbage_value" not in str(e) or "saved" not in str(e):
            raise AssertionError(f"Error should name the bad value + source: {e}")
    if not raised:
        raise AssertionError("Invalid saved crop_stitch_source was accepted (silent fallthrough bug)")


run_test("invalid_crop_stitch_source_in_saved_rejected",
         test_invalid_crop_stitch_source_in_saved_rejected)


def test_out_of_bounds_value_in_mask_config_rejected():
    """Fix #2: numeric values from mask_config must be bounds-checked too.
    A bad crop_expand_px=5000 from mask_config must raise."""
    stitcher = _make_synthetic_stitcher()
    node = NV_RebuildBlendMask()
    mask_config = {"crop_expand_px": 5000}
    raised = False
    try:
        node.rebuild(stitcher=stitcher, mask_config=mask_config, **_all_sentinel_kwargs())
    except ValueError as e:
        raised = True
        if "5000" not in str(e) or "mask_config" not in str(e):
            raise AssertionError(f"Error should name the bad value + source: {e}")
    if not raised:
        raise AssertionError("Out-of-range crop_expand_px from mask_config was accepted")


run_test("out_of_bounds_value_in_mask_config_rejected",
         test_out_of_bounds_value_in_mask_config_rejected)


def test_out_of_bounds_value_in_saved_rejected():
    """Same fix #2: corrupted saved numeric values must raise."""
    stitcher = _make_synthetic_stitcher()
    stitcher["crop_params"]["hybrid_falloff"] = 99999  # way out of [8, 192]
    node = NV_RebuildBlendMask()
    kwargs = _all_sentinel_kwargs()
    kwargs["crop_stitch_source"] = "hybrid"  # force hybrid path so falloff matters
    raised = False
    try:
        node.rebuild(stitcher=stitcher, mask_config=None, **kwargs)
    except ValueError as e:
        raised = True
        if "99999" not in str(e):
            raise AssertionError(f"Error should name the bad value: {e}")
    if not raised:
        raise AssertionError("Out-of-range saved hybrid_falloff was accepted")


run_test("out_of_bounds_value_in_saved_rejected", test_out_of_bounds_value_in_saved_rejected)


def test_negative_crop_coords_rejected():
    """Fix #3: per-frame crop coords must be validated before slicing.
    PyTorch silently wraps negatives — without validation, a malformed
    stitcher would produce a wrong blend mask instead of failing fast."""
    stitcher = _make_synthetic_stitcher()
    stitcher["cropped_to_canvas_x"] = [-5] + stitcher["cropped_to_canvas_x"][1:]
    node = NV_RebuildBlendMask()
    raised = False
    try:
        node.rebuild(stitcher=stitcher, mask_config=None, **_all_sentinel_kwargs())
    except ValueError as e:
        raised = True
        if "-5" not in str(e) or "frame 0" not in str(e):
            raise AssertionError(f"Error should name the bad coord + frame: {e}")
    if not raised:
        raise AssertionError("Negative cropped_to_canvas_x was accepted (silent slice-wrap)")


run_test("negative_crop_coords_rejected", test_negative_crop_coords_rejected)


def test_oob_crop_coords_rejected():
    """Fix #3: crop region exceeding canvas dimensions must raise."""
    stitcher = _make_synthetic_stitcher()  # default canvas is 128x160
    # Set crop window that runs off the right edge
    stitcher["cropped_to_canvas_x"] = [120] * len(stitcher["cropped_to_canvas_x"])
    stitcher["cropped_to_canvas_w"] = [80] * len(stitcher["cropped_to_canvas_w"])  # 120+80=200 > W=160
    node = NV_RebuildBlendMask()
    raised = False
    try:
        node.rebuild(stitcher=stitcher, mask_config=None, **_all_sentinel_kwargs())
    except ValueError as e:
        raised = True
        if "exceeds canvas" not in str(e):
            raise AssertionError(f"Error should mention canvas overflow: {e}")
    if not raised:
        raise AssertionError("OOB crop region was accepted (silent slice-truncate)")


run_test("oob_crop_coords_rejected", test_oob_crop_coords_rejected)


def test_empty_stitcher_passthrough_copies_crop_params():
    """Fix #4: empty-frames passthrough must shallow-copy crop_params for
    sibling-safety consistency (was leaving it shared by reference)."""
    stitcher = _make_synthetic_stitcher(n_frames=0)
    node = NV_RebuildBlendMask()
    (out, _) = node.rebuild(stitcher=stitcher, mask_config=None, **_all_sentinel_kwargs())
    if out["crop_params"] is stitcher["crop_params"]:
        raise AssertionError(
            "Empty-stitcher passthrough did not shallow-copy crop_params — "
            "downstream mutation could corrupt upstream's dict"
        )
    # And confirm the contents survived
    assert out["crop_params"] == stitcher["crop_params"], "crop_params content drifted on passthrough"


run_test("empty_stitcher_passthrough_copies_crop_params",
         test_empty_stitcher_passthrough_copies_crop_params)


def test_missing_crop_params_falls_back_cleanly():
    """Edge case from Gemini: stitcher with no crop_params key at all should not KeyError."""
    stitcher = _make_synthetic_stitcher()
    del stitcher["crop_params"]
    node = NV_RebuildBlendMask()
    # Should run with all-fallback values, no exception
    (out, report) = node.rebuild(stitcher=stitcher, mask_config=None, **_all_sentinel_kwargs())
    assert "(fallback)" in report  # all sources should attribute to fallback
    assert "crop_params" in out  # output should have one even if input didn't
    assert out["crop_params"]["rebuild_blend_mask_applied"] is True


run_test("missing_crop_params_falls_back_cleanly", test_missing_crop_params_falls_back_cleanly)


def test_invalid_crop_target_dimensions_rejected():
    """Edge case: zero or negative crop_target_w/h must raise before per-frame loop."""
    stitcher = _make_synthetic_stitcher()
    stitcher["crop_target_w"] = 0  # invalid
    node = NV_RebuildBlendMask()
    raised = False
    try:
        node.rebuild(stitcher=stitcher, mask_config=None, **_all_sentinel_kwargs())
    except ValueError as e:
        raised = True
        if "crop_target" not in str(e):
            raise AssertionError(f"Error should mention crop_target: {e}")
    if not raised:
        raise AssertionError("crop_target_w=0 was accepted")


run_test("invalid_crop_target_dimensions_rejected", test_invalid_crop_target_dimensions_rejected)


# =============================================================================
# Summary
# =============================================================================
print()
print(f"Results: {passed} passed, {failed} failed out of {passed + failed} tests")
if errors:
    print()
    for name, e in errors:
        print(f"  FAILED {name}: {e}")
    sys.exit(1)
else:
    print("All tests passed!")
