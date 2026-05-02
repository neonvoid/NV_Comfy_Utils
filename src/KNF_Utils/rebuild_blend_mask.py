"""
NV Rebuild Blend Mask — re-derive cropped_mask_for_blend on a loaded stitcher.

Use case: load a saved stitcher, tweak the blend mask shape (wider crop_expand_px,
different stitch_source, etc.), restitch — without recropping. Closes the
iteration loop for procedural mask editing post-save.

Re-derives blend mask FROM SCRATCH using the stitcher's `canvas_mask` (the raw
original SAM3 mask, no processing applied at save time per D-070) plus the new
crop-time params. Defaults come from `stitcher['crop_params']`, so passing
through with all sentinels reproduces the saved blend mask exactly.

Wiring pattern:
    NV_LoadStitcher_V2 → NV_RebuildBlendMask → NV_InpaintStitch_V2
                         (tweak widgets here)

What's tweakable here:
  - cleanup_fill_holes / cleanup_remove_noise / cleanup_smooth
  - crop_expand_px              (most-tweaked param)
  - crop_blend_feather_px
  - crop_stitch_source          (tight / processed / hybrid / bbox)
  - hybrid_falloff / hybrid_curve

What's NOT tweakable (would need a full re-crop via NV_InpaintCrop_V2):
  - target_width / target_height (output resolution)
  - cropped_to_canvas_*          (the crop region itself)
  - anomaly_threshold            (per-frame keep/skip decisions)

Output is a SHALLOW-COPIED stitcher dict with the heavy tensor lists shared by
reference (memory-friendly — a 277-frame 1080p stitcher is multi-GB; deepcopying
all of canvas_image is wasteful when we only mutate cropped_mask_for_blend).
The cropped_mask_for_blend list itself is replaced with new tensors so sibling
branches that share the input stitcher are not affected.
"""

import copy

import torch
import torchvision.transforms.v2.functional as TVF

from .mask_ops import (
    mask_erode_dilate as _op_erode_dilate,
    mask_fill_holes as _op_fill_holes,
    mask_remove_noise as _op_remove_noise,
    mask_smooth as _op_smooth,
    mask_blur,
    rescale_mask,
)


_SENTINEL_INT = -999
_SENTINEL_FLOAT = -999.0
_USE_SAVED = "use_saved"

# Fallback defaults when stitcher.crop_params is absent or missing a key
# (matches the widget defaults on NV_InpaintCrop_V2 so behavior is consistent).
_FALLBACKS = {
    "crop_stitch_source": "tight",
    "crop_expand_px": 0,
    "crop_blend_feather_px": 16,
    "cleanup_fill_holes": 0,
    "cleanup_remove_noise": 0,
    "cleanup_smooth": 0,
    "hybrid_falloff": 48,
    "hybrid_curve": 0.6,
}

# Per-key real-range bounds for the sentinel-gap defensive check
# (matches NV_InpaintCrop_V2's widget ranges + MASK_CONFIG_SCHEMA).
_REAL_BOUNDS = {
    "crop_expand_px": (-128, 128),
    "crop_blend_feather_px": (0, 64),
    "cleanup_fill_holes": (0, 128),
    "cleanup_remove_noise": (0, 32),
    "cleanup_smooth": (0, 127),
    "hybrid_falloff": (8, 192),
    "hybrid_curve": (0.1, 2.0),
}

_ALLOWED_STITCH_SOURCES = ("tight", "processed", "hybrid", "bbox")


class NV_RebuildBlendMask:
    """Rebuild cropped_mask_for_blend on a stitcher post-save (procedural mask iteration)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stitcher": ("STITCHER", {
                    "tooltip": (
                        "Stitcher dict from NV_LoadStitcher_V2 (or live from NV_InpaintCrop_V2). "
                        "Must be a V2 stitcher containing canvas_mask + cropped_to_canvas_* + "
                        "crop_target_w/h."
                    ),
                }),
                "crop_stitch_source": ([_USE_SAVED, "tight", "processed", "hybrid", "bbox"], {
                    "default": _USE_SAVED,
                    "tooltip": (
                        "'use_saved' (default) = preserve saved value from stitcher.crop_params. "
                        "tight = original SAM3 mask. processed = post-cleanup mask. "
                        "hybrid = wide soft falloff (most aggressive blend-back). "
                        "bbox = full crop region (everything inside the box gets blended)."
                    ),
                }),
                "crop_expand_px": ("INT", {
                    "default": _SENTINEL_INT, "min": _SENTINEL_INT, "max": 128, "step": 1,
                    "tooltip": (
                        f"Shrink (negative) or expand (positive) the processed mask. "
                        f"Real range -128 to 128. {_SENTINEL_INT} = use saved value from "
                        f"stitcher.crop_params. Most-tweaked param for widening the blend region."
                    ),
                }),
                "crop_blend_feather_px": ("INT", {
                    "default": _SENTINEL_INT, "min": _SENTINEL_INT, "max": 64, "step": 1,
                    "tooltip": (
                        f"Feather blend mask edges. Real range 0-64. "
                        f"{_SENTINEL_INT} = use saved value."
                    ),
                }),
                "cleanup_fill_holes": ("INT", {
                    "default": _SENTINEL_INT, "min": _SENTINEL_INT, "max": 128, "step": 1,
                    "tooltip": (
                        f"Fill holes in mask (morphological closing). Real range 0-128. "
                        f"{_SENTINEL_INT} = use saved value."
                    ),
                }),
                "cleanup_remove_noise": ("INT", {
                    "default": _SENTINEL_INT, "min": _SENTINEL_INT, "max": 32, "step": 1,
                    "tooltip": (
                        f"Remove isolated pixels (morphological opening). Real range 0-32. "
                        f"{_SENTINEL_INT} = use saved value."
                    ),
                }),
                "cleanup_smooth": ("INT", {
                    "default": _SENTINEL_INT, "min": _SENTINEL_INT, "max": 127, "step": 1,
                    "tooltip": (
                        f"Smooth jagged edges (binarize + blur). Real range 0-127. "
                        f"{_SENTINEL_INT} = use saved value."
                    ),
                }),
                "hybrid_falloff": ("INT", {
                    "default": _SENTINEL_INT, "min": _SENTINEL_INT, "max": 192, "step": 4,
                    "tooltip": (
                        f"Hybrid mode: soft transition radius (px). Real range 8-192. "
                        f"{_SENTINEL_INT} = use saved value. Only used when "
                        f"crop_stitch_source=hybrid."
                    ),
                }),
                "hybrid_curve": ("FLOAT", {
                    "default": _SENTINEL_FLOAT, "min": _SENTINEL_FLOAT, "max": 2.0, "step": 0.05,
                    "tooltip": (
                        f"Hybrid mode: falloff curve power. Real range 0.1-2.0. "
                        f"{_SENTINEL_FLOAT} = use saved value. Only used when "
                        f"crop_stitch_source=hybrid."
                    ),
                }),
            },
            "optional": {
                "mask_config": ("MASK_BLEND_CONFIG", {
                    "tooltip": (
                        "Optional MASK_BLEND_CONFIG override bus from NV_MaskBlendConfig. "
                        "When wired, its keys (cleanup_*, crop_expand_px, crop_blend_feather_px) "
                        "are applied where the corresponding widget is at its sentinel. "
                        "Resolution priority: widget value > mask_config value > stitcher.crop_params "
                        "> fallback default. NV_RebuildBlendMask is a BLEND-pipeline node "
                        "(D-189) — operates on the BLEND mask, distinct from the GEN mask."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("STITCHER", "STRING")
    RETURN_NAMES = ("modified_stitcher", "rebuild_report")
    FUNCTION = "rebuild"
    CATEGORY = "NV_Utils/Stitcher"
    DESCRIPTION = (
        "Rebuild cropped_mask_for_blend on a stitcher post-save. Re-derives blend "
        "mask from canvas_mask using new crop-time params, so you can iterate on mask "
        "shape (crop_expand_px, stitch_source, hybrid, etc.) without recropping. "
        "Default sentinels preserve the saved blend mask exactly. Wire between "
        "NV_LoadStitcher_V2 and NV_InpaintStitch_V2."
    )

    def rebuild(self, stitcher, crop_stitch_source, crop_expand_px, crop_blend_feather_px,
                cleanup_fill_holes, cleanup_remove_noise, cleanup_smooth,
                hybrid_falloff, hybrid_curve, mask_config=None):
        # --- Validate stitcher contract --------------------------------------
        if not isinstance(stitcher, dict):
            raise TypeError(
                f"[NV_RebuildBlendMask] stitcher must be a STITCHER dict, "
                f"got {type(stitcher).__name__}"
            )
        required_keys = (
            "canvas_mask", "cropped_to_canvas_x", "cropped_to_canvas_y",
            "cropped_to_canvas_w", "cropped_to_canvas_h",
            "crop_target_w", "crop_target_h", "cropped_mask_for_blend",
        )
        missing = [k for k in required_keys if k not in stitcher]
        if missing:
            raise ValueError(
                f"[NV_RebuildBlendMask] stitcher missing required V2 keys: {missing}. "
                f"Pass a V2 stitcher (from NV_InpaintCrop_V2 or NV_LoadStitcher_V2). "
                f"Note: V1/legacy stitchers don't have canvas_mask — they require a re-crop."
            )

        if mask_config is not None and not isinstance(mask_config, dict):
            raise TypeError(
                f"[NV_RebuildBlendMask] mask_config must be a dict (MASK_BLEND_CONFIG), "
                f"got {type(mask_config).__name__}"
            )

        # --- Resolve effective params: widget > mask_config > saved > fallback --
        # Per multi-AI impl-review (2026-05-01):
        #   * crop_stitch_source MUST also check mask_config (was skipped → inconsistent)
        #   * resolved values from mask_config / saved MUST be bounds-checked
        #     (previously only widget input was validated → bad saved/mask_config
        #     values silently produced garbage masks)
        saved_params = stitcher.get("crop_params") or {}

        def _bounds_check(key, value, source):
            """Validate a resolved numeric value against its real-range bounds.
            Source is included in the error message so the user knows where to fix it."""
            if key not in _REAL_BOUNDS:
                return
            lo, hi = _REAL_BOUNDS[key]
            if not (lo <= value <= hi):
                raise ValueError(
                    f"[NV_RebuildBlendMask] {key} = {value} (from {source}) is out of "
                    f"bounds [{lo}, {hi}]. {'Set widget back to the sentinel or use a value in range.' if source == 'widget' else f'Fix the upstream {source} so {key} is in [{lo}, {hi}].'}"
                )

        def _resolve(key, widget_val, sentinel, allow_mask_config=True):
            """Single-key resolution + post-resolution validation.

            Priority: widget > mask_config > saved > fallback.
            Validates the final value regardless of source (except fallback,
            which we trust as our own hardcoded constants).
            """
            if widget_val != sentinel:
                value, source = widget_val, "widget"
            elif allow_mask_config and mask_config is not None and key in mask_config:
                value, source = mask_config[key], "mask_config"
            elif key in saved_params:
                value, source = saved_params[key], "saved"
            else:
                value, source = _FALLBACKS[key], "fallback"
            if source != "fallback":
                _bounds_check(key, value, source)
            return value, source

        # crop_stitch_source — string sentinel, NOW also consults mask_config (Codex+Gemini fix)
        if crop_stitch_source != _USE_SAVED:
            eff_stitch_source, src_stitch_source = crop_stitch_source, "widget"
        elif mask_config is not None and "crop_stitch_source" in mask_config:
            eff_stitch_source, src_stitch_source = mask_config["crop_stitch_source"], "mask_config"
        elif "crop_stitch_source" in saved_params:
            eff_stitch_source, src_stitch_source = saved_params["crop_stitch_source"], "saved"
        else:
            eff_stitch_source, src_stitch_source = _FALLBACKS["crop_stitch_source"], "fallback"
        # Validate the resolved string regardless of source (except fallback)
        if src_stitch_source != "fallback" and eff_stitch_source not in _ALLOWED_STITCH_SOURCES:
            raise ValueError(
                f"[NV_RebuildBlendMask] crop_stitch_source = {eff_stitch_source!r} "
                f"(from {src_stitch_source}) not in {_ALLOWED_STITCH_SOURCES}"
            )

        eff_expand, src_expand = _resolve("crop_expand_px", crop_expand_px, _SENTINEL_INT)
        eff_feather, src_feather = _resolve("crop_blend_feather_px", crop_blend_feather_px, _SENTINEL_INT)
        eff_fill, src_fill = _resolve("cleanup_fill_holes", cleanup_fill_holes, _SENTINEL_INT)
        eff_noise, src_noise = _resolve("cleanup_remove_noise", cleanup_remove_noise, _SENTINEL_INT)
        eff_smooth, src_smooth = _resolve("cleanup_smooth", cleanup_smooth, _SENTINEL_INT)
        # hybrid_* — also consult mask_config now (semantic uniformity per impl-review).
        # The MASK_CONFIG_SCHEMA doesn't include them, so a stock NV_MaskProcessingConfig
        # output won't carry these keys. But a custom dict / future schema extension
        # could, and we want the same priority order to apply uniformly.
        eff_falloff, src_falloff = _resolve("hybrid_falloff", hybrid_falloff, _SENTINEL_INT)
        eff_curve, src_curve = _resolve("hybrid_curve", hybrid_curve, _SENTINEL_FLOAT)

        # --- Per-frame rebuild ----------------------------------------------
        n_blend = len(stitcher["cropped_mask_for_blend"])
        n_canvas_mask = len(stitcher["canvas_mask"])
        if n_canvas_mask != n_blend:
            raise ValueError(
                f"[NV_RebuildBlendMask] stitcher has {n_canvas_mask} canvas_mask entries "
                f"but {n_blend} cropped_mask_for_blend entries — these must match. "
                f"Stitcher is malformed; re-save from NV_InpaintCrop_V2."
            )
        if n_blend == 0:
            # Even on passthrough, follow the copy discipline: shallow-copy crop_params
            # so a downstream consumer can't mutate the upstream's dict via this branch.
            # (Gemini impl-review fix.)
            shallow = dict(stitcher)
            if "crop_params" in stitcher:
                shallow["crop_params"] = dict(stitcher["crop_params"])
            return (shallow, "[NV_RebuildBlendMask] No frames in stitcher; nothing to rebuild.")

        target_w = int(stitcher["crop_target_w"])
        target_h = int(stitcher["crop_target_h"])
        if target_w <= 0 or target_h <= 0:
            raise ValueError(
                f"[NV_RebuildBlendMask] invalid crop_target dimensions: "
                f"{target_w}x{target_h} (both must be > 0)"
            )
        resize_algo = stitcher.get("resize_algorithm", "bicubic")

        new_blend_masks = []
        for i in range(n_blend):
            canvas_mask_t = stitcher["canvas_mask"][i]
            # canvas_mask is uint8 [H, W] per V2 storage convention (D-070).
            # Convert to fp32 [1, H, W] for mask_ops which expect [B, H, W].
            if canvas_mask_t.dim() == 2:
                cm_fp32 = (canvas_mask_t.float() / 255.0).unsqueeze(0)
            elif canvas_mask_t.dim() == 3:
                # Defensive — accept [1, H, W] in case a future loader change drifts
                cm_fp32 = canvas_mask_t.float() / 255.0
                if cm_fp32.shape[0] != 1:
                    raise ValueError(
                        f"[NV_RebuildBlendMask] frame {i} canvas_mask has unexpected shape "
                        f"{tuple(canvas_mask_t.shape)}; expected [H,W] or [1,H,W]"
                    )
            else:
                raise ValueError(
                    f"[NV_RebuildBlendMask] frame {i} canvas_mask must be 2D [H,W] or 3D [1,H,W], "
                    f"got dim={canvas_mask_t.dim()}, shape={tuple(canvas_mask_t.shape)}"
                )

            original_mask = cm_fp32.clone()

            # --- Apply cleanup ops to derive processed_mask ------------------
            # (mirrors NV_InpaintCrop_V2 lines 530-540; same op order)
            processed_mask = original_mask.clone()
            if eff_fill > 0:
                processed_mask = _op_fill_holes(processed_mask, eff_fill)
            if eff_noise > 0:
                processed_mask = _op_remove_noise(processed_mask, eff_noise)
            if eff_expand != 0:
                processed_mask = _op_erode_dilate(processed_mask, eff_expand)
            if eff_smooth > 0:
                processed_mask = _op_smooth(processed_mask, eff_smooth)

            # --- Crop to ctc_x/y/w/h (with validation — Codex impl-review) ----
            # PyTorch slicing is permissive: negatives wrap, OOB truncates silently.
            # Validate explicitly so a malformed stitcher fails fast instead of
            # producing a plausible-but-wrong blend mask.
            ctc_x = int(stitcher["cropped_to_canvas_x"][i])
            ctc_y = int(stitcher["cropped_to_canvas_y"][i])
            ctc_w = int(stitcher["cropped_to_canvas_w"][i])
            ctc_h = int(stitcher["cropped_to_canvas_h"][i])
            canvas_H, canvas_W = original_mask.shape[1], original_mask.shape[2]
            if ctc_x < 0 or ctc_y < 0 or ctc_w <= 0 or ctc_h <= 0:
                raise ValueError(
                    f"[NV_RebuildBlendMask] frame {i}: invalid crop coords "
                    f"(x={ctc_x}, y={ctc_y}, w={ctc_w}, h={ctc_h}) — origin must be >=0 "
                    f"and width/height must be >0"
                )
            if ctc_x + ctc_w > canvas_W or ctc_y + ctc_h > canvas_H:
                raise ValueError(
                    f"[NV_RebuildBlendMask] frame {i}: crop region "
                    f"[{ctc_x}+{ctc_w}, {ctc_y}+{ctc_h}] exceeds canvas size "
                    f"[{canvas_W}, {canvas_H}]"
                )

            cropped_orig = original_mask[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w]
            cropped_proc = processed_mask[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w]

            # --- Resize to target_w x target_h -------------------------------
            if (ctc_w, ctc_h) != (target_w, target_h):
                cropped_orig = rescale_mask(cropped_orig, target_w, target_h, resize_algo)
                cropped_proc = rescale_mask(cropped_proc, target_w, target_h, resize_algo)

            # --- Build blend mask per stitch_source mode ---------------------
            # (mirrors NV_InpaintCrop_V2 lines 571-589)
            if eff_stitch_source == "bbox":
                blend_mask = torch.ones_like(cropped_orig)
            elif eff_stitch_source == "hybrid":
                proc = cropped_proc.clamp(0, 1)
                sigma = float(eff_falloff) / 3.0
                kernel_size = max(3, int(sigma * 6) | 1)  # odd, >=3
                falloff = TVF.gaussian_blur(proc.unsqueeze(1), kernel_size, sigma).squeeze(1)
                falloff = falloff.clamp(0, 1).pow(eff_curve)
                blend_mask = 1.0 - (1.0 - proc) * (1.0 - falloff)
            elif eff_stitch_source == "processed":
                blend_mask = cropped_proc.clone()
            else:  # tight
                blend_mask = cropped_orig.clone()

            # --- Apply blend feather (mirrors NV_InpaintCrop_V2 lines 591-602) -
            if eff_feather > 0:
                if eff_stitch_source == "bbox":
                    blend_mask = _op_erode_dilate(blend_mask, -eff_feather)
                    blend_mask = mask_blur(blend_mask, eff_feather)
                elif eff_stitch_source == "hybrid":
                    blend_mask = mask_blur(blend_mask, eff_feather)
                else:
                    blend_mask = _op_erode_dilate(blend_mask, eff_feather)
                    blend_mask = mask_blur(blend_mask, eff_feather)

            # Squeeze to [H, W] to match V2 storage convention
            new_blend_masks.append(blend_mask.squeeze(0))

        # --- Build new stitcher (shallow-copy + replace cropped_mask_for_blend) -
        # Heavy tensor lists (canvas_image, canvas_mask, etc.) are shared by
        # reference — we never mutate them, so siblings of the input stitcher
        # remain safe. Only the dict + cropped_mask_for_blend list + crop_params
        # dict are new objects.
        new_stitcher = dict(stitcher)
        new_stitcher["cropped_mask_for_blend"] = new_blend_masks

        # Update crop_params to reflect what was used (additive — preserve other keys)
        new_stitcher["crop_params"] = dict(stitcher.get("crop_params") or {})
        new_stitcher["crop_params"].update({
            "crop_stitch_source": eff_stitch_source,
            "crop_expand_px": eff_expand,
            "crop_blend_feather_px": eff_feather,
            "cleanup_fill_holes": eff_fill,
            "cleanup_remove_noise": eff_noise,
            "cleanup_smooth": eff_smooth,
            "hybrid_falloff": eff_falloff,
            "hybrid_curve": eff_curve,
            "rebuild_blend_mask_applied": True,
        })

        # --- Build report ---------------------------------------------------
        report_lines = [
            "=" * 64,
            f"REBUILD BLEND MASK REPORT  (frames rebuilt: {n_blend})",
            "=" * 64,
            f"Effective params (source in parentheses):",
            f"  crop_stitch_source:    {eff_stitch_source}  ({src_stitch_source})",
            f"  crop_expand_px:        {eff_expand}  ({src_expand})",
            f"  crop_blend_feather_px: {eff_feather}  ({src_feather})",
            f"  cleanup_fill_holes:    {eff_fill}  ({src_fill})",
            f"  cleanup_remove_noise:  {eff_noise}  ({src_noise})",
            f"  cleanup_smooth:        {eff_smooth}  ({src_smooth})",
            f"  hybrid_falloff:        {eff_falloff}  ({src_falloff})",
            f"  hybrid_curve:          {eff_curve}  ({src_curve})",
            "=" * 64,
        ]
        report = "\n".join(report_lines)
        print(report)
        return (new_stitcher, report)


NODE_CLASS_MAPPINGS = {
    "NV_RebuildBlendMask": NV_RebuildBlendMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_RebuildBlendMask": "NV Rebuild Blend Mask",
}
