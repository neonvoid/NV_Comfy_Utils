"""
NV Mask Union - Combine multiple MASK tensors with seam-aware combine modes
+ optional post-union close/smooth and image-guided edge refinement.

Primary use case: combining per-region SAM3 masks (face + hair + body) into a
single seamless whole-subject mask. Each region is optimally segmented in its
own SAM3 track, then unioned here at the end.

Pipeline (all stages additive, default OFF — drop-in compatible with the
original max-only behavior):

    per-input: nan_to_num -> optional pre_feather (Gaussian)
    combine:   max | probabilistic_or
    post:      optional close (bridge 2-5 px gaps between mask supports)
    post:      optional smooth (Gaussian)
    post:      optional guided refine (snap union edge to source image gradients)
    final:     optional clamp [0,1] + cast fp32

Backward compat note: `clamp_output` is kept as the first widget (preserves
positional `widgets_values` mapping for existing workflows). New widgets are
appended after it.
"""

import torch

from .guided_filter import refine_mask
from .mask_ops import mask_blur, mask_fill_holes


_COMBINE_MODES = ("max", "probabilistic_or")


class NV_MaskUnion:
    """Union of up to 4 MASK tensors with optional pre/post seam-repair."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask_a": ("MASK", {"tooltip": "First mask (required)."}),
                "mask_b": ("MASK", {"tooltip": "Second mask (required)."}),
                # KEEP clamp_output FIRST — existing workflows have a single widget
                # value at this position. Inserting new widgets before it would
                # scramble positional widgets_values mapping in saved graphs.
                "clamp_output": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Clamp result to [0,1]. REQUIRED when combine_mode != 'max' — "
                        "probabilistic_or math is undefined for out-of-range inputs and "
                        "the node will reject clamp_output=False with combine_mode='probabilistic_or'."
                    ),
                }),
                "combine_mode": (list(_COMBINE_MODES), {
                    "default": "max",
                    "tooltip": (
                        "How to combine inputs. 'max' (default, backward compatible): pixel-wise "
                        "torch.maximum — preserves the brightest mask at every pixel. "
                        "'probabilistic_or': 1 - prod(1 - m_i) — soft OR that preserves "
                        "anti-aliased edges and saturates smoothly in overlap zones. Use "
                        "probabilistic_or for non-nested layouts (e.g. face + neck side-by-side); "
                        "for nested masks (body superset of face union hair) it won't reconcile "
                        "boundary disagreement — use post_close_px for that."
                    ),
                }),
                "pre_feather_px": ("INT", {
                    "default": 0, "min": 0, "max": 8, "step": 1,
                    "tooltip": (
                        "Gaussian kernel size applied to EACH input before combine. 0 = off. "
                        "Cheap way to soften step-edge SAM3 masks before they stack into the union. "
                        "Note: on already-tight masks this can fatten hair wisps — leave at 0 "
                        "unless you see hard inter-region steps."
                    ),
                }),
                "post_close_px": ("INT", {
                    "default": 0, "min": 0, "max": 32, "step": 1,
                    "tooltip": (
                        "Greyscale closing (dilate then erode) on the COMBINED union. Bridges "
                        "2-5 px gaps where mask supports disagree (e.g. SAM3 face ends 3 px before "
                        "hair starts). This is the main seam-repair lever for the body+hair+face "
                        "use case. Note: on soft-alpha inputs, greyscale closing can slightly "
                        "thicken semi-transparent fringes — keep small (4-8) for hair, larger "
                        "(12-24) for clothing seams."
                    ),
                }),
                "post_smooth_px": ("INT", {
                    "default": 0, "min": 0, "max": 32, "step": 1,
                    "tooltip": (
                        "Gaussian kernel size on the COMBINED union after closing. 0 = off. "
                        "Softens any morphology harshness from post_close. Preserves gradients "
                        "(unlike binarize-then-blur) so probabilistic_or anti-aliasing survives."
                    ),
                }),
            },
            "optional": {
                "mask_c": ("MASK", {"tooltip": "Third mask (optional)."}),
                "mask_d": ("MASK", {"tooltip": "Fourth mask (optional)."}),
                "guided_image": ("IMAGE", {
                    "tooltip": (
                        "Optional source IMAGE [B,H,W,3] used as the guide for edge-aware "
                        "refinement of the union output. When wired, runs guided filter "
                        "(He et al. 2013) — snaps the outer silhouette to luminance gradients. "
                        "Especially useful for hair edges. Spatial dims must match the union mask; "
                        "batch must match union batch or be 1 (broadcast)."
                    ),
                }),
                "guided_radius": ("INT", {
                    "default": 0, "min": 0, "max": 64, "step": 1,
                    "tooltip": (
                        "Guided filter window radius. 0 = auto (max(4, min(H,W) // 64) — ~16 "
                        "at 1080p, ~33 at 4K). Larger = smoother boundary, may over-relax fine "
                        "detail. Only used when guided_image is wired."
                    ),
                }),
                "guided_eps": ("FLOAT", {
                    "default": 0.001, "min": 0.0001, "max": 0.1, "step": 0.0001,
                    "tooltip": (
                        "Guided filter regularization. Lower = sharper edge tracking, higher = "
                        "smoother. 0.001 default is good for hair-edge alignment. Only used "
                        "when guided_image is wired."
                    ),
                }),
                "guided_strength": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": (
                        "Lerp between unrefined union (0.0) and guided-refined union (1.0). "
                        "0.7 default avoids over-erosion when the guide is unreliable (motion "
                        "blur, low contrast). Only used when guided_image is wired."
                    ),
                }),
                "verbose_debug": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Print per-input mask statistics (shape, fg-pixel count, range, mean) "
                        "+ which post-stages ran + per-stage timing to the ComfyUI console. "
                        "Off by default to avoid log spam. Turn on while debugging which input "
                        "is dominating the union or whether guided refine is actually firing."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("unioned_mask",)
    FUNCTION = "union"
    CATEGORY = "NV_Utils/Mask"
    DESCRIPTION = (
        "Combines up to 4 MASK tensors with optional pre/post seam-repair. Designed for "
        "per-region SAM3 segmentation (face + hair + body): combine via max or probabilistic "
        "OR, optionally close 2-5 px gaps between mask supports, smooth, and snap to source "
        "image edges via guided filter (He et al. 2013)."
    )

    @staticmethod
    def _normalize(name, mask):
        """Promote 2D [H,W] to 3D [B,H,W]; reject other shapes with a clear error."""
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        elif mask.ndim != 3:
            raise ValueError(
                f"[NV_MaskUnion] mask_{name} must be 2D [H,W] or 3D [B,H,W], "
                f"got shape {tuple(mask.shape)}."
            )
        return mask

    @staticmethod
    def _combine(masks, mode):
        """Combine a list of [B,H,W] masks via the requested mode."""
        if mode == "max":
            result = masks[0].clone()
            for m in masks[1:]:
                torch.maximum(result, m, out=result)
            return result
        if mode == "probabilistic_or":
            # 1 - prod(1 - m_i). Inputs assumed to be in [0, 1] (clamp_output gate
            # ensures the caller has accepted that contract).
            inv = 1.0 - masks[0]
            for m in masks[1:]:
                inv = inv * (1.0 - m)
            return 1.0 - inv
        raise ValueError(f"[NV_MaskUnion] unknown combine_mode {mode!r}")

    def union(
        self,
        mask_a, mask_b,
        clamp_output=True,
        combine_mode="max",
        pre_feather_px=0,
        post_close_px=0,
        post_smooth_px=0,
        mask_c=None, mask_d=None,
        guided_image=None,
        guided_radius=0,
        guided_eps=0.001,
        guided_strength=0.7,
    ):
        # --- Param validation ------------------------------------------------
        if combine_mode not in _COMBINE_MODES:
            raise ValueError(
                f"[NV_MaskUnion] combine_mode must be one of {_COMBINE_MODES}, "
                f"got {combine_mode!r}."
            )
        if combine_mode != "max" and not clamp_output:
            raise ValueError(
                f"[NV_MaskUnion] clamp_output=False is only valid with combine_mode='max'. "
                f"combine_mode={combine_mode!r} requires inputs/outputs in [0,1] — "
                f"out-of-range values produce undefined math."
            )

        # --- Normalize + sanitize inputs -------------------------------------
        inputs = [("a", mask_a), ("b", mask_b), ("c", mask_c), ("d", mask_d)]
        pairs = [(n, self._normalize(n, m)) for n, m in inputs if m is not None]
        ref_name, ref_mask = pairs[0]
        # nan_to_num BEFORE pre_feather so a NaN can't propagate into the Gaussian.
        ref_mask = torch.nan_to_num(ref_mask, nan=0.0, posinf=1.0, neginf=0.0)

        prepared = []
        for name, m in pairs:
            if name == ref_name:
                m_clean = ref_mask
            else:
                m_clean = m.to(device=ref_mask.device, dtype=ref_mask.dtype)
                m_clean = torch.nan_to_num(m_clean, nan=0.0, posinf=1.0, neginf=0.0)
            if m_clean.shape != ref_mask.shape:
                raise ValueError(
                    f"[NV_MaskUnion] shape mismatch after promotion: "
                    f"mask_{ref_name}={tuple(ref_mask.shape)}, "
                    f"mask_{name}={tuple(m_clean.shape)}. "
                    f"All masks must share the same [B,H,W] shape."
                )
            if pre_feather_px > 0:
                m_clean = mask_blur(m_clean, int(pre_feather_px))
            prepared.append(m_clean)

        # --- Combine ---------------------------------------------------------
        result = self._combine(prepared, combine_mode)

        # --- Post-union seam repair (the main lever for the "no seams" goal) -
        if post_close_px > 0:
            result = mask_fill_holes(result, int(post_close_px))
        if post_smooth_px > 0:
            result = mask_blur(result, int(post_smooth_px))

        # --- Optional guided refine on the union -----------------------------
        # Short-circuit on strength=0: refine_mask would just lerp back to the
        # original mask but pay the full guided-filter cost first.
        if guided_image is not None and float(guided_strength) > 0.0:
            self._validate_guide(guided_image, result)
            # Align device with the union (mask path normalizes to mask_a's
            # device; guide may be on a different device entirely).
            guide = guided_image.to(device=result.device, dtype=result.dtype)
            if guide.shape[0] == 1 and result.shape[0] > 1:
                # .expand() returns a non-contiguous view; some guided-filter
                # implementations require contiguous input.
                guide = guide.expand(result.shape[0], -1, -1, -1).contiguous()
            radius = (
                int(guided_radius) if guided_radius > 0
                else max(4, min(result.shape[1], result.shape[2]) // 64)
            )
            # refine_mask handles the [B,H,W,C] -> internal layout, fast variant
            # for 1080p+, and the strength lerp.
            result = refine_mask(
                result, guide,
                radius=radius,
                eps=float(guided_eps),
                strength=float(guided_strength),
                mode="color",
            )

        # --- Final clamp + dtype --------------------------------------------
        if clamp_output:
            result = result.clamp(0.0, 1.0)
        if result.dtype != torch.float32:
            result = result.to(dtype=torch.float32)

        return (result,)

    @staticmethod
    def _validate_guide(guided_image, mask):
        """Fail fast on shape/batch/device mismatch in the IMAGE guide."""
        if guided_image.dim() != 4 or guided_image.shape[-1] != 3:
            raise ValueError(
                f"[NV_MaskUnion] guided_image must be IMAGE [B,H,W,3], "
                f"got shape {tuple(guided_image.shape)}."
            )
        if (guided_image.shape[1] != mask.shape[1]
                or guided_image.shape[2] != mask.shape[2]):
            raise ValueError(
                f"[NV_MaskUnion] guided_image spatial dims "
                f"{tuple(guided_image.shape[1:3])} must match union mask "
                f"{tuple(mask.shape[1:])}."
            )
        if guided_image.shape[0] != mask.shape[0] and guided_image.shape[0] != 1:
            raise ValueError(
                f"[NV_MaskUnion] guided_image batch={guided_image.shape[0]} must "
                f"match union batch={mask.shape[0]} or be 1 (broadcast)."
            )


NODE_CLASS_MAPPINGS = {
    "NV_MaskUnion": NV_MaskUnion,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_MaskUnion": "NV Mask Union",
}
