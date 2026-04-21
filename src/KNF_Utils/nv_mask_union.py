"""
NV Mask Union - Combine multiple MASK tensors via pixel-wise max (logical OR).

Primary use case: combining per-region SAM3 masks (e.g. face + hair + neck)
into a single whole-head mask. Each region is optimally segmented in its own
SAM3 track, then unioned here at the end.

Supports 2-4 input masks; omitted optional inputs are ignored. Inputs are
normalized to the first mask's device/dtype and promoted 2D→3D automatically,
so masks from different upstream nodes can be mixed without manual alignment.
"""

import torch


class NV_MaskUnion:
    """Union (pixel-wise max) of up to 4 MASK tensors. Any pixel that is
    foreground in ANY input is foreground in the output."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask_a": ("MASK", {"tooltip": "First mask (required)."}),
                "mask_b": ("MASK", {"tooltip": "Second mask (required)."}),
                "clamp_output": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Clamp result to [0,1]. Leave on unless you "
                               "intentionally want out-of-range soft-mask values.",
                }),
            },
            "optional": {
                "mask_c": ("MASK", {"tooltip": "Third mask (optional)."}),
                "mask_d": ("MASK", {"tooltip": "Fourth mask (optional)."}),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("unioned_mask",)
    FUNCTION = "union"
    CATEGORY = "NV_Utils/Mask"
    DESCRIPTION = (
        "Combines up to 4 MASK tensors via pixel-wise max (logical OR). "
        "Typical use: union per-region SAM3 masks (face + hair) into a single "
        "whole-head mask. Auto-promotes 2D [H,W] masks to 3D [B,H,W] and "
        "aligns device/dtype to the first input."
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

    def union(self, mask_a, mask_b, clamp_output=True, mask_c=None, mask_d=None):
        # Explicit (name, tensor) pairing — avoids index-vs-label drift when
        # mask_c is None but mask_d is provided.
        inputs = [("a", mask_a), ("b", mask_b), ("c", mask_c), ("d", mask_d)]
        pairs = [(n, self._normalize(n, m)) for n, m in inputs if m is not None]

        ref_name, ref_mask = pairs[0]
        # Sanitize NaN/inf — one corrupt upstream mask shouldn't poison the output.
        # NaN propagates through max, and clamp() doesn't fix NaN.
        ref_mask = torch.nan_to_num(ref_mask, nan=0.0, posinf=1.0, neginf=0.0)
        # Clone so iterative in-place max doesn't mutate upstream mask_a.
        result = ref_mask.clone()

        for name, m in pairs[1:]:
            # Align device + dtype to reference — masks from different upstream
            # nodes may live on CPU vs GPU or be fp16 vs fp32.
            m = m.to(device=result.device, dtype=result.dtype)
            m = torch.nan_to_num(m, nan=0.0, posinf=1.0, neginf=0.0)
            if m.shape != result.shape:
                raise ValueError(
                    f"[NV_MaskUnion] shape mismatch after promotion: "
                    f"mask_{ref_name}={tuple(result.shape)}, "
                    f"mask_{name}={tuple(m.shape)}. "
                    f"All masks must share the same [B,H,W] shape."
                )
            torch.maximum(result, m, out=result)

        if clamp_output:
            result = result.clamp(0.0, 1.0)

        # ComfyUI MASK convention is float32; coerce if upstream sent fp16/fp64.
        if result.dtype != torch.float32:
            result = result.to(dtype=torch.float32)

        return (result,)


NODE_CLASS_MAPPINGS = {
    "NV_MaskUnion": NV_MaskUnion,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_MaskUnion": "NV Mask Union",
}
