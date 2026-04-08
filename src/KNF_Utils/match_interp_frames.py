"""
NV MatchInterpFrames - Expand a mask/image batch to match frame-interpolated video length.

When you run frame interpolation (e.g., GIMM-VFI 2x) on an RGB video but don't want to
re-run a model on the mask, this node expands the mask batch to match the interpolated
frame count using nearest-neighbor duplication or linear interpolation (with optional threshold).

Example: 90-frame mask + interpolation_factor=2 → 179-frame mask that aligns with
the 179-frame interpolated RGB.
"""

import torch


class NV_MatchInterpFrames:
    """Expand a mask or image batch to match frame-interpolated video length."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "interpolation_factor": ("INT", {
                    "default": 2,
                    "min": 2,
                    "max": 16,
                    "step": 1,
                    "tooltip": "Same factor used in your frame interpolation node (e.g., GIMM-VFI interpolation_factor)"
                }),
                "method": (["nearest", "lerp", "lerp_threshold"], {
                    "default": "nearest",
                    "tooltip": "nearest = duplicate previous frame until next original. lerp = blend neighbors (preserves soft alpha). lerp_threshold = blend then binarize at threshold."
                }),
            },
            "optional": {
                "images": ("IMAGE", {
                    "tooltip": "IMAGE batch [B,H,W,C] to expand"
                }),
                "masks": ("MASK", {
                    "tooltip": "MASK batch [B,H,W] to expand"
                }),
                "target_frame_count": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100000,
                    "step": 1,
                    "tooltip": "Override output frame count. 0 = auto-compute from interpolation_factor."
                }),
                "threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Binarization threshold for lerp_threshold method. Ignored by nearest and lerp."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("images", "masks")
    FUNCTION = "execute"
    CATEGORY = "NV_Utils/utils"
    DESCRIPTION = "Expand a mask/image batch to match frame-interpolated video. Avoids re-running expensive models on interpolated frames."

    def _expand(self, source: torch.Tensor, interpolation_factor: int, method: str, target_frame_count: int, threshold: float) -> torch.Tensor:
        T = source.shape[0]
        device = source.device
        out_len = target_frame_count if target_frame_count > 0 else (T - 1) * interpolation_factor + 1

        if method == "nearest":
            indices = (torch.arange(out_len, device=device) // interpolation_factor).clamp_max(T - 1)
            return source.index_select(0, indices)

        # lerp and lerp_threshold share the blending step
        t_positions = torch.linspace(0, T - 1, out_len, device=device)
        floor_idx = torch.floor(t_positions).long()
        ceil_idx = torch.clamp(floor_idx + 1, max=T - 1)
        frac = (t_positions - floor_idx.float()).view(out_len, *([1] * (source.ndim - 1)))

        lo = source.index_select(0, floor_idx)
        hi = source.index_select(0, ceil_idx)
        blended = lo * (1.0 - frac) + hi * frac

        if method == "lerp_threshold":
            return (blended >= threshold).to(source.dtype)
        return blended

    def execute(self, interpolation_factor: int, method: str,
                images: torch.Tensor | None = None, masks: torch.Tensor | None = None,
                target_frame_count: int = 0, threshold: float = 0.5):
        if images is None and masks is None:
            raise ValueError("At least one of images or masks must be provided")

        out_images = torch.empty(0)
        out_masks = torch.empty(0)

        if images is not None:
            out_images = self._expand(images, interpolation_factor, method, target_frame_count, threshold)
            print(f"[NV_MatchInterpFrames] images: {images.shape[0]} → {out_images.shape[0]} frames (factor={interpolation_factor}, method={method})")

        if masks is not None:
            out_masks = self._expand(masks, interpolation_factor, method, target_frame_count, threshold)
            print(f"[NV_MatchInterpFrames] masks: {masks.shape[0]} → {out_masks.shape[0]} frames (factor={interpolation_factor}, method={method})")

        return (out_images, out_masks)


NODE_CLASS_MAPPINGS = {
    "NV_MatchInterpFrames": NV_MatchInterpFrames,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_MatchInterpFrames": "NV Match Interp Frames",
}
