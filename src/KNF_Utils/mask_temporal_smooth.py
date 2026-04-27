"""
NV Mask Temporal Smooth - Pure temporal median filter for SAM3 mask jitter.

Drop-in between SAM3 mask output and NV_VaceControlVideoPrep to reduce per-frame
silhouette wiggle. SAM3 produces stochastic per-frame masks (System Constraint:
"different per-frame inputs ALWAYS produce different masks"). This wiggle propagates
into the 64-channel packed VACE control mask + the inactive/reactive VAE encodes,
causing per-frame variation in the additive residual injected into the WAN denoiser.

Implementation: per-pixel temporal median over a sliding window. No flow, no
geodesic, no consensus algorithm — just median over time. Translation-only motion
compensation is unnecessary at this scope because Track B (NV_BboxAlignedMaskStabilizer)
stabilizes the bbox upstream; once the crop is stable, raw temporal median averages
out the per-frame silhouette wiggle without smearing legitimate motion.

Memory characteristics: at 512x512 / 277 frames / float32, peak ~290 MB. Chunked
along the temporal axis to bound VRAM under back-to-back queue pressure.
"""

import torch
import torch.nn.functional as F


def _temporal_median(mask: torch.Tensor, window: int) -> torch.Tensor:
    """Per-pixel temporal median filter over a sliding window.

    Args:
        mask: [T, H, W] or [T, 1, H, W], float in [0, 1]
        window: odd int, window size in frames (3-9 typical)

    Returns:
        Same shape as input. Each frame replaced by median of window centered on it.
        Edge frames pad-replicate.
    """
    squeeze_after = False
    if mask.ndim == 4 and mask.shape[1] == 1:
        mask = mask.squeeze(1)
        squeeze_after = True

    T, H, W = mask.shape
    half = window // 2

    # Replicate-pad temporally so edge frames have a full window
    padded = F.pad(
        mask.unsqueeze(0).unsqueeze(0),  # [1, 1, T, H, W]
        (0, 0, 0, 0, half, half),
        mode='replicate'
    ).squeeze(0).squeeze(0)  # [T + 2*half, H, W]

    # Stack windows then median over the window axis
    out = torch.empty_like(mask)
    chunk_size = 32  # bound peak memory: 32 frames * window * H * W * 4 bytes
    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        chunk_views = []
        for offset in range(window):
            chunk_views.append(padded[start + offset:end + offset])
        stack = torch.stack(chunk_views, dim=0)  # [window, chunk, H, W]
        out[start:end] = stack.median(dim=0).values

    if squeeze_after:
        out = out.unsqueeze(1)

    return out


class NV_MaskTemporalSmooth:
    """Apply temporal median smoothing to a mask sequence.

    Reduces per-frame SAM3 silhouette jitter that propagates into VACE conditioning
    and causes head jitter in the diffusion output. Wire between SAM3 mask output
    and NV_VaceControlVideoPrep.

    This is the cheap test before extending Track B to output a stabilized mask.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK", {
                    "tooltip": "Input mask sequence [T, H, W] from SAM3 or upstream segmentor.",
                }),
                "window": ("INT", {
                    "default": 5, "min": 3, "max": 15, "step": 2,
                    "tooltip": "Temporal window size (odd). 3=aggressive, 5=balanced, 7+=may smear motion. "
                               "Use 5 as the starting point for face-swap pipelines.",
                }),
                "binarize": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Threshold output at 0.5 for hard binary mask. Off=preserve soft values "
                               "(recommended — soft mask preserves stabilization signal at boundary).",
                }),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "execute"
    CATEGORY = "NV_Utils/mask"
    DESCRIPTION = (
        "Pure temporal median filter for mask jitter reduction. Drop in between SAM3 and "
        "NV_VaceControlVideoPrep. Window 5 = balanced; lower for fast motion, higher for "
        "static subjects. No flow, no consensus algorithm — just per-pixel median over time. "
        "Reduces per-frame silhouette wiggle that propagates into VACE conditioning."
    )

    def execute(self, mask, window, binarize):
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        if mask.shape[0] < 2:
            print(f"[NV_MaskTemporalSmooth] Single frame input; passthrough")
            return (mask,)

        # Ensure odd window
        if window % 2 == 0:
            window += 1

        # Clamp window to sequence length
        if window > mask.shape[0]:
            window = mask.shape[0] if mask.shape[0] % 2 == 1 else mask.shape[0] - 1

        T = mask.shape[0]
        result = _temporal_median(mask.float(), window)

        if binarize:
            result = (result > 0.5).float()

        # Diagnostic: how much did smoothing change the input?
        delta = (result - mask.float()).abs().mean().item()
        print(f"[NV_MaskTemporalSmooth] T={T} window={window} mean_delta={delta:.4f} "
              f"binarize={binarize}")

        return (result,)


NODE_CLASS_MAPPINGS = {
    "NV_MaskTemporalSmooth": NV_MaskTemporalSmooth,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_MaskTemporalSmooth": "NV Mask Temporal Smooth",
}
