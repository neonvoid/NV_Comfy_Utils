"""
NV Temporal Retime — Slowdown-for-rediffusion bookend nodes.

High-motion video produces poor rediffusion results because per-frame delta
is too large for the diffusion model. This pair of nodes enables:

  1. NV_RetimePrep   — compute retiming metadata, output fps values for
                       an external frame interpolator (e.g. GIMM-VFI)
  2. NV_RetimeRestore — select frames back to original timing after rediffusion

Pipeline:
  RetimePrep → GIMM-VFI (external) → rediffuse → RetimeRestore
"""

import torch

from .chunk_utils import is_wan_aligned


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_slowed_frame_count(original: int, factor: int) -> int:
    """Compute frame count after slowdown: (N-1) * factor + 1.

    This preserves WAN alignment automatically:
      original = 4k+1  →  slowed = 4*(k*factor) + 1
    """
    return (original - 1) * factor + 1


# ---------------------------------------------------------------------------
# NV_RetimePrep
# ---------------------------------------------------------------------------

class NV_RetimePrep:
    """Compute retiming metadata and fps values for external frame interpolation.

    Pass images through unchanged; output source/target fps for GIMM-VFI
    and a RETIME_CONFIG for NV_RetimeRestore.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Original video frames [B,H,W,C]"
                }),
                "slowdown_factor": ("INT", {
                    "default": 2,
                    "min": 2,
                    "max": 8,
                    "step": 1,
                    "tooltip": "How many times to slow down. 2x = half the motion per frame, double the frame count."
                }),
                "source_fps": ("FLOAT", {
                    "default": 24.0,
                    "min": 1.0,
                    "max": 120.0,
                    "step": 0.1,
                    "tooltip": "FPS of input video. Used to compute target fps for the interpolator."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "FLOAT", "FLOAT", "INT", "RETIME_CONFIG")
    RETURN_NAMES = ("images", "source_fps", "target_fps", "expected_frames", "retime_config")
    FUNCTION = "execute"
    CATEGORY = "NV_Utils/temporal"
    DESCRIPTION = "Prepare video for slowdown rediffusion. Outputs fps values for an external frame interpolator (e.g. GIMM-VFI) and a config for NV_RetimeRestore."

    def execute(self, images, slowdown_factor, source_fps):
        original_count = images.shape[0]
        slowed_count = _compute_slowed_frame_count(original_count, slowdown_factor)
        target_fps = source_fps / slowdown_factor

        wan_note = ""
        if is_wan_aligned(original_count) and is_wan_aligned(slowed_count):
            wan_note = " (WAN-aligned)"
        elif not is_wan_aligned(slowed_count):
            wan_note = f" (WARNING: slowed count {slowed_count} is NOT WAN-aligned)"

        print(f"[NV_RetimePrep] {original_count} frames @ {source_fps}fps → "
              f"{slowed_count} frames @ {target_fps:.1f}fps ({slowdown_factor}x slowdown){wan_note}")

        config = {
            "original_frame_count": original_count,
            "slowdown_factor": slowdown_factor,
            "expected_slowed_count": slowed_count,
            "source_fps": source_fps,
            "target_fps": target_fps,
        }

        return (images, source_fps, target_fps, slowed_count, config)


# ---------------------------------------------------------------------------
# NV_RetimeRestore
# ---------------------------------------------------------------------------

class NV_RetimeRestore:
    """Restore original timing by selecting frames from rediffused slow video.

    After rediffusing a slowed-down video, this node picks every Nth frame
    to return to the original frame count and timing.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Rediffused slowed video frames [B,H,W,C]"
                }),
                "retime_config": ("RETIME_CONFIG", {
                    "tooltip": "Config from NV_RetimePrep"
                }),
                "method": (["select", "blend"], {
                    "default": "select",
                    "tooltip": "select: pick every Nth frame (sharp). blend: weighted average of neighbors (smooth, handles count mismatches)."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "execute"
    CATEGORY = "NV_Utils/temporal"
    DESCRIPTION = "Restore original frame timing from a rediffused slow video. Pair with NV_RetimePrep."

    def execute(self, images, retime_config, method):
        factor = retime_config["slowdown_factor"]
        original_count = retime_config["original_frame_count"]
        expected = retime_config["expected_slowed_count"]
        available = images.shape[0]

        if available != expected:
            print(f"[NV_RetimeRestore] WARNING: expected {expected} slowed frames but got {available} "
                  f"(GIMM-VFI rounding). Using proportional mapping.")

        if method == "select":
            output = self._select_frames(images, original_count, available)
        else:
            output = self._blend_frames(images, original_count, available)

        print(f"[NV_RetimeRestore] {available} slowed frames → {output.shape[0]} restored frames "
              f"(factor={factor}, method={method})")

        return (output,)

    @staticmethod
    def _select_frames(images, original_count, available):
        """Pick evenly-spaced frames using proportional mapping.

        Maps original frame i to slowed frame position proportionally,
        handling any count mismatch from interpolator rounding.
        """
        if original_count == 1:
            return images[0:1]
        step = (available - 1) / (original_count - 1)
        indices = [min(round(i * step), available - 1) for i in range(original_count)]
        return images[indices]

    @staticmethod
    def _blend_frames(images, original_count, available):
        """Weighted average of two nearest source frames for each target position.

        Uses proportional positioning so it handles any slowed frame count,
        not just exact multiples.
        """
        if original_count == 1:
            return images[0:1]
        step = (available - 1) / (original_count - 1)
        frames = []
        for i in range(original_count):
            pos = i * step
            if pos >= available - 1:
                frames.append(images[available - 1:available])
            else:
                lo = int(pos)
                hi = min(lo + 1, available - 1)
                t = pos - lo
                blended = images[lo] * (1.0 - t) + images[hi] * t
                frames.append(blended.unsqueeze(0))
        return torch.cat(frames, dim=0)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "NV_RetimePrep": NV_RetimePrep,
    "NV_RetimeRestore": NV_RetimeRestore,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_RetimePrep": "NV Retime Prep",
    "NV_RetimeRestore": "NV Retime Restore",
}
