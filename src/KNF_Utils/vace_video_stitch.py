"""
NV VACE Video Stitch — Pixel-space alpha composite for the VACE dual-mask workflow.

Designed for the pattern: bbox mask for VACE conditioning (better quality),
tight mask for pixel compositing (no hallucinated surroundings).

Formula: output = stitch_mask * vace_output + (1 - stitch_mask) * original

Pair with NV_VaceControlVideoPrep's stitch_mask output.
"""

import torch


class NV_VaceVideoStitch:
    """Alpha-composite VACE output onto original video using a stitch mask.

    Use the stitch_mask output from NV_VaceControlVideoPrep (tight mask, feathered)
    to composite only the subject from the VACE render, discarding hallucinated
    content outside the tight mask boundary.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_video": ("IMAGE", {
                    "tooltip": "Original video/image before VACE processing."
                }),
                "vace_output": ("IMAGE", {
                    "tooltip": "VACE render output (same resolution as original)."
                }),
                "stitch_mask": ("MASK", {
                    "tooltip": "Tight mask for pixel compositing (from NV_VaceControlVideoPrep's "
                               "stitch_mask output). Feathered edges produce seamless blending."
                }),
            },
            "optional": {
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Blend strength. 1.0 = full VACE replacement in masked area. "
                               "Lower values mix original content back in for subtler effects."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "NV_Utils/VACE"
    DESCRIPTION = (
        "Alpha-composite VACE output onto original video using a tight stitch mask. "
        "Designed for the dual-mask workflow: bbox mask feeds VACE for quality, "
        "tight mask composites the result for precision. Unmasked pixels are "
        "pixel-identical to the original."
    )

    def execute(self, original_video, vace_output, stitch_mask, strength=1.0):
        B_orig, H_orig, W_orig, C_orig = original_video.shape
        B_vace, H_vace, W_vace, C_vace = vace_output.shape

        # Validate spatial dimensions match
        if H_orig != H_vace or W_orig != W_vace:
            raise ValueError(
                f"Spatial dimensions must match: original {W_orig}x{H_orig}, "
                f"vace_output {W_vace}x{H_vace}. Resize one to match the other."
            )

        # Handle mask batch dimension
        if stitch_mask.dim() == 2:
            stitch_mask = stitch_mask.unsqueeze(0)

        B_mask = stitch_mask.shape[0]

        # Determine output batch size and broadcast
        B_out = max(B_orig, B_vace)

        if B_orig == 1 and B_out > 1:
            original_video = original_video.expand(B_out, -1, -1, -1)
        if B_vace == 1 and B_out > 1:
            vace_output = vace_output.expand(B_out, -1, -1, -1)
        if B_mask == 1 and B_out > 1:
            stitch_mask = stitch_mask.expand(B_out, -1, -1)

        # Validate mask spatial dims
        if stitch_mask.shape[1] != H_orig or stitch_mask.shape[2] != W_orig:
            raise ValueError(
                f"Stitch mask spatial dims ({stitch_mask.shape[2]}x{stitch_mask.shape[1]}) "
                f"must match video dims ({W_orig}x{H_orig})."
            )

        # [B, H, W] -> [B, H, W, 1] for broadcasting with [B, H, W, C]
        mask = stitch_mask.unsqueeze(-1)

        if strength < 1.0:
            mask = mask * strength

        result = mask * vace_output + (1.0 - mask) * original_video

        return (result.clamp(0.0, 1.0),)


NODE_CLASS_MAPPINGS = {
    "NV_VaceVideoStitch": NV_VaceVideoStitch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_VaceVideoStitch": "NV VACE Video Stitch",
}
