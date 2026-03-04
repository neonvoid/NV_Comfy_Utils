"""
NV Reference Frame Prep

Selects and prepares reference frames for re-denoising in cascaded pipelines.

Uniformly samples N frames from a Stage 1 output video (same selection logic as
NV_VacePrePassReference), upscales to target resolution, and VAE-encodes to latent.

Intended workflow:
    Stage 1 decode → this node → KSampler (denoise 0.5-0.7) → VAE Decode → VacePrePassReference

By re-denoising the selected reference frames at higher resolution before they enter
VacePrePassReference, faces and details that were too coarse at low-res get refined
while sharing attention context (batch denoise) for cross-frame consistency.
"""

import torch
import comfy.utils
import comfy.model_management
from .streaming_vace_to_video import streaming_vae_encode


class NV_ReferenceFramePrep:
    """
    Select reference frames from a video, upscale, and VAE-encode for re-denoising.

    Outputs a LATENT ready for KSampler and the upscaled IMAGE for preview.
    Frame selection uses the same uniform sampling as NV_VacePrePassReference
    so the indices match when fed back.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Stage 1 decoded video (full length). N frames will be uniformly sampled."
                }),
                "vae": ("VAE",),
                "num_refs": ("INT", {"default": 5, "min": 1, "max": 20, "step": 1,
                             "tooltip": "Number of frames to sample. Match this to VacePrePassReference's num_refs_per_chunk."}),
                "width": ("INT", {"default": 832, "min": 16, "max": 8192, "step": 16,
                          "tooltip": "Target width for upscaling (should match Stage 2 target resolution)."}),
                "height": ("INT", {"default": 480, "min": 16, "max": 8192, "step": 16,
                           "tooltip": "Target height for upscaling (should match Stage 2 target resolution)."}),
                "frame_repeat": ("INT", {"default": 4, "min": 1, "max": 8, "step": 1,
                                "tooltip": "Repeat each frame N times before VAE encode. "
                                           "WAN 3D VAE has 4x temporal compression — without repeat, adjacent "
                                           "frames blend into shared latent frames, blurring faces. "
                                           "4x repeat = each ref maps to exactly 1 latent frame. "
                                           "Match VacePrePassReference's frame_repeat setting."}),
            },
        }

    RETURN_TYPES = ("LATENT", "IMAGE",)
    RETURN_NAMES = ("latent", "selected_frames",)
    FUNCTION = "execute"
    CATEGORY = "NV_Utils"
    DESCRIPTION = (
        "Select and prepare reference frames for re-denoising. "
        "Uniformly samples N frames, upscales to target resolution, and VAE-encodes. "
        "Wire the LATENT output to a KSampler, decode, then feed into VacePrePassReference."
    )

    def execute(self, images, vae, num_refs, width, height, frame_repeat):
        total = images.shape[0]
        num_refs = min(num_refs, total)

        # Uniform sampling — same logic as VacePrePassReference
        if num_refs >= total:
            indices = list(range(total))
        else:
            indices = torch.linspace(0, total - 1, num_refs).long().tolist()

        selected = images[indices]  # [N, H_in, W_in, C]

        # Upscale to target resolution
        selected = comfy.utils.common_upscale(
            selected.movedim(-1, 1), width, height, "bilinear", "center"
        ).movedim(1, -1)

        print(f"[NV_ReferenceFramePrep] Selected {num_refs} frames "
              f"(indices: {indices}) from {total} total")
        print(f"[NV_ReferenceFramePrep] Upscaled to {width}x{height}")

        # Frame repeat: duplicate each frame N times for 3D VAE temporal compression.
        # Without this, adjacent frames blend into shared latent frames.
        # 4x repeat = each reference maps to exactly 1 clean latent frame.
        if frame_repeat > 1:
            repeated = selected.unsqueeze(1).expand(-1, frame_repeat, -1, -1, -1)
            repeated = repeated.reshape(-1, *selected.shape[1:])
        else:
            repeated = selected

        total_pixel_frames = repeated.shape[0]
        print(f"[NV_ReferenceFramePrep] Frame repeat {frame_repeat}x -> "
              f"{total_pixel_frames} pixel frames -> {(total_pixel_frames - 1) // 4 + 1} latent frames")

        # VAE encode (streaming for OOM safety)
        latent = streaming_vae_encode(vae, repeated[:, :, :, :3])
        del repeated
        # latent shape: [1, 16, T, H/8, W/8] where T = num_refs (with 4x repeat)

        # Place on intermediate device (matches ComfyUI convention)
        latent = latent.to(comfy.model_management.intermediate_device())

        print(f"[NV_ReferenceFramePrep] Encoded to latent shape: {latent.shape}")

        return ({"samples": latent}, selected)


NODE_CLASS_MAPPINGS = {
    "NV_ReferenceFramePrep": NV_ReferenceFramePrep,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_ReferenceFramePrep": "NV Reference Frame Prep",
}
