"""
NV Reference Frame Prep

Selects and prepares reference frames for re-denoising in cascaded pipelines.

Uniformly samples N frames from a Stage 1 output video (same selection logic as
NV_VacePrePassReference), upscales to target resolution, and VAE-encodes to latent.

When a chunk plan JSON + chunk_index are provided, samples only from that chunk's
frame range — giving each chunk its own pose/lighting-matched references.

Intended workflow:
    Stage 1 decode → this node → KSampler (denoise 0.5-0.7) → VAE Decode → VacePrePassReference

By re-denoising the selected reference frames at higher resolution before they enter
VacePrePassReference, faces and details that were too coarse at low-res get refined
while sharing attention context (batch denoise) for cross-frame consistency.
"""

import json
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

    Optional chunk plan input restricts selection to a specific chunk's frame range.
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
            "optional": {
                "plan_json_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to chunk_plan.json from NV_ParallelChunkPlanner. "
                               "When provided with chunk_index, samples refs only from that chunk's frame range."
                }),
                "chunk_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 15,
                    "tooltip": "Which chunk to select references for (0-indexed). "
                               "Requires plan_json_path to be set."
                }),
            },
        }

    RETURN_TYPES = ("LATENT", "IMAGE",)
    RETURN_NAMES = ("latent", "selected_frames",)
    FUNCTION = "execute"
    CATEGORY = "NV_Utils"
    DESCRIPTION = (
        "Select and prepare reference frames for re-denoising. "
        "Uniformly samples N frames, upscales to target resolution, and VAE-encodes. "
        "Wire the LATENT output to a KSampler, decode, then feed into VacePrePassReference. "
        "Optional: provide a chunk plan JSON to select refs per-chunk."
    )

    def execute(self, images, vae, num_refs, width, height, frame_repeat,
                plan_json_path="", chunk_index=0):
        total = images.shape[0]

        # Determine frame range — full video or chunk-specific
        start_frame = 0
        end_frame = total

        if plan_json_path:
            try:
                with open(plan_json_path, 'r') as f:
                    plan = json.load(f)
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Chunk plan not found: {plan_json_path}\n"
                    f"Run NV_ParallelChunkPlanner first to create the plan."
                )

            chunks = plan.get("chunks", [])
            if chunk_index >= len(chunks):
                raise ValueError(
                    f"Invalid chunk_index {chunk_index}. Plan has {len(chunks)} chunks (indices 0-{len(chunks)-1})."
                )

            chunk = chunks[chunk_index]
            start_frame = chunk["start_frame"]
            end_frame = min(chunk["end_frame"], total)

            print(f"[NV_ReferenceFramePrep] Chunk {chunk_index}: "
                  f"frames {start_frame}-{end_frame-1} ({end_frame - start_frame} frames)")

        # Slice to frame range
        chunk_images = images[start_frame:end_frame]
        chunk_total = chunk_images.shape[0]
        num_refs = min(num_refs, chunk_total)

        # Uniform sampling within the range — same logic as VacePrePassReference
        if num_refs >= chunk_total:
            local_indices = list(range(chunk_total))
        else:
            local_indices = torch.linspace(0, chunk_total - 1, num_refs).long().tolist()

        # Map back to global indices for logging
        global_indices = [start_frame + i for i in local_indices]

        selected = chunk_images[local_indices]  # [N, H_in, W_in, C]
        del chunk_images

        # Upscale to target resolution
        selected = comfy.utils.common_upscale(
            selected.movedim(-1, 1), width, height, "bilinear", "center"
        ).movedim(1, -1)

        print(f"[NV_ReferenceFramePrep] Selected {num_refs} frames "
              f"(global indices: {global_indices}) from {total} total")
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
