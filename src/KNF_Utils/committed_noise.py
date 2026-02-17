"""
NV Committed Noise Generator

Generates full-length noise for entire video with optional FreeNoise correlation.
This noise can be:
1. Used directly by NV_VideoSampler for consistent chunked sampling
2. Saved to disk for parallel workers to load their slices

The key insight: with same seed, each chunk regenerates noise from position 0,
so overlapping frames get DIFFERENT noise. Committed noise ensures overlapping
frames get IDENTICAL noise across chunks.
"""

import torch
import comfy.sample
import os


def video_to_latent_frames(video_frames: int) -> int:
    """Convert video frame count to latent frame count (Wan 4:1 compression)."""
    if video_frames <= 0:
        return 0
    return (video_frames - 1) // 4 + 1


def apply_freenoise_temporal_correlation(
    noise: torch.Tensor,
    seed: int,
    context_length: int,
    context_overlap: int,
    temporal_dim: int = 2
) -> torch.Tensor:
    """
    Apply FreeNoise-style temporal correlation to noise tensor.

    FreeNoise works by copying noise from earlier frames to later frames,
    creating temporal correlation that improves consistency across chunks.

    Based on comfy.context_windows.apply_freenoise() but with explicit parameters.

    Args:
        noise: Full noise tensor [B, C, T, H, W]
        seed: Seed for shuffle randomization
        context_length: Context window size in LATENT frames
        context_overlap: Overlap between windows in LATENT frames
        temporal_dim: Temporal dimension index (2 for video latents)

    Returns:
        Noise tensor with temporal correlation applied
    """
    generator = torch.Generator(device='cpu').manual_seed(seed)
    latent_video_length = noise.shape[temporal_dim]
    delta = context_length - context_overlap

    if delta <= 0:
        return noise

    for start_idx in range(0, latent_video_length - context_length, delta):
        place_idx = start_idx + context_length

        actual_delta = min(delta, latent_video_length - place_idx)
        if actual_delta <= 0:
            break

        # Randomly select source frames from the established region
        list_idx = torch.randperm(actual_delta, generator=generator, device='cpu') + start_idx

        # Build slice objects for the temporal dimension
        source_slice = [slice(None)] * noise.ndim
        source_slice[temporal_dim] = list_idx
        target_slice = [slice(None)] * noise.ndim
        target_slice[temporal_dim] = slice(place_idx, place_idx + actual_delta)

        # Copy noise from earlier frames to later frames
        noise[tuple(target_slice)] = noise[tuple(source_slice)]

    return noise


class NV_CommittedNoise:
    """
    Generate committed noise tensor with optional FreeNoise correlation.

    This node pre-generates noise for the entire video, ensuring that
    overlapping frames between chunks get identical noise values.

    FreeNoise improves temporal consistency by correlating noise across
    context windows - frames share similar noise patterns with their
    temporal neighbors.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent_image": ("LATENT", {
                    "tooltip": "Reference latent for shape (full video length)"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Seed for noise generation"
                }),
            },
            "optional": {
                "enable_freenoise": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply FreeNoise temporal correlation for smoother transitions"
                }),
                "context_length": ("INT", {
                    "default": 81,
                    "min": 1,
                    "max": 500,
                    "tooltip": "Context window size in VIDEO frames (will be converted to latent internally)"
                }),
                "context_overlap": ("INT", {
                    "default": 16,
                    "min": 0,
                    "max": 200,
                    "tooltip": "Overlap between context windows in VIDEO frames"
                }),
            }
        }

    RETURN_TYPES = ("COMMITTED_NOISE",)
    RETURN_NAMES = ("committed_noise",)
    FUNCTION = "generate"
    CATEGORY = "NV_Utils/sampling"
    DESCRIPTION = "Pre-generate noise for entire video with optional FreeNoise temporal correlation. Ensures overlapping frames get identical noise across chunks."

    def generate(
        self,
        latent_image: dict,
        seed: int,
        enable_freenoise: bool = True,
        context_length: int = 81,
        context_overlap: int = 16
    ) -> tuple:
        latent = latent_image["samples"]

        # Get batch indices if present
        batch_inds = latent_image.get("batch_index", None)

        # Generate full noise tensor using ComfyUI's standard method
        noise = comfy.sample.prepare_noise(latent, seed, batch_inds)

        # Get shape info for logging
        total_latent_frames = latent.shape[2] if latent.ndim >= 4 else 1

        # Apply FreeNoise correlation if enabled and video is long enough
        if enable_freenoise:
            # Convert video frames to latent frames
            context_length_latent = video_to_latent_frames(context_length)
            context_overlap_latent = video_to_latent_frames(context_overlap)

            if total_latent_frames > context_length_latent:
                noise = apply_freenoise_temporal_correlation(
                    noise,
                    seed=seed,
                    context_length=context_length_latent,
                    context_overlap=context_overlap_latent,
                    temporal_dim=2  # [B, C, T, H, W]
                )
                print(f"[NV_CommittedNoise] Applied FreeNoise: {total_latent_frames} latent frames, "
                      f"context={context_length_latent} (from {context_length} video), "
                      f"overlap={context_overlap_latent} (from {context_overlap} video)")
            else:
                print(f"[NV_CommittedNoise] Skipped FreeNoise: video ({total_latent_frames} latent frames) "
                      f"shorter than context ({context_length_latent} latent frames)")
        else:
            print(f"[NV_CommittedNoise] Generated noise without FreeNoise: {total_latent_frames} latent frames")

        return ({
            "noise": noise,
            "seed": seed,
            "freenoise_applied": enable_freenoise and total_latent_frames > video_to_latent_frames(context_length),
            "context_length_video": context_length,
            "context_overlap_video": context_overlap,
            "shape": list(noise.shape),
        },)


class NV_SaveCommittedNoise:
    """
    Save committed noise to disk for parallel workers.

    Parallel chunk processing requires all workers to use the same noise.
    This node saves the committed noise tensor to a file that workers can load.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "committed_noise": ("COMMITTED_NOISE", {
                    "tooltip": "Committed noise from NV_CommittedNoise"
                }),
                "output_path": ("STRING", {
                    "default": "committed_noise.pt",
                    "tooltip": "Path to save the noise file (.pt format)"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = "NV_Utils/sampling"
    DESCRIPTION = "Save committed noise to disk for parallel workers to load."

    def save(self, committed_noise: dict, output_path: str) -> tuple:
        # Ensure directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Save the committed noise dict
        torch.save(committed_noise, output_path)

        shape = committed_noise["shape"]
        print(f"[NV_SaveCommittedNoise] Saved noise to {output_path}")
        print(f"  Shape: {shape}, Seed: {committed_noise['seed']}, "
              f"FreeNoise: {committed_noise['freenoise_applied']}")

        return (output_path,)


class NV_LoadCommittedNoiseSlice:
    """
    Load a slice of committed noise for a specific chunk.

    For parallel workers: load only the noise slice needed for your chunk,
    avoiding the need to load the entire noise tensor into memory.

    When using NV_VacePrePassReference, the input latent has prepended reference
    frames. Set prepend_ref_frames to match trim_latent so the noise tensor
    has the correct temporal size for the sampler.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise_path": ("STRING", {
                    "default": "committed_noise.pt",
                    "tooltip": "Path to saved committed noise file"
                }),
                "chunk_start_frame": ("INT", {
                    "default": 0,
                    "min": 0,
                    "tooltip": "Starting VIDEO frame of the chunk"
                }),
                "chunk_frame_count": ("INT", {
                    "default": 81,
                    "min": 1,
                    "tooltip": "Number of VIDEO frames in the chunk"
                }),
            },
            "optional": {
                "prepend_ref_frames": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 20,
                    "tooltip": "Number of LATENT frames to prepend for VACE reference padding. "
                               "Connect trim_latent from NV_VacePrePassReference here. "
                               "These frames get random noise (seeded deterministically) "
                               "and are trimmed after generation."
                }),
            }
        }

    RETURN_TYPES = ("COMMITTED_NOISE",)
    RETURN_NAMES = ("committed_noise",)
    FUNCTION = "load_slice"
    CATEGORY = "NV_Utils/sampling"
    DESCRIPTION = "Load a specific chunk's noise slice from saved committed noise. Use prepend_ref_frames with NV_VacePrePassReference."

    def load_slice(
        self,
        noise_path: str,
        chunk_start_frame: int,
        chunk_frame_count: int,
        prepend_ref_frames: int = 0
    ) -> tuple:
        # Load the committed noise
        data = torch.load(noise_path, weights_only=False)
        noise = data["noise"]

        # Convert video frame index to latent frame index.
        # video_to_latent_frames() converts a frame COUNT: (n-1)//4 + 1
        # For a frame INDEX, the mapping is: index // 4 (Wan's 4:1 temporal stride)
        latent_start = chunk_start_frame // 4
        chunk_latent_count = video_to_latent_frames(chunk_frame_count)
        latent_end = latent_start + chunk_latent_count

        # Slice on temporal dimension (dim=2 for [B, C, T, H, W])
        chunk_noise = noise[:, :, latent_start:latent_end, :, :].clone()

        print(f"[NV_LoadCommittedNoiseSlice] Loaded slice from {noise_path}")
        print(f"  Video frames {chunk_start_frame}-{chunk_start_frame + chunk_frame_count} "
              f"-> Latent frames {latent_start}-{latent_end}")
        print(f"  Slice shape: {list(chunk_noise.shape)}")

        # Prepend reference frame noise if needed (for NV_VacePrePassReference compatibility)
        # The reference positions have mask=0 (preserve) and get trimmed after generation,
        # so the noise content doesn't affect the output — we just need the right shape.
        if prepend_ref_frames > 0:
            seed = data.get("seed", 0)
            ref_noise = torch.randn(
                chunk_noise.shape[0], chunk_noise.shape[1], prepend_ref_frames,
                chunk_noise.shape[3], chunk_noise.shape[4],
                generator=torch.Generator(device='cpu').manual_seed(seed + 1)
            )
            chunk_noise = torch.cat([ref_noise, chunk_noise], dim=2)
            print(f"  Prepended {prepend_ref_frames} reference noise frames -> {list(chunk_noise.shape)}")

        # Return as COMMITTED_NOISE format for compatibility with samplers
        return ({
            "noise": chunk_noise,
            "seed": data.get("seed", 0),
            "freenoise_applied": data.get("freenoise_applied", False),
            "context_length_video": data.get("context_length_video", 81),
            "context_overlap_video": data.get("context_overlap_video", 16),
            "shape": list(chunk_noise.shape),
        },)


# Node registration
NODE_CLASS_MAPPINGS = {
    "NV_CommittedNoise": NV_CommittedNoise,
    "NV_SaveCommittedNoise": NV_SaveCommittedNoise,
    "NV_LoadCommittedNoiseSlice": NV_LoadCommittedNoiseSlice,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_CommittedNoise": "NV Committed Noise",
    "NV_SaveCommittedNoise": "NV Save Committed Noise",
    "NV_LoadCommittedNoiseSlice": "NV Load Committed Noise Slice",
}
