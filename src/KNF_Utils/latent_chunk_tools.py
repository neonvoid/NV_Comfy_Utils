"""
Latent-Space Chunk Stitching Tools

Saves per-chunk denoised latents to disk and stitches them with latent-space
crossfade blending. Replaces pixel-space blending (Hann/Hamming after VAE decode)
with latent-space blending (before VAE decode).

Key insight: pixel blend averages two decoded images → ghosting at boundaries.
Latent blend gives the VAE decoder one blended input → one coherent image.
Same principle as latent interpolation in image generation.

Usage:
    Per-chunk workflow (sweep loader):
        [KSampler] → [NV_SaveChunkLatent] → chunk_N.pt

    Stitch workflow (run once after all chunks):
        [NV_LatentChunkStitcher] → stitched latent → [NV_StreamingVAEDecode]
"""

import os
import re
import math
import json
import torch
import comfy.model_management


class NV_SaveChunkLatent:
    """
    Save a chunk's denoised latent to disk with a predictable indexed filename.

    Wire chunk_index from the sweep loader to get chunk_0.pt, chunk_1.pt, etc.
    The stitcher loads these by index to blend in latent space.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "directory": ("STRING", {
                    "default": "chunk_latents",
                    "tooltip": "Directory to save chunk latent files. Created if it doesn't exist."
                }),
                "chunk_index": ("INT", {
                    "default": 0, "min": 0, "max": 1000, "step": 1,
                    "tooltip": "Chunk index for filename. Wire from sweep_loader chunk index."
                }),
                "prefix": ("STRING", {
                    "default": "chunk",
                    "tooltip": "Filename prefix. Files saved as {prefix}_{index}.pt"
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "save"
    CATEGORY = "NV_Utils/latent"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Save chunk latent to disk with indexed filename for latent-space stitching. "
        "Wire chunk_index from sweep_loader."
    )

    def save(self, latent, directory, chunk_index, prefix):
        os.makedirs(directory, exist_ok=True)

        samples = latent["samples"]  # [B, C, T, H, W]
        filename = f"{prefix}_{chunk_index}.pt"
        filepath = os.path.join(directory, filename)

        data = {
            "samples": samples.cpu().half(),
            "chunk_index": chunk_index,
            "shape": list(samples.shape),
        }
        torch.save(data, filepath)

        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"[SaveChunkLatent] Saved {filename}: shape {list(samples.shape)}, "
              f"{file_size_mb:.1f} MB")

        return (filepath,)


class NV_LatentChunkStitcher:
    """
    Load all chunk latents from a directory and blend overlapping regions
    in latent space. Output a single stitched latent for streaming VAE decode.

    Replaces pixel-space chunk stitching with latent-space blending.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "chunk_directory": ("STRING", {
                    "default": "chunk_latents",
                    "tooltip": "Directory containing chunk_N.pt files from NV_SaveChunkLatent."
                }),
                "overlap_video_frames": ("INT", {
                    "default": 32, "min": 0, "max": 256, "step": 1,
                    "tooltip": "Overlap in VIDEO frames between adjacent chunks. "
                               "Must match your pixel-space stitcher / chunk planner overlap. "
                               "Converted to latent frames internally using Wan's temporal compression."
                }),
                "blend_mode": (["linear", "cosine"],  {
                    "default": "linear",
                    "tooltip": "Blend weight curve. Linear: straight ramp. "
                               "Cosine: smooth S-curve (same as Hann)."
                }),
                "prefix": ("STRING", {
                    "default": "chunk",
                    "tooltip": "Filename prefix to match. Loads {prefix}_0.pt, {prefix}_1.pt, etc."
                }),
            },
        }

    RETURN_TYPES = ("LATENT", "INT", "STRING")
    RETURN_NAMES = ("latent", "num_chunks", "stitch_info")
    FUNCTION = "stitch"
    CATEGORY = "NV_Utils/latent"
    DESCRIPTION = (
        "Load chunk latents from disk and stitch with latent-space crossfade blending. "
        "Output a single stitched latent for streaming VAE decode."
    )

    def stitch(self, chunk_directory, overlap_video_frames, blend_mode, prefix):
        # === Step 1: Discover and load chunk files ===
        pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)\.pt$")
        files = []
        for f in os.listdir(chunk_directory):
            m = pattern.match(f)
            if m:
                files.append((int(m.group(1)), f))

        if not files:
            raise FileNotFoundError(
                f"No chunk files matching '{prefix}_*.pt' found in {chunk_directory}"
            )

        files.sort(key=lambda x: x[0])
        print(f"[LatentChunkStitcher] Found {len(files)} chunk files: "
              f"{[f[1] for f in files]}")

        # Load all chunks
        chunks = []
        for idx, filename in files:
            filepath = os.path.join(chunk_directory, filename)
            data = torch.load(filepath, map_location="cpu", weights_only=False)
            samples = data["samples"].float()  # Upcast from fp16 to fp32
            chunks.append(samples)
            print(f"  {filename}: shape {list(samples.shape)}, "
                  f"chunk_index={data.get('chunk_index', idx)}")

        num_chunks = len(chunks)

        # === Step 2: Validate spatial dimensions ===
        ref_shape = chunks[0].shape
        for i, chunk in enumerate(chunks[1:], 1):
            if chunk.shape[3:] != ref_shape[3:]:
                raise ValueError(
                    f"Chunk {i} spatial dims {chunk.shape[3:]} don't match "
                    f"chunk 0 spatial dims {ref_shape[3:]}. "
                    f"All chunks must be encoded at the same resolution."
                )

        # === Step 3: Compute overlap in latent frames ===
        if overlap_video_frames <= 0:
            # No overlap — simple concatenation
            result = torch.cat(chunks, dim=2)
            print(f"[LatentChunkStitcher] No overlap — concatenated {num_chunks} chunks: "
                  f"{list(result.shape)}")
            out = {"samples": result.to(comfy.model_management.intermediate_device())}
            info = json.dumps({"num_chunks": num_chunks, "overlap": 0, "boundaries": []})
            return (out, num_chunks, info)

        # Wan VAE temporal compression: latent = (video - 1) // 4 + 1
        # For the overlap region, we compute from both sides and use the min.
        #
        # From the END of chunk N: the last `overlap_video_frames` video frames
        # span some number of latent frames from the chunk's end.
        #
        # From the START of chunk N+1: the first `overlap_video_frames` video frames
        # span some latent frames from the chunk's start.
        #
        # These can differ by 1 due to Wan's asymmetric first-frame encoding
        # (frame 0 → latent 0 alone; subsequent groups of 4 → 1 latent each).
        #
        # overlap_from_end: for a chunk of T_video frames, last overlap_video_frames
        #   occupy latent frames from T_latent - end_overlap_latent to T_latent.
        #   end_overlap_latent depends on chunk length.
        #
        # overlap_from_start: first overlap_video_frames of a chunk.
        #   start_overlap_latent = (overlap_video_frames - 1) // 4 + 1
        #
        # Use: min(end, start) as the blend count to be safe.
        overlap_from_start = (overlap_video_frames - 1) // 4 + 1

        # For the end: if chunk has T_video frames, the unique region has
        # T_video - overlap_video_frames frames. Unique latent count:
        # unique_latent = (unique_video - 1) // 4 + 1
        # overlap_from_end = T_latent - unique_latent
        # But we don't know T_video directly from the latent. Use approximate:
        overlap_latent = overlap_from_start  # Best general approximation

        print(f"[LatentChunkStitcher] Overlap: {overlap_video_frames} video frames "
              f"≈ {overlap_latent} latent frames")

        # === Step 4: Blend and stitch ===
        # Compute blend weights
        w = _compute_blend_weights(overlap_latent, blend_mode)
        # Reshape for broadcasting: [1, 1, overlap_latent, 1, 1]
        w = w.reshape(1, 1, -1, 1, 1)

        boundaries = []
        result = chunks[0]
        output_frames = result.shape[2]

        for i in range(1, num_chunks):
            chunk = chunks[i]

            # Clamp overlap to not exceed either chunk's temporal length
            actual_overlap = min(overlap_latent, result.shape[2], chunk.shape[2])
            if actual_overlap != overlap_latent:
                print(f"  Boundary {i-1}↔{i}: clamped overlap from {overlap_latent} "
                      f"to {actual_overlap} (chunk too short)")
                w_actual = _compute_blend_weights(actual_overlap, blend_mode)
                w_actual = w_actual.reshape(1, 1, -1, 1, 1)
            else:
                w_actual = w

            # Extract overlap regions
            result_overlap = result[:, :, -actual_overlap:, :, :]
            chunk_overlap = chunk[:, :, :actual_overlap, :, :]

            # Blend
            blended = (1.0 - w_actual) * result_overlap + w_actual * chunk_overlap

            # Stitch: result_unique + blended + chunk_unique
            result_unique = result[:, :, :-actual_overlap, :, :]
            chunk_unique = chunk[:, :, actual_overlap:, :, :]

            boundary_pos = result_unique.shape[2]
            result = torch.cat([result_unique, blended, chunk_unique], dim=2)

            boundaries.append({
                "boundary": f"{i-1}↔{i}",
                "position_latent": boundary_pos,
                "overlap_latent": actual_overlap,
                "result_frames_after": result.shape[2],
            })

            print(f"  Boundary {i-1}↔{i}: blended {actual_overlap} frames at position {boundary_pos}, "
                  f"result now {result.shape[2]} frames")

        # === Step 5: Output ===
        print(f"[LatentChunkStitcher] Final stitched shape: {list(result.shape)}")
        print(f"  {num_chunks} chunks, {len(boundaries)} boundaries, "
              f"blend_mode={blend_mode}")

        out_latent = {"samples": result.to(comfy.model_management.intermediate_device())}
        info = json.dumps({
            "num_chunks": num_chunks,
            "overlap_video_frames": overlap_video_frames,
            "overlap_latent_frames": overlap_latent,
            "blend_mode": blend_mode,
            "boundaries": boundaries,
            "final_shape": list(result.shape),
        }, indent=2)

        return (out_latent, num_chunks, info)


def _compute_blend_weights(num_frames, mode):
    """Compute blend weights from 0 (keep chunk A) to 1 (keep chunk B)."""
    if num_frames <= 1:
        return torch.ones(max(1, num_frames))

    t = torch.linspace(0, 1, num_frames)

    if mode == "linear":
        return t
    elif mode == "cosine":
        return 0.5 * (1.0 - torch.cos(math.pi * t))
    else:
        return t  # Fallback to linear


# Node registration
NODE_CLASS_MAPPINGS = {
    "NV_SaveChunkLatent": NV_SaveChunkLatent,
    "NV_LatentChunkStitcher": NV_LatentChunkStitcher,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_SaveChunkLatent": "NV Save Chunk Latent",
    "NV_LatentChunkStitcher": "NV Latent Chunk Stitcher",
}
