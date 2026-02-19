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

    Boundary refinement (Phase 2, optional):
        [NV_LatentChunkStitcher] → [NV_BoundaryNoiseMask] → [KSampler low denoise] → [VAE Decode]
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
            "optional": {
                "chunk_video_frames": ("INT", {
                    "default": 0, "min": 0, "max": 100000, "step": 1,
                    "tooltip": "Number of video frames in this chunk. Optional but improves "
                               "overlap precision in the stitcher. Wire from sweep_loader."
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

    def save(self, latent, directory, chunk_index, prefix, chunk_video_frames=0):
        os.makedirs(directory, exist_ok=True)

        samples = latent["samples"]  # [B, C, T, H, W]
        filename = f"{prefix}_{chunk_index}.pt"
        filepath = os.path.join(directory, filename)

        data = {
            "samples": samples.cpu().half(),
            "chunk_index": chunk_index,
            "shape": list(samples.shape),
        }
        if chunk_video_frames > 0:
            data["chunk_video_frames"] = chunk_video_frames
        torch.save(data, filepath)

        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        vid_info = f", video_frames={chunk_video_frames}" if chunk_video_frames > 0 else ""
        print(f"[SaveChunkLatent] Saved {filename}: shape {list(samples.shape)}, "
              f"{file_size_mb:.1f} MB{vid_info}")

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
                "overlap_latent_adjust": ("INT", {
                    "default": 0, "min": -4, "max": 4, "step": 1,
                    "tooltip": "Manual adjustment to computed latent overlap. The auto-computed "
                               "overlap accounts for Wan's first-frame encoding asymmetry, but "
                               "may be off by ±1 for edge cases. Positive = wider blend zone."
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

    def stitch(self, chunk_directory, overlap_video_frames, blend_mode, prefix,
               overlap_latent_adjust=0):
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

        # Load all chunks + metadata
        chunks = []
        chunk_video_frames_list = []
        for idx, filename in files:
            filepath = os.path.join(chunk_directory, filename)
            data = torch.load(filepath, map_location="cpu", weights_only=False)
            samples = data["samples"].float()  # Upcast from fp16 to fp32
            chunks.append(samples)
            # Get video frame count if saved, otherwise estimate from latent
            vid_frames = data.get("chunk_video_frames", 0)
            if vid_frames <= 0:
                vid_frames = (samples.shape[2] - 1) * 4 + 1  # Estimated
            chunk_video_frames_list.append(vid_frames)
            print(f"  {filename}: shape {list(samples.shape)}, "
                  f"chunk_index={data.get('chunk_index', idx)}, "
                  f"video_frames={vid_frames}"
                  f"{'(est)' if 'chunk_video_frames' not in data else ''}")

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

        # === Step 3: No overlap — simple concatenation ===
        if overlap_video_frames <= 0:
            result = torch.cat(chunks, dim=2)
            print(f"[LatentChunkStitcher] No overlap — concatenated {num_chunks} chunks: "
                  f"{list(result.shape)}")
            out = {"samples": result.to(comfy.model_management.intermediate_device())}
            info = json.dumps({"num_chunks": num_chunks, "overlap": 0, "boundaries": []})
            return (out, num_chunks, info)

        # === Step 4: Blend and stitch with per-boundary overlap ===
        #
        # Wan VAE temporal compression: latent = (video - 1) // 4 + 1
        # Each independently-encoded chunk has a "first-frame bonus" — frame 0 gets
        # its own latent slot. This means two chunks overlapping by N video frames
        # do NOT overlap by exactly (N-1)//4+1 latent frames. The second chunk's
        # first-frame bonus adds ~1 extra latent frame.
        #
        # Precise per-boundary formula:
        #   combined_video = video_A + video_B - overlap_video
        #   combined_latent = (combined_video - 1) // 4 + 1
        #   overlap_latent = latent_A + latent_B - combined_latent
        #
        # This accounts for the first-frame asymmetry automatically.

        boundaries = []
        result = chunks[0]

        for i in range(1, num_chunks):
            chunk = chunks[i]

            # Compute per-boundary overlap from chunk video frame counts
            latent_a = result.shape[2]  # Accumulated result (acts as "chunk A")
            latent_b = chunk.shape[2]
            # For accumulated result, use the running video frame count
            # For the first boundary, this is just chunk 0's video frames
            # For subsequent boundaries, we track the accumulated total
            if i == 1:
                video_a = chunk_video_frames_list[0]
            else:
                # After previous stitches, accumulated video = sum of unique portions
                video_a = sum(chunk_video_frames_list[:i]) - (i - 1) * overlap_video_frames
            video_b = chunk_video_frames_list[i]

            combined_video = video_a + video_b - overlap_video_frames
            combined_latent = (combined_video - 1) // 4 + 1
            overlap_latent = latent_a + latent_b - combined_latent + overlap_latent_adjust

            # Clamp to valid range
            overlap_latent = max(1, min(overlap_latent, latent_a, latent_b))

            # Compute blend weights for this boundary
            w = _compute_blend_weights(overlap_latent, blend_mode)
            w = w.reshape(1, 1, -1, 1, 1)

            # Extract overlap regions
            result_overlap = result[:, :, -overlap_latent:, :, :]
            chunk_overlap = chunk[:, :, :overlap_latent, :, :]

            # Blend
            blended = (1.0 - w) * result_overlap + w * chunk_overlap

            # Stitch: result_unique + blended + chunk_unique
            result_unique = result[:, :, :-overlap_latent, :, :]
            chunk_unique = chunk[:, :, overlap_latent:, :, :]

            boundary_pos = result_unique.shape[2]
            result = torch.cat([result_unique, blended, chunk_unique], dim=2)

            boundaries.append({
                "boundary": f"{i-1}↔{i}",
                "position_latent": boundary_pos,
                "overlap_latent": overlap_latent,
                "video_a": video_a,
                "video_b": video_b,
                "result_frames_after": result.shape[2],
            })

            adj_str = f" (adjust={overlap_latent_adjust:+d})" if overlap_latent_adjust != 0 else ""
            print(f"  Boundary {i-1}↔{i}: overlap={overlap_latent} latent frames{adj_str}, "
                  f"blended at position {boundary_pos}, "
                  f"result now {result.shape[2]} frames")

        # === Step 5: Output ===
        print(f"[LatentChunkStitcher] Final stitched shape: {list(result.shape)}")
        print(f"  {num_chunks} chunks, {len(boundaries)} boundaries, "
              f"blend_mode={blend_mode}")
        if overlap_latent_adjust != 0:
            print(f"  overlap_latent_adjust={overlap_latent_adjust:+d}")

        out_latent = {"samples": result.to(comfy.model_management.intermediate_device())}
        info = json.dumps({
            "num_chunks": num_chunks,
            "overlap_video_frames": overlap_video_frames,
            "overlap_latent_adjust": overlap_latent_adjust,
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


class NV_BoundaryNoiseMask:
    """
    Create a 5D noise_mask for boundary refinement of stitched latents.

    Takes stitch_info JSON from NV_LatentChunkStitcher and creates a soft-ramp
    noise_mask that targets chunk boundary regions for re-diffusion. Clean frames
    surrounding boundaries are preserved (mask=0) and act as implicit conditioning
    through the model's self-attention — all tokens are processed regardless of
    mask value.

    The mask is attached directly to the latent dict (NOT via SetLatentNoiseMask,
    which flattens 5D temporal structure). Output goes straight to KSampler.

    Pipeline:
        [NV_LatentChunkStitcher] -> [NV_BoundaryNoiseMask] -> [KSampler denoise=0.10-0.20]
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "stitch_info": ("STRING", {
                    "multiline": True,
                    "tooltip": "JSON from NV_LatentChunkStitcher stitch_info output. "
                               "Contains boundary positions and overlap sizes."
                }),
                "context_frames": ("INT", {
                    "default": 4, "min": 0, "max": 16, "step": 1,
                    "tooltip": "Clean frames on each side of boundary (mask=0.0). "
                               "Preserved exactly. Anchor spatial layout through attention."
                }),
                "ramp_frames": ("INT", {
                    "default": 4, "min": 0, "max": 16, "step": 1,
                    "tooltip": "Soft ramp frames on each side. Gradual transition from "
                               "preserved to fully re-diffused. Prevents hard mask edges."
                }),
                "core_extra": ("INT", {
                    "default": 0, "min": 0, "max": 16, "step": 1,
                    "tooltip": "Extra fully-masked frames beyond the overlap region on each "
                               "side. 0 = core is exactly the overlap zone from stitching."
                }),
                "ramp_curve": (["cosine", "linear"], {
                    "default": "cosine",
                    "tooltip": "Ramp weight curve. Cosine: smooth S-curve (recommended). "
                               "Linear: straight ramp."
                }),
            },
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "mask_info")
    FUNCTION = "create_mask"
    CATEGORY = "NV_Utils/latent"
    DESCRIPTION = (
        "Create boundary refinement noise_mask from stitch_info. Soft ramp targets "
        "chunk boundaries for re-diffusion while preserving surrounding context. "
        "Output goes directly to KSampler (do NOT use SetLatentNoiseMask)."
    )

    def create_mask(self, latent, stitch_info, context_frames, ramp_frames,
                    core_extra, ramp_curve):
        samples = latent["samples"]  # [B, C, T, H, W]
        total_t = samples.shape[2]

        # Parse stitch_info
        info = json.loads(stitch_info)
        boundaries = info.get("boundaries", [])

        if not boundaries:
            print("[BoundaryNoiseMask] No boundaries in stitch_info — returning all-zero mask (no refinement)")
            out = latent.copy()
            out["noise_mask"] = torch.zeros(1, 1, total_t, 1, 1)
            mask_info = json.dumps({"boundaries": [], "total_frames": total_t,
                                    "masked_frames": 0}, indent=2)
            return (out, mask_info)

        # Build mask: 0.0 = preserve, 1.0 = fully re-diffuse
        mask = torch.zeros(total_t)
        mask_details = []

        for b in boundaries:
            pos = b["position_latent"]    # Start of blended region in stitched latent
            overlap = b["overlap_latent"]  # Width of blended region

            # Core region: the blended overlap zone + optional extra on each side
            core_start = pos - core_extra
            core_end = pos + overlap + core_extra
            core_start = max(0, core_start)
            core_end = min(total_t, core_end)

            # Set core to 1.0
            mask[core_start:core_end] = 1.0

            # Ramp on the left side (before core)
            if ramp_frames > 0:
                ramp_left_start = max(0, core_start - ramp_frames)
                ramp_left_len = core_start - ramp_left_start
                if ramp_left_len > 0:
                    ramp_w = _compute_ramp(ramp_left_len, ramp_curve)
                    # ramp_w goes 0->1, and we want to ramp up toward the core
                    for j in range(ramp_left_len):
                        idx = ramp_left_start + j
                        mask[idx] = max(mask[idx].item(), ramp_w[j].item())

            # Ramp on the right side (after core)
            if ramp_frames > 0:
                ramp_right_end = min(total_t, core_end + ramp_frames)
                ramp_right_len = ramp_right_end - core_end
                if ramp_right_len > 0:
                    ramp_w = _compute_ramp(ramp_right_len, ramp_curve)
                    # ramp_w goes 0->1, flip it so it ramps down from core
                    ramp_w = ramp_w.flip(0)
                    for j in range(ramp_right_len):
                        idx = core_end + j
                        mask[idx] = max(mask[idx].item(), ramp_w[j].item())

            # Context frames are already 0.0 (default) — no action needed

            # Track for diagnostics
            full_left = max(0, core_start - ramp_frames) - context_frames
            full_right = min(total_t, core_end + ramp_frames) + context_frames
            mask_details.append({
                "boundary": b.get("boundary", "?"),
                "core_range": [int(core_start), int(core_end)],
                "core_width": int(core_end - core_start),
                "left_ramp": [int(max(0, core_start - ramp_frames)), int(core_start)],
                "right_ramp": [int(core_end), int(min(total_t, core_end + ramp_frames))],
                "context_zone": [int(max(0, full_left)), int(min(total_t, full_right))],
            })

        # Reshape to 5D: [1, 1, T, 1, 1] — broadcasts over B, C, H, W
        mask_5d = mask.reshape(1, 1, total_t, 1, 1)

        # Count masked frames for diagnostics
        masked_any = (mask > 0).sum().item()
        masked_full = (mask >= 0.99).sum().item()

        print(f"[BoundaryNoiseMask] Created mask for {len(boundaries)} boundaries")
        print(f"  Total: {total_t} frames, {masked_full} fully masked, "
              f"{masked_any - masked_full} partially masked, "
              f"{total_t - masked_any} preserved")
        print(f"  Config: context={context_frames}, ramp={ramp_frames}, "
              f"core_extra={core_extra}, curve={ramp_curve}")
        for d in mask_details:
            print(f"  {d['boundary']}: core [{d['core_range'][0]}:{d['core_range'][1]}] "
                  f"({d['core_width']} frames), "
                  f"ramps [{d['left_ramp'][0]}:{d['left_ramp'][1]}] / "
                  f"[{d['right_ramp'][0]}:{d['right_ramp'][1]}]")

        # Attach directly to latent dict (skip SetLatentNoiseMask — it flattens 5D)
        out = latent.copy()
        out["noise_mask"] = mask_5d

        mask_info = json.dumps({
            "total_frames": total_t,
            "masked_full": masked_full,
            "masked_partial": masked_any - masked_full,
            "preserved": total_t - masked_any,
            "config": {
                "context_frames": context_frames,
                "ramp_frames": ramp_frames,
                "core_extra": core_extra,
                "ramp_curve": ramp_curve,
            },
            "boundaries": mask_details,
            "mask_values": [round(mask[i].item(), 3) for i in range(total_t)],
        }, indent=2)

        return (out, mask_info)


def _compute_ramp(num_frames, mode):
    """Compute ramp weights with exclusive endpoints (no 0.0 or 1.0).

    For 4 frames: returns ~[0.1, 0.3, 0.7, 0.9] (cosine) or [0.2, 0.4, 0.6, 0.8] (linear).
    This ensures every ramp frame has a truly intermediate value — the 0.0 and 1.0
    values belong to the context and core zones respectively.
    """
    if num_frames <= 0:
        return torch.tensor([])
    if num_frames == 1:
        return torch.tensor([0.5])

    # Exclusive endpoints: skip the 0.0 and 1.0 at ends
    t = torch.linspace(0, 1, num_frames + 2)[1:-1]
    if mode == "cosine":
        return 0.5 * (1.0 - torch.cos(math.pi * t))
    else:
        return t


# Node registration
NODE_CLASS_MAPPINGS = {
    "NV_SaveChunkLatent": NV_SaveChunkLatent,
    "NV_LatentChunkStitcher": NV_LatentChunkStitcher,
    "NV_BoundaryNoiseMask": NV_BoundaryNoiseMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_SaveChunkLatent": "NV Save Chunk Latent",
    "NV_LatentChunkStitcher": "NV Latent Chunk Stitcher",
    "NV_BoundaryNoiseMask": "NV Boundary Noise Mask",
}
