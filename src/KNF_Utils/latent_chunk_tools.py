"""
Latent-Space Chunk Stitching Tools

Saves per-chunk denoised latents to disk and stitches them with latent-space
crossfade blending. Replaces pixel-space blending (Hann/Hamming after VAE decode)
with latent-space blending (before VAE decode).

Key insight: pixel blend averages two decoded images -> ghosting at boundaries.
Latent blend gives the VAE decoder one blended input -> one coherent image.
Same principle as latent interpolation in image generation.

Usage:
    Per-chunk workflow (sweep loader):
        [KSampler] -> [NV_SaveChunkLatent] -> chunk_N.pt

    Stitch workflow (run once after all chunks):
        [NV_LatentChunkStitcher] -> stitched latent -> [NV_StreamingVAEDecode]

    Boundary refinement (optional):
        [NV_LatentChunkStitcher] -> [NV_BoundaryNoiseMask] -> [KSampler denoise=0.10-0.20]

v2.0: plan_json_path auto-reads overlap from chunk plan; boundary mask auto-computes
      context/ramp from overlap size.
"""

import os
import re
import json
import torch
import comfy.model_management

from .chunk_utils import (
    compute_blend_weights,
    compute_ramp_weights,
    compute_latent_overlap,
    video_to_latent_frames,
)


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

    v2.0: Wire plan_json_path from the chunk planner to auto-read overlap
    and blend mode. No manual overlap entry needed.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "chunk_directory": ("STRING", {
                    "default": "chunk_latents",
                    "tooltip": "Directory containing chunk_N.pt files from NV_SaveChunkLatent."
                }),
                "prefix": ("STRING", {
                    "default": "chunk",
                    "tooltip": "Filename prefix to match. Loads {prefix}_0.pt, {prefix}_1.pt, etc."
                }),
            },
            "optional": {
                "plan_json_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to chunk_plan.json from NV_ParallelChunkPlanner. "
                               "Auto-reads overlap_video_frames and blend_mode from plan. "
                               "Overrides manual settings when provided."
                }),
                "overlap_video_frames": ("INT", {
                    "default": 0, "min": 0, "max": 256, "step": 1,
                    "tooltip": "Overlap in VIDEO frames between adjacent chunks. "
                               "0 = read from plan_json_path. "
                               "Must match the chunk planner overlap setting."
                }),
                "blend_mode": (["cosine", "linear", "hamming"], {
                    "default": "cosine",
                    "tooltip": "Blend weight curve. Cosine: smooth S-curve (recommended). "
                               "Linear: straight ramp. "
                               "Hamming: S-curve with ~0.08 floor."
                }),
                "overlap_latent_adjust": ("INT", {
                    "default": 0, "min": -4, "max": 4, "step": 1,
                    "tooltip": "Manual adjustment to computed latent overlap. The auto-computed "
                               "overlap accounts for Wan's first-frame encoding asymmetry, but "
                               "may be off by +/-1 for edge cases. Positive = wider blend zone."
                }),
            },
        }

    RETURN_TYPES = ("LATENT", "INT", "STRING")
    RETURN_NAMES = ("latent", "num_chunks", "stitch_info")
    FUNCTION = "stitch"
    CATEGORY = "NV_Utils/latent"
    DESCRIPTION = (
        "Load chunk latents from disk and stitch with latent-space crossfade blending. "
        "Wire plan_json_path from the chunk planner for auto-config, or set overlap manually."
    )

    def stitch(self, chunk_directory, prefix, plan_json_path="",
               overlap_video_frames=0, blend_mode="cosine", overlap_latent_adjust=0):

        # === Step 0: Read plan JSON if provided ===
        if plan_json_path and os.path.isfile(plan_json_path):
            with open(plan_json_path, 'r') as f:
                plan = json.load(f)
            print(f"[LatentChunkStitcher] Loaded plan from {plan_json_path} (v{plan.get('version', '?')})")

            # Auto-read overlap from plan
            if overlap_video_frames <= 0:
                # v2.0 plans have latent_stitch_config
                stitch_cfg = plan.get("latent_stitch_config", {})
                if stitch_cfg.get("overlap_video_frames", 0) > 0:
                    overlap_video_frames = stitch_cfg["overlap_video_frames"]
                    print(f"  Auto-read overlap: {overlap_video_frames} video frames (from latent_stitch_config)")
                elif plan.get("overlap_frames", 0) > 0:
                    # v1.x fallback
                    overlap_video_frames = plan["overlap_frames"]
                    print(f"  Auto-read overlap: {overlap_video_frames} video frames (from overlap_frames)")

            # Auto-read blend mode from plan
            stitch_cfg = plan.get("latent_stitch_config", {})
            if stitch_cfg.get("blend_mode"):
                blend_mode = stitch_cfg["blend_mode"]
                print(f"  Auto-read blend_mode: {blend_mode}")
        elif plan_json_path:
            print(f"[LatentChunkStitcher] Warning: plan_json_path '{plan_json_path}' not found, using manual settings")

        # Validate we have overlap
        if overlap_video_frames <= 0:
            print("[LatentChunkStitcher] Warning: overlap_video_frames=0. "
                  "Wire plan_json_path or set overlap manually for proper blending.")

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
        # Precise per-boundary formula (via compute_latent_overlap):
        #   combined_video = video_A + video_B - overlap_video
        #   combined_latent = (combined_video - 1) // 4 + 1
        #   overlap_latent = latent_A + latent_B - combined_latent

        boundaries = []
        result = chunks[0]

        for i in range(1, num_chunks):
            chunk = chunks[i]

            # Compute running video frame count for accumulated result
            if i == 1:
                video_a = chunk_video_frames_list[0]
            else:
                video_a = sum(chunk_video_frames_list[:i]) - (i - 1) * overlap_video_frames
            video_b = chunk_video_frames_list[i]

            # Use shared utility for precise overlap computation
            overlap_latent = compute_latent_overlap(
                video_a, video_b, overlap_video_frames, adjust=overlap_latent_adjust
            )

            # Additional clamping against actual tensor sizes
            latent_a = result.shape[2]
            latent_b = chunk.shape[2]
            overlap_latent = max(1, min(overlap_latent, latent_a, latent_b))

            # Compute blend weights for this boundary
            w = compute_blend_weights(overlap_latent, blend_mode)
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
                "boundary": f"{i-1}<->{i}",
                "position_latent": boundary_pos,
                "overlap_latent": overlap_latent,
                "video_a": video_a,
                "video_b": video_b,
                "result_frames_after": result.shape[2],
            })

            adj_str = f" (adjust={overlap_latent_adjust:+d})" if overlap_latent_adjust != 0 else ""
            print(f"  Boundary {i-1}<->{i}: overlap={overlap_latent} latent frames{adj_str}, "
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


class NV_BoundaryNoiseMask:
    """
    Create a 5D noise_mask for boundary refinement of stitched latents.

    Takes stitch_info JSON from NV_LatentChunkStitcher and creates a soft-ramp
    noise_mask that targets chunk boundary regions for re-diffusion. Clean frames
    surrounding boundaries are preserved (mask=0) and act as implicit conditioning
    through the model's self-attention -- all tokens are processed regardless of
    mask value.

    The mask is attached directly to the latent dict (NOT via SetLatentNoiseMask,
    which flattens 5D temporal structure). Output goes straight to KSampler.

    auto_compute mode: derives context_frames, ramp_frames, core_extra from
    the overlap size in stitch_info. No manual parameter tuning needed.

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
            },
            "optional": {
                "auto_compute": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Auto-compute context, ramp, and core_extra from overlap size. "
                               "When True, manual values below are ignored."
                }),
                "context_frames": ("INT", {
                    "default": 4, "min": 0, "max": 16, "step": 1,
                    "tooltip": "Clean frames on each side of boundary (mask=0.0). "
                               "Preserved exactly. Anchor spatial layout through attention. "
                               "Ignored when auto_compute=True."
                }),
                "ramp_frames": ("INT", {
                    "default": 4, "min": 0, "max": 16, "step": 1,
                    "tooltip": "Soft ramp frames on each side. Gradual transition from "
                               "preserved to fully re-diffused. Prevents hard mask edges. "
                               "Ignored when auto_compute=True."
                }),
                "core_extra": ("INT", {
                    "default": 0, "min": 0, "max": 16, "step": 1,
                    "tooltip": "Extra fully-masked frames beyond the overlap region on each "
                               "side. 0 = core is exactly the overlap zone from stitching. "
                               "Ignored when auto_compute=True."
                }),
                "ramp_curve": (["cosine", "linear"], {
                    "default": "cosine",
                    "tooltip": "Ramp weight curve. Cosine: smooth S-curve (recommended). "
                               "Linear: straight ramp."
                }),
            },
        }

    RETURN_TYPES = ("LATENT", "FLOAT", "STRING")
    RETURN_NAMES = ("latent", "recommended_denoise", "mask_info")
    FUNCTION = "create_mask"
    CATEGORY = "NV_Utils/latent"
    DESCRIPTION = (
        "Create boundary refinement noise_mask from stitch_info. With auto_compute=True, "
        "derives context/ramp from overlap size automatically. Outputs recommended_denoise "
        "(0.12) for the KSampler. Output goes directly to KSampler (do NOT use SetLatentNoiseMask)."
    )

    # Sweet spot from research: 0.10-0.20 range, 0.12 optimal
    RECOMMENDED_DENOISE = 0.12

    def create_mask(self, latent, stitch_info, auto_compute=True,
                    context_frames=4, ramp_frames=4, core_extra=0,
                    ramp_curve="cosine"):
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
            return (out, self.RECOMMENDED_DENOISE, mask_info)

        # Auto-compute parameters from overlap size
        if auto_compute and boundaries:
            # Use the average overlap across all boundaries
            avg_overlap = sum(b["overlap_latent"] for b in boundaries) // len(boundaries)
            context_frames = max(2, avg_overlap // 3)
            ramp_frames = max(2, avg_overlap // 3)
            core_extra = 0
            print(f"[BoundaryNoiseMask] Auto-computed from overlap={avg_overlap}: "
                  f"context={context_frames}, ramp={ramp_frames}, core_extra={core_extra}")

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
                    ramp_w = compute_ramp_weights(ramp_left_len, ramp_curve)
                    # ramp_w goes 0->1, and we want to ramp up toward the core
                    for j in range(ramp_left_len):
                        idx = ramp_left_start + j
                        mask[idx] = max(mask[idx].item(), ramp_w[j].item())

            # Ramp on the right side (after core)
            if ramp_frames > 0:
                ramp_right_end = min(total_t, core_end + ramp_frames)
                ramp_right_len = ramp_right_end - core_end
                if ramp_right_len > 0:
                    ramp_w = compute_ramp_weights(ramp_right_len, ramp_curve)
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
              f"core_extra={core_extra}, curve={ramp_curve}"
              f"{' (auto-computed)' if auto_compute else ''}")
        print(f"  Recommended denoise: {self.RECOMMENDED_DENOISE}")
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
            "auto_compute": auto_compute,
            "recommended_denoise": self.RECOMMENDED_DENOISE,
            "config": {
                "context_frames": context_frames,
                "ramp_frames": ramp_frames,
                "core_extra": core_extra,
                "ramp_curve": ramp_curve,
            },
            "boundaries": mask_details,
            "mask_values": [round(mask[i].item(), 3) for i in range(total_t)],
        }, indent=2)

        return (out, self.RECOMMENDED_DENOISE, mask_info)


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
