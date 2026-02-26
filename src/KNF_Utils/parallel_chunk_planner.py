"""
NV Parallel Chunk Planner v2.0

Plans video chunk splits for parallel processing across multiple GPUs or machines.
Exports a JSON configuration file with chunk boundaries and shared sampling parameters.

v2.0 additions:
- VRAM-aware auto-computation of max_frames_per_worker (wire a MODEL input)
- Auto-compute overlap from max_frames when overlap_frames=0
- Plan JSON v2.0 schema with latent_stitch_config for downstream nodes
- Per-chunk latent_frames pre-computed

Use Case:
- Long V2V upscaling workflows that take hours on a single GPU
- Split the video into chunks, process each on a different GPU/machine
- Each worker uses the same seed and parameters for consistent results
- Final stitching uses latent-space blending (NV_LatentChunkStitcher)

Workflow:
1. NV_ParallelChunkPlanner → exports chunk_plan.json
2. NV_ChunkLoader (on each worker) → extracts specific chunk
3. [Existing V2V workflow] → processes chunk
4. [NV_SaveChunkLatent] → saves denoised latent per chunk
5. NV_LatentChunkStitcher → blends all chunks in latent space
6. NV_BoundaryNoiseMask → optional boundary refinement

WAN Frame Alignment:
- WAN models require frame counts to satisfy: (frames % 4) == 1
- Valid counts: 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65...
- VHS VideoHelperSuite truncates frames to meet this constraint AFTER loading
- This planner can generate WAN-safe chunks that won't be truncated
"""

import json
import os
import logging
import comfy.samplers
import comfy.model_management

from .chunk_utils import (
    is_wan_aligned,
    nearest_wan_aligned,
    adjust_for_wan_alignment,
    video_to_latent_frames,
    estimate_max_inference_frames,
)

logger = logging.getLogger(__name__)


# ============================================================================
# File Path Helpers
# ============================================================================

def get_unique_filepath(filepath: str) -> str:
    """
    Return a unique filepath by appending _N if file exists.

    chunk_plan.json → chunk_plan_1.json → chunk_plan_2.json → ...
    """
    if not os.path.exists(filepath):
        return filepath

    base, ext = os.path.splitext(filepath)
    counter = 1
    while True:
        new_path = f"{base}_{counter}{ext}"
        if not os.path.exists(new_path):
            return new_path
        counter += 1


class NV_ParallelChunkPlanner:
    """
    Plans chunk splits for parallel video processing.

    Calculates optimal split points with overlap for crossfade blending.
    Exports configuration to JSON for use by multiple workers.

    VRAM-aware mode: wire a MODEL input and set max_frames_per_worker=0
    to auto-compute the maximum chunk size that fits in your GPU's VRAM.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("IMAGE", {
                    "tooltip": "Input video frames [T, H, W, C] - used to get total frame count and resolution"
                }),
                "max_frames_per_worker": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "tooltip": "Maximum frames per chunk. 0 = auto-compute from VRAM (requires MODEL input). "
                               "Worker count is auto-calculated to ensure all chunks fit within this limit."
                }),
                "overlap_frames": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 128,
                    "tooltip": "Overlap in video frames between chunks for blending. "
                               "0 = auto-compute from max_frames (recommended: ~25% of chunk size)."
                }),
                "output_json_path": ("STRING", {
                    "default": "chunk_plan.json",
                    "tooltip": "Path to save the chunk plan JSON file"
                }),
            },
            "optional": {
                # VRAM-aware auto-computation
                "model": ("MODEL", {
                    "tooltip": "WAN model for VRAM-aware auto-computation of max_frames_per_worker. "
                               "Required when max_frames_per_worker=0."
                }),
                "available_vram_gb": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 512.0,
                    "step": 0.1,
                    "tooltip": "VRAM budget in GB. 0 = auto-detect from GPU."
                }),
                "use_vace": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Account for VACE conditioning overhead in VRAM estimation."
                }),
                "use_cfg": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Account for CFG (classifier-free guidance) doubling batch size."
                }),
                # Sampling parameters to share across all workers
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Random seed for reproducible results across workers"
                }),
                "steps": ("INT", {
                    "default": 20,
                    "min": 1,
                    "max": 200,
                    "tooltip": "Number of sampling steps"
                }),
                "cfg": ("FLOAT", {
                    "default": 5.0,
                    "min": 0.0,
                    "max": 30.0,
                    "step": 0.5,
                    "tooltip": "Classifier-free guidance scale"
                }),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {
                    "default": "euler",
                    "tooltip": "Sampler algorithm"
                }),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {
                    "default": "sgm_uniform",
                    "tooltip": "Noise schedule"
                }),
                "denoise": ("FLOAT", {
                    "default": 0.75,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Denoising strength for V2V workflows"
                }),
                # Context window parameters
                "context_window_size": ("INT", {
                    "default": 81,
                    "min": 17,
                    "max": 200,
                    "tooltip": "Context window size in pixel frames"
                }),
                "context_overlap": ("INT", {
                    "default": 16,
                    "min": 4,
                    "max": 64,
                    "tooltip": "Overlap between context windows"
                }),
                "wan_frame_alignment": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Plan chunks to satisfy WAN (frames%4)==1 constraint. Prevents VHS truncation issues."
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("json_path", "plan_summary",)
    OUTPUT_NODE = True
    FUNCTION = "plan_chunks"
    CATEGORY = "NV_Utils"
    DESCRIPTION = (
        "Plans video chunk splits for parallel processing. "
        "Wire a MODEL input and set max_frames=0 for VRAM-aware auto-computation. "
        "Exports JSON config for latent-space stitching pipeline."
    )

    def _auto_compute_max_frames(self, model, height, width, available_vram_bytes,
                                 use_vace, use_cfg):
        """Auto-compute max frames that fit in VRAM using the VRAM estimator."""
        try:
            from .wan_vram_estimator import (
                extract_wan_config,
                detect_attention_backend,
                get_model_dtype,
                estimate_inference_peak,
            )
            from .wan_memory_estimator import get_model_size

            config = extract_wan_config(model)
            if config is None:
                logger.warning("[ChunkPlanner] Non-WAN model detected, using default 81 frames")
                return 81, None

            backend = detect_attention_backend()
            dtype = get_model_dtype(model)
            weight_bytes = get_model_size(model)

            # Build estimator function with all params bound except frame count
            def estimate_fn(pixel_frames):
                breakdown = estimate_inference_peak(
                    config=config,
                    model_weight_bytes=weight_bytes,
                    model_dtype=dtype,
                    total_pixel_frames=pixel_frames,
                    height=height,
                    width=width,
                    backend=backend,
                    use_vace=use_vace,
                    use_cfg=use_cfg,
                    safety_margin_pct=20.0,
                    model=model,
                )
                return breakdown.total

            max_frames = estimate_max_inference_frames(
                estimate_fn=estimate_fn,
                available_vram_bytes=available_vram_bytes,
                low=5,
                high=2001,
            )

            # Get the peak for the chosen frame count for reporting
            peak_breakdown = estimate_inference_peak(
                config=config,
                model_weight_bytes=weight_bytes,
                model_dtype=dtype,
                total_pixel_frames=max_frames,
                height=height,
                width=width,
                backend=backend,
                use_vace=use_vace,
                use_cfg=use_cfg,
                safety_margin_pct=20.0,
                model=model,
            )

            vram_info = {
                "estimated_peak_gb": round(peak_breakdown.total_gb, 2),
                "available_vram_gb": round(available_vram_bytes / (1024 ** 3), 2),
                "headroom_gb": round((available_vram_bytes - peak_breakdown.total) / (1024 ** 3), 2),
                "attention_backend": backend.name,
            }

            logger.info(
                f"[ChunkPlanner] VRAM auto-compute: {max_frames} frames @ {width}x{height}, "
                f"peak={peak_breakdown.total_gb:.1f}GB / {available_vram_bytes / (1024**3):.1f}GB available"
            )
            return max_frames, vram_info

        except Exception as e:
            logger.warning(f"[ChunkPlanner] VRAM auto-compute failed: {e}. Falling back to 81 frames.")
            return 81, None

    def _auto_compute_overlap(self, max_frames_per_worker):
        """Auto-compute overlap from chunk size. ~25% of chunk, aligned to 4."""
        raw = max(16, max_frames_per_worker // 4)
        # Align to multiple of 4 for clean WAN alignment
        aligned = (raw // 4) * 4
        return max(16, aligned)

    def plan_chunks(self, video, max_frames_per_worker, overlap_frames, output_json_path,
                    model=None, available_vram_gb=0.0, use_vace=False, use_cfg=True,
                    seed=0, steps=20, cfg=5.0, sampler_name="euler", scheduler="sgm_uniform",
                    denoise=0.75, context_window_size=81, context_overlap=16,
                    wan_frame_alignment=True):
        """
        Calculate chunk boundaries and export plan to JSON.
        """

        # Extract video metadata from tensor shape [T, H, W, C]
        total_frames = video.shape[0]
        height = video.shape[1]
        width = video.shape[2]

        # Resolve VRAM budget
        if available_vram_gb > 0:
            available_vram_bytes = int(available_vram_gb * 1024 ** 3)
        else:
            # Auto-detect from GPU
            try:
                available_vram_bytes = comfy.model_management.get_free_memory()
            except Exception:
                available_vram_bytes = 0

        # Track sources for plan JSON
        max_frames_source = "manual"
        overlap_source = "manual"
        vram_info = None

        # Auto-compute max_frames from VRAM if set to 0
        if max_frames_per_worker == 0:
            if model is not None:
                max_frames_per_worker, vram_info = self._auto_compute_max_frames(
                    model, height, width, available_vram_bytes, use_vace, use_cfg
                )
                max_frames_source = "vram_auto"
                print(f"[NV_ParallelChunkPlanner] VRAM auto-computed: {max_frames_per_worker} max frames per worker")
            else:
                max_frames_per_worker = 81  # Safe default
                max_frames_source = "default"
                print(f"[NV_ParallelChunkPlanner] No MODEL input — using default {max_frames_per_worker} frames")

        # Auto-compute overlap if set to 0
        if overlap_frames == 0:
            overlap_frames = self._auto_compute_overlap(max_frames_per_worker)
            overlap_source = "auto"
            print(f"[NV_ParallelChunkPlanner] Auto-computed overlap: {overlap_frames} frames")

        # Store original max for JSON output
        original_max_frames = max_frames_per_worker

        # If WAN alignment enabled, snap max_frames to valid WAN size (4n+1)
        if wan_frame_alignment:
            wan_max = nearest_wan_aligned(max_frames_per_worker)  # rounds down
            if wan_max < overlap_frames + 5:  # minimum viable chunk (overlap + at least 5 unique frames)
                raise ValueError(
                    f"max_frames_per_worker ({max_frames_per_worker}) too small for WAN alignment. "
                    f"Need at least {overlap_frames + 5} frames."
                )
            if wan_max != max_frames_per_worker:
                print(f"[NV_ParallelChunkPlanner] Adjusted max_frames {max_frames_per_worker} → {wan_max} for WAN alignment")
            max_frames_per_worker = wan_max

        # Calculate effective unique frames per chunk
        effective_unique = max_frames_per_worker - overlap_frames
        if effective_unique <= 0:
            raise ValueError(
                f"max_frames_per_worker ({max_frames_per_worker}) must be > overlap_frames ({overlap_frames})"
            )

        # Calculate number of workers needed
        if total_frames <= max_frames_per_worker:
            num_workers = 1
        else:
            num_workers = max(1, -(-(total_frames - overlap_frames) // effective_unique))  # Ceiling division

        print(f"[NV_ParallelChunkPlanner] Auto-calculated {num_workers} workers for max_frames_per_worker={max_frames_per_worker}")

        print(f"[NV_ParallelChunkPlanner] Video metadata:")
        print(f"  Total frames: {total_frames}")
        print(f"  Resolution: {width}x{height}")
        print(f"  Num workers: {num_workers}")
        print(f"  WAN frame alignment: {wan_frame_alignment}")

        warnings = []

        # Check if input video is WAN-aligned
        input_wan_aligned = is_wan_aligned(total_frames)
        if wan_frame_alignment and not input_wan_aligned:
            aligned_total = nearest_wan_aligned(total_frames)
            warnings.append(
                f"Input video ({total_frames} frames) is not WAN-aligned. "
                f"Nearest valid: {aligned_total} frames. "
                f"VHS may truncate to {aligned_total} before processing."
            )

        # Generate chunk boundaries
        chunks = []

        for i in range(num_workers):
            if i == 0:
                start_frame = 0
                end_frame = min(max_frames_per_worker, total_frames)
            elif i < num_workers - 1:
                start_frame = chunks[i-1]["end_frame"] - overlap_frames
                end_frame = start_frame + max_frames_per_worker
            else:
                start_frame = chunks[i-1]["end_frame"] - overlap_frames
                end_frame = total_frames

            frame_count = end_frame - start_frame

            # If WAN alignment enabled, enforce aligned chunk sizes
            if wan_frame_alignment and not is_wan_aligned(frame_count):
                aligned_count = nearest_wan_aligned(frame_count)  # rounds down
                if i < num_workers - 1:
                    end_frame = start_frame + aligned_count
                    frame_count = aligned_count
                else:
                    warnings.append(
                        f"Chunk {i} (last): {frame_count} frames is not WAN-aligned. "
                        f"Will be truncated to {aligned_count} by WAN. "
                        f"Consider adjusting input video length."
                    )

            chunk_wan_aligned = is_wan_aligned(frame_count)
            wan_adjusted_count = nearest_wan_aligned(frame_count) if not chunk_wan_aligned else frame_count

            chunks.append({
                "chunk_idx": i,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "frame_count": frame_count,
                "latent_frames": video_to_latent_frames(frame_count),
                "wan_aligned": chunk_wan_aligned,
                "expected_wan_frames": wan_adjusted_count,
                "vace_start": start_frame,
                "vace_end": end_frame,
            })

        # Build the plan (v2.0 schema)
        plan = {
            "version": "2.0",
            "video_metadata": {
                "total_frames": total_frames,
                "height": height,
                "width": width,
                "wan_aligned": input_wan_aligned,
            },
            "num_workers": num_workers,
            "max_frames_per_worker": original_max_frames,
            "effective_max_frames": max_frames_per_worker,
            "max_frames_source": max_frames_source,
            "overlap_frames": overlap_frames,
            "overlap_source": overlap_source,
            "wan_frame_alignment": wan_frame_alignment,
            "stitch_method": "latent",
            "chunks": chunks,
            "shared_params": {
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": denoise,
                "context_window_size": context_window_size,
                "context_overlap": context_overlap,
            },
            "blend_config": {
                "crossfade_frames": overlap_frames,
                "blend_method": "cosine",
            },
            "latent_stitch_config": {
                "overlap_video_frames": overlap_frames,
                "blend_mode": "cosine",
                "recommended_denoise": 0.12,
            },
        }

        # Add VRAM info if auto-computed
        if vram_info is not None:
            plan["vram_info"] = vram_info

        # Add warnings to plan if any
        if warnings:
            plan["warnings"] = warnings

        # Ensure output directory exists
        output_dir = os.path.dirname(output_json_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Get unique filepath (don't overwrite existing files)
        output_json_path = get_unique_filepath(output_json_path)

        # Write JSON file
        with open(output_json_path, 'w') as f:
            json.dump(plan, f, indent=2)

        # Generate summary
        summary_lines = [
            "=" * 60,
            "PARALLEL CHUNK PLAN v2.0",
            "=" * 60,
            f"Video: {total_frames} frames @ {width}x{height}",
        ]

        # Show max frames info
        if max_frames_source == "vram_auto":
            summary_lines.append(f"Max frames per worker: {max_frames_per_worker} (VRAM auto-computed)")
            if vram_info:
                summary_lines.append(
                    f"  VRAM: {vram_info['estimated_peak_gb']:.1f}GB peak / "
                    f"{vram_info['available_vram_gb']:.1f}GB available "
                    f"({vram_info['headroom_gb']:.1f}GB headroom, {vram_info['attention_backend']})"
                )
        elif wan_frame_alignment and original_max_frames != max_frames_per_worker:
            summary_lines.append(f"Max frames per worker: {original_max_frames} → {max_frames_per_worker} (WAN-aligned)")
        else:
            summary_lines.append(f"Max frames per worker: {max_frames_per_worker}")

        if overlap_source == "auto":
            summary_lines.append(f"Overlap: {overlap_frames} frames (auto-computed)")
        else:
            summary_lines.append(f"Overlap: {overlap_frames} frames")

        summary_lines.extend([
            f"Workers (auto-calculated): {num_workers}",
            f"Stitch method: latent",
            f"WAN alignment mode: {wan_frame_alignment}",
        ])

        if wan_frame_alignment:
            if input_wan_aligned:
                summary_lines.append(f"Input WAN status: OK ({total_frames} % 4 == 1)")
            else:
                summary_lines.append(f"Input WAN status: {total_frames} frames (needs {nearest_wan_aligned(total_frames)})")

        summary_lines.append("")
        summary_lines.append("CHUNK ASSIGNMENTS:")

        for chunk in chunks:
            chunk_line = (
                f"  Worker {chunk['chunk_idx']}: frames {chunk['start_frame']}-{chunk['end_frame']-1} "
                f"({chunk['frame_count']} frames, {chunk['latent_frames']} latent)"
            )
            if wan_frame_alignment:
                if chunk['wan_aligned']:
                    chunk_line += " [WAN OK]"
                else:
                    chunk_line += f" [WAN -> {chunk['expected_wan_frames']}]"
            summary_lines.append(chunk_line)

        # Add warnings if any
        if warnings:
            summary_lines.append("")
            summary_lines.append("WARNINGS:")
            for w in warnings:
                summary_lines.append(f"   - {w}")

        summary_lines.extend([
            "",
            "SHARED PARAMETERS:",
            f"  Seed: {seed}",
            f"  Steps: {steps}",
            f"  CFG: {cfg}",
            f"  Sampler: {sampler_name}",
            f"  Scheduler: {scheduler}",
            f"  Denoise: {denoise}",
            "",
            "LATENT STITCH CONFIG:",
            f"  Overlap: {overlap_frames} video frames",
            f"  Blend mode: cosine",
            f"  Recommended boundary denoise: 0.12",
            "",
            f"Plan saved to: {output_json_path}",
            "=" * 60,
        ])

        summary = "\n".join(summary_lines)
        print(summary)

        return {"ui": {"text": [summary]}, "result": (output_json_path, summary)}


# Node registration
NODE_CLASS_MAPPINGS = {
    "NV_ParallelChunkPlanner": NV_ParallelChunkPlanner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_ParallelChunkPlanner": "NV Parallel Chunk Planner",
}
