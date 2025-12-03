"""
NV Parallel Chunk Planner

Plans video chunk splits for parallel processing across multiple GPUs or machines.
Exports a JSON configuration file with chunk boundaries and shared sampling parameters.

Use Case:
- Long V2V upscaling workflows that take hours on a single GPU
- Split the video into chunks, process each on a different GPU/machine
- Each worker uses the same seed and parameters for consistent results
- Final stitching applies crossfade blending on decoded pixels (no re-encoding)

Workflow:
1. NV_ParallelChunkPlanner → exports chunk_plan.json
2. NV_ChunkLoader (on each worker) → extracts specific chunk
3. [Existing V2V workflow] → processes chunk
4. Save output video per chunk
5. NV_ChunkStitcher → blends all chunks into final video
"""

import json
import os
import comfy.samplers


class NV_ParallelChunkPlanner:
    """
    Plans chunk splits for parallel video processing.

    Calculates optimal split points with overlap for crossfade blending.
    Exports configuration to JSON for use by multiple workers.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("IMAGE", {
                    "tooltip": "Input video frames [T, H, W, C] - used to get total frame count and resolution"
                }),
                "num_workers": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 16,
                    "tooltip": "Number of GPUs or machines to split work across"
                }),
                "overlap_frames": ("INT", {
                    "default": 32,
                    "min": 8,
                    "max": 128,
                    "tooltip": "Number of frames to overlap between chunks for crossfade blending"
                }),
                "output_json_path": ("STRING", {
                    "default": "chunk_plan.json",
                    "tooltip": "Path to save the chunk plan JSON file"
                }),
            },
            "optional": {
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
            }
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("json_path", "plan_summary",)
    OUTPUT_NODE = True
    FUNCTION = "plan_chunks"
    CATEGORY = "NV_Utils"
    DESCRIPTION = "Plans video chunk splits for parallel processing. Exports JSON config for multiple workers."

    def plan_chunks(self, video, num_workers, overlap_frames, output_json_path,
                    seed=0, steps=20, cfg=5.0, sampler_name="euler", scheduler="sgm_uniform",
                    denoise=0.75, context_window_size=81, context_overlap=16):
        """
        Calculate chunk boundaries and export plan to JSON.
        """

        # Extract video metadata from tensor shape [T, H, W, C]
        total_frames = video.shape[0]
        height = video.shape[1]
        width = video.shape[2]

        print(f"[NV_ParallelChunkPlanner] Video metadata:")
        print(f"  Total frames: {total_frames}")
        print(f"  Resolution: {width}x{height}")

        # Validate inputs
        if overlap_frames >= total_frames // num_workers:
            raise ValueError(
                f"Overlap ({overlap_frames}) is too large for the number of workers ({num_workers}). "
                f"Each chunk would need at least {overlap_frames * 2} frames."
            )

        # Calculate chunk boundaries
        # Each chunk needs: unique_frames + overlap_frames
        # Total unique frames = total_frames - (num_workers - 1) * overlap_frames
        total_overlap = (num_workers - 1) * overlap_frames
        unique_frames_total = total_frames - total_overlap

        if unique_frames_total <= 0:
            raise ValueError(
                f"Too much overlap! Total overlap ({total_overlap}) exceeds available frames ({total_frames}). "
                f"Reduce overlap_frames or num_workers."
            )

        # Distribute unique frames evenly
        base_unique_per_worker = unique_frames_total // num_workers
        extra_frames = unique_frames_total % num_workers

        chunks = []
        current_frame = 0

        for i in range(num_workers):
            # This worker gets base + 1 extra frame if there are remainders
            unique_frames = base_unique_per_worker + (1 if i < extra_frames else 0)

            # First chunk starts at 0, subsequent chunks start overlap_frames before their unique section
            if i == 0:
                start_frame = 0
            else:
                start_frame = current_frame - overlap_frames

            # End frame is start + unique + overlap (except last chunk has no trailing overlap)
            if i < num_workers - 1:
                end_frame = start_frame + unique_frames + overlap_frames
            else:
                end_frame = total_frames  # Last chunk goes to the end

            chunks.append({
                "chunk_idx": i,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "frame_count": end_frame - start_frame,
                # For VACE reference, use same boundaries
                "vace_start": start_frame,
                "vace_end": end_frame,
            })

            # Move to next unique section
            current_frame += unique_frames

        # Build the plan
        plan = {
            "version": "1.0",
            "video_metadata": {
                "total_frames": total_frames,
                "height": height,
                "width": width,
            },
            "num_workers": num_workers,
            "overlap_frames": overlap_frames,
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
                "blend_method": "linear",
            }
        }

        # Ensure output directory exists
        output_dir = os.path.dirname(output_json_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Write JSON file
        with open(output_json_path, 'w') as f:
            json.dump(plan, f, indent=2)

        # Generate summary
        summary_lines = [
            "=" * 50,
            "PARALLEL CHUNK PLAN",
            "=" * 50,
            f"Video: {total_frames} frames @ {width}x{height}",
            f"Workers: {num_workers}",
            f"Overlap: {overlap_frames} frames",
            "",
            "CHUNK ASSIGNMENTS:",
        ]

        for chunk in chunks:
            summary_lines.append(
                f"  Worker {chunk['chunk_idx']}: frames {chunk['start_frame']}-{chunk['end_frame']-1} "
                f"({chunk['frame_count']} frames)"
            )

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
            f"Plan saved to: {output_json_path}",
            "=" * 50,
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
