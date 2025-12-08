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

WAN Frame Alignment:
- WAN models require frame counts to satisfy: (frames % 4) == 1
- Valid counts: 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65...
- VHS VideoHelperSuite truncates frames to meet this constraint AFTER loading
- This planner can generate WAN-safe chunks that won't be truncated
"""

import json
import os
import comfy.samplers


# ============================================================================
# WAN Frame Alignment Helpers
# ============================================================================

def is_wan_aligned(frames: int) -> bool:
    """Check if frame count satisfies WAN constraint: (frames % 4) == 1"""
    return (frames % 4) == 1


def nearest_wan_aligned(frames: int, round_up: bool = False) -> int:
    """
    Return the nearest valid WAN frame count.

    Args:
        frames: Input frame count
        round_up: If True, round up to next valid count. If False (default), round down.
    """
    if is_wan_aligned(frames):
        return frames
    if round_up:
        return ((frames - 1) // 4 + 1) * 4 + 1
    return ((frames - 1) // 4) * 4 + 1


def adjust_for_wan_alignment(frame_count: int, min_frames: int = 5) -> int:
    """
    Adjust frame count to be WAN-aligned.

    Rounds down to nearest valid count, but ensures at least min_frames.
    """
    aligned = nearest_wan_aligned(frame_count)
    return max(aligned, min_frames)


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
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("IMAGE", {
                    "tooltip": "Input video frames [T, H, W, C] - used to get total frame count and resolution"
                }),
                "num_workers": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 16,
                    "tooltip": "Number of workers (0 = use max_frames_per_worker instead)"
                }),
                "max_frames_per_worker": ("INT", {
                    "default": 300,
                    "min": 0,
                    "max": 10000,
                    "tooltip": "Max frames per chunk (0 = use num_workers instead). Automatically calculates worker count."
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
    DESCRIPTION = "Plans video chunk splits for parallel processing. Exports JSON config for multiple workers."

    def plan_chunks(self, video, num_workers, max_frames_per_worker, overlap_frames, output_json_path,
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

        # Determine num_workers from inputs
        if max_frames_per_worker > 0:
            # Calculate workers needed for max frames per worker
            # Account for overlap: each chunk has unique_frames + overlap, except last chunk
            # effective_unique = max_frames - overlap (what each chunk contributes uniquely)
            effective_unique = max_frames_per_worker - overlap_frames
            if effective_unique <= 0:
                raise ValueError(
                    f"max_frames_per_worker ({max_frames_per_worker}) must be > overlap_frames ({overlap_frames})"
                )
            # Calculate how many workers we need
            # total_frames = unique_per_worker * num_workers + (num_workers - 1) * overlap... simplified:
            num_workers = max(1, -(-total_frames // effective_unique))  # Ceiling division
            print(f"[NV_ParallelChunkPlanner] Auto-calculated {num_workers} workers from max_frames_per_worker={max_frames_per_worker}")
        elif num_workers > 0:
            # Use explicit num_workers
            pass
        else:
            raise ValueError("Either num_workers or max_frames_per_worker must be > 0")

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

        for i in range(num_workers):
            # This worker gets base + 1 extra frame if there are remainders
            unique_frames = base_unique_per_worker + (1 if i < extra_frames else 0)

            # Calculate chunk boundaries
            if i == 0:
                # First chunk starts at 0
                start_frame = 0
                end_frame = unique_frames + overlap_frames
            elif i < num_workers - 1:
                # Middle chunks: start where previous chunk's overlap begins
                start_frame = chunks[i-1]["end_frame"] - overlap_frames
                end_frame = start_frame + unique_frames + overlap_frames
            else:
                # Last chunk: starts at overlap, ends at total_frames
                start_frame = chunks[i-1]["end_frame"] - overlap_frames
                end_frame = total_frames

            frame_count = end_frame - start_frame

            # Check WAN alignment for this chunk
            chunk_wan_aligned = is_wan_aligned(frame_count)
            wan_adjusted_count = nearest_wan_aligned(frame_count) if wan_frame_alignment else frame_count

            if wan_frame_alignment and not chunk_wan_aligned:
                warnings.append(
                    f"Chunk {i}: {frame_count} frames is not WAN-aligned. "
                    f"May be truncated to {wan_adjusted_count} by VHS."
                )

            chunks.append({
                "chunk_idx": i,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "frame_count": frame_count,
                "wan_aligned": chunk_wan_aligned,
                "expected_wan_frames": wan_adjusted_count,
                # For VACE reference, use same boundaries
                "vace_start": start_frame,
                "vace_end": end_frame,
            })

        # Build the plan
        plan = {
            "version": "1.1",
            "video_metadata": {
                "total_frames": total_frames,
                "height": height,
                "width": width,
                "wan_aligned": input_wan_aligned,
            },
            "num_workers": num_workers,
            "overlap_frames": overlap_frames,
            "wan_frame_alignment": wan_frame_alignment,
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
            "PARALLEL CHUNK PLAN",
            "=" * 60,
            f"Video: {total_frames} frames @ {width}x{height}",
            f"Workers: {num_workers}",
            f"Overlap: {overlap_frames} frames",
            f"WAN alignment mode: {wan_frame_alignment}",
        ]

        if wan_frame_alignment:
            if input_wan_aligned:
                summary_lines.append(f"Input WAN status: ✓ OK ({total_frames} % 4 == 1)")
            else:
                summary_lines.append(f"Input WAN status: ⚠️  {total_frames} frames (needs {nearest_wan_aligned(total_frames)})")

        summary_lines.append("")
        summary_lines.append("CHUNK ASSIGNMENTS:")

        for chunk in chunks:
            chunk_line = (
                f"  Worker {chunk['chunk_idx']}: frames {chunk['start_frame']}-{chunk['end_frame']-1} "
                f"({chunk['frame_count']} frames)"
            )
            if wan_frame_alignment:
                if chunk['wan_aligned']:
                    chunk_line += " [WAN ✓]"
                else:
                    chunk_line += f" [WAN ⚠️  → {chunk['expected_wan_frames']}]"
            summary_lines.append(chunk_line)

        # Add warnings if any
        if warnings:
            summary_lines.append("")
            summary_lines.append("⚠️  WARNINGS:")
            for w in warnings:
                summary_lines.append(f"   • {w}")

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
