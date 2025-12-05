"""
NV Chunk Stitcher

Stitches processed video chunks back together with crossfade blending.
Loads chunk videos from disk and applies smooth transitions at overlap regions.

Key Features:
- Linear crossfade blending (same algorithm as v1 NV_VideoSampler, but on final pixels)
- NO re-encoding through VAE - pure pixel operation
- Loads chunks from numbered video files (chunk_0.mp4, chunk_1.mp4, etc.)
- Outputs stitched video frames ready for saving

Usage:
1. Process all chunks on separate GPUs/machines
2. Save each output as chunk_0.mp4, chunk_1.mp4, etc. in a shared directory
3. Run NV_ChunkStitcher to combine them
4. Save the final stitched output
"""

import os
import json
import torch
import numpy as np


# ============================================================================
# WAN Frame Alignment Helpers
# ============================================================================
# WAN models require frame counts to satisfy: (frames % 4) == 1
# Valid counts: 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65...
# VHS VideoHelperSuite truncates frames to meet this constraint AFTER loading,
# which can break chunk boundaries if not accounted for.

def is_wan_aligned(frames: int) -> bool:
    """Check if frame count satisfies WAN constraint: (frames % 4) == 1"""
    return (frames % 4) == 1


def nearest_wan_aligned(frames: int) -> int:
    """Return the nearest valid WAN frame count (always rounds down)"""
    if is_wan_aligned(frames):
        return frames
    return ((frames - 1) // 4) * 4 + 1


def validate_wan_alignment(frames: int) -> tuple:
    """
    Check if frame count satisfies WAN constraint.

    Returns:
        (is_valid, nearest_valid, difference)
    """
    is_valid = is_wan_aligned(frames)
    nearest = nearest_wan_aligned(frames)
    diff = frames - nearest
    return (is_valid, nearest, diff)


def calculate_adjusted_overlap(expected_frames: int, actual_frames: int, planned_overlap: int) -> int:
    """
    Calculate adjusted overlap given frame count mismatch.

    If chunks were truncated by WAN alignment, the overlap region may be smaller
    than planned. This calculates the actual usable overlap.
    """
    diff = expected_frames - actual_frames
    adjusted = max(0, planned_overlap - diff)
    return adjusted


def load_video_frames(video_path):
    """
    Load video file and return frames as tensor [T, H, W, C].

    Uses OpenCV for video loading.
    """
    try:
        import cv2
    except ImportError:
        raise ImportError(
            "OpenCV (cv2) is required for video loading.\n"
            "Install with: pip install opencv-python"
        )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB and normalize to [0, 1]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32) / 255.0
        frames.append(frame)

    cap.release()

    if not frames:
        raise ValueError(f"No frames found in video: {video_path}")

    # Stack to tensor [T, H, W, C]
    return torch.from_numpy(np.stack(frames, axis=0))


class NV_ChunkStitcher:
    """
    Stitches video chunks with crossfade blending.

    Loads chunk videos from a directory and applies linear crossfade
    at overlap regions. No VAE re-encoding - pure pixel blending.

    Features:
    - Validates chunk frame counts against plan
    - WAN mode (default ON) checks frame alignment constraints
    - Detailed reporting of any issues detected
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plan_json_path": ("STRING", {
                    "default": "chunk_plan.json",
                    "tooltip": "Path to the chunk_plan.json file"
                }),
                "chunk_directory": ("STRING", {
                    "default": "./chunks",
                    "tooltip": "Directory containing chunk_0.mp4, chunk_1.mp4, etc."
                }),
            },
            "optional": {
                "chunk_prefix": ("STRING", {
                    "default": "chunk_",
                    "tooltip": "Filename prefix for chunk videos (e.g., 'chunk_' for chunk_0.mp4)"
                }),
                "chunk_extension": ("STRING", {
                    "default": ".mp4",
                    "tooltip": "File extension for chunk videos"
                }),
                "crossfade_override": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 128,
                    "tooltip": "Override crossfade frames (-1 = use plan's overlap_frames)"
                }),
                "wan_mode": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "WAN frame alignment awareness. Validates (N%4)==1 constraint and reports issues."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("stitched_video", "stitch_report",)
    FUNCTION = "stitch_chunks"
    CATEGORY = "NV_Utils"
    DESCRIPTION = "Stitches processed video chunks with crossfade blending and WAN alignment validation."

    def stitch_chunks(self, plan_json_path, chunk_directory,
                      chunk_prefix="chunk_", chunk_extension=".mp4",
                      crossfade_override=-1, wan_mode=True):
        """
        Load and stitch all chunks with crossfade blending.
        """

        # Load the plan
        try:
            with open(plan_json_path, 'r') as f:
                plan = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Chunk plan not found: {plan_json_path}")

        num_chunks = len(plan.get("chunks", []))
        blend_config = plan.get("blend_config", {})
        planned_overlap = plan.get("overlap_frames", 32)

        # Determine crossfade frames
        if crossfade_override >= 0:
            crossfade_frames = crossfade_override
        else:
            crossfade_frames = blend_config.get("crossfade_frames", planned_overlap)

        # Build report header
        report_lines = [
            "=" * 60,
            "CHUNK STITCH REPORT",
            "=" * 60,
            f"Plan: {plan_json_path}",
            f"Plan overlap_frames: {planned_overlap}",
            f"Effective crossfade: {crossfade_frames}",
            f"WAN mode: {wan_mode}",
        ]

        if crossfade_override >= 0 and crossfade_override != planned_overlap:
            report_lines.append(
                f"⚠️  WARNING: crossfade_override ({crossfade_override}) != plan overlap ({planned_overlap})"
            )

        print(f"[NV_ChunkStitcher] Loading {num_chunks} chunks from {chunk_directory}")
        print(f"[NV_ChunkStitcher] Crossfade: {crossfade_frames} frames, WAN mode: {wan_mode}")

        report_lines.append("")
        report_lines.append("LOADING CHUNKS:")

        # Load all chunks with validation
        chunks = []
        warnings = []
        for i in range(num_chunks):
            video_path = os.path.join(chunk_directory, f"{chunk_prefix}{i}{chunk_extension}")

            if not os.path.exists(video_path):
                raise FileNotFoundError(
                    f"Chunk video not found: {video_path}\n"
                    f"Make sure all chunks have been processed and saved."
                )

            print(f"  Loading chunk {i}: {video_path}")
            chunk_frames = load_video_frames(video_path)
            chunks.append(chunk_frames)

            # Get expected frame count from plan
            actual_frames = chunk_frames.shape[0]
            expected_frames = plan["chunks"][i].get("frame_count") if i < len(plan.get("chunks", [])) else None

            # Check WAN alignment
            is_aligned, nearest, diff = validate_wan_alignment(actual_frames)

            # Build chunk report line
            chunk_info = f"  Chunk {i}: {actual_frames} frames"
            if expected_frames is not None:
                if actual_frames != expected_frames:
                    chunk_info += f" (expected {expected_frames}, diff={actual_frames - expected_frames})"
                    warnings.append(f"Chunk {i} frame count mismatch: {actual_frames} vs expected {expected_frames}")
                else:
                    chunk_info += " ✓"

            if wan_mode:
                if is_aligned:
                    chunk_info += " [WAN OK]"
                else:
                    chunk_info += f" [WAN: needs {nearest}, off by {diff}]"
                    if actual_frames == expected_frames:  # Only warn if we haven't already
                        warnings.append(f"Chunk {i} not WAN-aligned: {actual_frames} frames")

            report_lines.append(chunk_info)
            print(f"    Shape: {chunk_frames.shape}")

        # Add warnings section
        if warnings:
            report_lines.append("")
            report_lines.append("⚠️  WARNINGS:")
            for w in warnings:
                report_lines.append(f"   • {w}")

        # Stitch chunks with crossfade
        report_lines.append("")
        report_lines.append("STITCHING:")
        print(f"\n[NV_ChunkStitcher] Stitching {num_chunks} chunks...")

        # First chunk: use all frames
        output = chunks[0]
        report_lines.append(f"  Chunk 0: {chunks[0].shape[0]} frames (base)")

        for i in range(1, num_chunks):
            current_chunk = chunks[i]

            if crossfade_frames > 0:
                # Check if we have enough frames for crossfade
                if crossfade_frames >= output.shape[0] or crossfade_frames >= current_chunk.shape[0]:
                    report_lines.append(
                        f"  ⚠️  Chunk {i}: crossfade ({crossfade_frames}) >= chunk size, concatenating instead"
                    )
                    output = torch.cat([output, current_chunk], dim=0)
                    continue

                # Get overlap regions
                prev_end = output[-crossfade_frames:]  # Last N frames of accumulated
                curr_start = current_chunk[:crossfade_frames]  # First N frames of current

                # Validate dimensions match
                if prev_end.shape[1:] != curr_start.shape[1:]:
                    raise ValueError(
                        f"Chunk dimension mismatch at chunk {i}!\n"
                        f"Previous end shape: {prev_end.shape}\n"
                        f"Current start shape: {curr_start.shape}"
                    )

                # Linear crossfade weights: 0 → 1 over crossfade_frames
                weights = torch.linspace(0, 1, crossfade_frames)
                weights = weights.view(-1, 1, 1, 1)  # [T, 1, 1, 1] for broadcasting

                # Blend: (1 - weight) * prev + weight * current
                blended = (1.0 - weights) * prev_end + weights * curr_start

                # Replace overlap region in output and append remaining frames
                output = torch.cat([
                    output[:-crossfade_frames],  # Everything before overlap
                    blended,  # Blended overlap region
                    current_chunk[crossfade_frames:]  # Rest of current chunk
                ], dim=0)

                new_frames = current_chunk.shape[0] - crossfade_frames
                report_lines.append(
                    f"  Chunk {i}: {current_chunk.shape[0]} frames "
                    f"({crossfade_frames} blended, {new_frames} new)"
                )
            else:
                # No crossfade - just concatenate (not recommended)
                output = torch.cat([output, current_chunk], dim=0)
                report_lines.append(
                    f"  Chunk {i}: {current_chunk.shape[0]} frames (no blend)"
                )

        # Clamp to valid range
        output = torch.clamp(output, 0.0, 1.0)

        # Final report
        report_lines.append("")
        report_lines.append("RESULT:")
        report_lines.append(f"  Total output: {output.shape[0]} frames")
        report_lines.append(f"  Resolution: {output.shape[2]}x{output.shape[1]}")

        # Check final output WAN alignment
        if wan_mode:
            final_aligned, final_nearest, final_diff = validate_wan_alignment(output.shape[0])
            if final_aligned:
                report_lines.append(f"  WAN alignment: ✓ OK")
            else:
                report_lines.append(f"  WAN alignment: ⚠️  {output.shape[0]} frames (needs {final_nearest})")

        report_lines.append("=" * 60)

        stitch_report = "\n".join(report_lines)
        print(stitch_report)

        return (output, stitch_report)


class NV_ChunkStitcherFromImages:
    """
    Stitches video chunks that are passed as IMAGE inputs.

    Use this when chunks are already loaded in ComfyUI rather than on disk.
    Supports up to 4 chunk inputs.

    Features:
    - Optional plan JSON validation to catch frame count mismatches
    - WAN mode (default ON) validates frame alignment constraints
    - Detailed reporting of any adjustments made

    IMPORTANT: crossfade_frames MUST match overlap_frames from NV_ParallelChunkPlanner!
    If they don't match, you'll get visible ghosting at chunk boundaries because
    the crossfade will blend non-matching frame content.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "chunk_0": ("IMAGE", {"tooltip": "First video chunk"}),
                "crossfade_frames": ("INT", {
                    "default": 32,
                    "min": 0,
                    "max": 128,
                    "tooltip": "MUST match overlap_frames from NV_ParallelChunkPlanner! Mismatched values cause ghosting."
                }),
            },
            "optional": {
                "chunk_1": ("IMAGE", {"tooltip": "Second video chunk (optional)"}),
                "chunk_2": ("IMAGE", {"tooltip": "Third video chunk (optional)"}),
                "chunk_3": ("IMAGE", {"tooltip": "Fourth video chunk (optional)"}),
                "plan_json_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to chunk_plan.json for validation (optional but recommended)"
                }),
                "wan_mode": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "WAN frame alignment awareness. Validates (N%4)==1 constraint and adjusts overlap if needed."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("stitched_video", "stitch_info",)
    FUNCTION = "stitch_chunks"
    CATEGORY = "NV_Utils"
    DESCRIPTION = "Stitches video chunks from IMAGE inputs with optional plan validation and WAN alignment checks."

    def stitch_chunks(self, chunk_0, crossfade_frames,
                      chunk_1=None, chunk_2=None, chunk_3=None,
                      plan_json_path="", wan_mode=True):
        """
        Stitch chunk IMAGE inputs with crossfade.
        """

        # Collect non-None chunks
        chunks = [chunk_0]
        if chunk_1 is not None:
            chunks.append(chunk_1)
        if chunk_2 is not None:
            chunks.append(chunk_2)
        if chunk_3 is not None:
            chunks.append(chunk_3)

        num_chunks = len(chunks)
        report_lines = [
            "=" * 60,
            "CHUNK STITCH REPORT",
            "=" * 60,
        ]

        # Load plan for validation if provided
        plan = None
        planned_overlap = crossfade_frames
        if plan_json_path and os.path.exists(plan_json_path):
            try:
                with open(plan_json_path, 'r') as f:
                    plan = json.load(f)
                planned_overlap = plan.get("overlap_frames", crossfade_frames)
                report_lines.append(f"Plan loaded: {plan_json_path}")
                report_lines.append(f"Plan overlap_frames: {planned_overlap}")

                # Check if plan chunk count matches
                plan_chunks = plan.get("chunks", [])
                if len(plan_chunks) != num_chunks:
                    report_lines.append(
                        f"⚠️  WARNING: Plan has {len(plan_chunks)} chunks, but {num_chunks} provided!"
                    )
            except Exception as e:
                report_lines.append(f"⚠️  Could not load plan: {e}")

        # Validate crossfade matches plan
        if plan and crossfade_frames != planned_overlap:
            report_lines.append(
                f"⚠️  WARNING: crossfade_frames ({crossfade_frames}) != plan overlap_frames ({planned_overlap})"
            )
            report_lines.append("   This may cause ghosting at chunk boundaries!")

        report_lines.append("")
        report_lines.append("CHUNK ANALYSIS:")

        # Validate chunks and check WAN alignment
        warnings = []
        for i, chunk in enumerate(chunks):
            actual_frames = chunk.shape[0]
            is_aligned, nearest, diff = validate_wan_alignment(actual_frames)

            # Get expected frames from plan if available
            expected_frames = None
            if plan and i < len(plan.get("chunks", [])):
                expected_frames = plan["chunks"][i].get("frame_count")

            # Build chunk report line
            chunk_info = f"  Chunk {i}: {actual_frames} frames"
            if expected_frames is not None:
                if actual_frames != expected_frames:
                    chunk_info += f" (expected {expected_frames}, diff={actual_frames - expected_frames})"
                    warnings.append(f"Chunk {i} frame count mismatch: {actual_frames} vs expected {expected_frames}")
                else:
                    chunk_info += " ✓"

            if wan_mode:
                if is_aligned:
                    chunk_info += " [WAN OK]"
                else:
                    chunk_info += f" [WAN: needs {nearest}, truncated by {diff}]"
                    if not expected_frames:  # Only warn if we didn't already warn about mismatch
                        warnings.append(f"Chunk {i} not WAN-aligned: {actual_frames} frames (nearest valid: {nearest})")

            report_lines.append(chunk_info)

        # Add warnings section
        if warnings:
            report_lines.append("")
            report_lines.append("⚠️  WARNINGS:")
            for w in warnings:
                report_lines.append(f"   • {w}")

        report_lines.append("")
        report_lines.append("STITCHING:")

        print(f"[NV_ChunkStitcherFromImages] Stitching {num_chunks} chunks")
        print(f"  Crossfade: {crossfade_frames} frames")
        print(f"  WAN mode: {wan_mode}")

        # First chunk: use all frames
        output = chunks[0].clone()
        report_lines.append(f"  Chunk 0: {chunks[0].shape[0]} frames (base)")

        for i in range(1, num_chunks):
            current_chunk = chunks[i]
            effective_crossfade = crossfade_frames

            # Check if we have enough frames for crossfade
            if effective_crossfade > 0 and effective_crossfade < min(output.shape[0], current_chunk.shape[0]):
                # Get overlap regions
                prev_end = output[-effective_crossfade:]
                curr_start = current_chunk[:effective_crossfade]

                # Linear crossfade
                weights = torch.linspace(0, 1, effective_crossfade, device=output.device)
                weights = weights.view(-1, 1, 1, 1)

                blended = (1.0 - weights) * prev_end + weights * curr_start

                output = torch.cat([
                    output[:-effective_crossfade],
                    blended,
                    current_chunk[effective_crossfade:]
                ], dim=0)

                new_frames = current_chunk.shape[0] - effective_crossfade
                report_lines.append(
                    f"  Chunk {i}: {current_chunk.shape[0]} frames "
                    f"({effective_crossfade} blended, {new_frames} new)"
                )
            else:
                output = torch.cat([output, current_chunk], dim=0)
                report_lines.append(f"  Chunk {i}: {current_chunk.shape[0]} frames (concat, no blend)")

        output = torch.clamp(output, 0.0, 1.0)

        # Final summary
        report_lines.append("")
        report_lines.append("RESULT:")
        report_lines.append(f"  Total output: {output.shape[0]} frames")
        report_lines.append(f"  Resolution: {output.shape[2]}x{output.shape[1]}")

        # Check final output WAN alignment
        if wan_mode:
            final_aligned, final_nearest, final_diff = validate_wan_alignment(output.shape[0])
            if final_aligned:
                report_lines.append(f"  WAN alignment: ✓ OK")
            else:
                report_lines.append(f"  WAN alignment: ⚠️  {output.shape[0]} frames (needs {final_nearest})")

        report_lines.append("=" * 60)

        stitch_info = "\n".join(report_lines)
        print(stitch_info)

        return (output, stitch_info)


# Node registration
NODE_CLASS_MAPPINGS = {
    "NV_ChunkStitcher": NV_ChunkStitcher,
    "NV_ChunkStitcherFromImages": NV_ChunkStitcherFromImages,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_ChunkStitcher": "NV Chunk Stitcher (From Files)",
    "NV_ChunkStitcherFromImages": "NV Chunk Stitcher (From Images)",
}
