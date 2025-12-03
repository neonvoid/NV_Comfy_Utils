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
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("stitched_video", "stitch_report",)
    FUNCTION = "stitch_chunks"
    CATEGORY = "NV_Utils"
    DESCRIPTION = "Stitches processed video chunks with crossfade blending. No VAE re-encoding."

    def stitch_chunks(self, plan_json_path, chunk_directory,
                      chunk_prefix="chunk_", chunk_extension=".mp4",
                      crossfade_override=-1):
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

        # Determine crossfade frames
        if crossfade_override >= 0:
            crossfade_frames = crossfade_override
        else:
            crossfade_frames = blend_config.get("crossfade_frames", plan.get("overlap_frames", 32))

        print(f"[NV_ChunkStitcher] Loading {num_chunks} chunks from {chunk_directory}")
        print(f"[NV_ChunkStitcher] Crossfade: {crossfade_frames} frames")

        # Load all chunks
        chunks = []
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
            print(f"    Shape: {chunk_frames.shape}")

        # Stitch chunks with crossfade
        print(f"\n[NV_ChunkStitcher] Stitching {num_chunks} chunks...")

        # First chunk: use all frames
        output = chunks[0]
        report_lines = [
            "=" * 50,
            "CHUNK STITCH REPORT",
            "=" * 50,
            f"Chunk 0: {chunks[0].shape[0]} frames (used all)",
        ]

        for i in range(1, num_chunks):
            current_chunk = chunks[i]

            if crossfade_frames > 0:
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
                    f"Chunk {i}: {current_chunk.shape[0]} frames "
                    f"({crossfade_frames} blended, {new_frames} new)"
                )
            else:
                # No crossfade - just concatenate (not recommended)
                output = torch.cat([output, current_chunk], dim=0)
                report_lines.append(
                    f"Chunk {i}: {current_chunk.shape[0]} frames (no blend)"
                )

        # Clamp to valid range
        output = torch.clamp(output, 0.0, 1.0)

        # Final report
        report_lines.extend([
            "",
            f"Total output: {output.shape[0]} frames",
            f"Resolution: {output.shape[2]}x{output.shape[1]}",
            "=" * 50,
        ])

        stitch_report = "\n".join(report_lines)
        print(stitch_report)

        return (output, stitch_report)


class NV_ChunkStitcherFromImages:
    """
    Stitches video chunks that are passed as IMAGE inputs.

    Use this when chunks are already loaded in ComfyUI rather than on disk.
    Supports up to 4 chunk inputs.
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
                    "tooltip": "Number of frames to crossfade between chunks"
                }),
            },
            "optional": {
                "chunk_1": ("IMAGE", {"tooltip": "Second video chunk (optional)"}),
                "chunk_2": ("IMAGE", {"tooltip": "Third video chunk (optional)"}),
                "chunk_3": ("IMAGE", {"tooltip": "Fourth video chunk (optional)"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("stitched_video", "stitch_info",)
    FUNCTION = "stitch_chunks"
    CATEGORY = "NV_Utils"
    DESCRIPTION = "Stitches video chunks from IMAGE inputs with crossfade blending."

    def stitch_chunks(self, chunk_0, crossfade_frames,
                      chunk_1=None, chunk_2=None, chunk_3=None):
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

        print(f"[NV_ChunkStitcherFromImages] Stitching {len(chunks)} chunks")
        print(f"  Crossfade: {crossfade_frames} frames")

        # First chunk: use all frames
        output = chunks[0].clone()
        info_parts = [f"Chunk 0: {chunks[0].shape[0]} frames"]

        for i in range(1, len(chunks)):
            current_chunk = chunks[i]

            if crossfade_frames > 0 and crossfade_frames < min(output.shape[0], current_chunk.shape[0]):
                # Get overlap regions
                prev_end = output[-crossfade_frames:]
                curr_start = current_chunk[:crossfade_frames]

                # Linear crossfade
                weights = torch.linspace(0, 1, crossfade_frames, device=output.device)
                weights = weights.view(-1, 1, 1, 1)

                blended = (1.0 - weights) * prev_end + weights * curr_start

                output = torch.cat([
                    output[:-crossfade_frames],
                    blended,
                    current_chunk[crossfade_frames:]
                ], dim=0)

                info_parts.append(
                    f"Chunk {i}: {current_chunk.shape[0]} frames ({crossfade_frames} blended)"
                )
            else:
                output = torch.cat([output, current_chunk], dim=0)
                info_parts.append(f"Chunk {i}: {current_chunk.shape[0]} frames (concat)")

        output = torch.clamp(output, 0.0, 1.0)

        stitch_info = f"Stitched {len(chunks)} chunks → {output.shape[0]} frames\n" + "\n".join(info_parts)
        print(f"[NV_ChunkStitcherFromImages] Output: {output.shape}")

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
