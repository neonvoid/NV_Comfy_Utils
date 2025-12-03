"""
NV Chunk Loader

Loads a specific chunk from a video based on the plan JSON created by NV_ParallelChunkPlanner.
Each worker (GPU/machine) uses this node with a different chunk_index to process their assigned portion.

Outputs:
- chunk_video: The extracted video frames for this chunk (IMAGE type)
- seed: The shared seed from the plan
- steps: The shared steps from the plan
- cfg: The shared CFG from the plan
- denoise: The shared denoise strength from the plan
- chunk_info: String with chunk metadata for logging

Usage:
1. Copy chunk_plan.json to all worker machines
2. On each worker, set chunk_index to their assigned chunk (0, 1, 2, ...)
3. Connect chunk_video to your VACE/upscaling workflow
4. Use the output parameters (seed, steps, cfg, denoise) in your sampler
"""

import json
import torch


class NV_ChunkLoader:
    """
    Loads a specific video chunk based on the parallel chunk plan.

    Reads the plan JSON and extracts the frame range for the specified chunk.
    Also outputs the shared sampling parameters for consistent results.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("IMAGE", {
                    "tooltip": "Full input video frames [T, H, W, C]"
                }),
                "plan_json_path": ("STRING", {
                    "default": "chunk_plan.json",
                    "tooltip": "Path to the chunk_plan.json file from NV_ParallelChunkPlanner"
                }),
                "chunk_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 15,
                    "tooltip": "Which chunk to load (0-indexed). Each worker uses a different index."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "FLOAT", "FLOAT", "STRING",)
    RETURN_NAMES = ("chunk_video", "seed", "steps", "cfg", "denoise", "chunk_info",)
    FUNCTION = "load_chunk"
    CATEGORY = "NV_Utils"
    DESCRIPTION = "Loads a specific video chunk for parallel processing. Each worker uses a different chunk_index."

    def load_chunk(self, video, plan_json_path, chunk_index):
        """
        Extract the specified chunk from the video.
        """

        # Load the plan
        try:
            with open(plan_json_path, 'r') as f:
                plan = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Chunk plan not found: {plan_json_path}\n"
                f"Run NV_ParallelChunkPlanner first to create the plan."
            )
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in chunk plan: {e}")

        # Validate chunk index
        num_chunks = len(plan.get("chunks", []))
        if chunk_index >= num_chunks:
            raise ValueError(
                f"Invalid chunk_index {chunk_index}. Plan has {num_chunks} chunks (indices 0-{num_chunks-1})."
            )

        # Get this chunk's info
        chunk = plan["chunks"][chunk_index]
        start_frame = chunk["start_frame"]
        end_frame = chunk["end_frame"]

        # Validate video length
        total_video_frames = video.shape[0]
        if end_frame > total_video_frames:
            raise ValueError(
                f"Chunk {chunk_index} requires frames {start_frame}-{end_frame-1}, "
                f"but video only has {total_video_frames} frames.\n"
                f"Make sure you're using the same video that was planned for."
            )

        # Extract chunk frames
        chunk_video = video[start_frame:end_frame].clone()

        # Get shared parameters
        params = plan.get("shared_params", {})
        seed = params.get("seed", 0)
        steps = params.get("steps", 20)
        cfg = params.get("cfg", 5.0)
        denoise = params.get("denoise", 0.75)

        # Build info string
        info_lines = [
            f"Chunk {chunk_index}/{num_chunks-1}",
            f"Frames: {start_frame}-{end_frame-1} ({end_frame - start_frame} total)",
            f"Seed: {seed}",
            f"Steps: {steps}",
            f"CFG: {cfg}",
            f"Denoise: {denoise}",
        ]

        # Add context window info if present
        if "context_window_size" in params:
            info_lines.append(f"Context Window: {params['context_window_size']}")
        if "context_overlap" in params:
            info_lines.append(f"Context Overlap: {params['context_overlap']}")

        chunk_info = "\n".join(info_lines)

        print(f"[NV_ChunkLoader] Loaded chunk {chunk_index}:")
        print(f"  Frames: {start_frame} to {end_frame-1} ({chunk_video.shape[0]} frames)")
        print(f"  Shape: {chunk_video.shape}")

        return (chunk_video, seed, steps, cfg, denoise, chunk_info)


class NV_ChunkLoaderAdvanced:
    """
    Advanced chunk loader that outputs all parameters as separate outputs.

    Use this version when you need access to all sampling parameters
    including sampler_name, scheduler, and context window settings.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("IMAGE", {
                    "tooltip": "Full input video frames [T, H, W, C]"
                }),
                "plan_json_path": ("STRING", {
                    "default": "chunk_plan.json",
                    "tooltip": "Path to the chunk_plan.json file from NV_ParallelChunkPlanner"
                }),
                "chunk_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 15,
                    "tooltip": "Which chunk to load (0-indexed)"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "FLOAT", "FLOAT", "STRING", "STRING", "INT", "INT", "STRING",)
    RETURN_NAMES = ("chunk_video", "seed", "steps", "cfg", "denoise", "sampler_name", "scheduler",
                    "context_window_size", "context_overlap", "chunk_info",)
    FUNCTION = "load_chunk"
    CATEGORY = "NV_Utils"
    DESCRIPTION = "Advanced chunk loader with all sampling parameters as separate outputs."

    def load_chunk(self, video, plan_json_path, chunk_index):
        """
        Extract the specified chunk with all parameters.
        """

        # Load the plan
        with open(plan_json_path, 'r') as f:
            plan = json.load(f)

        # Validate chunk index
        num_chunks = len(plan.get("chunks", []))
        if chunk_index >= num_chunks:
            raise ValueError(f"Invalid chunk_index {chunk_index}. Plan has {num_chunks} chunks.")

        # Get this chunk's info
        chunk = plan["chunks"][chunk_index]
        start_frame = chunk["start_frame"]
        end_frame = chunk["end_frame"]

        # Extract chunk frames
        chunk_video = video[start_frame:end_frame].clone()

        # Get all shared parameters
        params = plan.get("shared_params", {})
        seed = params.get("seed", 0)
        steps = params.get("steps", 20)
        cfg = params.get("cfg", 5.0)
        denoise = params.get("denoise", 0.75)
        sampler_name = params.get("sampler_name", "euler")
        scheduler = params.get("scheduler", "sgm_uniform")
        context_window_size = params.get("context_window_size", 81)
        context_overlap = params.get("context_overlap", 16)

        # Build info string
        chunk_info = (
            f"Chunk {chunk_index}/{num_chunks-1} | "
            f"Frames {start_frame}-{end_frame-1} | "
            f"Seed {seed} | Steps {steps} | CFG {cfg}"
        )

        print(f"[NV_ChunkLoaderAdvanced] Loaded chunk {chunk_index}: frames {start_frame}-{end_frame-1}")

        return (chunk_video, seed, steps, cfg, denoise, sampler_name, scheduler,
                context_window_size, context_overlap, chunk_info)


# Node registration
NODE_CLASS_MAPPINGS = {
    "NV_ChunkLoader": NV_ChunkLoader,
    "NV_ChunkLoaderAdvanced": NV_ChunkLoaderAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_ChunkLoader": "NV Chunk Loader",
    "NV_ChunkLoaderAdvanced": "NV Chunk Loader (Advanced)",
}
