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

        # Validate video dimensions match plan
        total_video_frames = video.shape[0]
        video_height = video.shape[1]
        video_width = video.shape[2]

        # Check against video_metadata if present (new format)
        video_metadata = plan.get("video_metadata", {})
        expected_frames = video_metadata.get("total_frames", plan.get("total_frames"))
        expected_height = video_metadata.get("height")
        expected_width = video_metadata.get("width")

        if expected_frames and total_video_frames != expected_frames:
            raise ValueError(
                f"Video frame count mismatch! Plan expects {expected_frames} frames, "
                f"but input video has {total_video_frames} frames.\n"
                f"Make sure you're using the same video that was planned for."
            )

        if expected_height and expected_width:
            if video_height != expected_height or video_width != expected_width:
                raise ValueError(
                    f"Video resolution mismatch! Plan expects {expected_width}x{expected_height}, "
                    f"but input video is {video_width}x{video_height}.\n"
                    f"Make sure you're using the same video that was planned for."
                )

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

        # Validate video dimensions match plan
        total_video_frames = video.shape[0]
        video_height = video.shape[1]
        video_width = video.shape[2]

        video_metadata = plan.get("video_metadata", {})
        expected_frames = video_metadata.get("total_frames", plan.get("total_frames"))
        expected_height = video_metadata.get("height")
        expected_width = video_metadata.get("width")

        if expected_frames and total_video_frames != expected_frames:
            raise ValueError(
                f"Video frame count mismatch! Plan expects {expected_frames} frames, "
                f"but input video has {total_video_frames} frames."
            )

        if expected_height and expected_width:
            if video_height != expected_height or video_width != expected_width:
                raise ValueError(
                    f"Video resolution mismatch! Plan expects {expected_width}x{expected_height}, "
                    f"but input video is {video_width}x{video_height}."
                )

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


class NV_ChunkLoaderVACE:
    """
    Chunk loader with VACE control video support.

    Slices main video AND control video(s) using the same frame range.
    This ensures VACE conditioning is built from chunk-sized control videos,
    which is required for parallel VACE processing across multiple machines.

    Use Case:
    - V2V upscaling with VACE (depth, edge, pose, etc.)
    - Each machine processes a different chunk
    - Control videos must be sliced in sync with main video
    - Long VACE Patcher handles context windows within each chunk
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
            "optional": {
                # Flexible control inputs - connect as many as needed
                "control_1": ("IMAGE", {
                    "tooltip": "First control video (depth, edge, etc.) - will be sliced in sync"
                }),
                "control_2": ("IMAGE", {
                    "tooltip": "Second control video (optional) - will be sliced in sync"
                }),
                "control_3": ("IMAGE", {
                    "tooltip": "Third control video (optional) - will be sliced in sync"
                }),
                "control_4": ("IMAGE", {
                    "tooltip": "Fourth control video (optional) - will be sliced in sync"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE",
                    "INT", "INT", "INT", "FLOAT", "FLOAT", "STRING", "STRING", "INT", "INT", "STRING",)
    RETURN_NAMES = ("chunk_video", "chunk_ctrl_1", "chunk_ctrl_2", "chunk_ctrl_3", "chunk_ctrl_4",
                    "chunk_index", "seed", "steps", "cfg", "denoise", "sampler_name", "scheduler",
                    "context_window_size", "context_overlap", "chunk_info",)
    FUNCTION = "load_chunk"
    CATEGORY = "NV_Utils"
    DESCRIPTION = "Loads video chunk AND control videos for VACE parallel processing. Controls are sliced in sync."

    def load_chunk(self, video, plan_json_path, chunk_index,
                   control_1=None, control_2=None, control_3=None, control_4=None):
        """
        Extract the specified chunk from video and all control videos.
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

        # Validate video dimensions match plan
        total_video_frames = video.shape[0]
        video_metadata = plan.get("video_metadata", {})
        expected_frames = video_metadata.get("total_frames", plan.get("total_frames"))

        if expected_frames and total_video_frames != expected_frames:
            raise ValueError(
                f"Video frame count mismatch! Plan expects {expected_frames} frames, "
                f"but input video has {total_video_frames} frames.\n"
                f"Make sure you're using the same video that was planned for."
            )

        if end_frame > total_video_frames:
            raise ValueError(
                f"Chunk {chunk_index} requires frames {start_frame}-{end_frame-1}, "
                f"but video only has {total_video_frames} frames.\n"
                f"Make sure you're using the same video that was planned for."
            )

        # Extract chunk from main video
        chunk_video = video[start_frame:end_frame].clone()

        # Extract chunks from control videos (same frame range)
        controls = [control_1, control_2, control_3, control_4]
        chunk_controls = []

        for i, ctrl in enumerate(controls):
            if ctrl is not None:
                # Validate control video has same frame count as main video
                if ctrl.shape[0] != total_video_frames:
                    raise ValueError(
                        f"Control video {i+1} has {ctrl.shape[0]} frames, "
                        f"but main video has {total_video_frames} frames. "
                        f"Control videos must match main video frame count."
                    )
                chunk_controls.append(ctrl[start_frame:end_frame].clone())
            else:
                chunk_controls.append(None)

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
        num_controls = sum(1 for c in chunk_controls if c is not None)
        info_lines = [
            f"Chunk {chunk_index}/{num_chunks-1}",
            f"Frames: {start_frame}-{end_frame-1} ({end_frame - start_frame} total)",
            f"Controls: {num_controls} sliced",
            f"Seed: {seed}",
            f"Steps: {steps}",
            f"CFG: {cfg}",
            f"Denoise: {denoise}",
            f"Sampler: {sampler_name}",
            f"Scheduler: {scheduler}",
            f"Context Window: {context_window_size}",
            f"Context Overlap: {context_overlap}",
        ]

        chunk_info = "\n".join(info_lines)

        print(f"[NV_ChunkLoaderVACE] Loaded chunk {chunk_index}:")
        print(f"  Frames: {start_frame} to {end_frame-1} ({chunk_video.shape[0]} frames)")
        print(f"  Controls sliced: {num_controls}")

        return (chunk_video, chunk_controls[0], chunk_controls[1], chunk_controls[2], chunk_controls[3],
                chunk_index, seed, steps, cfg, denoise, sampler_name, scheduler,
                context_window_size, context_overlap, chunk_info)


# Node registration
NODE_CLASS_MAPPINGS = {
    "NV_ChunkLoader": NV_ChunkLoader,
    "NV_ChunkLoaderAdvanced": NV_ChunkLoaderAdvanced,
    "NV_ChunkLoaderVACE": NV_ChunkLoaderVACE,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_ChunkLoader": "NV Chunk Loader",
    "NV_ChunkLoaderAdvanced": "NV Chunk Loader (Advanced)",
    "NV_ChunkLoaderVACE": "NV Chunk Loader (VACE)",
}
