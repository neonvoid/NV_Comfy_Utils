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
import math
import torch

from .latent_constants import NV_CASCADED_CONFIG_KEY, LATENT_SAFE_KEYS


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


class NV_ChunkPlanReader:
    """
    Reads chunk plan JSON and outputs metadata WITHOUT loading video.

    Use this when you just need the frame ranges, parameters, or other metadata
    from a chunk plan - e.g., for workflow orchestration, passing to video loaders
    that handle their own slicing, or for debugging/logging.

    No video input required - just reads the JSON file.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plan_json_path": ("STRING", {
                    "default": "chunk_plan.json",
                    "tooltip": "Path to the chunk_plan.json file from NV_ParallelChunkPlanner"
                }),
                "chunk_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 99,
                    "tooltip": "Which chunk to read (0-indexed)"
                }),
            },
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "INT", "INT",
                    "INT", "INT", "FLOAT", "FLOAT", "STRING", "STRING", "INT", "INT",
                    "INT", "INT", "INT", "INT", "STRING",)
    RETURN_NAMES = ("start_frame", "end_frame", "frame_count", "vace_start", "vace_end", "chunk_index",
                    "seed", "steps", "cfg", "denoise", "sampler_name", "scheduler", "context_window_size", "context_overlap",
                    "total_frames", "num_chunks", "video_width", "video_height", "chunk_info",)
    FUNCTION = "read_plan"
    CATEGORY = "NV_Utils"
    DESCRIPTION = "Reads chunk plan JSON metadata without loading video. Use for orchestration or passing to external loaders."

    def read_plan(self, plan_json_path, chunk_index):
        """
        Read chunk plan and return metadata for the specified chunk.
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
        chunks = plan.get("chunks", [])
        num_chunks = len(chunks)
        if chunk_index >= num_chunks:
            raise ValueError(
                f"Invalid chunk_index {chunk_index}. Plan has {num_chunks} chunks (indices 0-{num_chunks-1})."
            )

        # Get this chunk's info
        chunk = chunks[chunk_index]
        start_frame = chunk["start_frame"]
        end_frame = chunk["end_frame"]
        frame_count = chunk.get("frame_count", end_frame - start_frame)
        vace_start = chunk.get("vace_start", start_frame)
        vace_end = chunk.get("vace_end", end_frame)

        # Get video metadata
        video_metadata = plan.get("video_metadata", {})
        total_frames = video_metadata.get("total_frames", plan.get("total_frames", 0))
        video_width = video_metadata.get("width", 0)
        video_height = video_metadata.get("height", 0)

        # Get shared parameters
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
        info_lines = [
            f"Chunk {chunk_index}/{num_chunks-1}",
            f"Frames: {start_frame}-{end_frame-1} ({frame_count} total)",
            f"VACE range: {vace_start}-{vace_end}",
            f"Video: {video_width}x{video_height}, {total_frames} frames",
            f"Seed: {seed} | Steps: {steps} | CFG: {cfg} | Denoise: {denoise}",
            f"Sampler: {sampler_name} | Scheduler: {scheduler}",
            f"Context: {context_window_size} window, {context_overlap} overlap",
        ]
        chunk_info = "\n".join(info_lines)

        print(f"[NV_ChunkPlanReader] Read chunk {chunk_index}/{num_chunks-1}:")
        print(f"  Frames: {start_frame} to {end_frame-1} ({frame_count} frames)")
        print(f"  VACE: {vace_start} to {vace_end}")

        return (start_frame, end_frame, frame_count, vace_start, vace_end, chunk_index,
                seed, steps, cfg, denoise, sampler_name, scheduler, context_window_size, context_overlap,
                total_frames, num_chunks, video_width, video_height, chunk_info)


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
                "latent": ("LATENT", {
                    "tooltip": "Pre-pass latent (full video). Temporally sliced to chunk range using Wan VAE frame mapping. "
                               "Use with LatentUpscale to bypass VAE encode of upscaled pixels."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE",
                    "INT", "INT", "INT", "INT", "INT", "FLOAT", "FLOAT", "STRING", "STRING", "INT", "INT", "STRING",
                    "LATENT",
                    "FLOAT", "INT", "INT", "STRING",)
    RETURN_NAMES = ("chunk_video", "chunk_ctrl_1", "chunk_ctrl_2", "chunk_ctrl_3", "chunk_ctrl_4",
                    "chunk_index", "start_frame", "frame_count", "seed", "steps", "cfg", "denoise", "sampler_name", "scheduler",
                    "context_window_size", "context_overlap", "chunk_info",
                    "chunk_latent",
                    "cascade_shift", "cascade_expanded_steps", "cascade_start_at_step", "cascade_add_noise",)
    FUNCTION = "load_chunk"
    CATEGORY = "NV_Utils"
    DESCRIPTION = ("Loads video chunk AND control videos for VACE parallel processing. Controls are sliced in sync. "
                   "Optional latent input for latent-space upscale path. "
                   "If plan contains cascaded_config (from NV_PreNoiseLatent), outputs cascade parameters "
                   "for direct NV_MultiModelSampler wiring. In cascaded mode, denoise is automatically "
                   "set to 1.0 (encoded in expanded_steps/start_at_step) and cascade_add_noise='disable'.")

    def load_chunk(self, video, plan_json_path, chunk_index,
                   control_1=None, control_2=None, control_3=None, control_4=None,
                   latent=None):
        """
        Extract the specified chunk from video and all control videos.
        Optionally slices a latent tensor using Wan VAE temporal frame mapping.
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
        expected_chunk_frames = chunk.get("frame_count", end_frame - start_frame)

        # Get video metadata
        input_video_frames = video.shape[0]
        video_metadata = plan.get("video_metadata", {})
        expected_total_frames = video_metadata.get("total_frames", plan.get("total_frames"))

        # Determine mode: pre-sliced chunk OR full video
        # Pre-sliced: input matches chunk frame count
        # Full video: input matches total frame count
        is_pre_sliced = (input_video_frames == expected_chunk_frames)
        is_full_video = (expected_total_frames and input_video_frames == expected_total_frames)

        if not is_pre_sliced and not is_full_video:
            raise ValueError(
                f"Video frame count mismatch! Input has {input_video_frames} frames.\n"
                f"Expected either:\n"
                f"  - Pre-sliced chunk: {expected_chunk_frames} frames (chunk {chunk_index})\n"
                f"  - Full video: {expected_total_frames} frames (will be sliced)\n"
                f"Make sure you're using the correct video."
            )

        if is_full_video and not is_pre_sliced:
            # Full video mode - validate and slice
            if end_frame > input_video_frames:
                raise ValueError(
                    f"Chunk {chunk_index} requires frames {start_frame}-{end_frame-1}, "
                    f"but video only has {input_video_frames} frames."
                )
            chunk_video = video[start_frame:end_frame].clone()
            slice_mode = "sliced"
        else:
            # Pre-sliced mode - pass through as-is
            chunk_video = video
            slice_mode = "pre-sliced"

        # Extract chunks from control videos (same logic)
        controls = [control_1, control_2, control_3, control_4]
        chunk_controls = []

        for i, ctrl in enumerate(controls):
            if ctrl is not None:
                ctrl_frames = ctrl.shape[0]
                ctrl_is_pre_sliced = (ctrl_frames == expected_chunk_frames)
                ctrl_is_full = (expected_total_frames and ctrl_frames == expected_total_frames)

                if not ctrl_is_pre_sliced and not ctrl_is_full:
                    raise ValueError(
                        f"Control video {i+1} has {ctrl_frames} frames.\n"
                        f"Expected either {expected_chunk_frames} (pre-sliced) or {expected_total_frames} (full)."
                    )

                if ctrl_is_full and not ctrl_is_pre_sliced:
                    chunk_controls.append(ctrl[start_frame:end_frame].clone())
                else:
                    chunk_controls.append(ctrl)
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

        # Get cascaded pipeline config (injected by NV_PreNoiseLatent)
        cascade = plan.get("cascaded_config", {})
        cascade_shift = cascade.get("shift_override", 0.0)
        cascade_expanded_steps = cascade.get("expanded_steps", 0)
        cascade_start_at_step = cascade.get("start_at_step", 0)
        cascade_add_noise = "disable" if (cascade and cascade_expanded_steps > 0) else "enable"

        # In cascaded mode, denoise is already encoded in expanded_steps/start_at_step.
        # Passing the shared_params denoise (<1.0) would double-truncate the sigma schedule,
        # resulting in zero effective denoising steps. Override to 1.0 so the sampler uses
        # the full expanded schedule and start_at_step handles the truncation.
        if cascade and cascade_expanded_steps > 0:
            original_denoise = denoise
            original_steps = steps
            denoise = 1.0
            steps = cascade_expanded_steps
            print(f"[NV_ChunkLoaderVACE] Cascaded mode: denoise overridden {original_denoise} → 1.0, "
                  f"steps overridden {original_steps} → {cascade_expanded_steps} "
                  f"(start_at_step={cascade_start_at_step}, effective={cascade_expanded_steps - cascade_start_at_step})")

        # --- Cross-validate plan config vs latent config ---
        if latent is not None and cascade and cascade_expanded_steps > 0:
            latent_config = latent.get(NV_CASCADED_CONFIG_KEY, None)
            if latent_config is not None:
                # Both plan and latent have config — check they agree
                mismatches = []
                for key in ("shift_override", "start_sigma", "expanded_steps", "start_at_step", "scheduler"):
                    plan_val = cascade.get(key)
                    latent_val = latent_config.get(key)
                    if plan_val is not None and latent_val is not None:
                        # Float comparison with tolerance for sigma
                        if isinstance(plan_val, float):
                            if not math.isclose(plan_val, latent_val, rel_tol=1e-4):
                                mismatches.append(f"  {key}: plan={plan_val}, latent={latent_val}")
                        elif plan_val != latent_val:
                            mismatches.append(f"  {key}: plan={plan_val}, latent={latent_val}")
                if mismatches:
                    raise ValueError(
                        f"[NV_ChunkLoaderVACE] CASCADED CONFIG MISMATCH!\n"
                        f"The plan JSON and the latent have different cascaded parameters.\n"
                        f"This means the pre-noised latent was generated with different settings\n"
                        f"than what the plan expects. The sampler would start at the wrong sigma,\n"
                        f"producing garbage output.\n\n"
                        f"Mismatched parameters:\n" + "\n".join(mismatches) + "\n\n"
                        f"Fix: Re-run NV_PreNoiseLatent with matching parameters, or regenerate\n"
                        f"the plan JSON to match the latent's config."
                    )
                print(f"[NV_ChunkLoaderVACE] Cascaded config validated: plan matches latent "
                      f"(shift={cascade_shift}, sigma={cascade.get('start_sigma', '?')})")
            else:
                # Plan has cascaded config but latent doesn't — likely a legacy .pt file
                print(f"[NV_ChunkLoaderVACE] WARNING: Plan has cascaded_config but the loaded "
                      f"latent has no embedded config (nv_cascaded_config key missing).\n"
                      f"  This could mean:\n"
                      f"  1. The .pt file was saved before config embedding was added\n"
                      f"  2. The wrong .pt file is being loaded (clean instead of pre-noised)\n"
                      f"  Expected: shift={cascade_shift}, sigma={cascade.get('start_sigma', '?')}\n"
                      f"  Cannot validate — proceeding with plan config. If output is noisy,\n"
                      f"  re-run NV_PreNoiseLatent to generate a latent with embedded config.")

        # Build info string
        num_controls = sum(1 for c in chunk_controls if c is not None)
        info_lines = [
            f"Chunk {chunk_index}/{num_chunks-1} ({slice_mode})",
            f"Frames: {start_frame}-{end_frame-1} ({chunk_video.shape[0]} frames)",
            f"Controls: {num_controls} processed",
            f"Seed: {seed}",
            f"Steps: {steps}",
            f"CFG: {cfg}",
            f"Denoise: {denoise}",
            f"Sampler: {sampler_name}",
            f"Scheduler: {scheduler}",
            f"Context Window: {context_window_size}",
            f"Context Overlap: {context_overlap}",
        ]
        if cascade:
            info_lines.extend([
                f"--- Cascaded Config (from NV_PreNoiseLatent) ---",
                f"Cascade Shift: {cascade_shift}",
                f"Cascade Expanded Steps: {cascade_expanded_steps}",
                f"Cascade Start At Step: {cascade_start_at_step}",
                f"Cascade Effective Steps: {cascade_expanded_steps - cascade_start_at_step}",
                f"Signal Preserved: {cascade.get('signal_preserved_pct', '?')}%",
                f"Add Noise: disable (latent is pre-noised)",
            ])

        chunk_info = "\n".join(info_lines)

        print(f"[NV_ChunkLoaderVACE] Loaded chunk {chunk_index} ({slice_mode}):")
        print(f"  Frames: {start_frame} to {end_frame-1} ({chunk_video.shape[0]} frames)")
        print(f"  Controls: {num_controls}")

        frame_count = end_frame - start_frame

        # Slice latent if provided (Wan VAE temporal frame mapping)
        chunk_latent = None
        if latent is not None:
            samples = latent["samples"]  # [B, C, T, H, W]
            total_latent_frames = samples.shape[2]

            # Wan VAE mapping: video frame V -> latent frame L
            #   V == 0 -> L = 0 (first frame gets its own latent)
            #   V > 0  -> L = (V - 1) // 4 + 1
            if start_frame == 0:
                latent_start = 0
            else:
                latent_start = (start_frame - 1) // 4 + 1

            chunk_latent_frames = (frame_count - 1) // 4 + 1
            latent_end = min(latent_start + chunk_latent_frames, total_latent_frames)

            # Pre-sliced detection: if latent temporal frames already match chunk
            if total_latent_frames == chunk_latent_frames:
                chunk_latent = latent  # pass through as-is
                print(f"  Latent: pre-sliced ({total_latent_frames} frames)")
            else:
                chunk_samples = samples[:, :, latent_start:latent_end, :, :].clone()
                # Build clean output dict — only copy safe keys
                # (latent.copy() would carry stale noise_mask/batch_index)
                chunk_latent = {"samples": chunk_samples}
                for k in LATENT_SAFE_KEYS:
                    if k in latent:
                        chunk_latent[k] = latent[k]
                print(f"  Latent: sliced frames {latent_start}-{latent_end-1} "
                      f"({chunk_samples.shape[2]} latent frames from {total_latent_frames} total)")

        return (chunk_video, chunk_controls[0], chunk_controls[1], chunk_controls[2], chunk_controls[3],
                chunk_index, start_frame, frame_count, seed, steps, cfg, denoise, sampler_name, scheduler,
                context_window_size, context_overlap, chunk_info,
                chunk_latent,
                cascade_shift, cascade_expanded_steps, cascade_start_at_step, cascade_add_noise)


# Node registration
NODE_CLASS_MAPPINGS = {
    "NV_ChunkLoader": NV_ChunkLoader,
    "NV_ChunkLoaderAdvanced": NV_ChunkLoaderAdvanced,
    "NV_ChunkPlanReader": NV_ChunkPlanReader,
    "NV_ChunkLoaderVACE": NV_ChunkLoaderVACE,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_ChunkLoader": "NV Chunk Loader",
    "NV_ChunkLoaderAdvanced": "NV Chunk Loader (Advanced)",
    "NV_ChunkPlanReader": "NV Chunk Plan Reader",
    "NV_ChunkLoaderVACE": "NV Chunk Loader (VACE)",
}
