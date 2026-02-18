"""
NV Workflow Logger

Captures workflow execution data for memory analysis and debugging.
Unobtrusive node that hooks into Python logging to capture:
- Input dimensions (pixels, latents, VACE controls)
- Models loaded with sizes
- GPU memory stats
- Outcome (success/OOM/error)

Outputs structured JSON for building memory estimation datasets.
"""

import json
import logging
import os
import re
import statistics
import sys
import time
import traceback
import atexit
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch
from comfy.comfy_types.node_typing import IO


class LogCaptureHandler(logging.Handler):
    """Custom logging handler that captures log messages."""

    def __init__(self):
        super().__init__()
        self.logs: List[Dict[str, Any]] = []
        self.start_time = time.time()

    def emit(self, record):
        try:
            self.logs.append({
                "time_offset": record.created - self.start_time,
                "level": record.levelname,
                "message": self.format(record),
                "logger": record.name,
            })
        except Exception:
            pass  # Don't let logging errors break the workflow

    def clear(self):
        self.logs = []
        self.start_time = time.time()


# Global capture handler and state
_capture_handler: Optional[LogCaptureHandler] = None
_pending_log: Optional[Dict[str, Any]] = None
_output_path: Optional[str] = None
_log_name: Optional[str] = None


def _get_memory_stats() -> Dict[str, float]:
    """Get current GPU memory statistics."""
    if not torch.cuda.is_available():
        return {}

    try:
        return {
            "gpu_allocated_mb": torch.cuda.memory_allocated() / 1024**2,
            "gpu_reserved_mb": torch.cuda.memory_reserved() / 1024**2,
            "gpu_max_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
            "gpu_max_reserved_mb": torch.cuda.max_memory_reserved() / 1024**2,
        }
    except Exception:
        return {}


def _get_gpu_info() -> Dict[str, Any]:
    """Get GPU device information."""
    if not torch.cuda.is_available():
        return {"available": False}

    try:
        props = torch.cuda.get_device_properties(0)
        return {
            "available": True,
            "name": props.name,
            "total_memory_mb": props.total_memory / 1024**2,
            "compute_capability": f"{props.major}.{props.minor}",
        }
    except Exception:
        return {"available": True, "error": "Could not get properties"}


def _get_environment_info() -> Dict[str, str]:
    """Get software environment versions."""
    env = {
        "python_version": sys.version.split()[0],
        "pytorch_version": torch.__version__,
    }
    if torch.cuda.is_available():
        env["cuda_version"] = torch.version.cuda or "unknown"

    # Try to get ComfyUI version
    try:
        import folder_paths
        version_file = os.path.join(folder_paths.base_path, "version.txt")
        if os.path.exists(version_file):
            with open(version_file) as f:
                env["comfyui_version"] = f.read().strip()
    except Exception:
        pass

    return env


def _get_memory_settings() -> Dict[str, Any]:
    """Get current memory management settings."""
    try:
        from comfy import model_management
        settings = {
            "pinned_memory_enabled": getattr(model_management, 'ENABLE_PINNED_MEMORY', False),
        }
        if hasattr(model_management, 'vram_state'):
            settings["vram_state"] = str(model_management.vram_state.name)
        return settings
    except Exception:
        return {}


def _detect_workflow_type(logs: List[Dict[str, Any]]) -> str:
    """Detect workflow type based on log messages."""
    has_vace = False
    has_image_input = False
    has_wan = False
    has_controlnet = False
    sampler_count = 0

    for log_entry in logs:
        msg = log_entry.get("message", "").lower()

        # Check for VACE
        if "vace" in msg or "control_video_latent" in msg:
            has_vace = True

        # Check for WAN model
        if "wan" in msg and ("model" in msg or "loaded" in msg):
            has_wan = True

        # Check for image input (img2vid)
        if "input image" in msg or "load image" in msg or "image_to_video" in msg:
            has_image_input = True

        # Check for ControlNet
        if "controlnet" in msg:
            has_controlnet = True

        # Count samplers
        if "ksampler" in msg or "sampler" in msg and "starting" in msg:
            sampler_count += 1

    # Determine workflow type
    if has_vace:
        return "vace"
    elif has_controlnet:
        return "controlnet"
    elif has_wan:
        if has_image_input:
            return "img2vid"
        else:
            return "txt2vid"
    elif sampler_count > 1:
        return "multi_model"
    else:
        return "unknown"


def _parse_logs_for_data(logs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Parse captured logs to extract structured workflow data."""
    data = {
        "inputs": {},
        "vae": {},
        "vace": {},
        "models": [],
        "context_window": None,
        "sampling": {},
        "multi_model": {},
        "adaptive_shift": {},
        "chunk": {},
        "timing": {
            "per_step_times": [],
            "per_iteration_times": [],
        },
        "errors": [],
    }

    # Regex patterns for extracting data from log messages
    patterns = {
        # Input shapes (legacy)
        "pixel_shape": r"Input shape: torch\.Size\(\[(\d+), (\d+), (\d+), (\d+)\]\)",
        "latent_shape": r"Output latent shape: torch\.Size\(\[(\d+), (\d+), (\d+), (\d+), (\d+)\]\)",
        "latent_input": r"Input shape: torch\.Size\(\[(\d+), (\d+), (\d+), (\d+), (\d+)\]\)",

        # VACE shapes (legacy)
        "vace_frames": r"control_video_latent shape: torch\.Size\(\[([^\]]+)\]\)",
        "vace_mask": r"vace_mask shape: torch\.Size\(\[([^\]]+)\]\)",

        # === VACE Slicer shapes (new) ===
        "vace_slicer_frames": r"\[VACE Slicer\] Sliced vace_frames:.*?-> torch\.Size\(\[([^\]]+)\]\)",
        "vace_slicer_mask": r"\[VACE Slicer\] Sliced vace_mask:.*?-> torch\.Size\(\[([^\]]+)\]\)",
        "vace_slicer_context": r"\[VACE Slicer\] Sliced vace_context:.*?-> torch\.Size\(\[([^\]]+)\]\)",
        "vace_original_frames": r"\[VACE Slicer\] Cached original vace_frames: shape torch\.Size\(\[([^\]]+)\]\)",
        "vace_original_mask": r"\[VACE Slicer\] Cached original vace_mask: shape torch\.Size\(\[([^\]]+)\]\)",
        "x_in_shape": r"x_in shape: torch\.Size\(\[([^\]]+)\]\)",

        # === VAE Encode/Decode ===
        "vae_encode_input": r"\[NV_StreamingVAEEncode\] Input shape: torch\.Size\(\[([^\]]+)\]\)",
        "vae_encode_output": r"\[NV_StreamingVAEEncode\].*Output latent shape: torch\.Size\(\[([^\]]+)\]\)",
        "vae_encode_memory": r"\[NV_StreamingVAEEncode\] Memory: ([\d.]+) GB pixels -> ([\d.]+) MB latents",
        "vae_decode_input": r"\[NV_StreamingVAEDecode\] Input shape: torch\.Size\(\[([^\]]+)\]\)",
        "vae_decode_output": r"\[NV_StreamingVAEDecode\].*Output shape: torch\.Size\(\[([^\]]+)\]\)",
        "vae_decode_frames": r"\[NV_StreamingVAEDecode\] Starting decode of (\d+) latent frames",

        # === Multi-Model Sampler ===
        "multi_model_mode": r"\[NV_MultiModelSampler\] Sequential mode: (\d+) models",
        "multi_model_steps": r"\[NV_MultiModelSampler\] Step distribution: \[([^\]]+)\]",
        "multi_model_config": r"\[NV_MultiModelSampler\] Model (\d+): steps (\d+)-(\d+), cfg=([\d.]+)",

        # === Adaptive Shift ===
        "adaptive_shift": r"\[NV_AdaptiveShiftApplier\] (?:manual )?shift=([\d.]+)",

        # === Chunk Loader ===
        "chunk_loader": r"\[NV_ChunkLoaderVACE\] Loaded chunk (\d+):\s*Frames: (\d+) to (\d+) \((\d+) frames\)",

        # Model loading
        "model_loaded": r"Requested to load (\w+)",
        "model_size": r"loaded completely.*?(\d+\.?\d*) MB loaded",
        "model_dtype": r"dtype[=:\s]*(bf16|fp16|fp32|float16|float32|bfloat16)",
        "lowvram_patches": r"lowvram patches:\s*(\d+)",

        # Context window - enhanced to capture overlap
        "context_window": r"Using context windows?\s*(\d+)\s*(?:for|with)?\s*(\d+)\s*frames?",
        "context_window_full": r"context windows?\s*(\d+)\s*with overlap\s*(\d+)\s*for\s*(\d+)\s*frames",
        "context_overlap": r"overlap[=:\s]+(\d+)",
        "context_stride": r"stride[=:\s]+(\d+)",

        # Sampling info
        "sampler_steps": r"(\d+)\s*(?:steps?|iterations?)",
        "sampler_name": r"sampler[=:\s]+['\"]?(\w+)['\"]?",
        "cfg_scale": r"cfg[=:\s]+(\d+\.?\d*)",

        # Timing - iteration times from progress logs (fixed patterns)
        "tqdm_progress": r"(\d+)/(\d+)\s*\[[^\]]+,\s*(\d+\.?\d*)s/it\]",
        "iteration_time": r"(\d+\.?\d*)s/it",
        "it_per_sec": r"(\d+\.?\d*)it/s",
        "step_progress": r"(\d+)/(\d+)\s*\[",

        # Execution time
        "prompt_executed": r"Prompt executed in (\d+):(\d+):(\d+)",

        # Attention type
        "attention_type": r"(sage|flash|pytorch|sdpa)[\s_-]?attention",

        # Errors
        "oom_error": r"OutOfMemoryError|Allocation on device",
        "error_location": r"File \"[^\"]+\\([^\"]+)\", line \d+, in (\w+)",
    }

    current_model = None
    seen_models = set()
    vace_shapes_seen = set()  # Track unique VACE shapes for condition counting

    for log_entry in logs:
        msg = log_entry.get("message", "")
        level = log_entry.get("level", "")

        # Parse pixel input shape
        match = re.search(patterns["pixel_shape"], msg)
        if match:
            frames, h, w, c = map(int, match.groups())
            data["inputs"]["pixel_frames"] = frames
            data["inputs"]["pixel_resolution"] = [h, w]

        # Parse latent output shape
        match = re.search(patterns["latent_shape"], msg)
        if match:
            b, c, t, h, w = map(int, match.groups())
            data["inputs"]["latent_shape"] = [b, c, t, h, w]
            data["inputs"]["latent_frames"] = t
            data["inputs"]["latent_spatial"] = [h, w]

        # Parse latent input shape (for decode)
        if "latent frames" in msg.lower():
            match = re.search(patterns["latent_input"], msg)
            if match:
                b, c, t, h, w = map(int, match.groups())
                data["inputs"]["latent_shape"] = [b, c, t, h, w]
                data["inputs"]["latent_frames"] = t
                data["inputs"]["latent_spatial"] = [h, w]

        # Parse VACE shapes (track unique shapes for condition counting)
        match = re.search(patterns["vace_frames"], msg)
        if match:
            shape_str = match.group(1)
            data["vace"]["control_shape"] = [int(x.strip()) for x in shape_str.split(",")]
            vace_shapes_seen.add(shape_str)  # Track for condition counting

        match = re.search(patterns["vace_mask"], msg)
        if match:
            shape_str = match.group(1)
            data["vace"]["mask_shape"] = [int(x.strip()) for x in shape_str.split(",")]

        # === Parse VACE Slicer shapes (new) ===
        match = re.search(patterns["vace_original_frames"], msg)
        if match:
            shape_str = match.group(1)
            shape = [int(x.strip()) for x in shape_str.split(",")]
            data["vace"]["original_frames_shape"] = shape
            # Extract dimensions: [B, C, T, H, W]
            if len(shape) >= 5:
                data["inputs"]["latent_frames"] = shape[2]
                data["inputs"]["latent_spatial"] = [shape[3], shape[4]]

        match = re.search(patterns["vace_original_mask"], msg)
        if match:
            shape_str = match.group(1)
            data["vace"]["original_mask_shape"] = [int(x.strip()) for x in shape_str.split(",")]

        match = re.search(patterns["vace_slicer_frames"], msg)
        if match:
            shape_str = match.group(1)
            data["vace"]["sliced_frames_shape"] = [int(x.strip()) for x in shape_str.split(",")]

        match = re.search(patterns["vace_slicer_mask"], msg)
        if match:
            shape_str = match.group(1)
            data["vace"]["sliced_mask_shape"] = [int(x.strip()) for x in shape_str.split(",")]

        match = re.search(patterns["x_in_shape"], msg)
        if match:
            shape_str = match.group(1)
            data["vace"]["x_in_shape"] = [int(x.strip()) for x in shape_str.split(",")]

        # === Parse VAE Encode/Decode ===
        match = re.search(patterns["vae_encode_input"], msg)
        if match:
            shape_str = match.group(1)
            shape = [int(x.strip()) for x in shape_str.split(",")]
            data["vae"]["encode_input_shape"] = shape
            # Extract pixel dimensions: [F, H, W, C]
            if len(shape) >= 4:
                data["inputs"]["pixel_frames"] = shape[0]
                data["inputs"]["pixel_resolution"] = [shape[1], shape[2]]

        match = re.search(patterns["vae_encode_output"], msg)
        if match:
            shape_str = match.group(1)
            shape = [int(x.strip()) for x in shape_str.split(",")]
            data["vae"]["encode_output_shape"] = shape
            # Extract latent dimensions: [B, C, T, H, W]
            if len(shape) >= 5:
                data["inputs"]["latent_shape"] = shape
                data["inputs"]["latent_frames"] = shape[2]
                data["inputs"]["latent_spatial"] = [shape[3], shape[4]]

        match = re.search(patterns["vae_encode_memory"], msg)
        if match:
            data["vae"]["encode_memory_gb"] = float(match.group(1))
            data["vae"]["encode_output_mb"] = float(match.group(2))

        match = re.search(patterns["vae_decode_input"], msg)
        if match:
            shape_str = match.group(1)
            data["vae"]["decode_input_shape"] = [int(x.strip()) for x in shape_str.split(",")]

        match = re.search(patterns["vae_decode_output"], msg)
        if match:
            shape_str = match.group(1)
            shape = [int(x.strip()) for x in shape_str.split(",")]
            data["vae"]["decode_output_shape"] = shape
            # Also update pixel dimensions from decode output
            if len(shape) >= 4:
                data["inputs"]["pixel_frames"] = shape[0]
                data["inputs"]["pixel_resolution"] = [shape[1], shape[2]]

        # === Parse Multi-Model Sampler ===
        match = re.search(patterns["multi_model_mode"], msg)
        if match:
            data["multi_model"]["num_models"] = int(match.group(1))

        match = re.search(patterns["multi_model_steps"], msg)
        if match:
            steps_str = match.group(1)
            data["multi_model"]["step_distribution"] = [int(x.strip()) for x in steps_str.split(",")]

        match = re.search(patterns["multi_model_config"], msg)
        if match:
            model_num, start_step, end_step, cfg = match.groups()
            if "configs" not in data["multi_model"]:
                data["multi_model"]["configs"] = []
            data["multi_model"]["configs"].append({
                "model": int(model_num),
                "steps": f"{start_step}-{end_step}",
                "cfg": float(cfg),
            })

        # === Parse Adaptive Shift ===
        match = re.search(patterns["adaptive_shift"], msg)
        if match:
            shift, frames, denoise, shift_type = match.groups()
            data["adaptive_shift"] = {
                "shift": float(shift),
                "frames": int(frames),
                "denoise": float(denoise),
                "type": shift_type,
            }

        # === Parse Chunk Loader ===
        match = re.search(patterns["chunk_loader"], msg)
        if match:
            chunk_idx, frame_start, frame_end, num_frames = match.groups()
            data["chunk"] = {
                "chunk_index": int(chunk_idx),
                "frame_start": int(frame_start),
                "frame_end": int(frame_end),
                "num_frames": int(num_frames),
            }

        # Parse model loading
        match = re.search(patterns["model_loaded"], msg)
        if match:
            current_model = match.group(1)

        match = re.search(patterns["model_size"], msg)
        if match and current_model and current_model not in seen_models:
            size_mb = float(match.group(1))
            data["models"].append({
                "name": current_model,
                "size_mb": size_mb,
            })
            seen_models.add(current_model)
            current_model = None

        # Parse model dtype
        match = re.search(patterns["model_dtype"], msg, re.IGNORECASE)
        if match:
            dtype = match.group(1).lower()
            # Normalize dtype names
            dtype_map = {"float16": "fp16", "float32": "fp32", "bfloat16": "bf16"}
            dtype = dtype_map.get(dtype, dtype)
            if data["models"]:
                data["models"][-1]["dtype"] = dtype

        # Parse lowvram patches count
        match = re.search(patterns["lowvram_patches"], msg)
        if match:
            data["memory_settings"] = data.get("memory_settings", {})
            data["memory_settings"]["lowvram_patches_count"] = int(match.group(1))

        # Parse context window with full pattern (includes overlap)
        match = re.search(patterns["context_window_full"], msg)
        if match:
            window_size, overlap, total_frames = map(int, match.groups())
            data["context_window"] = {
                "window_size": window_size,
                "overlap": overlap,
                "total_frames": total_frames,
                "stride": window_size - overlap,
            }
        else:
            # Fallback to legacy pattern
            match = re.search(patterns["context_window"], msg)
            if match:
                window_size, total_frames = map(int, match.groups())
                if data["context_window"] is None:
                    data["context_window"] = {
                        "window_size": window_size,
                        "total_frames": total_frames,
                    }

        # Parse context overlap (legacy fallback)
        match = re.search(patterns["context_overlap"], msg)
        if match and data["context_window"] and "overlap" not in data["context_window"]:
            data["context_window"]["overlap"] = int(match.group(1))
            # Calculate stride if we have overlap
            if "window_size" in data["context_window"]:
                data["context_window"]["stride"] = data["context_window"]["window_size"] - data["context_window"]["overlap"]

        # Parse context stride (if explicitly provided)
        match = re.search(patterns["context_stride"], msg)
        if match and data["context_window"]:
            data["context_window"]["stride"] = int(match.group(1))

        # Parse sampling info
        match = re.search(patterns["sampler_name"], msg, re.IGNORECASE)
        if match:
            data["sampling"]["sampler"] = match.group(1)

        match = re.search(patterns["cfg_scale"], msg)
        if match:
            data["sampling"]["cfg_scale"] = float(match.group(1))

        # Parse step progress to get total steps
        match = re.search(patterns["step_progress"], msg)
        if match:
            current_step, total_steps = map(int, match.groups())
            data["sampling"]["total_steps"] = total_steps

        # Parse iteration times from progress logs
        match = re.search(patterns["iteration_time"], msg)
        if match:
            it_time = float(match.group(1))
            data["timing"]["per_iteration_times"].append(it_time)

        # Parse attention type
        match = re.search(patterns["attention_type"], msg, re.IGNORECASE)
        if match:
            data["gpu_settings"] = data.get("gpu_settings", {})
            data["gpu_settings"]["attention_type"] = match.group(1).lower()

        # Parse errors
        if level == "ERROR" or re.search(patterns["oom_error"], msg):
            # Try to extract error location
            match = re.search(patterns["error_location"], msg)
            location = match.group(2) if match else None

            data["errors"].append({
                "message": msg[:500],  # Truncate long messages
                "location": location,
                "is_oom": bool(re.search(patterns["oom_error"], msg)),
            })

    # Calculate VACE condition count from unique shapes seen
    if vace_shapes_seen:
        data["vace"]["num_conditions"] = len(vace_shapes_seen)

    # Calculate number of windows if we have context window data
    if data["context_window"] and "total_frames" in data["context_window"]:
        cw = data["context_window"]
        total_frames = cw["total_frames"]
        window_size = cw.get("window_size", 16)
        stride = cw.get("stride", window_size // 2)  # Default to 50% overlap

        if stride > 0 and total_frames > window_size:
            num_windows = ((total_frames - window_size) // stride) + 1
            if (total_frames - window_size) % stride > 0:
                num_windows += 1
            data["context_window"]["num_windows"] = num_windows
        else:
            data["context_window"]["num_windows"] = 1

    # Calculate timing statistics
    if data["timing"]["per_iteration_times"]:
        times = data["timing"]["per_iteration_times"]
        data["timing"]["iteration_stats"] = {
            "count": len(times),
            "mean_sec": round(statistics.mean(times), 2),
            "min_sec": round(min(times), 2),
            "max_sec": round(max(times), 2),
            "std_sec": round(statistics.stdev(times), 2) if len(times) > 1 else 0,
            "total_sec": round(sum(times), 2),
        }

    # Clean up empty dicts
    if not data["vace"]:
        del data["vace"]
    if not data["vae"]:
        del data["vae"]
    if not data["sampling"]:
        del data["sampling"]
    if not data["multi_model"]:
        del data["multi_model"]
    if not data["adaptive_shift"]:
        del data["adaptive_shift"]
    if not data["chunk"]:
        del data["chunk"]
    if not data["timing"]["per_step_times"] and not data["timing"]["per_iteration_times"]:
        del data["timing"]
    if not data["inputs"]:
        del data["inputs"]

    return data


def _dump_log():
    """Write the pending log to JSON file."""
    global _pending_log, _output_path, _log_name

    if _pending_log is None or _output_path is None:
        return

    try:
        # Ensure output directory exists
        os.makedirs(_output_path, exist_ok=True)

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = _log_name if _log_name else "workflow"
        filename = f"{timestamp}_{prefix}.json"
        filepath = os.path.join(_output_path, filename)

        # Write JSON
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(_pending_log, f, indent=2, default=str)

        print(f"[NV_WorkflowLogger] Log saved to: {filepath}")

    except Exception as e:
        print(f"[NV_WorkflowLogger] Failed to save log: {e}")

    finally:
        _pending_log = None


def _on_exit():
    """Called on program exit - ensures crash data is captured."""
    global _pending_log, _capture_handler

    if _pending_log is not None:
        # If we're exiting with a pending log, it's likely a crash
        _pending_log["outcome"] = "crash"
        _pending_log["memory"]["at_crash"] = _get_memory_stats()

        if _capture_handler:
            parsed = _parse_logs_for_data(_capture_handler.logs)
            _pending_log.update(parsed)

            # Check for OOM in errors
            for error in parsed.get("errors", []):
                if error.get("is_oom"):
                    _pending_log["outcome"] = "OOM"
                    _pending_log["error_message"] = "Allocation on device"
                    _pending_log["error_location"] = error.get("location")
                    break

        _dump_log()


# Register atexit handler
atexit.register(_on_exit)


class NV_WorkflowLogger:
    """
    Workflow Logger for Memory Analysis.

    Captures workflow execution data including:
    - Input/output tensor shapes
    - Models loaded and their sizes
    - GPU memory usage
    - Execution outcome (success/OOM/error)

    Outputs structured JSON for building memory estimation datasets.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "output_path": ("STRING", {
                    "default": "",
                    "tooltip": "Directory to save workflow logs. Leave empty to use ComfyUI output folder."
                }),
            },
            "optional": {
                "log_name": ("STRING", {
                    "default": "workflow",
                    "tooltip": "Prefix for log filename. Output: {timestamp}_{log_name}.json"
                }),
                "any_input": (IO.ANY, {
                    "tooltip": "Connect to a late-executing node to ensure logger captures full workflow."
                }),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        # Accept any input type for 'any_input' wildcard parameter
        return True

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("log_path",)
    FUNCTION = "log_workflow"
    CATEGORY = "NV_Utils"
    OUTPUT_NODE = True
    DESCRIPTION = "Captures workflow execution data for memory analysis. Connect to final node to ensure complete capture."

    def __init__(self):
        self.start_time = None

    def log_workflow(self, output_path: str, log_name: str = "workflow", any_input=None):
        """Capture and save workflow execution data."""
        global _capture_handler, _pending_log, _output_path, _log_name

        # Determine output path
        if not output_path:
            # Default to ComfyUI output folder
            import folder_paths
            output_path = os.path.join(folder_paths.get_output_directory(), "workflow_logs")

        _output_path = output_path
        _log_name = log_name

        # Initialize log capture if not already done
        if _capture_handler is None:
            _capture_handler = LogCaptureHandler()
            _capture_handler.setLevel(logging.DEBUG)

            # Add to root logger to capture all logs
            root_logger = logging.getLogger()
            root_logger.addHandler(_capture_handler)

        # Build the log entry
        timestamp = datetime.now().isoformat()
        execution_time = time.time() - _capture_handler.start_time

        # Parse logs first to detect workflow type
        parsed = _parse_logs_for_data(_capture_handler.logs)
        workflow_type = _detect_workflow_type(_capture_handler.logs)

        log_entry = {
            "timestamp": timestamp,
            "execution_time_sec": round(execution_time, 2),
            "outcome": "success",  # Will be updated if crash/OOM
            "workflow_type": workflow_type,

            # Environment info
            "environment": _get_environment_info(),

            # GPU info
            "gpu": _get_gpu_info(),

            # Memory settings and state
            "memory_settings": _get_memory_settings(),

            # Memory stats
            "memory": {
                "peak": _get_memory_stats(),
            },

            "raw_logs_count": len(_capture_handler.logs),
        }

        # Merge memory_settings from parsed data if present
        if "memory_settings" in parsed:
            log_entry["memory_settings"].update(parsed.pop("memory_settings"))

        # Merge gpu_settings from parsed data if present
        if "gpu_settings" in parsed:
            log_entry["gpu"].update(parsed.pop("gpu_settings"))

        # Add parsed data
        log_entry.update(parsed)

        # Store as pending in case of crash
        _pending_log = log_entry

        # Reset memory tracking for next run
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Save the log
        _dump_log()

        # Clear capture handler for next workflow
        _capture_handler.clear()

        # Return the log path
        log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filepath = os.path.join(output_path, f"{log_timestamp}_{log_name}.json")

        return (log_filepath,)


class NV_WorkflowLoggerStart:
    """
    Workflow Logger Start Node.

    Place at the beginning of your workflow to start capturing logs.
    Use with NV_WorkflowLogger (end) to capture the full workflow.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "trigger": ("*", {"tooltip": "Optional trigger input"}),
            },
        }

    RETURN_TYPES = ("WORKFLOW_LOG_SESSION",)
    RETURN_NAMES = ("session",)
    FUNCTION = "start_capture"
    CATEGORY = "NV_Utils"
    DESCRIPTION = "Starts workflow log capture. Place at beginning of workflow."

    def start_capture(self, trigger=None):
        """Initialize log capture session."""
        global _capture_handler

        # Initialize or reset capture handler
        if _capture_handler is None:
            _capture_handler = LogCaptureHandler()
            _capture_handler.setLevel(logging.DEBUG)

            # Add to root logger
            root_logger = logging.getLogger()
            root_logger.addHandler(_capture_handler)
        else:
            _capture_handler.clear()

        # Reset GPU memory stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        print("[NV_WorkflowLogger] Log capture started")

        # Return a session identifier
        session_id = datetime.now().isoformat()
        return (session_id,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "NV_WorkflowLogger": NV_WorkflowLogger,
    "NV_WorkflowLoggerStart": NV_WorkflowLoggerStart,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_WorkflowLogger": "NV Workflow Logger",
    "NV_WorkflowLoggerStart": "NV Workflow Logger (Start)",
}
