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
import time
import traceback
import atexit
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch


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


def _parse_logs_for_data(logs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Parse captured logs to extract structured workflow data."""
    data = {
        "inputs": {},
        "vace": {},
        "models": [],
        "context_window": None,
        "errors": [],
    }

    # Regex patterns for extracting data from log messages
    patterns = {
        # Input shapes
        "pixel_shape": r"Input shape: torch\.Size\(\[(\d+), (\d+), (\d+), (\d+)\]\)",
        "latent_shape": r"Output latent shape: torch\.Size\(\[(\d+), (\d+), (\d+), (\d+), (\d+)\]\)",
        "latent_input": r"Input shape: torch\.Size\(\[(\d+), (\d+), (\d+), (\d+), (\d+)\]\)",

        # VACE shapes
        "vace_frames": r"control_video_latent shape: torch\.Size\(\[([^\]]+)\]\)",
        "vace_mask": r"vace_mask shape: torch\.Size\(\[([^\]]+)\]\)",

        # Model loading
        "model_loaded": r"Requested to load (\w+)",
        "model_size": r"loaded completely.*?(\d+\.?\d*) MB loaded",

        # Context window
        "context_window": r"Using context windows (\d+) for (\d+) frames",

        # Errors
        "oom_error": r"OutOfMemoryError|Allocation on device",
        "error_location": r"File \"[^\"]+\\([^\"]+)\", line \d+, in (\w+)",
    }

    current_model = None
    seen_models = set()

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

        # Parse VACE shapes
        match = re.search(patterns["vace_frames"], msg)
        if match:
            shape_str = match.group(1)
            data["vace"]["control_shape"] = [int(x.strip()) for x in shape_str.split(",")]

        match = re.search(patterns["vace_mask"], msg)
        if match:
            shape_str = match.group(1)
            data["vace"]["mask_shape"] = [int(x.strip()) for x in shape_str.split(",")]

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

        # Parse context window
        match = re.search(patterns["context_window"], msg)
        if match:
            window_size, total_frames = map(int, match.groups())
            data["context_window"] = {
                "window_size": window_size,
                "total_frames": total_frames,
            }

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
                "any_input": ("*", {
                    "tooltip": "Connect to a late-executing node to ensure logger captures full workflow."
                }),
            },
        }

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

        log_entry = {
            "timestamp": timestamp,
            "execution_time_sec": round(execution_time, 2),
            "gpu_info": _get_gpu_info(),
            "memory": {
                "peak": _get_memory_stats(),
            },
            "outcome": "success",  # Will be updated if crash/OOM
            "raw_logs_count": len(_capture_handler.logs),
        }

        # Parse logs for structured data
        parsed = _parse_logs_for_data(_capture_handler.logs)
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
