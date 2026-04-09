"""
NV Temporal Retime — Slowdown-for-rediffusion bookend nodes.

High-motion video produces poor rediffusion results because per-frame delta
is too large for the diffusion model. This pair of nodes enables:

  1. NV_RetimePrep   — compute retiming metadata, output fps values for
                       an external frame interpolator (e.g. GIMM-VFI)
  2. NV_RetimeRestore — select frames back to original timing after rediffusion

Pipeline:
  RetimePrep → GIMM-VFI (external) → rediffuse → RetimeRestore
"""

import json
import os
import tempfile
import torch

import folder_paths

from .chunk_utils import is_wan_aligned


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_slowed_frame_count(original: int, factor: int) -> int:
    """Compute frame count after slowdown: (N-1) * factor + 1."""
    return (original - 1) * factor + 1


def _default_config_dir():
    return os.path.join(folder_paths.get_output_directory(), "retime_configs")


def _resolve_config_path(path: str) -> str:
    """Resolve a save/load path to an absolute .json path."""
    if not path:
        return ""
    resolved = path
    if not os.path.isabs(resolved):
        resolved = os.path.join(_default_config_dir(), resolved)
    root, ext = os.path.splitext(resolved)
    if ext.lower() != ".json":
        resolved = resolved + ".json"
    return os.path.normpath(resolved)


def _validate_retime_config(config: dict) -> dict:
    """Validate and normalize config dict. Raises ValueError on bad data."""
    required_keys = ["original_frame_count", "slowdown_factor", "expected_slowed_count", "source_fps", "target_fps"]
    missing = [k for k in required_keys if k not in config]
    if missing:
        raise ValueError(f"[NV_Retime] Config missing keys: {', '.join(missing)}")
    try:
        original = int(config["original_frame_count"])
        factor = int(config["slowdown_factor"])
        expected = int(config["expected_slowed_count"])
        source_fps = float(config["source_fps"])
        target_fps = float(config["target_fps"])
    except (TypeError, ValueError) as e:
        raise ValueError(f"[NV_Retime] Invalid config values: {e}") from e
    if original < 1:
        raise ValueError("[NV_Retime] original_frame_count must be >= 1")
    if factor < 2:
        raise ValueError("[NV_Retime] slowdown_factor must be >= 2")
    return {
        "original_frame_count": original,
        "slowdown_factor": factor,
        "expected_slowed_count": expected,
        "source_fps": source_fps,
        "target_fps": target_fps,
    }


def _save_config_atomic(config: dict, save_path: str) -> str:
    """Write config to JSON atomically (temp file + rename)."""
    resolved = _resolve_config_path(save_path)
    parent = os.path.dirname(resolved)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".retime_", suffix=".json", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, resolved)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    return resolved


def _load_config(config_path: str) -> dict:
    """Load and validate config from JSON file."""
    resolved = _resolve_config_path(config_path)
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"[NV_Retime] Config not found: {resolved}")
    try:
        with open(resolved, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"[NV_Retime] Invalid JSON at {resolved}: {e}") from e
    return _validate_retime_config(data)


# ---------------------------------------------------------------------------
# NV_RetimePrep
# ---------------------------------------------------------------------------

class NV_RetimePrep:
    """Compute retiming metadata and fps values for external frame interpolation.

    Pass images through unchanged; output source/target fps for GIMM-VFI
    and a RETIME_CONFIG for NV_RetimeRestore.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Original video frames [B,H,W,C]"
                }),
                "slowdown_factor": ("INT", {
                    "default": 2,
                    "min": 2,
                    "max": 8,
                    "step": 1,
                    "tooltip": "How many times to slow down. 2x = half the motion per frame, double the frame count."
                }),
                "source_fps": ("FLOAT", {
                    "default": 24.0,
                    "min": 1.0,
                    "max": 120.0,
                    "step": 0.1,
                    "tooltip": "FPS of input video. Target fps = source * factor."
                }),
            },
            "optional": {
                "save_path": ("STRING", {
                    "default": "",
                    "tooltip": "Save config to JSON. Relative paths resolve to ComfyUI output/retime_configs/. Leave empty to skip."
                }),
                "config_only": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Compute and save config only — returns a 1-frame placeholder image. Disconnect downstream image consumers when using this."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "FLOAT", "FLOAT", "INT", "RETIME_CONFIG")
    RETURN_NAMES = ("images", "source_fps", "target_fps", "expected_frames", "retime_config")
    FUNCTION = "execute"
    CATEGORY = "NV_Utils/temporal"
    DESCRIPTION = "Prepare video for slowdown rediffusion. Outputs fps values for GIMM-VFI and a config for NV_RetimeRestore."

    def execute(self, images, slowdown_factor, source_fps, save_path="", config_only=False):
        original_count = int(images.shape[0])
        if original_count < 1:
            raise ValueError("[NV_RetimePrep] Requires at least 1 input frame")

        slowdown_factor = int(slowdown_factor)
        source_fps = float(source_fps)
        slowed_count = _compute_slowed_frame_count(original_count, slowdown_factor)
        target_fps = source_fps * slowdown_factor

        wan_note = ""
        if is_wan_aligned(original_count) and is_wan_aligned(slowed_count):
            wan_note = " (WAN-aligned)"
        elif not is_wan_aligned(slowed_count):
            wan_note = f" (WARNING: slowed count {slowed_count} is NOT WAN-aligned)"

        print(f"[NV_RetimePrep] {original_count} frames @ {source_fps}fps → "
              f"{slowed_count} frames @ {target_fps:.1f}fps ({slowdown_factor}x slowdown){wan_note}")

        config = _validate_retime_config({
            "original_frame_count": original_count,
            "slowdown_factor": slowdown_factor,
            "expected_slowed_count": slowed_count,
            "source_fps": source_fps,
            "target_fps": target_fps,
        })

        if save_path:
            resolved = _save_config_atomic(config, save_path)
            print(f"[NV_RetimePrep] Config saved to {resolved}")

        if config_only:
            placeholder = images.new_zeros((1, *images.shape[1:]))
            print(f"[NV_RetimePrep] Config-only mode — returning 1-frame placeholder")
            return (placeholder, source_fps, target_fps, slowed_count, config)

        return (images, source_fps, target_fps, slowed_count, config)


# ---------------------------------------------------------------------------
# NV_RetimeRestore
# ---------------------------------------------------------------------------

class NV_RetimeRestore:
    """Restore original timing by selecting frames from rediffused slow video.

    After rediffusing a slowed-down video, this node picks every Nth frame
    to return to the original frame count and timing.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Rediffused slowed video frames [B,H,W,C]"
                }),
                "method": (["select", "blend"], {
                    "default": "select",
                    "tooltip": "select: pick every Nth frame (sharp). blend: weighted average of neighbors (smooth)."
                }),
            },
            "optional": {
                "retime_config": ("RETIME_CONFIG", {
                    "forceInput": True,
                    "tooltip": "Config from NV_RetimePrep (direct connection)"
                }),
                "config_path": ("STRING", {
                    "default": "",
                    "tooltip": "Load config from JSON file. Same filename used in RetimePrep save_path. Ignored if retime_config is connected."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "execute"
    CATEGORY = "NV_Utils/temporal"
    DESCRIPTION = "Restore original frame timing from a rediffused slow video. Pair with NV_RetimePrep."

    def execute(self, images, method, retime_config=None, config_path=""):
        available = int(images.shape[0])
        if available < 1:
            raise ValueError("[NV_RetimeRestore] Requires at least 1 input frame")

        if retime_config is not None:
            retime_config = _validate_retime_config(retime_config)
        elif config_path:
            retime_config = _load_config(config_path)
        else:
            raise ValueError("[NV_RetimeRestore] Either connect retime_config or provide a config_path")

        factor = retime_config["slowdown_factor"]
        original_count = retime_config["original_frame_count"]
        expected = retime_config["expected_slowed_count"]

        if available != expected:
            print(f"[NV_RetimeRestore] WARNING: expected {expected} slowed frames but got {available}. Using proportional mapping.")

        if method == "select":
            output = self._select_frames(images, original_count, available)
        else:
            output = self._blend_frames(images, original_count, available)

        print(f"[NV_RetimeRestore] {available} slowed frames → {output.shape[0]} restored frames "
              f"(factor={factor}, method={method})")

        return (output,)

    @staticmethod
    def _select_frames(images, original_count, available):
        """Pick evenly-spaced frames using proportional mapping."""
        if original_count == 1:
            return images[0:1]
        step = (available - 1) / (original_count - 1)
        indices = [min(round(i * step), available - 1) for i in range(original_count)]
        return images[indices]

    @staticmethod
    def _blend_frames(images, original_count, available):
        """Weighted average of two nearest source frames for each target position."""
        if original_count == 1:
            return images[0:1]
        step = (available - 1) / (original_count - 1)
        frames = []
        for i in range(original_count):
            pos = i * step
            if pos >= available - 1:
                frames.append(images[available - 1:available])
            else:
                lo = int(pos)
                hi = min(lo + 1, available - 1)
                t = pos - lo
                blended = images[lo] * (1.0 - t) + images[hi] * t
                frames.append(blended.unsqueeze(0))
        return torch.cat(frames, dim=0)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "NV_RetimePrep": NV_RetimePrep,
    "NV_RetimeRestore": NV_RetimeRestore,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_RetimePrep": "NV Retime Prep",
    "NV_RetimeRestore": "NV Retime Restore",
}
