"""
NV Generation Info Node

Single node that automatically measures generation time by hooking into
ComfyUI's prompt submission system. Drop at the end of any workflow to get
timing metrics, throughput stats, and CSV logging for tracking the relationship
between resolution, frame count, FPS, and generation time.

Auto-detects resolution and frame count from the passthrough tensor shape
(IMAGE or LATENT). Manual inputs override when non-zero.
"""

import csv
import os
import time
from datetime import datetime

import torch
from comfy.comfy_types.node_typing import IO
import folder_paths
import server

# ---------------------------------------------------------------------------
# Module-level hook: capture prompt submission time automatically
# ---------------------------------------------------------------------------
_prompt_start_times = {}


def _on_prompt_submitted(json_data):
    """Called when a prompt is submitted to the server."""
    _prompt_start_times["_latest"] = time.time()
    return json_data


try:
    server.PromptServer.instance.add_on_prompt_handler(_on_prompt_submitted)
except Exception:
    pass  # Server may not be initialized during import in some contexts


# ---------------------------------------------------------------------------
# Auto-detection helpers
# ---------------------------------------------------------------------------
def _detect_from_passthrough(passthrough):
    """Try to extract width, height, num_frames from the passthrough tensor.

    Returns (width, height, num_frames) or (0, 0, 0) if detection fails.
    """
    try:
        # LATENT dict: samples shape is [B, C, T, H, W]
        if isinstance(passthrough, dict) and "samples" in passthrough:
            samples = passthrough["samples"]
            if hasattr(samples, "shape") and len(samples.shape) == 5:
                _, _, t, h, w = samples.shape
                pixel_frames = int(t) * 4 - 3
                return int(w) * 8, int(h) * 8, max(pixel_frames, 1)

        # IMAGE tensor: shape is [F, H, W, C]
        if hasattr(passthrough, "shape") and len(passthrough.shape) == 4:
            f, h, w, _ = passthrough.shape
            return int(w), int(h), int(f)
    except Exception:
        pass

    return 0, 0, 0


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------
class NV_GenerationInfo:
    """
    Measures total generation time and computes throughput metrics.

    Hooks into ComfyUI's prompt system to automatically capture start time.
    Auto-detects resolution and frame count from the passthrough input.
    Displays a report and optionally logs to CSV for cross-run analysis.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "passthrough": (IO.ANY, {
                    "tooltip": "Connect to a late-pipeline output (IMAGE, LATENT, etc.) to set execution order and enable auto-detection"
                }),
            },
            "optional": {
                "width": ("INT", {
                    "default": 0, "min": 0, "max": 8192, "step": 8,
                    "tooltip": "Pixel width (0 = auto-detect from passthrough)"
                }),
                "height": ("INT", {
                    "default": 0, "min": 0, "max": 8192, "step": 8,
                    "tooltip": "Pixel height (0 = auto-detect from passthrough)"
                }),
                "num_frames": ("INT", {
                    "default": 0, "min": 0, "max": 100000,
                    "tooltip": "Total frames generated (0 = auto-detect from passthrough)"
                }),
                "fps": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 240.0, "step": 0.01,
                    "tooltip": "Output video FPS (for realtime ratio calculation)"
                }),
                "steps": ("INT", {
                    "default": 0, "min": 0, "max": 10000,
                    "tooltip": "Number of sampling steps"
                }),
                "cfg": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 100.0, "step": 0.01,
                    "tooltip": "CFG scale used"
                }),
                "sampler_name": ("STRING", {
                    "default": "",
                    "tooltip": "Sampler name (e.g., euler, dpmpp_2m)"
                }),
                "model_name": ("STRING", {
                    "default": "",
                    "tooltip": "Model identifier for tracking"
                }),
                "notes": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Freeform notes about this run"
                }),
                "csv_path": ("STRING", {
                    "default": "gen_timer_log.csv",
                    "tooltip": "CSV log file path (relative to output dir). Empty to disable."
                }),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    RETURN_TYPES = ("STRING", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", IO.ANY)
    RETURN_NAMES = ("report", "gen_time_sec", "sec_per_frame", "frames_per_min",
                    "total_megapixels", "megapixels_per_sec", "output_seconds",
                    "realtime_ratio", "passthrough")
    OUTPUT_NODE = True
    FUNCTION = "measure"
    CATEGORY = "NV_Utils/Benchmark"
    DESCRIPTION = "Measures generation time and computes throughput metrics. Drop at end of workflow."

    def measure(self, passthrough, width=0, height=0, num_frames=0,
                fps=0.0, steps=0, cfg=0.0, sampler_name="", model_name="",
                notes="", csv_path="gen_timer_log.csv"):

        end_time = time.time()
        start_time = _prompt_start_times.get("_latest", end_time)
        gen_time_sec = end_time - start_time

        # --- Auto-detect from passthrough, manual overrides non-zero ---
        det_w, det_h, det_f = _detect_from_passthrough(passthrough)
        if width == 0:
            width = det_w
        if height == 0:
            height = det_h
        if num_frames == 0:
            num_frames = det_f

        # --- VRAM peak ---
        vram_peak_gb = 0.0
        if torch.cuda.is_available():
            vram_peak_gb = torch.cuda.max_memory_allocated() / 1024**3

        # --- Time formatting ---
        hours, remainder = divmod(int(gen_time_sec), 3600)
        minutes, seconds = divmod(remainder, 60)
        time_formatted = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        # --- Derived metrics ---
        sec_per_frame = gen_time_sec / num_frames if num_frames > 0 else 0.0
        frames_per_min = (num_frames / gen_time_sec * 60) if (num_frames > 0 and gen_time_sec > 0) else 0.0
        total_megapixels = (width * height * max(num_frames, 1)) / 1e6 if (width > 0 and height > 0) else 0.0
        megapixels_per_sec = total_megapixels / gen_time_sec if (total_megapixels > 0 and gen_time_sec > 0) else 0.0
        output_seconds = num_frames / fps if (num_frames > 0 and fps > 0) else 0.0
        realtime_ratio = (output_seconds / gen_time_sec * 60) if (output_seconds > 0 and gen_time_sec > 0) else 0.0
        sec_per_step = gen_time_sec / steps if steps > 0 else 0.0

        # --- Build report ---
        report = self._build_report(
            gen_time_sec, time_formatted, vram_peak_gb,
            width, height, num_frames, fps, steps, cfg,
            sampler_name, model_name,
            sec_per_frame, frames_per_min, total_megapixels,
            megapixels_per_sec, sec_per_step, output_seconds, realtime_ratio,
            notes,
        )

        print(report)

        # --- CSV logging ---
        if csv_path.strip():
            self._write_csv(
                csv_path, gen_time_sec, time_formatted,
                width, height, num_frames, fps, steps, cfg,
                sampler_name, model_name,
                sec_per_frame, frames_per_min, total_megapixels,
                megapixels_per_sec, sec_per_step, output_seconds,
                realtime_ratio, vram_peak_gb, notes,
            )

        return {
            "ui": {"text": [report]},
            "result": (
                report, gen_time_sec, sec_per_frame, frames_per_min,
                total_megapixels, megapixels_per_sec, output_seconds,
                realtime_ratio, passthrough,
            ),
        }

    @staticmethod
    def _build_report(gen_time_sec, time_formatted, vram_peak_gb,
                      width, height, num_frames, fps, steps, cfg,
                      sampler_name, model_name,
                      sec_per_frame, frames_per_min, total_megapixels,
                      megapixels_per_sec, sec_per_step, output_seconds,
                      realtime_ratio, notes):
        lines = []
        sep = "=" * 50
        lines.append(sep)
        lines.append("GENERATION TIMING REPORT")
        lines.append(sep)
        lines.append(f"Generation Time:  {time_formatted} ({gen_time_sec:.1f}s)")

        if vram_peak_gb > 0:
            lines.append(f"VRAM Peak:        {vram_peak_gb:.2f} GB")

        # Parameters section
        params = []
        if width > 0 and height > 0:
            params.append(f"Resolution:       {width} x {height}")
        if num_frames > 0:
            params.append(f"Frames:           {num_frames}")
        if fps > 0:
            params.append(f"FPS:              {fps}")
        if steps > 0:
            params.append(f"Steps:            {steps}")
        if cfg > 0:
            params.append(f"CFG:              {cfg}")
        if sampler_name:
            params.append(f"Sampler:          {sampler_name}")
        if model_name:
            params.append(f"Model:            {model_name}")

        if params:
            lines.append("")
            lines.extend(params)

        # Per-frame metrics
        if num_frames > 0:
            lines.append("")
            lines.append(f"Per-Frame:        {sec_per_frame:.2f} sec/frame | {frames_per_min:.2f} frames/min")

        # Throughput metrics
        if total_megapixels > 0:
            parts = [f"{total_megapixels:.2f} MP total | {megapixels_per_sec:.4f} MP/s"]
            if sec_per_step > 0:
                parts.append(f"{sec_per_step:.2f} sec/step")
            lines.append(f"Throughput:       {' | '.join(parts)}")

        # Video output metrics
        if output_seconds > 0:
            lines.append(f"Video Output:     {output_seconds:.2f}s duration | {realtime_ratio:.2f} vid_sec/gen_min")

        if notes:
            lines.append("")
            lines.append(f"Notes: {notes}")

        lines.append(sep)
        return "\n".join(lines)

    @staticmethod
    def _write_csv(csv_path, gen_time_sec, time_formatted,
                   width, height, num_frames, fps, steps, cfg,
                   sampler_name, model_name,
                   sec_per_frame, frames_per_min, total_megapixels,
                   megapixels_per_sec, sec_per_step, output_seconds,
                   realtime_ratio, vram_peak_gb, notes):
        if not os.path.isabs(csv_path):
            output_dir = folder_paths.get_output_directory()
            csv_path = os.path.join(output_dir, csv_path)

        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        file_exists = os.path.exists(csv_path)

        fieldnames = [
            "timestamp", "gen_time_sec", "time_formatted",
            "width", "height", "num_frames", "fps",
            "steps", "cfg", "sampler_name", "model_name",
            "sec_per_frame", "frames_per_min",
            "total_megapixels", "megapixels_per_sec",
            "output_seconds", "realtime_ratio", "sec_per_step",
            "vram_peak_gb", "notes",
        ]

        row = {
            "timestamp": datetime.now().isoformat(),
            "gen_time_sec": round(gen_time_sec, 2),
            "time_formatted": time_formatted,
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "fps": fps,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": sampler_name,
            "model_name": model_name,
            "sec_per_frame": round(sec_per_frame, 4),
            "frames_per_min": round(frames_per_min, 4),
            "total_megapixels": round(total_megapixels, 4),
            "megapixels_per_sec": round(megapixels_per_sec, 6),
            "output_seconds": round(output_seconds, 4),
            "realtime_ratio": round(realtime_ratio, 4),
            "sec_per_step": round(sec_per_step, 4),
            "vram_peak_gb": round(vram_peak_gb, 3),
            "notes": notes.replace("\n", " "),
        }

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

        print(f"[NV_GenTimer] Logged to {csv_path}")


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "NV_GenerationInfo": NV_GenerationInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_GenerationInfo": "NV Generation Info",
}
