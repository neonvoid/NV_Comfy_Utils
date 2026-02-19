"""
Pipeline Benchmark Timer

Checkpoint-style timer for comparing pipeline variants (e.g., pixel-space vs
latent-space stitching). Place at stage boundaries to accumulate per-stage timing.
The final instance writes a comparison CSV row.

Two chains flow through the workflow:
  1. Data chain: latent → mask → KSampler → decode → images (normal pipeline)
  2. Timing chain: timing_data STRING → from benchmark to benchmark

Usage:
    [Benchmark "start"]
        ↓ timing_data
    [LatentChunkStitcher] → latent
        ↓                      ↓
    [Benchmark "stitch", dep_latent=latent]
        ↓ timing_data
    [BoundaryNoiseMask] → [KSampler] → refined_latent
        ↓                                    ↓
    [Benchmark "refine", dep_latent=refined_latent]
        ↓ timing_data
    [StreamingVAEDecode] → images
        ↓                      ↓
    [Benchmark "decode", dep_images=images, write_csv=true]
        ↓
    comparison CSV row written
"""

import csv
import json
import os
import time
from datetime import datetime

import folder_paths


class NV_PipelineBenchmark:
    """
    Pipeline timing checkpoint. Place at stage boundaries to measure per-stage
    and total pipeline time. Chain timing_data from node to node.

    The first instance (stage="start") records the start time.
    Subsequent instances record elapsed time for each stage.
    The final instance (write_csv=True) writes a comparison CSV row.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stage": ("STRING", {
                    "default": "start",
                    "tooltip": "Name of the stage that just completed. Use 'start' for "
                               "the first checkpoint (records start time only)."
                }),
                "variant": ("STRING", {
                    "default": "default",
                    "tooltip": "Pipeline variant name for A/B comparison "
                               "(e.g., 'latent_blend', 'pixel_blend')."
                }),
            },
            "optional": {
                "timing_data": ("STRING", {
                    "forceInput": True,
                    "tooltip": "JSON from previous benchmark node. Leave unconnected "
                               "for the first (start) checkpoint."
                }),
                "dep_latent": ("LATENT", {
                    "tooltip": "Connect from stage output (LATENT) for DAG ordering."
                }),
                "dep_images": ("IMAGE", {
                    "tooltip": "Connect from stage output (IMAGE) for DAG ordering."
                }),
                "chunk_gen_seconds": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 100000.0, "step": 0.1,
                    "tooltip": "Total chunk generation time in seconds (from sweep). "
                               "Enter manually from GenTimer data. Only used by the "
                               "final node (write_csv=True) for total pipeline time."
                }),
                "quality_score": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1,
                    "tooltip": "User-assigned quality rating (0-10). Only used by "
                               "the final node (write_csv=True)."
                }),
                "csv_file": ("STRING", {
                    "default": "pipeline_comparison.csv",
                    "tooltip": "Path for comparison CSV. Relative paths resolve "
                               "against ComfyUI output dir. Empty to skip CSV."
                }),
                "write_csv": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable on the LAST benchmark node to write the CSV "
                               "row and print the full timing summary."
                }),
                "notes": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Freeform notes for this run."
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("timing_data", "summary")
    OUTPUT_NODE = True
    FUNCTION = "checkpoint"
    CATEGORY = "NV_Utils/Benchmark"
    DESCRIPTION = (
        "Pipeline timing checkpoint. Chain between stages to measure per-stage "
        "timing. Enable write_csv on the final node for comparison output."
    )

    def checkpoint(self, stage, variant, timing_data=None, dep_latent=None,
                   dep_images=None, chunk_gen_seconds=0.0, quality_score=0.0,
                   csv_file="pipeline_comparison.csv", write_csv=False, notes=""):
        now = time.time()

        # Parse or initialize timing data
        if timing_data and timing_data.strip():
            data = json.loads(timing_data)
        else:
            data = {"variant": variant, "checkpoints": []}

        # Always use the latest variant name
        data["variant"] = variant

        # Append this checkpoint
        data["checkpoints"].append({
            "stage": stage,
            "timestamp": now,
        })

        # Compute per-stage durations
        stages = _compute_stage_durations(data["checkpoints"])

        # Build summary
        summary = _build_summary(data["variant"], stages, chunk_gen_seconds,
                                 quality_score, notes)

        # Print progress
        if stage == "start":
            print(f"[PipelineBenchmark] Started timing for variant '{variant}'")
        else:
            elapsed = stages[-1]["duration"] if stages else 0.0
            print(f"[PipelineBenchmark] Stage '{stage}': {elapsed:.1f}s")

        # Write CSV if this is the final node
        if write_csv:
            print(summary)
            if csv_file and csv_file.strip():
                _write_csv(csv_file, data, stages, chunk_gen_seconds,
                           quality_score, notes)

        timing_json = json.dumps(data, indent=2)
        return (timing_json, summary)


def _compute_stage_durations(checkpoints):
    """Compute duration for each stage from consecutive timestamps."""
    stages = []
    for i in range(1, len(checkpoints)):
        prev = checkpoints[i - 1]
        curr = checkpoints[i]
        duration = curr["timestamp"] - prev["timestamp"]
        stages.append({
            "name": curr["stage"],
            "duration": duration,
        })
    return stages


def _build_summary(variant, stages, chunk_gen_seconds, quality_score, notes):
    """Build a human-readable timing summary."""
    sep = "=" * 50
    lines = [sep, f"PIPELINE BENCHMARK: {variant}", sep]

    if chunk_gen_seconds > 0:
        lines.append(f"  Chunk generation:  {_fmt_time(chunk_gen_seconds)} (manual input)")

    post_chunk_total = 0.0
    for s in stages:
        lines.append(f"  {s['name']:20s} {_fmt_time(s['duration'])}")
        post_chunk_total += s["duration"]

    if stages:
        lines.append(f"  {'Post-chunk total:':20s} {_fmt_time(post_chunk_total)}")

    total = chunk_gen_seconds + post_chunk_total
    if total > 0:
        lines.append(f"  {'FULL PIPELINE:':20s} {_fmt_time(total)}")

    if quality_score > 0:
        lines.append(f"  Quality score:     {quality_score:.1f}/10")

    if notes:
        lines.append(f"  Notes: {notes}")

    lines.append(sep)
    return "\n".join(lines)


def _fmt_time(seconds):
    """Format seconds as a human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {secs}s ({seconds:.1f}s)"
    hours, mins = divmod(minutes, 60)
    return f"{hours}h {mins}m {secs}s ({seconds:.1f}s)"


def _write_csv(csv_file, data, stages, chunk_gen_seconds, quality_score, notes):
    """Write a comparison CSV row."""
    if not os.path.isabs(csv_file):
        output_dir = folder_paths.get_output_directory()
        csv_file = os.path.join(output_dir, csv_file)

    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    file_exists = os.path.exists(csv_file)

    # Build stage duration map
    stage_map = {s["name"]: s["duration"] for s in stages}
    post_chunk_total = sum(s["duration"] for s in stages)
    total = chunk_gen_seconds + post_chunk_total

    # Dynamic fieldnames: fixed columns + any stage names found
    fixed_fields = [
        "timestamp", "variant", "chunk_gen_sec",
    ]
    stage_fields = [f"stage_{s['name']}_sec" for s in stages]
    tail_fields = [
        "post_chunk_sec", "total_sec", "quality_score", "notes",
    ]
    fieldnames = fixed_fields + stage_fields + tail_fields

    row = {
        "timestamp": datetime.now().isoformat(),
        "variant": data["variant"],
        "chunk_gen_sec": round(chunk_gen_seconds, 2),
    }
    for s in stages:
        row[f"stage_{s['name']}_sec"] = round(s["duration"], 2)
    row["post_chunk_sec"] = round(post_chunk_total, 2)
    row["total_sec"] = round(total, 2)
    row["quality_score"] = round(quality_score, 1)
    row["notes"] = notes.replace("\n", " ") if notes else ""

    with open(csv_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"[PipelineBenchmark] Logged to {csv_file}")


# Node registration
NODE_CLASS_MAPPINGS = {
    "NV_PipelineBenchmark": NV_PipelineBenchmark,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_PipelineBenchmark": "NV Pipeline Benchmark",
}
