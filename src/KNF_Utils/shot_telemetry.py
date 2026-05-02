"""NV_ShotMeasure + NV_ShotRecord — the consolidated telemetry pipeline.

Replaces the scattered analyzer + manifest + memory + timing nodes. All
logic lives in shared ops modules; these classes are thin wrappers that
make the ops available in ComfyUI workflows.

WIRING (typical):
    [pipeline] → result_image
    [InpaintCrop2] → source_image, mask, stitcher
                ↓
        NV_ShotMeasure(source_image, result_image, mask, stitcher,
                       chunk_plan_json?) → SHOT_METRICS dict
                ↓
        NV_ShotRecord(metrics, manifest_in, shot_id, verdict_*, trigger)
                → writes one .jsonl line to agent_log_inbox/{MACHINE_ID}.jsonl
"""

import json
import math
import time
import uuid

from comfy.comfy_types.node_typing import IO

from .diff_ops import compute_diff_metrics
from .manifest_ops import summarize_mask_config, summarize_stitcher
from .seam_ops import compute_seam_metrics
from .shot_fingerprint import compute_fingerprint
from .shot_jsonl_writer import (
    append_record,
    count_records,
    get_machine_id,
    resolve_inbox_path,
)
from .shot_runtime import get_runtime_snapshot
from .shot_telemetry_types import (
    RUBRIC_VERSION,
    SCHEMA_VERSION,
)


_JSON_SAFE_PRIMITIVES = (str, int, bool, type(None))


def _to_json_safe(value, _depth=0):
    """Recursively convert arbitrary values into JSON-safe primitives.

    Tensors, numpy arrays, paths, sets, and other non-JSON types get
    stringified rather than failing the whole record. Non-finite floats
    (NaN/Infinity) become None — strict JSON readers reject them.

    Capped at depth 8 to defend against pathological cycles.
    """
    if _depth > 8:
        return f"<truncated:depth>"
    if isinstance(value, _JSON_SAFE_PRIMITIVES):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v, _depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(v, _depth + 1) for v in value]
    if isinstance(value, (set, frozenset)):
        return [_to_json_safe(v, _depth + 1) for v in value]
    # numpy scalars / tensors / arrays / Paths / etc.
    for attr in ("tolist", "item"):
        fn = getattr(value, attr, None)
        if callable(fn):
            try:
                return _to_json_safe(fn(), _depth + 1)
            except (TypeError, ValueError):
                pass
    return f"<non_json:{type(value).__name__}>"


# ---------------------------------------------------------------------------
# NV_ShotMeasure — pure compute, emits structured dict
# ---------------------------------------------------------------------------
class NV_ShotMeasure:
    """Computes diff metrics + seam metrics + shot fingerprint into one dict.

    No file I/O, no widgets for runtime/verdict — that's NV_ShotRecord's job.
    Place between the pipeline output and NV_ShotRecord.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_image": ("IMAGE", {
                    "tooltip": "Pre-render input plate (post-InpaintCrop2 crop). What VACE sees as input."
                }),
                "result_image": ("IMAGE", {
                    "tooltip": "Post-render output (post-stitch or pre-stitch crop)."
                }),
            },
            "optional": {
                "mask": ("MASK", {
                    "tooltip": "Segmentation mask. Used for diff zone splits and fingerprint occupancy."
                }),
                "stitcher": ("STITCHER", {
                    "tooltip": "Stitcher dict from InpaintCrop2. Used for fingerprint motion and bbox trajectory."
                }),
                "chunk_plan_json": ("STRING", {
                    "default": "", "forceInput": True,
                    "tooltip": "Optional JSON from NV_AspectChunkPlanner. If wired, seam metrics computed at chunk boundaries."
                }),
                "boundary_width_px": ("INT", {
                    "default": 16, "min": 1, "max": 128, "step": 1,
                    "tooltip": "Pixels around mask edge that count as 'boundary kept' for seam-localized diff stats."
                }),
            },
        }

    RETURN_TYPES = ("SHOT_METRICS", "STRING")
    RETURN_NAMES = ("metrics", "metrics_summary")
    FUNCTION = "measure"
    CATEGORY = "NV_Utils/Telemetry"
    DESCRIPTION = (
        "Computes shot fingerprint, diff metrics (interior vs boundary), "
        "and seam continuity scalars. Feeds NV_ShotRecord."
    )

    def measure(self, source_image, result_image, mask=None, stitcher=None,
                chunk_plan_json="", boundary_width_px=16):
        result = {
            "schema_version": SCHEMA_VERSION,
            "fingerprint": compute_fingerprint(source_image, mask, stitcher),
            "diff": compute_diff_metrics(source_image, result_image, mask, boundary_width=boundary_width_px),
            "seam": compute_seam_metrics(result_image, chunk_plan_json=chunk_plan_json),
        }
        # Human-readable one-line summary for inline workflow inspection
        fp = result["fingerprint"]
        diff = result["diff"]
        seam = result["seam"]
        summary = (
            f"regime={'/'.join(fp.get('regime_tags', []))} | "
            f"diff_boundary_p95={diff.get('boundary_p95', 'n/a')} | "
            f"diff_interior_mean={diff.get('interior_mean', 'n/a')} | "
            f"seam_count={seam.get('seam_count', 0)} | "
            f"psnr_min={seam.get('psnr_min', 'n/a')}"
        )
        return (result, summary)


# ---------------------------------------------------------------------------
# NV_ShotRecord — sink, writes one .jsonl record per render
# ---------------------------------------------------------------------------
class NV_ShotRecord:
    """End-of-workflow drain. Bundles params + fingerprint + metrics + runtime
    into one structured record and appends one line to
    agent_log_inbox/{MACHINE_ID}.jsonl.

    VERDICT FIELDS ARE WRITTEN AS NULL placeholders. The operator fills them
    in by editing the .jsonl AFTER viewing the render output. The widgets-at-
    queue-time pattern was abandoned because it asks the operator to rate a
    render they haven't seen yet — that data is noise, not signal.

    Each record's `verdict` block is shaped:
        "verdict": {
            "overall":  null,   // set to 1-5 after viewing
            "identity": null,
            "temporal": null,
            "artifacts": [],    // controlled-vocab veto tags (see DISQUALIFYING_ARTIFACT_TAGS)
            "notes":    ""
        }

    Trigger from SaveVideo to guarantee post-render execution.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trigger": (IO.ANY, {
                    "tooltip": "Wire from SaveVideo / final node so this runs after the render completes."
                }),
                "shot_id": ("STRING", {
                    "default": "unnamed_shot",
                    "tooltip": "Stable cross-render key for this shot. Use the SAME id across re-runs of the same source."
                }),
                "inbox_dir": ("STRING", {
                    "default": "node_notes/agent_log_inbox/{MACHINE_ID}.jsonl",
                    "tooltip": "Path to .jsonl inbox. {MACHINE_ID} is auto-substituted from hostname (or NV_MACHINE_ID env var)."
                }),
            },
            "optional": {
                "metrics": ("SHOT_METRICS", {
                    "tooltip": "From NV_ShotMeasure. Optional — record can persist params without metrics."
                }),
                "manifest_in": ("RENDER_MANIFEST", {
                    "tooltip": "From NV_RenderManifest or upstream node. Param state of the render."
                }),
                "stitcher": ("STITCHER", {
                    "tooltip": "Optional fallback if no manifest_in — extracts bbox trajectory directly."
                }),
                "mask_config": ("MASK_BLEND_CONFIG", {
                    "tooltip": "Optional fallback if no manifest_in — extracts BLEND-side mask "
                               "processing params. GEN-side params come via the consuming node's "
                               "info string."
                }),
                "render_status": (["completed", "failed", "oom", "aborted"], {
                    "default": "completed",
                    "tooltip": "Outcome status. 'failed' / 'oom' records still useful — they teach safe bounds."
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", IO.ANY)
    RETURN_NAMES = ("record_json", "inbox_path_resolved", "record_count", "trigger_passthrough")
    FUNCTION = "record"
    CATEGORY = "NV_Utils/Telemetry"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Writes one structured record per render to agent_log_inbox/{MACHINE_ID}.jsonl. "
        "Append-only, atomic, lock-protected. Verdict fields are null placeholders — "
        "fill them in by editing the .jsonl after viewing the render. Feeds NV_AgentParamPlanner."
    )

    def record(self, trigger, shot_id, inbox_dir,
               metrics=None, manifest_in=None, stitcher=None, mask_config=None,
               render_status="completed"):

        # Build params dict — prefer manifest_in (richer), fall back to direct stitcher/mask_config.
        # Sanitize through _to_json_safe: manifests can carry tensors/numpy/paths
        # that crash json.dumps. Sanitizing at the boundary keeps the JSONL
        # contract strict without forcing every upstream node to be JSON-aware.
        params = {}
        if isinstance(manifest_in, dict):
            params = _to_json_safe(manifest_in)
        if stitcher is not None and "stitcher" not in params:
            sti = summarize_stitcher(stitcher)
            if sti is not None:
                params["stitcher"] = _to_json_safe(sti)
        if mask_config is not None and "mask_processing_config" not in params:
            mc = summarize_mask_config(mask_config)
            if mc is not None:
                params["mask_processing_config"] = _to_json_safe(mc)

        # Runtime snapshot — non-destructive, returns None for unavailable fields
        runtime = get_runtime_snapshot()

        record = {
            "schema_version": SCHEMA_VERSION,
            "rubric_version": RUBRIC_VERSION,
            "record_id": uuid.uuid4().hex,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "machine_id": get_machine_id(),
            "shot_id": shot_id.strip() or "unnamed_shot",
            "render_status": render_status,
            "params": params,
            "fingerprint": _to_json_safe(metrics.get("fingerprint")) if isinstance(metrics, dict) else None,
            "metrics": {
                "diff": _to_json_safe(metrics.get("diff")) if isinstance(metrics, dict) else None,
                "seam": _to_json_safe(metrics.get("seam")) if isinstance(metrics, dict) else None,
            },
            "performance": {
                # gen_time_sec is None when the prompt hook didn't install — never silently 0.
                "gen_time_sec": runtime.get("gen_time_sec"),
                "gen_time_scope": runtime.get("gen_time_scope"),
                "vram_session_peak_gb": runtime.get("vram_session_peak_gb"),  # NOT shot-scoped
                "vram_total_gb": runtime.get("vram_total_gb"),
                "ram_used_gb": runtime.get("ram_used_gb"),
                "gpu_name": runtime.get("gpu_name"),
                "hook_installed": runtime.get("hook_installed"),
                "hook_install_error": runtime.get("hook_install_error"),
                "vram_error": runtime.get("vram_error"),
                "ram_error": runtime.get("ram_error"),
            },
            # Verdict block — null placeholders. Operator fills these in by
            # editing the .jsonl directly AFTER viewing the render output.
            # Schema is fixed so the agent's parser can rely on shape; values
            # are null/empty until human judgment is recorded.
            #   - overall/identity/temporal: integer 1-5 per the locked rubric
            #     (see VERDICT_RUBRIC in shot_telemetry_types.py)
            #   - artifacts: array of strings from DISQUALIFYING_ARTIFACT_TAGS
            #     (any non-empty entry is a binary VETO for the agent)
            #   - notes: free-form
            "verdict": {
                "overall": None,
                "identity": None,
                "temporal": None,
                "artifacts": [],
                "notes": "",
            },
        }

        path = resolve_inbox_path(inbox_dir)
        # Let exceptions propagate. Silent telemetry loss is the failure mode
        # we explicitly designed against — better the workflow surfaces a
        # red error node than the agent corpus quietly misses records.
        append_record(path, record)
        count = count_records(path)

        record_json_out = json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False)
        print(f"[NV_ShotRecord] appended record {record['record_id'][:8]} to {path} ({count} total)")
        return (record_json_out, path, count, trigger)


NODE_CLASS_MAPPINGS = {
    "NV_ShotMeasure": NV_ShotMeasure,
    "NV_ShotRecord": NV_ShotRecord,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_ShotMeasure": "NV Shot Measure",
    "NV_ShotRecord": "NV Shot Record",
}
