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

import copy
import json
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
from .shot_runtime import get_gen_time_sec, get_memory_snapshot
from .shot_telemetry_types import (
    DISQUALIFYING_ARTIFACT_TAGS,
    RUBRIC_VERSION,
    SCHEMA_VERSION,
)


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
    """End-of-workflow drain. Bundles params + fingerprint + metrics + verdict
    + runtime (timing/memory) into one structured record and appends one line
    to agent_log_inbox/{MACHINE_ID}.jsonl.

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
                "verdict_overall": ("INT", {
                    "default": 0, "min": 0, "max": 5, "step": 1,
                    "tooltip": "Overall quality 1-5 per locked rubric. 0 = not yet rated (record kept, marked unrated)."
                }),
                "verdict_identity": ("INT", {
                    "default": 0, "min": 0, "max": 5, "step": 1,
                    "tooltip": "Identity preservation 1-5. 0 = not rated."
                }),
                "verdict_temporal": ("INT", {
                    "default": 0, "min": 0, "max": 5, "step": 1,
                    "tooltip": "Temporal stability 1-5. 0 = not rated."
                }),
            },
            "optional": {
                "metrics": ("SHOT_METRICS", {
                    "tooltip": "From NV_ShotMeasure. Optional — record can persist params+verdict without metrics."
                }),
                "manifest_in": ("RENDER_MANIFEST", {
                    "tooltip": "From NV_RenderManifest or upstream node. Param state of the render."
                }),
                "stitcher": ("STITCHER", {
                    "tooltip": "Optional fallback if no manifest_in — extracts bbox trajectory directly."
                }),
                "mask_config": ("MASK_PROCESSING_CONFIG", {
                    "tooltip": "Optional fallback if no manifest_in — extracts mask processing params."
                }),
                "disqualifying_artifacts": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": (
                        "Comma- or newline-separated tags from controlled vocab. Any non-empty tag = "
                        "binary VETO for the agent. Vocab: " + ", ".join(DISQUALIFYING_ARTIFACT_TAGS)
                    ),
                }),
                "render_status": (["completed", "failed", "oom", "aborted"], {
                    "default": "completed",
                    "tooltip": "Outcome status. 'failed' / 'oom' records still useful — they teach safe bounds."
                }),
                "notes": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "Free-form session notes. Will be summarized into agent context, not parsed structurally."
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
        "Append-only, atomic, lock-protected. Feeds NV_AgentParamPlanner."
    )

    @staticmethod
    def _parse_artifact_tags(raw):
        if not raw or not raw.strip():
            return []
        # Accept comma- or newline-separated; strip whitespace; dedupe; preserve order
        chunks = []
        for line in raw.replace(",", "\n").split("\n"):
            t = line.strip().lower()
            if t and t not in chunks:
                chunks.append(t)
        return chunks

    def record(self, trigger, shot_id, inbox_dir,
               verdict_overall, verdict_identity, verdict_temporal,
               metrics=None, manifest_in=None, stitcher=None, mask_config=None,
               disqualifying_artifacts="", render_status="completed", notes=""):

        # Build params dict — prefer manifest_in (richer), fall back to direct stitcher/mask_config
        params = {}
        if isinstance(manifest_in, dict):
            params = copy.deepcopy(manifest_in)
        if stitcher is not None and "stitcher" not in params:
            sti = summarize_stitcher(stitcher)
            if sti is not None:
                params["stitcher"] = sti
        if mask_config is not None and "mask_processing_config" not in params:
            mc = summarize_mask_config(mask_config)
            if mc is not None:
                params["mask_processing_config"] = mc

        # Runtime snapshot — non-destructive
        gen_time = get_gen_time_sec()
        mem = get_memory_snapshot()

        record = {
            "schema_version": SCHEMA_VERSION,
            "rubric_version": RUBRIC_VERSION,
            "record_id": uuid.uuid4().hex,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "machine_id": get_machine_id(),
            "shot_id": shot_id.strip() or "unnamed_shot",
            "render_status": render_status,
            "params": params,
            "fingerprint": metrics.get("fingerprint") if isinstance(metrics, dict) else None,
            "metrics": {
                "diff": metrics.get("diff") if isinstance(metrics, dict) else None,
                "seam": metrics.get("seam") if isinstance(metrics, dict) else None,
            },
            "performance": {
                "gen_time_sec": round(gen_time, 3) if gen_time > 0 else None,
                "vram_peak_gb": mem.get("vram_peak_gb"),
                "vram_total_gb": mem.get("vram_total_gb"),
                "ram_used_gb": mem.get("ram_used_gb"),
                "gpu_name": mem.get("gpu_name"),
                "hook_installed": mem.get("hook_installed"),
            },
            "verdict": {
                "overall": verdict_overall if verdict_overall > 0 else None,
                "identity": verdict_identity if verdict_identity > 0 else None,
                "temporal": verdict_temporal if verdict_temporal > 0 else None,
                "artifacts": self._parse_artifact_tags(disqualifying_artifacts),
                "notes": notes.strip(),
            },
        }

        path = resolve_inbox_path(inbox_dir)
        success = append_record(path, record)
        count = count_records(path) if success else 0

        record_json_out = json.dumps(record, indent=2, ensure_ascii=False)
        if success:
            print(f"[NV_ShotRecord] appended record {record['record_id'][:8]} to {path} ({count} total)")
        else:
            print(f"[NV_ShotRecord] WARNING: append failed; record returned but not persisted")

        return (record_json_out, path, count, trigger)


NODE_CLASS_MAPPINGS = {
    "NV_ShotMeasure": NV_ShotMeasure,
    "NV_ShotRecord": NV_ShotRecord,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_ShotMeasure": "NV Shot Measure",
    "NV_ShotRecord": "NV Shot Record",
}
