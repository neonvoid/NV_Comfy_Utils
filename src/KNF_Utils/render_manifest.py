"""
NV Render Manifest - Consolidate all pipeline parameters into a JSON sidecar.

The user's workflow scatters config across 60+ nodes. When debugging a render
weeks later, recovering "what was the CoTracker strength? what was the bbox
expand? was Track B on?" requires reading the workflow JSON in PNG metadata
and chasing connections — useless for "what did I actually render with."

This node consolidates everything into one structured JSON file written next
to the video output. Survives ComfyUI restarts, browser sessions, etc.

Design:
- Optional inputs for every metadata source (info STRINGs, config buses, raw values)
- Auto-extracts what it can from structured inputs (stitcher dict, mask_config bus)
- Free-form notes field for human commentary
- Trigger input (passthrough) lets the node execute LAST in the dependency graph
- Output: manifest STRING + trigger passthrough

Place at the end of the pipeline, gated by a trigger input from the video saver
(or any final node) so it captures the post-execution state.
"""

import copy
import json
import time

# Shared ops — single source of truth. Helpers used to live here; they're now
# in manifest_ops.py so NV_RenderManifest and NV_ShotRecord see identical
# summaries and there's no risk of one copy drifting from the other.
from .manifest_ops import (
    summarize_mask_config as _summarize_mask_config,
    summarize_stitcher as _summarize_stitcher,
)


class NV_RenderManifest:
    """Per-pass manifest aggregator + cross-pass flow carrier.

    Two roles:
      1. Aggregates scattered upstream config (info STRINGs from per-node info
         outputs, sampler widget values, stitcher/mask_config summaries) into
         one structured dict.
      2. Carries that dict between passes via the RENDER_MANIFEST custom type
         (manifest_in → manifest_out). Multi-pass workflows (e.g. SimpleColorMatch
         chain) need this so cross-pass state (`color_state`, etc.) survives.

    Disk-write was removed 2026-04-29 — NV_ShotRecord now writes structured
    JSONL records to agent_log_inbox/, which supersedes the per-render JSON
    sidecar this node used to emit. Wire `manifest_out` into NV_ShotRecord's
    `manifest_in` to persist.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shot_name": ("STRING", {
                    "default": "render",
                    "tooltip": "Shot identifier — recorded in the manifest dict so downstream "
                               "NV_ShotRecord can pick it up if no shot_id widget is set there."
                }),
            },
            "optional": {
                "trigger_in": ("*", {
                    "tooltip": "Optional trigger to gate execution. Wire from any final-stage node "
                               "(e.g., video saver output) so the manifest writes AFTER the render "
                               "completes. ComfyUI executes nodes in dependency order."
                }),
                "notes": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "Free-form session notes. What were you testing? What's different "
                               "about this render? Future-you will thank present-you."
                }),
                # Structured upstream metadata sources
                "stitcher": ("STITCHER", {
                    "tooltip": "Stitcher dict from NV_InpaintCrop2 / NV_CoTrackerBridge. Auto-extracts "
                               "bbox extents per frame, canvas dims, content_warp_mode, etc."
                }),
                "mask_config": ("MASK_BLEND_CONFIG", {
                    "tooltip": "NV_MaskBlendConfig output. Auto-extracts BLEND-side mask processing "
                               "overrides (cleanup, crop_expand, blend_feather, stitch erosion/feather). "
                               "GEN-side params are recorded via the consuming node's info string instead."
                }),
                # Info STRING inputs (from each node's info output)
                "static_vace_mask_info": ("STRING", {"forceInput": True, "tooltip": "info from NV_StaticVaceMask"}),
                "vace_prep_info": ("STRING", {"forceInput": True, "tooltip": "info from NV_VaceControlVideoPrep"}),
                "vace_prepass_info": ("STRING", {"forceInput": True, "tooltip": "info from NV_VacePrePassReference"}),
                "track_b_info": ("STRING", {"forceInput": True, "tooltip": "info from NV_BboxAlignedMaskStabilizer"}),
                "point_bbox_info": ("STRING", {"forceInput": True, "tooltip": "info from NV_PointDrivenBBox"}),
                "opt_crop_info": ("STRING", {"forceInput": True, "tooltip": "info from NV_OptimizeCropTrajectory"}),
                "cotracker_info": ("STRING", {"forceInput": True, "tooltip": "info from NV_CoTrackerBridge"}),
                "color_match_info": ("STRING", {"forceInput": True, "tooltip": "info from NV_SimpleColorMatch or NV_CropColorFix"}),
                "texture_harmonize_info": ("STRING", {"forceInput": True, "tooltip": "info from NV_TextureHarmonize"}),
                "stitch_info": ("STRING", {"forceInput": True, "tooltip": "info from NV_InpaintStitch2 (free-form, not a real output yet)"}),
                # Raw values (force_input — wire from widgets that don't have info outputs)
                "cotracker_strength": ("FLOAT", {"forceInput": True, "tooltip": "CoTrackerBridge strength"}),
                "cfg": ("FLOAT", {"forceInput": True, "tooltip": "KSampler CFG"}),
                "denoise": ("FLOAT", {"forceInput": True, "tooltip": "KSampler denoise"}),
                "sampler_steps": ("INT", {"forceInput": True, "tooltip": "KSampler total steps"}),
                "sampler_name": ("STRING", {"forceInput": True, "tooltip": "Sampler name"}),
                # Custom JSON blob — paste anything else
                "custom_json_extras": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "Additional JSON to merge into manifest under 'extras' key. "
                               "Must parse as JSON object. Empty = skipped."
                }),
                # Flowing manifest dict from upstream nodes (e.g., NV_SimpleColorMatch).
                # Forms the BASE manifest; widget/info inputs above merge on top so they
                # only OVERRIDE when explicitly set (non-empty). This is what makes the
                # manifest a single flowing source of truth across the workflow.
                "manifest_in": ("RENDER_MANIFEST", {
                    "tooltip": "Optional flowing render-manifest dict from upstream nodes "
                               "(typically the last NV_SimpleColorMatch in the chain). When "
                               "wired, its keys form the BASE manifest; widget/info inputs "
                               "above are merged ON TOP (widget values override only when "
                               "explicitly non-empty). Carries `color_state` and any other "
                               "node-contributed metadata into the JSON sidecar with zero "
                               "user paste effort."
                }),
            },
        }

    RETURN_TYPES = ("STRING", "*", "RENDER_MANIFEST")
    RETURN_NAMES = ("manifest_json", "trigger_out", "manifest_out")
    FUNCTION = "execute"
    CATEGORY = "NV_Utils/Debug"
    DESCRIPTION = (
        "Aggregates pipeline params from upstream info STRINGs + sampler widget "
        "values + stitcher/mask_config into one manifest dict. Pipes via "
        "RENDER_MANIFEST custom type for cross-pass workflows. Wire "
        "manifest_out → NV_ShotRecord.manifest_in to persist."
    )

    def execute(
        self,
        shot_name,
        trigger_in=None,
        notes="",
        stitcher=None,
        mask_config=None,
        static_vace_mask_info=None,
        vace_prep_info=None,
        vace_prepass_info=None,
        track_b_info=None,
        point_bbox_info=None,
        opt_crop_info=None,
        cotracker_info=None,
        color_match_info=None,
        texture_harmonize_info=None,
        stitch_info=None,
        cotracker_strength=None,
        cfg=None,
        denoise=None,
        sampler_steps=None,
        sampler_name=None,
        custom_json_extras="",
        manifest_in=None,
    ):
        TAG = "[NV_RenderManifest]"

        # Start from upstream manifest if wired, else fresh dict.
        # Multi-AI fix: DEEP-copy (was shallow). Branching workflows where two
        # downstream nodes consume the same upstream manifest must each get an
        # independent snapshot — shallow copy aliased nested objects and broke
        # the flowing-carrier contract.
        if isinstance(manifest_in, dict):
            manifest = copy.deepcopy(manifest_in)
        else:
            manifest = {}

        # Always refresh timestamp; widget values override upstream only when explicitly set
        manifest["render_timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        if shot_name:
            manifest["shot_name"] = shot_name
        elif "shot_name" not in manifest:
            manifest["shot_name"] = "unnamed"
        if notes:
            manifest["notes"] = notes
        elif "notes" not in manifest:
            manifest["notes"] = ""

        # Structured extractions
        sti_summary = _summarize_stitcher(stitcher)
        if sti_summary is not None:
            manifest["stitcher"] = sti_summary

        mc_summary = _summarize_mask_config(mask_config)
        if mc_summary is not None:
            manifest["mask_processing_config"] = mc_summary

        # Info STRINGs — MERGE with any upstream node_info rather than overwrite.
        # Widget-supplied values override upstream per-key; upstream-only entries
        # (e.g., a node downstream of an upstream RenderManifest that contributed
        # something we don't have a widget for) are preserved.
        existing_info = manifest.get("node_info")
        info_block = dict(existing_info) if isinstance(existing_info, dict) else {}
        for name, value in [
            ("static_vace_mask", static_vace_mask_info),
            ("vace_control_video_prep", vace_prep_info),
            ("vace_prepass_reference", vace_prepass_info),
            ("track_b_mask_stabilizer", track_b_info),
            ("point_driven_bbox", point_bbox_info),
            ("optimize_crop_trajectory", opt_crop_info),
            ("cotracker_bridge", cotracker_info),
            ("color_match", color_match_info),
            ("texture_harmonize", texture_harmonize_info),
            ("inpaint_stitch", stitch_info),
        ]:
            if value:
                info_block[name] = value
        if info_block:
            manifest["node_info"] = info_block

        # Sampler/scalar block — same merge semantics as node_info.
        existing_sampler = manifest.get("sampler")
        sampler_block = dict(existing_sampler) if isinstance(existing_sampler, dict) else {}
        if cotracker_strength is not None:
            sampler_block["cotracker_strength"] = float(cotracker_strength)
        if cfg is not None:
            sampler_block["cfg"] = float(cfg)
        if denoise is not None:
            sampler_block["denoise"] = float(denoise)
        if sampler_steps is not None:
            sampler_block["sampler_steps"] = int(sampler_steps)
        if sampler_name:
            sampler_block["sampler_name"] = str(sampler_name)
        if sampler_block:
            manifest["sampler"] = sampler_block

        # Custom extras — explicit clear-on-state-change to prevent stale data
        # leaking across runs (multi-AI fix). Three branches:
        #   - widget non-empty + parses OK   → set extras, drop any prior parse_error
        #   - widget non-empty + parse fails → drop any prior extras, set parse_error
        #   - widget empty                   → drop any prior parse_error (keep upstream extras)
        if custom_json_extras and custom_json_extras.strip():
            try:
                manifest["extras"] = json.loads(custom_json_extras)
                manifest.pop("extras_parse_error", None)
            except json.JSONDecodeError as e:
                manifest.pop("extras", None)
                manifest["extras_parse_error"] = f"{e}"
        else:
            manifest.pop("extras_parse_error", None)

        # Serialize for the human-readable STRING output (workflow inspection).
        # No `default=str` — non-serializable payloads fail loudly so upstream
        # contributors don't silently ship lossy data downstream into the
        # JSONL agent corpus via NV_ShotRecord.
        try:
            manifest_json = json.dumps(manifest, indent=2, sort_keys=True)
        except TypeError as e:
            raise TypeError(
                f"{TAG} Manifest contains non-JSON-serializable payload: {e}. "
                f"All upstream contributors must emit JSON-safe values "
                f"(strings, numbers, bools, lists, dicts, None)."
            )

        print(f"{TAG} manifest contains {len(manifest)} top-level keys; "
              f"wire manifest_out → NV_ShotRecord to persist.")

        # Emit (json_string, trigger_passthrough, manifest_dict).
        # The dict output lets downstream nodes (chained NV_RenderManifest
        # instances or NV_ShotRecord) keep flowing the central source of truth
        # without re-parsing JSON.
        return (manifest_json, trigger_in if trigger_in is not None else manifest_json, manifest)


NODE_CLASS_MAPPINGS = {
    "NV_RenderManifest": NV_RenderManifest,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_RenderManifest": "NV Render Manifest",
}
