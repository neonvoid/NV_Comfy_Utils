"""
NV Fixture Dumper — ad-hoc parity-test fixture extraction.

A passive observer node for capturing the inputs + golden outputs of
NV_InpaintCrop2 + NV_InpaintStitch2 to disk, for use as parity-test
fixtures by the NV_Interactive_Masking_Suite editor.

Wire pattern in a workflow:

    NV_InpaintCrop2 ─┬─> KSampler ─┬─> NV_InpaintStitch2 ─> output
                     │             │           │
                     │             │           └─> NV_FixtureDumper.comfy_output
                     │             └─────────────> NV_FixtureDumper.inpainted_image
                     └───────────────────────────> NV_FixtureDumper.stitcher

The node is purely side-effecting: it reads its inputs, slices a window
of frames, writes a .safetensors + .json pair under output_dir, and
returns a short info string. It does NOT mutate any inputs and is safe
to add to a working pipeline.

Output format is deliberately ad-hoc (not the eventual production bundle
schema) — the editor's bundle format isn't locked yet, and we don't want
to design it before the renderer's runtime needs are known.

Output layout:
  {output_dir}/{shot_id}_{num_frames}f.safetensors   stacked tensors
  {output_dir}/{shot_id}_{num_frames}f.json          geometry + params + metadata
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

import folder_paths


def _parse_pseudo_yaml(s: str) -> dict:
    """Minimal `key: value` parser. Numeric values are int/float; "true"/"false"
    are bools; everything else is a string. Lines starting with # are skipped.

    We deliberately don't pull in PyYAML for this — the param dump is small
    and hand-edited, and the user shouldn't need a yaml dependency to capture
    a fixture.
    """
    out: dict = {}
    for line in s.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()
        if val.lower() in ("true", "false"):
            out[key] = val.lower() == "true"
            continue
        try:
            if "." in val or "e" in val.lower():
                out[key] = float(val)
            else:
                out[key] = int(val)
        except ValueError:
            out[key] = val
    return out


def _slice_list(lst, offset: int, count: int):
    """Slice a list to a window, gracefully handling shorter inputs."""
    if not lst:
        return []
    end = min(len(lst), offset + count)
    return lst[offset:end]


def _stack_or_none(tensor_list, name: str):
    """Stack a list of same-shape tensors, or return None if empty."""
    if not tensor_list:
        return None
    try:
        return torch.stack([t.cpu() for t in tensor_list], dim=0).contiguous()
    except RuntimeError as e:
        raise RuntimeError(f"[NV_FixtureDumper] Failed to stack '{name}': {e}") from e


class NV_FixtureDumper:
    """Dump NV_InpaintCrop2 + NV_InpaintStitch2 inputs/outputs to disk for parity tests.

    Output is a (.safetensors, .json) pair under {output_dir}. The .safetensors
    holds stacked per-frame tensors; the .json holds geometry, params, and metadata.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stitcher": ("STITCHER", {
                    "tooltip": "STITCHER dict from NV_InpaintCrop2 (canvas_image, canvas_mask, "
                               "canvas_mask_processed, cropped_mask_for_blend, geometry lists)."
                }),
                "inpainted_image": ("IMAGE", {
                    "tooltip": "KSampler output before NV_InpaintStitch2 — the diffused crop."
                }),
                "comfy_output": ("IMAGE", {
                    "tooltip": "Output of NV_InpaintStitch2 — the golden composite that the "
                               "editor must reproduce within tolerance."
                }),
                "shot_id": ("STRING", {
                    "default": "jcrew_001",
                    "tooltip": "Naming prefix for the output files. e.g. 'jcrew_walking_001'."
                }),
                "num_frames": ("INT", {
                    "default": 8, "min": 1, "max": 64, "step": 1,
                    "tooltip": "How many frames to dump. Smaller = faster, smaller fixture."
                }),
                "frame_offset": ("INT", {
                    "default": 0, "min": 0, "max": 999, "step": 1,
                    "tooltip": "Starting frame index. Pick a representative segment "
                               "(e.g. a head turn or motion peak)."
                }),
                "crop_params": ("STRING", {
                    "multiline": True,
                    "default": (
                        "# Copy these from your NV_InpaintCrop2 widget values.\n"
                        "# build_blend_mask parity tests will read this back.\n"
                        "crop_stitch_source: processed\n"
                        "crop_blend_feather_px: 16\n"
                        "hybrid_falloff: 48\n"
                        "hybrid_curve: 0.6\n"
                        "cleanup_fill_holes: 0\n"
                        "cleanup_remove_noise: 0\n"
                        "cleanup_smooth: 0\n"
                        "crop_expand_px: 0\n"
                    ),
                    "tooltip": "Paste your NV_InpaintCrop2 widget values here. "
                               "Used by the parity test harness to reproduce the same blend mask."
                }),
                "stitch_params": ("STRING", {
                    "multiline": True,
                    "default": (
                        "# Copy these from your NV_InpaintStitch2 widget values.\n"
                        "# composite_frame parity tests will read this back.\n"
                        "blend_mode: multiband\n"
                        "multiband_levels: 5\n"
                        "guided_refine: false\n"
                        "guided_radius: 8\n"
                        "guided_eps: 0.001\n"
                        "guided_strength: 0.7\n"
                    ),
                    "tooltip": "Paste your NV_InpaintStitch2 widget values here."
                }),
                "output_dir": ("STRING", {
                    "default": "fixtures",
                    "tooltip": "Output dir relative to ComfyUI's output folder. "
                               "Will be created if missing."
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    FUNCTION = "dump"
    CATEGORY = "NV_Utils/Debug"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Dumps NV_InpaintCrop2 + NV_InpaintStitch2 tensors to disk as a parity-test "
        "fixture for the NV Interactive Masking Suite editor. Tee this in parallel "
        "with NV_InpaintStitch2 on a working pipeline; it side-effects only, never "
        "mutates the dataflow."
    )

    def dump(self, stitcher, inpainted_image, comfy_output,
             shot_id, num_frames, frame_offset,
             crop_params, stitch_params, output_dir):

        out_root = Path(folder_paths.get_output_directory()) / output_dir
        out_root.mkdir(parents=True, exist_ok=True)

        base_name = f"{shot_id}_{num_frames}f"
        tensor_path = out_root / f"{base_name}.safetensors"
        meta_path = out_root / f"{base_name}.json"

        # ----- Slice the per-frame data to the requested window ---------
        offset, count = int(frame_offset), int(num_frames)

        canvas_images = _slice_list(stitcher.get("canvas_image", []), offset, count)
        canvas_masks = _slice_list(stitcher.get("canvas_mask", []), offset, count)
        canvas_masks_processed = _slice_list(stitcher.get("canvas_mask_processed", []), offset, count)
        cropped_blend_masks = _slice_list(stitcher.get("cropped_mask_for_blend", []), offset, count)

        ctc_x = _slice_list(stitcher.get("cropped_to_canvas_x", []), offset, count)
        ctc_y = _slice_list(stitcher.get("cropped_to_canvas_y", []), offset, count)
        ctc_w = _slice_list(stitcher.get("cropped_to_canvas_w", []), offset, count)
        ctc_h = _slice_list(stitcher.get("cropped_to_canvas_h", []), offset, count)
        cto_x = _slice_list(stitcher.get("canvas_to_orig_x", []), offset, count)
        cto_y = _slice_list(stitcher.get("canvas_to_orig_y", []), offset, count)
        cto_w = _slice_list(stitcher.get("canvas_to_orig_w", []), offset, count)
        cto_h = _slice_list(stitcher.get("canvas_to_orig_h", []), offset, count)

        if not canvas_images:
            raise ValueError(
                f"[NV_FixtureDumper] No frames to dump. "
                f"frame_offset={offset} but stitcher only has "
                f"{len(stitcher.get('canvas_image', []))} canvas images."
            )

        actual_count = len(canvas_images)

        # ----- Stack tensors for safetensors --------------------------
        end = offset + actual_count
        tensors: dict[str, torch.Tensor] = {}

        ci = _stack_or_none(canvas_images, "canvas_image")
        if ci is not None:
            tensors["canvas_image"] = ci
        cm = _stack_or_none(canvas_masks, "canvas_mask")
        if cm is not None:
            tensors["canvas_mask"] = cm  # uint8 [0, 255]
        cmp = _stack_or_none(canvas_masks_processed, "canvas_mask_processed")
        if cmp is not None:
            tensors["canvas_mask_processed"] = cmp  # uint8 [0, 255]
        cbm = _stack_or_none(cropped_blend_masks, "cropped_mask_for_blend")
        if cbm is not None:
            tensors["cropped_mask_for_blend"] = cbm  # fp32 [0, 1]

        if isinstance(inpainted_image, torch.Tensor):
            tensors["inpainted_image"] = inpainted_image[offset:end].cpu().contiguous()
        if isinstance(comfy_output, torch.Tensor):
            tensors["comfy_output"] = comfy_output[offset:end].cpu().contiguous()

        save_file(tensors, str(tensor_path))

        # ----- Build metadata JSON ------------------------------------
        skipped_full = list(stitcher.get("skipped_indices", []))
        skipped_in_window = [int(i) for i in skipped_full if offset <= i < end]

        meta = {
            "schema_version": "fixture-0.1.0",
            "shot_id": shot_id,
            "frame_offset": offset,
            "num_frames": actual_count,
            "total_frames_in_stitcher": int(stitcher.get("total_frames", 0)),
            "skipped_indices_in_window": skipped_in_window,
            "geometry": {
                "cropped_to_canvas_x": list(map(int, ctc_x)),
                "cropped_to_canvas_y": list(map(int, ctc_y)),
                "cropped_to_canvas_w": list(map(int, ctc_w)),
                "cropped_to_canvas_h": list(map(int, ctc_h)),
                "canvas_to_orig_x": list(map(int, cto_x)),
                "canvas_to_orig_y": list(map(int, cto_y)),
                "canvas_to_orig_w": list(map(int, cto_w)),
                "canvas_to_orig_h": list(map(int, cto_h)),
                "crop_target_w": int(stitcher.get("crop_target_w", 0)),
                "crop_target_h": int(stitcher.get("crop_target_h", 0)),
                "resize_algorithm": str(stitcher.get("resize_algorithm", "bicubic")),
            },
            "blend_pixels_baked": int(stitcher.get("blend_pixels", 0)),
            "content_warp_mode": stitcher.get("content_warp_mode"),
            "crop_params": _parse_pseudo_yaml(crop_params),
            "stitch_params": _parse_pseudo_yaml(stitch_params),
            "tensor_inventory": {k: list(v.shape) for k, v in tensors.items()},
            "tensor_dtypes": {k: str(v.dtype) for k, v in tensors.items()},
            "produced_by": "NV_FixtureDumper",
            "produced_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        info_lines = [
            f"[NV_FixtureDumper] Dumped fixture:",
            f"  Tensors: {tensor_path}",
            f"  Meta:    {meta_path}",
            f"  Shot:    {shot_id}  Frames: {actual_count} (offset {offset})",
            f"  Tensor inventory:",
        ]
        for k, t in tensors.items():
            mb = t.element_size() * t.nelement() / 1e6
            info_lines.append(f"    {k:32s} {tuple(t.shape)}  {t.dtype}  {mb:.1f} MB")
        info_lines.append(f"  Skipped frames in window: {skipped_in_window or 'none'}")
        info = "\n".join(info_lines)
        print(info)
        return (info,)


NODE_CLASS_MAPPINGS = {
    "NV_FixtureDumper": NV_FixtureDumper,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_FixtureDumper": "NV Fixture Dumper",
}
