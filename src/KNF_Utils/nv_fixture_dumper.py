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

import datetime
import json
import os
from pathlib import Path

import torch
from safetensors.torch import save_file

import folder_paths


# --- Memory warning threshold for total dumped tensor bytes ---------------
_MEM_WARN_MB = 2000


def _stack_or_none(tensor_list, name: str):
    """Stack a list of same-shape, same-dtype tensors. Returns None if empty.

    Validates explicitly so that a non-tensor / mixed-dtype / mismatched-shape
    list produces a targeted error instead of a cryptic AttributeError or
    silent dtype promotion (uint8 mask quietly becoming float32).
    """
    if not tensor_list:
        return None

    first = tensor_list[0]
    if not isinstance(first, torch.Tensor):
        raise TypeError(
            f"[NV_FixtureDumper] '{name}' entry 0 is not a torch.Tensor "
            f"(got {type(first).__name__}). Did the stitcher dict get corrupted?"
        )
    expected_shape = tuple(first.shape)
    expected_dtype = first.dtype

    for i, t in enumerate(tensor_list):
        if not isinstance(t, torch.Tensor):
            raise TypeError(
                f"[NV_FixtureDumper] '{name}' entry {i} is non-tensor: "
                f"{type(t).__name__}"
            )
        if tuple(t.shape) != expected_shape:
            raise RuntimeError(
                f"[NV_FixtureDumper] '{name}' entry {i} shape {tuple(t.shape)} "
                f"!= entry 0 shape {expected_shape}"
            )
        if t.dtype != expected_dtype:
            raise RuntimeError(
                f"[NV_FixtureDumper] '{name}' entry {i} dtype {t.dtype} "
                f"!= entry 0 dtype {expected_dtype} — refusing to silently "
                f"promote (would lose uint8 -> float distinction)"
            )

    try:
        return torch.stack([t.cpu() for t in tensor_list], dim=0).contiguous()
    except RuntimeError as e:
        raise RuntimeError(
            f"[NV_FixtureDumper] Failed to stack '{name}' (shape {expected_shape}, "
            f"dtype {expected_dtype}): {e}"
        ) from e


def _build_global_to_pos_map(total_frames: int, skipped_set: set[int]) -> dict[int, int]:
    """Map global frame index -> position in the stitcher's non-skipped lists.

    The stitcher's per-frame lists (canvas_image, canvas_mask, etc.) only
    contain non-skipped frames, indexed by enumeration order. This map lets
    callers translate a global frame index into the corresponding list position.
    """
    out: dict[int, int] = {}
    pos = 0
    for g in range(total_frames):
        if g not in skipped_set:
            out[g] = pos
            pos += 1
    return out


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
             shot_id, num_frames, frame_offset, output_dir):

        # ----- Validate inputs upfront, fail with useful messages --------
        if not isinstance(stitcher, dict):
            raise TypeError(
                f"[NV_FixtureDumper] 'stitcher' must be the STITCHER dict from "
                f"NV_InpaintCrop2, got {type(stitcher).__name__}. Did you wire "
                f"the wrong port? (Check that the cyan STITCHER output is "
                f"connected, not an IMAGE.)"
            )

        required_stitcher_keys = (
            "canvas_image", "canvas_mask", "canvas_mask_processed",
            "cropped_mask_for_blend",
            "cropped_to_canvas_x", "cropped_to_canvas_y",
            "cropped_to_canvas_w", "cropped_to_canvas_h",
            "canvas_to_orig_x", "canvas_to_orig_y",
            "canvas_to_orig_w", "canvas_to_orig_h",
            "total_frames",
        )
        missing = [k for k in required_stitcher_keys if k not in stitcher]
        if missing:
            raise KeyError(
                f"[NV_FixtureDumper] Stitcher dict is missing required keys: "
                f"{missing}. Is this an NV_InpaintCrop2 output? (Old/legacy "
                f"stitcher formats may need an upgrade path.)"
            )

        if not isinstance(inpainted_image, torch.Tensor):
            raise TypeError(
                f"[NV_FixtureDumper] 'inpainted_image' must be an IMAGE tensor "
                f"(KSampler output), got {type(inpainted_image).__name__}."
            )
        if not isinstance(comfy_output, torch.Tensor):
            raise TypeError(
                f"[NV_FixtureDumper] 'comfy_output' must be an IMAGE tensor "
                f"(NV_InpaintStitch2 output), got {type(comfy_output).__name__}."
            )

        out_root = Path(folder_paths.get_output_directory()) / output_dir
        out_root.mkdir(parents=True, exist_ok=True)

        # ----- Build the global<->position map --------------------------
        # The stitcher's per-frame lists are indexed by enumeration of
        # NON-SKIPPED frames. comfy_output (from NV_InpaintStitch2) is indexed
        # by GLOBAL frame index — skipped frames are pass-through copies.
        # inpainted_image (KSampler output) is non-skipped indexed.
        # Everything in the fixture must be aligned, so we slice each by its
        # native index space but to the SAME set of non-skipped global frames.
        offset, count = int(frame_offset), int(num_frames)
        total_frames = int(stitcher["total_frames"])
        skipped_full = set(int(i) for i in stitcher.get("skipped_indices", []))
        global_to_pos = _build_global_to_pos_map(total_frames, skipped_full)

        end = min(offset + count, total_frames)
        global_window = list(range(offset, end))
        non_skipped_globals = [g for g in global_window if g not in skipped_full]
        skipped_in_window = [g for g in global_window if g in skipped_full]

        if not non_skipped_globals:
            if not global_window:
                raise ValueError(
                    f"[NV_FixtureDumper] Empty window: frame_offset={offset} "
                    f">= total_frames={total_frames}."
                )
            raise ValueError(
                f"[NV_FixtureDumper] Every frame in window [{offset}, {end}) "
                f"was skipped (empty mask): {skipped_in_window}. Pick a window "
                f"that contains at least one non-skipped frame."
            )

        stitcher_positions = [global_to_pos[g] for g in non_skipped_globals]
        actual_count = len(non_skipped_globals)

        # ----- Slice stitcher lists by stitcher_positions ---------------
        canvas_images = [stitcher["canvas_image"][p] for p in stitcher_positions]
        canvas_masks = [stitcher["canvas_mask"][p] for p in stitcher_positions]
        canvas_masks_processed = [stitcher["canvas_mask_processed"][p] for p in stitcher_positions]
        cropped_blend_masks = [stitcher["cropped_mask_for_blend"][p] for p in stitcher_positions]
        ctc_x = [stitcher["cropped_to_canvas_x"][p] for p in stitcher_positions]
        ctc_y = [stitcher["cropped_to_canvas_y"][p] for p in stitcher_positions]
        ctc_w = [stitcher["cropped_to_canvas_w"][p] for p in stitcher_positions]
        ctc_h = [stitcher["cropped_to_canvas_h"][p] for p in stitcher_positions]
        cto_x = [stitcher["canvas_to_orig_x"][p] for p in stitcher_positions]
        cto_y = [stitcher["canvas_to_orig_y"][p] for p in stitcher_positions]
        cto_w = [stitcher["canvas_to_orig_w"][p] for p in stitcher_positions]
        cto_h = [stitcher["canvas_to_orig_h"][p] for p in stitcher_positions]

        # ----- Stack tensors, indexing by appropriate space -------------
        # inpainted_image is non-skipped indexed -> use stitcher_positions
        # comfy_output is global indexed -> use non_skipped_globals
        # All resulting tensors have the SAME first-dim length = actual_count,
        # 1:1 aligned by enumerated position.
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

        # Index by python-list of ints; this triggers fancy indexing along dim 0
        idx_pos = torch.tensor(stitcher_positions, dtype=torch.long)
        idx_glb = torch.tensor(non_skipped_globals, dtype=torch.long)
        tensors["inpainted_image"] = inpainted_image.index_select(0, idx_pos).cpu().contiguous()
        tensors["comfy_output"] = comfy_output.index_select(0, idx_glb).cpu().contiguous()

        # ----- Memory sanity check --------------------------------------
        total_bytes = sum(t.element_size() * t.nelement() for t in tensors.values())
        total_mb = total_bytes / 1e6
        if total_mb > _MEM_WARN_MB:
            print(
                f"[NV_FixtureDumper] WARNING: dump size is {total_mb:.0f} MB. "
                f"Large fixtures slow disk I/O and parity tests; consider "
                f"reducing num_frames (currently {actual_count})."
            )

        # ----- Atomic write: write to .tmp first, then os.replace -------
        # JSON is the commit marker (renamed last). Tensors first; if the JSON
        # rename fails the orphan .safetensors is harmless and gets overwritten
        # on next run.
        base_name = f"{shot_id}_{actual_count}f"
        tensor_path = out_root / f"{base_name}.safetensors"
        meta_path = out_root / f"{base_name}.json"
        tmp_tensor_path = out_root / f"{base_name}.safetensors.tmp"
        tmp_meta_path = out_root / f"{base_name}.json.tmp"

        save_file(tensors, str(tmp_tensor_path))

        meta = {
            "schema_version": "fixture-0.1.0",
            "shot_id": shot_id,
            "frame_offset": offset,
            "num_frames": actual_count,
            "total_frames_in_stitcher": total_frames,
            "frame_global_indices": non_skipped_globals,
            "skipped_indices_in_window": skipped_in_window,
            "global_window": [offset, end],
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
            # Read params directly from the stitcher dict — NV_InpaintCrop2 records
            # crop_params at construction; NV_InpaintStitch2 mutates the same dict
            # to add stitch_params at the start of stitch(). The dumper depends on
            # comfy_output (which forces NV_InpaintStitch2 to run first), so by the
            # time we read stitcher here both keys are populated.
            "crop_params": stitcher.get("crop_params", {}),
            "stitch_params": stitcher.get("stitch_params", {}),
            "tensor_inventory": {k: list(v.shape) for k, v in tensors.items()},
            "tensor_dtypes": {k: str(v.dtype) for k, v in tensors.items()},
            "tensor_total_mb": round(total_mb, 1),
            "produced_by": "NV_FixtureDumper",
            "produced_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

        with open(tmp_meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        os.replace(tmp_tensor_path, tensor_path)
        os.replace(tmp_meta_path, meta_path)

        info_lines = [
            f"[NV_FixtureDumper] Dumped fixture:",
            f"  Tensors: {tensor_path}",
            f"  Meta:    {meta_path}",
            f"  Shot:    {shot_id}  Frames: {actual_count} (window [{offset}, {end}))",
            f"  Globals: {non_skipped_globals}",
            f"  Tensor inventory:",
        ]
        for k, t in tensors.items():
            mb = t.element_size() * t.nelement() / 1e6
            info_lines.append(f"    {k:32s} {tuple(t.shape)}  {t.dtype}  {mb:.1f} MB")
        info_lines.append(f"  Total dump size: {total_mb:.1f} MB")
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
