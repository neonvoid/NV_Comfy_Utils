"""
NV Stitcher Save/Load Nodes — V2

Save and load STITCHER objects produced by NV_InpaintCrop_V2 (and mutated by
NV_InpaintStitch_V2 / NV_CoTrackerBridge / NV_AETrackingBridge).

Why V2: the V1 module (stitcher_io.py) was written for the original
comfy-inpaint-crop-fork schema. Since then the V2 STITCHER dict has gained:

  - canvas_mask / canvas_mask_processed   (uint8 [H,W], stored as 0-255)
  - crop_params / stitch_params           (resolved-param records)
  - content_warp_mode / content_warp_data (CoTrackerBridge stabilization)
  - resize_algorithm                      (single key replaces v1 down/up split)
  - crop_target_w / crop_target_h

The V1 saver/loader silently dropped all of those AND would corrupt the
uint8 masks via a stray *255 quantization. V2 round-trips everything.

Layout on disk (output_path/):
    metadata.json              schema_version=2.0; coords; param dicts; pointers
    canvas_images/             frame_0000.png ... (RGB / RGBA, lossless uint8)
    canvas_masks/              frame_0000.png ... (uint8, no double *255)
    canvas_masks_processed/    frame_0000.png ...
    blend_masks/               frame_0000.png ... (fp→uint8 quantized OK)
    original_frames/           frame_0000.png ... (skipped indices)
    warp/
        centroid_data.json     {dx, dy} per frame (centroid mode only)
        flows/frame_0000.pt    torch.save'd flow tensors (optical_flow mode only)

Backward compatibility: V2 loader REFUSES v1 saves (version=1.0) with a
migration hint pointing at the V1 nodes. V1 nodes remain registered under
NV_SaveStitcher / NV_LoadStitcher / NV_StitcherInfo for legacy saves.
"""

import os
import json
import datetime

import numpy as np
import torch
from PIL import Image


SCHEMA_VERSION = "2.0"
KNOWN_LEGACY_VERSIONS = ("1.0",)


# =============================================================================
# I/O helpers
# =============================================================================

def _save_image_png(tensor, filepath):
    """Save a fp tensor [H,W,C] / [H,W] / [B,H,W,C] / [B,H,W] as a uint8 PNG.

    Used for canvas_image, original_frames, cropped_mask_for_blend. Round-trip
    is uint8-quantized but lossless w.r.t. the encoder's 8-bit destination.
    """
    while tensor.dim() > 3:
        tensor = tensor.squeeze(0)
    arr = tensor.detach().cpu().float().clamp(0.0, 1.0).numpy()
    u8 = (arr * 255.0).round().astype(np.uint8)
    if u8.ndim == 2:
        Image.fromarray(u8, mode="L").save(filepath, compress_level=1)
        return
    if u8.ndim == 3 and u8.shape[2] == 1:
        Image.fromarray(u8[:, :, 0], mode="L").save(filepath, compress_level=1)
        return
    if u8.ndim == 3 and u8.shape[2] == 3:
        Image.fromarray(u8, mode="RGB").save(filepath, compress_level=1)
        return
    if u8.ndim == 3 and u8.shape[2] == 4:
        Image.fromarray(u8, mode="RGBA").save(filepath, compress_level=1)
        return
    raise ValueError(f"[stitcher_io_v2] unsupported tensor shape for image save: {u8.shape}")


def _save_mask_passthrough(tensor, filepath):
    """Save a mask tensor as a uint8 PNG, preserving V2's uint8 storage convention.

    canvas_mask / canvas_mask_processed are stored as uint8 [H,W] with values
    0-255 already (per inpaint_crop.py D-070). The V1 saver multiplied by 255
    a SECOND time, saturating everything. This helper branches on dtype.
    """
    while tensor.dim() > 2:
        tensor = tensor.squeeze(0)
    if tensor.dtype == torch.uint8:
        u8 = tensor.detach().cpu().numpy()
    else:
        arr = tensor.detach().cpu().float().clamp(0.0, 1.0).numpy()
        u8 = (arr * 255.0).round().astype(np.uint8)
    if u8.ndim != 2:
        raise ValueError(f"[stitcher_io_v2] mask tensor must reduce to 2D, got shape {u8.shape}")
    Image.fromarray(u8, mode="L").save(filepath, compress_level=1)


def _load_image_fp32(filepath, device="cpu"):
    """Load a PNG as [1, H, W, 3] fp32 in [0, 1]."""
    img = Image.open(filepath)
    arr = np.array(img).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.shape[2] == 4:
        arr = arr[:, :, :3]
    tensor = torch.from_numpy(arr).to(device)
    return tensor.unsqueeze(0)


def _load_mask_uint8(filepath, device="cpu"):
    """Load a PNG as [1, H, W] uint8 — matches V2 canvas_mask storage."""
    img = Image.open(filepath)
    arr = np.array(img)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return torch.from_numpy(arr).to(device=device, dtype=torch.uint8).unsqueeze(0)


def _load_mask_fp32(filepath, device="cpu"):
    """Load a PNG as [1, H, W] fp32 in [0, 1] — for cropped_mask_for_blend."""
    img = Image.open(filepath)
    arr = np.array(img).astype(np.float32) / 255.0
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return torch.from_numpy(arr).to(device).unsqueeze(0)


# =============================================================================
# Warp data serialization
# =============================================================================

def _save_warp_data(stitcher, output_path, report_lines):
    """Serialize content_warp_mode + content_warp_data.

    centroid mode    -> warp/centroid_data.json (small, JSON)
    optical_flow     -> warp/flows/frame_NNNN.pt (torch.save, dtype-preserving)

    Returns (mode, payload_metadata) where payload_metadata is the dict to merge
    into the top-level metadata.json so the loader knows what to reconstruct.
    """
    mode = stitcher.get("content_warp_mode")
    data = stitcher.get("content_warp_data")

    if mode is None or not data:
        return None, {}

    warp_dir = os.path.join(output_path, "warp")
    os.makedirs(warp_dir, exist_ok=True)

    if mode == "centroid":
        cleaned = []
        for entry in data:
            if entry is None:
                cleaned.append(None)
                continue
            cleaned.append({
                "dx": float(entry.get("dx", 0.0)),
                "dy": float(entry.get("dy", 0.0)),
            })
        json_path = os.path.join(warp_dir, "centroid_data.json")
        with open(json_path, "w") as f:
            json.dump(cleaned, f)
        report_lines.append(
            f"Saved {sum(1 for e in cleaned if e is not None)} centroid warp entries to {json_path}"
        )
        return mode, {
            "content_warp_mode": "centroid",
            "content_warp_centroid_path": "warp/centroid_data.json",
            "content_warp_entry_count": len(cleaned),
        }

    if mode == "optical_flow":
        flows_dir = os.path.join(warp_dir, "flows")
        os.makedirs(flows_dir, exist_ok=True)
        nonempty = 0
        index_map = []
        for i, entry in enumerate(data):
            if entry is None or "flow" not in entry:
                index_map.append(None)
                continue
            flow_tensor = entry["flow"].detach().cpu()
            flow_path = os.path.join(flows_dir, f"frame_{i:04d}.pt")
            torch.save(flow_tensor, flow_path)
            index_map.append(f"frame_{i:04d}.pt")
            nonempty += 1
        report_lines.append(
            f"Saved {nonempty}/{len(data)} optical-flow tensors to {flows_dir} "
            f"(WARNING: optical-flow tensors at HD resolutions are large — "
            f"~17 MB/frame at 1080p × 2-channel fp32)"
        )
        return mode, {
            "content_warp_mode": "optical_flow",
            "content_warp_flows_dir": "warp/flows",
            "content_warp_flow_index": index_map,
            "content_warp_entry_count": len(data),
        }

    report_lines.append(f"WARNING: unknown content_warp_mode={mode!r} — warp data NOT saved")
    return mode, {"content_warp_mode": mode, "content_warp_warning": "unrecognized mode"}


def _load_warp_data(metadata, input_path, device, report_lines):
    """Reconstruct content_warp_mode + content_warp_data from metadata + side files."""
    mode = metadata.get("content_warp_mode")
    if mode is None:
        return None, None

    if mode == "centroid":
        rel = metadata.get("content_warp_centroid_path", "warp/centroid_data.json")
        path = os.path.join(input_path, rel)
        if not os.path.exists(path):
            report_lines.append(f"WARNING: centroid warp file missing at {path}; warp disabled on load")
            return None, None
        with open(path, "r") as f:
            raw = json.load(f)
        report_lines.append(
            f"Loaded {sum(1 for e in raw if e is not None)} centroid warp entries from {path}"
        )
        return "centroid", raw

    if mode == "optical_flow":
        rel_dir = metadata.get("content_warp_flows_dir", "warp/flows")
        flows_dir = os.path.join(input_path, rel_dir)
        index_map = metadata.get("content_warp_flow_index", [])
        out = []
        loaded = 0
        for fname in index_map:
            if fname is None:
                out.append(None)
                continue
            flow_path = os.path.join(flows_dir, fname)
            if not os.path.exists(flow_path):
                report_lines.append(f"WARNING: missing flow file {flow_path}; entry replaced with None")
                out.append(None)
                continue
            flow_tensor = torch.load(flow_path, map_location=device)
            out.append({"flow": flow_tensor})
            loaded += 1
        report_lines.append(f"Loaded {loaded}/{len(index_map)} optical-flow tensors from {flows_dir}")
        return "optical_flow", out

    report_lines.append(f"WARNING: unknown content_warp_mode={mode!r} in metadata; warp NOT restored")
    return None, None


# =============================================================================
# NV_SaveStitcher_V2
# =============================================================================

class NV_SaveStitcher_V2:
    """Save a V2 STITCHER object (full schema, including warp + param records)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stitcher": ("STITCHER",),
                "output_path": ("STRING", {
                    "default": "stitcher_data_v2",
                    "tooltip": (
                        "Directory path to save the stitcher data. Created if missing; "
                        "existing files are overwritten."
                    ),
                }),
            },
            "optional": {
                "save_canvas_images": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Save canvas_image frames (required for any later stitch). "
                        "Disable to save metadata-only (param records, coords, warp)."
                    ),
                }),
                "save_canvas_masks": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Save canvas_mask + canvas_mask_processed (V2-only — needed by "
                        "NV_CoTrackerBridge / NV_AETrackingBridge for warp source-space "
                        "consistency per D-061). Stored as uint8 PNG without double-quantization."
                    ),
                }),
                "save_warp_data": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Save content_warp_mode + content_warp_data when present. "
                        "Centroid mode = small JSON. Optical-flow mode = per-frame torch tensors "
                        "(WARNING: ~17 MB/frame at 1080p × 2ch fp32; ~4.5 GB for a 277f shot). "
                        "Disable to skip warp persistence (stitch will run unwarped on load)."
                    ),
                }),
                "verbose_report": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Print a save report to the ComfyUI console + return as a string.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("saved_path", "save_report")
    OUTPUT_NODE = True
    FUNCTION = "save_stitcher"
    CATEGORY = "NV_Utils/Stitcher"
    DESCRIPTION = (
        "Save a V2 STITCHER (full schema: canvas_mask, canvas_mask_processed, "
        "crop_params, stitch_params, content_warp_mode/data, resize_algorithm). "
        "Pairs with NV_LoadStitcher_V2."
    )

    def save_stitcher(self, stitcher, output_path,
                      save_canvas_images=True, save_canvas_masks=True,
                      save_warp_data=True, verbose_report=True):
        os.makedirs(output_path, exist_ok=True)

        report_lines = [
            "=" * 64,
            f"STITCHER V2 SAVE REPORT  (schema_version={SCHEMA_VERSION})",
            "=" * 64,
            f"Output path: {output_path}",
        ]

        # --- Top-level metadata (everything JSON-clean lives here) ----------
        metadata = {
            "version": SCHEMA_VERSION,
            "saved_at": datetime.datetime.utcnow().isoformat() + "Z",

            # V2 single resize algorithm (replaces v1 down/up split)
            "resize_algorithm": stitcher.get("resize_algorithm", "bicubic"),
            "blend_pixels": int(stitcher.get("blend_pixels", 16)),
            "crop_target_w": stitcher.get("crop_target_w"),
            "crop_target_h": stitcher.get("crop_target_h"),

            # Coordinate arrays (always per-frame lists)
            "canvas_to_orig_x": list(stitcher.get("canvas_to_orig_x", [])),
            "canvas_to_orig_y": list(stitcher.get("canvas_to_orig_y", [])),
            "canvas_to_orig_w": list(stitcher.get("canvas_to_orig_w", [])),
            "canvas_to_orig_h": list(stitcher.get("canvas_to_orig_h", [])),
            "cropped_to_canvas_x": list(stitcher.get("cropped_to_canvas_x", [])),
            "cropped_to_canvas_y": list(stitcher.get("cropped_to_canvas_y", [])),
            "cropped_to_canvas_w": list(stitcher.get("cropped_to_canvas_w", [])),
            "cropped_to_canvas_h": list(stitcher.get("cropped_to_canvas_h", [])),

            "skipped_indices": list(stitcher.get("skipped_indices", [])),
            "total_frames": int(stitcher.get("total_frames", 0)),

            # Resolved param records (NEW in V2 — editor parity contract)
            "crop_params": stitcher.get("crop_params"),
        }
        if "stitch_params" in stitcher:
            metadata["stitch_params"] = stitcher["stitch_params"]

        # Frame counts for header report
        n_canvas = len(stitcher.get("canvas_image", []))
        n_canvas_mask = len(stitcher.get("canvas_mask", []))
        n_canvas_mask_proc = len(stitcher.get("canvas_mask_processed", []))
        n_blend = len(stitcher.get("cropped_mask_for_blend", []))
        n_orig = len(stitcher.get("original_frames", []))
        report_lines.append(
            f"total_frames={metadata['total_frames']}  "
            f"canvas={n_canvas} canvas_mask={n_canvas_mask} "
            f"canvas_mask_proc={n_canvas_mask_proc} blend={n_blend} skipped_orig={n_orig}"
        )
        if metadata["crop_params"] is not None:
            cp = metadata["crop_params"]
            report_lines.append(
                f"crop_params: stitch_source={cp.get('crop_stitch_source')} "
                f"target_mode={cp.get('target_mode')} resize={cp.get('resize_algorithm')} "
                f"mask_config_used={cp.get('mask_config_used')}"
            )
        if "stitch_params" in metadata:
            sp = metadata["stitch_params"]
            report_lines.append(
                f"stitch_params: blend_mode={sp.get('blend_mode')} "
                f"output_dtype={sp.get('output_dtype')} "
                f"guided_refine={sp.get('guided_refine')}"
            )

        # --- Save canvas images --------------------------------------------
        if save_canvas_images and n_canvas > 0:
            d = os.path.join(output_path, "canvas_images")
            os.makedirs(d, exist_ok=True)
            for i, t in enumerate(stitcher["canvas_image"]):
                _save_image_png(t, os.path.join(d, f"frame_{i:04d}.png"))
            metadata["canvas_images_dir"] = "canvas_images"
            metadata["canvas_images_count"] = n_canvas
            report_lines.append(f"Saved {n_canvas} canvas images")

        # --- Save canvas masks (uint8 0-255 passthrough) -------------------
        if save_canvas_masks and n_canvas_mask > 0:
            d = os.path.join(output_path, "canvas_masks")
            os.makedirs(d, exist_ok=True)
            for i, t in enumerate(stitcher["canvas_mask"]):
                _save_mask_passthrough(t, os.path.join(d, f"frame_{i:04d}.png"))
            metadata["canvas_masks_dir"] = "canvas_masks"
            metadata["canvas_masks_count"] = n_canvas_mask
            report_lines.append(f"Saved {n_canvas_mask} canvas_mask frames (uint8 passthrough)")

        if save_canvas_masks and n_canvas_mask_proc > 0:
            d = os.path.join(output_path, "canvas_masks_processed")
            os.makedirs(d, exist_ok=True)
            for i, t in enumerate(stitcher["canvas_mask_processed"]):
                _save_mask_passthrough(t, os.path.join(d, f"frame_{i:04d}.png"))
            metadata["canvas_masks_processed_dir"] = "canvas_masks_processed"
            metadata["canvas_masks_processed_count"] = n_canvas_mask_proc
            report_lines.append(f"Saved {n_canvas_mask_proc} canvas_mask_processed frames")

        # --- Save blend masks ----------------------------------------------
        if n_blend > 0:
            d = os.path.join(output_path, "blend_masks")
            os.makedirs(d, exist_ok=True)
            for i, t in enumerate(stitcher["cropped_mask_for_blend"]):
                _save_mask_passthrough(t, os.path.join(d, f"frame_{i:04d}.png"))
            metadata["blend_masks_dir"] = "blend_masks"
            metadata["blend_masks_count"] = n_blend
            report_lines.append(f"Saved {n_blend} blend masks")

        # --- Save original frames (skipped indices) ------------------------
        if n_orig > 0:
            d = os.path.join(output_path, "original_frames")
            os.makedirs(d, exist_ok=True)
            for i, t in enumerate(stitcher["original_frames"]):
                _save_image_png(t, os.path.join(d, f"frame_{i:04d}.png"))
            metadata["original_frames_dir"] = "original_frames"
            metadata["original_frames_count"] = n_orig
            report_lines.append(f"Saved {n_orig} original (skipped) frames")

        # --- Save content warp ---------------------------------------------
        if save_warp_data:
            mode, warp_meta = _save_warp_data(stitcher, output_path, report_lines)
            if warp_meta:
                metadata.update(warp_meta)
            if mode is None:
                report_lines.append("No content_warp data to save (mode=None)")
        else:
            report_lines.append("save_warp_data=False — skipping content warp persistence")

        # --- Write metadata.json -------------------------------------------
        metadata_path = os.path.join(output_path, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        report_lines.append(f"Wrote metadata.json to {metadata_path}")
        report_lines.append("=" * 64)

        report = "\n".join(report_lines)
        if verbose_report:
            print(report)
        return (output_path, report)


# =============================================================================
# NV_LoadStitcher_V2
# =============================================================================

class NV_LoadStitcher_V2:
    """Load a V2 STITCHER from disk. Refuses v1 saves with a migration hint."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_path": ("STRING", {
                    "default": "stitcher_data_v2",
                    "tooltip": (
                        "Directory containing a V2 stitcher save (metadata.json with "
                        "version=2.0). For older v1 saves, use the V1 NV_LoadStitcher node."
                    ),
                }),
            },
            "optional": {
                "load_warp_data": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Reconstruct content_warp_mode + content_warp_data when present. "
                        "Disable to load metadata-only (skip optical-flow tensor reads — "
                        "useful when the stitcher will be re-stabilized rather than reused as-is)."
                    ),
                }),
                "verbose_report": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Print a load report to the ComfyUI console + return as a string.",
                }),
            },
        }

    RETURN_TYPES = ("STITCHER", "STRING")
    RETURN_NAMES = ("stitcher", "load_report")
    FUNCTION = "load_stitcher"
    CATEGORY = "NV_Utils/Stitcher"
    DESCRIPTION = (
        "Load a STITCHER saved with NV_SaveStitcher_V2 (full V2 schema). "
        "Refuses v1 saves with a clear migration message — use NV_LoadStitcher (v1) for those."
    )

    def load_stitcher(self, input_path, load_warp_data=True, verbose_report=True):
        import comfy.model_management
        intermediate = comfy.model_management.intermediate_device()

        report_lines = [
            "=" * 64,
            "STITCHER V2 LOAD REPORT",
            "=" * 64,
            f"Input path: {input_path}",
        ]

        metadata_path = os.path.join(input_path, "metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"[NV_LoadStitcher_V2] metadata.json not found at {metadata_path}. "
                f"Either the path is wrong, or this is not a stitcher save directory."
            )

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        version = metadata.get("version", "unknown")
        if version in KNOWN_LEGACY_VERSIONS:
            raise ValueError(
                f"[NV_LoadStitcher_V2] save at {input_path!r} is schema_version={version!r} "
                f"(legacy V1 format). This loader only handles V2 (schema_version={SCHEMA_VERSION!r}). "
                f"Use the V1 NV_LoadStitcher node instead, or re-save through V2 to upgrade."
            )
        if not str(version).startswith("2."):
            raise ValueError(
                f"[NV_LoadStitcher_V2] unsupported schema_version={version!r}. "
                f"This loader expects 2.x (got {version!r})."
            )

        report_lines.append(f"schema_version={version}  saved_at={metadata.get('saved_at', 'unknown')}")

        # --- Reconstruct top-level scalar/list fields ----------------------
        stitcher = {
            "resize_algorithm": metadata.get("resize_algorithm", "bicubic"),
            "blend_pixels": int(metadata.get("blend_pixels", 16)),
            "canvas_to_orig_x": list(metadata.get("canvas_to_orig_x", [])),
            "canvas_to_orig_y": list(metadata.get("canvas_to_orig_y", [])),
            "canvas_to_orig_w": list(metadata.get("canvas_to_orig_w", [])),
            "canvas_to_orig_h": list(metadata.get("canvas_to_orig_h", [])),
            "cropped_to_canvas_x": list(metadata.get("cropped_to_canvas_x", [])),
            "cropped_to_canvas_y": list(metadata.get("cropped_to_canvas_y", [])),
            "cropped_to_canvas_w": list(metadata.get("cropped_to_canvas_w", [])),
            "cropped_to_canvas_h": list(metadata.get("cropped_to_canvas_h", [])),
            "skipped_indices": list(metadata.get("skipped_indices", [])),
            "total_frames": int(metadata.get("total_frames", 0)),
            "canvas_image": [],
            "canvas_mask": [],
            "canvas_mask_processed": [],
            "cropped_mask_for_blend": [],
            "original_frames": [],
        }
        # Optional V2-only fields
        if metadata.get("crop_target_w") is not None:
            stitcher["crop_target_w"] = int(metadata["crop_target_w"])
        if metadata.get("crop_target_h") is not None:
            stitcher["crop_target_h"] = int(metadata["crop_target_h"])
        if metadata.get("crop_params") is not None:
            stitcher["crop_params"] = metadata["crop_params"]
        if "stitch_params" in metadata:
            stitcher["stitch_params"] = metadata["stitch_params"]

        # --- Load canvas images --------------------------------------------
        d = metadata.get("canvas_images_dir")
        if d:
            count = int(metadata.get("canvas_images_count", 0))
            base = os.path.join(input_path, d)
            for i in range(count):
                fp = os.path.join(base, f"frame_{i:04d}.png")
                if os.path.exists(fp):
                    stitcher["canvas_image"].append(_load_image_fp32(fp, device=intermediate))
            report_lines.append(f"Loaded {len(stitcher['canvas_image'])}/{count} canvas images")

        # --- Load canvas masks (uint8 — V2 storage convention) -------------
        d = metadata.get("canvas_masks_dir")
        if d:
            count = int(metadata.get("canvas_masks_count", 0))
            base = os.path.join(input_path, d)
            for i in range(count):
                fp = os.path.join(base, f"frame_{i:04d}.png")
                if os.path.exists(fp):
                    stitcher["canvas_mask"].append(_load_mask_uint8(fp, device=intermediate))
            report_lines.append(
                f"Loaded {len(stitcher['canvas_mask'])}/{count} canvas_mask frames (uint8)"
            )

        d = metadata.get("canvas_masks_processed_dir")
        if d:
            count = int(metadata.get("canvas_masks_processed_count", 0))
            base = os.path.join(input_path, d)
            for i in range(count):
                fp = os.path.join(base, f"frame_{i:04d}.png")
                if os.path.exists(fp):
                    stitcher["canvas_mask_processed"].append(_load_mask_uint8(fp, device=intermediate))
            report_lines.append(
                f"Loaded {len(stitcher['canvas_mask_processed'])}/{count} canvas_mask_processed frames"
            )

        # --- Load blend masks (fp32 [0,1]) ---------------------------------
        d = metadata.get("blend_masks_dir")
        if d:
            count = int(metadata.get("blend_masks_count", 0))
            base = os.path.join(input_path, d)
            for i in range(count):
                fp = os.path.join(base, f"frame_{i:04d}.png")
                if os.path.exists(fp):
                    stitcher["cropped_mask_for_blend"].append(_load_mask_fp32(fp, device=intermediate))
            report_lines.append(
                f"Loaded {len(stitcher['cropped_mask_for_blend'])}/{count} blend masks"
            )

        # --- Load skipped original frames ----------------------------------
        d = metadata.get("original_frames_dir")
        if d:
            count = int(metadata.get("original_frames_count", 0))
            base = os.path.join(input_path, d)
            for i in range(count):
                fp = os.path.join(base, f"frame_{i:04d}.png")
                if os.path.exists(fp):
                    stitcher["original_frames"].append(_load_image_fp32(fp, device=intermediate))
            report_lines.append(
                f"Loaded {len(stitcher['original_frames'])}/{count} original (skipped) frames"
            )

        # --- Load content warp data ----------------------------------------
        if load_warp_data:
            mode, data = _load_warp_data(metadata, input_path, intermediate, report_lines)
            if mode is not None:
                stitcher["content_warp_mode"] = mode
                stitcher["content_warp_data"] = data
        else:
            report_lines.append("load_warp_data=False — content_warp_mode/data NOT restored")

        report_lines.append(
            f"Stitcher loaded: total_frames={stitcher['total_frames']}, "
            f"skipped={len(stitcher['skipped_indices'])}, "
            f"warp={stitcher.get('content_warp_mode') or 'none'}"
        )
        report_lines.append("=" * 64)

        report = "\n".join(report_lines)
        if verbose_report:
            print(report)
        return (stitcher, report)


# =============================================================================
# NV_StitcherInfo_V2
# =============================================================================

class NV_StitcherInfo_V2:
    """Display the V2 STITCHER schema in human-readable form (V1 fields removed)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stitcher": ("STITCHER",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    FUNCTION = "get_info"
    CATEGORY = "NV_Utils/Stitcher"
    DESCRIPTION = "Display info about a V2 STITCHER (warp + crop_params + stitch_params + V2 keys)."

    def get_info(self, stitcher):
        lines = [
            "=" * 64,
            "STITCHER INFO (V2)",
            "=" * 64,
        ]

        lines.append("SETTINGS:")
        lines.append(f"  resize_algorithm: {stitcher.get('resize_algorithm', 'N/A')}")
        lines.append(f"  blend_pixels:     {stitcher.get('blend_pixels', 'N/A')}")
        lines.append(
            f"  crop_target:      {stitcher.get('crop_target_w', '?')}x{stitcher.get('crop_target_h', '?')}"
        )

        lines.append("")
        lines.append("FRAME COUNTS:")
        lines.append(f"  total_frames:           {stitcher.get('total_frames', 0)}")
        lines.append(f"  canvas_image:           {len(stitcher.get('canvas_image', []))}")
        lines.append(f"  canvas_mask:            {len(stitcher.get('canvas_mask', []))}")
        lines.append(f"  canvas_mask_processed:  {len(stitcher.get('canvas_mask_processed', []))}")
        lines.append(f"  cropped_mask_for_blend: {len(stitcher.get('cropped_mask_for_blend', []))}")
        lines.append(f"  skipped indices:        {len(stitcher.get('skipped_indices', []))}")
        if stitcher.get("skipped_indices"):
            lines.append(f"    {stitcher['skipped_indices']}")

        # Coord summary
        cx = stitcher.get("canvas_to_orig_x", [])
        cy = stitcher.get("canvas_to_orig_y", [])
        cw = stitcher.get("canvas_to_orig_w", [])
        ch = stitcher.get("canvas_to_orig_h", [])
        if cx:
            lines.append("")
            lines.append("CANVAS->ORIG (x,y,w,h ranges):")
            lines.append(f"  x: {min(cx)}–{max(cx)}   y: {min(cy)}–{max(cy)}")
            lines.append(f"  w: {min(cw)}–{max(cw)}   h: {min(ch)}–{max(ch)}")

        ctcx = stitcher.get("cropped_to_canvas_x", [])
        ctcy = stitcher.get("cropped_to_canvas_y", [])
        ctcw = stitcher.get("cropped_to_canvas_w", [])
        ctch = stitcher.get("cropped_to_canvas_h", [])
        if ctcx:
            lines.append("")
            lines.append("CROPPED->CANVAS (x,y,w,h ranges):")
            lines.append(f"  x: {min(ctcx)}–{max(ctcx)}   y: {min(ctcy)}–{max(ctcy)}")
            lines.append(f"  w: {min(ctcw)}–{max(ctcw)}   h: {min(ctch)}–{max(ctch)}")

        # Param records
        cp = stitcher.get("crop_params")
        if cp:
            lines.append("")
            lines.append("CROP_PARAMS (resolved post-override):")
            for k in (
                "crop_stitch_source", "crop_blend_feather_px", "hybrid_falloff", "hybrid_curve",
                "cleanup_fill_holes", "cleanup_remove_noise", "cleanup_smooth", "crop_expand_px",
                "resize_algorithm", "target_mode", "target_width", "target_height",
                "auto_preset", "padding_multiple", "anomaly_threshold", "mask_config_used",
            ):
                if k in cp:
                    lines.append(f"  {k}: {cp[k]}")

        sp = stitcher.get("stitch_params")
        if sp:
            lines.append("")
            lines.append("STITCH_PARAMS (recorded at stitch time):")
            for k in (
                "blend_mode", "multiband_levels", "guided_refine",
                "guided_radius", "guided_eps", "guided_strength", "output_dtype",
            ):
                if k in sp:
                    lines.append(f"  {k}: {sp[k]}")

        # Content warp
        warp_mode = stitcher.get("content_warp_mode")
        warp_data = stitcher.get("content_warp_data")
        lines.append("")
        lines.append("CONTENT WARP:")
        if warp_mode is None:
            lines.append("  mode: none")
        else:
            count = len(warp_data) if warp_data else 0
            nonempty = sum(1 for e in (warp_data or []) if e is not None)
            lines.append(f"  mode: {warp_mode}")
            lines.append(f"  entries: {nonempty}/{count} non-null")

        # Tensor shape peek
        canvas = stitcher.get("canvas_image", [])
        if canvas:
            t0 = canvas[0]
            if hasattr(t0, "shape"):
                lines.append("")
                lines.append("CANVAS_IMAGE[0]:")
                lines.append(f"  shape: {tuple(t0.shape)}")
                lines.append(f"  dtype: {t0.dtype}")
                lines.append(f"  device: {getattr(t0, 'device', 'N/A')}")

        cmask = stitcher.get("canvas_mask", [])
        if cmask:
            t0 = cmask[0]
            if hasattr(t0, "shape"):
                lines.append("")
                lines.append("CANVAS_MASK[0]:")
                lines.append(f"  shape: {tuple(t0.shape)}")
                lines.append(f"  dtype: {t0.dtype}  (V2 stores uint8 0-255 — see D-070)")

        lines.append("=" * 64)
        info = "\n".join(lines)
        print(info)
        return (info,)


# =============================================================================
# Registration
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "NV_SaveStitcher_V2": NV_SaveStitcher_V2,
    "NV_LoadStitcher_V2": NV_LoadStitcher_V2,
    "NV_StitcherInfo_V2": NV_StitcherInfo_V2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_SaveStitcher_V2": "NV Save Stitcher v2",
    "NV_LoadStitcher_V2": "NV Load Stitcher v2",
    "NV_StitcherInfo_V2": "NV Stitcher Info v2",
}
