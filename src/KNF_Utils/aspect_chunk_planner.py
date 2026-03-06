"""NV_AspectChunkPlanner — temporal segmentation by bounding-box aspect ratio.

Analyzes per-frame bbox masks, detects aspect ratio change points, and
segments the video into temporal chunks where each chunk has a stable
aspect ratio.  Each chunk gets its own union bbox and target resolution,
written to a JSON plan file.
"""

import json
import math
import os

import torch
import folder_paths

from .chunk_utils import (
    nearest_wan_aligned,
    is_wan_aligned,
    video_to_latent_frames,
)
from .latent_constants import LATENT_SAFE_KEYS
from .mask_tracking_bbox import extract_bboxes
from .inpaint_crop import compute_auto_resolution, WAN_PRESETS
from .latent_inpaint_crop import snap_to_vae_grid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_unique_filepath(filepath: str) -> str:
    """Return a unique filepath by appending _N if file exists."""
    if not os.path.exists(filepath):
        return filepath
    base, ext = os.path.splitext(filepath)
    counter = 1
    while True:
        new_path = f"{base}_{counter}{ext}"
        if not os.path.exists(new_path):
            return new_path
        counter += 1


def _compute_per_frame_aspects(x1s, y1s, x2s, y2s):
    """Per-frame width/height ratios.  0.0 sentinel for zero-area bboxes."""
    aspects = []
    for i in range(len(x1s)):
        w = x2s[i] - x1s[i]
        h = y2s[i] - y1s[i]
        aspects.append(w / h if w > 0 and h > 0 else 0.0)
    return aspects


def _greedy_segment(aspects, threshold):
    """Split into segments wherever aspect deviates from running mean.

    Uses log-ratio distance: ``|log(ar / mean)| > threshold``.

    Returns list of ``(start, end_exclusive)`` tuples.
    """
    if not aspects:
        return []

    segments = []
    seg_start = 0
    seg_sum = 0.0
    seg_count = 0

    for i, ar in enumerate(aspects):
        if ar <= 0:
            continue  # skip empty frames, keep in current segment

        if seg_count == 0:
            seg_sum = ar
            seg_count = 1
            continue

        seg_mean = seg_sum / seg_count
        if abs(math.log(ar / seg_mean)) > threshold:
            segments.append((seg_start, i))
            seg_start = i
            seg_sum = ar
            seg_count = 1
        else:
            seg_sum += ar
            seg_count += 1

    segments.append((seg_start, len(aspects)))
    return segments


def _segment_mean_ar(aspects, start, end):
    """Mean aspect ratio for valid frames in [start, end)."""
    valid = [aspects[i] for i in range(start, end) if aspects[i] > 0]
    return sum(valid) / len(valid) if valid else 1.0


def _merge_small_segments(segments, aspects, min_chunk_frames):
    """Repeatedly merge the smallest segment into its nearest neighbour.

    "Nearest" = smallest log-ratio distance between segment mean aspects.
    Ties broken by preferring the left neighbour.
    """
    if len(segments) <= 1:
        return segments

    working = list(segments)

    while len(working) > 1:
        # Find smallest
        sizes = [(s[1] - s[0], idx) for idx, s in enumerate(working)]
        sizes.sort()
        smallest_size, smallest_idx = sizes[0]

        if smallest_size >= min_chunk_frames:
            break

        # Pick neighbour with closest mean AR
        small_ar = _segment_mean_ar(aspects, *working[smallest_idx])
        best_neighbor = None
        best_dist = float("inf")

        for n_idx in [smallest_idx - 1, smallest_idx + 1]:
            if 0 <= n_idx < len(working):
                n_ar = _segment_mean_ar(aspects, *working[n_idx])
                if small_ar > 0 and n_ar > 0:
                    dist = abs(math.log(small_ar / n_ar))
                else:
                    dist = 0.0
                if dist < best_dist or (dist == best_dist and n_idx < smallest_idx):
                    best_dist = dist
                    best_neighbor = n_idx

        if best_neighbor is None:
            break

        lo = min(smallest_idx, best_neighbor)
        hi = max(smallest_idx, best_neighbor)
        merged = (working[lo][0], working[hi][1])
        working[lo : hi + 1] = [merged]

    return working


def _snap_segments_to_wan(segments, total_frames):
    """Adjust boundaries so each segment's frame_count is WAN-aligned.

    Last segment absorbs any remainder.  Tiny post-snap segments (< 5)
    merge into the previous one.
    """
    if not segments:
        return segments

    result = []
    cursor = 0

    for i in range(len(segments)):
        if i == len(segments) - 1:
            result.append((cursor, total_frames))
        else:
            count = segments[i][1] - segments[i][0]
            aligned = nearest_wan_aligned(count)
            aligned = max(5, aligned)
            end = min(cursor + aligned, total_frames)
            result.append((cursor, end))
            cursor = end

    # Merge segments that became too small after snapping
    final = []
    for seg in result:
        count = seg[1] - seg[0]
        if count < 5 and final:
            prev = final[-1]
            final[-1] = (prev[0], seg[1])
        else:
            final.append(seg)

    return final


def _compute_segment_union_bbox(x1s, y1s, x2s, y2s, start, end, padding, H, W):
    """Union bbox for frames [start, end) with padding and VAE grid snap.

    Returns (x, y, w, h) as ints or None if no valid bbox.
    """
    valid = [
        i for i in range(start, end)
        if (x2s[i] - x1s[i]) > 0 and (y2s[i] - y1s[i]) > 0
    ]
    if not valid:
        return None

    ux1 = min(x1s[i] for i in valid)
    uy1 = min(y1s[i] for i in valid)
    ux2 = max(x2s[i] for i in valid)
    uy2 = max(y2s[i] for i in valid)

    bw = ux2 - ux1
    bh = uy2 - uy1
    pad_x = bw * padding
    pad_y = bh * padding
    ux1 = max(0, ux1 - pad_x)
    uy1 = max(0, uy1 - pad_y)
    ux2 = min(W, ux2 + pad_x)
    uy2 = min(H, uy2 + pad_y)

    return snap_to_vae_grid(ux1, uy1, ux2 - ux1, uy2 - uy1, H, W)


# ---------------------------------------------------------------------------
# Core analysis (shared by planner + preview)
# ---------------------------------------------------------------------------

def analyze_aspect_chunks(
    mask: torch.Tensor,
    aspect_threshold: float,
    min_chunk_frames: int,
    wan_alignment: bool,
    auto_preset: str,
    padding: float,
):
    """Run the full aspect-ratio segmentation pipeline.

    Returns (plan_dict, info_lines).
    """
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    B, H, W = mask.shape
    info_lines = [f"[AspectChunkPlanner] {B} frames, {W}x{H}px"]

    # ---- 1. Per-frame bboxes ----
    x1s, y1s, x2s, y2s, present = extract_bboxes(mask, info_lines)
    aspects = _compute_per_frame_aspects(x1s, y1s, x2s, y2s)

    valid_count = sum(1 for a in aspects if a > 0)
    if valid_count == 0:
        info_lines.append("  WARNING: no valid bboxes — returning single full-frame chunk")
        segments = [(0, B)]
    else:
        # ---- 2. Greedy segmentation ----
        segments = _greedy_segment(aspects, aspect_threshold)
        info_lines.append(f"  Greedy: {len(segments)} raw segments")

        # ---- 3. Merge small segments ----
        segments = _merge_small_segments(segments, aspects, min_chunk_frames)
        info_lines.append(f"  After merge: {len(segments)} segments")

    # ---- 4. WAN alignment ----
    if wan_alignment:
        segments = _snap_segments_to_wan(segments, B)
        info_lines.append(f"  After WAN snap: {len(segments)} segments")

    # ---- 5. Build chunk descriptors ----
    chunks = []
    for seg_idx, (start, end) in enumerate(segments):
        frame_count = end - start
        latent_frames = video_to_latent_frames(frame_count)
        wan_ok = is_wan_aligned(frame_count)
        mean_ar = _segment_mean_ar(aspects, start, end)

        bbox = _compute_segment_union_bbox(
            x1s, y1s, x2s, y2s, start, end, padding, H, W,
        )
        if bbox:
            bx, by, bw, bh = bbox
            bbox_aspect = bw / bh if bh > 0 else 1.0
        else:
            bx, by, bw, bh = 0, 0, W, H
            bbox_aspect = W / H

        target_w, target_h = compute_auto_resolution(bbox_aspect, auto_preset, 0)

        chunks.append({
            "chunk_idx": seg_idx,
            "start_frame": start,
            "end_frame": end,
            "frame_count": frame_count,
            "latent_frames": latent_frames,
            "wan_aligned": wan_ok,
            "mean_aspect": round(mean_ar, 4),
            "bbox_x": int(bx),
            "bbox_y": int(by),
            "bbox_w": int(bw),
            "bbox_h": int(bh),
            "target_width": target_w,
            "target_height": target_h,
            "aspect_segment_id": seg_idx,
        })

    # ---- 6. Build plan dict ----
    plan = {
        "version": "1.0",
        "planner": "aspect_ratio",
        "video_metadata": {
            "total_frames": B,
            "height": H,
            "width": W,
        },
        "aspect_config": {
            "threshold": aspect_threshold,
            "min_chunk_frames": min_chunk_frames,
            "wan_aligned": wan_alignment,
            "auto_preset": auto_preset,
            "padding": padding,
        },
        "num_chunks": len(chunks),
        "chunks": chunks,
    }

    # ---- 7. Info summary ----
    info_lines.append("")
    info_lines.append(f"Plan: {len(chunks)} aspect-based chunks")
    for c in chunks:
        wan_tag = "WAN OK" if c["wan_aligned"] else "WAN MISS"
        info_lines.append(
            f"  Chunk {c['chunk_idx']}: frames {c['start_frame']}-{c['end_frame'] - 1} "
            f"({c['frame_count']}f, {c['latent_frames']}L) "
            f"AR={c['mean_aspect']:.2f} bbox={c['bbox_w']}x{c['bbox_h']} "
            f"target={c['target_width']}x{c['target_height']} [{wan_tag}]"
        )

    return plan, info_lines


# ---------------------------------------------------------------------------
# Shared input definitions
# ---------------------------------------------------------------------------

_ANALYSIS_INPUTS = {
    "mask": ("MASK", {
        "tooltip": "Per-frame masks [B,H,W] from MaskTrackingBBox or SAM3.",
    }),
    "aspect_threshold": ("FLOAT", {
        "default": 0.25, "min": 0.05, "max": 1.0, "step": 0.05,
        "tooltip": (
            "Log-ratio threshold for splitting. "
            "0.25 catches 4:3 to 16:9 transitions. "
            "Lower = more splits, higher = fewer."
        ),
    }),
    "min_chunk_frames": ("INT", {
        "default": 21, "min": 5, "max": 201, "step": 4,
        "tooltip": (
            "Minimum video frames per chunk. Segments smaller "
            "than this merge into the nearest neighbour by AR."
        ),
    }),
    "wan_alignment": ("BOOLEAN", {
        "default": True,
        "tooltip": "Snap boundaries to WAN frame alignment (frame_count % 4 == 1).",
    }),
    "auto_preset": (list(WAN_PRESETS.keys()), {
        "default": "WAN_480p",
        "tooltip": "Resolution preset for target dimensions from aspect ratio.",
    }),
    "padding": ("FLOAT", {
        "default": 0.1, "min": 0.0, "max": 0.5, "step": 0.05,
        "tooltip": "Expand union bbox by this fraction per side before VAE snap.",
    }),
}


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

class NV_AspectChunkPreview:
    """Preview aspect-ratio segmentation without saving any files.

    Shows chunk breakdown, bbox regions, and target resolutions.
    Outputs plan as a JSON string for inspection or downstream use.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": dict(_ANALYSIS_INPUTS)}

    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("plan_json", "num_chunks", "info")
    OUTPUT_NODE = True
    FUNCTION = "execute"
    CATEGORY = "NV_Utils"
    DESCRIPTION = (
        "Preview aspect-ratio chunk segmentation without saving files. "
        "Shows how the video would be split based on bbox AR changes."
    )

    def execute(self, mask, aspect_threshold, min_chunk_frames, wan_alignment, auto_preset, padding):
        plan, info_lines = analyze_aspect_chunks(
            mask, aspect_threshold, min_chunk_frames, wan_alignment, auto_preset, padding,
        )
        info = "\n".join(info_lines)
        plan_json = json.dumps(plan, indent=2)
        print(info)
        return {"ui": {"text": [info]}, "result": (plan_json, plan["num_chunks"], info)}


class NV_AspectChunkPlanner:
    """Segment video temporally by bounding-box aspect ratio changes.

    Each chunk gets its own union bbox and WAN-aligned target resolution,
    exported as a JSON plan file.
    """

    @classmethod
    def INPUT_TYPES(cls):
        inputs = dict(_ANALYSIS_INPUTS)
        inputs["output_dir"] = ("STRING", {
            "default": "",
            "tooltip": "JSON output directory. Empty = ComfyUI output dir.",
        })
        inputs["filename_prefix"] = ("STRING", {
            "default": "aspect_chunk_plan",
            "tooltip": "Prefix for the output JSON filename.",
        })
        return {"required": inputs}

    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("plan_json", "num_chunks", "info")
    OUTPUT_NODE = True
    FUNCTION = "execute"
    CATEGORY = "NV_Utils"
    DESCRIPTION = (
        "Segments a video temporally based on bounding-box aspect ratio "
        "changes. Each chunk gets its own optimal crop region and target "
        "resolution, written to a JSON plan file."
    )

    def execute(
        self, mask, aspect_threshold, min_chunk_frames, wan_alignment,
        auto_preset, padding, output_dir, filename_prefix,
    ):
        plan, info_lines = analyze_aspect_chunks(
            mask, aspect_threshold, min_chunk_frames, wan_alignment, auto_preset, padding,
        )

        # Write JSON
        out_dir = output_dir.strip() or folder_paths.get_output_directory()
        os.makedirs(out_dir, exist_ok=True)
        filepath = _get_unique_filepath(
            os.path.join(out_dir, f"{filename_prefix}.json")
        )
        with open(filepath, "w") as f:
            json.dump(plan, f, indent=2)

        info_lines.append(f"\nSaved: {filepath}")
        info = "\n".join(info_lines)
        print(info)

        return {"ui": {"text": [info]}, "result": (filepath, plan["num_chunks"], info)}


class NV_AspectChunkReader:
    """Load a specific chunk from an aspect plan, slicing video and controls.

    Reads a plan (JSON string or file path), extracts per-chunk metadata
    (frame range, bbox, target resolution), and temporally slices video
    and control inputs to the chunk's frame range.

    Supports both full-video and pre-sliced inputs (auto-detected).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("IMAGE", {
                    "tooltip": "Full input video frames [T,H,W,C] or pre-sliced chunk.",
                }),
                "plan_json": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Aspect chunk plan — either a JSON string (from AspectChunkPreview) "
                        "or a file path (from AspectChunkPlanner)."
                    ),
                }),
                "chunk_index": ("INT", {
                    "default": 0, "min": 0, "max": 99,
                    "tooltip": "Which chunk to read (0-indexed).",
                }),
            },
            "optional": {
                "control_1": ("IMAGE", {
                    "tooltip": "First control video (depth, edge, etc.) — sliced in sync.",
                }),
                "control_2": ("IMAGE", {
                    "tooltip": "Second control video — sliced in sync.",
                }),
                "control_3": ("IMAGE", {
                    "tooltip": "Third control video — sliced in sync.",
                }),
                "control_4": ("IMAGE", {
                    "tooltip": "Fourth control video — sliced in sync.",
                }),
                "latent": ("LATENT", {
                    "tooltip": (
                        "Pre-pass latent (full video). Temporally sliced to chunk range "
                        "using WAN VAE frame mapping."
                    ),
                }),
            },
        }

    RETURN_TYPES = (
        "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE",
        "INT", "INT", "INT", "INT",
        "INT", "INT", "INT", "INT",
        "INT", "INT",
        "FLOAT", "BOOLEAN",
        "INT",
        "LATENT",
        "STRING",
    )
    RETURN_NAMES = (
        "chunk_video", "chunk_ctrl_1", "chunk_ctrl_2", "chunk_ctrl_3", "chunk_ctrl_4",
        "start_frame", "end_frame", "frame_count", "latent_frames",
        "bbox_x", "bbox_y", "bbox_w", "bbox_h",
        "target_width", "target_height",
        "mean_aspect", "wan_aligned",
        "num_chunks",
        "chunk_latent",
        "chunk_info",
    )
    FUNCTION = "execute"
    CATEGORY = "NV_Utils"
    DESCRIPTION = (
        "Reads an aspect chunk plan and slices video + controls to the "
        "specified chunk's frame range. Outputs bbox crop region and "
        "target resolution for downstream processing."
    )

    def execute(
        self, video, plan_json, chunk_index,
        control_1=None, control_2=None, control_3=None, control_4=None,
        latent=None,
    ):
        # ---- Parse plan (JSON string or file path) ----
        plan_str = plan_json.strip()
        if not plan_str:
            raise ValueError("[NV_AspectChunkReader] plan_json is empty.")

        if plan_str.startswith("{"):
            plan = json.loads(plan_str)
        else:
            with open(plan_str, "r") as f:
                plan = json.load(f)

        chunks = plan.get("chunks", [])
        num_chunks = len(chunks)
        if chunk_index >= num_chunks:
            raise ValueError(
                f"[NV_AspectChunkReader] chunk_index {chunk_index} out of range. "
                f"Plan has {num_chunks} chunks (0-{num_chunks - 1})."
            )

        chunk = chunks[chunk_index]
        start_frame = chunk["start_frame"]
        end_frame = chunk["end_frame"]
        expected_frames = chunk.get("frame_count", end_frame - start_frame)

        video_meta = plan.get("video_metadata", {})
        expected_total = video_meta.get("total_frames", 0)

        # ---- Detect pre-sliced vs full video ----
        input_frames = video.shape[0]
        is_pre_sliced = (input_frames == expected_frames)
        is_full_video = (expected_total > 0 and input_frames == expected_total)

        if not is_pre_sliced and not is_full_video:
            raise ValueError(
                f"[NV_AspectChunkReader] Video has {input_frames} frames.\n"
                f"Expected either:\n"
                f"  - Pre-sliced chunk: {expected_frames} frames\n"
                f"  - Full video: {expected_total} frames"
            )

        if is_full_video and not is_pre_sliced:
            chunk_video = video[start_frame:end_frame].clone()
            slice_mode = "sliced"
        else:
            chunk_video = video
            slice_mode = "pre-sliced"

        # ---- Slice control videos ----
        controls = [control_1, control_2, control_3, control_4]
        chunk_controls = []

        for i, ctrl in enumerate(controls):
            if ctrl is not None:
                ctrl_frames = ctrl.shape[0]
                ctrl_pre = (ctrl_frames == expected_frames)
                ctrl_full = (expected_total > 0 and ctrl_frames == expected_total)

                if not ctrl_pre and not ctrl_full:
                    raise ValueError(
                        f"[NV_AspectChunkReader] Control {i + 1} has {ctrl_frames} frames.\n"
                        f"Expected either {expected_frames} (pre-sliced) or {expected_total} (full)."
                    )

                if ctrl_full and not ctrl_pre:
                    chunk_controls.append(ctrl[start_frame:end_frame].clone())
                else:
                    chunk_controls.append(ctrl)
            else:
                chunk_controls.append(None)

        # ---- Slice latent (WAN VAE temporal mapping) ----
        chunk_latent = None
        if latent is not None:
            samples = latent["samples"]  # [B, C, T, H, W]
            total_latent_t = samples.shape[2]
            chunk_latent_t = (expected_frames - 1) // 4 + 1

            if total_latent_t == chunk_latent_t:
                chunk_latent = latent
            else:
                latent_start = 0 if start_frame == 0 else (start_frame - 1) // 4 + 1
                latent_end = min(latent_start + chunk_latent_t, total_latent_t)
                chunk_samples = samples[:, :, latent_start:latent_end, :, :].clone()
                chunk_latent = {"samples": chunk_samples}
                for k in LATENT_SAFE_KEYS:
                    if k in latent:
                        chunk_latent[k] = latent[k]

        # ---- Extract aspect-specific fields ----
        bbox_x = chunk.get("bbox_x", 0)
        bbox_y = chunk.get("bbox_y", 0)
        bbox_w = chunk.get("bbox_w", video_meta.get("width", 0))
        bbox_h = chunk.get("bbox_h", video_meta.get("height", 0))
        target_w = chunk.get("target_width", 0)
        target_h = chunk.get("target_height", 0)
        mean_aspect = chunk.get("mean_aspect", 1.0)
        wan_ok = chunk.get("wan_aligned", False)
        l_frames = chunk.get("latent_frames", (expected_frames - 1) // 4 + 1)

        # ---- Info ----
        num_ctrls = sum(1 for c in chunk_controls if c is not None)
        wan_tag = "WAN OK" if wan_ok else "WAN MISS"
        info_lines = [
            f"[AspectChunkReader] Chunk {chunk_index}/{num_chunks - 1} ({slice_mode})",
            f"  Frames: {start_frame}-{end_frame - 1} ({expected_frames}f, {l_frames}L) [{wan_tag}]",
            f"  BBox: x={bbox_x} y={bbox_y} w={bbox_w} h={bbox_h}",
            f"  Target: {target_w}x{target_h}  AR={mean_aspect:.2f}",
            f"  Controls: {num_ctrls}",
        ]
        if chunk_latent is not None:
            info_lines.append(f"  Latent: {chunk_latent['samples'].shape[2]} temporal frames")
        chunk_info = "\n".join(info_lines)
        print(chunk_info)

        return (
            chunk_video, chunk_controls[0], chunk_controls[1], chunk_controls[2], chunk_controls[3],
            start_frame, end_frame, expected_frames, l_frames,
            bbox_x, bbox_y, bbox_w, bbox_h,
            target_w, target_h,
            mean_aspect, wan_ok,
            num_chunks,
            chunk_latent,
            chunk_info,
        )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "NV_AspectChunkPlanner": NV_AspectChunkPlanner,
    "NV_AspectChunkPreview": NV_AspectChunkPreview,
    "NV_AspectChunkReader": NV_AspectChunkReader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_AspectChunkPlanner": "NV Aspect Chunk Planner",
    "NV_AspectChunkPreview": "NV Aspect Chunk Preview",
    "NV_AspectChunkReader": "NV Aspect Chunk Reader",
}
