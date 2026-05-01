"""
NV VACE Control Video Prep — Single-node VACE control video + mask preparation.

Replaces the multi-node pipeline of: Image To Mask → InvertMask → VaceMaskPrep → ImageRemoveAlpha
                                      Image To Mask → VaceMaskPrep → WanVaceToVideo

The problem with separate nodes: two VaceMaskPrep instances process different masks
(original vs inverted), producing non-complementary results. Erosion on the inverted
mask EXPANDS the grey zone, while erosion on the original mask SHRINKS the VACE zone,
creating an N-pixel gap where grey leaks into the "preserve" channel → visible halo.

This node processes the mask ONCE, then uses that same result for both:
  1. Compositing the control video (fill masked area with VACE-neutral color)
  2. Output VACE mask for WanVaceToVideo's control_masks input

Guarantees perfect pixel-level alignment between grey zone and VACE mask boundary.

# =============================================================================
# Two-Mask Architecture (2026-05-01)
# =============================================================================
# This node emits TWO logically-distinct masks for TWO different jobs. Read this
# before tweaking — same-named-looking params shape DIFFERENT masks at DIFFERENT
# pipeline stages with DIFFERENT failure modes.
#
#   ┌─────────────────────────────────────────────────────────────────────────┐
#   │ Job 1: GENERATION MASK  (output: control_masks)                         │
#   │   Job   : Tells VACE/WAN what region to repaint.                        │
#   │   Stage : Pre-sampler — encoded into latent conditioning.               │
#   │   Goal  : Cover BOTH source-subject silhouette AND target-subject       │
#   │           silhouette (so original doesn't leak through, AND the model   │
#   │           has room to generate the new shape). NOT too wide — over-     │
#   │           expansion makes WAN hallucinate in obviously-not-subject      │
#   │           areas.                                                        │
#   │   Knobs : erosion_blocks, feather_blocks, vace_input_grow_px,           │
#   │           vace_halo_px, fill_mode, fill_value.                          │
#   │           Tooltips for these are tagged [GEN MASK].                     │
#   │                                                                         │
#   │ Job 2: STITCH/BLEND MASK  (output: stitch_mask, tight_mask)             │
#   │   Job   : Tells the compositor what region to alpha-blend.              │
#   │   Stage : Post-sampler — pure pixel-space alpha composite.              │
#   │   Goal  : Hide the seam between AI output and original frame. Can be   │
#   │           tighter than the GEN mask (pure compositing job, no model    │
#   │           encoding constraints). Soft edges are a feature here.        │
#   │   Knobs : vace_stitch_source, vace_stitch_erosion_px,                   │
#   │           vace_stitch_feather_px.                                       │
#   │           Tooltips for these are tagged [STITCH MASK].                  │
#   └─────────────────────────────────────────────────────────────────────────┘
#
# Other tooltip tag legend:
#   [SHAPE]    — pre-stage geometry decisions (tight vs bbox, bbox padding)
#   [CLEANUP]  — applied to raw mask BEFORE the gen/stitch split
#   [CHUNK]    — multi-chunk continuity (previous_chunk_tail, tail_overlap)
#   [ADVANCED] — VAE-specific or rarely-tweaked
#
# Source verification (multi-AI VACE deep-dive, 2026-05-01): VACE injection at
# WAN model.py:802 is `x += c_skip * vace_strength[iii]` — additive residual,
# NOT a hard mask gate. The mask multiplicatively splits control_video into
# inactive/reactive branches at nodes_wan.py:339-340 BEFORE VAE encoding, then
# gets a 64-channel 8x8-block decomposition (nodes_wan.py:351-354). This
# confirms: feather_blocks ≥ 1 matters for clean ENCODING of the boundary
# (VAE block alignment), not for some abstract "gate cliff." See D-122.
# =============================================================================

Mask shape modes:
  - "as_is": Use the input mask directly (tight segmentation mask).
  - "bbox": Convert to per-frame bounding boxes with temporal smoothing. Produces
    cleaner VACE results because: (a) rectangular boundaries align with the VAE's
    8x8 spatial blocks, (b) the mask perimeter is shorter with no irregular notches,
    (c) VACE was trained with bbox masks as a first-class augmentation mode.

Fill modes:
  - "soft": Alpha-composite using the feathered mask. Both VAE channels (inactive/reactive)
    are smooth, but the inactive signal has quadratic attenuation in the transition zone
    [(real-0.5)*(1-m)² instead of (real-0.5)*(1-m)] due to double-masking.
  - "none": No fill — pass through real content. The VACE formula alone handles the
    transition, producing the mathematically ideal linear attenuation. Old face content
    appears in the reactive channel but is attenuated by the mask and overridden by LoRA/prompt.
    Best for face replacement with LoRA where you want the cleanest transition.

The correct VACE neutral value is 0.5 in [0,1] space. WanVaceToVideo centers by
subtracting 0.5, so fill=0.5 → centered=0.0 → both inactive and reactive get pure 0.5
(true neutral). The original #999999 (0.6) produces a slight brightness bias.
"""

import torch
import numpy as np
from scipy.ndimage import distance_transform_edt

from .mask_ops import mask_erode_dilate, mask_blur, mask_fill_holes, mask_remove_noise, mask_smooth
from .bbox_ops import (
    extract_bboxes, build_bbox_masks,
    gaussian_smooth_1d, median_filter_1d,
)


def compute_inscribed_radius(mask: torch.Tensor) -> tuple[list[float], int]:
    """Compute inscribed radius per frame via EDT. Returns (radii_list, min_radius_frame_idx)."""
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)

    radii = []
    for b in range(mask.shape[0]):
        mask_bin = (mask[b] > 0.5).cpu().numpy().astype(np.uint8)
        if np.any(mask_bin):
            dist = distance_transform_edt(mask_bin)
            radii.append(float(np.max(dist)))
        else:
            radii.append(0.0)

    min_idx = int(np.argmin(radii)) if radii else 0
    return radii, min_idx


def masks_to_bboxes(mask, padding, smooth_window, vae_stride, info_lines):
    """Convert per-frame masks to temporally-smoothed, padded, VAE-aligned bounding box masks.

    Delegates to bbox_ops for extraction, smoothing, and mask building.
    """
    B, H, W = mask.shape

    x1s, y1s, x2s, y2s, present = extract_bboxes(mask, info_lines)

    num_present = sum(present)
    if num_present == 0:
        return torch.zeros_like(mask)

    # Temporal smoothing (median + gaussian)
    if smooth_window > 1 and B > 1:
        win = min(smooth_window, B)
        x1s = gaussian_smooth_1d(median_filter_1d(x1s, win), win)
        y1s = gaussian_smooth_1d(median_filter_1d(y1s, win), win)
        x2s = gaussian_smooth_1d(median_filter_1d(x2s, win), win)
        y2s = gaussian_smooth_1d(median_filter_1d(y2s, win), win)
        info_lines.append(f"  BBox smoothing: median + gaussian, window={smooth_window}")

    return build_bbox_masks(x1s, y1s, x2s, y2s, padding, H, W,
                            info_lines=info_lines, vae_stride=vae_stride)


class NV_VaceControlVideoPrep:
    """Prepare both control_video and control_masks for WanVaceToVideo in a single node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": (
                        "Source video frames [B, H, W, 3]. Will be composited with neutral fill "
                        "inside the generation mask to produce control_video for WanVaceToVideo."
                    ),
                }),
                "mask": ("MASK", {
                    "tooltip": (
                        "Raw input mask [B, H, W]. Goes through CLEANUP → SHAPE → splits into "
                        "GEN MASK (drives VACE) + STITCH MASK (drives compositing). See module "
                        "docstring for the two-mask architecture."
                    ),
                }),
                "mask_shape": (["as_is", "bbox"], {
                    "default": "as_is",
                    "tooltip": (
                        "[SHAPE] Pre-stage geometry decision — affects BOTH gen and stitch masks. "
                        "as_is: use input mask directly (tight segmentation). "
                        "bbox: convert to per-frame bounding boxes with temporal smoothing. "
                        "Box masks produce cleaner VACE results because boundaries align with "
                        "VAE 8x8 blocks and match VACE training distribution."
                    ),
                }),
                "mode": (["auto", "manual"], {
                    "default": "auto",
                    "tooltip": (
                        "[GEN MASK] How to derive the GENERATION-mask erosion/feather values. "
                        "auto: analyze mask geometry and pick optimal values (capped by safety "
                        "limits derived from the mask itself). The widget values below act as "
                        "TARGETS in this mode. "
                        "manual: use erosion_blocks and feather_blocks widget values directly."
                    ),
                }),
                "erosion_blocks": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 4.0, "step": 0.25,
                    "tooltip": (
                        "[GEN MASK] Erode the GENERATION mask inward by this many VAE blocks (8px each "
                        "for WAN). 0.5 blocks = 4px. NEVER set to 0 — sharp 1-pixel edges produce "
                        "ALIASED encoded conditioning at the VAE-block grid (see D-122). "
                        "In auto mode this is the target value capped by mask geometry. "
                        "DOES NOT affect the stitch mask — for that, see vace_stitch_erosion_px."
                    ),
                }),
                "feather_blocks": ("FLOAT", {
                    "default": 1.5, "min": 0.0, "max": 8.0, "step": 0.25,
                    "tooltip": (
                        "[GEN MASK] Feather the GENERATION mask edge over this many VAE blocks. "
                        "1.5 blocks = 12px for WAN. NEVER set to 0 — feather over >1 VAE block "
                        "ensures the boundary gradient is captured cleanly in encoded space "
                        "(VACE encodes the mask via 8x8 block decomposition; sharp pixel-aligned "
                        "edges produce aliased conditioning). "
                        "DOES NOT affect the stitch mask — for that, see vace_stitch_feather_px."
                    ),
                }),
                "fill_mode": (["soft", "none"], {
                    "default": "soft",
                    "tooltip": (
                        "[GEN MASK] How control_video is filled inside the generation mask. "
                        "soft: alpha-composite fill using feathered mask. Smooth channels, "
                        "mild double-attenuation in transition zone. "
                        "none: no fill, real content everywhere. Mathematically ideal "
                        "inactive channel. Best for LoRA face replacement."
                    ),
                }),
            },
            "optional": {
                "mask_config": ("MASK_PROCESSING_CONFIG", {
                    "tooltip": (
                        "Optional shared config bus from NV_MaskProcessingConfig. When connected, "
                        "overrides matching widgets across BOTH gen and stitch stages: "
                        "[GEN] erosion/feather/input_grow/halo; "
                        "[STITCH] stitch_erosion/stitch_feather; "
                        "[CLEANUP] fill_holes/remove_noise/smooth. "
                        "Note: a future v2.1 may split this into separate MASK_GEN_CONFIG and "
                        "MASK_BLEND_CONFIG buses to make the gen/stitch separation explicit at "
                        "the bus level."
                    ),
                }),
                "bbox_padding": ("FLOAT", {
                    "default": 0.15, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": (
                        "[SHAPE] BBox padding as fraction of bbox dimensions (0.15 = 15%). "
                        "Only used when mask_shape='bbox'. VACE training uses 10-20%."
                    ),
                }),
                "bbox_smooth_frames": ("INT", {
                    "default": 5, "min": 0, "max": 31, "step": 2,
                    "tooltip": (
                        "[SHAPE] Temporal smoothing window for bbox coordinates (odd values recommended). "
                        "0 = disabled. Only used when mask_shape='bbox'."
                    ),
                }),
                "fill_value": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": (
                        "[GEN MASK] Fill color for masked area in control_video. 0.5 = true VACE neutral "
                        "(WanVaceToVideo centers control_video by -0.5, so 0.5 → 0.0 in the encoded "
                        "inactive channel). Only used when fill_mode='soft'."
                    ),
                }),
                "threshold": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "[CLEANUP] Binarize raw input mask at 0.5 before all downstream processing. "
                        "Use when input is a soft probability map and you want crisp downstream behavior."
                    ),
                }),
                "vace_input_grow_px": ("INT", {
                    "default": 0, "min": -128, "max": 128, "step": 1,
                    "tooltip": (
                        "[GEN MASK] Uniform grow (positive) or shrink (negative) of the GENERATION mask "
                        "BEFORE VACE processing. Use to ensure full coverage of BOTH source-subject "
                        "silhouette AND target-subject silhouette (e.g., head-swap with size mismatch — "
                        "original head must be hidden, AND target shape must have room to generate). "
                        "0 = input mask already covers the right area. "
                        "DOES NOT affect the stitch mask. (Previously: mask_grow)"
                    ),
                }),
                "cleanup_fill_holes": ("INT", {
                    "default": 0, "min": 0, "max": 128, "step": 1,
                    "tooltip": (
                        "[CLEANUP] Fill gaps/holes in raw mask (morphological closing). Applied BEFORE "
                        "the gen/stitch split — affects both downstream masks. "
                        "Legacy: prefer NV_MaskBinaryCleanup upstream for true binary fill_holes "
                        "(scipy.ndimage.binary_fill_holes is strictly more powerful). "
                        "(Previously: mask_fill_holes)"
                    ),
                }),
                "cleanup_remove_noise": ("INT", {
                    "default": 0, "min": 0, "max": 32, "step": 1,
                    "tooltip": (
                        "[CLEANUP] Remove isolated pixels (morphological opening). Applied BEFORE the "
                        "gen/stitch split. "
                        "Legacy: prefer NV_MaskBinaryCleanup.min_component_area_px upstream. "
                        "(Previously: mask_remove_noise)"
                    ),
                }),
                "cleanup_smooth": ("INT", {
                    "default": 0, "min": 0, "max": 127, "step": 1,
                    "tooltip": (
                        "[CLEANUP] Smooth jagged edges of raw mask (binarize + blur). Applied BEFORE "
                        "the gen/stitch split. Useful for SAM3 stairstep edges. (Previously: mask_smooth)"
                    ),
                }),
                "vae_stride": ("INT", {
                    "default": 8, "min": 4, "max": 32, "step": 4,
                    "tooltip": (
                        "[ADVANCED] VAE spatial stride in pixels. WAN VAE = 8. Verified at "
                        "comfy_extras/nodes_wan.py:348. Don't change unless using a non-WAN model."
                    ),
                }),
                "vace_stitch_source": (["tight", "bbox"], {
                    "default": "tight",
                    "tooltip": (
                        "[STITCH MASK] Source mask for the pixel-space STITCH/COMPOSITE output. "
                        "'tight': original input mask (precise subject boundary). "
                        "'bbox': bbox mask eroded inward (hides VACE generation seams at boundary). "
                        "DOES NOT affect what VACE generates — only what gets composited back to frame. "
                        "(Previously: stitch_source)"
                    ),
                }),
                "vace_halo_px": ("INT", {
                    "default": 16, "min": 0, "max": 48, "step": 4,
                    "tooltip": (
                        "[GEN MASK] Seam-Absorbing Halo: expand the GENERATION mask OUTWARD by N px so "
                        "WAN actually repaints a margin OUTSIDE the eventual stitch boundary. The stitch "
                        "composite seam then falls INSIDE repainted content rather than on a hard "
                        "AI/original boundary — major reduction in visible seam. 16px = recommended "
                        "(2 VAE blocks). 0 = disabled. "
                        "Despite the name, this shapes the GEN mask, not the stitch mask. "
                        "(Previously: halo_pixels)"
                    ),
                }),
                "vace_stitch_erosion_px": ("INT", {
                    "default": 0, "min": -32, "max": 32, "step": 1,
                    "tooltip": (
                        "[STITCH MASK] Erode (negative) or dilate (positive) the pixel-space STITCH "
                        "mask. Tightens or widens the composite boundary; INDEPENDENT of VACE "
                        "conditioning erosion. Use to fine-tune the visible seam without affecting "
                        "what WAN generated. (Previously: stitch_erosion)"
                    ),
                }),
                "vace_stitch_feather_px": ("INT", {
                    "default": 8, "min": 0, "max": 64, "step": 1,
                    "tooltip": (
                        "[STITCH MASK] Feather pixel-space STITCH mask edges for soft compositing. "
                        "8-16px = subtle, 24-32px = visible softening, 0 = hard edge. "
                        "Soft edges here are FINE — this is pure pixel-space alpha blending, not "
                        "encoded conditioning. (Previously: stitch_feather)"
                    ),
                }),
                "previous_chunk_tail": ("IMAGE", {
                    "tooltip": (
                        "[CHUNK] Last N frames of previous chunk's output (post-CropColorFix, pre-stitch). "
                        "Prepended to control video with mask=0 (preserve exactly). VACE continues "
                        "generation from these frames — trained inpainting continuation behavior. "
                        "No domain mismatch: tail goes through same VAE encode as control video. "
                        "Leave unconnected for chunk 0."
                    ),
                }),
                "tail_overlap_frames": ("INT", {
                    "default": 4, "min": 0, "max": 16, "step": 4,
                    "tooltip": (
                        "[CHUNK] Number of tail frames to prepend from previous_chunk_tail. "
                        "0 = disabled. MUST be a multiple of 4 (WAN temporal compression rule: "
                        "adding 4k frames to a valid 4k+1 count preserves validity). "
                        "4 = 1 latent frame, 8 = 2 latent frames. "
                        "Strip tail_trim frames from output after generation."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "MASK", "STRING", "INT")
    RETURN_NAMES = ("control_video", "control_masks", "stitch_mask", "tight_mask", "info", "tail_trim")
    FUNCTION = "execute"
    CATEGORY = "NV_Utils/VACE"
    DESCRIPTION = (
        "Single-node VACE control video + mask preparation with two-mask architecture.\n"
        "\n"
        "TWO logically-distinct masks for TWO different jobs (see module docstring):\n"
        "  • control_masks (GEN MASK)    → tells VACE/WAN what region to repaint.\n"
        "    Shaped by [GEN MASK] tagged params: erosion_blocks, feather_blocks,\n"
        "    vace_input_grow_px, vace_halo_px, fill_mode, fill_value.\n"
        "    Goal: cover the union of source-subject + target-subject silhouettes,\n"
        "    feathered in VAE-block units for clean encoded conditioning.\n"
        "  • stitch_mask (BLEND MASK)    → tells the compositor what region to alpha-blend.\n"
        "    Shaped by [STITCH MASK] tagged params: vace_stitch_source,\n"
        "    vace_stitch_erosion_px, vace_stitch_feather_px.\n"
        "    Goal: hide the seam in pixel space; soft edges are a feature here.\n"
        "  • tight_mask                  → backup output: tight original mask with stitch\n"
        "    erosion/feather applied. Use when you want a different boundary downstream\n"
        "    than the bbox-vs-tight choice in vace_stitch_source.\n"
        "\n"
        "Pre-stage params [SHAPE] + [CLEANUP] apply to the raw mask BEFORE the gen/stitch split.\n"
        "Wire control_masks → WanVaceToVideo. Wire stitch_mask → NV_VaceVideoStitch (or any\n"
        "downstream pixel compositor) for the post-sampler blend."
    )

    def execute(self, image, mask, mask_shape, mode, erosion_blocks, feather_blocks, fill_mode,
                mask_config=None, bbox_padding=0.15, bbox_smooth_frames=5,
                fill_value=0.5, threshold=False,
                vace_input_grow_px=0, cleanup_fill_holes=0, cleanup_remove_noise=0,
                cleanup_smooth=0, vae_stride=8,
                vace_stitch_source="tight", vace_halo_px=16,
                vace_stitch_erosion_px=0, vace_stitch_feather_px=8,
                previous_chunk_tail=None, tail_overlap_frames=4):

        # Apply shared config override if connected
        from .mask_processing_config import apply_vace_mask_config
        vals = apply_vace_mask_config(mask_config,
            cleanup_fill_holes=cleanup_fill_holes,
            cleanup_remove_noise=cleanup_remove_noise,
            cleanup_smooth=cleanup_smooth,
            erosion_blocks=erosion_blocks,
            feather_blocks=feather_blocks,
            vace_stitch_erosion_px=vace_stitch_erosion_px,
            vace_stitch_feather_px=vace_stitch_feather_px,
            vace_input_grow_px=vace_input_grow_px,
            vace_halo_px=vace_halo_px,
        )
        fill_holes_v = vals["cleanup_fill_holes"]
        remove_noise_v = vals["cleanup_remove_noise"]
        smooth_v = vals["cleanup_smooth"]
        erosion_blocks = vals["erosion_blocks"]
        feather_blocks = vals["feather_blocks"]
        stitch_erosion = vals["vace_stitch_erosion_px"]
        stitch_feather = vals["vace_stitch_feather_px"]
        vace_input_grow_px = vals["vace_input_grow_px"]
        vace_halo_px = vals["vace_halo_px"]

        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        # --- Step 0: Prepare tail frames (DON'T prepend yet) ---
        # Tail is prepended AFTER all mask analysis to avoid contaminating inscribed
        # radius, auto-mode geometry, bbox temporal smoothing. Only control_video and
        # control_masks get the prepend; stitch_mask/tight_mask stay at original length.
        tail_trim = 0
        _tail_frames = None
        if previous_chunk_tail is not None and tail_overlap_frames > 0:
            # Snap to multiple of 4 (WAN 4k+1 rule: adding 4k preserves valid frame counts)
            tail_overlap_frames = (tail_overlap_frames // 4) * 4
            if tail_overlap_frames > 0:
                available = previous_chunk_tail.shape[0]
                actual_tail = (min(tail_overlap_frames, available) // 4) * 4
                if actual_tail > 0:
                    start_idx = available - actual_tail
                    _tail_frames = previous_chunk_tail[start_idx:]

                    # Resize tail to match control video if dimensions differ
                    if _tail_frames.shape[1:] != image.shape[1:]:
                        _tail_frames = torch.nn.functional.interpolate(
                            _tail_frames.movedim(-1, 1),
                            size=(image.shape[1], image.shape[2]),
                            mode="bilinear", align_corners=False
                        ).movedim(1, -1)

                    _tail_frames = _tail_frames.to(device=image.device, dtype=image.dtype)
                    tail_trim = actual_tail
                    print(f"[NV_VaceControlVideoPrep] Tail: {actual_tail} frames prepared "
                          f"(will prepend to control_video/control_masks after mask processing)")

        info_lines = [
            f"[NV_VaceControlVideoPrep] shape={mask_shape} | mode={mode} | "
            f"fill={fill_mode} | vae_stride={vae_stride}px"
        ]

        # Helper: apply tail prepend to control_video + control_masks for any exit path
        def _apply_tail(ctrl_video, ctrl_mask):
            if _tail_frames is not None and tail_trim > 0:
                ctrl_video = torch.cat([_tail_frames, ctrl_video], dim=0)
                zeros = torch.zeros(tail_trim, ctrl_mask.shape[1], ctrl_mask.shape[2],
                                    device=ctrl_mask.device, dtype=ctrl_mask.dtype)
                ctrl_mask = torch.cat([zeros, ctrl_mask], dim=0)
            return ctrl_video, ctrl_mask

        # --- Edge case: trivial masks ---
        mask_min = mask.min().item()
        mask_max = mask.max().item()

        if mask_max < 0.01:
            info_lines.append("  Mask is all zeros — passing through unchanged")
            cv, cm = _apply_tail(image, mask)
            if tail_trim > 0:
                info_lines.append(f"  Tail prepended: {tail_trim} frames (trivial mask path)")
            info = "\n".join(info_lines)
            print(info)
            return (cv, cm, mask, mask, info, tail_trim)

        if mask_min > 0.99:
            fill = torch.full_like(image, fill_value) if fill_mode == "soft" else image
            info_lines.append("  Mask is all ones — full fill applied")
            cv, cm = _apply_tail(fill, mask)
            if tail_trim > 0:
                info_lines.append(f"  Tail prepended: {tail_trim} frames (trivial mask path)")
            info = "\n".join(info_lines)
            print(info)
            return (cv, cm, mask, mask, info, tail_trim)

        result_mask = mask.clone()

        # --- Step 1: Threshold ---
        if threshold:
            result_mask = (result_mask > 0.5).float()
            info_lines.append("  Threshold: applied at 0.5")

        # --- Step 1b: Mask pre-processing (grow/shrink, fill holes, denoise, smooth) ---
        preproc_applied = []
        if vace_input_grow_px != 0:
            result_mask = mask_erode_dilate(result_mask, vace_input_grow_px)
            preproc_applied.append(f"grow={vace_input_grow_px}px")
        if fill_holes_v > 0:
            result_mask = mask_fill_holes(result_mask, fill_holes_v)
            preproc_applied.append(f"fill_holes={fill_holes_v}px")
        if remove_noise_v > 0:
            result_mask = mask_remove_noise(result_mask, remove_noise_v)
            preproc_applied.append(f"remove_noise={remove_noise_v}px")
        if smooth_v > 0:
            result_mask = mask_smooth(result_mask, smooth_v)
            preproc_applied.append(f"smooth={smooth_v}px")
        if preproc_applied:
            result_mask = result_mask.clamp(0.0, 1.0)
            info_lines.append(f"  Mask pre-processing: {', '.join(preproc_applied)}")

        # Save preprocessed mask before bbox conversion (used for stitch mask recomputation)
        preprocessed_mask = result_mask.clone()

        # --- Step 2: BBox conversion (before erosion/feather) ---
        if mask_shape == "bbox":
            result_mask = masks_to_bboxes(
                result_mask, bbox_padding, bbox_smooth_frames,
                vae_stride, info_lines
            )

            # Re-check for trivial after bbox conversion
            if result_mask.max().item() < 0.01:
                cv, cm = _apply_tail(image, result_mask)
                if tail_trim > 0:
                    info_lines.append(f"  Tail prepended: {tail_trim} frames (post-bbox trivial path)")
                info = "\n".join(info_lines)
                print(info)
                return (cv, cm, mask, mask, info, tail_trim)

        # --- Step 3: Analyze mask geometry ---
        radii, min_frame = compute_inscribed_radius(result_mask)
        min_radius = radii[min_frame] if radii else 0.0
        radius_blocks = min_radius / vae_stride

        info_lines.append(
            f"  Inscribed radius: {min_radius:.1f}px ({radius_blocks:.1f} VAE blocks)"
            f" — min at frame {min_frame}/{len(radii)}"
        )

        # --- Step 4: Compute erosion/feather pixel values ---
        if mode == "auto":
            # Use widget values as TARGETS, capped by mask geometry safety limits
            erosion_target = erosion_blocks * vae_stride
            erosion_max_safe = max(1.0, min_radius / 3.0)
            erosion_px = int(round(min(erosion_target, erosion_max_safe)))

            feather_target = feather_blocks * vae_stride
            feather_max_safe = max(float(vae_stride), min_radius / 2.0)
            feather_px = int(round(min(feather_target, feather_max_safe)))

            info_lines.append(
                f"  Erosion (auto): {erosion_px}px "
                f"— target: {erosion_target:.0f}px, max safe: {erosion_max_safe:.0f}px"
            )
            info_lines.append(
                f"  Feather (auto): {feather_px}px "
                f"— target: {feather_target:.0f}px, max safe: {feather_max_safe:.0f}px"
            )

            if min_radius < vae_stride:
                info_lines.append(
                    f"  WARNING: Mask thinner than 1 VAE block ({min_radius:.1f}px < {vae_stride}px). "
                    f"Auto values are heavily constrained."
                )

        else:  # manual
            erosion_px = int(round(erosion_blocks * vae_stride))
            feather_px = int(round(feather_blocks * vae_stride))
            info_lines.append(
                f"  Erosion (manual): {erosion_blocks} blocks = {erosion_px}px"
            )
            info_lines.append(
                f"  Feather (manual): {feather_blocks} blocks = {feather_px}px"
            )

        # --- Step 5: Erode ---
        if erosion_px > 0:
            result_mask = mask_erode_dilate(result_mask, -erosion_px)

        # --- Step 5b: Seam-Absorbing Control Halo ---
        # Expand the BINARY eroded mask outward BEFORE feathering, so the halo gets a
        # clean uniform expansion. Dilating after feather would distort the gradient profile
        # (grey morphology expands high-value contours more than low-value).
        if vace_halo_px > 0:
            result_mask = mask_erode_dilate(result_mask, vace_halo_px)  # positive = dilate outward
            info_lines.append(
                f"  Halo: expanded VACE mask outward by {vace_halo_px}px "
                f"({vace_halo_px / vae_stride:.1f} VAE blocks) — applied pre-feather"
            )

        # --- Step 6: Blur (feather) ---
        if feather_px > 0:
            result_mask = mask_blur(result_mask, feather_px)
            effective_kernel = feather_px if feather_px % 2 == 1 else feather_px + 1
            info_lines.append(f"  Blur kernel: {effective_kernel}px (odd)")

        # --- Step 7: Clamp ---
        result_mask = result_mask.clamp(0.0, 1.0)

        # --- Step 8: Composite control video ---
        # Expand mask for broadcasting: [B, H, W] → [B, H, W, 1] for IMAGE [B, H, W, C]
        mask_expanded = result_mask.unsqueeze(-1)

        if fill_mode == "soft":
            # Alpha composite: real * (1 - mask) + fill * mask
            control_video = image * (1.0 - mask_expanded) + fill_value * mask_expanded
            info_lines.append(
                f"  Fill: soft composite with value={fill_value:.2f} "
                f"(VACE neutral={'yes' if abs(fill_value - 0.5) < 0.01 else 'NO — 0.5 is neutral'})"
            )
        else:  # none
            control_video = image
            info_lines.append("  Fill: none — real content passed through")

        # Transition summary
        transition_px = erosion_px + feather_px
        transition_blocks = transition_px / vae_stride
        transition_patches = transition_px / (vae_stride * 2)  # VACE patch stride (1,2,2)
        info_lines.append(
            f"  Transition zone: ~{transition_px}px "
            f"({transition_blocks:.1f} VAE blocks, {transition_patches:.1f} VACE attention patches)"
        )

        if fill_mode == "soft" and feather_px > 0:
            info_lines.append(
                "  Note: soft fill + feathered mask creates mild double-attenuation in "
                "transition zone. Switch to fill_mode='none' for linear attenuation."
            )

        # --- Step 9: Build tight mask (always from original input mask) ---
        tight_mask = mask.clone()
        if threshold:
            tight_mask = (tight_mask > 0.5).float()

        # Apply stitch erosion/feather to tight mask
        tight_processed = tight_mask.clone()
        if stitch_erosion != 0:
            tight_processed = mask_erode_dilate(tight_processed, stitch_erosion)
        if stitch_feather > 0:
            tight_processed = mask_blur(tight_processed, stitch_feather)
        tight_processed = tight_processed.clamp(0.0, 1.0)

        # --- Step 10: Build stitch mask (source depends on vace_stitch_source param) ---
        if vace_stitch_source == "bbox" and mask_shape == "bbox":
            # Derive stitch mask from the bbox (result_mask before erosion/feather for VACE).
            # Re-run bbox computation on the pre-erosion mask to get clean binary boxes,
            # then apply stitch_erosion/feather independently.
            # result_mask at this point has VACE erosion+feather applied, so we rebuild
            # from the bbox step output. We can use the binary bbox by thresholding result_mask
            # before VACE feathering was applied — but we already modified result_mask.
            # Recompute bboxes from preprocessed mask (matches VACE bbox input).
            bbox_binary = masks_to_bboxes(
                preprocessed_mask,
                bbox_padding, bbox_smooth_frames, vae_stride, []  # discard info
            )
            stitch_mask = bbox_binary.clone()

            # Erode inward to cut inside the bbox seam
            default_bbox_erosion = vae_stride  # 1 VAE block = 8px default inset
            effective_erosion = stitch_erosion if stitch_erosion != 0 else -default_bbox_erosion
            if effective_erosion != 0:
                stitch_mask = mask_erode_dilate(stitch_mask, effective_erosion)

            if stitch_feather > 0:
                stitch_mask = mask_blur(stitch_mask, stitch_feather)

            stitch_mask = stitch_mask.clamp(0.0, 1.0)

            info_lines.append(
                f"  Stitch mask: source=bbox, erosion={effective_erosion}px "
                f"({'default 1 VAE block inset' if stitch_erosion == 0 else 'manual'}),"
                f" feather={stitch_feather}px"
            )
        else:
            # Use tight mask (original behavior)
            stitch_mask = tight_processed
            if vace_stitch_source == "bbox" and mask_shape != "bbox":
                info_lines.append(
                    "  Stitch source: 'bbox' requested but mask_shape is not 'bbox' "
                    "— falling back to tight mask."
                )
            info_lines.append(
                f"  Stitch mask: source=tight, erosion={stitch_erosion}px, "
                f"feather={stitch_feather}px"
            )

        if stitch_feather == 0:
            info_lines.append(
                "  WARNING: stitch_feather=0 produces hard edges — visible seams likely."
            )

        # --- Step 11: Prepend tail to control_video and control_masks ONLY ---
        # stitch_mask and tight_processed stay at original N-frame length
        # (they're used for pixel compositing against the original video which has N frames)
        control_video, result_mask = _apply_tail(control_video, result_mask)
        if _tail_frames is not None and tail_trim > 0:
            info_lines.append(
                f"  Tail prepended: {tail_trim} frames (mask=0). "
                f"control_video={control_video.shape[0]} frames, "
                f"stitch/tight_mask={stitch_mask.shape[0]} frames (unchanged)")

        info = "\n".join(info_lines)
        print(info)

        return (control_video, result_mask, stitch_mask, tight_processed, info, tail_trim)


NODE_CLASS_MAPPINGS = {
    "NV_VaceControlVideoPrep": NV_VaceControlVideoPrep,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_VaceControlVideoPrep": "NV VACE Control Video Prep",
}
