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
# Two-Mask Architecture (2026-05-01, simplified post-research)
# =============================================================================
# This node emits TWO logically-distinct masks for TWO different jobs.
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
#   │   Knobs : vace_input_grow_px (the one mask-shape knob), vae_grid_snap   │
#   │           (optional VAE 8x8 quantization). Tagged [GEN MASK].           │
#   │                                                                         │
#   │ Job 2: STITCH/BLEND MASK  (output: stitch_mask, tight_mask)             │
#   │   Job   : Tells the compositor what region to alpha-blend.              │
#   │   Stage : Post-sampler — pure pixel-space alpha composite.              │
#   │   Goal  : Hide the seam between AI output and original frame.           │
#   │   Knobs : vace_stitch_source, vace_stitch_erosion_px,                   │
#   │           vace_stitch_feather_px. Tagged [STITCH MASK].                 │
#   └─────────────────────────────────────────────────────────────────────────┘
#
# Other tooltip tag legend:
#   [SHAPE]    — pre-stage geometry decisions (tight vs bbox, bbox padding)
#   [CLEANUP]  — applied to raw mask BEFORE the gen/stitch split
#   [CHUNK]    — multi-chunk continuity (previous_chunk_tail, tail_overlap)
#   [ADVANCED] — VAE-specific or rarely-tweaked
#
# =============================================================================
# RETIREMENTS — 2026-05-01 (post research + multi-AI debate)
# =============================================================================
# The following GEN-MASK params were retired after research showed VACE was
# trained on BINARY masks (paper § 3.2; arxiv 2503.07598) and feathered/soft
# input is OUT OF DISTRIBUTION:
#
#   feather_blocks  → REMOVED. Soft GEN masks are OOD per VACE training data.
#                     Hard cliff: was defaulted to 1.5 ("NEVER 0" lore); the rule
#                     was retconned project-internal misattribution. See CHANGELOG.
#   erosion_blocks  → REMOVED. Mathematically equivalent to vace_input_grow_px
#                     with negative values; the "VAE block alignment" framing did
#                     NOT survive scrutiny — erosion shifts boundary inward without
#                     snapping to the 8x8 grid. For real grid alignment use the new
#                     vae_grid_snap toggle (per-block majority quantization).
#   vace_halo_px    → REMOVED. Functionally identical to vace_input_grow_px (both
#                     uniformly expand the GEN mask). The "Seam-Absorbing Halo"
#                     framing was just labeling for a uniform expansion operation.
#   fill_mode       → REMOVED. "soft" with gray-127 fill IS the canonical training
#                     distribution per VACE User Guide. "none" mode was OOD. Hardcoded
#                     to soft.
#   fill_value      → REMOVED. Locked to 0.5 by VAE centering math — WanVaceToVideo
#                     centers control_video by -0.5, so 0.5 maps to 0 in encoded
#                     inactive channel. Anything else is OOD. Hardcoded to 0.5.
#   mode (auto/manual) → REMOVED. With erosion/feather retired, nothing is auto-derived.
#
# NEW PARAMETER:
#   vae_grid_snap (BOOLEAN, default False) — quantizes the binary mask boundary
#                     to multiples of vae_stride via per-block majority voting.
#                     Genuinely snaps to the VAE encoder grid (unlike erosion which
#                     just shifted the boundary inward).
#
# Source verification (multi-AI VACE deep-dive, 2026-05-01):
#   - VACE paper § 3.2: "the mask is binary, with 1 and 0 indicating edit vs keep"
#   - Official User Guide: src_mask = white-generate / black-retain
#   - Training data: SAM2 instance masks + binary morphological augmentation
#   - VACE injection at wan/model.py:802 is `x += c_skip * vace_strength[iii]`
#     (additive residual, not hard gate)
#   - Mask path at nodes_wan.py:339-354: control video multiplicatively split,
#     mask gets 64-channel 8x8 block decomposition via nearest-exact interpolate
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

import cv2
import numpy as np
import torch

from .mask_ops import mask_erode_dilate, mask_blur, mask_fill_holes, mask_remove_noise, mask_smooth
from .bbox_ops import (
    extract_bboxes, build_bbox_masks,
    gaussian_smooth_1d, median_filter_1d,
)


def compute_inscribed_radius(mask):
    """Per-frame inscribed-circle radius via Euclidean distance transform.

    For each frame, returns the largest distance any mask pixel has from a non-mask
    pixel — i.e., the radius of the largest disk that fits entirely inside the mask.
    Used by `vace_mask_prep.py` to cap auto-mode erosion/feather to safe values:
    erosion ≤ min_radius / 3, feather ≤ min_radius / 2.

    Args:
        mask: torch.Tensor [B, H, W] or [H, W] in [0, 1].

    Returns:
        (radii, min_frame): radii is a list[float] of per-frame inscribed radii in
                             pixels; min_frame is the index of the smallest radius
                             (the bottleneck frame for safe erosion sizing).
    """
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    radii = []
    arr = (mask > 0.5).detach().cpu().numpy().astype(np.uint8)
    for i in range(arr.shape[0]):
        m = arr[i]
        if m.sum() == 0:
            radii.append(0.0)
            continue
        dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
        radii.append(float(dist.max()))
    min_frame = int(np.argmin(radii)) if radii else 0
    return radii, min_frame


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


def _snap_mask_to_vae_grid(mask, stride=8):
    """Snap mask boundary to multiples of `stride` via per-block majority quantization.

    For each `stride x stride` block of pixels, count foreground pixels (mask > 0.5).
    If more than 50% of the block is foreground, the entire block becomes 1.0; otherwise
    the entire block becomes 0.0. Result: every stride-aligned block has a uniform value,
    perfectly aligned with the VACE encoder's VAE 8x8 spatial stride.

    This eliminates fractional-block ambiguity that arises when an arbitrary binary
    boundary falls inside a single VAE block (the encoder sees a partial block which
    encodes to an in-between latent value). With grid-snapping on, every block is
    fully foreground or fully background -- the encoded mask becomes crisp.

    Different from erosion: erosion shifts the boundary inward by a fixed offset,
    leaving its alignment to the grid arbitrary. Grid-snap actually quantizes the
    boundary onto stride-aligned positions.

    Args:
        mask: [B, H, W] tensor in [0, 1]. Values >0.5 treated as foreground.
        stride: VAE spatial stride (8 for WAN).

    Returns:
        [B, H, W] tensor with grid-quantized boundary; each stride x stride block
        is uniformly 0.0 or 1.0.
    """
    if mask.dim() != 3:
        raise ValueError(
            f"_snap_mask_to_vae_grid expects [B,H,W], got shape {tuple(mask.shape)}"
        )
    B, H, W = mask.shape
    pad_h = (stride - H % stride) % stride
    pad_w = (stride - W % stride) % stride

    if pad_h or pad_w:
        # Pad to clean stride multiples for the reshape, snap, then crop back.
        # Pad value 0 (background) is the safe default — won't introduce false fg.
        # BUT: we must also track which pixels in each edge block are real vs. padded
        # so the per-block majority vote is computed against actual valid pixel count,
        # not the full stride*stride area. Otherwise a fully-foreground edge strip
        # whose partial block contains <stride*stride/2 valid pixels gets erased by
        # phantom zeros (valid pixels are about to be cropped away anyway, so the
        # "no" votes from padding shouldn't influence the block's snap state).
        mask_p = torch.nn.functional.pad(mask, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
        validity = torch.zeros_like(mask_p)
        validity[:, :H, :W] = 1.0
    else:
        mask_p = mask
        validity = None
    Hp, Wp = mask_p.shape[1], mask_p.shape[2]

    # `.contiguous()` before `.view()` — pad/slicing can leave the tensor non-contiguous
    # on some upstreams; .view() would raise. Cheap no-op when already contiguous.
    binary = (mask_p > 0.5).to(mask.dtype).contiguous()
    blocks = binary.view(B, Hp // stride, stride, Wp // stride, stride)
    fg_count = blocks.sum(dim=(2, 4))                                   # [B, H/s, W/s]

    if validity is not None:
        valid_blocks = validity.contiguous().view(B, Hp // stride, stride, Wp // stride, stride)
        valid_count = valid_blocks.sum(dim=(2, 4))                      # [B, H/s, W/s]
        # Per-block half-threshold over actual valid pixel count.
        # Strict `>` preserves tie-goes-to-off (matches old full-block convention:
        # exactly stride*stride/2 fg out of stride*stride → OFF).
        threshold = valid_count * 0.5
    else:
        threshold = (stride * stride) * 0.5

    block_on = (fg_count > threshold).to(mask.dtype)                    # [B, H/s, W/s]

    # Expand each block back to full resolution
    snapped = block_on.unsqueeze(2).unsqueeze(4).expand(
        B, Hp // stride, stride, Wp // stride, stride
    ).reshape(B, Hp, Wp)
    return snapped[:, :H, :W]


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
                    "default": "bbox",
                    "tooltip": (
                        "[SHAPE] Pre-stage geometry decision — affects BOTH gen and stitch masks. "
                        "bbox (DEFAULT): convert to per-frame bounding boxes with temporal smoothing — "
                        "in-distribution per VACE training (paper § 3.2 + bbox augmentation), "
                        "naturally 8x8-block-aligned when sized correctly. "
                        "as_is: use input mask directly (tight segmentation) — out-of-distribution "
                        "for VACE; only use for tight subject masks at high denoise (>~0.75) where "
                        "bbox produces visible rectangular pasted-on artifacts."
                    ),
                }),
            },
            "optional": {
                "mask_config": ("MASK_PROCESSING_CONFIG", {
                    "tooltip": (
                        "Optional shared config bus from NV_MaskProcessingConfig. When connected, "
                        "overrides matching widgets: "
                        "[GEN] vace_input_grow_px; "
                        "[STITCH] stitch_erosion/stitch_feather; "
                        "[CLEANUP] fill_holes/remove_noise/smooth. "
                        "Bus keys for RETIRED params (vace_erosion_blocks, vace_feather_blocks, "
                        "vace_halo_px) are silently ignored — those operations are no longer applied. "
                        "Future v2.1 may split this into MASK_GEN_CONFIG + MASK_BLEND_CONFIG buses."
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
                "vae_grid_snap": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "[GEN MASK] Quantize the binary mask boundary to multiples of vae_stride (8 px) "
                        "via per-block majority voting. Each 8x8 block becomes uniformly 1.0 or 0.0 — "
                        "perfectly aligned with VACE's encoder grid, eliminating fractional-block "
                        "ambiguity at arbitrary boundaries. Different from erosion (retired) — does NOT "
                        "shift the boundary inward, just snaps it to grid lines. Useful for tight "
                        "irregular masks where boundary alignment with VAE blocks reduces encoded "
                        "artifacts. False (default) = pass binary mask as-is; True = snap to grid."
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
                        "[GEN MASK] The ONE mask-shape knob for VACE. Uniformly grow (positive) or "
                        "shrink (negative) the GENERATION mask in pixels. "
                        "USE POSITIVE TO: cover BOTH source-subject silhouette AND target-subject "
                        "silhouette (head-swap with size mismatch — original head hidden + target shape "
                        "has room to generate); add a repaint margin so the eventual stitch seam falls "
                        "inside repainted content (the old 'Seam-Absorbing Halo' role — now subsumed). "
                        "USE NEGATIVE TO: shrink the editable region inward (the old 'erosion_blocks' "
                        "role — now subsumed; both ops were morphological erosion, just expressed in "
                        "different units). "
                        "0 = input mask already covers the right area. "
                        "Subsumes retired params: vace_halo_px (= positive grow) and erosion_blocks "
                        "(= negative grow). DOES NOT affect the stitch mask."
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
        "Single-node VACE control video + mask preparation. Slim version: research showed "
        "VACE was trained on BINARY masks (paper § 3.2), so feathering / erosion / soft-fill "
        "params have been retired as out-of-distribution.\n"
        "\n"
        "GEN MASK (control_masks) → drives WAN regeneration. Binary by default; one knob "
        "(vace_input_grow_px) for uniform expand/shrink. Optional vae_grid_snap quantizes "
        "boundaries to the VAE 8x8 grid for tight irregular masks.\n"
        "\n"
        "STITCH MASK (stitch_mask, tight_mask) → drives post-sampler pixel compositing. "
        "Soft edges fine here; vace_stitch_* params shape it independently.\n"
        "\n"
        "Pre-stage params [SHAPE]/[CLEANUP] apply before the gen/stitch split. Control video "
        "is always soft-filled with VACE neutral 0.5 (canonical training distribution).\n"
        "\n"
        "Wire control_masks → WanVaceToVideo. Wire stitch_mask → NV_VaceVideoStitch / "
        "NV_InpaintStitch_V2 for post-sampler blend."
    )

    def execute(self, image, mask, mask_shape,
                mask_config=None, bbox_padding=0.15, bbox_smooth_frames=5,
                vae_grid_snap=False, threshold=False,
                vace_input_grow_px=0, cleanup_fill_holes=0, cleanup_remove_noise=0,
                cleanup_smooth=0, vae_stride=8,
                vace_stitch_source="tight",
                vace_stitch_erosion_px=0, vace_stitch_feather_px=8,
                previous_chunk_tail=None, tail_overlap_frames=4):

        # Hardcoded canonical VACE values (per research 2026-05-01):
        # - Fill mode is always "soft" alpha-composite (VACE training distribution).
        # - Fill value is locked to 0.5 (gray-127), the VACE neutral. WanVaceToVideo
        #   centers control_video by -0.5, so 0.5 maps to 0 in the encoded inactive
        #   channel. Anything else is OOD per the training data + paper.
        FILL_VALUE = 0.5

        # Apply shared config override if connected. Only references SURVIVING keys.
        # Bus keys for retired params (vace_erosion_blocks, vace_feather_blocks,
        # vace_halo_px) are silently ignored.
        from .mask_processing_config import apply_vace_mask_config
        vals = apply_vace_mask_config(mask_config,
            cleanup_fill_holes=cleanup_fill_holes,
            cleanup_remove_noise=cleanup_remove_noise,
            cleanup_smooth=cleanup_smooth,
            vace_stitch_erosion_px=vace_stitch_erosion_px,
            vace_stitch_feather_px=vace_stitch_feather_px,
            vace_input_grow_px=vace_input_grow_px,
        )
        fill_holes_v = vals["cleanup_fill_holes"]
        remove_noise_v = vals["cleanup_remove_noise"]
        smooth_v = vals["cleanup_smooth"]
        stitch_erosion = vals["vace_stitch_erosion_px"]
        stitch_feather = vals["vace_stitch_feather_px"]
        vace_input_grow_px = vals["vace_input_grow_px"]

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
            f"[NV_VaceControlVideoPrep v2.1] shape={mask_shape} | grid_snap={vae_grid_snap} | "
            f"vae_stride={vae_stride}px | fill=soft@0.5 (canonical VACE neutral)"
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

        # All-ones early return: skipped when vace_input_grow_px < 0 because the operator
        # explicitly asked to SHRINK an already-saturated mask. Without this gate the
        # negative grow is silently ignored (saturated input + grow=-16 stays saturated).
        # Fall through to the normal pipeline so Step 1b erosion runs.
        if mask_min > 0.99 and vace_input_grow_px >= 0:
            fill = torch.full_like(image, FILL_VALUE)
            info_lines.append("  Mask is all ones — full fill applied (0.5 neutral)")
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

        # --- Step 3: Binarize the GEN mask ---
        # VACE was trained on binary masks (paper § 3.2). Cleanup ops above use
        # greyscale morphology which can leave soft values; enforce binary at the
        # boundary between mask processing and VACE consumption. This is the
        # binary-first principle from the 2026-05-01 research+debate convergence.
        result_mask = (result_mask > 0.5).to(result_mask.dtype)
        info_lines.append("  Binarized for VACE (binary-first per training distribution)")

        # --- Step 4: Optional VAE grid snap ---
        # Genuinely quantizes the binary boundary onto stride-aligned positions
        # via per-block majority voting. Different from the retired erosion_blocks
        # (which just shifted the boundary inward without grid alignment). When
        # enabled, every 8x8 VAE block is uniformly fully foreground or fully
        # background — eliminates fractional-block ambiguity at arbitrary boundaries.
        if vae_grid_snap:
            result_mask = _snap_mask_to_vae_grid(result_mask, stride=vae_stride)
            info_lines.append(
                f"  VAE grid snap applied: boundary quantized to {vae_stride}px-aligned blocks "
                f"via per-block majority vote"
            )

        # --- Step 5: Composite control video (always soft-fill with VACE neutral 0.5) ---
        # WanVaceToVideo centers control_video by -0.5 so 0.5 maps to 0 in encoded
        # inactive channel. This is canonical per the VACE User Guide. The retired
        # fill_mode='none' path was OOD relative to the training distribution.
        mask_expanded = result_mask.unsqueeze(-1)
        control_video = image * (1.0 - mask_expanded) + FILL_VALUE * mask_expanded
        info_lines.append(
            f"  Fill: soft composite with VACE neutral {FILL_VALUE:.2f} (gray-127)"
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
