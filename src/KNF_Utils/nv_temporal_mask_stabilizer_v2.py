"""
NV Temporal Mask Stabilizer V2 - Motion-compensated short-window temporal
consensus for jittery / lossy mask sequences.

Designed to follow NV_MaskBinaryCleanup in the canonical mask pipeline.
Solves the "per-frame mask is fine but the shot looks jittery" problem by
enforcing frame-to-frame coherence through a motion-compensated median.

Pipeline per frame t:
  1. For each neighbor in the window (t-1, t+1; or t-2..t+2 for window=5),
     compute backward optical flow from frame t to that neighbor (Farneback)
  2. Warp the neighbor's mask into frame t's coordinate space using the flow
  3. Stack [warped_neighbors, original_mask[t]]
  4. Per-pixel median (or mean) consensus — robust to single-frame outliers
     and naturally fills lost / empty mask frames
  5. Optional auto-repair: if current frame disagrees badly with both warped
     neighbors (IoU below threshold), replace with median of warped neighbors
     only (drop current frame's contribution from the consensus)
  6. Optional final Gaussian feather

Why "V2": the previous NV_TemporalMaskStabilizer was shelved 2026-04-24 (D-099)
due to critical bugs. V2 explicitly avoids each:
  - Frame-0 seed poison-pill → median of warped neighbors, no single-frame anchor
  - Infinity-ratio area normalization → median is scale-free, no area math
  - 10-px geodesic reach too small → optical flow is unbounded by design
  - Not motion-compensated → per-frame Farneback flow warp
  - OOM on 277f 1080p → streaming per-frame compute, NO batch flow precompute

MVP design from multi-AI brainstorm (2026-04-30) — Codex + Gemini converged
independently on this same architecture.

Performance notes:
  - Farneback runs on CPU; ~75-100ms per 1080p flow. With window=3, that's
    2 flows per frame ≈ 150-200ms per frame ≈ 50s for a 277f shot.
  - Memory: uint8 grayscale + float32 masks held in RAM upfront. ~5 GB for
    277f at 1080p. Acceptable for offline pipeline; tile-streaming is a
    follow-up if it ever blocks.
"""

import cv2
import numpy as np
import torch

from .mask_ops import mask_blur


_WINDOW_SIZES = (3, 5)
_CONSENSUS_MODES = ("median", "mean")
# Empty-mask threshold: a frame is considered "empty" if its max value is below
# this. Used by the dropout-recovery special case in auto_repair.
_EMPTY_MASK_EPS = 0.01


# =============================================================================
# Optical flow + warp helpers (CPU, numpy/cv2 path)
# =============================================================================

def _compute_backward_flow_farneback(curr_gray_u8, neighbor_gray_u8):
    """Backward flow from curr → neighbor, suitable for grid_sample/remap warp.

    Per project Farneback convention (status board D-103): `Farneback(dst, src)`
    returns backward flow from dst→src — for each pixel in `curr`, says where in
    `neighbor` the corresponding source pixel is. Use to warp the neighbor's
    mask INTO `curr`'s coordinate space (cv2.remap with x+dx, y+dy).

    Returns flow [H, W, 2] float32.
    """
    return cv2.calcOpticalFlowFarneback(
        curr_gray_u8, neighbor_gray_u8, None,
        pyr_scale=0.5, levels=3, winsize=15, iterations=3,
        poly_n=5, poly_sigma=1.2, flags=0,
    )


def _warp_mask_with_flow(mask_2d, flow):
    """Warp [H, W] mask using [H, W, 2] backward-flow field via cv2.remap.

    For each (y, x) in output, samples source mask at (x + flow_dx, y + flow_dy).
    Border: zero-fill (a mask region warped past frame boundary disappears
    rather than wrapping). Bilinear interpolation.
    """
    H, W = mask_2d.shape
    y_idx, x_idx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    map_x = (x_idx + flow[..., 0]).astype(np.float32)
    map_y = (y_idx + flow[..., 1]).astype(np.float32)
    return cv2.remap(
        mask_2d.astype(np.float32), map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
    )


def _binary_iou(a, b, threshold=0.5):
    """IoU of two soft masks after thresholding at 0.5. Both empty = 1.0."""
    ab = a > threshold
    bb = b > threshold
    intersection = int(np.logical_and(ab, bb).sum())
    union = int(np.logical_or(ab, bb).sum())
    if union == 0:
        return 1.0
    return intersection / union


# =============================================================================
# Node
# =============================================================================

class NV_TemporalMaskStabilizer_V2:
    """Motion-compensated short-window temporal mask stabilizer."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK", {
                    "tooltip": (
                        "Input mask sequence [B,H,W]. Should be the cleaned "
                        "WHOLE-SUBJECT mask (output of NV_MaskBinaryCleanup), not the "
                        "per-region SAM3 outputs. Stabilizing per-region masks "
                        "independently creates flicker at their abutments."
                    ),
                }),
                "image": ("IMAGE", {
                    "tooltip": (
                        "Source video [B,H,W,3] in [0,1]. Used by Farneback to compute "
                        "per-frame backward flow → warp neighbor masks into each frame's "
                        "coordinate space. Must match mask batch + spatial dims."
                    ),
                }),
                "window_size": ([3, 5], {
                    "default": 3,
                    "tooltip": (
                        "Number of frames in consensus window. 3 = (t-1, t, t+1) — "
                        "fastest, tightest consensus. 5 = (t-2..t+2) — smoother, more "
                        "robust to back-to-back failures, but ~2× slower and tends to "
                        "over-smooth fast motion. Start with 3."
                    ),
                }),
                "consensus_mode": (list(_CONSENSUS_MODES), {
                    "default": "median",
                    "tooltip": (
                        "How to combine the warped stack. 'median' = robust to "
                        "single-frame outliers (RECOMMENDED — implicitly handles lost "
                        "frames, anomalies). 'mean' = smoother but vulnerable to bad "
                        "frames pulling the average."
                    ),
                }),
                "binarize_first": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Threshold input to binary at 0.5 before stabilizing. Set TRUE "
                        "if input is soft (e.g. post-guided-filter); set FALSE if input "
                        "is already binary (e.g. post-NV_MaskBinaryCleanup with no final "
                        "feather). Binarizing first eliminates soft-edge flicker but "
                        "destroys anti-aliasing."
                    ),
                }),
            },
            "optional": {
                "auto_repair": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Detect anomalous frames (low IoU vs all warped neighbors) and "
                        "replace with median of warped neighbors only (skip current "
                        "frame's contribution). Catches single-frame SAM3 failures, "
                        "empty-mask frames, identity jumps. Recommended ON."
                    ),
                }),
                "anomaly_iou_threshold": ("FLOAT", {
                    "default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": (
                        "Frame is flagged anomalous when IoU(current, neighbor_consensus) "
                        "< threshold. Lower = stricter (fewer repairs, less risk of over-"
                        "smoothing valid fast motion). 0.35 = balanced default (walking "
                        "subjects, mixed motion). 0.50-0.70 for static subjects only. "
                        "0.15-0.30 for fast action shots. 0.0 = off (always trust current "
                        "frame; empty-current dropout recovery still fires regardless)."
                    ),
                }),
                "final_feather_px": ("INT", {
                    "default": 0, "min": 0, "max": 32, "step": 1,
                    "tooltip": (
                        "Optional Gaussian softening on the stabilized output. 0 = off "
                        "(preserves whatever softness median consensus produced). 2-4 = "
                        "subtle smoothing for downstream blend masks."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("MASK", "MASK", "STRING")
    RETURN_NAMES = ("stabilized_mask", "repaired_flags", "info")
    FUNCTION = "stabilize"
    CATEGORY = "NV_Utils/Mask"
    DESCRIPTION = (
        "Motion-compensated short-window temporal mask stabilizer. Removes per-frame "
        "jitter, fills lost frames, repairs anomalies via median consensus over "
        "Farneback-warped neighbors. Designed to follow NV_MaskBinaryCleanup in the "
        "canonical chain. V2 of the shelved 2026-04-24 stabilizer — small window, "
        "motion-compensated, memory-bounded, no frame-0 seed."
    )

    def stabilize(
        self,
        mask, image,
        window_size, consensus_mode,
        binarize_first,
        auto_repair=True,
        anomaly_iou_threshold=0.35,
        final_feather_px=0,
    ):
        # --- Validate inputs -------------------------------------------------
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        elif mask.ndim != 3:
            raise ValueError(
                f"[NV_TemporalMaskStabilizer_V2] mask must be 2D [H,W] or 3D [B,H,W], "
                f"got shape {tuple(mask.shape)}."
            )
        if image.ndim != 4 or image.shape[-1] != 3:
            raise ValueError(
                f"[NV_TemporalMaskStabilizer_V2] image must be IMAGE [B,H,W,3], "
                f"got shape {tuple(image.shape)}."
            )
        if mask.shape[0] != image.shape[0]:
            raise ValueError(
                f"[NV_TemporalMaskStabilizer_V2] mask batch={mask.shape[0]} must match "
                f"image batch={image.shape[0]}."
            )
        if mask.shape[1] != image.shape[1] or mask.shape[2] != image.shape[2]:
            raise ValueError(
                f"[NV_TemporalMaskStabilizer_V2] mask spatial dims {tuple(mask.shape[1:])} "
                f"must match image spatial dims {tuple(image.shape[1:3])}."
            )

        N, H, W = mask.shape
        device = mask.device

        # Single/zero-frame: pass through (need ≥2 for any temporal consensus).
        if N < 2:
            info = (
                f"[NV_TemporalMaskStabilizer_V2] N={N} frames; pass-through "
                f"(need ≥2 for temporal consensus)."
            )
            empty_flags = torch.zeros((N, H, W), dtype=torch.float32, device=device)
            return (mask.to(torch.float32), empty_flags, info)

        # --- Sanitize, optionally binarize ----------------------------------
        mask_clean = torch.nan_to_num(mask, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        if binarize_first:
            mask_clean = (mask_clean > 0.5).to(torch.float32)

        # --- Convert IMAGE to grayscale uint8 (numpy) — Farneback's input ----
        # Luminance: Y = 0.299*R + 0.587*G + 0.114*B per ITU-R BT.601.
        image_clean = image.clamp(0.0, 1.0)
        gray = (
            image_clean[..., 0] * 0.299
            + image_clean[..., 1] * 0.587
            + image_clean[..., 2] * 0.114
        ) * 255.0
        gray_np = gray.to(torch.uint8).cpu().numpy()  # [N, H, W] uint8
        # Free intermediates BEFORE the per-frame loop allocates more numpy buffers.
        del gray, image_clean
        # No redundant .astype(np.float32): mask_clean is already fp32, .cpu().numpy()
        # produces a fp32 numpy view (no copy).
        mask_np = mask_clean.cpu().numpy()  # [N, H, W] fp32
        del mask_clean

        # --- Per-frame stabilize (streaming, no flow precompute) -------------
        radius = window_size // 2  # 3→1, 5→2
        output_np = np.zeros_like(mask_np)
        # Per-frame repair flag for debug viz (1.0 = repaired, 0.0 = trusted).
        repaired_flags_np = np.zeros((N,), dtype=np.float32)
        n_warps_computed = 0
        n_repaired = 0
        n_dropout_recovered = 0
        n_passthrough = 0

        for t in range(N):
            # Collect window: warped neighbors only (current frame handled separately).
            current_mask = mask_np[t]
            warped_neighbors = []

            for offset in range(-radius, radius + 1):
                if offset == 0:
                    continue
                tn = t + offset
                if tn < 0 or tn >= N:
                    continue

                flow = _compute_backward_flow_farneback(gray_np[t], gray_np[tn])
                warped = _warp_mask_with_flow(mask_np[tn], flow)
                warped_neighbors.append(warped)
                n_warps_computed += 1

            # No neighbors available — pass current through.
            if len(warped_neighbors) == 0:
                output_np[t] = current_mask
                n_passthrough += 1
                continue

            # Compute neighbor consensus (used for both anomaly check + repair).
            neighbor_stack = np.stack(warped_neighbors, axis=0)
            if consensus_mode == "median":
                neighbor_consensus = np.median(neighbor_stack, axis=0)
            else:
                neighbor_consensus = np.mean(neighbor_stack, axis=0)

            # --- Anomaly detection (consensus-based, with dropout recovery) ---
            do_repair = False
            current_is_empty = float(current_mask.max()) < _EMPTY_MASK_EPS
            if auto_repair:
                # Special case: current frame is empty but at least one neighbor
                # carries signal. Codex Bug #2 — max(individual IoU) was wrongly
                # certifying empty-current as good when any neighbor was also
                # empty. Explicit dropout recovery beats threshold tuning here.
                any_neighbor_nonempty = any(
                    float(wn.max()) >= _EMPTY_MASK_EPS for wn in warped_neighbors
                )
                if current_is_empty and any_neighbor_nonempty:
                    do_repair = True
                    n_dropout_recovered += 1
                elif anomaly_iou_threshold > 0.0:
                    # Score current vs the neighbor CONSENSUS (not max-of-individuals),
                    # so a single matching-but-bad neighbor can't certify a bad frame.
                    iou = _binary_iou(current_mask, neighbor_consensus)
                    if iou < float(anomaly_iou_threshold):
                        do_repair = True
                        n_repaired += 1

            if do_repair:
                # Use precomputed neighbor consensus (drop current frame's contribution).
                output_np[t] = neighbor_consensus
                repaired_flags_np[t] = 1.0
            else:
                # Combine current + neighbors via the same consensus rule.
                stack = np.stack([current_mask] + warped_neighbors, axis=0)
                if consensus_mode == "median":
                    output_np[t] = np.median(stack, axis=0)
                else:
                    output_np[t] = np.mean(stack, axis=0)

            # Re-binarize if user asked for binary I/O — linear-interp warp +
            # median both reintroduce fractional values, which violate intent.
            if binarize_first:
                output_np[t] = (output_np[t] > 0.5).astype(np.float32)

        # --- Back to torch + optional final feather --------------------------
        result = torch.from_numpy(output_np).to(device=device, dtype=torch.float32)
        if final_feather_px > 0:
            result = mask_blur(result, int(final_feather_px))
        result = result.clamp(0.0, 1.0)

        # Per-frame repair flags as a [N, H, W] MASK for downstream viz/debug
        # (1.0 fills the whole frame on repaired frames; 0.0 elsewhere). Wire to
        # NV_PreviewAnimation or VHS_VideoCombine to flash on repaired frames.
        repaired_flags = torch.from_numpy(repaired_flags_np).to(
            device=device, dtype=torch.float32
        )
        repaired_flags = repaired_flags.view(N, 1, 1).expand(N, H, W).contiguous()

        info = (
            f"[NV_TemporalMaskStabilizer_V2] N={N}, window={window_size}, "
            f"consensus={consensus_mode}, binarize_first={binarize_first}, "
            f"auto_repair={auto_repair} (threshold={anomaly_iou_threshold:.2f}), "
            f"warps_computed={n_warps_computed}, "
            f"repaired_anomaly={n_repaired}, dropout_recovered={n_dropout_recovered}, "
            f"passthrough_boundary={n_passthrough}, "
            f"final_feather_px={final_feather_px}."
        )
        return (result, repaired_flags, info)


NODE_CLASS_MAPPINGS = {
    "NV_TemporalMaskStabilizer_V2": NV_TemporalMaskStabilizer_V2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_TemporalMaskStabilizer_V2": "NV Temporal Mask Stabilizer V2",
}
