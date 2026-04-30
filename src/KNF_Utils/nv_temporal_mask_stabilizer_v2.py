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

import time

import cv2
import numpy as np
import scipy.ndimage
import torch

from .mask_ops import mask_blur


_WINDOW_SIZES = (3, 5)
_CONSENSUS_MODES = ("median", "mean")
_WARP_INTERPOLATIONS = ("nearest", "linear")
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


def _warp_mask_with_flow(mask_2d, flow, interpolation_mode="nearest"):
    """Warp [H, W] mask using [H, W, 2] backward-flow field via cv2.remap.

    For each (y, x) in output, samples source mask at (x + flow_dx, y + flow_dy).
    Border: zero-fill (a mask region warped past frame boundary disappears
    rather than wrapping).

    interpolation_mode:
      - "nearest": preserves binarity (binary in → binary out). No grey halo at
        boundaries. Cost: stairstep edges where the warp is not pixel-aligned.
        DEFAULT — multi-AI review flagged this as the right root-cause fix for
        the bilinear-warp grey halo bug.
      - "linear": bilinear interpolation. Smoother edges but reintroduces
        fractional values at binary boundaries (the original halo bug). Use
        only if downstream wants soft alpha output AND `binarize_output=False`.
    """
    if interpolation_mode == "linear":
        flag = cv2.INTER_LINEAR
    else:
        flag = cv2.INTER_NEAREST
    H, W = mask_2d.shape
    y_idx, x_idx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    map_x = (x_idx + flow[..., 0]).astype(np.float32)
    map_y = (y_idx + flow[..., 1]).astype(np.float32)
    return cv2.remap(
        mask_2d.astype(np.float32), map_x, map_y,
        interpolation=flag,
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
                    "default": True,
                    "tooltip": (
                        "INPUT-SIDE binarize. Threshold input to binary at 0.5 BEFORE "
                        "stabilizing. **DEFAULT TRUE** — the suite is tuned for crisp "
                        "binary output, so all internal stages (warp, median, IoU) run "
                        "on binary values end-to-end. No-op when input is already binary "
                        "(the canonical MaskUnion → MaskBinaryCleanup → Stabilizer chain). "
                        "Set FALSE only if you specifically want soft input → soft "
                        "intermediate computation (e.g. with `binarize_output=False` for "
                        "a downstream consumer that wants soft alpha)."
                    ),
                }),
            },
            "optional": {
                "warp_interpolation": (list(_WARP_INTERPOLATIONS), {
                    "default": "nearest",
                    "tooltip": (
                        "Interpolation mode for cv2.remap when warping neighbor masks "
                        "into the current frame's coords. 'nearest' (DEFAULT) preserves "
                        "binarity → no grey halo at source. 'linear' = bilinear, smoother "
                        "edges but reintroduces fractional values at binary boundaries "
                        "(the original halo bug). Use 'linear' only if you specifically "
                        "want soft alpha output AND `binarize_output=False`."
                    ),
                }),
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
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": (
                        "Frame is flagged anomalous when IoU(current, neighbor_consensus) "
                        "< threshold. **DEFAULT 0.0 = OFF** — multi-AI review flagged that "
                        "full-frame replacement on IoU-anomaly tends to over-fire on legitimate "
                        "fast motion (head turns, swinging arms, hair motion), creating new "
                        "smearing artifacts where good frames get replaced with blurry warped "
                        "consensus. Empty-current dropout recovery still fires regardless of "
                        "this threshold. Opt back in for shots with known SAM3 dropouts: "
                        "0.20-0.30 for fast action, 0.35-0.45 for walking subjects, 0.50-0.70 "
                        "for static-camera talking heads."
                    ),
                }),
                "binarize_output": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "OUTPUT-SIDE binarize. Threshold the stabilized result at "
                        "`binarize_output_threshold` AFTER consensus. Default ON because "
                        "bilinear warp (cv2.remap) + median consensus reintroduce fractional "
                        "values at mask boundaries even on binary inputs — produces a visible "
                        "grey halo around the subject. Turn OFF only if you specifically want "
                        "soft alpha output (downstream blend feathering with non-binary edges)."
                    ),
                }),
                "binarize_output_threshold": ("FLOAT", {
                    "default": 0.5, "min": 0.01, "max": 0.99, "step": 0.01,
                    "tooltip": (
                        "Threshold for binarize_output. 0.5 = standard. Lower values "
                        "(0.3) preserve more of the soft-consensus halo as foreground; "
                        "higher values (0.7) tighten the silhouette but may cut into the "
                        "subject. Only used when binarize_output=True."
                    ),
                }),
                "post_fill_holes": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "True binary fill of internal holes (scipy.ndimage.binary_fill_holes) "
                        "AFTER consensus + binarize_output. Default ON because median "
                        "consensus + bilinear warp artifacts can punch micro-holes inside "
                        "the subject silhouette: median([1.0_current, 0.0_warped_n1, "
                        "0.0_warped_n2]) = 0 even when current was correctly foreground. "
                        "This step closes any-sized internal holes regardless of kernel "
                        "(unlike greyscale closing). Different from `final_feather_px` — "
                        "fill_holes runs first, feather optionally softens after."
                    ),
                }),
                "final_feather_px": ("INT", {
                    "default": 0, "min": 0, "max": 32, "step": 1,
                    "tooltip": (
                        "Optional Gaussian softening on the stabilized output. 0 = off. "
                        "Runs AFTER binarize_output, so set this > 0 + binarize_output ON "
                        "to get a clean binary silhouette with a soft feathered edge for "
                        "downstream blend compositing. 2-4 = subtle. 8+ = heavy falloff."
                    ),
                }),
                "verbose_debug": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Print structured diagnostics to the ComfyUI console: per-call "
                        "config, per-frame repair status (flagged/dropout/trusted), IoU "
                        "values, timing breakdown (flow vs warp+consensus). Off by default "
                        "to avoid log spam. Turn on while tuning anomaly_iou_threshold + "
                        "comparing repair counts against actual shot motion."
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
        "Farneback-warped neighbors. **Defaults tuned for crisp binary output** "
        "(nearest warp + binarize_first + binarize_output + post_fill_holes all ON; "
        "IoU-anomaly-repair OFF to avoid smearing legitimate fast motion). Designed "
        "to follow NV_MaskBinaryCleanup in the canonical chain. V2 of the shelved "
        "2026-04-24 stabilizer — small window, motion-compensated, memory-bounded, "
        "no frame-0 seed."
    )

    def stabilize(
        self,
        mask, image,
        window_size, consensus_mode,
        binarize_first,
        warp_interpolation="nearest",
        auto_repair=True,
        anomaly_iou_threshold=0.0,
        binarize_output=True,
        binarize_output_threshold=0.5,
        post_fill_holes=True,
        final_feather_px=0,
        verbose_debug=False,
    ):
        # Wall-clock for the full call. Cheap; runs whether or not verbose_debug.
        t_start = time.perf_counter()
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

        if verbose_debug:
            # Single .min/.max/.mean call set; uses CPU-GPU sync but only when opted in.
            mn, mx, mean = float(mask.min()), float(mask.max()), float(mask.mean())
            print(f"[NV_TemporalMaskStabilizer_V2] === call ===")
            print(f"[NV_TemporalMaskStabilizer_V2]   input mask: shape=({N},{H},{W}), "
                  f"range=[{mn:.3f},{mx:.3f}], mean={mean:.4f}, dtype={mask.dtype}, device={device}")
            print(f"[NV_TemporalMaskStabilizer_V2]   input image: shape={tuple(image.shape)}, "
                  f"dtype={image.dtype}, device={image.device}")
            print(f"[NV_TemporalMaskStabilizer_V2]   config: window={window_size}, "
                  f"consensus={consensus_mode}, warp={warp_interpolation}, "
                  f"binarize_first={binarize_first}, "
                  f"auto_repair={auto_repair} (threshold={anomaly_iou_threshold:.2f}, "
                  f"{'IoU-anomaly OFF' if anomaly_iou_threshold <= 0 else 'IoU-anomaly ON'}), "
                  f"binarize_output={binarize_output} "
                  f"(threshold={binarize_output_threshold:.2f}), "
                  f"post_fill_holes={post_fill_holes}, "
                  f"final_feather_px={final_feather_px}")

        # Single/zero-frame: pass through (need ≥2 for any temporal consensus).
        if N < 2:
            info = (
                f"[NV_TemporalMaskStabilizer_V2] N={N} frames; pass-through "
                f"(need ≥2 for temporal consensus)."
            )
            if verbose_debug:
                print(f"[NV_TemporalMaskStabilizer_V2]   {info}")
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
        # Verbose-mode timing accumulators (zero overhead when verbose_debug=False)
        t_flow = 0.0
        t_consensus = 0.0

        if verbose_debug:
            mask_mb = mask_np.nbytes / (1024 * 1024)
            gray_mb = gray_np.nbytes / (1024 * 1024)
            print(f"[NV_TemporalMaskStabilizer_V2]   memory: mask_np={mask_mb:.1f} MB "
                  f"(fp32), gray_np={gray_mb:.1f} MB (uint8), "
                  f"output_np={mask_mb:.1f} MB. Peak ~{(mask_mb*2 + gray_mb):.1f} MB.")
            print(f"[NV_TemporalMaskStabilizer_V2]   per-frame log "
                  f"(repaired frames + every 50th progress tick):")

        for t in range(N):
            # Collect window: warped neighbors only (current frame handled separately).
            current_mask = mask_np[t]
            warped_neighbors = []

            t_flow_start = time.perf_counter() if verbose_debug else 0.0
            for offset in range(-radius, radius + 1):
                if offset == 0:
                    continue
                tn = t + offset
                if tn < 0 or tn >= N:
                    continue

                flow = _compute_backward_flow_farneback(gray_np[t], gray_np[tn])
                warped = _warp_mask_with_flow(mask_np[tn], flow, interpolation_mode=warp_interpolation)
                warped_neighbors.append(warped)
                n_warps_computed += 1
            if verbose_debug:
                t_flow += time.perf_counter() - t_flow_start

            # No neighbors available — pass current through.
            if len(warped_neighbors) == 0:
                output_np[t] = current_mask
                n_passthrough += 1
                if verbose_debug:
                    print(f"[NV_TemporalMaskStabilizer_V2]     [t={t:>4}/{N-1}] "
                          f"PASSTHROUGH (no neighbors in window)")
                continue

            t_cons_start = time.perf_counter() if verbose_debug else 0.0
            # Compute neighbor consensus (used for both anomaly check + repair).
            neighbor_stack = np.stack(warped_neighbors, axis=0)
            if consensus_mode == "median":
                neighbor_consensus = np.median(neighbor_stack, axis=0)
            else:
                neighbor_consensus = np.mean(neighbor_stack, axis=0)

            # --- Anomaly detection (consensus-based, with dropout recovery) ---
            do_repair = False
            repair_reason = None  # for verbose log
            iou_used = None       # for verbose log
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
                    repair_reason = "DROPOUT"
                elif anomaly_iou_threshold > 0.0:
                    # Score current vs the neighbor CONSENSUS (not max-of-individuals),
                    # so a single matching-but-bad neighbor can't certify a bad frame.
                    iou = _binary_iou(current_mask, neighbor_consensus)
                    iou_used = iou
                    if iou < float(anomaly_iou_threshold):
                        do_repair = True
                        n_repaired += 1
                        repair_reason = "ANOMALY"

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

            # NOTE: previous code re-binarized here using `binarize_first` with hardcoded
            # 0.5. Removed — output-side binarization is now `binarize_output` (with its
            # own threshold) applied AFTER the loop. Multi-AI review caught the redundancy.

            if verbose_debug:
                t_consensus += time.perf_counter() - t_cons_start
                if do_repair:
                    n_nonempty = sum(
                        1 for wn in warped_neighbors if float(wn.max()) >= _EMPTY_MASK_EPS
                    )
                    if repair_reason == "DROPOUT":
                        print(f"[NV_TemporalMaskStabilizer_V2]     [t={t:>4}/{N-1}] "
                              f"REPAIRED (dropout) — current empty, "
                              f"{n_nonempty}/{len(warped_neighbors)} neighbors carry signal")
                    else:
                        print(f"[NV_TemporalMaskStabilizer_V2]     [t={t:>4}/{N-1}] "
                              f"REPAIRED (anomaly) — IoU={iou_used:.3f} "
                              f"< threshold={anomaly_iou_threshold:.2f}")
                elif (t + 1) % 50 == 0 or t == N - 1:
                    elapsed = time.perf_counter() - t_start
                    rate = (t + 1) / max(elapsed, 1e-3)
                    eta = (N - t - 1) / max(rate, 1e-3)
                    print(f"[NV_TemporalMaskStabilizer_V2]     progress: {t+1}/{N} "
                          f"({rate:.1f} fps, elapsed {elapsed:.1f}s, ETA {eta:.1f}s)")

        # --- Output-side binarize (fixes the bilinear-warp-grey-halo issue) ---
        # Bilinear cv2.remap + median consensus reintroduce fractional values at
        # mask boundaries even on binary input. Without this step the output is
        # a soft mask with a visible grey halo on the subject. Default ON.
        if binarize_output:
            output_np = (output_np > float(binarize_output_threshold)).astype(np.float32)

        # --- Post-fill internal holes from median + warp artifacts ----------
        # Median over warped neighbors can punch micro-holes inside the subject
        # silhouette when Farneback flow misaligns on textureless interior
        # regions: median([1.0_current, 0.0_warped_n1, 0.0_warped_n2]) = 0.
        # True binary fill (scipy) closes any-sized holes regardless of kernel.
        # Per-frame loop (scipy is CPU-only). Skip if not binarized — fill_holes
        # is a binary operation; thresholds soft input internally at non-zero.
        if post_fill_holes and binarize_output:
            t_fill_start = time.perf_counter() if verbose_debug else 0.0
            n_holes_filled = 0
            for t in range(N):
                frame_u8 = output_np[t].astype(np.uint8)
                filled = scipy.ndimage.binary_fill_holes(frame_u8).astype(np.float32)
                if verbose_debug:
                    delta = int((filled > 0.5).sum() - (frame_u8 > 0).sum())
                    if delta > 0:
                        n_holes_filled += 1
                output_np[t] = filled
            if verbose_debug:
                t_fill = time.perf_counter() - t_fill_start
                print(f"[NV_TemporalMaskStabilizer_V2]   post_fill_holes: "
                      f"frames_with_holes_filled={n_holes_filled}/{N}, "
                      f"time={t_fill:.2f}s")
        elif post_fill_holes and not binarize_output and verbose_debug:
            print(f"[NV_TemporalMaskStabilizer_V2]   post_fill_holes SKIPPED "
                  f"(requires binarize_output=True; soft alpha hole-fill is undefined)")

        # --- Back to torch + optional final feather --------------------------
        # final_feather AFTER binarize_output: clean binary silhouette gets a
        # consistent feathered edge for downstream blend compositing.
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

        total_elapsed = time.perf_counter() - t_start
        info = (
            f"[NV_TemporalMaskStabilizer_V2] N={N}, window={window_size}, "
            f"consensus={consensus_mode}, warp={warp_interpolation}, "
            f"binarize_first={binarize_first}, "
            f"binarize_output={binarize_output} (thr={binarize_output_threshold:.2f}), "
            f"post_fill_holes={post_fill_holes}, "
            f"auto_repair={auto_repair} (threshold={anomaly_iou_threshold:.2f}), "
            f"warps_computed={n_warps_computed}, "
            f"repaired_anomaly={n_repaired}, dropout_recovered={n_dropout_recovered}, "
            f"passthrough_boundary={n_passthrough}, "
            f"final_feather_px={final_feather_px}, "
            f"total_time={total_elapsed:.1f}s."
        )
        if verbose_debug:
            other = max(0.0, total_elapsed - t_flow - t_consensus)
            print(f"[NV_TemporalMaskStabilizer_V2]   summary:")
            print(f"[NV_TemporalMaskStabilizer_V2]     warps_computed={n_warps_computed}, "
                  f"repaired_anomaly={n_repaired}, dropout_recovered={n_dropout_recovered}, "
                  f"passthrough_boundary={n_passthrough}")
            n_trusted = N - n_repaired - n_dropout_recovered - n_passthrough
            print(f"[NV_TemporalMaskStabilizer_V2]     frame disposition: "
                  f"trusted={n_trusted}/{N} ({100.0 * n_trusted / N:.1f}%), "
                  f"repaired={n_repaired + n_dropout_recovered}/{N} "
                  f"({100.0 * (n_repaired + n_dropout_recovered) / N:.1f}%)")
            print(f"[NV_TemporalMaskStabilizer_V2]     timing: "
                  f"flow+warp={t_flow:.1f}s ({100.0 * t_flow / max(total_elapsed, 1e-3):.0f}%), "
                  f"consensus={t_consensus:.1f}s ({100.0 * t_consensus / max(total_elapsed, 1e-3):.0f}%), "
                  f"other={other:.1f}s, total={total_elapsed:.1f}s")
            print(f"[NV_TemporalMaskStabilizer_V2] === done ===")
        return (result, repaired_flags, info)


NODE_CLASS_MAPPINGS = {
    "NV_TemporalMaskStabilizer_V2": NV_TemporalMaskStabilizer_V2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_TemporalMaskStabilizer_V2": "NV Temporal Mask Stabilizer V2",
}
