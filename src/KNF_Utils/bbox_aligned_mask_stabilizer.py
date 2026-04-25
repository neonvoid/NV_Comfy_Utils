"""
NV BboxAligned Mask Stabilizer — Track B of the D-098 ownership-principle fix.

Stabilizes a per-frame SAM3 silhouette mask by motion-compensating consecutive
frames using the temporally stable bbox center trajectory from
NV_OptimizeCropTrajectory, then fusing aligned neighbors via a streaming
weighted-mean accumulator (or optional median).

Architecture (after 3-round multi-AI debate, 2026-04-24):

  motion model:    bbox-center translation only
                   - scale alignment is no-op under NV_OptimizeCropTrajectory's
                     default size_mode=lock_largest (per-frame extents constant),
                     and harmful under unlocked sizing because bbox-extent
                     fluctuation is annotation noise, not camera motion
                   - flow rejected: aperture problem on uniform mask interiors

  consensus:       distance-weighted mean (default), median (opt-in)
                   - weighted-mean encodes uncertainty as soft transition,
                     which all downstream consumers in this pipeline want
                     (NV_FaceHarmonizePyramid, NV_FrameTemporalStabilize,
                     InpaintCrop2, InpaintStitch2 — none binarize the master
                     mask before further processing; VACE dilates 80-100px
                     before VAE encoding so any soft band is killed upstream)
                   - median preserves morphological boundaries; offered as
                     escape hatch if downstream binarization ever appears in
                     the pipeline

  window:          fixed ±N (default 2 = 5-frame consensus) with dynamic
                   valid-count shrinking — boundary frames and trajectory
                   gaps just have fewer survivors

  neighbor reject: center-jump magnitude only (no IoU magic numbers).
                   Skips neighbors whose bbox-center delta exceeds
                   `max_center_jump_frac × min(H, W)`.

  failure modes:   - bbox missing for ≤3 frames → linear-interpolate centers
                   - longer bbox gaps → raw-mask passthrough on those frames
                   - subject enter/exit → window naturally shrinks via
                     valid-count, no zero-padding pollution

  output:          soft mask (default); binary as opt-in. Aux quality scalar
                   tensor opt-in for production debugging.

  performance:     streaming weighted-mean accumulator — O(H×W) memory, no
                   T×K×H×W stack materialization. Median path uses small
                   per-frame stack (≤5 × H × W ≤ ~50 MB at 1080p).

Key references in this repo:
  - bbox_ops.extract_bboxes — extract per-frame bbox extents from a MASK
  - bbox_ops.forward_backward_fill — pattern for gap-filling per-frame data
  - low_freq_recompose / face_harmonize_pyramid — sibling post-decode nodes
"""

import torch
import torch.nn.functional as F

from .bbox_ops import extract_bboxes


LOG_PREFIX = "[NV_BboxAlignedMaskStabilizer]"

# Fixed weighted-mean weights table for offsets [0, 1, 2, 3, 4] from current.
# Mirrored on negative side via abs(dt). Values >= 5 unused in practice (window
# capped at 4 by the schema).
_WEIGHTS_TABLE = [0.50, 0.30, 0.20, 0.12, 0.07]


# =============================================================================
# Center extraction + interpolation
# =============================================================================

def _extract_centers(bbox_mask):
    """Per-frame (cx, cy) from a bbox MASK. Returns list[Optional[(cx, cy)]]."""
    x1s, y1s, x2s, y2s, present = extract_bboxes(bbox_mask, info_lines=None)
    centers = []
    for x1, y1, x2, y2, p in zip(x1s, y1s, x2s, y2s, present):
        if not p:
            centers.append(None)
        else:
            centers.append((0.5 * (x1 + x2), 0.5 * (y1 + y2)))
    return centers


def _interpolate_short_gaps(centers, max_gap):
    """Linearly interpolate (cx, cy) across gaps of length <= max_gap.

    Gaps that are too long, or that touch the clip start/end without anchors
    on both sides, are left as None — those frames will fall back to raw-mask
    passthrough downstream.
    """
    if max_gap <= 0:
        return list(centers)

    interpolated = list(centers)
    T = len(interpolated)

    i = 0
    while i < T:
        if interpolated[i] is not None:
            i += 1
            continue
        # Find span of consecutive None entries
        gap_start = i
        while i < T and interpolated[i] is None:
            i += 1
        gap_end = i  # exclusive
        gap_len = gap_end - gap_start

        if gap_len > max_gap:
            continue
        left = interpolated[gap_start - 1] if gap_start > 0 else None
        right = interpolated[gap_end] if gap_end < T else None
        if left is None or right is None:
            continue  # boundary-touching gap — leave as None
        for k, idx in enumerate(range(gap_start, gap_end)):
            alpha = (k + 1) / (gap_len + 1)
            cx = left[0] * (1.0 - alpha) + right[0] * alpha
            cy = left[1] * (1.0 - alpha) + right[1] * alpha
            interpolated[idx] = (cx, cy)
    return interpolated


# =============================================================================
# Sub-pixel mask translation via grid_sample
# =============================================================================

def _translate_mask(mask_HW, delta_x, delta_y, kernel_grid_xs, kernel_grid_ys):
    """Shift mask by (delta_x, delta_y) so that out[x, y] = mask[x - dx, y - dy].

    Pre-built coordinate grids (`kernel_grid_xs`, `kernel_grid_ys`) are passed
    in to avoid re-allocating the meshgrid every call.

    Padding mode "zeros" — when a sample falls outside the mask bounds, it
    reads 0 (no subject). reflection or border would leak subject content
    from the silhouette edges into the padded region, which is exactly the
    failure mode we are trying to avoid.
    """
    H, W = mask_HW.shape
    sample_x = kernel_grid_xs - delta_x
    sample_y = kernel_grid_ys - delta_y
    norm_x = (sample_x / max(W - 1, 1)) * 2.0 - 1.0
    norm_y = (sample_y / max(H - 1, 1)) * 2.0 - 1.0
    grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0)  # [1, H, W, 2]
    mask_bchw = mask_HW.unsqueeze(0).unsqueeze(0)
    warped = F.grid_sample(
        mask_bchw, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    return warped.squeeze(0).squeeze(0)


# =============================================================================
# Per-frame consensus
# =============================================================================

def _frame_consensus_weighted_mean(
    raw_mask, t, valid_dts, deltas_xy, kernel_grid_xs, kernel_grid_ys
):
    """Streaming weighted-mean accumulator over valid neighbors.

    Returns (out_HW, weight_sum, valid_count). If weight_sum == 0, caller
    should fall back to raw_mask[t].
    """
    H, W = raw_mask.shape[1], raw_mask.shape[2]
    accumulator = torch.zeros((H, W), device=raw_mask.device, dtype=raw_mask.dtype)
    weight_sum = 0.0
    for dt, (dx, dy) in zip(valid_dts, deltas_xy):
        w = _WEIGHTS_TABLE[min(abs(dt), len(_WEIGHTS_TABLE) - 1)]
        if dt == 0:
            aligned = raw_mask[t]
        else:
            aligned = _translate_mask(raw_mask[t + dt], dx, dy, kernel_grid_xs, kernel_grid_ys)
        accumulator.add_(aligned, alpha=w)
        weight_sum += w
    return accumulator, weight_sum, len(valid_dts)


def _frame_consensus_median(
    raw_mask, t, valid_dts, deltas_xy, kernel_grid_xs, kernel_grid_ys
):
    """Median consensus over valid neighbors. Builds a small per-frame stack
    (max ~5 entries × H × W) and calls torch.median(dim=0).

    Important: torch.median on an EVEN-sized dimension returns the LOWER of
    the two middle values, not the mean of them. For soft masks in [0, 1]
    this acts like a logical-AND / morphological erosion, aggressively
    shrinking the mask. We special-case k=2 to use the mean explicitly,
    which matches the architecture's promise of "graceful degradation to
    mean at low valid-count" (multi-AI review caught this 2026-04-24).

    At k=1 median is identity. At k=3+ median works as expected.
    """
    if not valid_dts:
        return None, 0
    candidates = []
    for dt, (dx, dy) in zip(valid_dts, deltas_xy):
        if dt == 0:
            candidates.append(raw_mask[t])
        else:
            candidates.append(
                _translate_mask(raw_mask[t + dt], dx, dy, kernel_grid_xs, kernel_grid_ys)
            )
    stack = torch.stack(candidates, dim=0)  # [K, H, W]
    if stack.shape[0] == 2:
        # torch.median picks lower-of-two; explicit mean is what we want
        return stack.mean(dim=0), 2
    return torch.median(stack, dim=0).values, len(candidates)


# =============================================================================
# Node
# =============================================================================

class NV_BboxAlignedMaskStabilizer:
    """Stabilize a per-frame mask by motion-compensating neighbors using the
    smoothed bbox-center trajectory, then fusing via weighted-mean (default)
    or median.

    Pipeline placement (the master mask fan-out):

        SAM3 → raw_mask
        NV_PointDrivenBBox → CoTracker3 bbox
        NV_OptimizeCropTrajectory(bbox) → bbox_optimized (stable signal)
        NV_BboxAlignedMaskStabilizer(raw_mask, bbox_optimized) → MASTER_MASK
                                ┌────────────────────────────┐
              MASTER_MASK ──────┤── InpaintCrop2.mask
                                ├── VaceControlVideoPrep.mask
                                ├── NV_FaceHarmonizePyramid.mask
                                ├── NV_FrameTemporalStabilize.mask
                                └── (InpaintStitch2 reads via stitcher)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "raw_mask": ("MASK", {
                    "tooltip": "Per-frame SAM3 silhouette [T, H, W]. The unstable signal — "
                               "this is what we're stabilizing."
                }),
                "bbox_optimized": ("MASK", {
                    "tooltip": "Smoothed bbox MASK from NV_OptimizeCropTrajectory [T, H, W]. "
                               "Provides the stable per-frame motion signal (bbox center)."
                }),
                "temporal_window": ("INT", {
                    "default": 2, "min": 1, "max": 4, "step": 1,
                    "tooltip": "± window size. 2 = 5-frame consensus (default). 1 = 3-frame "
                               "(faster, less smoothing). 3-4 = wider temporal smoothing but "
                               "slower and risks oversmooth on fast motion."
                }),
            },
            "optional": {
                "consensus": (["weighted_mean", "median"], {
                    "default": "weighted_mean",
                    "tooltip": "weighted_mean (default): streaming distance-weighted accumulator. "
                               "Soft transitions where mask oscillates between contours. Best for "
                               "downstream soft-mask consumers (this pipeline). "
                               "median: morphologically decisive boundaries. Opt-in when downstream "
                               "binarizes the mask. Slower (builds per-frame stack), but exact."
                }),
                "max_center_jump_frac": ("FLOAT", {
                    "default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Skip neighbor if bbox-center delta exceeds this fraction of "
                               "min(H, W). Prevents alignment when motion is so large that "
                               "translation is meaningless (scene cut, fast pan). 0.25 = 25%% of "
                               "the shorter side."
                }),
                "interpolate_gap": ("INT", {
                    "default": 3, "min": 0, "max": 20, "step": 1,
                    "tooltip": "Linearly interpolate bbox-center across gaps up to this length. "
                               "Longer gaps → raw-mask passthrough on those frames. 0 disables "
                               "interpolation entirely."
                }),
                "output_mode": (["soft", "binary"], {
                    "default": "soft",
                    "tooltip": "soft: passthrough float [0, 1] consensus output. "
                               "binary: threshold at 0.5 (loses gradient information; only "
                               "useful if a downstream consumer requires a hard mask)."
                }),
                "return_quality": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If True, embed per-frame valid-count quality stats in the info "
                               "string output. Useful for debugging which frames had degraded "
                               "consensus (boundaries, trajectory gaps, extreme motion)."
                }),
            },
        }

    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("master_mask", "info")
    FUNCTION = "execute"
    CATEGORY = "NV_Utils/Mask"
    DESCRIPTION = (
        "Stabilize a per-frame mask by motion-compensating neighbors via the "
        "smoothed bbox trajectory, then fusing via weighted-mean (default) or "
        "median. The 'master mask' upstream of all temporal-decision-making nodes."
    )

    def execute(self, raw_mask, bbox_optimized, temporal_window,
                consensus="weighted_mean",
                max_center_jump_frac=0.25,
                interpolate_gap=3,
                output_mode="soft",
                return_quality=False):
        info_lines = []

        # ── Param validation (defensive — schema clamps to 1-4 but bypassable) ─
        if not (1 <= int(temporal_window) <= 4):
            raise ValueError(
                f"temporal_window must be in [1, 4]; got {temporal_window}. "
                f"_WEIGHTS_TABLE only defines weights for offsets 0-4."
            )

        # ── Shape validation + 2D auto-promotion ────────────────────────────
        # ComfyUI MASK type can arrive as [H, W] for single-frame inputs.
        if raw_mask.dim() == 2:
            raw_mask = raw_mask.unsqueeze(0)
        if bbox_optimized.dim() == 2:
            bbox_optimized = bbox_optimized.unsqueeze(0)
        if raw_mask.dim() != 3 or bbox_optimized.dim() != 3:
            raise ValueError(
                f"raw_mask and bbox_optimized must be [T, H, W] (or [H, W] for single frame); "
                f"got {list(raw_mask.shape)} / {list(bbox_optimized.shape)}"
            )
        if raw_mask.shape != bbox_optimized.shape:
            raise ValueError(
                f"raw_mask {list(raw_mask.shape)} must match bbox_optimized "
                f"{list(bbox_optimized.shape)} in T, H, W"
            )
        T, H, W = raw_mask.shape

        # ── NaN / inf sanitization (production safety) ──────────────────────
        # Real pipelines occasionally produce NaN at mask boundaries from
        # upstream warps. One NaN poisons the consensus accumulator, so we
        # sanitize up front and report it.
        n_nonfinite_raw = int((~torch.isfinite(raw_mask)).sum().item())
        if n_nonfinite_raw > 0:
            info_lines.append(f"WARN: sanitized {n_nonfinite_raw} non-finite values in raw_mask")
            raw_mask = torch.nan_to_num(raw_mask, nan=0.0, posinf=1.0, neginf=0.0)

        # Single-frame fast path — nothing to stabilize against. Always emit
        # float32 [0, 1] regardless of input dtype (ComfyUI MASK convention).
        if T < 2:
            info_lines.append(f"passthrough (T={T})")
            info = f"{LOG_PREFIX} " + " | ".join(info_lines)
            print(info)
            return (raw_mask.clone().float().clamp(0.0, 1.0), info)

        device = raw_mask.device
        work_mask = raw_mask.float()  # accumulator math wants fp32 stability
        max_jump = float(max_center_jump_frac) * float(min(H, W))

        # ── Center extraction + gap interpolation ───────────────────────────
        # IMPORTANT: keep `original_centers` separate from the interpolated
        # list. We allow interpolated centers to define alignment for THIS
        # frame's coordinate space, but only ORIGINALLY-present frames get to
        # contribute their raw mask as a neighbor — interpolated frames have
        # an estimated center but a potentially-garbage raw_mask (SAM3 may
        # have failed for the same reason the bbox detector did). Multi-AI
        # review (2026-04-24) flagged self-pollution from contributing
        # garbage masks at dt=0 for interpolated frames.
        original_centers = _extract_centers(bbox_optimized)
        n_present = sum(1 for c in original_centers if c is not None)
        centers = _interpolate_short_gaps(list(original_centers), int(interpolate_gap))
        n_after_interp = sum(1 for c in centers if c is not None)
        n_interpolated = n_after_interp - n_present
        if n_after_interp < T:
            info_lines.append(
                f"bbox: {n_present}/{T} present, +{n_interpolated} interpolated, "
                f"{T - n_after_interp} gap-passthrough"
            )
        elif n_interpolated > 0:
            info_lines.append(f"bbox: {n_present}/{T} present, +{n_interpolated} interpolated")
        else:
            info_lines.append(f"bbox: {T}/{T} present")

        # ── Pre-build coordinate grids reused across all warp calls ─────────
        # Float32 indices to match the dtype we're operating in.
        ys_grid, xs_grid = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing="ij",
        )

        # ── Output buffer + diagnostics ─────────────────────────────────────
        out = torch.empty((T, H, W), device=device, dtype=torch.float32)
        valid_counts = [0] * T
        n_passthrough = 0

        # ── Per-frame consensus loop ────────────────────────────────────────
        for t in range(T):
            if centers[t] is None:
                # Anchor frame has no bbox → can't define alignment for THIS
                # frame's coordinate space. Pass raw mask through (clamped).
                out[t] = work_mask[t].clamp(0.0, 1.0)
                n_passthrough += 1
                continue

            cx_t, cy_t = centers[t]

            # Collect valid neighbors and their deltas. ONLY originally-
            # present frames (not interpolated) contribute their raw mask —
            # interpolated frames have an estimated center but their raw
            # mask might be garbage (SAM3 may have failed where bbox failed).
            valid_dts = []
            deltas_xy = []
            for dt in range(-temporal_window, temporal_window + 1):
                j = t + dt
                if j < 0 or j >= T:
                    continue
                # `centers[j]` may be interpolated (truthy) — we use the
                # original_centers check to gate raw-mask contribution.
                if original_centers[j] is None:
                    continue
                cx_j, cy_j = centers[j]
                dx = cx_t - cx_j
                dy = cy_t - cy_j
                if (dx * dx + dy * dy) ** 0.5 > max_jump:
                    continue
                valid_dts.append(dt)
                deltas_xy.append((dx, dy))

            if not valid_dts:
                out[t] = work_mask[t].clamp(0.0, 1.0)
                n_passthrough += 1
                continue

            if consensus == "median":
                stabilized, k = _frame_consensus_median(
                    work_mask, t, valid_dts, deltas_xy, xs_grid, ys_grid
                )
                if k == 0 or stabilized is None:
                    out[t] = work_mask[t].clamp(0.0, 1.0)
                    n_passthrough += 1
                else:
                    out[t] = stabilized.clamp(0.0, 1.0)
                    valid_counts[t] = k
            else:  # weighted_mean (default)
                accum, wsum, k = _frame_consensus_weighted_mean(
                    work_mask, t, valid_dts, deltas_xy, xs_grid, ys_grid
                )
                if wsum <= 0.0:
                    out[t] = work_mask[t].clamp(0.0, 1.0)
                    n_passthrough += 1
                else:
                    out[t] = (accum / wsum).clamp(0.0, 1.0)
                    valid_counts[t] = k

        # ── Diagnostics computed BEFORE binarization ───────────────────────
        # If we measured after binarization, a soft 0.49 -> 0.51 wiggle would
        # report as a 1.0 frame-to-frame delta — meaningless artifact.
        with torch.no_grad():
            raw_delta = (work_mask[1:] - work_mask[:-1]).abs().mean().item() * 255
            out_delta = (out[1:] - out[:-1]).abs().mean().item() * 255
            reduction = (
                (1.0 - out_delta / max(raw_delta, 1e-12)) * 100.0 if raw_delta > 0 else 0.0
            )

        # ── Output mode (after diagnostics) ─────────────────────────────────
        if output_mode == "binary":
            out = (out > 0.5).float()

        if n_passthrough > 0:
            info_lines.append(f"raw passthrough: {n_passthrough}/{T} frames")

        info_lines.append(
            f"frame-to-frame mask delta: raw={raw_delta:.2f}/255 -> "
            f"smoothed={out_delta:.2f}/255 (reduction {reduction:.0f}%)"
        )

        info_lines.append(
            f"T={T} window=+/-{temporal_window} consensus={consensus} "
            f"output={output_mode}"
        )

        if return_quality:
            full = sum(1 for k in valid_counts if k == 2 * temporal_window + 1)
            partial = sum(1 for k in valid_counts if 0 < k < 2 * temporal_window + 1)
            empty = sum(1 for k in valid_counts if k == 0)
            info_lines.append(
                f"quality: {full} full / {partial} partial / {empty} empty (passthrough)"
            )

        info = f"{LOG_PREFIX} " + " | ".join(info_lines)
        print(info)
        # Always emit float32 [0, 1] regardless of input dtype. ComfyUI MASK
        # convention is fp32; preserving an upstream uint8/bool dtype here
        # would quantize soft consensus values to 0/1 and destroy the entire
        # benefit of stabilization.
        return (out.float(), info)


# ── Registration ─────────────────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    "NV_BboxAlignedMaskStabilizer": NV_BboxAlignedMaskStabilizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_BboxAlignedMaskStabilizer": "NV BboxAligned Mask Stabilizer (master mask)",
}
