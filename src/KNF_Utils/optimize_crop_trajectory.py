"""
NV Optimize Crop Trajectory — solve for the smoothest crop bbox path that
keeps the subject inside the crop with margin.

Root cause this node attacks:
  Both NV_MaskTrackingBBox (mask centroid + one_euro) and NV_PointDrivenBBox
  (CoTracker3 feature) faithfully track ~2 px/frame of real subject motion
  for a walking person. That faithful tracking IS the visible jitter — the
  VACE inpaint region bobs with the natural head-bob, and its boundary walks
  ~2 px/frame in world frame. The fix has to break the 1:1 crop↔subject
  motion coupling.

Solution: formulate the crop center trajectory as a constrained smoothing
problem. Solve

    minimize  λ_p · Σ‖s - c‖²
            + λ_v · Σ‖Δs‖²        (1st diff = velocity)
            + λ_a · Σ‖Δ²s‖²       (2nd diff = acceleration / bobbing)
            + λ_j · Σ‖Δ³s‖²       (3rd diff = jerk; favours constant-velocity segments)

    subject to: per-frame containment bounds
                  (subject mask + margin must fit inside the crop window)
                frame bounds
                  (crop must stay inside the image)

Where:
  s = smoothed crop centers (decision variable, length N)
  c = raw input bbox centers (data, from upstream bbox node)
  λ_* = penalty weights

Solved as two independent 1D convex QPs (axis-separable). Y typically gets
heavier weights than X — vertical head-bob is the perceptually painful
artifact for walking shots; slow horizontal pan is fine.

Drop-in: insert between any bbox-producing node and InpaintCrop2.bounding_box_mask.
The output bbox sequence has the SAME size per frame (locked via size_mode)
but a smoothed center trajectory.

Acknowledgments:
  Multi-AI debate 2026-04-23 — Codex+Gemini both ranked this approach #1
  for the failed-hypothesis case (faithful tracking of real motion produces
  the visible jitter, not mask noise).

References:
  Grundmann et al., "Auto-Directed Video Stabilization with Robust L1 Optimal
    Camera Paths", CVPR 2011.
  Liu et al., "Bundled Camera Paths for Video Stabilization", SIGGRAPH 2013.
"""

import torch
import numpy as np
from scipy.optimize import minimize

from .bbox_ops import extract_bboxes, build_bbox_masks, print_bbox_trajectory_debug
from .mask_ops import mask_smooth


LOG_PREFIX = "[NV_OptimizeCropTrajectory]"


# =============================================================================
# Core solver: 1D constrained QP via L-BFGS-B
# =============================================================================

def _solve_axis(centers, lower, upper,
                lambda_p, lambda_v, lambda_a, lambda_j,
                pos_weight=None,
                max_iter=2000, ftol=1e-14, gtol=1e-12):
    """Solve a single-axis convex QP with box bounds via L-BFGS-B.

    Objective (residual-count-normalized, so lambdas are per-step penalties):
        L(s) = lambda_p * mean( w_t * (s_t - c_t)² )
             + lambda_v * mean( (s_{t+1} - s_t)² )
             + lambda_a * mean( (s_{t+2} - 2 s_{t+1} + s_t)² )
             + lambda_j * mean( (s_{t+3} - 3 s_{t+2} + 3 s_{t+1} - s_t)² )

    Subject to:
        lower_t <= s_t <= upper_t  for all t

    Args:
        centers:    (N,) raw input centers along this axis
        lower:      (N,) per-frame lower bound for smoothed center
        upper:      (N,) per-frame upper bound for smoothed center
        lambda_p/v/a/j: penalty weights (position pull, velocity, acceleration, jerk)
        pos_weight: (N,) per-frame multiplier on the position term (1.0 default).
                    Set to 0 for missing/dropped-out frames so the optimizer is not
                    pulled toward stale forward-fill values.

    Returns:
        smoothed: (N,) smoothed centers (clipped to bounds)
        cost:     final objective value
        success:  bool from scipy result
        n_iter:   iterations used
        n_infeas: int — number of frames where lower > upper (constraint impossible)
    """
    centers = np.asarray(centers, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    N = len(centers)

    if pos_weight is None:
        pos_weight = np.ones(N, dtype=np.float64)
    else:
        pos_weight = np.asarray(pos_weight, dtype=np.float64)

    # Repair infeasible bounds (lower > upper). Snap both to midpoint.
    # NOTE: This is least-bad-compromise behavior, NOT containment satisfaction.
    # Caller surfaces n_infeas in info so the user knows containment failed.
    infeas = lower > upper
    n_infeas = int(infeas.sum())
    if n_infeas:
        mid = 0.5 * (lower + upper)
        lower = np.where(infeas, mid, lower)
        upper = np.where(infeas, mid, upper)

    # Feasible warm-start
    x0 = np.clip(centers, lower, upper)

    # Per-step normalization factors (count of non-zero residuals per term).
    # Makes lambdas portable across clip lengths.
    inv_n_p = 1.0 / max(N, 1)
    inv_n_v = 1.0 / max(N - 1, 1)
    inv_n_a = 1.0 / max(N - 2, 1)
    inv_n_j = 1.0 / max(N - 3, 1)

    def cost(s):
        c_p = (lambda_p * inv_n_p * float(np.sum(pos_weight * (s - centers) ** 2))
               if lambda_p > 0 else 0.0)
        c_v = 0.0
        c_a = 0.0
        c_j = 0.0
        if N > 1 and lambda_v > 0:
            dv = s[1:] - s[:-1]
            c_v = lambda_v * inv_n_v * float(np.sum(dv ** 2))
        if N > 2 and lambda_a > 0:
            da = s[2:] - 2.0 * s[1:-1] + s[:-2]
            c_a = lambda_a * inv_n_a * float(np.sum(da ** 2))
        if N > 3 and lambda_j > 0:
            dj = s[3:] - 3.0 * s[2:-1] + 3.0 * s[1:-2] - s[:-3]
            c_j = lambda_j * inv_n_j * float(np.sum(dj ** 2))
        return c_p + c_v + c_a + c_j

    def grad(s):
        g = np.zeros_like(s)
        if lambda_p > 0:
            g += 2.0 * lambda_p * inv_n_p * pos_weight * (s - centers)
        if N > 1 and lambda_v > 0:
            dv = s[1:] - s[:-1]
            g[:-1] -= 2.0 * lambda_v * inv_n_v * dv
            g[1:]  += 2.0 * lambda_v * inv_n_v * dv
        if N > 2 and lambda_a > 0:
            da = s[2:] - 2.0 * s[1:-1] + s[:-2]
            g[:-2]  += 2.0 * lambda_a * inv_n_a * da
            g[1:-1] -= 4.0 * lambda_a * inv_n_a * da
            g[2:]   += 2.0 * lambda_a * inv_n_a * da
        if N > 3 and lambda_j > 0:
            dj = s[3:] - 3.0 * s[2:-1] + 3.0 * s[1:-2] - s[:-3]
            g[:-3]  -= 2.0 * lambda_j * inv_n_j * dj
            g[1:-2] += 6.0 * lambda_j * inv_n_j * dj
            g[2:-1] -= 6.0 * lambda_j * inv_n_j * dj
            g[3:]   += 2.0 * lambda_j * inv_n_j * dj
        return g

    bounds = list(zip(lower.tolist(), upper.tolist()))

    res = minimize(
        cost, x0, jac=grad, method='L-BFGS-B', bounds=bounds,
        options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol}
    )

    # Hard-clip in case solver tolerance left a tiny violation
    smoothed = np.clip(res.x, lower, upper)
    # OptimizeResult fields vary across scipy versions; `.get()` with default
    # avoids AttributeError on builds where `nit` isn't populated.
    nit = int(res.get('nit', -1)) if hasattr(res, 'get') else int(getattr(res, 'nit', -1))
    return smoothed, float(res.fun), bool(res.success), nit, n_infeas


# =============================================================================
# Deadband post-process (non-convex; collapses near-static runs to constant)
# =============================================================================

def _apply_deadband(smoothed, lower, upper, deadband_radius):
    """Collapse consecutive frames into constant segments while:
      (a) value spread within the segment stays <= 2 * deadband_radius
          (uses running min/max rather than first-element anchoring)
      (b) feasible bound intersection [max(lower), min(upper)] stays non-empty

    Each closed segment's value = mean of its members, clipped to the segment's
    feasible bound intersection. Per (b), the intersection is guaranteed
    non-empty for any segment we close, so the constant value is always
    feasible — no infeasible-segment fallback exists or should exist.

    This breaks small-amplitude residual oscillation that the QP couldn't fully
    flatten without making the trajectory rigid.
    """
    if deadband_radius <= 0 or len(smoothed) < 2:
        return smoothed.copy()

    s = smoothed.copy()
    N = len(s)
    radius2 = 2.0 * float(deadband_radius)
    out = s.copy()

    def close_segment(start, end_exclusive):
        """Replace [start, end_exclusive) with the segment mean, clipped to its
        guaranteed-non-empty bound intersection."""
        if end_exclusive - start <= 1:
            return  # singletons unchanged
        run_mean = float(np.mean(s[start:end_exclusive]))
        lo = float(np.max(lower[start:end_exclusive]))
        hi = float(np.min(upper[start:end_exclusive]))
        # Invariant from grow loop: lo <= hi for any closed segment.
        out[start:end_exclusive] = float(np.clip(run_mean, lo, hi))

    seg_start = 0
    seg_min = s[0]
    seg_max = s[0]
    seg_lo = lower[0]
    seg_hi = upper[0]

    for t in range(1, N):
        new_min = min(seg_min, s[t])
        new_max = max(seg_max, s[t])
        new_lo  = max(seg_lo, lower[t])
        new_hi  = min(seg_hi, upper[t])

        spread_ok = (new_max - new_min) <= radius2
        feasible_ok = new_lo <= new_hi

        if spread_ok and feasible_ok:
            seg_min, seg_max, seg_lo, seg_hi = new_min, new_max, new_lo, new_hi
        else:
            close_segment(seg_start, t)
            seg_start = t
            seg_min = s[t]
            seg_max = s[t]
            seg_lo = lower[t]
            seg_hi = upper[t]

    close_segment(seg_start, N)
    return out


# =============================================================================
# Node
# =============================================================================

class NV_OptimizeCropTrajectory:
    """Smooth a crop bbox trajectory via constrained convex optimization.

    Reads a per-frame bbox mask, extracts centers + extents, locks the bbox
    size, then solves a 1D QP per axis for the smoothest crop center sequence
    that still contains the subject with the requested margin. Outputs a
    rebuilt per-frame bbox mask with smoothed centers + locked size.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bbox_mask": ("MASK", {
                    "tooltip": "Per-frame bbox mask from upstream (NV_MaskTrackingBBox / "
                               "NV_PointDrivenBBox / any rectangular bbox source). "
                               "Centers extracted per frame, smoothed via constrained QP, "
                               "size locked, mask rebuilt."
                }),
                "lambda_velocity": ("FLOAT", {
                    "default": 0.25, "min": 0.0, "max": 1000.0, "step": 0.05,
                    "tooltip": "Penalty on frame-to-frame crop center motion (1st derivative). "
                               "Higher = smoother trajectory, more lag through fast subject motion. "
                               "Lambdas are normalized per-step (clip-length-portable)."
                }),
                "lambda_acceleration": ("FLOAT", {
                    "default": 30.0, "min": 0.0, "max": 10000.0, "step": 1.0,
                    "tooltip": "Penalty on bobbing / direction reversal (2nd derivative). "
                               "Higher = kills oscillation. Primary weight for damping head-bob. "
                               "Try 30-60 for walking shots with painful vertical bob."
                }),
                "lambda_jerk": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 10000.0, "step": 0.5,
                    "tooltip": "Penalty on jerkiness (3rd derivative). Promotes constant-velocity "
                               "segments. Often unnecessary if acceleration is sufficient. "
                               "0 = disabled (recommended unless residual jerkiness remains)."
                }),
                "containment_margin_px": ("INT", {
                    "default": 16, "min": 0, "max": 256, "step": 1,
                    "tooltip": "Required padding between subject extents and crop edge. "
                               "Larger = more freedom for crop to drift away from subject = "
                               "smoother trajectory possible. Smaller = subject can press against "
                               "crop edge during fast motion."
                }),
            },
            "optional": {
                "size_mode": (["lock_largest", "lock_median", "lock_first", "follow_input"], {
                    "default": "lock_largest",
                    "tooltip": "How to handle bbox size across frames. "
                               "lock_largest: max observed dims (recommended — fits all poses). "
                               "lock_median: median dims (tighter but may clip fast motion). "
                               "lock_first: frame 0 dims (riskiest for changing poses). "
                               "follow_input: per-frame variable size (smoothing affects only position)."
                }),
                "lambda_y_multiplier": ("FLOAT", {
                    "default": 3.0, "min": 0.5, "max": 20.0, "step": 0.5,
                    "tooltip": "Multiplier on Y-axis smoothing weights (vel + acc + jerk). "
                               ">1 damps vertical head-bob harder than horizontal pan — "
                               "perceptually correct for walking subjects. 1.0 = isotropic. "
                               "Try 3-5 for walking shots, 1-2 for static or panning shots."
                }),
                "deadband_radius_px": ("INT", {
                    "default": 0, "min": 0, "max": 64, "step": 1,
                    "tooltip": "Post-process: collapse runs where consecutive |Δs| < radius "
                               "into constant segments. Creates static-hold behavior between "
                               "corrective moves. 0 = disabled. Try 2-4 for ultra-stable holds."
                }),
                "lambda_position": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 100.0, "step": 0.05,
                    "tooltip": "Pull toward raw input centers. Higher = follows raw motion more "
                               "closely (good for panning shots). Lower = trusts smoothness terms "
                               "(good for static-camera walking shots). Set to 0 to anchor purely "
                               "via constraints. Per-frame weight is auto-zeroed for missing frames."
                }),
                "lambda_x_multiplier": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 20.0, "step": 0.1,
                    "tooltip": "Multiplier on X-axis smoothing weights. Usually 1.0; raise above "
                               "1 if horizontal jitter is also painful (e.g., side-to-side weave)."
                }),
                "solver_max_iter": ("INT", {
                    "default": 2000, "min": 100, "max": 50000, "step": 100,
                    "tooltip": "L-BFGS-B max iterations per axis. Convex QP usually converges "
                               "in <500 iters; raise if you see solver=False in the info output."
                }),
                "verbose_debug": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Print A/B trajectory stats: input vs smoothed Δ distribution, "
                               "spike frames, displacement comparison. Format matches "
                               "NV_MaskTrackingBBox / NV_PointDrivenBBox for direct comparison."
                }),
            }
        }

    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("bbox_mask", "info")
    FUNCTION = "execute"
    CATEGORY = "NV_Utils/Mask"
    DESCRIPTION = (
        "Smooth a crop bbox trajectory via constrained convex optimization. "
        "Decouples crop motion from subject motion by minimizing weighted "
        "velocity/acceleration/jerk subject to subject-containment bounds. "
        "Drop-in replacement for raw bbox output of MaskTrackingBBox / PointDrivenBBox."
    )

    def execute(self, bbox_mask, lambda_velocity, lambda_acceleration,
                lambda_jerk, containment_margin_px,
                size_mode="lock_largest",
                lambda_y_multiplier=3.0, deadband_radius_px=0,
                lambda_position=0.05, lambda_x_multiplier=1.0,
                solver_max_iter=2000, verbose_debug=False):
        info_lines = []

        # ── Validate ───────────────────────────────────────────────────────────
        if bbox_mask.dim() != 3:
            raise ValueError(f"bbox_mask must be [B, H, W], got {list(bbox_mask.shape)}")
        B, H, W = bbox_mask.shape

        info_lines.append(f"Input: {B} frames, {W}x{H}")

        # ── Single-frame fast path ─────────────────────────────────────────────
        if B < 2:
            info_lines.append("Single frame — no smoothing needed, passing through.")
            info = "\n".join(info_lines)
            print(f"{LOG_PREFIX} {info}")
            return (bbox_mask.clone(), info)

        # ── Extract per-frame bbox extents ─────────────────────────────────────
        x1s, y1s, x2s, y2s, present = extract_bboxes(bbox_mask, info_lines=None)
        x1s = np.asarray(x1s, dtype=np.float64)
        y1s = np.asarray(y1s, dtype=np.float64)
        x2s = np.asarray(x2s, dtype=np.float64)
        y2s = np.asarray(y2s, dtype=np.float64)
        present = np.asarray(present, dtype=bool)
        n_present = int(present.sum())
        n_missing = B - n_present

        if n_present == 0:
            info_lines.append("WARNING: bbox_mask contains no non-empty frames — passing through unchanged.")
            info = "\n".join(info_lines)
            print(f"{LOG_PREFIX} {info}")
            return (bbox_mask.clone(), info)

        if n_missing:
            info_lines.append(
                f"Missing detections: {n_missing}/{B} frames. Containment bounds "
                f"relaxed to frame-only on those frames; position term zeroed "
                f"(no pull toward forward-filled stale centers)."
            )

        widths  = np.maximum(1.0, x2s - x1s)
        heights = np.maximum(1.0, y2s - y1s)
        cx_raw  = 0.5 * (x1s + x2s)
        cy_raw  = 0.5 * (y1s + y2s)

        # ── Determine locked size (only over PRESENT frames) ───────────────────
        widths_p  = widths[present]
        heights_p = heights[present]
        if size_mode == "follow_input":
            target_w = widths.copy()
            target_h = heights.copy()
            info_lines.append("Size: follow_input (per-frame variable)")
        elif size_mode == "lock_largest":
            tw, th = float(widths_p.max()), float(heights_p.max())
            target_w = np.full(B, tw)
            target_h = np.full(B, th)
            info_lines.append(f"Size locked (lock_largest): {tw:.0f}x{th:.0f}")
        elif size_mode == "lock_median":
            tw, th = float(np.median(widths_p)), float(np.median(heights_p))
            target_w = np.full(B, tw)
            target_h = np.full(B, th)
            info_lines.append(f"Size locked (lock_median): {tw:.0f}x{th:.0f}")
        elif size_mode == "lock_first":
            first_idx = int(np.argmax(present))  # first present frame
            tw, th = float(widths[first_idx]), float(heights[first_idx])
            target_w = np.full(B, tw)
            target_h = np.full(B, th)
            info_lines.append(f"Size locked (lock_first, frame {first_idx}): {tw:.0f}x{th:.0f}")
        else:
            raise ValueError(f"Unknown size_mode: {size_mode}")

        if np.any(target_w > W) or np.any(target_h > H):
            info_lines.append(
                f"WARNING: locked crop size ({target_w.max():.0f}x{target_h.max():.0f}) "
                f"exceeds frame ({W}x{H}). Containment is impossible; results will be "
                f"midpoint-clamped on affected frames."
            )

        # ── Containment bounds per frame ───────────────────────────────────────
        # For PRESENT frames with margin m and crop size W_t:
        #   s_t.x - W_t/2 + m  <=  subject_xmin_t
        #   s_t.x + W_t/2 - m  >=  subject_xmax_t
        # i.e. lower = subject_xmax_t + m - W_t/2,  upper = subject_xmin_t + W_t/2 - m
        # Plus frame bounds: W_t/2 <= s_t.x <= frame_W - W_t/2
        # For MISSING frames: relax to frame bounds only (no subject containment).
        m = float(containment_margin_px)
        half_w = 0.5 * target_w
        half_h = 0.5 * target_h

        # Subject-containment bounds (only meaningful where present)
        sub_lower_x = x2s + m - half_w
        sub_upper_x = x1s + half_w - m
        sub_lower_y = y2s + m - half_h
        sub_upper_y = y1s + half_h - m

        # Frame-only bounds (always valid)
        frame_lower_x = half_w
        frame_upper_x = W - half_w
        frame_lower_y = half_h
        frame_upper_y = H - half_h

        # Combine: present → intersect subject + frame; missing → frame-only.
        lower_x = np.where(present, np.maximum(frame_lower_x, sub_lower_x), frame_lower_x)
        upper_x = np.where(present, np.minimum(frame_upper_x, sub_upper_x), frame_upper_x)
        lower_y = np.where(present, np.maximum(frame_lower_y, sub_lower_y), frame_lower_y)
        upper_y = np.where(present, np.minimum(frame_upper_y, sub_upper_y), frame_upper_y)

        n_infeas_x = int(np.sum(lower_x > upper_x))
        n_infeas_y = int(np.sum(lower_y > upper_y))
        if n_infeas_x or n_infeas_y:
            info_lines.append(
                f"WARNING: containment infeasible on {n_infeas_x} X frames, "
                f"{n_infeas_y} Y frames (subject + 2×margin > crop_size, OR crop > frame). "
                f"Affected frames will snap to bound midpoint — containment NOT preserved. "
                f"Try larger size_mode (lock_largest), smaller containment_margin_px, "
                f"or check for oversized subject masks."
            )

        # ── Per-frame position weight: zero out missing frames ─────────────────
        pos_weight = present.astype(np.float64)

        # ── Solve per-axis QPs ─────────────────────────────────────────────────
        lam_p = float(lambda_position)
        lam_vx = float(lambda_velocity) * float(lambda_x_multiplier)
        lam_vy = float(lambda_velocity) * float(lambda_y_multiplier)
        lam_ax = float(lambda_acceleration) * float(lambda_x_multiplier)
        lam_ay = float(lambda_acceleration) * float(lambda_y_multiplier)
        lam_jx = float(lambda_jerk) * float(lambda_x_multiplier)
        lam_jy = float(lambda_jerk) * float(lambda_y_multiplier)

        info_lines.append(
            f"Weights: pos={lam_p:.3f}  "
            f"X(v={lam_vx:.2f}, a={lam_ax:.2f}, j={lam_jx:.2f})  "
            f"Y(v={lam_vy:.2f}, a={lam_ay:.2f}, j={lam_jy:.2f})  "
            f"margin={m:.0f}px"
        )

        sx, cost_x, ok_x, nit_x, _ = _solve_axis(
            cx_raw, lower_x, upper_x,
            lam_p, lam_vx, lam_ax, lam_jx,
            pos_weight=pos_weight,
            max_iter=int(solver_max_iter),
        )
        sy, cost_y, ok_y, nit_y, _ = _solve_axis(
            cy_raw, lower_y, upper_y,
            lam_p, lam_vy, lam_ay, lam_jy,
            pos_weight=pos_weight,
            max_iter=int(solver_max_iter),
        )
        info_lines.append(
            f"Solver: X(ok={ok_x}, iter={nit_x}, cost={cost_x:.3e})  "
            f"Y(ok={ok_y}, iter={nit_y}, cost={cost_y:.3e})"
        )
        if not ok_x:
            info_lines.append(f"WARNING: X-axis solver did not report success — check solver_max_iter or lambda magnitudes.")
        if not ok_y:
            info_lines.append(f"WARNING: Y-axis solver did not report success — check solver_max_iter or lambda magnitudes.")

        # ── Optional deadband post-process ─────────────────────────────────────
        if deadband_radius_px > 0:
            sx_db = _apply_deadband(sx, lower_x, upper_x, float(deadband_radius_px))
            sy_db = _apply_deadband(sy, lower_y, upper_y, float(deadband_radius_px))
            n_changed = int(np.sum(np.abs(sx_db - sx) + np.abs(sy_db - sy) > 1e-6))
            sx, sy = sx_db, sy_db
            info_lines.append(
                f"Deadband {deadband_radius_px}px: collapsed to constant segments, "
                f"{n_changed} frames adjusted"
            )

        # ── Build smoothed bbox coordinates ────────────────────────────────────
        sx1 = sx - half_w
        sy1 = sy - half_h
        sx2 = sx + half_w
        sy2 = sy + half_h

        # ── Build output mask (no extra padding — size already includes upstream) ─
        smoothed_mask = build_bbox_masks(
            sx1.tolist(), sy1.tolist(), sx2.tolist(), sy2.tolist(),
            padding=0.0,
            H=H, W=W,
            info_lines=info_lines,
        )
        # Preserve input device + dtype (build_bbox_masks returns CPU float32 by default)
        smoothed_mask = smoothed_mask.to(device=bbox_mask.device, dtype=bbox_mask.dtype)

        # ── Verbose debug: A/B against raw input ───────────────────────────────
        if verbose_debug:
            raw_positions = list(zip(cx_raw.tolist(), cy_raw.tolist()))
            smoothed_positions = list(zip(sx.tolist(), sy.tolist()))
            print_bbox_trajectory_debug(
                positions=smoothed_positions,
                compare_positions=raw_positions,
                compare_label="Smoothed ↔ raw input divergence "
                              "(this is the work the optimizer did — large divergence "
                              "= heavy smoothing was applied to remove jitter)",
                log_prefix=LOG_PREFIX,
            )

        # ── Summary ────────────────────────────────────────────────────────────
        # Compute smoothed Δ stats for the info string
        d_smoothed = np.sqrt(np.diff(sx) ** 2 + np.diff(sy) ** 2)
        d_raw = np.sqrt(np.diff(cx_raw) ** 2 + np.diff(cy_raw) ** 2)
        info_lines.append(
            f"Trajectory Δ: raw mean={d_raw.mean():.2f}px max={d_raw.max():.2f}px  →  "
            f"smoothed mean={d_smoothed.mean():.2f}px max={d_smoothed.max():.2f}px  "
            f"(reduction: {(1 - d_smoothed.mean() / max(d_raw.mean(), 1e-6)) * 100:.0f}%)"
        )
        info = "\n".join(info_lines)
        print(f"{LOG_PREFIX} {info}")

        return (smoothed_mask, info)


# ── Registration ────────────────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    "NV_OptimizeCropTrajectory": NV_OptimizeCropTrajectory,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_OptimizeCropTrajectory": "NV Optimize Crop Trajectory",
}
