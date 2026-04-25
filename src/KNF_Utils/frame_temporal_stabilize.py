"""
NV Frame Temporal Stabilize — post-spatial temporal coherence pass.

Companion to NV_FaceHarmonizePyramid (and standalone for any video workflow).
Motion-adaptive bidirectional EMA with an optional Farneback optical-flow
alignment step. Separates temporal logic from spatial harmonization.

This node is distinct from `temporal_stabilize.py` (which is for control-pass
workflows — depth/edge/canny V2V propagation). This one operates on
already-harmonized RGB IMAGE sequences and smooths per-frame variance.

Post-review architecture (2026-04-24 round 2):

  Correct bidirectional fusion:
    fwd = pass(frames, reverse=False)       # both passes see RAW frames,
    bwd = pass(frames, reverse=True)        # not each other — avoids
    result = 0.5 * (fwd + bwd)               # EMA-on-top-of-EMA ghosting

  Exponential motion adaptation:
    effective_alpha = temporal_strength * exp(-motion_sensitivity * motion_delta)
    Replaces the prior 1/(1 + k*d) form which was too weak at realistic motion
    magnitudes.

  Vectorized motion delta precompute:
    motion_deltas = torch.mean(|frames[1:] - frames[:-1]|, in-mask, over HxW)
    Computed ONCE as a (T-1,) tensor, no per-frame .item() syncs in the loop.

  Flow direction:
    Farneback(B, A) returns backward flow from B -> A, i.e., flow[y, x] at
    position (x, y) in B points to the corresponding location in A.
    Then `sample = (X + flow_x, Y + flow_y)` inside grid_sample gives us
    `output = A aligned to B's coordinates`. The prior code had the args
    swapped and was using forward flow with a backward-flow sampling
    formula — fixed here.

  Flow cache:
    For T frames we compute two arrays of backward flows:
      - flow_to_prev[t] for t in [1, T-1]: used in forward pass
      - flow_to_next[t] for t in [0, T-2]: used in backward pass
    Each is computed exactly once; both passes reuse them.

  Grid-sample padding:
    reflection -> border. Reflection can mirror face/background back into the
    subject region near frame edges. Border clamps to edge pixels, which is
    more appropriate for subject warps.
"""

import torch
import torch.nn.functional as F


LOG_PREFIX = "[NV_FrameTemporalStabilize]"


# =============================================================================
# Vectorized motion delta precompute (no per-frame .item() sync)
# =============================================================================

def _precompute_motion_deltas(frames, mask):
    """Compute per-pair scalar motion magnitudes for all (t-1, t) pairs.

    Args:
        frames: [T, H, W, 3] in [0, 1], fp32
        mask:   [T, H, W] in [0, 1] or None

    Returns:
        deltas: [T-1] tensor (same device as frames). deltas[i] is the mean
                absolute channel difference between frames[i] and frames[i+1],
                weighted by mask[i+1] if provided.
    """
    if frames.shape[0] < 2:
        return torch.zeros(0, device=frames.device, dtype=frames.dtype)

    diffs = (frames[1:] - frames[:-1]).abs().mean(dim=-1)  # [T-1, H, W]

    if mask is not None:
        m = mask[1:].clamp(0.0, 1.0)  # [T-1, H, W], aligned with the "current" frame
        weight = m.sum(dim=(1, 2)).clamp_min(1.0)  # [T-1]
        return (diffs * m).sum(dim=(1, 2)) / weight  # [T-1]
    return diffs.mean(dim=(1, 2))  # [T-1]


# =============================================================================
# Optical flow — cached precompute
# =============================================================================

def _cv2_available():
    try:
        import cv2  # noqa: F401
        return True
    except ImportError:
        return False


def _compute_backward_flow(src_hwc, dst_hwc):
    """Compute backward flow from dst -> src via Farneback.

    "Backward flow" here means: flow[y, x] at each dst position (x, y) gives
    (dx, dy) such that sampling src at (x + dx, y + dy) yields the pixel
    that moved to (x, y) in dst. This is the flow we need for grid_sample
    to warp src into dst's coordinate system.

    Internally we invoke Farneback as Farneback(dst, src), which gives
    forward flow from dst -> src — i.e., "where did each dst pixel come
    from in src". That IS the backward-mapping flow we want at each dst
    position. (The name "backward" vs. "forward" depends on whether you
    mean direction of time or direction of mapping; we use mapping
    direction consistently throughout this module.)

    Args:
        src_hwc, dst_hwc: [H, W, 3] float tensors in [0, 1].
    Returns:
        flow: [H, W, 2] float32 tensor with (dx, dy) in dst coords,
              or None if cv2 unavailable / compute failed.
    """
    try:
        import cv2
    except ImportError:
        return None

    src_np = (src_hwc.detach().cpu().numpy() * 255.0).clip(0, 255).astype("uint8")
    dst_np = (dst_hwc.detach().cpu().numpy() * 255.0).clip(0, 255).astype("uint8")
    src_gray = cv2.cvtColor(src_np, cv2.COLOR_RGB2GRAY)
    dst_gray = cv2.cvtColor(dst_np, cv2.COLOR_RGB2GRAY)

    try:
        # Farneback(dst, src) gives flow from dst -> src, which is the
        # backward-mapping flow we want for "warp src into dst coords".
        flow = cv2.calcOpticalFlowFarneback(
            dst_gray, src_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
    except Exception:
        return None

    return torch.from_numpy(flow).to(dtype=src_hwc.dtype, device=src_hwc.device)


def _precompute_flow_arrays(orig_frames, need_to_prev, need_to_next):
    """Precompute (T,)-indexed arrays of backward flows in both directions.

    Args:
        orig_frames: [T, H, W, 3]
        need_to_prev: bool, precompute flows used by the forward EMA pass
        need_to_next: bool, precompute flows used by the backward EMA pass

    Returns:
        (flow_to_prev, flow_to_next): each is a list of length T, with None at
        boundary positions that have no neighbor (flow_to_prev[0] is None,
        flow_to_next[T-1] is None). Each entry is a [H, W, 2] tensor.
    """
    T = orig_frames.shape[0]
    flow_to_prev = [None] * T
    flow_to_next = [None] * T

    if need_to_prev:
        for t in range(1, T):
            flow_to_prev[t] = _compute_backward_flow(orig_frames[t - 1], orig_frames[t])
    if need_to_next:
        for t in range(0, T - 1):
            flow_to_next[t] = _compute_backward_flow(orig_frames[t + 1], orig_frames[t])
    return flow_to_prev, flow_to_next


# =============================================================================
# Warp + confidence
# =============================================================================

def _warp_by_flow(img_hwc, flow_hw2, padding_mode="border"):
    """Warp img by a dense backward-mapping flow via grid_sample.

    For each output pixel (x, y), samples img at (x + dx, y + dy) where
    (dx, dy) = flow[y, x]. Use `padding_mode='border'` (default) to clamp
    out-of-bounds samples to the edge — safer than reflection for subject
    warps where the boundary is background, not a mirrored feature.
    """
    H, W, C = img_hwc.shape
    device = img_hwc.device
    dtype = img_hwc.dtype

    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    sample_x = xs + flow_hw2[..., 0]
    sample_y = ys + flow_hw2[..., 1]
    norm_x = (sample_x / max(W - 1, 1)) * 2.0 - 1.0
    norm_y = (sample_y / max(H - 1, 1)) * 2.0 - 1.0
    grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0)  # [1, H, W, 2]

    img_bchw = img_hwc.permute(2, 0, 1).unsqueeze(0)
    warped = F.grid_sample(img_bchw, grid, mode="bilinear", padding_mode=padding_mode, align_corners=True)
    return warped.squeeze(0).permute(1, 2, 0)


def _flow_confidence(warped_prev, cur, occlusion_threshold):
    """Confidence map from photometric residual. 1 = reliable, 0 = occluded."""
    residual = (warped_prev - cur).abs().mean(dim=-1)
    if occlusion_threshold <= 0:
        return torch.ones_like(residual)
    conf = 1.0 - (residual / occlusion_threshold).clamp(0.0, 1.0)
    return conf


# =============================================================================
# Core temporal pass — consumes RAW frames (not smoothed output)
# =============================================================================

def _temporal_pass(frames, mask, orig_frames, flow_cache, motion_deltas, *,
                   temporal_strength, motion_sensitivity,
                   smooth_scope, flow_mode, occlusion_threshold,
                   reverse=False):
    """Single-direction EMA pass.

    Critical: the motion delta used for alpha attenuation is measured between
    RAW frames[t] and RAW frames[prev_t] (from the precomputed table), NOT
    from the smoothed prior. That prevents the stabilizer from "locking in"
    whatever state it first produced.

    The prior used for blending IS the previous output (so the EMA actually
    accumulates), but motion adaptation sees raw signal.

    Args:
        frames:        [T, H, W, 3] RAW input sequence
        mask:          [T, H, W] or None
        orig_frames:   [T, H, W, 3] for flow (or None if flow_mode=off)
        flow_cache:    list of length T of [H, W, 2] tensors (backward flows)
                       or Nones at boundaries
        motion_deltas: [T-1] tensor — motion_deltas[i] is delta between
                       frames[i] and frames[i+1]
        temporal_strength: base EMA alpha in [0, 1]
        motion_sensitivity: k in `alpha_eff = alpha * exp(-k * delta)`
        smooth_scope: "mask_only" or "full_frame"
        flow_mode: "off" or "farneback"
        occlusion_threshold: residual threshold that zeros alpha (flow only)
        reverse: if True, run backward through time

    Returns:
        [T, H, W, 3] smoothed sequence
    """
    T, H, W, _ = frames.shape
    order = range(T - 2, -1, -1) if reverse else range(1, T)
    seed_idx = T - 1 if reverse else 0
    out = frames.clone()
    out[seed_idx] = frames[seed_idx]

    use_flow = (flow_mode == "farneback" and orig_frames is not None and flow_cache is not None)

    for t in order:
        prev_t = t + 1 if reverse else t - 1

        # Motion delta from precomputed table (raw-frames signal)
        # motion_deltas[i] corresponds to (frames[i], frames[i+1]). For the
        # forward pass the pair is (prev_t, t) -> index = prev_t = t-1.
        # For backward pass the pair is (t, prev_t) where prev_t = t+1 -> also index = t.
        delta_idx = min(t, prev_t)
        motion_delta = motion_deltas[delta_idx]  # 0-dim tensor, no .item()

        # Exponential motion-adaptive alpha (much stronger decay than 1/(1+k*d))
        # At delta=0: alpha_eff = temporal_strength
        # At delta=0.1, k=1.5: alpha_eff = temporal_strength * 0.86
        # At delta=0.3, k=1.5: alpha_eff = temporal_strength * 0.64
        effective_alpha = temporal_strength * torch.exp(-motion_sensitivity * motion_delta)

        # Establish the prior (possibly flow-warped)
        prior = out[prev_t]
        flow_conf_hw = None

        if use_flow:
            flow = flow_cache[t]
            if flow is not None:
                prior = _warp_by_flow(out[prev_t], flow, padding_mode="border")
                orig_warped = _warp_by_flow(orig_frames[prev_t], flow, padding_mode="border")
                flow_conf_hw = _flow_confidence(orig_warped, orig_frames[t], occlusion_threshold)

        # Blend
        cur = frames[t]
        if flow_conf_hw is not None:
            alpha_map = (effective_alpha * flow_conf_hw).unsqueeze(-1)  # [H, W, 1]
            blended = alpha_map * prior + (1.0 - alpha_map) * cur
        else:
            blended = effective_alpha * prior + (1.0 - effective_alpha) * cur

        # Scope
        if smooth_scope == "mask_only" and mask is not None:
            m = mask[t].clamp(0.0, 1.0).unsqueeze(-1)
            blended = m * blended + (1.0 - m) * cur

        out[t] = blended

    return out


# =============================================================================
# Node
# =============================================================================

class NV_FrameTemporalStabilize:
    """Post-spatial temporal coherence pass for harmonized frame sequences.

    Motion-adaptive bidirectional EMA with correct forward+backward fusion.
    Dampens per-frame variance (VAE fizz, sampling noise, texture shimmer)
    while preserving intentional motion via exponential alpha decay.

    Pipeline placement:
        NV_FaceHarmonizePyramid → NV_FrameTemporalStabilize → NV_InpaintStitch2
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE", {
                    "tooltip": "Input frame sequence [T, H, W, C]. Typically from "
                               "NV_FaceHarmonizePyramid output."
                }),
                "temporal_strength": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 0.9, "step": 0.05,
                    "tooltip": "Base EMA alpha. 0 = passthrough. 0.3 = moderate. "
                               "0.5 = heavy. Values above 0.7 cause temporal smearing on fast motion."
                }),
            },
            "optional": {
                "mask": ("MASK", {
                    "tooltip": "Optional per-frame subject mask. With smooth_scope=mask_only, EMA "
                               "blending is confined to masked pixels; outside passes through. Also "
                               "used to gate motion-delta estimation to the subject."
                }),
                "original_frames": ("IMAGE", {
                    "tooltip": "Optional original plate frames (pre-VACE). REQUIRED for "
                               "flow_mode!=off — flow is computed from these as a stable motion "
                               "ground truth."
                }),
                "motion_sensitivity": ("FLOAT", {
                    "default": 5.0, "min": 0.0, "max": 40.0, "step": 0.5,
                    "tooltip": "Motion-adaptive alpha decay rate k in exp(-k * motion_delta). "
                               "Higher = EMA backs off faster on fast motion. "
                               "With motion_delta ~0.05 (typical walking subject), k=5 gives ~22%% "
                               "alpha reduction; k=10 gives ~39%%. k=0 disables adaptation."
                }),
                "smooth_scope": (["mask_only", "full_frame"], {
                    "default": "mask_only",
                    "tooltip": "mask_only: blend only where mask>0 per frame. full_frame: blend "
                               "everywhere (BG smoothing too — usually not wanted for inpaint)."
                }),
                "direction": (["bidirectional", "causal"], {
                    "default": "bidirectional",
                    "tooltip": "bidirectional: forward+backward passes run INDEPENDENTLY on raw "
                               "frames, then averaged — zero phase lag. causal: forward only."
                }),
                "flow_mode": (["off", "farneback"], {
                    "default": "off",
                    "tooltip": "off: prior frame is used directly for EMA (fast; may leave lip "
                               "doubling / texture crawl during motion). "
                               "farneback: warp prior frame to current by backward flow from plate "
                               "before blending. Requires opencv. Flows are precomputed once and "
                               "cached across both directions — ~50-100 ms/frame for the precompute."
                }),
                "occlusion_threshold": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Flow confidence threshold. Where warped-prev vs current residual "
                               "exceeds this, EMA alpha is scaled toward 0 (occlusion skip). "
                               "0.3 = tolerant, 0.1 = strict. flow_mode=farneback only."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("frames", "info")
    FUNCTION = "execute"
    CATEGORY = "NV_Utils/Color"
    DESCRIPTION = (
        "Motion-adaptive bidirectional EMA with correct fwd+bwd fusion and "
        "optional Farneback flow warp. Post-spatial temporal coherence pass."
    )

    def execute(self, frames, temporal_strength,
                mask=None, original_frames=None,
                motion_sensitivity=5.0, smooth_scope="mask_only",
                direction="bidirectional", flow_mode="off",
                occlusion_threshold=0.3):
        info_lines = []

        # ── Shape validation ─────────────────────────────────────────────────
        if frames.dim() != 4:
            raise ValueError(f"frames must be [T, H, W, C]; got {list(frames.shape)}")
        T, H, W, C = frames.shape
        if C not in (3, 4):
            raise ValueError(f"frames must have 3 or 4 channels, got {C}")

        # Short-circuit passthrough
        if T < 2 or temporal_strength <= 0.0:
            info_lines.append(f"passthrough (T={T}, strength={temporal_strength})")
            info = f"{LOG_PREFIX} " + " | ".join(info_lines)
            print(info)
            return (frames, info)

        # Work in RGB (drop alpha for processing; re-attach at end)
        work_frames = frames[..., :3].float().contiguous()
        out_dtype = frames.dtype

        # Mask validation
        work_mask = None
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
            if mask.shape[0] == 1 and T > 1:
                mask = mask.expand(T, -1, -1)
            if mask.shape != (T, H, W):
                raise ValueError(
                    f"mask shape {list(mask.shape)} must be [T={T}, H={H}, W={W}]"
                )
            work_mask = mask.float().contiguous()

        # Original plate (for flow)
        work_orig = None
        if original_frames is not None:
            if original_frames.shape[:3] != (T, H, W):
                if original_frames.shape[0] == 1:
                    original_frames = original_frames.expand(T, -1, -1, -1)
                if original_frames.shape[:3] != (T, H, W):
                    raise ValueError(
                        f"original_frames shape {list(original_frames.shape)} must match "
                        f"frames {[T, H, W, 3]} in T/H/W"
                    )
            work_orig = original_frames[..., :3].float().contiguous()

        # Flow mode graceful downgrades
        if flow_mode == "farneback" and work_orig is None:
            info_lines.append("WARN: flow_mode=farneback but original_frames missing -- falling back to off")
            flow_mode = "off"
        if flow_mode == "farneback" and not _cv2_available():
            info_lines.append("WARN: opencv not available -- flow_mode disabled")
            flow_mode = "off"

        # ── Precompute motion deltas (vectorized, one-shot) ───────────────────
        motion_deltas = _precompute_motion_deltas(work_frames, work_mask)

        # ── Precompute flow arrays (both directions, each flow computed ONCE) ──
        flow_to_prev = None
        flow_to_next = None
        if flow_mode == "farneback":
            need_prev = True  # forward pass always needs t -> t-1 flows
            need_next = (direction == "bidirectional")  # backward pass needs t -> t+1
            flow_to_prev, flow_to_next = _precompute_flow_arrays(work_orig, need_prev, need_next)
            # Count successful computes for diagnostics
            fwd_ok = sum(1 for f in flow_to_prev if f is not None) if flow_to_prev else 0
            bwd_ok = sum(1 for f in flow_to_next if f is not None) if flow_to_next else 0
            info_lines.append(f"flow precomputed: {fwd_ok} to_prev, {bwd_ok} to_next")

        # ── Forward pass on RAW frames ────────────────────────────────────────
        fwd = _temporal_pass(
            work_frames, work_mask, work_orig, flow_to_prev, motion_deltas,
            temporal_strength=float(temporal_strength),
            motion_sensitivity=float(motion_sensitivity),
            smooth_scope=smooth_scope,
            flow_mode=flow_mode,
            occlusion_threshold=float(occlusion_threshold),
            reverse=False,
        )

        if direction == "bidirectional":
            # Backward pass ALSO consumes RAW frames (not fwd) — independent
            # forward and backward estimates are then averaged. This is the
            # textbook bidirectional EMA and avoids the EMA-on-top-of-EMA
            # ghosting that the v1 MVP had.
            bwd = _temporal_pass(
                work_frames, work_mask, work_orig, flow_to_next, motion_deltas,
                temporal_strength=float(temporal_strength),
                motion_sensitivity=float(motion_sensitivity),
                smooth_scope=smooth_scope,
                flow_mode=flow_mode,
                occlusion_threshold=float(occlusion_threshold),
                reverse=True,
            )
            result = 0.5 * fwd + 0.5 * bwd
            del bwd
            info_lines.append("bidirectional (independent fwd+bwd, averaged)")
        else:
            result = fwd
            info_lines.append("causal")
        del fwd

        # ── Diagnostic: temporal delta before/after ──────────────────────────
        with torch.no_grad():
            raw_delta = (work_frames[1:] - work_frames[:-1]).abs().mean().item()
            out_delta = (result[1:] - result[:-1]).abs().mean().item()
            info_lines.append(
                f"frame-to-frame abs delta: raw={raw_delta*255:.2f}/255 -> "
                f"smoothed={out_delta*255:.2f}/255 "
                f"(reduction {(1.0 - out_delta/max(raw_delta, 1e-12))*100:.0f}%)"
            )

        info_lines.append(
            f"T={T} strength={temporal_strength:.2f} motion_sens={motion_sensitivity:.1f} "
            f"scope={smooth_scope} flow={flow_mode}"
        )

        if C == 4:
            result = torch.cat([result, frames[..., 3:4].float()], dim=-1)

        result = result.to(dtype=out_dtype)
        info = f"{LOG_PREFIX} " + " | ".join(info_lines)
        print(info)
        return (result, info)


# ── Registration ─────────────────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    "NV_FrameTemporalStabilize": NV_FrameTemporalStabilize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_FrameTemporalStabilize": "NV Frame Temporal Stabilize",
}
