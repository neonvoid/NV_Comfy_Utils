"""
NV Warp Alignment Check - Diagnostic checker for the NV_PreCompWarp + mask suite pipeline.

Sits between NV_PreCompWarp and the downstream VACE V2V refinement so you can verify
alignment cheaply (numbers + visualization) before committing to a multi-minute render.

OPERATES IN TWO MODES (selected by `face_swap_mode` boolean input):
  - face_swap_mode=True (DEFAULT): geometry-only gating via edge_IoU + phase_offset, with
                                    looser thresholds. PSNR/SSIM still computed and reported
                                    but NOT gated — they will always be low between two
                                    different actors and would falsely fail is_aligned.
                                    Use this for face/body swap, identity transfer, any
                                    workflow where the warped content is semantically
                                    different from the original.
  - face_swap_mode=False (strict): PSNR + SSIM + edge_IoU + phase_offset all gated.
                                    Use ONLY when comparing the SAME content re-warped
                                    (same actor, same scene, just geometric correction).

NUMERICAL METRICS (mask-restricted if a mask is provided — strongly recommended):
  - PSNR (dB)        — pixel similarity. >27 = good for V2V swap; <22 = obviously off.
                       PSNR is sensitive to appearance differences (denoise, color drift)
                       not just geometry — use as a coarse signal, not primary go/no-go.
  - SSIM             — structural similarity (Gaussian-windowed, reflect-padded).
                       0..1. >0.85 mean = aligned; <0.75 = misregistered.
  - L1 / RMSE        — mean absolute / RMS pixel error in [0, 1] units.
  - edge_IoU         — Sobel-edge overlap inside the mask, with per-frame adaptive
                       thresholding (normalize by per-frame max gradient). >0.4 = aligned.
  - phase_dx, phase_dy — sub-pixel residual translation between original and warped
                         per frame (cv2.phaseCorrelate with Hann window on masked bbox).
                         Magnitude near 0 = no residual drift; >2 px = systematic warp drift.
                         SIGN CONVENTION: cv2.phaseCorrelate(original, warped) returns the
                         (dx, dy) shift required to register `warped` to `original`. So
                         positive dx = "warped is shifted left of original by dx px"
                         (you'd add dx to warped's coordinates to align it).

OUTPUT BOOLEAN `is_aligned`:
  Convenience signal derived from PSNR/SSIM/phase_offset percentiles. Wire this into
  a Switch node downstream to dynamically bypass VACE rendering when alignment fails.

VISUALIZATION MODES:
  - anaglyph     (DEFAULT) — original→cyan, warped→red. Aligned = grayscale; misaligned
                              = colored fringes. Sub-pixel offsets visible.
  - translucent  — original at full opacity + warped tinted (chosen color), alpha-blended.
  - difference   — |original - warped| × gain.
  - checkerboard — alternating tiles. Tearing at boundaries = misalignment.
  - edge_overlay — GPU Sobel edges of warped, drawn in tint over original (no CPU Canny).

USAGE PATTERN (post-NV_PreCompWarp):
  source_plate_crop ──► original
  warped_render     ──► warped
  source_actor_mask ──► mask    (the SOURCE plate's stabilized binary mask — NOT the
                                 warped mask. The warped mask follows the warped silhouette
                                 and biases edge_IoU + phase-corr crop toward the warp's
                                 own answer.)

Read the summary STRING. The is_aligned BOOLEAN gives a one-shot pass/fail. If misaligned,
per-frame metrics tell you which frames are worst — go fix anchor placement there.
"""

import json
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.nn.functional as F


# Color presets for translucent / edge_overlay modes. RGB in [0, 1].
_TINT_COLORS = OrderedDict([
    ("red",     (1.00, 0.20, 0.20)),
    ("green",   (0.20, 1.00, 0.30)),
    ("blue",    (0.30, 0.50, 1.00)),
    ("yellow",  (1.00, 0.92, 0.25)),
    ("magenta", (1.00, 0.25, 0.95)),
    ("cyan",    (0.25, 0.95, 1.00)),
])
_TINT_NAMES = list(_TINT_COLORS.keys())

_VIZ_MODES = ["anaglyph", "translucent", "difference", "checkerboard", "edge_overlay"]

# Reading-guide thresholds, recalibrated for low-denoise face/body swap pipeline.
# Used by both the printed summary and the is_aligned BOOLEAN output.
_THRESH_PSNR_MEDIAN = 27.0       # PSNR median > 27 dB = aligned
_THRESH_SSIM_MEAN = 0.85         # SSIM mean > 0.85 = aligned
_THRESH_PHASE_P90 = 2.0          # 90% of frames must have phase_offset_magnitude < 2 px
_THRESH_EDGE_IOU = 0.40          # edge_IoU > 0.4 (with adaptive thresholding) = aligned

# Looser thresholds for face_swap_mode (different actor, may have different pose).
# In this mode PSNR/SSIM are NOT gated — pixel content differs by design (different person)
# regardless of how well the warp aligned. Only the geometry-truth metrics (edge_IoU,
# phase_offset) gate the verdict. Thresholds loosened to tolerate the residual pose
# difference that a 4-DOF similarity warp can't fully correct between two actors.
_THRESH_EDGE_IOU_FACE_SWAP = 0.20    # different person, partially-aligned pose → looser
_THRESH_PHASE_P90_FACE_SWAP = 5.0    # tolerate residual translation drift the warp can't correct

# Phase-correlation requires a minimum bbox area to produce meaningful FFT estimates.
# 16 px is enough for divide-by-zero protection but produces garbage on tiny crops.
_PHASE_MIN_BBOX_DIM = 32         # min height/width of mask bbox for phase correlation
_PHASE_MIN_ACTIVE_PX = 256       # min active pixel count for phase correlation


class NV_WarpAlignmentCheck:
    """Per-frame alignment metrics + diagnostic visualization for the warp + mask pipeline."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original": ("IMAGE", {
                    "tooltip": "Source plate cropped region (typically from InpaintCrop2). "
                               "RGB or RGBA accepted; alpha is ignored."
                }),
                "warped": ("IMAGE", {
                    "tooltip": "Warped/corrected output to check (typically warped_render from NV_PreCompWarp). "
                               "Must depict the same geometry as `original` — different content but same coordinate frame."
                }),
                "viz_mode": (_VIZ_MODES, {
                    "default": "anaglyph",
                    "tooltip": (
                        "anaglyph (default): original→cyan, warped→red. Aligned = grayscale; misaligned = colored fringes. "
                        "Best for sub-pixel alignment check. "
                        "translucent: warped tinted + alpha-blended. "
                        "difference: |orig - warped| amplified. "
                        "checkerboard: alternating tiles. "
                        "edge_overlay: GPU Sobel edges of warped over original."
                    ),
                }),
                "viz_strength": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 10.0, "step": 0.05,
                    "tooltip": (
                        "Mode-dependent. translucent: alpha (0-1, default 0.5). "
                        "difference: gain (1-10, try 5-10 for subtle errors). "
                        "edge_overlay: edge alpha (0-1). "
                        "anaglyph/checkerboard: ignored."
                    ),
                }),
                "tint_color": (_TINT_NAMES, {
                    "default": "red",
                    "tooltip": "Color tint for translucent and edge_overlay modes."
                }),
                "checkerboard_tile_px": ("INT", {
                    "default": 64, "min": 8, "max": 256, "step": 8,
                    "tooltip": "Tile size in pixels for checkerboard viz mode. Smaller = more cuts = more visible tearing."
                }),
                "compute_phase_correlation": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Compute residual translation offset per frame via cv2.phaseCorrelate (Hann-windowed, masked-bbox cropped). "
                               "The most useful single number for warp drift detection."
                }),
                "face_swap_mode": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "When True (default): is_aligned gates ONLY on edge_IoU + phase_offset (geometry-truth metrics), "
                        "with looser thresholds appropriate for face/body swap output. PSNR/SSIM are still computed "
                        "and reported as informational, but NOT gated — they will always be low between two different "
                        "actors and would falsely fail is_aligned. "
                        "When False: strict mode — gates on PSNR/SSIM/edge_IoU/phase_offset all together. "
                        "Use False only when comparing the SAME content re-warped (same actor, same scene, just geometric correction)."
                    ),
                }),
                "print_summary": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Print summary table to stdout."
                }),
                "allow_silent_resize": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If False (default), raise an error when `original` and `warped` shapes differ. "
                               "If True, silently bilinear-resize `warped` to match — convenience, but can hide upstream wiring bugs."
                }),
            },
            "optional": {
                "mask": ("MASK", {
                    "tooltip": "SOURCE plate's stabilized subject mask (NOT the warped mask). "
                               "Strongly recommended — full-frame metrics are dominated by background pixels that don't matter. "
                               "Wiring the warped mask instead biases edge_IoU + phase-corr toward the warp's own answer."
                }),
                "edge_iou_threshold": ("FLOAT", {
                    "default": 0.10, "min": 0.01, "max": 0.50, "step": 0.01,
                    "tooltip": "Threshold for binarizing NORMALIZED Sobel edge magnitude (per-frame normalized by max). "
                               "0.10 = top-90% gradients counted as edges. Lower = looser; higher = stricter."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "BOOLEAN")
    RETURN_NAMES = ("viz", "metrics_json", "summary", "per_frame_csv", "is_aligned")
    FUNCTION = "check"
    CATEGORY = "NV_Utils/Debug"
    DESCRIPTION = (
        "Compare a warped/corrected image against an original crop. Outputs visualization + concrete numerical "
        "metrics (PSNR, SSIM, L1, RMSE, edge IoU, phase-correlation residual offset) + an is_aligned BOOLEAN "
        "for verifying the NV_PreCompWarp + mask suite pipeline alignment. Vectorized batched metrics; "
        "Hann-windowed phase correlation; adaptive edge thresholding."
    )

    # ===================================================================
    # Static helpers
    # ===================================================================

    @staticmethod
    def _luminance_batched(img_rgb):
        """[B, H, W, 3] tensor → [B, H, W] luminance via Rec.709 weights."""
        return img_rgb[..., 0] * 0.2126 + img_rgb[..., 1] * 0.7152 + img_rgb[..., 2] * 0.0722

    @staticmethod
    def _gaussian_kernel_2d(win, sigma, device, dtype):
        """Build a normalized 2D Gaussian kernel of shape [1, 1, win, win]."""
        coords = torch.arange(win, dtype=torch.float32, device=device) - (win // 2)
        g = torch.exp(-coords ** 2 / (2.0 * sigma ** 2))
        g = g / g.sum()
        kernel = (g.view(win, 1) * g.view(1, win)).view(1, 1, win, win).to(dtype)
        return kernel

    @classmethod
    def _ssim_batched(cls, a, b, win=11, sigma=1.5):
        """Gaussian-windowed reflect-padded SSIM on [B, H, W] luminance tensors. Returns [B, H, W] map."""
        device = a.device
        dtype = a.dtype
        kernel = cls._gaussian_kernel_2d(win, sigma, device, dtype)
        a4 = a.unsqueeze(1)  # [B, 1, H, W]
        b4 = b.unsqueeze(1)
        pad = win // 2
        # Reflect-pad once, conv with no padding — eliminates zero-padding bias near image borders.
        a_p = F.pad(a4, (pad, pad, pad, pad), mode="reflect")
        b_p = F.pad(b4, (pad, pad, pad, pad), mode="reflect")
        aa_p = F.pad(a4 * a4, (pad, pad, pad, pad), mode="reflect")
        bb_p = F.pad(b4 * b4, (pad, pad, pad, pad), mode="reflect")
        ab_p = F.pad(a4 * b4, (pad, pad, pad, pad), mode="reflect")
        mu_a = F.conv2d(a_p, kernel)
        mu_b = F.conv2d(b_p, kernel)
        mu_a2 = mu_a * mu_a
        mu_b2 = mu_b * mu_b
        mu_ab = mu_a * mu_b
        # Clamp variances at 0: floating-point noise can make these slightly negative,
        # which propagates into NaN through the SSIM denominator on edge cases.
        var_a = (F.conv2d(aa_p, kernel) - mu_a2).clamp(min=0.0)
        var_b = (F.conv2d(bb_p, kernel) - mu_b2).clamp(min=0.0)
        cov_ab = F.conv2d(ab_p, kernel) - mu_ab
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        ssim_map = ((2 * mu_ab + c1) * (2 * cov_ab + c2)) / ((mu_a2 + mu_b2 + c1) * (var_a + var_b + c2))
        return ssim_map.squeeze(1)  # [B, H, W]

    @staticmethod
    def _sobel_magnitude_batched(img_lum):
        """[B, H, W] → [B, H, W] Sobel gradient magnitude with reflect padding."""
        device = img_lum.device
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
        a = img_lum.unsqueeze(1).float()  # [B, 1, H, W]
        a_p = F.pad(a, (1, 1, 1, 1), mode="reflect")
        gx = F.conv2d(a_p, kx)
        gy = F.conv2d(a_p, ky)
        return torch.sqrt(gx * gx + gy * gy).squeeze(1)  # [B, H, W]

    @staticmethod
    def _phase_correlate_hann(a_np, b_np):
        """cv2.phaseCorrelate with Hann window. Both [H, W] float32."""
        h, w = a_np.shape
        try:
            hann = cv2.createHanningWindow((w, h), cv2.CV_32F)
            (dx, dy), _resp = cv2.phaseCorrelate(
                a_np.astype(np.float32),
                b_np.astype(np.float32),
                window=hann,
            )
            return float(dx), float(dy)
        except Exception:
            return float("nan"), float("nan")

    @staticmethod
    def _percentile(arr, p):
        return float("nan") if len(arr) == 0 else float(np.percentile(arr, p))

    # ===================================================================
    # Main entry
    # ===================================================================

    def check(self, original, warped, viz_mode, viz_strength, tint_color,
              checkerboard_tile_px, compute_phase_correlation, face_swap_mode,
              print_summary, allow_silent_resize, mask=None, edge_iou_threshold=0.10):

        # ---- shape / channel validation ----
        if original.ndim != 4 or warped.ndim != 4:
            raise ValueError(
                f"[NV_WarpAlignmentCheck] expected IMAGE [B,H,W,C], got "
                f"original={tuple(original.shape)}, warped={tuple(warped.shape)}"
            )

        if original.shape[-1] not in (3, 4) or warped.shape[-1] not in (3, 4):
            raise ValueError(
                f"[NV_WarpAlignmentCheck] expected 3- or 4-channel IMAGE, got "
                f"C_original={original.shape[-1]}, C_warped={warped.shape[-1]}. "
                f"Convert grayscale to 3-channel upstream."
            )

        # Slice to RGB for all metric/viz computation; alpha is silently dropped.
        original_rgb = original[..., :3].contiguous()
        warped_rgb = warped[..., :3].contiguous()

        # Resize warped if shape mismatch (gated by allow_silent_resize).
        if original_rgb.shape != warped_rgb.shape:
            if not allow_silent_resize:
                raise ValueError(
                    f"[NV_WarpAlignmentCheck] shape mismatch: original={tuple(original_rgb.shape)}, "
                    f"warped={tuple(warped_rgb.shape)}. Set allow_silent_resize=True to bilinear-resize warped, "
                    f"or fix upstream wiring."
                )
            print(f"[NV_WarpAlignmentCheck] WARNING: resizing warped {tuple(warped_rgb.shape)} → {tuple(original_rgb.shape)}")
            w_chw = warped_rgb.permute(0, 3, 1, 2)
            w_chw = F.interpolate(w_chw, size=(original_rgb.shape[1], original_rgb.shape[2]),
                                  mode="bilinear", align_corners=False)
            warped_rgb = w_chw.permute(0, 2, 3, 1).contiguous()

        B, H, W, _ = original_rgb.shape
        device = original_rgb.device

        # Promote and resize mask if provided.
        mask_3d = self._prepare_mask(mask, B, H, W, device)

        # ---- vectorized batched metrics ----
        # Cast once. Everything below stays on-device until the very last `.cpu().numpy()` call.
        orig_f = original_rgb.float()
        warp_f = warped_rgb.float()

        # Luminance
        lum_o = self._luminance_batched(orig_f)  # [B, H, W]
        lum_w = self._luminance_batched(warp_f)

        # Active mask per frame: m > 0.05; fallback to full-frame for frames with too-small masks.
        if mask_3d is not None:
            m_active = (mask_3d > 0.05)  # [B, H, W] bool
            per_frame_area = m_active.view(B, -1).sum(dim=1)  # [B]
            fallback_per_frame = (per_frame_area < 16)  # [B] bool
            # For fallback frames, use full-frame mask (all True).
            full_true = torch.ones((H, W), dtype=torch.bool, device=device)
            eff_mask = torch.where(
                fallback_per_frame.view(B, 1, 1).expand(-1, H, W),
                full_true.unsqueeze(0).expand(B, -1, -1),
                m_active,
            )
        else:
            eff_mask = torch.ones((B, H, W), dtype=torch.bool, device=device)
            fallback_per_frame = torch.zeros(B, dtype=torch.bool, device=device)

        eff_mask_f = eff_mask.float()
        per_frame_count = eff_mask_f.view(B, -1).sum(dim=1).clamp(min=1.0)  # [B]

        # L1 / MSE / RMSE / PSNR (all [B] tensors)
        diff_per_pix = (orig_f - warp_f).abs().mean(dim=-1)            # [B, H, W]
        sq_per_pix = ((orig_f - warp_f) ** 2).mean(dim=-1)             # [B, H, W]
        l1_per_frame = (diff_per_pix * eff_mask_f).view(B, -1).sum(dim=1) / per_frame_count
        mse_per_frame = (sq_per_pix * eff_mask_f).view(B, -1).sum(dim=1) / per_frame_count
        rmse_per_frame = mse_per_frame.clamp(min=0.0).sqrt()
        psnr_per_frame = torch.where(
            mse_per_frame < 1e-10,
            torch.full_like(mse_per_frame, 99.0),
            10.0 * torch.log10(1.0 / mse_per_frame.clamp(min=1e-12)),
        )

        # SSIM batched (Gaussian, reflect-padded)
        ssim_map = self._ssim_batched(lum_o, lum_w, win=11, sigma=1.5)
        ssim_per_frame = (ssim_map * eff_mask_f).view(B, -1).sum(dim=1) / per_frame_count

        # Edge IoU batched, with per-frame adaptive thresholding. Normalize Sobel magnitudes
        # by their per-frame max INSIDE the effective mask — using whole-frame max would let
        # bright background edges compress in-mask gradients and depress IoU artificially.
        edge_o = self._sobel_magnitude_batched(lum_o)  # [B, H, W]
        edge_w = self._sobel_magnitude_batched(lum_w)
        edge_o_masked = edge_o.masked_fill(~eff_mask, 0.0)
        edge_w_masked = edge_w.masked_fill(~eff_mask, 0.0)
        max_o = edge_o_masked.view(B, -1).max(dim=1).values.clamp(min=1e-5).view(B, 1, 1)
        max_w = edge_w_masked.view(B, -1).max(dim=1).values.clamp(min=1e-5).view(B, 1, 1)
        e_o_bin = (edge_o / max_o) > edge_iou_threshold
        e_w_bin = (edge_w / max_w) > edge_iou_threshold
        inter_count = (e_o_bin & e_w_bin & eff_mask).view(B, -1).sum(dim=1).float()
        union_count = ((e_o_bin | e_w_bin) & eff_mask).view(B, -1).sum(dim=1).float()
        edge_iou_per_frame = inter_count / union_count.clamp(min=1.0)

        # Single CPU transfer for all batched metrics.
        l1_np = l1_per_frame.detach().cpu().numpy()
        rmse_np = rmse_per_frame.detach().cpu().numpy()
        psnr_np = psnr_per_frame.detach().cpu().numpy()
        ssim_np = ssim_per_frame.detach().cpu().numpy()
        edge_iou_np = edge_iou_per_frame.detach().cpu().numpy()
        fallback_np = fallback_per_frame.detach().cpu().numpy()

        # ---- phase correlation (per-frame, CPU-bound, but with Hann window + masked bbox) ----
        phase_dx = [None] * B
        phase_dy = [None] * B
        if compute_phase_correlation:
            # Move luminance batches to CPU once. Faster than per-frame transfers.
            lum_o_cpu = lum_o.detach().cpu().numpy()
            lum_w_cpu = lum_w.detach().cpu().numpy()
            mask_cpu_bool = m_active.detach().cpu().numpy() if mask_3d is not None else None

            for i in range(B):
                a_np = lum_o_cpu[i]
                b_np = lum_w_cpu[i]
                if mask_3d is not None and not bool(fallback_np[i]):
                    m_i = mask_cpu_bool[i]
                    rows = np.where(m_i.any(axis=1))[0]
                    cols = np.where(m_i.any(axis=0))[0]
                    if rows.size and cols.size:
                        y0, y1 = int(rows[0]), int(rows[-1]) + 1
                        x0, x1 = int(cols[0]), int(cols[-1]) + 1
                        ch, cw = y1 - y0, x1 - x0
                        active_in_bbox = int(m_i[y0:y1, x0:x1].sum())
                        if (ch >= _PHASE_MIN_BBOX_DIM and cw >= _PHASE_MIN_BBOX_DIM
                                and active_in_bbox >= _PHASE_MIN_ACTIVE_PX):
                            # Pass raw crop to phaseCorrelate. Multiplying by a hard binary
                            # mask creates a razor-sharp edge the FFT latches onto, biasing
                            # dx/dy toward (0, 0). The Hann window applied inside
                            # _phase_correlate_hann smoothly fades the borders without that
                            # artifact.
                            crop_o = a_np[y0:y1, x0:x1]
                            crop_w = b_np[y0:y1, x0:x1]
                            dx, dy = self._phase_correlate_hann(crop_o, crop_w)
                            phase_dx[i] = None if np.isnan(dx) else round(dx, 3)
                            phase_dy[i] = None if np.isnan(dy) else round(dy, 3)
                            continue
                # Fallback: full-frame phase correlation (still Hann-windowed).
                if a_np.shape[0] >= _PHASE_MIN_BBOX_DIM and a_np.shape[1] >= _PHASE_MIN_BBOX_DIM:
                    dx, dy = self._phase_correlate_hann(a_np, b_np)
                    phase_dx[i] = None if np.isnan(dx) else round(dx, 3)
                    phase_dy[i] = None if np.isnan(dy) else round(dy, 3)

        # ---- assemble per-frame entries ----
        per_frame = []
        for i in range(B):
            per_frame.append({
                "frame": i,
                "psnr_db": round(float(psnr_np[i]), 3),
                "ssim": round(float(ssim_np[i]), 4),
                "l1": round(float(l1_np[i]), 5),
                "rmse": round(float(rmse_np[i]), 5),
                "edge_iou": round(float(edge_iou_np[i]), 4),
                "phase_dx_px": phase_dx[i],
                "phase_dy_px": phase_dy[i],
                "mask_used": "fallback_full_frame" if bool(fallback_np[i]) else ("mask" if mask_3d is not None else "full_frame"),
            })

        # ---- aggregate (with correct frame-index tracking for phase_offset) ----
        agg = self._aggregate(per_frame, compute_phase_correlation)

        # ---- sanity check: structurally unrelated inputs? ----
        sanity_warning = None
        if "edge_iou" in agg and agg["edge_iou"]["mean"] < 0.05:
            sanity_warning = (
                "WARNING: mean edge_IoU < 0.05 — original and warped may not depict the same content. "
                "Check that you wired the SOURCE plate crop to `original` and the WARPED render to `warped`."
            )
        elif "phase_offset_magnitude_px" in agg and agg["phase_offset_magnitude_px"]["max"] > 50.0:
            sanity_warning = (
                f"WARNING: peak phase offset {agg['phase_offset_magnitude_px']['max']:.1f} px — extreme "
                f"misalignment on at least one frame. Verify wiring: original = source plate crop, "
                f"warped = warped_render from NV_PreCompWarp."
            )

        # Aggregate fallback count
        fallback_count = int(fallback_per_frame.sum().item())
        if fallback_count > 0 and mask_3d is not None:
            fallback_warning = f"WARNING: {fallback_count}/{B} frames used fallback_full_frame because mask area was <16 px."
        else:
            fallback_warning = None

        # ---- is_aligned BOOLEAN (the gatekeeper output) + gate_eval explainability ----
        is_aligned, gate_eval = self._compute_is_aligned(agg, face_swap_mode)

        # ---- viz ----
        viz = self._build_viz(original_rgb, warped_rgb, viz_mode, viz_strength, tint_color, checkerboard_tile_px)

        # ---- format outputs ----
        # Thresholds the gate actually used (mode-dependent).
        if face_swap_mode:
            thresholds_used = {
                "mode": "face_swap",
                "edge_iou_mean_min": _THRESH_EDGE_IOU_FACE_SWAP,
                "phase_offset_p90_max": _THRESH_PHASE_P90_FACE_SWAP,
                "psnr_ssim_gated": False,
            }
        else:
            thresholds_used = {
                "mode": "strict",
                "psnr_median_min": _THRESH_PSNR_MEDIAN,
                "ssim_mean_min": _THRESH_SSIM_MEAN,
                "edge_iou_mean_min": _THRESH_EDGE_IOU,
                "phase_offset_p90_max": _THRESH_PHASE_P90,
                "psnr_ssim_gated": True,
            }

        metrics_json = json.dumps({
            "frames": B,
            "mask_used": "mask" if mask_3d is not None else "full_frame",
            "fallback_frame_count": fallback_count,
            "viz_mode": viz_mode,
            "edge_iou_threshold": edge_iou_threshold,
            "face_swap_mode": face_swap_mode,
            "is_aligned": is_aligned,
            "thresholds": thresholds_used,
            "gate_eval": gate_eval,
            "warnings": [w for w in (sanity_warning, fallback_warning) if w is not None],
            "aggregate": agg,
            "per_frame": per_frame,
        }, indent=2)
        per_frame_csv = self._make_csv(per_frame)
        summary = self._make_summary(B, agg, mask_3d is not None, viz_mode, edge_iou_threshold,
                                     is_aligned, fallback_count, sanity_warning, fallback_warning,
                                     face_swap_mode, gate_eval)

        if print_summary:
            print(f"\n[NV_WarpAlignmentCheck]\n{summary}\n")

        return (viz, metrics_json, summary, per_frame_csv, is_aligned)

    # ===================================================================
    # Mask prep
    # ===================================================================

    @staticmethod
    def _prepare_mask(mask, B, H, W, device):
        if mask is None:
            return None
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        elif mask.ndim != 3:
            raise ValueError(f"[NV_WarpAlignmentCheck] mask must be 2D or 3D, got {mask.ndim}D shape={tuple(mask.shape)}")
        if mask.shape[0] == 1 and B > 1:
            mask = mask.expand(B, -1, -1)
        elif mask.shape[0] != B:
            raise ValueError(f"[NV_WarpAlignmentCheck] mask batch {mask.shape[0]} != image batch {B}")
        if tuple(mask.shape[1:]) != (H, W):
            m4 = mask.unsqueeze(1).float()
            m4 = F.interpolate(m4, size=(H, W), mode="nearest")
            mask = m4.squeeze(1)
        return mask.to(device).float()

    # ===================================================================
    # Aggregation (with correct frame-index tracking for phase_offset)
    # ===================================================================

    def _aggregate(self, per_frame, compute_phase_correlation):
        agg = {}

        def numeric_col(key):
            return [
                f[key] for f in per_frame
                if f[key] is not None and not (isinstance(f[key], float) and np.isnan(f[key]))
            ]

        for key in ("psnr_db", "ssim", "l1", "rmse", "edge_iou"):
            vals = numeric_col(key)
            if not vals:
                continue
            arr = np.array(vals, dtype=np.float64)
            agg[key] = {
                "min": round(float(arr.min()), 5),
                "p10": round(self._percentile(vals, 10), 5),
                "p50": round(self._percentile(vals, 50), 5),
                "mean": round(float(arr.mean()), 5),
                "max": round(float(arr.max()), 5),
                "std": round(float(arr.std()), 5),
            }
            similarity_metric = key in ("psnr_db", "ssim", "edge_iou")
            raw = [f[key] for f in per_frame]
            agg[key]["worst_frame"] = int(np.argmin(raw)) if similarity_metric else int(np.argmax(raw))

        if compute_phase_correlation:
            # Carry frame indices alongside values to fix the filtered-list worst_frame bug.
            valid = [
                (f["frame"], f["phase_dx_px"], f["phase_dy_px"])
                for f in per_frame
                if f["phase_dx_px"] is not None and f["phase_dy_px"] is not None
            ]
            if valid:
                mags = [(idx, float(np.sqrt(dx * dx + dy * dy))) for idx, dx, dy in valid]
                worst_idx, _ = max(mags, key=lambda t: t[1])
                magnitudes = [m for _, m in mags]
                agg["phase_offset_magnitude_px"] = {
                    "min": round(float(np.min(magnitudes)), 3),
                    "p50": round(self._percentile(magnitudes, 50), 3),
                    "p90": round(self._percentile(magnitudes, 90), 3),
                    "mean": round(float(np.mean(magnitudes)), 3),
                    "max": round(float(np.max(magnitudes)), 3),
                    "std": round(float(np.std(magnitudes)), 3),
                    "worst_frame": worst_idx,
                    "valid_frames": len(valid),
                }
                dx_only = [dx for _, dx, _ in valid]
                dy_only = [dy for _, _, dy in valid]
                agg["phase_dx_px"] = {
                    "min": round(float(np.min(dx_only)), 3),
                    "mean": round(float(np.mean(dx_only)), 3),
                    "max": round(float(np.max(dx_only)), 3),
                }
                agg["phase_dy_px"] = {
                    "min": round(float(np.min(dy_only)), 3),
                    "mean": round(float(np.mean(dy_only)), 3),
                    "max": round(float(np.max(dy_only)), 3),
                }

        return agg

    # ===================================================================
    # is_aligned gate
    # ===================================================================

    @staticmethod
    def _compute_is_aligned(agg, face_swap_mode):
        """Gate on alignment-truth metrics. Returns (is_aligned: bool, gate_eval: dict).

        gate_eval is a per-gate explainability block with structure:
            {gate_name: {"value": float|None, "threshold": float, "passed": bool, "present": bool}, ...}
        Distinguishes "metric missing" (present=False) from "metric measured but failed"
        (present=True, passed=False) so users can debug why is_aligned tripped.

        face_swap_mode=False (strict, same-content re-warp):
            PSNR p50 >= 27 AND SSIM mean >= 0.85 AND edge_IoU mean >= 0.40 AND phase_offset p90 <= 2.

        face_swap_mode=True (default — face/body swap with different actor):
            edge_IoU mean >= 0.20 AND phase_offset p90 <= 5.0.
            PSNR/SSIM are NOT gated — different people produce inherently low pixel similarity
            regardless of how good the warp is. Only geometric alignment is gated.
        """
        gate_eval = {"mode": "face_swap" if face_swap_mode else "strict"}

        def _gate(metric_key, value_path, threshold, comparator):
            present = metric_key in agg and agg[metric_key].get(value_path) is not None
            value = float(agg[metric_key][value_path]) if present else None
            passed = comparator(value, threshold) if present else False
            return {"value": value, "threshold": float(threshold), "passed": bool(passed), "present": bool(present)}

        ge = lambda a, b: a >= b
        le = lambda a, b: a <= b

        if face_swap_mode:
            gate_eval["edge_iou_mean"] = _gate("edge_iou", "mean", _THRESH_EDGE_IOU_FACE_SWAP, ge)
            # Phase gate is optional — soft-pass if metric absent (e.g., compute_phase_correlation=False).
            if "phase_offset_magnitude_px" in agg:
                gate_eval["phase_offset_p90"] = _gate("phase_offset_magnitude_px", "p90", _THRESH_PHASE_P90_FACE_SWAP, le)
            else:
                gate_eval["phase_offset_p90"] = {"value": None, "threshold": _THRESH_PHASE_P90_FACE_SWAP,
                                                  "passed": True, "present": False}
            is_aligned = gate_eval["edge_iou_mean"]["passed"] and gate_eval["phase_offset_p90"]["passed"]
        else:
            gate_eval["psnr_p50"] = _gate("psnr_db", "p50", _THRESH_PSNR_MEDIAN, ge)
            gate_eval["ssim_mean"] = _gate("ssim", "mean", _THRESH_SSIM_MEAN, ge)
            gate_eval["edge_iou_mean"] = _gate("edge_iou", "mean", _THRESH_EDGE_IOU, ge)
            if "phase_offset_magnitude_px" in agg:
                gate_eval["phase_offset_p90"] = _gate("phase_offset_magnitude_px", "p90", _THRESH_PHASE_P90, le)
            else:
                gate_eval["phase_offset_p90"] = {"value": None, "threshold": _THRESH_PHASE_P90,
                                                  "passed": True, "present": False}
            is_aligned = (gate_eval["psnr_p50"]["passed"] and gate_eval["ssim_mean"]["passed"]
                          and gate_eval["edge_iou_mean"]["passed"] and gate_eval["phase_offset_p90"]["passed"])

        gate_eval["overall"] = bool(is_aligned)
        return bool(is_aligned), gate_eval

    # ===================================================================
    # Visualization builders
    # ===================================================================

    def _build_viz(self, original, warped, mode, strength, tint, checker_tile):
        if mode == "anaglyph":
            return self._viz_anaglyph(original, warped)
        if mode == "translucent":
            return self._viz_translucent(original, warped, strength, tint)
        if mode == "difference":
            return self._viz_difference(original, warped, strength)
        if mode == "checkerboard":
            return self._viz_checkerboard(original, warped, checker_tile)
        if mode == "edge_overlay":
            return self._viz_edge_overlay(original, warped, strength, tint)
        raise ValueError(f"[NV_WarpAlignmentCheck] unknown viz_mode: {mode!r}")

    def _viz_anaglyph(self, orig, warped):
        # original → cyan (G+B), warped → red. Aligned → grayscale; misaligned → colored fringes.
        lum_o = self._luminance_batched(orig)
        lum_w = self._luminance_batched(warped)
        out = torch.zeros_like(orig)
        out[..., 0] = lum_w
        out[..., 1] = lum_o
        out[..., 2] = lum_o
        return out.clamp(0, 1)

    def _viz_translucent(self, orig, warped, alpha, tint_name):
        tint = torch.tensor(_TINT_COLORS[tint_name], device=orig.device, dtype=orig.dtype)
        lum_w = self._luminance_batched(warped).unsqueeze(-1)
        tinted_w = lum_w * tint  # [B, H, W, 3]
        a = max(0.0, min(1.0, float(alpha)))
        return (orig * (1.0 - a) + tinted_w * a).clamp(0, 1)

    def _viz_difference(self, orig, warped, gain):
        diff_lum = (orig - warped).abs().mean(dim=-1, keepdim=True)
        amplified = (diff_lum * float(gain)).clamp(0, 1)
        return amplified.expand(-1, -1, -1, 3)

    def _viz_checkerboard(self, orig, warped, tile):
        B, H, W, _ = orig.shape
        device = orig.device
        ys = (torch.arange(H, device=device) // tile).view(H, 1)
        xs = (torch.arange(W, device=device) // tile).view(1, W)
        cb = ((ys + xs) % 2 == 0).float().unsqueeze(0).unsqueeze(-1)  # [1, H, W, 1]
        return (orig * cb + warped * (1.0 - cb)).clamp(0, 1)

    def _viz_edge_overlay(self, orig, warped, alpha, tint_name):
        # GPU Sobel + threshold instead of cv2.Canny — fully batched, no CPU transfer.
        tint = torch.tensor(_TINT_COLORS[tint_name], device=orig.device, dtype=orig.dtype)
        a = max(0.0, min(1.0, float(alpha)))
        lum_w = self._luminance_batched(warped)
        edges = self._sobel_magnitude_batched(lum_w)
        # Per-frame normalization so the overlay reads on both flat and contrasty footage.
        max_per_frame = edges.view(edges.shape[0], -1).max(dim=1).values.clamp(min=1e-5).view(-1, 1, 1)
        edges_norm = (edges / max_per_frame).clamp(0, 1)
        # Threshold to binarize, then soft-falloff for cleaner viz.
        e = (edges_norm > 0.20).float().unsqueeze(-1)  # [B, H, W, 1]
        tint_full = tint.view(1, 1, 1, 3)
        return (orig * (1.0 - e * a) + tint_full * (e * a)).clamp(0, 1)

    # ===================================================================
    # Output formatters
    # ===================================================================

    @staticmethod
    def _make_csv(per_frame):
        if not per_frame:
            return ""
        keys = list(per_frame[0].keys())
        lines = [",".join(keys)]
        for f in per_frame:
            row = []
            for k in keys:
                v = f[k]
                if v is None:
                    row.append("")
                elif isinstance(v, float):
                    row.append(f"{v:.5f}")
                else:
                    row.append(str(v))
            lines.append(",".join(row))
        return "\n".join(lines)

    @staticmethod
    def _make_summary(B, agg, mask_used, viz_mode, edge_thresh, is_aligned, fallback_count,
                      sanity_warning, fallback_warning, face_swap_mode, gate_eval):
        bar = "=" * 84
        verdict = "ALIGNED ✓" if is_aligned else "MISALIGNED ✗"
        if face_swap_mode:
            gate_str = (f"face_swap_mode | gate: edge_IoU mean ≥ {_THRESH_EDGE_IOU_FACE_SWAP}, "
                        f"phase p90 ≤ {_THRESH_PHASE_P90_FACE_SWAP} px (PSNR/SSIM informational only)")
        else:
            gate_str = (f"strict mode | gate: PSNR p50 ≥ {_THRESH_PSNR_MEDIAN}, SSIM mean ≥ {_THRESH_SSIM_MEAN}, "
                        f"edge_IoU ≥ {_THRESH_EDGE_IOU}, phase p90 ≤ {_THRESH_PHASE_P90} px")
        lines = [
            bar,
            f" NV_WarpAlignmentCheck — {B} frames | mask={'YES' if mask_used else 'no (full-frame)'} | viz={viz_mode}",
            f" VERDICT: {verdict}",
            f"   {gate_str}",
        ]
        # Per-gate explainability — surfaces exactly which gate(s) tripped.
        gate_lines = []
        for gate_name, gate_data in gate_eval.items():
            if gate_name in ("mode", "overall"):
                continue
            if not gate_data.get("present", False):
                gate_lines.append(f"   {gate_name:<16s} {'(metric not computed)':<30s} {'soft-pass' if gate_data['passed'] else 'FAIL':<10s}")
            else:
                v = gate_data["value"]
                t = gate_data["threshold"]
                status = "✓" if gate_data["passed"] else "✗"
                gate_lines.append(f"   {gate_name:<16s} value={v:>8.4f}  threshold={t:>6.3f}  {status}")
        if gate_lines:
            lines.append("   GATE EVALUATION:")
            lines.extend(gate_lines)
        lines.append(bar)

        if sanity_warning:
            lines.append(f"  ⚠ {sanity_warning}")
        if fallback_warning:
            lines.append(f"  ⚠ {fallback_warning}")
        if sanity_warning or fallback_warning:
            lines.append("")

        def fmt(label, key, unit=""):
            if key not in agg:
                return None
            a = agg[key]
            return (
                f"  {label:<14s} min={a['min']:>8.4f} p10={a['p10']:>8.4f} p50={a['p50']:>8.4f}"
                f" mean={a['mean']:>8.4f} max={a['max']:>8.4f} std={a['std']:>7.4f}"
                f"  worst@frame={a['worst_frame']}{unit}"
            )

        for label, key in (("PSNR (dB)", "psnr_db"), ("SSIM", "ssim"), ("edge_IoU", "edge_iou"),
                          ("L1", "l1"), ("RMSE", "rmse")):
            row = fmt(label, key)
            if row:
                lines.append(row)

        if "phase_offset_magnitude_px" in agg:
            a = agg["phase_offset_magnitude_px"]
            lines.append(
                f"  {'phase_offset':<14s} min={a['min']:>8.3f} p10=    n/a    p50={a['p50']:>8.3f}"
                f" mean={a['mean']:>8.3f} max={a['max']:>8.3f} std={a['std']:>7.3f}"
                f"  worst@frame={a['worst_frame']}  (px)"
            )
            lines.append(f"                 p90={a['p90']:.3f}  (gate: <{_THRESH_PHASE_P90} px)  valid_frames={a['valid_frames']}")
            if "phase_dx_px" in agg:
                d = agg["phase_dx_px"]
                lines.append(f"  {'  dx (signed)':<14s} mean={d['mean']:>+8.3f} min={d['min']:>+8.3f} max={d['max']:>+8.3f}  (px)")
            if "phase_dy_px" in agg:
                d = agg["phase_dy_px"]
                lines.append(f"  {'  dy (signed)':<14s} mean={d['mean']:>+8.3f} min={d['min']:>+8.3f} max={d['max']:>+8.3f}  (px)")

        lines.append("")
        lines.append(" Reading guide (recalibrated for low-denoise face/body swap pipeline):")
        lines.append(f"   PSNR median > {_THRESH_PSNR_MEDIAN} dB     = visually aligned (PSNR is sensitive to appearance, not just geometry)")
        lines.append(f"   SSIM mean > {_THRESH_SSIM_MEAN}         = structurally aligned")
        lines.append(f"   edge_IoU mean > {_THRESH_EDGE_IOU}      = boundaries line up (adaptive threshold={edge_thresh})")
        lines.append(f"   |phase_offset| p90 < {_THRESH_PHASE_P90} px = no systematic translation drift across the clip")
        lines.append(f"   |phase_offset| > 2 px on a frame  = check anchor placement at worst_frame")
        if fallback_count:
            lines.append(f"   {fallback_count} frame(s) used fallback_full_frame — your mask may be too tight on those frames.")
        lines.append(bar)
        return "\n".join(lines)


NODE_CLASS_MAPPINGS = {
    "NV_WarpAlignmentCheck": NV_WarpAlignmentCheck,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_WarpAlignmentCheck": "NV Warp Alignment Check",
}
