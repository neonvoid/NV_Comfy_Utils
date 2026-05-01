"""
NV Keyframe Sampler — Pick K still frames adjacent to motion changes.

Selects keyframes for VLM landmark detection (NV_VLMLandmarkCorresponder) on
already-cropped video. The goal is to expose new pose configurations as the actor
moves, while always seeding the VLM with sharp non-blurry frames.

Why valleys, not peaks: motion peaks coincide with motion blur — the worst place
to seed Shi-Tomasi corner detection or CoTracker3 trajectories. Instead we detect
motion CHANGES (high gradient of the motion curve) and pick the nearest local
MINIMUM of motion within a forward-search window — i.e. the "hold" frame just
after the actor reached a new pose configuration but before they moved again.

Designed for v1.2 of the VLM-driven CoTracker anchor pipeline (see research handoff
2026-05-01_vlm_cotracker_anchor_architecture.md). Requires already-cropped frames
(post InpaintCrop2) so background/camera motion doesn't dominate the curve.
"""

import json
import time

import cv2
import numpy as np
import torch


def _laplacian_variance(gray_u8):
    """Laplacian variance — higher = sharper. Used as blur metric."""
    return float(cv2.Laplacian(gray_u8, cv2.CV_64F).var())


def _trimmed_mean(arr, top_drop_frac=0.10):
    """Upper-percentile-trimmed mean: drop top `top_drop_frac` outliers, take mean.

    Robust to crop-jitter / cloth-flutter / single-pixel outliers that would
    dominate a plain mean of optical-flow magnitude. Codex review note.
    """
    if arr.size == 0:
        return 0.0
    threshold = np.quantile(arr, 1.0 - top_drop_frac)
    kept = arr[arr <= threshold]
    if kept.size == 0:
        return float(arr.mean())
    return float(kept.mean())


def _smooth_1d(arr, window):
    """Centered moving-average smoothing with edge replication."""
    if window < 2 or arr.size < window:
        return arr.copy()
    pad = window // 2
    padded = np.pad(arr, pad, mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(padded, kernel, mode="valid")[: arr.size]


class NV_KeyframeSampler:
    """Pick K still hold-frames adjacent to motion changes for VLM landmark detection."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": (
                        "ALREADY-CROPPED video frames [B,H,W,3] from InpaintCrop2. "
                        "Background/camera motion will dominate the motion curve if "
                        "you pass uncropped video — make sure the subject fills the crop."
                    ),
                }),
                "num_keyframes": ("INT", {
                    "default": 5, "min": 2, "max": 16, "step": 1,
                    "tooltip": (
                        "Number of keyframes to sample. 5 is a good default for the VLM "
                        "anchor-detection pipeline — covers frame 0, end frame, and 3 "
                        "mid-sequence holds."
                    ),
                }),
                "min_spacing_frames": ("INT", {
                    "default": 8, "min": 2, "max": 64, "step": 1,
                    "tooltip": (
                        "Minimum frames between selected keyframes (NMS spacing). Prevents "
                        "selecting near-duplicate poses. Also bounds the forward-search "
                        "window when finding the post-change hold."
                    ),
                }),
            },
            "optional": {
                "flow_downscale": ("INT", {
                    "default": 4, "min": 1, "max": 16, "step": 1,
                    "tooltip": (
                        "Downscale factor for Farneback flow computation (4 = quarter res). "
                        "Higher = faster + more robust to high-frequency noise. 4 is fine "
                        "for 1024² crops; use 2 on 256² crops."
                    ),
                }),
                "blur_reject_percentile": ("FLOAT", {
                    "default": 25.0, "min": 0.0, "max": 50.0, "step": 5.0,
                    "tooltip": (
                        "Reject frames below this percentile of Laplacian variance "
                        "(per-clip blur metric). 25.0 = drop the bottom 25% blurriest "
                        "frames. 0.0 = no blur rejection."
                    ),
                }),
                "verbose_debug": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Print per-step diagnostics: motion curve stats, candidate frames, rejections.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("keyframe_indices", "motion_curve", "info")
    FUNCTION = "sample"
    CATEGORY = "NV_Utils/Tracking"
    DESCRIPTION = (
        "Pick K still hold-frames adjacent to motion changes for VLM landmark detection. "
        "Chooses post-change motion valleys (sharp frames just after pose changes) — never "
        "motion peaks (which coincide with motion blur). Designed for the VLM-driven "
        "CoTracker3 anchor pipeline. Requires already-cropped video (post InpaintCrop2)."
    )

    def sample(self, image, num_keyframes, min_spacing_frames,
               flow_downscale=4, blur_reject_percentile=25.0, verbose_debug=False):
        t_start = time.perf_counter()
        T = image.shape[0]
        H = image.shape[1]
        W = image.shape[2]

        if T < 2:
            raise ValueError(
                f"[NV_KeyframeSampler] need at least 2 frames for motion analysis, got T={T}"
            )
        if num_keyframes > T:
            print(f"[NV_KeyframeSampler] num_keyframes={num_keyframes} > T={T}; "
                  f"capping to T")
            num_keyframes = T

        # --- Convert to grayscale uint8 numpy + downscale -------------------
        gray = (image[..., 0] * 0.299 + image[..., 1] * 0.587 + image[..., 2] * 0.114) * 255.0
        gray_np = gray.clamp(0.0, 255.0).to(torch.uint8).cpu().numpy()  # [T, H, W]
        del gray

        if flow_downscale > 1:
            small_h = max(8, H // flow_downscale)
            small_w = max(8, W // flow_downscale)
            gray_small = np.empty((T, small_h, small_w), dtype=np.uint8)
            for i in range(T):
                gray_small[i] = cv2.resize(gray_np[i], (small_w, small_h),
                                           interpolation=cv2.INTER_AREA)
        else:
            gray_small = gray_np

        # --- Laplacian variance per frame on FULL res (blur metric) --------
        blur_scores = np.zeros(T, dtype=np.float64)
        for i in range(T):
            blur_scores[i] = _laplacian_variance(gray_np[i])

        if blur_reject_percentile > 0.0:
            blur_threshold = np.percentile(blur_scores, blur_reject_percentile)
        else:
            blur_threshold = -1.0  # accept all

        # --- Per-frame motion magnitude via Farneback flow -----------------
        # Frame 0 has no predecessor — motion[0] = motion[1] for symmetry
        motion = np.zeros(T, dtype=np.float64)
        for i in range(1, T):
            flow = cv2.calcOpticalFlowFarneback(
                gray_small[i - 1], gray_small[i], None,
                pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                poly_n=5, poly_sigma=1.2, flags=0,
            )
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            motion[i] = _trimmed_mean(mag.flatten(), top_drop_frac=0.10)
        if T >= 2:
            motion[0] = motion[1]

        # Smooth motion curve over a 5–9 frame window (use 7 by default)
        smooth_window = min(7, T)
        if smooth_window % 2 == 0:
            smooth_window = max(1, smooth_window - 1)
        motion_smooth = _smooth_1d(motion, smooth_window)

        if verbose_debug:
            print(f"[NV_KeyframeSampler] motion stats: "
                  f"min={motion_smooth.min():.3f}, max={motion_smooth.max():.3f}, "
                  f"mean={motion_smooth.mean():.3f}")
            print(f"[NV_KeyframeSampler] blur reject threshold "
                  f"(p{blur_reject_percentile:.0f}) = {blur_threshold:.2f}")

        # --- Find motion-change points (high gradient of motion curve) -----
        if T >= 3:
            motion_grad = np.abs(np.gradient(motion_smooth))
        else:
            motion_grad = np.zeros_like(motion_smooth)
        # NMS-style local maxima of |gradient| with min_spacing separation
        change_points = self._local_maxima_nms(motion_grad, min_spacing_frames)

        # --- For each change point, find post-change hold (motion minimum) -
        candidate_holds = []
        for cp in change_points:
            hold = self._find_post_change_hold(
                cp, motion_smooth, blur_scores, blur_threshold,
                forward_window=min_spacing_frames, T=T,
            )
            if hold is not None:
                candidate_holds.append(hold)

        # --- Always include frame 0 and last frame (with blur fallback) ----
        # Endpoint search uses a small dedicated window (not min_spacing_frames,
        # which can be up to 64). Spec: "search ±3 frames for sharper neighbor"
        # for the blur fallback.
        endpoint_window = min(min_spacing_frames, 5)
        first_frame = self._find_acceptable_frame(
            0, motion_smooth, blur_scores, blur_threshold,
            forward_window=endpoint_window, T=T, direction="forward",
        )
        last_frame = self._find_acceptable_frame(
            T - 1, motion_smooth, blur_scores, blur_threshold,
            forward_window=endpoint_window, T=T, direction="backward",
        )

        # --- Seed selected list with required endpoints FIRST (Codex+Gemini review #1) ---
        # Spec item 7: "Always include frame 0 and last sharp low-motion frame."
        # If we mix endpoints into the candidate pool and NMS sorts by motion, a low-motion
        # hold near an endpoint can suppress the endpoint. Seed selected[] BEFORE NMS so
        # endpoints are guaranteed inclusion; NMS over remaining holds only.
        selected = []
        if first_frame is not None:
            selected.append(first_frame)
        if (last_frame is not None and last_frame != first_frame
                and (not selected or all(abs(last_frame - s) >= min_spacing_frames
                                          for s in selected))):
            selected.append(last_frame)

        # NMS over candidate holds, sorted by motion (lowest first = best hold)
        holds_sorted = sorted(set(candidate_holds), key=lambda i: motion_smooth[i])
        for cand in holds_sorted:
            if all(abs(cand - s) >= min_spacing_frames for s in selected):
                selected.append(cand)
            if len(selected) >= num_keyframes:
                break

        # --- Uniform-spacing fallback if fewer than K holds ----------------
        if len(selected) < num_keyframes:
            if verbose_debug:
                print(f"[NV_KeyframeSampler] only {len(selected)} holds found; "
                      f"running uniform-spacing fallback for remaining "
                      f"{num_keyframes - len(selected)} keyframes")
            selected = self._uniform_fallback(
                selected, num_keyframes, T, motion_smooth, blur_scores,
                blur_threshold, min_spacing_frames,
            )

        if len(selected) == 0:
            raise RuntimeError(
                f"[NV_KeyframeSampler] cannot find any acceptable keyframes "
                f"in this clip (T={T}, blur_threshold={blur_threshold:.2f}). "
                f"Clip may be too short, too blurry, or too motion-saturated. "
                f"Reduce blur_reject_percentile or improve source footage."
            )

        selected_sorted = sorted(selected)

        # --- Build outputs --------------------------------------------------
        elapsed = time.perf_counter() - t_start
        info_lines = [
            f"NV_KeyframeSampler: selected {len(selected_sorted)}/{num_keyframes} keyframes",
            f"  Indices: {selected_sorted}",
            f"  Per-keyframe motion: " + ", ".join(
                f"t={i} m={motion_smooth[i]:.2f} blur={blur_scores[i]:.0f}"
                for i in selected_sorted
            ),
            f"  Total frames: {T}, blur threshold: {blur_threshold:.2f}",
            f"  Elapsed: {elapsed * 1000:.0f} ms",
        ]
        info = "\n".join(info_lines)
        print(f"[NV_KeyframeSampler] {info}")

        keyframe_indices_json = json.dumps(selected_sorted)
        motion_curve_json = json.dumps([
            {"t": int(i), "motion": float(motion_smooth[i]), "blur": float(blur_scores[i])}
            for i in range(T)
        ])

        return (keyframe_indices_json, motion_curve_json, info)

    @staticmethod
    def _local_maxima_nms(arr, min_spacing):
        """Return indices of local maxima with at least min_spacing separation."""
        T = arr.size
        if T < 3:
            return []
        # All strict local maxima
        cands = []
        for i in range(1, T - 1):
            if arr[i] > arr[i - 1] and arr[i] > arr[i + 1]:
                cands.append(i)
        # Sort by descending arr value, NMS
        cands.sort(key=lambda i: -arr[i])
        kept = []
        for c in cands:
            if all(abs(c - k) >= min_spacing for k in kept):
                kept.append(c)
        return kept

    @staticmethod
    def _find_post_change_hold(change_idx, motion, blur, blur_threshold,
                               forward_window, T):
        """After change_idx, find the local minimum of motion within forward_window
        that ALSO passes blur threshold. Returns frame index or None."""
        end = min(T, change_idx + forward_window + 1)
        best_idx = None
        best_motion = float("inf")
        for i in range(change_idx, end):
            if blur[i] < blur_threshold:
                continue
            # Local minimum check (motion[i] <= neighbors within window)
            is_local_min = True
            if i > 0 and motion[i] > motion[i - 1]:
                is_local_min = False
            if i < T - 1 and motion[i] > motion[i + 1]:
                is_local_min = False
            if is_local_min and motion[i] < best_motion:
                best_motion = motion[i]
                best_idx = i
        # If no strict local minimum in window, accept the lowest motion in window
        if best_idx is None:
            for i in range(change_idx, end):
                if blur[i] < blur_threshold:
                    continue
                if motion[i] < best_motion:
                    best_motion = motion[i]
                    best_idx = i
        return best_idx

    @staticmethod
    def _find_acceptable_frame(start_idx, motion, blur, blur_threshold,
                               forward_window, T, direction):
        """Search for a low-motion sharp frame from start_idx in given direction.

        Returns the index of a frame that passes the blur threshold AND has lowest
        motion in the search window. Returns None if no qualifying frame found —
        the caller is responsible for raising an informative error rather than
        silently degrading to start_idx (per Codex review: spec forbids silent
        return of high-blur frames).

        direction='forward': search start_idx..start_idx+forward_window
        direction='backward': search start_idx..start_idx-forward_window
        """
        if not (0 <= start_idx < T):
            return None
        if direction == "forward":
            end = min(T, start_idx + forward_window + 1)
            indices = range(start_idx, end)
        else:
            end = max(-1, start_idx - forward_window - 1)
            indices = range(start_idx, end, -1)
        best_idx = None
        best_motion = float("inf")
        for i in indices:
            if blur[i] >= blur_threshold and motion[i] < best_motion:
                best_motion = motion[i]
                best_idx = i
        return best_idx  # may be None — caller decides how to handle

    @staticmethod
    def _uniform_fallback(existing, target_count, T, motion, blur,
                          blur_threshold, min_spacing):
        """Fill remaining slots via uniform spacing, but each candidate must be
        a low-motion sharp frame within ±5 (then ±10) of its uniform position.

        Per v1.2: never silently insert high-blur or high-motion frames.
        """
        K = target_count
        result = list(existing)
        # Generate uniform candidates
        if K > 1:
            uniform_targets = [int(round(i * (T - 1) / (K - 1))) for i in range(K)]
        else:
            uniform_targets = [T // 2]

        for target in uniform_targets:
            if any(abs(target - s) < min_spacing for s in result):
                continue
            # Search ±5 then ±10 for low-motion sharp frame
            picked = None
            for radius in (5, 10):
                best_motion = float("inf")
                for di in range(-radius, radius + 1):
                    j = target + di
                    if not (0 <= j < T):
                        continue
                    if blur[j] < blur_threshold:
                        continue
                    if any(abs(j - s) < min_spacing for s in result):
                        continue
                    if motion[j] < best_motion:
                        best_motion = motion[j]
                        picked = j
                if picked is not None:
                    break
            if picked is not None:
                result.append(picked)
            if len(result) >= K:
                break

        if len(result) < K:
            raise RuntimeError(
                f"[NV_KeyframeSampler] uniform-spacing fallback could not find "
                f"{K} sharp low-motion frames (only got {len(result)}). Clip may "
                f"be too short, too blurry, or too motion-saturated. Reduce "
                f"num_keyframes or improve source footage."
            )
        return result


NODE_CLASS_MAPPINGS = {
    "NV_KeyframeSampler": NV_KeyframeSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_KeyframeSampler": "NV Keyframe Sampler",
}
