"""
NV Temporal Stabilize

Temporal stabilization for control passes (depth, edge, canny, etc.)
Reduces frame-to-frame jitter that causes hallucination artifacts in V2V pipelines.

When using inferred control signals (depth estimation, edge detection, etc.) on recorded video,
per-frame inference models introduce temporal jitter. This node smooths the control signal
across time to prevent VACE from interpreting jitter as actual geometry/structure changes.

Example: A static tree in the scene gets slightly different depth values each frame from
a per-frame depth model. Without stabilization, VACE interprets the fluctuating depth as
the tree growing/morphing. Temporal median or EMA collapses these to stable values.

Modes:
    median   - Sliding window median. Best for removing jitter without drift.
               Each pixel's value is replaced by the median of its values across
               neighboring frames. Robust to outliers.
    ema      - Bidirectional exponential moving average. Forward + backward pass
               averaged to eliminate directional lag. Fast and smooth.
    gaussian - Temporal Gaussian blur. Smooth weighted average across time.
               Softer than median but can over-smooth real transitions.

Optional motion-aware mode:
    Connect the SOURCE video (recorded footage) to reference_video.
    The node detects which pixels are static vs moving in the source.
    Stabilization is only applied to static regions — moving regions
    keep their original control values (real depth/edge changes preserved).
"""

import torch
import torch.nn.functional as F


class NV_TemporalStabilize:
    """
    Temporal stabilization for control passes.

    Fixes depth hallucination, edge flicker, and other artifacts caused by
    frame-to-frame jitter in per-frame inference models.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("IMAGE", {
                    "tooltip": "Control pass video to stabilize (depth, edge, etc.) [T, H, W, C]"
                }),
                "mode": (["median", "ema", "gaussian"], {
                    "default": "median",
                    "tooltip": "median: removes jitter without drift (best default). "
                               "ema: smooth + fast. gaussian: smooth but can soften transitions."
                }),
                "window_size": ("INT", {
                    "default": 7, "min": 3, "max": 31, "step": 2,
                    "tooltip": "Temporal window size in frames. Larger = more smoothing. Must be odd."
                }),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Blend between original (0.0) and stabilized (1.0)"
                }),
            },
            "optional": {
                "reference_video": ("IMAGE", {
                    "tooltip": "Source video for motion detection. When provided, only static "
                               "regions get stabilized. Moving regions keep original values."
                }),
                "motion_threshold": ("FLOAT", {
                    "default": 0.03, "min": 0.001, "max": 0.2, "step": 0.005,
                    "tooltip": "Motion detection threshold (per-pixel avg frame diff). "
                               "Lower = more regions treated as static. "
                               "Only used when reference_video is connected."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("video", "motion_mask",)
    FUNCTION = "stabilize"
    CATEGORY = "NV_Utils"
    DESCRIPTION = (
        "Temporal stabilization for control passes (depth, edge, etc.). "
        "Fixes jitter from per-frame inference models that causes hallucination in V2V pipelines."
    )

    def stabilize(self, video, mode, window_size, strength,
                  reference_video=None, motion_threshold=0.03):
        T, H, W, C = video.shape

        # Early exit
        if T <= 1 or strength == 0.0:
            empty_mask = torch.zeros(1, H, W)
            return (video, empty_mask)

        # Ensure odd window, clamp to video length
        if window_size % 2 == 0:
            window_size += 1
        window_size = min(window_size, T)

        print(f"[NV_TemporalStabilize] Mode: {mode}, window: {window_size}, "
              f"strength: {strength}, frames: {T}")

        # Apply temporal smoothing
        if mode == "median":
            stabilized = self._temporal_median(video, window_size)
        elif mode == "ema":
            stabilized = self._temporal_ema(video, window_size)
        elif mode == "gaussian":
            stabilized = self._temporal_gaussian(video, window_size)
        else:
            stabilized = video

        # Motion-aware masking (optional)
        # motion_mask: 1.0 = moving (keep original), 0.0 = static (stabilize)
        motion_mask = torch.zeros(T, H, W)

        if reference_video is not None:
            motion_mask = self._compute_motion_mask(
                reference_video, motion_threshold, T, H, W
            )
            # Blend: static regions get stabilized, moving regions keep original
            mask_4d = motion_mask.unsqueeze(-1)  # [T, H, W, 1]
            stabilized = stabilized * (1.0 - mask_4d) + video * mask_4d

            static_pct = (motion_mask < 0.5).float().mean().item() * 100
            print(f"  Motion mask: {static_pct:.1f}% static, "
                  f"{100 - static_pct:.1f}% moving")

        # Final blend with strength
        result = video * (1.0 - strength) + stabilized * strength
        result = result.clamp(0.0, 1.0)

        return (result, motion_mask)

    def _temporal_median(self, video, window_size):
        """Sliding window median per pixel across time."""
        T, H, W, C = video.shape
        radius = window_size // 2
        result = torch.empty_like(video)

        for t in range(T):
            start = max(0, t - radius)
            end = min(T, t + radius + 1)
            window = video[start:end]  # [window, H, W, C]
            result[t] = window.median(dim=0).values

            if t % 50 == 0 and t > 0:
                print(f"  Median: {t}/{T} frames")

        print(f"  Median: {T}/{T} frames (done)")
        return result

    def _temporal_ema(self, video, window_size):
        """Bidirectional EMA — forward + backward averaged. No directional lag."""
        alpha = 2.0 / (window_size + 1)
        T = video.shape[0]

        # Forward pass
        forward = torch.empty_like(video)
        forward[0] = video[0]
        for t in range(1, T):
            forward[t] = alpha * video[t] + (1.0 - alpha) * forward[t - 1]

        # Backward pass
        backward = torch.empty_like(video)
        backward[-1] = video[-1]
        for t in range(T - 2, -1, -1):
            backward[t] = alpha * video[t] + (1.0 - alpha) * backward[t + 1]

        return (forward + backward) / 2.0

    def _temporal_gaussian(self, video, window_size):
        """Temporal Gaussian blur — weighted average across neighboring frames."""
        T = video.shape[0]
        radius = window_size // 2
        sigma = max(radius / 2.0, 0.5)

        # Build Gaussian kernel weights
        x = torch.arange(-radius, radius + 1, dtype=video.dtype, device=video.device)
        kernel = torch.exp(-x ** 2 / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()

        result = torch.empty_like(video)

        for t in range(T):
            weighted_sum = torch.zeros_like(video[0])
            weight_sum = 0.0

            for k in range(-radius, radius + 1):
                src_t = t + k
                if 0 <= src_t < T:
                    w = kernel[k + radius].item()
                    weighted_sum += video[src_t] * w
                    weight_sum += w

            result[t] = weighted_sum / weight_sum

            if t % 50 == 0 and t > 0:
                print(f"  Gaussian: {t}/{T} frames")

        print(f"  Gaussian: {T}/{T} frames (done)")
        return result

    def _compute_motion_mask(self, reference_video, threshold, target_T, target_H, target_W):
        """
        Compute per-pixel motion mask from reference video.
        Returns: [T, H, W] where 1.0 = moving, 0.0 = static.
        """
        ref_T, ref_H, ref_W, ref_C = reference_video.shape

        # Frame-to-frame absolute differences, averaged across channels
        diffs = (reference_video[1:] - reference_video[:-1]).abs()  # [T-1, H, W, C]
        diffs_mag = diffs.mean(dim=-1)  # [T-1, H, W]

        # Mean motion magnitude per pixel across all frame pairs
        avg_motion = diffs_mag.mean(dim=0)  # [H, W]

        # Handle resolution mismatch between reference and control video
        if ref_H != target_H or ref_W != target_W:
            avg_motion = avg_motion.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            avg_motion = F.interpolate(
                avg_motion, size=(target_H, target_W),
                mode='bilinear', align_corners=False
            )
            avg_motion = avg_motion.squeeze(0).squeeze(0)  # [H, W]

        # Soft threshold via sigmoid for smooth transition at boundary
        motion_mask = torch.sigmoid(
            (avg_motion - threshold) / (threshold * 0.3 + 1e-6)
        )

        # Expand to target frame count
        motion_mask = motion_mask.unsqueeze(0).expand(target_T, -1, -1)

        return motion_mask


NODE_CLASS_MAPPINGS = {
    "NV_TemporalStabilize": NV_TemporalStabilize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_TemporalStabilize": "NV Temporal Stabilize",
}
