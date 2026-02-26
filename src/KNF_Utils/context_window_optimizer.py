"""
NV Context Window Optimizer - Compute optimal overlap to offset window boundaries.

Context windows create periodic boundaries where adjacent windows are blended.
When two sampling runs share the same temporal space (multi-pass refinement,
re-sampling with different settings, etc.), using identical context window
parameters causes boundaries to align. Artifacts at those positions compound.

NOTE - Iterative latent upscale finding:
  When performing iterative latent upscaling (e.g. 0.25x >> 0.5x >> 1.0x),
  using the same context window settings across passes causes boundary positions
  to land in the same places. Noise artifacts at those boundaries compound with
  each successive pass. This node offsets the boundaries between passes to
  prevent that compounding.

This node computes the optimal context_overlap such that window boundaries
land as far as possible from a reference stride's boundary positions, using
the actual video length to brute-force all real boundary pairs. Among
candidates with equal boundary distance, it prefers more overlap for
better blending quality.

Usage:
  Wire your control video (IMAGE) into this node. It extracts the frame count
  automatically, then outputs the optimal context_overlap to plug directly
  into WAN Context Windows (Manual).

  [Load Video] image --> [This Node] context_overlap --> [WAN Context Windows]
"""

import math


from .chunk_utils import video_to_latent_frames as _vtl_impl


def _pixel_to_latent(pixel_frames):
    """Convert WAN pixel frames to latent frames: ((px - 1) // 4) + 1"""
    return _vtl_impl(pixel_frames)


def _latent_to_pixel(latent_frames):
    """Convert latent frames back to the minimum pixel value that maps to it.
    Inverse of ((px - 1) // 4) + 1 = L  -->  px = 4 * (L - 1) + 1"""
    if latent_frames <= 0:
        return 0
    return 4 * (latent_frames - 1) + 1


def _get_boundaries(n_latent, stride):
    """Get all non-zero boundary positions for a stride within a video length."""
    if stride <= 0:
        return []
    bounds = []
    pos = stride
    while pos < n_latent:
        bounds.append(pos)
        pos += stride
    return bounds


def _actual_min_distance(n_latent, stride, ref_stride):
    """Compute the true minimum distance between any boundary pair.

    Generates all boundary positions for both strides within the video,
    then finds the closest pair. Returns the minimum distance, or n_latent
    if one or both have no boundaries.
    """
    ours = _get_boundaries(n_latent, stride)
    refs = _get_boundaries(n_latent, ref_stride)

    if not ours or not refs:
        return n_latent  # No overlap possible

    min_dist = n_latent
    for a in ours:
        for b in refs:
            d = abs(a - b)
            if d < min_dist:
                min_dist = d
                if min_dist == 0:
                    return 0  # Can't do worse than exact alignment

    return min_dist


class NV_ContextWindowOptimizer:
    """
    Compute optimal context_overlap for maximum quality: boundaries as far
    as possible from a reference configuration, preferring more overlap
    when distances are tied.

    IMPORTANT: This node only optimizes overlap. Set context_length to
    what your GPU can handle at the current pass resolution. VRAM for WAN
    attention scales with (CL_latent × H_latent × W_latent)^2 — a CL that
    worked at a lower resolution pass may OOM at higher resolution.

    Wire your control video (IMAGE) or total_frames to provide the video length.
    The node picks the smallest stride (most overlap) that avoids exact
    boundary collision with the reference pass. More overlap = gentler
    pyramid blending = fewer artifacts, even at the cost of boundary distance.

    Outputs context_overlap in pixel frames, ready to plug into
    NV WAN Context Windows or WAN Context Windows (Manual).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "context_length": ("INT", {
                    "default": 81, "min": 5, "max": 513, "step": 4,
                    "tooltip": "Context window length for THIS pass (pixel frames). "
                               "Set this to the CL your GPU can handle at the "
                               "current pass resolution. This node only optimizes "
                               "overlap — it does not change context_length."
                }),
                "reference_context_length": ("INT", {
                    "default": 81, "min": 5, "max": 513, "step": 4,
                    "tooltip": "Context window length of the PREVIOUS pass whose "
                               "boundaries you want to avoid (pixel frames). "
                               "This was the CL used at the previous resolution."
                }),
                "reference_context_overlap": ("INT", {
                    "default": 30, "min": 0, "max": 200, "step": 1,
                    "tooltip": "Context window overlap of the PREVIOUS pass whose "
                               "boundaries you want to avoid (pixel frames)."
                }),
            },
            "optional": {
                "images": ("IMAGE", {
                    "tooltip": "Control/input video. Frame count is extracted automatically "
                               "from tensor shape. Overrides total_frames if both provided."
                }),
                "total_frames": ("INT", {
                    "default": 0, "min": 0, "max": 10000, "step": 1,
                    "tooltip": "Total pixel frames in the video. Set to 0 to use the IMAGE "
                               "input instead. If both are provided, IMAGE takes precedence."
                }),
                "min_overlap": ("INT", {
                    "default": 13, "min": 0, "max": 200, "step": 1,
                    "tooltip": "Minimum acceptable overlap in pixel frames. "
                               "Ensures blending zones don't get too thin. "
                               "(13px = 4 latent frames)"
                }),
            },
        }

    RETURN_TYPES = ("INT", "INT", "INT", "FLOAT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = (
        "context_overlap",          # Pixel frames - plug directly into WAN CW node
        "latent_stride",            # Diagnostic: stride in latent frames
        "min_boundary_distance",    # True worst-case distance from any ref boundary
        "offset_quality",           # 0.0-1.0 (1.0 = no boundary pair closer than half ref stride)
        "num_windows",              # How many context windows this config produces
        "compute_multiplier",       # Total latent work vs single pass (num_windows * cl / n)
        "summary",                  # Human-readable summary string
    )
    FUNCTION = "optimize"
    CATEGORY = "NV_Utils/context_windows"
    DESCRIPTION = (
        "Computes optimal context_overlap to maximize boundary distance from a "
        "previous pass's window boundaries. Only adjusts overlap — set "
        "context_length to what your GPU can handle at the current resolution "
        "(VRAM scales quadratically with CL × spatial resolution). Wire your "
        "control video to the IMAGE input for automatic frame count detection."
    )

    def optimize(
        self,
        context_length,
        reference_context_length,
        reference_context_overlap,
        images=None,
        total_frames=0,
        min_overlap=13,
    ):
        # Resolve frame count: IMAGE tensor shape[0] takes precedence
        if images is not None:
            n_pixel = images.shape[0]
        elif total_frames > 0:
            n_pixel = total_frames
        else:
            # No frame count provided - fall back to heuristic mode
            return self._optimize_heuristic(
                context_length, reference_context_length,
                reference_context_overlap, min_overlap
            )

        # Convert everything to latent frames
        n_latent = _pixel_to_latent(n_pixel)
        cl_lat = _pixel_to_latent(context_length)
        ref_cl_lat = _pixel_to_latent(reference_context_length)
        ref_co_lat = _pixel_to_latent(reference_context_overlap) if reference_context_overlap > 0 else 0
        ref_stride = ref_cl_lat - ref_co_lat

        # Convert pixel overlap bound to latent
        min_co_lat = _pixel_to_latent(min_overlap) if min_overlap > 0 else 0

        # Edge case: video fits in a single context window
        if n_latent <= cl_lat:
            return (
                0, cl_lat, n_latent, 1.0, 1, 1.0,
                f"Video ({n_pixel}px={n_latent}lat) fits in one window "
                f"(CL={cl_lat}lat). No context windows needed."
            )

        # Edge case: reference had no meaningful stride
        if ref_stride <= 0:
            default_co_lat = max(cl_lat // 3, 1)
            default_stride = cl_lat - default_co_lat
            num_win = self._count_windows(n_latent, cl_lat, default_co_lat)
            cm = (num_win * cl_lat / n_latent) if n_latent > 0 else 1.0
            return (
                _latent_to_pixel(default_co_lat),
                default_stride, 0, 0.0, num_win, round(cm, 3),
                f"Reference stride <= 0 (overlap >= length). Using default."
            )

        # Stride search range: stride = cl_lat - overlap
        # max_stride = cl_lat - min_overlap (least overlap allowed)
        # min_stride = 1 (most overlap = cl_lat - 1)
        max_stride = cl_lat - min_co_lat
        min_stride = 1
        if max_stride < min_stride:
            max_stride = min_stride

        # Overlap-first: pick smallest stride (most overlap) that doesn't have
        # exact boundary collision (min_dist > 0). Empirical finding: larger blend
        # zones matter more than boundary distance — a gentle pyramid over 9 frames
        # masks disagreements better than a steep pyramid over 5 frames, even if
        # boundaries align slightly closer to reference positions.
        # Fallback: if ALL strides collide, pick the one with max boundary distance.
        best_stride = None
        best_min_dist = -1
        fallback_stride = None
        fallback_min_dist = -1

        for s in range(min_stride, max_stride + 1):
            min_dist = _actual_min_distance(n_latent, s, ref_stride)

            # Track best boundary distance as fallback
            if min_dist > fallback_min_dist or (
                min_dist == fallback_min_dist and fallback_stride is not None and s < fallback_stride
            ):
                fallback_min_dist = min_dist
                fallback_stride = s

            # Primary: first non-colliding stride = most overlap
            if best_stride is None and min_dist > 0:
                best_stride = s
                best_min_dist = min_dist

        if best_stride is None:
            best_stride = fallback_stride if fallback_stride is not None else max(1, max_stride)
            best_min_dist = fallback_min_dist if fallback_min_dist >= 0 else 0

        # Compute outputs
        latent_overlap = cl_lat - best_stride
        pixel_overlap = _latent_to_pixel(latent_overlap) if latent_overlap > 0 else 0
        num_windows = self._count_windows(n_latent, cl_lat, latent_overlap)

        # Quality: 1.0 means no pair closer than half the ref stride
        max_possible_dist = ref_stride / 2.0
        offset_quality = min(best_min_dist / max_possible_dist, 1.0) if max_possible_dist > 0 else 0.0

        # Compute multiplier: total latent frames processed vs video length
        # 1.0 = no overlap (each frame processed once), higher = more redundant work
        compute_mult = (num_windows * cl_lat / n_latent) if n_latent > 0 else 1.0

        # Build boundary map for summary
        opt_bounds = _get_boundaries(n_latent, best_stride)
        ref_bounds = _get_boundaries(n_latent, ref_stride)

        summary = " | ".join([
            f"Video: {n_pixel}px = {n_latent} latent frames",
            f"Reference: CL={ref_cl_lat}lat CO={ref_co_lat}lat stride={ref_stride}lat "
            f"({len(ref_bounds)} boundaries at {ref_bounds})",
            f"Optimized: CL={cl_lat}lat CO={latent_overlap}lat stride={best_stride}lat "
            f"({len(opt_bounds)} boundaries at {opt_bounds})",
            f"Min boundary distance: {best_min_dist}lat "
            f"({offset_quality:.0%} of optimal {math.floor(max_possible_dist)})",
            f"Windows: {num_windows} (compute: {compute_mult:.2f}x)",
            f">> Set context_overlap = {pixel_overlap} in WAN CW node",
        ])

        print(f"[CW Optimizer] {summary}")

        return (
            pixel_overlap,
            best_stride,
            best_min_dist,
            round(offset_quality, 3),
            num_windows,
            round(compute_mult, 3),
            summary,
        )

    def _optimize_heuristic(
        self,
        context_length,
        reference_context_length,
        reference_context_overlap,
        min_overlap,
    ):
        """Fallback when no frame count is available. Uses first-boundary heuristic."""
        cl_lat = _pixel_to_latent(context_length)
        ref_cl_lat = _pixel_to_latent(reference_context_length)
        ref_co_lat = _pixel_to_latent(reference_context_overlap) if reference_context_overlap > 0 else 0
        ref_stride = ref_cl_lat - ref_co_lat

        min_co_lat = _pixel_to_latent(min_overlap) if min_overlap > 0 else 0

        if ref_stride <= 0:
            default_co_lat = max(cl_lat // 3, 1)
            return (_latent_to_pixel(default_co_lat), cl_lat - default_co_lat, 0, 0.0, 0, 0.0,
                    "No frame count provided and reference stride <= 0. Using default.")

        max_stride = cl_lat - min_co_lat
        min_stride = 1
        if max_stride < min_stride:
            max_stride = min_stride

        # Overlap-first: smallest stride with non-colliding boundaries
        best_stride = None
        best_min_dist = -1
        fallback_stride = None
        fallback_min_dist = -1

        for s in range(min_stride, max_stride + 1):
            remainder = s % ref_stride
            min_dist = min(remainder, ref_stride - remainder)

            # Track best distance as fallback
            if min_dist > fallback_min_dist or (
                min_dist == fallback_min_dist and fallback_stride is not None and s < fallback_stride
            ):
                fallback_min_dist = min_dist
                fallback_stride = s

            # Primary: first non-colliding stride
            if best_stride is None and min_dist > 0:
                best_stride = s
                best_min_dist = min_dist

        if best_stride is None:
            best_stride = fallback_stride if fallback_stride is not None else max(1, max_stride)
            best_min_dist = fallback_min_dist if fallback_min_dist >= 0 else 0

        latent_overlap = cl_lat - best_stride
        pixel_overlap = _latent_to_pixel(latent_overlap) if latent_overlap > 0 else 0
        max_possible_dist = ref_stride / 2.0
        offset_quality = min(best_min_dist / max_possible_dist, 1.0) if max_possible_dist > 0 else 0.0

        summary = (
            f"HEURISTIC (no frame count): "
            f"Ref stride={ref_stride}lat | Opt stride={best_stride}lat CO={latent_overlap}lat | "
            f"First-boundary distance: {best_min_dist}lat ({offset_quality:.0%}) | "
            f">> Set context_overlap = {pixel_overlap}. "
            f"Wire IMAGE input for exact optimization."
        )
        print(f"[CW Optimizer] {summary}")
        return (pixel_overlap, best_stride, best_min_dist, round(offset_quality, 3), 0, 0.0, summary)

    @staticmethod
    def _count_windows(n_latent, cl_lat, co_lat):
        """Estimate window count using static_standard schedule logic."""
        if n_latent <= cl_lat:
            return 1
        stride = cl_lat - co_lat
        if stride <= 0:
            return 1
        # static_standard: step by stride, last window snaps back
        count = 0
        start = 0
        while start < n_latent:
            count += 1
            if start + cl_lat >= n_latent:
                break
            start += stride
        return count


NODE_CLASS_MAPPINGS = {
    "NV_ContextWindowOptimizer": NV_ContextWindowOptimizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_ContextWindowOptimizer": "NV Context Window Optimizer",
}
