"""
NV Context Window Optimizer - Compute optimal overlap for multi-pass pipelines.

When running a multi-pass V2V pipeline (generation → refinement passes), using
the same context window settings for every pass causes boundary positions to
align. Artifacts at those boundaries compound across passes.

This node computes the optimal context_overlap for the current pass such that
its window boundaries land as far as possible from the previous pass's boundaries.

Math:
  Given previous stride S_prev and current context_length CL (both in latent frames),
  for each candidate stride S, the minimum distance from any previous-pass boundary is:
      min_dist = min(S % S_prev, S_prev - S % S_prev)
  We pick the S that maximizes min_dist. The theoretical optimum is floor(S_prev / 2),
  achieved when the two strides are coprime and offset by ~half a stride.

Usage:
  [This Node] → context_overlap output → [WAN Context Windows (Manual)] context_overlap input
"""

import math


def _pixel_to_latent(pixel_frames):
    """Convert WAN pixel frames to latent frames: ((px - 1) // 4) + 1"""
    if pixel_frames <= 0:
        return 0
    return ((pixel_frames - 1) // 4) + 1


def _latent_to_pixel(latent_frames):
    """Convert latent frames back to the minimum pixel value that maps to it.
    Inverse of ((px - 1) // 4) + 1 = L  →  px = 4 * (L - 1) + 1"""
    if latent_frames <= 0:
        return 0
    return 4 * (latent_frames - 1) + 1


class NV_ContextWindowOptimizer:
    """
    Compute optimal context_overlap for a refinement pass to maximize
    boundary offset from a previous pass's context window positions.

    Connect the context_overlap output directly to WAN Context Windows (Manual).
    The latent_stride and min_boundary_distance outputs are diagnostic.

    ┌─────────────────────────────────────────────────────────────────┐
    │  Example:                                                       │
    │                                                                 │
    │  Previous pass: CL=81, CO=30  → latent stride=13               │
    │  This pass:     CL=81         → optimizer finds CO=53           │
    │    → latent stride=7, min boundary distance=6 (92% of optimal)  │
    │                                                                 │
    │  Pass 2 boundaries: 0  13  26  39  52  65  78 ...               │
    │  Pass 3 boundaries: 0   7  14  21  28  35  42 ...               │
    │                          ↑       ↑       ↑                      │
    │                 distance: 6   5   6   6   5   6                 │
    └─────────────────────────────────────────────────────────────────┘
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "context_length": ("INT", {
                    "default": 81, "min": 5, "max": 513, "step": 4,
                    "tooltip": "Context window length for THIS pass (pixel frames). "
                               "Same value you set in WAN Context Windows."
                }),
                "prev_context_length": ("INT", {
                    "default": 81, "min": 5, "max": 513, "step": 4,
                    "tooltip": "Context window length from the PREVIOUS pass (pixel frames)."
                }),
                "prev_context_overlap": ("INT", {
                    "default": 30, "min": 0, "max": 200, "step": 1,
                    "tooltip": "Context window overlap from the PREVIOUS pass (pixel frames)."
                }),
            },
            "optional": {
                "min_overlap_latent": ("INT", {
                    "default": 4, "min": 1, "max": 20, "step": 1,
                    "tooltip": "Minimum acceptable overlap in latent frames. "
                               "Lower = fewer windows but thinner blending zone."
                }),
                "max_overlap_latent": ("INT", {
                    "default": 0, "min": 0, "max": 20, "step": 1,
                    "tooltip": "Maximum overlap in latent frames (0 = no limit). "
                               "Set this to cap the number of windows for faster renders."
                }),
            },
        }

    RETURN_TYPES = ("INT", "INT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = (
        "context_overlap",          # Pixel frames - plug directly into WAN CW node
        "latent_stride",            # Diagnostic: stride in latent frames
        "min_boundary_distance",    # Diagnostic: worst-case distance from prev boundary
        "offset_quality",           # Diagnostic: 0.0-1.0 (1.0 = boundaries at exact midpoints)
        "summary",                  # Human-readable summary string
    )
    FUNCTION = "optimize"
    CATEGORY = "NV_Utils/context_windows"
    DESCRIPTION = (
        "Computes optimal context_overlap for a refinement pass to maximize "
        "boundary offset from the previous pass. Outputs pixel-frame overlap "
        "ready to plug into WAN Context Windows (Manual)."
    )

    def optimize(
        self,
        context_length,
        prev_context_length,
        prev_context_overlap,
        min_overlap_latent=4,
        max_overlap_latent=0,
    ):
        # Convert to latent frames (WAN VAE temporal compression)
        cl_lat = _pixel_to_latent(context_length)
        prev_cl_lat = _pixel_to_latent(prev_context_length)
        prev_co_lat = _pixel_to_latent(prev_context_overlap) if prev_context_overlap > 0 else 0
        prev_stride = prev_cl_lat - prev_co_lat

        # Edge case: previous pass had no meaningful stride
        if prev_stride <= 0:
            default_co = _latent_to_pixel(max(cl_lat // 3, 1))
            return (
                default_co,
                cl_lat - _pixel_to_latent(default_co),
                0,
                0.0,
                f"Previous stride <= 0 (overlap >= length). Using default overlap={default_co}px."
            )

        # Edge case: previous pass didn't use context windows (stride >= video)
        # Any overlap works, return default
        if prev_stride >= cl_lat:
            default_overlap = _latent_to_pixel(max(cl_lat // 3, 1))
            return (
                default_overlap,
                cl_lat - _pixel_to_latent(default_overlap),
                prev_stride,
                1.0,
                f"Previous stride ({prev_stride}lat) >= current CL ({cl_lat}lat). "
                f"No boundary alignment risk. Using default overlap={default_overlap}px."
            )

        # Determine stride search range
        # stride = cl_lat - overlap, so:
        #   max_stride corresponds to min_overlap
        #   min_stride corresponds to max_overlap
        max_stride = cl_lat - min_overlap_latent
        if max_overlap_latent > 0:
            min_stride = max(1, cl_lat - max_overlap_latent)
        else:
            min_stride = 1

        if max_stride < min_stride:
            max_stride = min_stride

        # Search for stride that maximizes minimum boundary distance
        best_stride = None
        best_min_dist = -1

        for s in range(min_stride, max_stride + 1):
            remainder = s % prev_stride
            min_dist = min(remainder, prev_stride - remainder)

            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_stride = s
            elif min_dist == best_min_dist and best_stride is not None:
                # Tie-break: prefer larger stride (fewer windows, faster)
                if s > best_stride:
                    best_stride = s

        if best_stride is None:
            best_stride = max(1, max_stride)
            best_min_dist = 0

        # Compute outputs
        latent_overlap = cl_lat - best_stride
        pixel_overlap = _latent_to_pixel(latent_overlap) if latent_overlap > 0 else 0

        # Quality score: 1.0 = boundaries exactly at midpoints of previous windows
        max_possible_dist = prev_stride / 2.0
        offset_quality = best_min_dist / max_possible_dist if max_possible_dist > 0 else 0.0

        # Verify the pixel→latent round-trip
        verify_lat = _pixel_to_latent(pixel_overlap)
        verify_stride = cl_lat - verify_lat

        # Build summary
        lines = [
            f"Previous pass: CL={prev_cl_lat}lat CO={prev_co_lat}lat stride={prev_stride}lat",
            f"This pass:     CL={cl_lat}lat CO={latent_overlap}lat stride={best_stride}lat",
            f"Min boundary distance: {best_min_dist} latent frames ({offset_quality:.0%} of optimal {math.floor(max_possible_dist)})",
            f">> Set context_overlap = {pixel_overlap} in WAN Context Windows node",
        ]
        summary = " | ".join(lines)

        print(f"[CW Optimizer] {summary}")

        return (
            pixel_overlap,
            best_stride,
            best_min_dist,
            round(offset_quality, 3),
            summary,
        )


NODE_CLASS_MAPPINGS = {
    "NV_ContextWindowOptimizer": NV_ContextWindowOptimizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_ContextWindowOptimizer": "NV Context Window Optimizer",
}
