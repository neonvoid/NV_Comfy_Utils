"""
NV Context Window Optimizer - Compute optimal overlap to offset window boundaries.

Context windows create periodic boundaries where adjacent windows are blended.
When two sampling runs share the same temporal space (multi-pass refinement,
re-sampling with different settings, etc.), using identical context window
parameters causes boundaries to align. Artifacts at those positions compound.

This node computes the optimal context_overlap such that window boundaries
land as far as possible from a reference stride's boundary positions.

Math:
  Given a reference stride S_ref and context_length CL (both in latent frames),
  for each candidate stride S, the minimum distance from any reference boundary is:
      min_dist = min(S % S_ref, S_ref - S % S_ref)
  We pick the S that maximizes min_dist. The theoretical optimum is floor(S_ref / 2),
  achieved when the two strides are coprime and offset by ~half a stride.

Usage:
  [This Node] context_overlap --> [WAN Context Windows (Manual)] context_overlap
"""

import math


def _pixel_to_latent(pixel_frames):
    """Convert WAN pixel frames to latent frames: ((px - 1) // 4) + 1"""
    if pixel_frames <= 0:
        return 0
    return ((pixel_frames - 1) // 4) + 1


def _latent_to_pixel(latent_frames):
    """Convert latent frames back to the minimum pixel value that maps to it.
    Inverse of ((px - 1) // 4) + 1 = L  -->  px = 4 * (L - 1) + 1"""
    if latent_frames <= 0:
        return 0
    return 4 * (latent_frames - 1) + 1


class NV_ContextWindowOptimizer:
    """
    Compute optimal context_overlap that maximizes boundary offset from
    a reference context window configuration.

    Use case: any time two sampling runs share temporal space and you want
    their context window boundaries to NOT align. Common scenarios:
      - Multi-pass V2V refinement (pass N vs pass N+1)
      - Re-sampling the same video with different settings
      - Any workflow where context window boundary artifacts are visible

    Connect the context_overlap output directly to WAN Context Windows (Manual).

    Example:
      Reference:  CL=81, CO=30  (latent stride=13)
      Optimized:  CL=81         --> CO=53 (latent stride=7)
        min boundary distance = 6 latent frames (92% of optimal)

      Reference boundaries: 0  13  26  39  52  65  78 ...
      Optimized boundaries: 0   7  14  21  28  35  42 ...
                   distance:    6   5   6   6   5   6
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "context_length": ("INT", {
                    "default": 81, "min": 5, "max": 513, "step": 4,
                    "tooltip": "Context window length you plan to use (pixel frames). "
                               "Same value you'll set in WAN Context Windows."
                }),
                "reference_context_length": ("INT", {
                    "default": 81, "min": 5, "max": 513, "step": 4,
                    "tooltip": "Context window length of the run whose boundaries "
                               "you want to avoid (pixel frames)."
                }),
                "reference_context_overlap": ("INT", {
                    "default": 30, "min": 0, "max": 200, "step": 1,
                    "tooltip": "Context window overlap of the run whose boundaries "
                               "you want to avoid (pixel frames)."
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
        "min_boundary_distance",    # Diagnostic: worst-case distance from ref boundary
        "offset_quality",           # Diagnostic: 0.0-1.0 (1.0 = boundaries at exact midpoints)
        "summary",                  # Human-readable summary string
    )
    FUNCTION = "optimize"
    CATEGORY = "NV_Utils/context_windows"
    DESCRIPTION = (
        "Computes optimal context_overlap to maximize boundary offset from a "
        "reference context window configuration. Outputs pixel-frame overlap "
        "ready to plug into WAN Context Windows (Manual)."
    )

    def optimize(
        self,
        context_length,
        reference_context_length,
        reference_context_overlap,
        min_overlap_latent=4,
        max_overlap_latent=0,
    ):
        # Convert to latent frames (WAN VAE temporal compression)
        cl_lat = _pixel_to_latent(context_length)
        ref_cl_lat = _pixel_to_latent(reference_context_length)
        ref_co_lat = _pixel_to_latent(reference_context_overlap) if reference_context_overlap > 0 else 0
        ref_stride = ref_cl_lat - ref_co_lat

        # Edge case: reference had no meaningful stride (overlap >= length)
        if ref_stride <= 0:
            default_co = _latent_to_pixel(max(cl_lat // 3, 1))
            return (
                default_co,
                cl_lat - _pixel_to_latent(default_co),
                0,
                0.0,
                f"Reference stride <= 0 (overlap >= length). Using default overlap={default_co}px."
            )

        # Edge case: reference stride larger than our context length
        # No boundary alignment risk - any overlap works
        if ref_stride >= cl_lat:
            default_overlap = _latent_to_pixel(max(cl_lat // 3, 1))
            return (
                default_overlap,
                cl_lat - _pixel_to_latent(default_overlap),
                ref_stride,
                1.0,
                f"Reference stride ({ref_stride}lat) >= context length ({cl_lat}lat). "
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
            remainder = s % ref_stride
            min_dist = min(remainder, ref_stride - remainder)

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

        # Quality score: 1.0 = boundaries exactly at midpoints of reference windows
        max_possible_dist = ref_stride / 2.0
        offset_quality = best_min_dist / max_possible_dist if max_possible_dist > 0 else 0.0

        # Build summary
        lines = [
            f"Reference: CL={ref_cl_lat}lat CO={ref_co_lat}lat stride={ref_stride}lat",
            f"Optimized: CL={cl_lat}lat CO={latent_overlap}lat stride={best_stride}lat",
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
