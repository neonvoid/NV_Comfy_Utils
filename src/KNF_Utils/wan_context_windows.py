"""
NV WAN Context Windows - Boundary-aware context window node for WAN models.

Drop-in replacement for the core WAN Context Windows (Manual) node. When
reference context window parameters are provided, wraps the selected fuse
method to blend more aggressively at positions near reference boundaries.

Finding: during iterative latent upscaling (e.g. 0.25x >> 0.5x >> 1.0x),
using the same context window settings across passes causes boundary positions
to align. Noise artifacts at those boundaries compound with each successive
pass. This node offsets boundaries (via overlap from the optimizer) AND smooths
residual artifacts at reference boundary positions (via fuse weight modification).

The boundary-aware fuse works by lerping weights toward flat (equal blending)
at positions near reference boundaries. This causes overlapping windows to
contribute more equally at those positions, smoothing out compounded artifacts.
"""

import torch
import logging

import comfy.context_windows
from comfy.context_windows import (
    IndexListContextHandler,
    ContextFuseMethod,
    ContextFuseMethods,
    ContextSchedules,
    FUSE_MAPPING,
    CONTEXT_MAPPING,
    get_matching_context_schedule,
    get_matching_fuse_method,
    create_prepare_sampling_wrapper,
    create_sampler_sample_wrapper,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pixel_to_latent(pixel_frames):
    """Convert WAN pixel frames to latent frames: max(((px - 1) // 4) + 1, 1)"""
    return max(((pixel_frames - 1) // 4) + 1, 1)


def _get_ref_boundaries(full_length, ref_stride):
    """Compute reference boundary positions within a video.

    A boundary is where a new context window starts in the reference config.
    Occurs at multiples of ref_stride, from ref_stride up to full_length.
    """
    if ref_stride <= 0:
        return []
    boundaries = []
    pos = ref_stride
    while pos < full_length:
        boundaries.append(pos)
        pos += ref_stride
    return boundaries


# ---------------------------------------------------------------------------
# Boundary-aware fuse weight factory
# ---------------------------------------------------------------------------

def make_boundary_aware_fuse(base_fuse_func, ref_stride):
    """Create a fuse weight function that wraps a base method with boundary awareness.

    At positions near reference boundaries, lerps weights toward 1.0 (flat/equal
    blending). At positions far from reference boundaries, weights are unchanged
    from the base method.

    The blend_radius is handler.context_overlap -- the zone where windows overlap
    and blending occurs. Within this radius of any reference boundary, the weight
    is lerped toward flat.

    Args:
        base_fuse_func: Original fuse function (pyramid, flat, overlap-linear).
        ref_stride: Reference stride in latent frames (ref_cl - ref_co).

    Returns:
        A callable with the standard fuse method signature:
        func(length, sigma=None, handler=None, full_length=0, idxs=[])
    """

    def boundary_aware_weights(length, sigma=None, handler=None, full_length=0, idxs=[]):
        # Get base weights from the underlying fuse method
        base = base_fuse_func(
            length, sigma=sigma, handler=handler,
            full_length=full_length, idxs=idxs
        )

        # Early return if no reference info or no frame indices
        if ref_stride <= 0 or not idxs or full_length <= 0:
            return base

        # Compute reference boundaries at runtime using actual video length
        ref_boundaries = _get_ref_boundaries(full_length, ref_stride)
        if not ref_boundaries:
            return base

        # Convert to mutable list (handles both list and torch.Tensor returns)
        if isinstance(base, torch.Tensor):
            weights = base.tolist()
        else:
            weights = list(base)

        # Blend radius: within this distance of a ref boundary, lerp toward flat.
        # context_overlap is the natural choice -- it's where windows overlap.
        blend_radius = handler.context_overlap if handler is not None else length // 4
        if blend_radius <= 0:
            return base

        for i, frame_idx in enumerate(idxs):
            # Find minimum distance to any reference boundary
            min_dist = min(abs(frame_idx - rb) for rb in ref_boundaries)

            if min_dist < blend_radius:
                # Proximity: 1.0 at the boundary, 0.0 at blend_radius
                t = 1.0 - (min_dist / blend_radius)
                # Lerp toward flat weight (1.0 = equal blending)
                weights[i] = weights[i] * (1.0 - t) + 1.0 * t

        return weights

    return boundary_aware_weights


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class NV_WanContextWindows:
    """
    WAN Context Windows with optional boundary-aware fuse weights.

    Drop-in replacement for the core WAN Context Windows (Manual) node.
    When reference_context_length and reference_context_overlap are connected,
    wraps the selected fuse method to blend more aggressively near reference
    boundary positions. Without reference inputs, behaves identically to the
    core node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "context_length": ("INT", {
                    "default": 81, "min": 1, "max": 16384, "step": 4,
                    "tooltip": "Context window length in pixel frames."
                }),
                "context_overlap": ("INT", {
                    "default": 30, "min": 0, "max": 512, "step": 1,
                    "tooltip": "Context window overlap in pixel frames. "
                               "Can be wired from NV Context Window Optimizer."
                }),
                "context_schedule": (list(CONTEXT_MAPPING.keys()), {
                    "tooltip": "Schedule algorithm for generating context windows."
                }),
                "fuse_method": (ContextFuseMethods.LIST_STATIC, {
                    "default": ContextFuseMethods.PYRAMID,
                    "tooltip": "Blending method for overlapping windows. "
                               "Boundary-aware adjustment applies to all except 'relative'."
                }),
                "freenoise": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Apply FreeNoise noise shuffling for better window blending."
                }),
            },
            "optional": {
                "reference_context_length": ("INT", {
                    "default": 81, "min": 1, "max": 16384, "step": 4,
                    "tooltip": "Context length (pixel frames) of the reference pass "
                               "whose boundaries you want to smooth. "
                               "Leave disconnected to disable boundary-aware blending."
                }),
                "reference_context_overlap": ("INT", {
                    "default": 30, "min": 0, "max": 512, "step": 1,
                    "tooltip": "Context overlap (pixel frames) of the reference pass "
                               "whose boundaries you want to smooth."
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "execute"
    CATEGORY = "NV_Utils/context_windows"
    DESCRIPTION = (
        "WAN Context Windows with boundary-aware fuse weights. When reference "
        "parameters are connected, blends more aggressively near reference "
        "boundary positions to prevent artifact compounding during multi-pass "
        "iterative latent upscaling. Without reference inputs, behaves "
        "identically to the core WAN Context Windows (Manual) node."
    )

    def execute(
        self,
        model,
        context_length,
        context_overlap,
        context_schedule,
        fuse_method,
        freenoise,
        reference_context_length=None,
        reference_context_overlap=None,
    ):
        # --- 1. Pixel to latent conversion (same as core WAN CW node) ---
        cl_lat = _pixel_to_latent(context_length)
        co_lat = max(((context_overlap - 1) // 4) + 1, 0) if context_overlap > 0 else 0

        # --- 2. Resolve fuse method ---
        has_ref = (
            reference_context_length is not None
            and reference_context_overlap is not None
        )
        use_boundary_aware = has_ref and fuse_method != ContextFuseMethods.RELATIVE

        if use_boundary_aware:
            # Compute reference stride in latent frames
            ref_cl_lat = _pixel_to_latent(reference_context_length)
            ref_co_lat = (
                max(((reference_context_overlap - 1) // 4) + 1, 0)
                if reference_context_overlap > 0 else 0
            )
            ref_stride = ref_cl_lat - ref_co_lat

            if ref_stride > 0:
                base_fuse_func = FUSE_MAPPING[fuse_method]
                wrapped_func = make_boundary_aware_fuse(base_fuse_func, ref_stride)
                fuse = ContextFuseMethod(name=fuse_method, func=wrapped_func)
                logger.info(
                    f"[NV_WanContextWindows] Boundary-aware fuse: "
                    f"base={fuse_method}, ref_stride={ref_stride}lat "
                    f"(ref_cl={ref_cl_lat}, ref_co={ref_co_lat})"
                )
            else:
                fuse = get_matching_fuse_method(fuse_method)
                logger.warning(
                    f"[NV_WanContextWindows] Reference stride <= 0, "
                    f"using standard {fuse_method} fuse"
                )
        else:
            fuse = get_matching_fuse_method(fuse_method)
            if has_ref and fuse_method == ContextFuseMethods.RELATIVE:
                logger.info(
                    f"[NV_WanContextWindows] RELATIVE fuse selected; "
                    f"boundary-aware adjustment not supported for RELATIVE. "
                    f"Using standard RELATIVE behavior."
                )

        # --- 3. Create handler ---
        handler = IndexListContextHandler(
            context_schedule=get_matching_context_schedule(context_schedule),
            fuse_method=fuse,
            context_length=cl_lat,
            context_overlap=co_lat,
            context_stride=1,
            closed_loop=False,
            dim=2,
            freenoise=freenoise,
        )

        # --- 4. Attach to model ---
        model = model.clone()
        model.model_options["context_handler"] = handler

        create_prepare_sampling_wrapper(model)
        if freenoise:
            create_sampler_sample_wrapper(model)

        stride_lat = cl_lat - co_lat
        logger.info(
            f"[NV_WanContextWindows] CL={cl_lat}lat CO={co_lat}lat "
            f"stride={stride_lat}lat schedule={context_schedule} "
            f"fuse={fuse_method} freenoise={freenoise}"
        )

        return (model,)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "NV_WanContextWindows": NV_WanContextWindows,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_WanContextWindows": "NV WAN Context Windows",
}
