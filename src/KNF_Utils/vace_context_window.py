"""
VACE Context Window Support

This module provides nodes to make VACE conditioning work with ComfyUI's
context window system. The core problem is that VACE stores tensors in lists,
but the context window's get_resized_cond method only slices plain tensors.

Solution: Register a callback on EVALUATE_CONTEXT_WINDOWS that slices the
VACE tensors before each window is processed.

All changes are isolated to NV_Comfy_Utils - no ComfyUI core modifications.
"""

import torch
from comfy.context_windows import IndexListCallbacks, IndexListContextWindow

# Keys in conditioning that contain list-wrapped tensors needing slicing
VACE_TENSOR_KEYS = ["vace_frames", "vace_mask"]

# Threshold for CPU offload: videos longer than this cache to CPU to avoid OOM
# 55 latent frames ≈ 220 pixel frames. Below this, GPU cache is fast and fits.
# Above this, we offload to CPU and transfer one window at a time.
LONG_VIDEO_THRESHOLD = 55

# Cache for original full-length tensors, keyed by cond_dict uuid
# This is necessary because we modify conds in-place, and need to preserve
# originals for subsequent windows within the same timestep
_VACE_ORIGINALS_CACHE = {}


def slice_vace_conditioning(handler, model, x_in, conds, timestep, model_options,
                            window_idx, window: IndexListContextWindow,
                            model_opts, device, first_device):
    """
    Callback to slice VACE conditioning tensors for context windows.

    Called before each context window evaluation via EVALUATE_CONTEXT_WINDOWS callback.
    Modifies `conds` in-place to slice VACE tensors on the temporal dimension.

    IMPORTANT: We cache original tensors because in-place modifications persist
    across windows within the same timestep. Without caching, window 0 would slice
    the tensors, then windows 1, 2, 3... would see already-sliced tensors and fail
    to slice to their correct indices.

    Parameters:
        handler: The IndexListContextHandler instance
        model: The model being used
        x_in: Input tensor [B, C, T_full, H, W]
        conds: List of conditioning lists (positive, negative, etc.)
        timestep: Current timestep
        model_options: Model options dict
        window_idx: Index of current window
        window: IndexListContextWindow with index_list for slicing
        model_opts: Same as model_options
        device: Target device
        first_device: First device for multi-GPU
    """
    global _VACE_ORIGINALS_CACHE

    dim = handler.dim  # dim=2 for WAN models (temporal dimension)
    full_temporal_size = x_in.size(dim)  # Full video length (e.g., 48 frames)

    # Adaptive CPU offload: for long videos, cache on CPU to avoid OOM
    use_cpu_cache = full_temporal_size > LONG_VIDEO_THRESHOLD

    # Debug: Log callback invocation (with parseable format for workflow logger)
    if window_idx == 0:  # Only log first window to reduce noise
        window_size = len(window.index_list)
        print(f"[VACE Slicer] Callback invoked for window {window_idx}")
        print(f"[VACE Slicer] x_in shape: {x_in.shape}, dim={dim}")
        print(f"[VACE Slicer] Window indices: {window.index_list[:5]}...{window.index_list[-5:] if len(window.index_list) > 5 else ''}")
        print(f"[VACE Slicer] Number of cond lists: {len(conds)}")
        # Parseable log line for workflow logger
        print(f"[VACE Slicer] Using context window {window_size} for {full_temporal_size} frames")
        if use_cpu_cache:
            print(f"[VACE Slicer] Long video detected ({full_temporal_size} frames > {LONG_VIDEO_THRESHOLD}), using CPU cache")

    for cond_list in conds:
        if cond_list is None:
            continue
        for cond_idx, cond_dict in enumerate(cond_list):
            # At callback time, conds have already been converted from [(tensor, dict), ...]
            # to [dict, ...] by convert_cond in sampler_helpers.py
            if not isinstance(cond_dict, dict):
                continue

            # Get unique identifier for this cond_dict
            cond_uuid = cond_dict.get("uuid", None)
            if cond_uuid is None:
                continue

            cache_key = str(cond_uuid)

            # Initialize cache entry if needed
            if cache_key not in _VACE_ORIGINALS_CACHE:
                _VACE_ORIGINALS_CACHE[cache_key] = {}
            cache = _VACE_ORIGINALS_CACHE[cache_key]

            # Debug: Print all keys in the first cond_dict of first window
            if window_idx == 0 and cond_idx == 0:
                print(f"[VACE Slicer] cond_dict keys: {list(cond_dict.keys())}")

            # Process vace_frames and vace_mask
            for key in VACE_TENSOR_KEYS:
                if key not in cond_dict:
                    continue

                # Cache original if not already cached (check by matching full temporal size)
                if key not in cache:
                    tensor_list = cond_dict[key]
                    if isinstance(tensor_list, list) and len(tensor_list) > 0:
                        first_tensor = tensor_list[0]
                        if isinstance(first_tensor, torch.Tensor) and first_tensor.ndim > dim:
                            if first_tensor.size(dim) == full_temporal_size:
                                # This is the original full-length tensor, cache it
                                if use_cpu_cache:
                                    # Long video: cache on CPU to avoid OOM
                                    cache[key] = [t.cpu().clone() if isinstance(t, torch.Tensor) else t for t in tensor_list]
                                else:
                                    # Short video: cache on GPU for speed
                                    cache[key] = [t.clone() if isinstance(t, torch.Tensor) else t for t in tensor_list]
                                if window_idx == 0 and cond_idx == 0:
                                    cache_loc = "CPU" if use_cpu_cache else "GPU"
                                    print(f"[VACE Slicer] Cached original {key}: shape {first_tensor.shape} ({cache_loc})")

                # Slice from cached original
                if key in cache:
                    original_list = cache[key]
                    sliced_list = []
                    for tensor in original_list:
                        if isinstance(tensor, torch.Tensor):
                            # If cached on CPU, move to GPU before slicing
                            # Explicitly preserve dtype to avoid autocast issues with lowvram patches
                            tensor_for_slice = tensor.to(device=device, dtype=tensor.dtype) if use_cpu_cache else tensor
                            sliced_tensor = window.get_tensor(tensor_for_slice, device, dim=dim)
                            sliced_list.append(sliced_tensor)
                            if window_idx == 0 and cond_idx == 0:
                                print(f"[VACE Slicer] Sliced {key}: {tensor.shape} -> {sliced_tensor.shape}")
                        else:
                            sliced_list.append(tensor)
                    cond_dict[key] = sliced_list

            # Also slice model_conds['vace_context'] which is a CONDRegular
            # After extra_conds(), vace_context has shape [B, num_vace, 96, T, H, W]
            # where temporal dimension is at index 3 (not 2!)
            model_conds = cond_dict.get("model_conds", {})
            if "vace_context" in model_conds:
                vace_context = model_conds["vace_context"]
                vace_context_key = "vace_context"
                vace_temporal_dim = 3

                # CONDRegular objects have a .cond attribute with the tensor
                if hasattr(vace_context, "cond") and isinstance(vace_context.cond, torch.Tensor):
                    vace_tensor = vace_context.cond

                    # Cache original vace_context if not already cached
                    if vace_context_key not in cache:
                        if vace_tensor.ndim > vace_temporal_dim and vace_tensor.size(vace_temporal_dim) == full_temporal_size:
                            if use_cpu_cache:
                                # Long video: cache on CPU to avoid OOM
                                cache[vace_context_key] = vace_context._copy_with(vace_tensor.cpu().clone())
                            else:
                                # Short video: cache on GPU for speed
                                cache[vace_context_key] = vace_context._copy_with(vace_tensor.clone())
                            if window_idx == 0 and cond_idx == 0:
                                cache_loc = "CPU" if use_cpu_cache else "GPU"
                                print(f"[VACE Slicer] Cached original vace_context: shape {vace_tensor.shape} ({cache_loc})")

                    # Slice from cached original
                    if vace_context_key in cache:
                        original_vace_context = cache[vace_context_key]
                        original_tensor = original_vace_context.cond
                        # If cached on CPU, move to GPU before slicing
                        # Explicitly preserve dtype to avoid autocast issues with lowvram patches
                        tensor_for_slice = original_tensor.to(device=device, dtype=original_tensor.dtype) if use_cpu_cache else original_tensor
                        sliced_tensor = window.get_tensor(tensor_for_slice, device, dim=vace_temporal_dim)
                        model_conds["vace_context"] = original_vace_context._copy_with(sliced_tensor)
                        if window_idx == 0 and cond_idx == 0:
                            print(f"[VACE Slicer] Sliced vace_context: {original_tensor.shape} -> {sliced_tensor.shape}")


class NV_VACEContextWindowPatcher:
    """
    Patches a model to properly handle VACE conditioning with context windows.

    Connect this node AFTER your context window node and BEFORE the sampler.
    The model will automatically slice VACE conditioning for each context window.

    Usage:
        [WAN Context Windows] -> [VACE Context Window Patcher] -> [KSampler]

    This enables using VACEtoVideo with context windows for long video generation.
    The VACE conditioning (vace_frames, vace_mask) will be automatically sliced
    to match each context window during sampling.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "NV_Utils/context_windows"
    DESCRIPTION = "Enables VACE conditioning to work with context windows by slicing VACE tensors per-window."

    def patch(self, model):
        import comfy.patcher_extension

        model = model.clone()

        # Get the context handler if it exists
        context_handler = model.model_options.get("context_handler", None)

        if context_handler is not None:
            # Register callback using ComfyUI's official API
            # get_all_callbacks expects: callbacks_dict["callbacks"][call_type][key] = [funcs]
            # So we need to nest under "callbacks" key
            comfy.patcher_extension.add_callback_with_key(
                IndexListCallbacks.EVALUATE_CONTEXT_WINDOWS,
                "nv_vace_slicer",
                slice_vace_conditioning,
                context_handler.callbacks,
                is_model_options=False
            )
            print("[NV_VACEContextWindowPatcher] Registered VACE slicing callback on context handler")
        else:
            # Context handler not yet attached - store callback in model_options
            # It will be picked up if context handler is attached later
            print("[NV_VACEContextWindowPatcher] Warning: No context handler found on model.")
            print("  Make sure to connect WAN Context Windows node before this patcher.")
            print("  The patcher will still work if context handler is added later in the workflow.")

            # Store in model_options using the official API
            comfy.patcher_extension.add_callback_with_key(
                IndexListCallbacks.EVALUATE_CONTEXT_WINDOWS,
                "nv_vace_slicer",
                slice_vace_conditioning,
                model.model_options,
                is_model_options=True
            )

        return (model,)


# Node mappings for registration
NODE_CLASS_MAPPINGS = {
    "NV_VACEContextWindowPatcher": NV_VACEContextWindowPatcher,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_VACEContextWindowPatcher": "NV VACE Context Window Patcher",
}
