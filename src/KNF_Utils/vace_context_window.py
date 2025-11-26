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


def slice_vace_conditioning(handler, model, x_in, conds, timestep, model_options,
                            window_idx, window: IndexListContextWindow,
                            model_opts, device, first_device):
    """
    Callback to slice VACE conditioning tensors for context windows.

    Called before each context window evaluation via EVALUATE_CONTEXT_WINDOWS callback.
    Modifies `conds` in-place to slice VACE tensors on the temporal dimension.

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
    dim = handler.dim  # dim=2 for WAN models (temporal dimension)

    for cond_list in conds:
        if cond_list is None:
            continue
        for cond in cond_list:
            if not isinstance(cond, (list, tuple)) or len(cond) < 2:
                continue
            cond_dict = cond[1]
            if not isinstance(cond_dict, dict):
                continue

            for key in VACE_TENSOR_KEYS:
                if key not in cond_dict:
                    continue

                tensor_list = cond_dict[key]
                if not isinstance(tensor_list, list):
                    continue

                sliced_list = []
                for tensor in tensor_list:
                    if isinstance(tensor, torch.Tensor):
                        # Check if tensor has the temporal dimension matching full input
                        if tensor.ndim > dim and tensor.size(dim) == x_in.size(dim):
                            # Use window.get_tensor to slice with correct indices
                            sliced_tensor = window.get_tensor(tensor, device, dim=dim)
                            sliced_list.append(sliced_tensor)
                        else:
                            # Tensor doesn't need slicing (already correct size or different dim)
                            sliced_list.append(tensor if device is None else tensor.to(device))
                    else:
                        # Not a tensor (e.g., strength floats), keep as-is
                        sliced_list.append(tensor)

                # Replace the list in-place
                cond_dict[key] = sliced_list


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
        model = model.clone()

        # Get the context handler if it exists
        context_handler = model.model_options.get("context_handler", None)

        if context_handler is not None:
            # Register callback directly on the context handler's callbacks dict
            # This follows the pattern used by IndexListContextHandler
            callbacks = context_handler.callbacks
            cb_dict = callbacks.setdefault(IndexListCallbacks.EVALUATE_CONTEXT_WINDOWS, {})
            cb_list = cb_dict.setdefault("nv_vace_slicer", [])

            # Only add if not already present (avoid duplicates)
            if slice_vace_conditioning not in cb_list:
                cb_list.append(slice_vace_conditioning)
                print("[NV_VACEContextWindowPatcher] Registered VACE slicing callback on context handler")
        else:
            # Context handler not yet attached - store callback in model_options
            # It will be picked up if context handler is attached later
            print("[NV_VACEContextWindowPatcher] Warning: No context handler found on model.")
            print("  Make sure to connect WAN Context Windows node before this patcher.")
            print("  The patcher will still work if context handler is added later in the workflow.")

            # Store in transformer_options for potential later use
            transformer_options = model.model_options.setdefault("transformer_options", {})
            callbacks = transformer_options.setdefault("callbacks", {})
            cb_dict = callbacks.setdefault(IndexListCallbacks.EVALUATE_CONTEXT_WINDOWS, {})
            cb_list = cb_dict.setdefault("nv_vace_slicer", [])
            if slice_vace_conditioning not in cb_list:
                cb_list.append(slice_vace_conditioning)

        return (model,)


# Node mappings for registration
NODE_CLASS_MAPPINGS = {
    "NV_VACEContextWindowPatcher": NV_VACEContextWindowPatcher,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_VACEContextWindowPatcher": "NV VACE Context Window Patcher",
}
