

import os

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",

]

__author__ = """elkkkk"""
__email__ = "you@gmail.com"
__version__ = "0.0.1"

# Import merged node mappings from subpackage (includes nodes.py + memory_monitor.py)
from .src.KNF_Utils import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# WEB_DIRECTORY is the directory that ComfyUI will link and auto-load for frontend extensions
WEB_DIRECTORY = "./web"

# Import Slack error handler - auto-registers if env vars are configured
# Does nothing if SLACK_BOT_TOKEN and SLACK_ERROR_CHANNEL are not set
from .src.KNF_Utils import slack_error_handler

# ---- Monkey-patch: fix dtype mismatch in context_windows.set_step ----
# ComfyUI stores sample_sigmas as float32, but third-party samplers (e.g. RES4LYF)
# may pass timestep as float64. torch.isclose() requires matching dtypes.
# See: node_notes/archive/bugfixes/CONTEXT_WINDOW_DTYPE_MISMATCH_FIX.md
try:
    import torch
    from comfy.context_windows import IndexListContextHandler

    _original_set_step = IndexListContextHandler.set_step

    def _patched_set_step(self, timestep: torch.Tensor, model_options: dict):
        sample_sigmas = model_options["transformer_options"]["sample_sigmas"]
        if sample_sigmas.dtype != timestep.dtype:
            model_options["transformer_options"]["sample_sigmas"] = sample_sigmas.to(timestep.dtype)
        return _original_set_step(self, timestep, model_options)

    IndexListContextHandler.set_step = _patched_set_step
    print("[NV_Comfy_Utils] Patched context_windows.IndexListContextHandler.set_step (dtype fix)")
except Exception as e:
    print(f"[NV_Comfy_Utils] Warning: could not patch context_windows set_step: {e}")


