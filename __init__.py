

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

# ---- Monkey-patch: fix context_windows.set_step for third-party samplers ----
# Two issues with ComfyUI's set_step when used with RK samplers (e.g. RES4LYF):
#   1. Dtype mismatch: sample_sigmas is float32, RES4LYF passes timestep as float64.
#      torch.isclose() requires matching dtypes.
#   2. Sub-step sigmas: Multi-stage RK methods evaluate the model at intermediate
#      sub-sigmas (e.g., midpoints) that don't exist in the original sigma schedule.
#      The original set_step raises an exception when no exact match is found.
# Fix: match dtype, then fall back to nearest sigma when no exact match exists.
# See: node_notes/archive/bugfixes/CONTEXT_WINDOW_DTYPE_MISMATCH_FIX.md
try:
    import torch
    from comfy.context_windows import IndexListContextHandler

    def _patched_set_step(self, timestep: torch.Tensor, model_options: dict):
        sample_sigmas = model_options["transformer_options"]["sample_sigmas"]
        ts = timestep[0].to(sample_sigmas.dtype)

        # Try exact match first (original behavior)
        mask = torch.isclose(sample_sigmas, ts, rtol=0.0001)
        matches = torch.nonzero(mask)
        if torch.numel(matches) > 0:
            self._step = int(matches[0].item())
            return

        # No exact match — RK sub-step sigma. Use nearest schedule entry.
        self._step = int(torch.argmin(torch.abs(sample_sigmas - ts)).item())

    IndexListContextHandler.set_step = _patched_set_step
    print("[NV_Comfy_Utils] Patched context_windows.IndexListContextHandler.set_step (dtype + substep fix)")
except Exception as e:
    print(f"[NV_Comfy_Utils] Warning: could not patch context_windows set_step: {e}")

# ---- Register custom samplers into KSampler dropdown ----
# Adds RF-Solver-2 and Flow-Solver-3 (rectified flow ODE-native solvers)
# Uses the same registration pattern as RES4LYF (__init__.py lines 40-56)
try:
    from .src.KNF_Utils.custom_samplers import sample_nv_rf_solver_2, sample_nv_flow_solver_3
    from comfy.samplers import KSampler, k_diffusion_sampling

    _nv_samplers = {
        "nv_rf_solver_2": sample_nv_rf_solver_2,
        "nv_flow_solver_3": sample_nv_flow_solver_3,
    }
    _added = 0
    for _name, _fn in _nv_samplers.items():
        if _name not in KSampler.SAMPLERS:
            try:
                _idx = KSampler.SAMPLERS.index("uni_pc_bh2")
                KSampler.SAMPLERS.insert(_idx + 1, _name)
            except ValueError:
                KSampler.SAMPLERS.append(_name)
            setattr(k_diffusion_sampling, f"sample_{_name}", _fn)
            _added += 1
    if _added > 0:
        print(f"[NV_Comfy_Utils] Registered {_added} custom sampler(s): {', '.join(_nv_samplers.keys())}")
except Exception as e:
    print(f"[NV_Comfy_Utils] Warning: could not register custom samplers: {e}")

