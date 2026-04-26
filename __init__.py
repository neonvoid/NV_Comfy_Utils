

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

# ---- Post-prompt memory flush hook (Windows-focused) ----
# Hooks PromptServer.send_sync to detect the "executing/node=None" event that
# ComfyUI broadcasts when a prompt finishes. Runs aggressive memory hygiene at
# that point — the only place where the prompt is provably complete, async I/O
# subprocesses (ffmpeg) have returned, and ComfyUI itself has finished its own
# bookkeeping. A custom DAG node CAN'T do this reliably because OUTPUT_NODE
# only guarantees inclusion, not last-execution ordering.
#
# Order of ops (per multi-AI debate 2026-04-26):
#   1. torch.cuda.synchronize()   — wait for in-flight GPU kernels
#   2. gc.collect()                — Python GC: tensor destruction returns
#                                    storage to PyTorch's caching allocator
#   3. soft_empty_cache(force=True) — release allocator pool back to driver;
#                                    soft variant adds ipc_collect over raw
#                                    torch.cuda.empty_cache. force=True
#                                    overrides ComfyUI's "I have runway" guard.
#   4. EmptyWorkingSet (Windows)   — release working-set pages back to OS;
#                                    Windows has no malloc_trim equivalent so
#                                    this is the only mechanism.
#
# We deliberately do NOT call unload_all_models() — the user's batch workflows
# rely on models staying resident across queue runs (overnight sweeps would
# otherwise pay 60s+ model reload per render). The recoverable memory we're
# after is intermediate tensors + Python heap fragmentation, not model weights.
#
# Disable via env var:
#   NV_FLUSH_DISABLE=1            — disable entire hook
#   NV_FLUSH_DISABLE_WS=1         — disable EmptyWorkingSet only (keep gc/cuda)
try:
    import sys
    import gc
    import logging
    import ctypes

    _flush_log = logging.getLogger("NV_Comfy_Utils.flush")
    _flush_disable = os.environ.get("NV_FLUSH_DISABLE", "").strip() == "1"
    _flush_ws_disable = os.environ.get("NV_FLUSH_DISABLE_WS", "").strip() == "1"

    if _flush_disable:
        print("[NV_Comfy_Utils] Memory flush hook DISABLED (NV_FLUSH_DISABLE=1)")
    else:
        # Pre-resolve Windows API entry points once at module load.
        _empty_working_set = None
        _get_current_process = None
        if sys.platform == "win32" and not _flush_ws_disable:
            try:
                _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
                _psapi = ctypes.WinDLL("psapi", use_last_error=True)
                _get_current_process = _kernel32.GetCurrentProcess
                _get_current_process.restype = ctypes.c_void_p
                _get_current_process.argtypes = []
                _empty_working_set = _psapi.EmptyWorkingSet
                _empty_working_set.restype = ctypes.c_int   # BOOL
                _empty_working_set.argtypes = [ctypes.c_void_p]  # HANDLE
            except Exception as e:
                print(f"[NV_Comfy_Utils] Warning: could not bind Windows EmptyWorkingSet: {e}")
                _empty_working_set = None

        def _nv_post_prompt_flush():
            """Run the flush sequence. Called by hook on prompt-complete event."""
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except Exception as e:
                _flush_log.debug(f"cuda.synchronize skipped: {e}")

            try:
                collected = gc.collect()
            except Exception as e:
                _flush_log.debug(f"gc.collect skipped: {e}")
                collected = -1

            try:
                # soft_empty_cache(force=True) is the ComfyUI-native abstraction
                # — it wraps torch.cuda.empty_cache() AND torch.cuda.ipc_collect().
                # force=True overrides ComfyUI's "I think I have runway" guard so
                # it actually empties even when free VRAM looks fine.
                import comfy.model_management
                comfy.model_management.soft_empty_cache(force=True)
            except Exception as e:
                _flush_log.debug(f"soft_empty_cache skipped: {e}")
                # Fall back to raw if soft variant unavailable in older ComfyUI
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

            ws_status = "skipped"
            if sys.platform == "win32" and _empty_working_set is not None:
                try:
                    h = _get_current_process()
                    ok = _empty_working_set(h)
                    if ok:
                        ws_status = "trimmed"
                    else:
                        err = ctypes.get_last_error()
                        ws_status = f"failed(err={err})"
                except Exception as e:
                    ws_status = f"error:{type(e).__name__}"

            print(f"[NV_Comfy_Utils] post-prompt flush: gc={collected} | ws={ws_status}")

        # Install hook on PromptServer. ComfyUI broadcasts an "executing" event
        # with data={"node": None, ...} when the entire prompt finishes. Wrap
        # send_sync so we observe that event and trigger flush.
        from server import PromptServer

        _server_inst = PromptServer.instance
        if not hasattr(_server_inst, "_nv_flush_installed"):
            _orig_send_sync = _server_inst.send_sync

            def _hijacked_send_sync(event, data, sid=None):
                try:
                    if event == "executing" and isinstance(data, dict) and data.get("node") is None:
                        _nv_post_prompt_flush()
                except Exception as e:
                    print(f"[NV_Comfy_Utils] flush hook error: {e}")
                return _orig_send_sync(event, data, sid)

            _server_inst.send_sync = _hijacked_send_sync
            _server_inst._nv_flush_installed = True
            ws_state = "off (env or non-Windows)" if _empty_working_set is None else "on"
            print(f"[NV_Comfy_Utils] Memory flush hook installed (EmptyWorkingSet: {ws_state})")
except Exception as e:
    print(f"[NV_Comfy_Utils] Warning: could not install memory flush hook: {e}")


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

