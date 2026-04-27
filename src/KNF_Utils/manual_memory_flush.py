"""
NV Manual Memory Flush - User-controlled cleanup between heavy pipeline stages.

The auto-flush hook installed at __init__ (D-119) fires only at PROMPT END.
Within a single prompt, heavy operations (multiple TH/CCF/VAE encodes/decodes)
accumulate transient allocations that aren't released until the prompt finishes.
This causes Windows access violations when a second heavy operation tries to
allocate before the first's intermediate tensors are reclaimed.

This node lets you manually trigger the same cleanup sequence the post-prompt
hook runs, but at any point in your workflow. Place between heavy operations:

  KSampler → ManualMemoryFlush → SecondVAE Decode → ManualMemoryFlush → TH ...

Passthrough design — wire any IMAGE/LATENT/* through it; it runs cleanup THEN
forwards the input unchanged. ComfyUI executes nodes in dependency order so
the flush happens after the upstream node completes its work.

Same operations as the auto-flush hook (D-119/D-120):
  1. torch.cuda.synchronize()           — wait for in-flight GPU kernels
  2. gc.collect()                        — Python GC (frees PyTorch caching alloc)
  3. soft_empty_cache(force=True)        — ComfyUI native CUDA cache release
  4. EmptyWorkingSet (Windows only)      — release working-set pages to OS

Models stay resident (per D-120) — no cleanup_models() call. Recoverable memory
is intermediate tensors + Python heap fragmentation + Windows working-set bloat,
NOT model weights.
"""

import gc
import os
import sys


class NV_ManualMemoryFlush:
    """User-controlled memory cleanup between heavy pipeline stages.

    Place between operations that produce/consume large transient tensors
    (multiple VAE encodes/decodes, multiple TH passes, multiple stitches).
    Passes input through unchanged; cleanup happens as a side effect during
    execution.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "passthrough": ("*", {
                    "tooltip": "Any input — IMAGE, LATENT, MASK, STITCHER, etc. Passes through "
                               "unchanged after cleanup runs. Use this to chain the flush into "
                               "your dependency graph between two heavy operations."
                }),
            },
            "optional": {
                "do_synchronize": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "torch.cuda.synchronize() before cleanup. Waits for in-flight GPU "
                               "kernels to complete. Required for accurate measurement; rarely "
                               "harmful but adds a tiny GPU sync cost."
                }),
                "do_gc": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Run gc.collect(). Triggers Python GC which lets PyTorch's caching "
                               "allocator reclaim freed tensor storage. Cheap but worth doing."
                }),
                "do_cuda_cache": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Run comfy.model_management.soft_empty_cache(force=True). Releases "
                               "the PyTorch caching allocator pool back to the CUDA driver. force=True "
                               "overrides ComfyUI's 'I have runway' guard. Includes ipc_collect."
                }),
                "do_working_set": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Windows-only: EmptyWorkingSet() releases working-set pages to OS. "
                               "Critical for back-to-back heavy ops on Windows where the working "
                               "set bloats monotonically. No effect on non-Windows. Disable if "
                               "you observe slowdowns from frequent OS page reloads."
                }),
                "verbose": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Print cleanup diagnostic to console. False = silent operation."
                }),
            },
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("passthrough",)
    FUNCTION = "execute"
    CATEGORY = "NV_Utils/Debug"
    OUTPUT_NODE = False
    DESCRIPTION = (
        "Manual memory cleanup between heavy pipeline stages. Same operations as "
        "the auto-flush hook (synchronize, gc, soft_empty_cache, EmptyWorkingSet) "
        "but triggered mid-prompt. Place between two heavy ops (e.g., between two "
        "TH passes, or between VAE decode and a second VAE encode) to prevent "
        "memory pressure access violations. Passes input through unchanged."
    )

    def execute(self, passthrough, do_synchronize=True, do_gc=True,
                do_cuda_cache=True, do_working_set=True, verbose=True):
        TAG = "[NV_ManualMemoryFlush]"
        results = []

        # 1. CUDA synchronize
        if do_synchronize:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    results.append("sync=ok")
                else:
                    results.append("sync=skipped(no_cuda)")
            except Exception as e:
                results.append(f"sync=err({type(e).__name__})")

        # 2. Python GC
        gc_count = -1
        if do_gc:
            try:
                gc_count = gc.collect()
                results.append(f"gc={gc_count}")
            except Exception as e:
                results.append(f"gc=err({type(e).__name__})")

        # 3. CUDA cache release (ComfyUI native)
        if do_cuda_cache:
            try:
                import comfy.model_management
                comfy.model_management.soft_empty_cache(force=True)
                results.append("cuda_cache=ok")
            except Exception as e:
                # Fallback to raw torch.cuda.empty_cache
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    results.append("cuda_cache=fallback")
                except Exception:
                    results.append(f"cuda_cache=err({type(e).__name__})")

        # 4. Windows working-set trim
        if do_working_set and sys.platform == "win32":
            try:
                import ctypes
                k32 = ctypes.WinDLL("kernel32", use_last_error=True)
                psapi = ctypes.WinDLL("psapi", use_last_error=True)
                hproc = k32.GetCurrentProcess()
                k32.GetCurrentProcess.restype = ctypes.c_void_p
                psapi.EmptyWorkingSet.restype = ctypes.c_int
                psapi.EmptyWorkingSet.argtypes = [ctypes.c_void_p]
                ok = psapi.EmptyWorkingSet(hproc)
                results.append("ws=trimmed" if ok else "ws=failed")
            except Exception as e:
                results.append(f"ws=err({type(e).__name__})")
        elif do_working_set:
            results.append("ws=skipped(non_windows)")

        if verbose:
            print(f"{TAG} {' | '.join(results)}")

        return (passthrough,)


NODE_CLASS_MAPPINGS = {
    "NV_ManualMemoryFlush": NV_ManualMemoryFlush,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_ManualMemoryFlush": "NV Manual Memory Flush",
}
