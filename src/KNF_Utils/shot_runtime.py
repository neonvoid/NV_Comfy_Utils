"""Runtime telemetry — prompt-scoped timing + non-destructive memory peak.

Hooks ComfyUI's PromptServer to record start time per prompt. Memory peak
is read non-destructively via torch's session-wide max_memory_allocated;
the destructive reset path that NV_MemoryReport had is gone.
"""

import time

import torch

try:
    import server  # ComfyUI's server module
    _SERVER_AVAILABLE = True
except Exception:
    _SERVER_AVAILABLE = False

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except Exception:
    _PSUTIL_AVAILABLE = False


# Prompt-scoped timing — single per-prompt start time captured by hook.
_prompt_start_times = {}


def _on_prompt_submitted(json_data):
    """Hook fires when a prompt is queued. Records start time."""
    _prompt_start_times["_latest"] = time.time()
    return json_data


def install_hook():
    """Install the PromptServer hook. Idempotent. Returns True on success.

    Unlike the bare-except in gen_timer.py, this fails loudly so silent
    zero-time records can't poison the agent corpus.
    """
    if not _SERVER_AVAILABLE:
        print("[shot_runtime] WARNING: server module not importable — gen_time will read 0")
        return False
    try:
        instance = server.PromptServer.instance
        if instance is None:
            print("[shot_runtime] WARNING: PromptServer.instance is None — gen_time will read 0")
            return False
        instance.add_on_prompt_handler(_on_prompt_submitted)
        return True
    except Exception as e:
        print(f"[shot_runtime] WARNING: failed to install prompt hook ({type(e).__name__}: {e})")
        return False


# Install at import time
_HOOK_INSTALLED = install_hook()


def get_gen_time_sec():
    """Seconds since the most recent prompt was submitted. 0.0 if hook
    failed or no prompt has run yet (caller should treat as None).
    """
    start = _prompt_start_times.get("_latest")
    if start is None:
        return 0.0
    return max(0.0, time.time() - start)


def get_memory_snapshot():
    """Non-destructive memory snapshot. Never resets peak counters."""
    snap = {
        "vram_allocated_gb": None,
        "vram_reserved_gb": None,
        "vram_peak_gb": None,
        "vram_total_gb": None,
        "ram_used_gb": None,
        "ram_total_gb": None,
        "gpu_name": None,
        "hook_installed": _HOOK_INSTALLED,
    }
    if torch.cuda.is_available():
        try:
            dev = torch.cuda.current_device()
            snap["vram_allocated_gb"] = round(torch.cuda.memory_allocated(dev) / 1024**3, 3)
            snap["vram_reserved_gb"]  = round(torch.cuda.memory_reserved(dev) / 1024**3, 3)
            snap["vram_peak_gb"]      = round(torch.cuda.max_memory_allocated(dev) / 1024**3, 3)
            snap["vram_total_gb"]     = round(torch.cuda.get_device_properties(dev).total_memory / 1024**3, 3)
            snap["gpu_name"]          = torch.cuda.get_device_name(dev)
        except Exception:
            pass

    if _PSUTIL_AVAILABLE:
        try:
            ram = psutil.virtual_memory()
            snap["ram_used_gb"]  = round(ram.used / 1024**3, 3)
            snap["ram_total_gb"] = round(ram.total / 1024**3, 3)
        except Exception:
            pass

    return snap
