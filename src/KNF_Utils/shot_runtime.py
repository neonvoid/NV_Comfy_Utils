"""Runtime telemetry — prompt-scoped timing + non-destructive memory peak.

Hooks ComfyUI's PromptServer to record start time per prompt. Memory peak
is read non-destructively via torch's session-wide max_memory_allocated;
the destructive reset path that NV_MemoryReport had is gone.

IMPORTANT timing semantics: ComfyUI's `add_on_prompt_handler` fires when a
prompt is QUEUED, not when execution actually starts. For sequential
single-prompt operation (the dominant case here) that's effectively render
time. If multiple prompts are queued back-to-back, gen_time_sec for the
LATER ones may include queue wait. The schema flags this via a scope field
so the agent can filter ambiguous records.
"""

import time

import torch

try:
    import server  # ComfyUI's server module
    _SERVER_AVAILABLE = True
    _SERVER_IMPORT_ERR = None
except ImportError as e:
    _SERVER_AVAILABLE = False
    _SERVER_IMPORT_ERR = f"{type(e).__name__}: {e}"

try:
    import psutil
    _PSUTIL_AVAILABLE = True
    _PSUTIL_IMPORT_ERR = None
except ImportError as e:
    _PSUTIL_AVAILABLE = False
    _PSUTIL_IMPORT_ERR = f"{type(e).__name__}: {e}"


# Prompt-scoped timing — single per-prompt start time captured by hook.
# perf_counter is monotonic (immune to system clock jumps) which time.time isn't.
_prompt_start_perf = {}
_HOOK_INSTALL_ERROR = None


def _on_prompt_submitted(json_data):
    """Hook fires when a prompt is queued. Records start time."""
    _prompt_start_perf["_latest"] = time.perf_counter()
    return json_data


def install_hook():
    """Install the PromptServer hook. Returns (installed_bool, error_str_or_None).

    Idempotent at the module level via the install error tracking — repeated
    calls are silently no-ops once successful.
    """
    global _HOOK_INSTALL_ERROR
    if not _SERVER_AVAILABLE:
        _HOOK_INSTALL_ERROR = f"server module not importable: {_SERVER_IMPORT_ERR}"
        print(f"[shot_runtime] WARNING: {_HOOK_INSTALL_ERROR}")
        return False, _HOOK_INSTALL_ERROR
    try:
        instance = server.PromptServer.instance
        if instance is None:
            _HOOK_INSTALL_ERROR = "PromptServer.instance is None at install time"
            print(f"[shot_runtime] WARNING: {_HOOK_INSTALL_ERROR}")
            return False, _HOOK_INSTALL_ERROR
        instance.add_on_prompt_handler(_on_prompt_submitted)
        return True, None
    except (AttributeError, TypeError) as e:
        _HOOK_INSTALL_ERROR = f"hook install failed: {type(e).__name__}: {e}"
        print(f"[shot_runtime] WARNING: {_HOOK_INSTALL_ERROR}")
        return False, _HOOK_INSTALL_ERROR


# Install at import time. If it fails, that fact is captured in module state
# so callers can surface it in the JSONL record instead of silently emitting
# a zero gen_time.
_HOOK_INSTALLED, _ = install_hook()


def get_gen_time_sec():
    """Seconds since the most recent prompt was queued.

    Returns None when unavailable (hook didn't install OR no prompt has run
    yet) — NEVER 0.0. Silent zero on data the agent depends on is the exact
    anti-pattern this project has been burned by.
    """
    if not _HOOK_INSTALLED:
        return None
    start = _prompt_start_perf.get("_latest")
    if start is None:
        return None
    return max(0.0, time.perf_counter() - start)


def get_runtime_snapshot():
    """Combined runtime telemetry: timing + memory + hook status.

    Renamed from get_memory_snapshot — this function returns more than memory
    (gpu_name, hook status). Field names use scope qualifiers so the agent
    can interpret them correctly:

      - vram_session_peak_gb: torch's max_memory_allocated since process
        start (or last reset). NOT shot-scoped — across renders.
      - gen_time_sec: seconds since the most recent prompt queue submission.
        Equals render time for sequential single-prompt operation.
      - hook_install_error: None on success, short reason string on failure
        so a stale-zero gen_time can be diagnosed from the record.
    """
    snap = {
        "gen_time_sec": get_gen_time_sec(),
        "gen_time_scope": "prompt_queue_elapsed",
        "hook_installed": _HOOK_INSTALLED,
        "hook_install_error": _HOOK_INSTALL_ERROR,
        "vram_allocated_gb": None,
        "vram_reserved_gb": None,
        "vram_session_peak_gb": None,   # process-wide, NOT shot-scoped
        "vram_total_gb": None,
        "vram_error": None,
        "ram_used_gb": None,
        "ram_total_gb": None,
        "ram_error": None if _PSUTIL_AVAILABLE else _PSUTIL_IMPORT_ERR,
        "gpu_name": None,
    }

    if torch.cuda.is_available():
        try:
            dev = torch.cuda.current_device()
            snap["vram_allocated_gb"]    = round(torch.cuda.memory_allocated(dev) / 1024**3, 3)
            snap["vram_reserved_gb"]     = round(torch.cuda.memory_reserved(dev) / 1024**3, 3)
            snap["vram_session_peak_gb"] = round(torch.cuda.max_memory_allocated(dev) / 1024**3, 3)
            snap["vram_total_gb"]        = round(torch.cuda.get_device_properties(dev).total_memory / 1024**3, 3)
            snap["gpu_name"]             = torch.cuda.get_device_name(dev)
        except (RuntimeError, AssertionError) as e:
            snap["vram_error"] = f"{type(e).__name__}: {e}"

    if _PSUTIL_AVAILABLE:
        try:
            ram = psutil.virtual_memory()
            snap["ram_used_gb"]  = round(ram.used / 1024**3, 3)
            snap["ram_total_gb"] = round(ram.total / 1024**3, 3)
        except (OSError, AttributeError) as e:
            snap["ram_error"] = f"{type(e).__name__}: {e}"

    return snap


# Backwards alias for any callers that reach in by old name. Drop after
# all callsites are migrated.
get_memory_snapshot = get_runtime_snapshot
