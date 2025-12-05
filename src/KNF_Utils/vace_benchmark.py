"""
VACE Benchmark Logger Node

Logs comprehensive VACE workflow metrics to disk for memory analysis and formula derivation.
Captures system info, GPU details, CUDA memory stats, timing, and tensor shapes.
"""

import torch
import json
import time
import sys
import platform
import socket
import uuid
from datetime import datetime
from pathlib import Path
from math import ceil

# Try to import psutil for system memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("[NV_VACEBenchmarkLogger] Warning: psutil not available, system RAM monitoring disabled")


class NV_VACEBenchmarkLogger:
    """
    Logs comprehensive VACE workflow metrics to disk for memory analysis.

    Place this node at different points in your workflow to capture
    VRAM snapshots. Results are appended to a JSON file for analysis.

    Captures:
    - System info (hostname, platform, Python/PyTorch/CUDA versions)
    - GPU details (name, compute capability, memory, SM count)
    - System RAM usage (if psutil available)
    - CUDA memory (allocated, reserved, peak, stats)
    - Timing (session ID, elapsed time between phases)
    - Tensor shapes (if conditioning/latent/model passed through)
    - Context window calculations

    Phases:
    - pre_encode: Before VACE streaming encode starts
    - post_encode: After VACE encode completes (VACE tensors on GPU)
    - pre_sample: Before KSampler starts (model loaded)
    - during_sample: During sampling (if you can capture mid-window)
    - post_sample: After sampling completes (success!)
    - error: If workflow crashes (connect to error handler if possible)
    """

    # Class-level session tracking
    _session_id = None
    _session_start = None
    _last_phase_time = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trigger": ("*",),
                "phase": (["pre_encode", "post_encode", "pre_sample", "during_sample", "post_sample", "error"],),
                "log_file": ("STRING", {
                    "default": "vace_benchmark.json",
                    "tooltip": "Path to JSON log file (relative or absolute)"
                }),
            },
            "optional": {
                # Workflow parameters
                "pixel_frames": ("INT", {
                    "default": 0, "min": 0, "max": 10000,
                    "tooltip": "Total pixel frames in video"
                }),
                "latent_frames": ("INT", {
                    "default": 0, "min": 0, "max": 2500,
                    "tooltip": "Latent frames = (pixel_frames + 3) // 4"
                }),
                "height": ("INT", {
                    "default": 0, "min": 0, "max": 4096, "step": 8,
                    "tooltip": "Video height in pixels"
                }),
                "width": ("INT", {
                    "default": 0, "min": 0, "max": 4096, "step": 8,
                    "tooltip": "Video width in pixels"
                }),
                "context_window_size": ("INT", {
                    "default": 0, "min": 0, "max": 256,
                    "tooltip": "Context window size (pixel frames) used in sampling"
                }),
                "context_overlap": ("INT", {
                    "default": 0, "min": 0, "max": 128,
                    "tooltip": "Context window overlap (pixel frames)"
                }),
                "model_name": ("STRING", {
                    "default": "wan_14b",
                    "tooltip": "Model identifier (wan_14b, wan_1.3b, etc.)"
                }),
                "notes": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Any notes about this run (e.g., 'crashed after 5 windows')"
                }),
                # Tensor inspection inputs
                "conditioning": ("CONDITIONING", {
                    "tooltip": "Pass conditioning to capture VACE tensor shapes"
                }),
                "latent": ("LATENT", {
                    "tooltip": "Pass latent to capture latent tensor shape/dtype"
                }),
                "model": ("MODEL", {
                    "tooltip": "Pass model to detect model size and dtype"
                }),
            }
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("passthrough",)
    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = (False,)
    FUNCTION = "log_metrics"
    CATEGORY = "NV_Utils/benchmark"
    DESCRIPTION = "Logs comprehensive VACE workflow metrics for memory analysis."

    def __init__(self):
        # Initialize session on first use
        if NV_VACEBenchmarkLogger._session_id is None:
            NV_VACEBenchmarkLogger._session_id = uuid.uuid4().hex[:8]
            NV_VACEBenchmarkLogger._session_start = time.time()
            NV_VACEBenchmarkLogger._last_phase_time = time.time()

    def log_metrics(self, trigger, phase, log_file,
                    pixel_frames=0, latent_frames=0, height=0, width=0,
                    context_window_size=0, context_overlap=0,
                    model_name="wan_14b", notes="",
                    conditioning=None, latent=None, model=None):

        current_time = time.time()

        # === SYSTEM INFO ===
        system_info = self._get_system_info()

        # === GPU INFO ===
        gpu_info = self._get_gpu_info()

        # === SYSTEM MEMORY ===
        system_memory = self._get_system_memory()

        # === CUDA MEMORY ===
        cuda_memory = self._get_cuda_memory()

        # === TIMING ===
        timing_info = self._get_timing_info(current_time)

        # Auto-calculate latent frames if not provided
        if latent_frames == 0 and pixel_frames > 0:
            latent_frames = (pixel_frames + 3) // 4

        # Calculate latent dimensions
        latent_height = height // 8 if height > 0 else 0
        latent_width = width // 8 if width > 0 else 0

        # === CONTEXT WINDOWS CALCULATION ===
        context_windows = self._calculate_context_windows(
            latent_frames, latent_height, latent_width,
            context_window_size, context_overlap
        )

        # === TENSOR INSPECTION ===
        tensor_info = self._inspect_tensors(conditioning, latent, model)

        # === CALCULATED ESTIMATES ===
        calculated_estimates = {
            "feat_cache_gb": self._estimate_feat_cache(height, width),
            "vace_tensors_gb": self._estimate_vace_tensors(latent_frames, height, width),
            "forward_activation_gb": self._estimate_forward_activations(height, width),
        }

        # === BUILD LOG ENTRY ===
        entry = {
            "session_id": NV_VACEBenchmarkLogger._session_id,
            "timestamp": datetime.now().isoformat(),
            "phase": phase,

            "system": system_info,
            "gpu": gpu_info,
            "system_memory": system_memory,

            "params": {
                "pixel_frames": pixel_frames,
                "latent_frames": latent_frames,
                "height": height,
                "width": width,
                "latent_height": latent_height,
                "latent_width": latent_width,
                "context_window_size": context_window_size,
                "context_overlap": context_overlap,
                "model_name": model_name,
            },

            "cuda_memory": cuda_memory,
            "context_windows": context_windows,
            "tensors": tensor_info,
            "calculated_estimates": calculated_estimates,
            "timing": timing_info,
            "notes": notes,
        }

        # === DERIVED METRICS ===
        entry["derived"] = self._calculate_derived_metrics(
            phase, cuda_memory, calculated_estimates, context_windows, model_name
        )

        # Update last phase time
        NV_VACEBenchmarkLogger._last_phase_time = current_time

        # === WRITE TO LOG FILE ===
        self._write_log(log_file, entry)

        # === CONSOLE OUTPUT ===
        allocated = cuda_memory.get("allocated_gb", 0)
        total = cuda_memory.get("total_gb", 0)
        peak = cuda_memory.get("max_allocated_gb", 0)
        util = cuda_memory.get("utilization_pct", 0)

        print(f"[Benchmark] {phase}: VRAM {allocated:.2f}/{total:.2f} GB "
              f"({util:.1f}%) | Peak: {peak:.2f} GB")

        return (trigger,)

    def _get_system_info(self):
        """Collect system/software version info."""
        info = {
            "hostname": socket.gethostname(),
            "platform": platform.system(),
            "platform_version": platform.version(),
            "platform_release": platform.release(),
            "python_version": sys.version.split()[0],
            "pytorch_version": torch.__version__,
        }

        # CUDA version
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda or "N/A"
            try:
                info["cudnn_version"] = torch.backends.cudnn.version()
            except:
                info["cudnn_version"] = "N/A"
        else:
            info["cuda_version"] = "N/A"
            info["cudnn_version"] = "N/A"

        return info

    def _get_gpu_info(self):
        """Collect GPU hardware info."""
        if not torch.cuda.is_available():
            return {"available": False, "name": "CPU only"}

        props = torch.cuda.get_device_properties(0)

        info = {
            "available": True,
            "name": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_gb": round(props.total_memory / 1024**3, 2),
            "multi_processor_count": props.multi_processor_count,
            "gpu_count": torch.cuda.device_count(),
        }

        # Try to get driver version via nvidia-smi (optional)
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                info["driver_version"] = result.stdout.strip().split('\n')[0]
        except:
            info["driver_version"] = "N/A"

        return info

    def _get_system_memory(self):
        """Collect system RAM info via psutil."""
        if not PSUTIL_AVAILABLE:
            return {"available": False, "reason": "psutil not installed"}

        try:
            vm = psutil.virtual_memory()
            return {
                "available": True,
                "total_ram_gb": round(vm.total / 1024**3, 2),
                "available_ram_gb": round(vm.available / 1024**3, 2),
                "used_ram_gb": round(vm.used / 1024**3, 2),
                "ram_percent_used": vm.percent,
            }
        except Exception as e:
            return {"available": False, "reason": str(e)}

    def _get_cuda_memory(self):
        """Collect detailed CUDA memory stats."""
        if not torch.cuda.is_available():
            return {"available": False}

        torch.cuda.synchronize()

        props = torch.cuda.get_device_properties(0)
        total = props.total_memory / 1024**3
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        max_reserved = torch.cuda.max_memory_reserved() / 1024**3

        info = {
            "available": True,
            "allocated_gb": round(allocated, 3),
            "reserved_gb": round(reserved, 3),
            "max_allocated_gb": round(max_allocated, 3),
            "max_reserved_gb": round(max_reserved, 3),
            "total_gb": round(total, 2),
            "free_gb": round(total - allocated, 3),
            "utilization_pct": round(allocated / total * 100, 1) if total > 0 else 0,
        }

        # Extended memory stats
        try:
            stats = torch.cuda.memory_stats()
            info["memory_stats"] = {
                "active_bytes": stats.get("active_bytes.all.current", 0),
                "inactive_split_bytes": stats.get("inactive_split_bytes.all.current", 0),
                "num_alloc_retries": stats.get("num_alloc_retries", 0),
                "num_ooms": stats.get("num_ooms", 0),
            }
        except:
            pass

        return info

    def _get_timing_info(self, current_time):
        """Calculate timing/session info."""
        elapsed_since_start = current_time - NV_VACEBenchmarkLogger._session_start
        elapsed_since_last = current_time - NV_VACEBenchmarkLogger._last_phase_time

        return {
            "session_start_iso": datetime.fromtimestamp(
                NV_VACEBenchmarkLogger._session_start
            ).isoformat(),
            "elapsed_since_start_sec": round(elapsed_since_start, 2),
            "elapsed_since_last_phase_sec": round(elapsed_since_last, 2),
        }

    def _calculate_context_windows(self, latent_frames, latent_height, latent_width,
                                    context_window_size, context_overlap):
        """Calculate context window metrics."""
        if context_window_size <= 0 or latent_frames <= 0:
            return {}

        window_latent_frames = (context_window_size + 3) // 4
        stride = context_window_size - context_overlap
        stride_latent = (stride + 3) // 4 if stride > 0 else 1

        if latent_frames > window_latent_frames and stride_latent > 0:
            total_windows = ceil((latent_frames - window_latent_frames) / stride_latent) + 1
        else:
            total_windows = 1

        tokens_per_window = window_latent_frames * latent_height * latent_width if latent_height > 0 else 0
        total_tokens = latent_frames * latent_height * latent_width if latent_height > 0 else 0

        return {
            "window_latent_frames": window_latent_frames,
            "effective_stride_pixels": stride,
            "effective_stride_latent": stride_latent,
            "total_windows": total_windows,
            "tokens_per_window": tokens_per_window,
            "total_tokens": total_tokens,
        }

    def _inspect_tensors(self, conditioning, latent, model):
        """Inspect actual tensor shapes if provided."""
        info = {}

        # Latent tensor
        if latent is not None:
            try:
                samples = latent.get("samples") if isinstance(latent, dict) else None
                if samples is not None:
                    info["latent_shape"] = list(samples.shape)
                    info["latent_dtype"] = str(samples.dtype)
                    info["latent_device"] = str(samples.device)
                    info["latent_size_mb"] = round(
                        samples.numel() * samples.element_size() / 1024**2, 2
                    )
            except Exception as e:
                info["latent_error"] = str(e)

        # Conditioning (VACE tensors)
        if conditioning is not None:
            try:
                # Conditioning is typically a list of tuples
                for i, cond in enumerate(conditioning):
                    if isinstance(cond, (list, tuple)) and len(cond) >= 2:
                        cond_dict = cond[1] if isinstance(cond[1], dict) else {}

                        # VACE frames
                        vace_frames = cond_dict.get("vace_frames")
                        if vace_frames is not None:
                            if isinstance(vace_frames, list) and len(vace_frames) > 0:
                                vf = vace_frames[0]
                                info["vace_frames_shape"] = list(vf.shape)
                                info["vace_frames_dtype"] = str(vf.dtype)
                                info["vace_frames_device"] = str(vf.device)
                                info["vace_frames_size_mb"] = round(
                                    vf.numel() * vf.element_size() / 1024**2, 2
                                )

                        # VACE mask
                        vace_mask = cond_dict.get("vace_mask")
                        if vace_mask is not None:
                            if isinstance(vace_mask, list) and len(vace_mask) > 0:
                                vm = vace_mask[0]
                                info["vace_mask_shape"] = list(vm.shape)
                                info["vace_mask_dtype"] = str(vm.dtype)
                                info["vace_mask_device"] = str(vm.device)
                                info["vace_mask_size_mb"] = round(
                                    vm.numel() * vm.element_size() / 1024**2, 2
                                )

                        # VACE strength
                        vace_strength = cond_dict.get("vace_strength")
                        if vace_strength is not None:
                            info["vace_strength"] = vace_strength[0] if isinstance(vace_strength, list) else vace_strength

                        break  # Only process first conditioning
            except Exception as e:
                info["conditioning_error"] = str(e)

        # Model info
        if model is not None:
            try:
                # Try to get model size
                model_obj = model.model if hasattr(model, 'model') else model
                if hasattr(model_obj, 'parameters'):
                    total_params = sum(p.numel() for p in model_obj.parameters())
                    info["model_params"] = total_params
                    info["model_params_millions"] = round(total_params / 1e6, 1)

                    # Estimate size based on first param dtype
                    first_param = next(model_obj.parameters(), None)
                    if first_param is not None:
                        info["model_dtype"] = str(first_param.dtype)
                        bytes_per_param = first_param.element_size()
                        info["model_size_gb"] = round(
                            total_params * bytes_per_param / 1024**3, 2
                        )
            except Exception as e:
                info["model_error"] = str(e)

        return info if info else None

    def _calculate_derived_metrics(self, phase, cuda_memory, calculated_estimates,
                                   context_windows, model_name):
        """Calculate derived analysis metrics."""
        derived = {}

        allocated = cuda_memory.get("allocated_gb", 0)
        vace_estimate = calculated_estimates.get("vace_tensors_gb", 0)

        # Model size estimate based on name
        model_size_gb = 32 if "14b" in model_name.lower() else 5.2

        if phase in ["post_encode", "pre_sample", "during_sample", "post_sample"]:
            # VRAM minus VACE estimate = model + activations
            derived["vram_minus_vace_gb"] = round(allocated - vace_estimate, 3)

            if phase in ["during_sample", "post_sample"]:
                # Estimate activation memory
                activation_mem = max(0, allocated - model_size_gb - vace_estimate)
                derived["estimated_activation_gb"] = round(activation_mem, 3)

                tokens = context_windows.get("tokens_per_window", 0)
                if tokens > 0:
                    derived["activation_per_million_tokens_gb"] = round(
                        activation_mem / (tokens / 1_000_000), 4
                    )

        return derived if derived else None

    def _estimate_feat_cache(self, H, W):
        """Calculate exact feat_cache size for WAN VAE encoder."""
        if H == 0 or W == 0:
            return 0

        CACHE_T = 2
        dtype_bytes = 4  # fp32

        conv1 = 3 * CACHE_T * H * W
        stage0 = 4 * 128 * CACHE_T * H * W + 128 * 1 * (H // 2) * (W // 2)
        stage1 = 4 * 256 * CACHE_T * (H // 2) * (W // 2) + 256 * 1 * (H // 4) * (W // 4)
        stage2 = 4 * 512 * CACHE_T * (H // 4) * (W // 4)
        stage3 = 4 * 512 * CACHE_T * (H // 8) * (W // 8)
        middle = 4 * 512 * CACHE_T * (H // 8) * (W // 8)
        head = 512 * CACHE_T * (H // 8) * (W // 8)

        total_elements = conv1 + stage0 + stage1 + stage2 + stage3 + middle + head
        total_bytes = total_elements * dtype_bytes

        return round(total_bytes / 1024**3, 3)

    def _estimate_vace_tensors(self, latent_frames, H, W):
        """Calculate VACE output tensor size."""
        if latent_frames == 0 or H == 0 or W == 0:
            return 0

        latent_H = H // 8
        latent_W = W // 8
        dtype_bytes = 4  # fp32

        control = 1 * 32 * latent_frames * latent_H * latent_W * dtype_bytes
        mask = 1 * 64 * latent_frames * latent_H * latent_W * dtype_bytes

        total_bytes = control + mask
        return round(total_bytes / 1024**3, 3)

    def _estimate_forward_activations(self, H, W, chunk_frames=4):
        """Estimate peak forward pass activation memory."""
        if H == 0 or W == 0:
            return 0

        dtype_bytes = 4
        stage0_peak = 2 * 3 * 128 * chunk_frames * H * W * dtype_bytes
        total_bytes = stage0_peak * 1.5

        return round(total_bytes / 1024**3, 3)

    def _write_log(self, log_file, entry):
        """Write entry to log file."""
        import folder_paths

        log_path = Path(log_file)
        if not log_path.is_absolute():
            output_dir = folder_paths.get_output_directory()
            log_path = Path(output_dir) / log_file

        existing = []
        if log_path.exists():
            try:
                with open(log_path, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
                if not isinstance(existing, list):
                    existing = [existing]
            except (json.JSONDecodeError, Exception) as e:
                print(f"[Benchmark] Warning: Could not read existing log: {e}")
                existing = []

        existing.append(entry)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(existing, f, indent=2)

        print(f"[Benchmark] Logged to: {log_path}")


# Utility function to reset session (e.g., for new workflow run)
def reset_benchmark_session():
    """Reset the benchmark session ID and timing."""
    NV_VACEBenchmarkLogger._session_id = uuid.uuid4().hex[:8]
    NV_VACEBenchmarkLogger._session_start = time.time()
    NV_VACEBenchmarkLogger._last_phase_time = time.time()
    print(f"[Benchmark] New session started: {NV_VACEBenchmarkLogger._session_id}")


# Node registration
NODE_CLASS_MAPPINGS = {
    "NV_VACEBenchmarkLogger": NV_VACEBenchmarkLogger,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_VACEBenchmarkLogger": "NV VACE Benchmark Logger",
}
