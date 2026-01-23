"""
Workflow Memory Monitor

Runtime measurement of VRAM and system RAM usage.
Place anywhere in workflow - no "start" or "end" needed.

Key insight: torch.cuda.max_memory_allocated() tracks peak across
the entire ComfyUI session, so the peak is accurate regardless of
when this node executes in the DAG.
"""

import torch
import psutil
import os
import json
from datetime import datetime
import folder_paths


class NV_MemoryReport:
    """
    Reports actual VRAM and system RAM usage.

    Place ANYWHERE in workflow - execution order doesn't matter.
    Peak memory reflects the highest point across the entire ComfyUI session.

    For per-run peaks, enable reset_after_report to clear counters after each report.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "trigger": ("*", {"tooltip": "Optional - connect to any output to control execution order"}),
                "log_to_file": ("BOOLEAN", {"default": True, "tooltip": "Write stats to output/memory_logs/"}),
                "reset_after_report": ("BOOLEAN", {"default": False, "tooltip": "Reset peak counter after report (for per-run tracking)"}),
                "workflow_name": ("STRING", {"default": "", "tooltip": "Optional identifier in logs"}),
            }
        }

    RETURN_TYPES = ("STRING", "FLOAT", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("report", "vram_peak_gb", "vram_total_gb", "ram_used_gb", "ram_total_gb")
    OUTPUT_NODE = True
    FUNCTION = "measure_memory"
    CATEGORY = "NV_Utils/Debug"
    DESCRIPTION = "Reports VRAM/RAM usage. Place anywhere in workflow. Returns data in API response for deployment scaling."

    def measure_memory(self, trigger=None, log_to_file=True, reset_after_report=False, workflow_name=""):
        # VRAM measurement
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            vram_allocated = torch.cuda.memory_allocated(device) / 1024**3
            vram_reserved = torch.cuda.memory_reserved(device) / 1024**3
            vram_total = torch.cuda.get_device_properties(device).total_memory / 1024**3
            vram_peak = torch.cuda.max_memory_allocated(device) / 1024**3
            gpu_name = torch.cuda.get_device_name(device)
        else:
            vram_allocated = vram_reserved = vram_total = vram_peak = 0.0
            gpu_name = "N/A (CUDA not available)"

        # System RAM measurement
        ram = psutil.virtual_memory()
        ram_used = ram.used / 1024**3
        ram_total = ram.total / 1024**3
        ram_available = ram.available / 1024**3

        # Calculate recommendations
        vram_utilization = (vram_peak / vram_total * 100) if vram_total > 0 else 0
        ram_utilization = (ram_used / ram_total * 100) if ram_total > 0 else 0
        recommended_vram = max(8, int(vram_peak * 1.3))  # 30% headroom
        recommended_ram = max(16, int(ram_used * 1.3))  # 30% headroom, minimum 16GB

        # Build report
        report_lines = [
            "",
            "=" * 50,
            "WORKFLOW MEMORY REPORT",
            "=" * 50,
            f"GPU: {gpu_name}",
            "",
            "VRAM:",
            f"  Currently Allocated: {vram_allocated:.2f} GB",
            f"  Session Peak:        {vram_peak:.2f} GB",
            f"  Reserved Pool:       {vram_reserved:.2f} GB",
            f"  Total Available:     {vram_total:.2f} GB",
            f"  Peak Utilization:    {vram_utilization:.1f}%",
            "",
            "System RAM:",
            f"  Used:        {ram_used:.2f} GB",
            f"  Available:   {ram_available:.2f} GB",
            f"  Total:       {ram_total:.2f} GB",
            f"  Utilization: {ram_utilization:.1f}%",
            "",
            "For API Deployment:",
            f"  Minimum GPU VRAM:    {vram_peak:.1f} GB (observed peak)",
            f"  Recommended GPU:     {recommended_vram} GB (with 30% headroom)",
            f"  Minimum System RAM:  {ram_used:.1f} GB (observed usage)",
            f"  Recommended RAM:     {recommended_ram} GB (with 30% headroom)",
            "=" * 50,
        ]

        if reset_after_report:
            report_lines.append("(Peak counter will reset after this report)")

        report_lines.append("")
        report = "\n".join(report_lines)
        print(report)

        # Log to file for aggregation
        if log_to_file:
            try:
                log_dir = os.path.join(folder_paths.get_output_directory(), "memory_logs")
                os.makedirs(log_dir, exist_ok=True)
                log_file = os.path.join(log_dir, "workflow_memory.jsonl")

                log_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "workflow_name": workflow_name if workflow_name else "unnamed",
                    "gpu_name": gpu_name,
                    "vram_peak_gb": round(vram_peak, 3),
                    "vram_allocated_gb": round(vram_allocated, 3),
                    "vram_total_gb": round(vram_total, 3),
                    "vram_utilization_pct": round(vram_utilization, 1),
                    "recommended_vram_gb": recommended_vram,
                    "ram_used_gb": round(ram_used, 3),
                    "ram_available_gb": round(ram_available, 3),
                    "ram_total_gb": round(ram_total, 3),
                    "ram_utilization_pct": round(ram_utilization, 1),
                    "recommended_ram_gb": recommended_ram,
                }

                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_entry) + "\n")

                print(f"[NV_MemoryReport] Logged to {log_file}")
            except Exception as e:
                print(f"[NV_MemoryReport] Warning: Could not write log: {e}")

        # Reset peak counter if requested (for per-run tracking)
        if reset_after_report and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            print("[NV_MemoryReport] Peak memory counter reset for next run")

        # Return with API-compatible format
        return {
            "ui": {
                "memory_report": [{
                    "gpu_name": gpu_name,
                    "vram_peak_gb": round(vram_peak, 3),
                    "vram_total_gb": round(vram_total, 3),
                    "vram_utilization_pct": round(vram_utilization, 1),
                    "recommended_vram_gb": recommended_vram,
                    "ram_used_gb": round(ram_used, 3),
                    "ram_total_gb": round(ram_total, 3),
                    "ram_utilization_pct": round(ram_utilization, 1),
                    "recommended_ram_gb": recommended_ram,
                }]
            },
            "result": (report, vram_peak, vram_total, ram_used, ram_total)
        }


# Node registration
NODE_CLASS_MAPPINGS = {
    "NV_MemoryReport": NV_MemoryReport,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_MemoryReport": "NV Memory Report",
}
