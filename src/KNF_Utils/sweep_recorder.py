"""
NV Sweep Iteration Recorder

Records completion status for sweep iterations and updates the sweep plan JSON.
Use this to track which iterations have been completed, failed, or skipped.

Use Case:
- Track progress through a parameter sweep
- Record output paths for each iteration
- Mark failed iterations for retry
- Log optional metrics for comparison

Workflow:
1. NV_SweepPlanner → exports sweep_plan.json (run once)
2. NV_SweepIterationLoader → loads params for iteration N
3. [Your sampling workflow] → processes with loaded params
4. NV_SweepIterationRecorder → updates sweep_plan.json with status (this node)
"""

import json
import os
from datetime import datetime
from comfy.comfy_types.node_typing import IO


class NV_SweepIterationRecorder:
    """
    Records completion status for a sweep iteration.

    Updates the sweep plan JSON with status and optional metadata.
    Pass-through trigger allows chaining in workflow.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trigger": (IO.ANY, {
                    "tooltip": "Connect to execution chain (e.g., save node output)"
                }),
                "sweep_json_path": ("STRING", {
                    "default": "sweep_plan.json",
                    "tooltip": "Path to the sweep_plan.json file"
                }),
                "iteration_id": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 999999,
                    "tooltip": "Iteration ID to record (from loader output)"
                }),
            },
            "optional": {
                "status": (["completed", "failed", "skipped"], {
                    "default": "completed",
                    "tooltip": "Status to record for this iteration"
                }),
                "output_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to the output file for this iteration"
                }),
                "notes": ("STRING", {
                    "default": "",
                    "tooltip": "Optional notes about this iteration"
                }),
                "metric_1_name": ("STRING", {
                    "default": "",
                    "tooltip": "Name for metric 1 (e.g., 'psnr', 'ssim')"
                }),
                "metric_1_value": ("FLOAT", {
                    "default": 0.0,
                    "tooltip": "Value for metric 1"
                }),
                "metric_2_name": ("STRING", {
                    "default": "",
                    "tooltip": "Name for metric 2"
                }),
                "metric_2_value": ("FLOAT", {
                    "default": 0.0,
                    "tooltip": "Value for metric 2"
                }),
            }
        }

    RETURN_TYPES = (IO.ANY, "STRING", "INT", "INT",)
    RETURN_NAMES = ("trigger_passthrough", "status_message", "completed_count", "remaining_count",)
    FUNCTION = "record_iteration"
    CATEGORY = "NV_Utils/Sweep"
    OUTPUT_NODE = True
    DESCRIPTION = "Records completion status for a sweep iteration. Updates the sweep plan JSON."

    def record_iteration(self, trigger, sweep_json_path, iteration_id,
                         status="completed", output_path="", notes="",
                         metric_1_name="", metric_1_value=0.0,
                         metric_2_name="", metric_2_value=0.0):
        """
        Record iteration status and update sweep JSON.
        """

        # Load the plan
        try:
            with open(sweep_json_path, 'r') as f:
                plan = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Sweep plan not found: {sweep_json_path}\n"
                f"Cannot record iteration status."
            )
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in sweep plan: {e}")

        # Validate iteration ID
        iterations = plan.get("iterations", [])
        total_iterations = len(iterations)

        if iteration_id >= total_iterations:
            raise ValueError(
                f"Invalid iteration_id {iteration_id}. "
                f"Plan has {total_iterations} iterations (indices 0-{total_iterations-1})."
            )

        # Update this iteration's status
        iteration = iterations[iteration_id]
        iteration["status"] = status
        iteration["completed_at"] = datetime.now().isoformat()

        if output_path.strip():
            iteration["output_path"] = output_path.strip()

        if notes.strip():
            iteration["notes"] = notes.strip()

        # Add metrics if provided
        if metric_1_name.strip():
            if "metrics" not in iteration:
                iteration["metrics"] = {}
            iteration["metrics"][metric_1_name.strip()] = metric_1_value

        if metric_2_name.strip():
            if "metrics" not in iteration:
                iteration["metrics"] = {}
            iteration["metrics"][metric_2_name.strip()] = metric_2_value

        # Update completed count
        completed_count = sum(1 for it in iterations if it.get("status") == "completed")
        failed_count = sum(1 for it in iterations if it.get("status") == "failed")
        skipped_count = sum(1 for it in iterations if it.get("status") == "skipped")
        remaining_count = total_iterations - completed_count - failed_count - skipped_count

        plan["completed_iterations"] = completed_count

        # Write updated JSON
        with open(sweep_json_path, 'w') as f:
            json.dump(plan, f, indent=2)

        # Build status message
        status_message = (
            f"Iteration {iteration_id} marked as {status}\n"
            f"Progress: {completed_count}/{total_iterations} completed"
        )
        if failed_count > 0:
            status_message += f", {failed_count} failed"
        if skipped_count > 0:
            status_message += f", {skipped_count} skipped"
        if remaining_count > 0:
            status_message += f", {remaining_count} remaining"

        print(f"[NV_SweepIterationRecorder] {status_message}")

        return (trigger, status_message, completed_count, remaining_count)


class NV_SweepProgressReader:
    """
    Reads current progress from a sweep plan without modifying it.

    Use this to check sweep status or display progress.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sweep_json_path": ("STRING", {
                    "default": "sweep_plan.json",
                    "tooltip": "Path to the sweep_plan.json file"
                }),
            },
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "INT", "STRING",)
    RETURN_NAMES = ("total", "completed", "failed", "skipped", "remaining", "summary",)
    FUNCTION = "read_progress"
    CATEGORY = "NV_Utils/Sweep"
    DESCRIPTION = "Reads current progress from a sweep plan. Does not modify the plan."

    def read_progress(self, sweep_json_path):
        """
        Read progress from sweep plan.
        """

        # Load the plan
        try:
            with open(sweep_json_path, 'r') as f:
                plan = json.load(f)
        except FileNotFoundError:
            return (0, 0, 0, 0, 0, f"Sweep plan not found: {sweep_json_path}")
        except json.JSONDecodeError as e:
            return (0, 0, 0, 0, 0, f"Invalid JSON: {e}")

        # Count statuses
        iterations = plan.get("iterations", [])
        total = len(iterations)
        completed = sum(1 for it in iterations if it.get("status") == "completed")
        failed = sum(1 for it in iterations if it.get("status") == "failed")
        skipped = sum(1 for it in iterations if it.get("status") == "skipped")
        remaining = total - completed - failed - skipped

        # Build summary
        sweep_name = plan.get("sweep_name", "unknown")
        summary_lines = [
            f"Sweep: {sweep_name}",
            f"Total: {total}",
            f"Completed: {completed}",
            f"Failed: {failed}",
            f"Skipped: {skipped}",
            f"Remaining: {remaining}",
            f"Progress: {completed}/{total} ({100*completed//total if total > 0 else 0}%)",
        ]
        summary = "\n".join(summary_lines)

        print(f"[NV_SweepProgressReader] {summary}")

        return (total, completed, failed, skipped, remaining, summary)


# Node registration
NODE_CLASS_MAPPINGS = {
    "NV_SweepIterationRecorder": NV_SweepIterationRecorder,
    "NV_SweepProgressReader": NV_SweepProgressReader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_SweepIterationRecorder": "NV Sweep Iteration Recorder",
    "NV_SweepProgressReader": "NV Sweep Progress Reader",
}
