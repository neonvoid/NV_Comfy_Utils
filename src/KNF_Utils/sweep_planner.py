"""
NV Sweep Planner

Plans parameter sweep configurations for systematic exploration of sampling parameters.
Generates a JSON file with all parameter combinations (grid search).

Use Case:
- Exploring CFG, steps, denoise, or any custom parameter ranges
- Systematic A/B testing of different parameter combinations
- Reproducible parameter sweeps with tracking

Workflow:
1. NV_SweepPlanner → exports sweep_plan.json (run once)
2. NV_SweepIterationLoader → loads params for iteration N (run per iteration)
3. [Your sampling workflow] → uses loaded parameters
4. NV_SweepIterationRecorder (optional) → tracks completion status
"""

import json
import os
from datetime import datetime
from itertools import product


def get_unique_filepath(filepath: str) -> str:
    """
    Return a unique filepath by appending _N if file exists.

    sweep_plan.json → sweep_plan_1.json → sweep_plan_2.json → ...
    """
    if not os.path.exists(filepath):
        return filepath

    base, ext = os.path.splitext(filepath)
    counter = 1
    while True:
        new_path = f"{base}_{counter}{ext}"
        if not os.path.exists(new_path):
            return new_path
        counter += 1


def make_filename_safe(val) -> str:
    """
    Convert a value to a filename-safe string.

    - Floats: 3.0 → "3", 3.5 → "3-5" (period replaced with hyphen)
    - Ints: passed through as string
    - Strings: spaces/special chars replaced with hyphens
    """
    if isinstance(val, float):
        # If it's a whole number, show without decimal
        if val == int(val):
            return str(int(val))
        else:
            # Replace period with hyphen for decimals
            return str(val).replace(".", "-")
    elif isinstance(val, int):
        return str(val)
    else:
        # String: replace problematic characters
        safe = str(val)
        for char in [" ", "/", "\\", ":", "*", "?", '"', "<", ">", "|", "."]:
            safe = safe.replace(char, "-")
        # Remove consecutive hyphens
        while "--" in safe:
            safe = safe.replace("--", "-")
        return safe.strip("-")


def generate_numeric_values(start: float, end: float, increment: float, value_type: str) -> list:
    """
    Generate list of values from start to end with given increment.

    Args:
        start: Starting value
        end: Ending value (inclusive if hit exactly)
        increment: Step size
        value_type: "float" or "int"

    Returns:
        List of values
    """
    if increment <= 0:
        raise ValueError(f"Increment must be positive, got {increment}")

    if start > end:
        raise ValueError(f"Start ({start}) must be <= end ({end})")

    values = []
    current = start

    # Use tolerance for floating point comparison
    tolerance = increment / 1000

    while current <= end + tolerance:
        if value_type == "int":
            values.append(int(round(current)))
        else:
            values.append(round(current, 6))  # Round to avoid floating point artifacts
        current += increment

    return values


class NV_SweepPlanner:
    """
    Plans parameter sweep configurations with flexible numeric and string parameters.

    Generates all combinations (grid search) and exports to JSON.
    Supports up to 8 numeric parameters and 2 string parameters.
    """

    @classmethod
    def INPUT_TYPES(cls):
        # Build input spec for 8 numeric param slots
        inputs = {
            "required": {
                "sweep_name": ("STRING", {
                    "default": "sweep_001",
                    "tooltip": "Name for this sweep (used in output filenames)"
                }),
                "output_json_path": ("STRING", {
                    "default": "sweep_plan.json",
                    "tooltip": "Path to save the sweep plan JSON file"
                }),
            },
            "optional": {}
        }

        # Add 8 numeric parameter slots
        for i in range(1, 9):
            inputs["optional"][f"param_{i}_name"] = ("STRING", {
                "default": "",
                "tooltip": f"Name for parameter {i} (e.g., 'cfg', 'steps', 'denoise')"
            })
            inputs["optional"][f"param_{i}_start"] = ("FLOAT", {
                "default": 0.0,
                "min": -1e10,
                "max": 1e10,
                "step": 0.01,
                "tooltip": f"Starting value for parameter {i}"
            })
            inputs["optional"][f"param_{i}_end"] = ("FLOAT", {
                "default": 0.0,
                "min": -1e10,
                "max": 1e10,
                "step": 0.01,
                "tooltip": f"Ending value for parameter {i}"
            })
            inputs["optional"][f"param_{i}_increment"] = ("FLOAT", {
                "default": 1.0,
                "min": 0.0001,
                "max": 1e10,
                "step": 0.01,
                "tooltip": f"Step size for parameter {i}"
            })
            inputs["optional"][f"param_{i}_type"] = (["float", "int"], {
                "default": "float",
                "tooltip": f"Output type for parameter {i}"
            })

        # Add 2 string parameter slots
        for i in range(1, 3):
            inputs["optional"][f"string_param_{i}_name"] = ("STRING", {
                "default": "",
                "tooltip": f"Name for string parameter {i} (e.g., 'sampler_name', 'scheduler')"
            })
            inputs["optional"][f"string_param_{i}_values"] = ("STRING", {
                "default": "",
                "tooltip": f"Comma-separated values for string parameter {i} (e.g., 'euler,dpmpp_2m,uni_pc')"
            })

        return inputs

    RETURN_TYPES = ("STRING", "INT", "STRING",)
    RETURN_NAMES = ("json_path", "total_iterations", "summary",)
    OUTPUT_NODE = True
    FUNCTION = "plan_sweep"
    CATEGORY = "NV_Utils/Sweep"
    DESCRIPTION = "Plans parameter sweep with flexible numeric and string parameters. Generates JSON with all combinations."

    def plan_sweep(self, sweep_name, output_json_path, **kwargs):
        """
        Generate sweep plan with all parameter combinations.
        """

        # Collect active numeric parameters
        numeric_params = []
        for i in range(1, 9):
            name = kwargs.get(f"param_{i}_name", "")
            if name and name.strip():
                start = kwargs.get(f"param_{i}_start", 0.0)
                end = kwargs.get(f"param_{i}_end", 0.0)
                increment = kwargs.get(f"param_{i}_increment", 1.0)
                value_type = kwargs.get(f"param_{i}_type", "float")

                # Generate values
                try:
                    values = generate_numeric_values(start, end, increment, value_type)
                except ValueError as e:
                    raise ValueError(f"Parameter {i} ({name}): {e}")

                if len(values) == 0:
                    raise ValueError(f"Parameter {i} ({name}): No values generated. Check start/end/increment.")

                numeric_params.append({
                    "slot": f"param_{i}",
                    "name": name.strip(),
                    "type": value_type,
                    "start": start,
                    "end": end,
                    "increment": increment,
                    "values": values,
                })

        # Collect active string parameters
        string_params = []
        for i in range(1, 3):
            name = kwargs.get(f"string_param_{i}_name", "")
            values_str = kwargs.get(f"string_param_{i}_values", "")
            if name and name.strip() and values_str and values_str.strip():
                values = [v.strip() for v in values_str.split(",") if v.strip()]
                if len(values) == 0:
                    raise ValueError(f"String parameter {i} ({name}): No values found. Check comma-separated list.")

                string_params.append({
                    "slot": f"string_param_{i}",
                    "name": name.strip(),
                    "values": values,
                })

        # Validate we have at least one parameter
        all_params = numeric_params + string_params
        if len(all_params) == 0:
            raise ValueError("At least one parameter must be defined. Fill in param_1_name and param_1_start/end/increment.")

        # Calculate total combinations
        total_iterations = 1
        param_counts = []
        for p in all_params:
            count = len(p["values"])
            total_iterations *= count
            param_counts.append(f"{p['name']}({count})")

        # Generate all iterations (grid search - cartesian product)
        all_value_lists = [p["values"] for p in all_params]
        all_combinations = list(product(*all_value_lists))

        iterations = []
        for idx, combo in enumerate(all_combinations):
            # Build params dict with actual names
            params_dict = {}
            for p_idx, p in enumerate(all_params):
                params_dict[p["name"]] = combo[p_idx]

            # Build human-readable label
            label_parts = [f"{p['name']}={combo[p_idx]}" for p_idx, p in enumerate(all_params)]
            label = f"iter {idx}/{total_iterations}: " + ", ".join(label_parts)

            # Build filename-safe suffix
            suffix_parts = []
            for p_idx, p in enumerate(all_params):
                val = combo[p_idx]
                safe_val = make_filename_safe(val)
                suffix_parts.append(f"{p['name']}{safe_val}")
            suffix = "_".join(suffix_parts)

            iterations.append({
                "id": idx,
                "status": "pending",
                "params": params_dict,
                "label": label,
                "output_suffix": suffix,
            })

        # Build parameters section for JSON
        parameters_section = {}
        for p in numeric_params:
            parameters_section[p["slot"]] = {
                "name": p["name"],
                "type": p["type"],
                "start": p["start"],
                "end": p["end"],
                "increment": p["increment"],
                "values": p["values"],
            }
        for p in string_params:
            parameters_section[p["slot"]] = {
                "name": p["name"],
                "type": "string",
                "values": p["values"],
            }

        # Build the plan
        plan = {
            "version": "1.0",
            "sweep_name": sweep_name,
            "created_at": datetime.now().isoformat(),
            "total_iterations": total_iterations,
            "completed_iterations": 0,
            "parameters": parameters_section,
            "iterations": iterations,
        }

        # Ensure output directory exists
        output_dir = os.path.dirname(output_json_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Get unique filepath (don't overwrite existing files)
        output_json_path = get_unique_filepath(output_json_path)

        # Write JSON file
        with open(output_json_path, 'w') as f:
            json.dump(plan, f, indent=2)

        # Generate summary
        summary_lines = [
            "=" * 60,
            "PARAMETER SWEEP PLAN",
            "=" * 60,
            f"Sweep name: {sweep_name}",
            f"Total iterations: {total_iterations}",
            f"Combination: {' x '.join(param_counts)}",
            "",
            "PARAMETERS:",
        ]

        for p in numeric_params:
            values_preview = p["values"][:5]
            preview_str = ", ".join(str(v) for v in values_preview)
            if len(p["values"]) > 5:
                preview_str += f", ... ({len(p['values'])} total)"
            summary_lines.append(f"  {p['name']} ({p['type']}): {preview_str}")

        for p in string_params:
            summary_lines.append(f"  {p['name']} (string): {', '.join(p['values'])}")

        summary_lines.extend([
            "",
            "FIRST 5 ITERATIONS:",
        ])

        for it in iterations[:5]:
            summary_lines.append(f"  [{it['id']}] {it['label']}")

        if total_iterations > 5:
            summary_lines.append(f"  ... and {total_iterations - 5} more")

        summary_lines.extend([
            "",
            f"Plan saved to: {output_json_path}",
            "=" * 60,
        ])

        summary = "\n".join(summary_lines)
        print(summary)

        return {"ui": {"text": [summary]}, "result": (output_json_path, total_iterations, summary)}


# Node registration
NODE_CLASS_MAPPINGS = {
    "NV_SweepPlanner": NV_SweepPlanner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_SweepPlanner": "NV Sweep Planner",
}
