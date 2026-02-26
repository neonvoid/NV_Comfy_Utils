"""
NV Sweep Iteration Loader

Loads parameters for a specific iteration from a sweep plan JSON.
Outputs values through generic slots that can be connected to any workflow input.

Use Case:
- Load iteration parameters from sweep plan
- Connect param_N_value outputs to sampler inputs
- Use iteration_label for logging/display
- Use output_suffix for filenames

Workflow:
1. NV_SweepPlanner → exports sweep_plan.json (run once)
2. NV_SweepIterationLoader → loads params for iteration N (this node)
3. [Your sampling workflow] → connect param_N_value outputs to inputs
4. NV_SweepIterationRecorder (optional) → tracks completion status
"""

import json


class NV_SweepIterationLoader:
    """
    Loads parameters for a specific sweep iteration.

    Reads the sweep plan JSON and outputs parameter values for the specified iteration.
    Supports up to 8 numeric parameters and 2 string parameters.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sweep_json_path": ("STRING", {
                    "default": "sweep_plan.json",
                    "tooltip": "Path to the sweep_plan.json file from NV_SweepPlanner"
                }),
                "iteration_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 999999,
                    "tooltip": "Which iteration to load (0-indexed)"
                }),
            },
        }

    # Output types: 8 numeric (as FLOAT since ComfyUI will accept for both), 2 string, 1 step_split, plus metadata
    RETURN_TYPES = (
        "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT",  # param_1-8_value
        "STRING", "STRING",  # string_param_1-2_value
        "STRING",  # step_split_1_value
        "INT", "INT",  # iteration_id, total_iterations
        "STRING", "STRING", "STRING",  # iteration_label, output_suffix, all_params_json
    )
    RETURN_NAMES = (
        "param_1_value", "param_2_value", "param_3_value", "param_4_value",
        "param_5_value", "param_6_value", "param_7_value", "param_8_value",
        "string_param_1_value", "string_param_2_value",
        "step_split_1_value",
        "iteration_id", "total_iterations",
        "iteration_label", "output_suffix", "all_params_json",
    )
    FUNCTION = "load_iteration"
    CATEGORY = "NV_Utils/Sweep"
    DESCRIPTION = "Loads parameters for a specific sweep iteration. Connect param_N_value outputs to your sampler inputs."

    def load_iteration(self, sweep_json_path, iteration_index):
        """
        Load parameters for the specified iteration.
        """

        # Load the plan
        try:
            with open(sweep_json_path, 'r') as f:
                plan = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Sweep plan not found: {sweep_json_path}\n"
                f"Run NV_SweepPlanner first to create the plan."
            )
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in sweep plan: {e}")

        # Validate iteration index
        iterations = plan.get("iterations", [])
        total_iterations = len(iterations)

        if iteration_index >= total_iterations:
            raise ValueError(
                f"Invalid iteration_index {iteration_index}. "
                f"Plan has {total_iterations} iterations (indices 0-{total_iterations-1})."
            )

        # Get this iteration's data
        iteration = iterations[iteration_index]
        params = iteration.get("params", {})
        label = iteration.get("label", f"iter {iteration_index}/{total_iterations}")
        output_suffix = iteration.get("output_suffix", f"iter{iteration_index:04d}")

        # Get parameter definitions to map slots
        parameters = plan.get("parameters", {})

        # Extract numeric parameter values (by slot order)
        numeric_values = [0.0] * 8
        for i in range(1, 9):
            slot_key = f"param_{i}"
            if slot_key in parameters:
                param_name = parameters[slot_key].get("name", "")
                param_type = parameters[slot_key].get("type", "float")
                if param_name and param_name in params:
                    val = params[param_name]
                    if param_type == "int":
                        numeric_values[i-1] = float(int(val))
                    else:
                        numeric_values[i-1] = float(val)

        # Extract string parameter values (by slot order)
        string_values = ["", ""]
        for i in range(1, 3):
            slot_key = f"string_param_{i}"
            if slot_key in parameters:
                param_name = parameters[slot_key].get("name", "")
                if param_name and param_name in params:
                    string_values[i-1] = str(params[param_name])

        # Extract step split parameter value
        step_split_value = ""
        if "step_split_1" in parameters:
            param_name = parameters["step_split_1"].get("name", "")
            if param_name and param_name in params:
                step_split_value = str(params[param_name])

        # Build all_params_json
        all_params_json = json.dumps(params, indent=2)

        print(f"[NV_SweepIterationLoader] Loaded iteration {iteration_index}/{total_iterations-1}:")
        print(f"  {label}")

        return (
            numeric_values[0], numeric_values[1], numeric_values[2], numeric_values[3],
            numeric_values[4], numeric_values[5], numeric_values[6], numeric_values[7],
            string_values[0], string_values[1],
            step_split_value,
            iteration_index, total_iterations,
            label, output_suffix, all_params_json,
        )


class NV_SweepIterationLoaderInt:
    """
    Alternative loader that outputs integers instead of floats.

    Use this when you need INT outputs for nodes that don't accept FLOAT.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sweep_json_path": ("STRING", {
                    "default": "sweep_plan.json",
                    "tooltip": "Path to the sweep_plan.json file from NV_SweepPlanner"
                }),
                "iteration_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 999999,
                    "tooltip": "Which iteration to load (0-indexed)"
                }),
            },
        }

    RETURN_TYPES = (
        "INT", "INT", "INT", "INT", "INT", "INT", "INT", "INT",  # param_1-8_value as INT
        "STRING", "STRING",  # string_param_1-2_value
        "STRING",  # step_split_1_value
        "INT", "INT",  # iteration_id, total_iterations
        "STRING", "STRING", "STRING",  # iteration_label, output_suffix, all_params_json
    )
    RETURN_NAMES = (
        "param_1_value", "param_2_value", "param_3_value", "param_4_value",
        "param_5_value", "param_6_value", "param_7_value", "param_8_value",
        "string_param_1_value", "string_param_2_value",
        "step_split_1_value",
        "iteration_id", "total_iterations",
        "iteration_label", "output_suffix", "all_params_json",
    )
    FUNCTION = "load_iteration"
    CATEGORY = "NV_Utils/Sweep"
    DESCRIPTION = "Loads parameters for a sweep iteration with INT outputs. Use when you need integer values."

    def load_iteration(self, sweep_json_path, iteration_index):
        """
        Load parameters for the specified iteration (INT version).
        """

        # Load the plan
        try:
            with open(sweep_json_path, 'r') as f:
                plan = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Sweep plan not found: {sweep_json_path}\n"
                f"Run NV_SweepPlanner first to create the plan."
            )
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in sweep plan: {e}")

        # Validate iteration index
        iterations = plan.get("iterations", [])
        total_iterations = len(iterations)

        if iteration_index >= total_iterations:
            raise ValueError(
                f"Invalid iteration_index {iteration_index}. "
                f"Plan has {total_iterations} iterations (indices 0-{total_iterations-1})."
            )

        # Get this iteration's data
        iteration = iterations[iteration_index]
        params = iteration.get("params", {})
        label = iteration.get("label", f"iter {iteration_index}/{total_iterations}")
        output_suffix = iteration.get("output_suffix", f"iter{iteration_index:04d}")

        # Get parameter definitions to map slots
        parameters = plan.get("parameters", {})

        # Extract numeric parameter values (by slot order) - cast to INT
        int_values = [0] * 8
        for i in range(1, 9):
            slot_key = f"param_{i}"
            if slot_key in parameters:
                param_name = parameters[slot_key].get("name", "")
                if param_name and param_name in params:
                    int_values[i-1] = int(round(params[param_name]))

        # Extract string parameter values (by slot order)
        string_values = ["", ""]
        for i in range(1, 3):
            slot_key = f"string_param_{i}"
            if slot_key in parameters:
                param_name = parameters[slot_key].get("name", "")
                if param_name and param_name in params:
                    string_values[i-1] = str(params[param_name])

        # Extract step split parameter value
        step_split_value = ""
        if "step_split_1" in parameters:
            param_name = parameters["step_split_1"].get("name", "")
            if param_name and param_name in params:
                step_split_value = str(params[param_name])

        # Build all_params_json
        all_params_json = json.dumps(params, indent=2)

        print(f"[NV_SweepIterationLoaderInt] Loaded iteration {iteration_index}/{total_iterations-1}:")
        print(f"  {label}")

        return (
            int_values[0], int_values[1], int_values[2], int_values[3],
            int_values[4], int_values[5], int_values[6], int_values[7],
            string_values[0], string_values[1],
            step_split_value,
            iteration_index, total_iterations,
            label, output_suffix, all_params_json,
        )


# Node registration
NODE_CLASS_MAPPINGS = {
    "NV_SweepIterationLoader": NV_SweepIterationLoader,
    "NV_SweepIterationLoaderInt": NV_SweepIterationLoaderInt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_SweepIterationLoader": "NV Sweep Iteration Loader",
    "NV_SweepIterationLoaderInt": "NV Sweep Iteration Loader (INT)",
}
