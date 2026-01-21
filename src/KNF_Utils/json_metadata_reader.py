"""
JSON Metadata Reader Node for ComfyUI - Reads values from JSON metadata files.
"""

import os
import json
from comfy.comfy_types.node_typing import IO


class NV_JsonMetadataReader:
    """
    Reads key-value metadata from a JSON file.
    Supports nested structure via parent_key.
    """

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "filepath": ("STRING", {"default": "", "tooltip": "Path to JSON file to read"}),
            },
            "optional": {
                "parent_key": ("STRING", {"default": "", "tooltip": "Optional parent key to read from (e.g., 'chunk_0')"}),
            }
        }
        # Add 5 key inputs
        for i in range(1, 6):
            inputs["optional"][f"key_{i}"] = ("STRING", {"default": "", "tooltip": f"Key {i} to read"})
        return inputs

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("value_1", "value_2", "value_3", "value_4", "value_5", "json_output")
    FUNCTION = "read_metadata"
    CATEGORY = "NV_Utils/Serverless"
    DESCRIPTION = "Reads values from a JSON metadata file by key names."

    def read_metadata(self, filepath, **kwargs):
        parent_key = kwargs.get("parent_key", "").strip()

        # Default outputs
        values = ["", "", "", "", ""]
        json_output = "{}"

        if not filepath.strip():
            print("[NV_JsonMetadataReader] Warning: No filepath provided")
            return (*values, json_output)

        if not os.path.exists(filepath):
            print(f"[NV_JsonMetadataReader] File not found: {filepath}")
            return (*values, json_output)

        # Read JSON
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"[NV_JsonMetadataReader] Error reading file: {e}")
            return (*values, json_output)

        # Get target dict (root or nested under parent_key)
        if parent_key:
            target = data.get(parent_key, {})
            if not isinstance(target, dict):
                target = {}
        else:
            target = data

        json_output = json.dumps(target, indent=2, ensure_ascii=False)

        # Read requested keys
        for i in range(1, 6):
            key = kwargs.get(f"key_{i}", "").strip()
            if key:
                value = target.get(key, "")
                # Convert non-string values to string
                if not isinstance(value, str):
                    value = json.dumps(value)
                values[i-1] = value

        print(f"[NV_JsonMetadataReader] Read from {filepath}" + (f" [{parent_key}]" if parent_key else ""))
        return (*values, json_output)
