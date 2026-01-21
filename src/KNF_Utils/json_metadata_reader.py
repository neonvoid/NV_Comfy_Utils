"""
JSON Metadata Reader Node for ComfyUI - Reads values from JSON metadata files.
"""

import os
import json
from comfy.comfy_types.node_typing import IO


class NV_JsonMetadataReader:
    """
    Reads key-value metadata from a JSON file.
    Supports up to 2 levels of nested structure via parent_key and parent_key_2.
    """

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "filepath": ("STRING", {"default": "", "tooltip": "Path to JSON file to read"}),
            },
            "optional": {
                "parent_key": ("STRING", {"default": "", "tooltip": "Level 1 parent key (e.g., 'clip_001')"}),
                "parent_key_2": ("STRING", {"default": "", "tooltip": "Level 2 parent key (e.g., 'chunk_0')"}),
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
    DESCRIPTION = "Reads values from a JSON metadata file. Supports up to 2 levels of nesting."

    def read_metadata(self, filepath, **kwargs):
        parent_key = kwargs.get("parent_key", "").strip()
        parent_key_2 = kwargs.get("parent_key_2", "").strip()

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

        # Navigate to target based on nesting level
        def get_nested(d, key):
            if not key:
                return d
            val = d.get(key, {})
            return val if isinstance(val, dict) else {}

        if parent_key and parent_key_2:
            # Two levels: data[parent_key][parent_key_2]
            level1 = get_nested(data, parent_key)
            target = get_nested(level1, parent_key_2)
            path_str = f" [{parent_key}][{parent_key_2}]"
        elif parent_key:
            # One level: data[parent_key]
            target = get_nested(data, parent_key)
            path_str = f" [{parent_key}]"
        else:
            # Root level
            target = data
            path_str = ""

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

        print(f"[NV_JsonMetadataReader] Read from {filepath}{path_str}")
        return (*values, json_output)
