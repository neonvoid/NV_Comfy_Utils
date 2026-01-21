"""
JSON Metadata Writer Node for ComfyUI - Creates/updates JSON files with key-value metadata.
"""

import os
import json
from comfy.comfy_types.node_typing import IO


class NV_JsonMetadataWriter:
    """
    Writes key-value metadata to a JSON file.
    Creates file if it doesn't exist, merges with existing data.
    Empty values delete the corresponding key.
    """

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "trigger": (IO.ANY, {"tooltip": "Connect to trigger execution order"}),
                "filepath": ("STRING", {"default": "", "tooltip": "Path to JSON file (created if doesn't exist)"}),
            },
            "optional": {
                "parent_key": ("STRING", {"default": "", "tooltip": "Optional parent key to nest all values under (e.g., 'chunk_0')"}),
            }
        }
        # Add 10 key-value pairs
        for i in range(1, 11):
            inputs["optional"][f"key_{i}"] = ("STRING", {"default": "", "tooltip": f"Metadata key {i}"})
            inputs["optional"][f"value_{i}"] = ("STRING", {"default": "", "tooltip": f"Value for key {i}"})
        return inputs

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("json_output",)
    FUNCTION = "write_metadata"
    CATEGORY = "NV_Utils/Serverless"
    OUTPUT_NODE = True
    DESCRIPTION = "Writes key-value metadata to a JSON file. Merges with existing data. Use parent_key to nest under a specific key."

    def write_metadata(self, trigger, filepath, **kwargs):
        if not filepath.strip():
            print("[NV_JsonMetadataWriter] Warning: No filepath provided")
            return ("{}",)

        parent_key = kwargs.get("parent_key", "").strip()

        # Read existing JSON or start with empty dict
        data = {}
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"[NV_JsonMetadataWriter] Loaded existing data with {len(data)} top-level keys")
            except (json.JSONDecodeError, IOError) as e:
                print(f"[NV_JsonMetadataWriter] Warning: Could not read existing file: {e}")
        else:
            print(f"[NV_JsonMetadataWriter] File does not exist, creating new: {filepath}")

        # Build the new key-value pairs
        new_pairs = {}
        for i in range(1, 11):
            key = kwargs.get(f"key_{i}", "").strip()
            value = kwargs.get(f"value_{i}", "")

            if not key:
                continue  # Skip empty keys

            if value == "":
                # Empty value = mark for deletion
                new_pairs[key] = None
            else:
                new_pairs[key] = value

        # Apply to data (with optional parent nesting)
        if parent_key:
            # Nest under parent_key, merge with existing nested data
            if parent_key not in data:
                data[parent_key] = {}
            elif not isinstance(data[parent_key], dict):
                # Parent exists but isn't a dict, overwrite it
                data[parent_key] = {}

            for key, value in new_pairs.items():
                if value is None:
                    data[parent_key].pop(key, None)
                else:
                    data[parent_key][key] = value

            print(f"[NV_JsonMetadataWriter] Updated '{parent_key}' with {len([v for v in new_pairs.values() if v is not None])} fields")
        else:
            # Flat structure at root level
            for key, value in new_pairs.items():
                if value is None:
                    data.pop(key, None)
                else:
                    data[key] = value

        # Ensure directory exists
        dir_path = os.path.dirname(filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        # Write JSON
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        json_output = json.dumps(data, indent=2, ensure_ascii=False)
        print(f"[NV_JsonMetadataWriter] Wrote to {filepath}")
        return (json_output,)
