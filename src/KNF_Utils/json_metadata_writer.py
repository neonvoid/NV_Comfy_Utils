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
                "parent_key": ("STRING", {"default": "", "tooltip": "Level 1 parent key (e.g., 'clip_001')"}),
                "parent_key_2": ("STRING", {"default": "", "tooltip": "Level 2 parent key (e.g., 'chunk_0')"}),
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
    DESCRIPTION = "Writes key-value metadata to a JSON file. Supports up to 2 levels of nesting via parent_key and parent_key_2."

    def write_metadata(self, trigger, filepath, **kwargs):
        if not filepath.strip():
            print("[NV_JsonMetadataWriter] Warning: No filepath provided")
            return ("{}",)

        parent_key = kwargs.get("parent_key", "").strip()
        parent_key_2 = kwargs.get("parent_key_2", "").strip()

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

        # Get or create target dict based on nesting level
        def ensure_dict(d, key):
            if key not in d:
                d[key] = {}
            elif not isinstance(d[key], dict):
                d[key] = {}
            return d[key]

        # Determine target location for writing
        if parent_key and parent_key_2:
            # Two levels of nesting: data[parent_key][parent_key_2][key] = value
            level1 = ensure_dict(data, parent_key)
            target = ensure_dict(level1, parent_key_2)
            path_str = f"'{parent_key}' > '{parent_key_2}'"
        elif parent_key:
            # One level of nesting: data[parent_key][key] = value
            target = ensure_dict(data, parent_key)
            path_str = f"'{parent_key}'"
        else:
            # Flat structure at root level
            target = data
            path_str = "root"

        # Apply new pairs to target
        for key, value in new_pairs.items():
            if value is None:
                target.pop(key, None)
            else:
                target[key] = value

        print(f"[NV_JsonMetadataWriter] Updated {path_str} with {len([v for v in new_pairs.values() if v is not None])} fields")

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
