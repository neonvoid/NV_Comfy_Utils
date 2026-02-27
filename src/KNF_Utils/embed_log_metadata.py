"""
NV Embed Log Metadata — Injects execution log data into video/image container metadata.

Passthrough node: place between data source and save node. Reads a JSON log file
and mutates extra_pnginfo in-place so the downstream SaveVideo/SaveWEBM/SaveImage
includes it alongside the workflow JSON.

Execution order is guaranteed by the passthrough DAG edge.
"""

import os
import json
from comfy.comfy_types.node_typing import IO


class NV_EmbedLogMetadata:
    """
    Reads a JSON log file and injects its contents into the output metadata.
    Place between data source and save node to embed execution logs alongside
    the workflow JSON in the saved file's container metadata.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "passthrough": (IO.ANY, {
                    "tooltip": "Data to pass through (IMAGE, VIDEO, LATENT, etc.). "
                               "Connect between data source and save node to guarantee execution order."
                }),
                "log_filepath": ("STRING", {
                    "default": "",
                    "tooltip": "Path to the JSON log file to embed. Accepts dynamic paths from sweep nodes."
                }),
            },
            "optional": {
                "metadata_key": ("STRING", {
                    "default": "execution_log",
                    "tooltip": "Key name under which the log data will be stored in the video metadata."
                }),
                "extra_key_1": ("STRING", {
                    "default": "",
                    "tooltip": "Optional additional key to embed (e.g., 'sweep_config')"
                }),
                "extra_value_1": ("STRING", {
                    "default": "",
                    "tooltip": "Value for extra_key_1"
                }),
                "extra_key_2": ("STRING", {
                    "default": "",
                    "tooltip": "Optional additional key to embed"
                }),
                "extra_value_2": ("STRING", {
                    "default": "",
                    "tooltip": "Value for extra_key_2"
                }),
            },
            "hidden": {
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True  # Accept any passthrough type

    RETURN_TYPES = (IO.ANY,)
    RETURN_NAMES = ("passthrough",)
    FUNCTION = "embed_metadata"
    CATEGORY = "NV_Utils/Serverless"
    DESCRIPTION = (
        "Reads a JSON log file and injects its contents into the video/image metadata. "
        "Place between data source and save node. The log will appear alongside the workflow JSON "
        "in the saved file's container metadata."
    )

    def embed_metadata(self, passthrough, log_filepath, extra_pnginfo=None,
                       metadata_key="execution_log",
                       extra_key_1="", extra_value_1="",
                       extra_key_2="", extra_value_2=""):
        if extra_pnginfo is None:
            print("[NV_EmbedLogMetadata] Warning: extra_pnginfo is None, cannot embed metadata")
            return (passthrough,)

        # Read and inject the JSON log file
        if log_filepath and log_filepath.strip():
            log_filepath = log_filepath.strip()
            if os.path.exists(log_filepath):
                try:
                    with open(log_filepath, 'r', encoding='utf-8') as f:
                        log_data = json.load(f)
                    extra_pnginfo[metadata_key] = log_data
                    print(f"[NV_EmbedLogMetadata] Embedded log from {os.path.basename(log_filepath)} as '{metadata_key}'")
                except (json.JSONDecodeError, IOError) as e:
                    print(f"[NV_EmbedLogMetadata] Warning: Could not read log file: {e}")
            else:
                print(f"[NV_EmbedLogMetadata] Warning: Log file not found: {log_filepath}")

        # Inject extra key-value pairs
        for key, value in [(extra_key_1, extra_value_1), (extra_key_2, extra_value_2)]:
            key = key.strip() if key else ""
            if key and value:
                # Try to parse value as JSON, fall back to string
                try:
                    parsed = json.loads(value)
                    extra_pnginfo[key] = parsed
                except (json.JSONDecodeError, TypeError):
                    extra_pnginfo[key] = value
                print(f"[NV_EmbedLogMetadata] Embedded extra key '{key}'")

        return (passthrough,)


NODE_CLASS_MAPPINGS = {
    "NV_EmbedLogMetadata": NV_EmbedLogMetadata,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_EmbedLogMetadata": "NV Embed Log Metadata",
}
