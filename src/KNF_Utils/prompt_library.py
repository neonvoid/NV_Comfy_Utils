"""
NV Prompt Library - Save and load reusable prompt templates.

Manages a JSON-based prompt library for use with GeminiVideoCaptioner
and other text-input nodes. Supports LoRA trigger word placeholders
({lora1}-{lora4}) for dynamic prompt assembly.
"""

import os
import json
from datetime import datetime


class NV_PromptSaver:
    """
    Save a named prompt to a JSON library file.
    Append-only: duplicate names get an auto-incremented suffix (_2, _3, etc.).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "library_path": ("STRING", {
                    "default": "prompt_library.json",
                    "tooltip": "Path to the JSON prompt library file. Created if it does not exist."
                }),
                "prompt_name": ("STRING", {
                    "default": "",
                    "tooltip": "Name/key for this prompt (e.g., 'cinematic_dolly'). Auto-suffixed if duplicate."
                }),
                "prompt_text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Prompt text. Use {lora1}-{lora4} as placeholders for LoRA trigger words."
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt_text", "saved_as", "status")
    FUNCTION = "save_prompt"
    CATEGORY = "NV_Utils/Prompt"
    OUTPUT_NODE = True
    DESCRIPTION = "Save a named prompt to a JSON library file. Append-only: never overwrites existing prompts. Supports {lora1}-{lora4} placeholders."

    def save_prompt(self, library_path, prompt_name, prompt_text):
        if not prompt_name.strip():
            status = "[NV_PromptSaver] Error: prompt_name is empty"
            print(status)
            return (prompt_text, "", status)

        if not library_path.strip():
            status = "[NV_PromptSaver] Error: library_path is empty"
            print(status)
            return (prompt_text, "", status)

        prompt_name = prompt_name.strip()
        now = datetime.now().isoformat()

        # Load existing library or create new
        data = {"version": "1.0", "prompts": {}}
        if os.path.exists(library_path):
            try:
                with open(library_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if "prompts" not in data:
                    data["prompts"] = {}
            except (json.JSONDecodeError, IOError) as e:
                print(f"[NV_PromptSaver] Warning: Could not read existing file ({e}), creating new")
                data = {"version": "1.0", "prompts": {}}

        # Auto-suffix to avoid overwriting existing prompts
        final_name = prompt_name
        if final_name in data["prompts"]:
            counter = 2
            while f"{prompt_name}_{counter}" in data["prompts"]:
                counter += 1
            final_name = f"{prompt_name}_{counter}"

        # Save the prompt
        data["prompts"][final_name] = {
            "text": prompt_text,
            "created_at": now,
            "updated_at": now,
        }

        # Ensure directory exists
        dir_path = os.path.dirname(os.path.abspath(library_path))
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        # Write JSON
        with open(library_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        suffix_note = f" (renamed from '{prompt_name}')" if final_name != prompt_name else ""
        total = len(data["prompts"])
        status = f"[NV_PromptSaver] Saved '{final_name}'{suffix_note} ({total} prompts in library)"
        print(status)

        return (prompt_text, final_name, status)


class NV_PromptSelector:
    """
    Load a prompt from the library, substitute LoRA trigger word placeholders,
    and optionally append additional trigger words.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "library_path": ("STRING", {
                    "default": "prompt_library.json",
                    "tooltip": "Path to the JSON prompt library file"
                }),
                "prompt_name": ("STRING", {
                    "default": "",
                    "tooltip": "Name of the prompt to load. Must match a key in the library."
                }),
            },
            "optional": {
                "lora_trigger_1": ("STRING", {
                    "default": "",
                    "tooltip": "Trigger word to replace {lora1} placeholder"
                }),
                "lora_trigger_2": ("STRING", {
                    "default": "",
                    "tooltip": "Trigger word to replace {lora2} placeholder"
                }),
                "lora_trigger_3": ("STRING", {
                    "default": "",
                    "tooltip": "Trigger word to replace {lora3} placeholder"
                }),
                "lora_trigger_4": ("STRING", {
                    "default": "",
                    "tooltip": "Trigger word to replace {lora4} placeholder"
                }),
                "lora_append": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Additional trigger words to append (comma or newline separated)"
                }),
                "append_separator": ("STRING", {
                    "default": ", ",
                    "tooltip": "Separator between prompt text and appended trigger words"
                }),
                "remove_unfilled_placeholders": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Strip unreplaced {lora1}-{lora4} placeholders from output"
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt_text", "raw_prompt", "available_prompts")
    FUNCTION = "select_prompt"
    CATEGORY = "NV_Utils/Prompt"
    DESCRIPTION = "Load a prompt from the library, substitute {lora1}-{lora4} placeholders, and optionally append trigger words."

    @classmethod
    def IS_CHANGED(cls, library_path, prompt_name, **kwargs):
        """Re-execute when the library file changes on disk."""
        if os.path.exists(library_path):
            return os.path.getmtime(library_path)
        return float("nan")

    def select_prompt(self, library_path, prompt_name, **kwargs):
        lora_triggers = {}
        for i in range(1, 5):
            val = kwargs.get(f"lora_trigger_{i}", "").strip()
            if val:
                lora_triggers[f"lora{i}"] = val

        lora_append = kwargs.get("lora_append", "").strip()
        append_separator = kwargs.get("append_separator", ", ")
        remove_unfilled = kwargs.get("remove_unfilled_placeholders", True)

        # Load the library
        if not os.path.exists(library_path):
            msg = f"[NV_PromptSelector] Library file not found: {library_path}"
            print(msg)
            return ("", "", msg)

        try:
            with open(library_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            msg = f"[NV_PromptSelector] Error reading library: {e}"
            print(msg)
            return ("", "", msg)

        prompts = data.get("prompts", {})
        available = ", ".join(sorted(prompts.keys())) if prompts else "(empty library)"

        # Validate prompt_name
        prompt_name = prompt_name.strip()
        if not prompt_name:
            msg = f"[NV_PromptSelector] No prompt_name specified. Available: {available}"
            print(msg)
            return ("", "", available)

        if prompt_name not in prompts:
            msg = f"[NV_PromptSelector] '{prompt_name}' not found. Available: {available}"
            print(msg)
            return ("", "", available)

        # Get raw prompt text
        raw_prompt = prompts[prompt_name].get("text", "")

        # Placeholder substitution
        assembled = raw_prompt
        for key, value in lora_triggers.items():
            assembled = assembled.replace(f"{{{key}}}", value)

        # Remove unfilled placeholders if requested
        if remove_unfilled:
            for i in range(1, 5):
                assembled = assembled.replace(f"{{lora{i}}}", "")
            while "  " in assembled:
                assembled = assembled.replace("  ", " ")
            assembled = assembled.strip()

        # Append trigger words
        if lora_append:
            append_words = []
            for line in lora_append.split("\n"):
                line = line.strip()
                if line:
                    for word in line.split(","):
                        word = word.strip()
                        if word:
                            append_words.append(word)

            if append_words:
                assembled = assembled + append_separator + ", ".join(append_words)

        print(f"[NV_PromptSelector] Loaded '{prompt_name}' ({len(assembled)} chars)")
        if lora_triggers:
            print(f"[NV_PromptSelector] Substituted: {list(lora_triggers.keys())}")
        if lora_append:
            print(f"[NV_PromptSelector] Appended trigger words")

        return (assembled, raw_prompt, available)


NODE_CLASS_MAPPINGS = {
    "NV_PromptSaver": NV_PromptSaver,
    "NV_PromptSelector": NV_PromptSelector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_PromptSaver": "NV Prompt Saver",
    "NV_PromptSelector": "NV Prompt Selector",
}
