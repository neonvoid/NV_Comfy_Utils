"""
NV R2V Prompt Splitter — Extract prompt and negative prompt from R2V captioner output.

Takes the captioner's output string (which ends with a NEGATIVE: line) and splits
it into separate prompt and negative_prompt strings suitable for the WAN 2.6 R2V API.
"""

import re


class NV_R2VPromptSplitter:
    """Split R2V captioner output into prompt + negative_prompt for API consumption."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "caption": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "R2V captioner output containing a NEGATIVE: line at the end."
                }),
            },
            "optional": {
                "max_prompt_chars": ("INT", {
                    "default": 800,
                    "min": 100,
                    "max": 5000,
                    "step": 50,
                    "tooltip": "Maximum characters for the prompt output. WAN 2.6 R2V limit is 800."
                }),
                "max_negative_chars": ("INT", {
                    "default": 500,
                    "min": 50,
                    "max": 2000,
                    "step": 50,
                    "tooltip": "Maximum characters for the negative prompt output. WAN 2.6 R2V limit is 500."
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "negative_prompt", "info")
    FUNCTION = "split_caption"
    CATEGORY = "NV_Utils/Prompt"
    DESCRIPTION = (
        "Split R2V captioner output into prompt and negative_prompt. "
        "Extracts the NEGATIVE: line and enforces character limits for "
        "the WAN 2.6 R2V API (800 chars prompt, 500 chars negative)."
    )

    def split_caption(self, caption, max_prompt_chars=800, max_negative_chars=500):
        caption = caption.strip()

        if not caption:
            return ("", "", "Empty caption input.")

        # Find the last NEGATIVE: line (case-insensitive)
        # Match "NEGATIVE:" at the start of a line, possibly with whitespace before it
        pattern = re.compile(r'^[ \t]*NEGATIVE\s*:', re.IGNORECASE | re.MULTILINE)
        matches = list(pattern.finditer(caption))

        if matches:
            last_match = matches[-1]
            prompt = caption[:last_match.start()].strip()
            # Extract everything after "NEGATIVE:" on that line and any following lines
            negative_raw = caption[last_match.end():].strip()
            negative_prompt = negative_raw
        else:
            prompt = caption
            negative_prompt = ""

        # Track truncation
        info_lines = []
        prompt_truncated = False
        negative_truncated = False

        if len(prompt) > max_prompt_chars:
            prompt = prompt[:max_prompt_chars].rstrip()
            prompt_truncated = True

        if negative_prompt and len(negative_prompt) > max_negative_chars:
            negative_prompt = negative_prompt[:max_negative_chars].rstrip()
            negative_truncated = True

        # Build info output
        info_lines.append(f"Prompt: {len(prompt)} chars" + (" (TRUNCATED)" if prompt_truncated else ""))
        info_lines.append(f"Negative: {len(negative_prompt)} chars" + (" (TRUNCATED)" if negative_truncated else ""))
        if not matches:
            info_lines.append("No NEGATIVE: line found — negative_prompt is empty.")
        if prompt_truncated:
            info_lines.append(f"WARNING: Prompt truncated to {max_prompt_chars} chars.")
        if negative_truncated:
            info_lines.append(f"WARNING: Negative prompt truncated to {max_negative_chars} chars.")

        info = "\n".join(info_lines)

        return (prompt, negative_prompt, info)


NODE_CLASS_MAPPINGS = {
    "NV_R2VPromptSplitter": NV_R2VPromptSplitter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_R2VPromptSplitter": "NV R2V Prompt Splitter",
}
