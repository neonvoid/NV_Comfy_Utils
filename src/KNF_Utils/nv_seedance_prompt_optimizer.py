"""NV Seedance Prompt Optimizer — Gemini-powered translation + bracket syntax.

Bridges English-authored prompts into Seedance-native Chinese with bracket
reference syntax. Per multi-AI research (2026-04-23), Seedance 2.0's text
encoder was trained predominantly on Chinese examples using bracketed
references like [图1]/[视频1]/[音频1] rather than @Image1/@Video1/@Audio1.

Wire this between your prompt source and NV Seedance Prep V2:

    prompt (EN) ──► NV Seedance Prompt Optimizer ──► NV Seedance Prep V2

Reuses the existing Gemini integration from prompt_refiner.py (same API key
resolution, same call infrastructure). Cheap models (Flash) recommended —
this is a translation task, not a creative one.
"""

from __future__ import annotations

import json
import re
import time

from .api_keys import resolve_api_key
from .prompt_refiner import _call_gemini_text


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

# To Chinese + bracket syntax.
# Front-loaded: state operation first per multi-AI prompting consensus.
_TO_CHINESE_SYSTEM_PROMPT = """You are a precise translator specializing in video generation prompts for ByteDance Seedance 2.0.

Translate the user's prompt to formal Chinese suitable for the Doubao text encoder. Apply these rules:

1. Convert tag syntax:
   - `@Image1`, `@Image 1`, `@image1` → `[图1]`
   - `@Image2` → `[图2]`, etc. (preserve numbering)
   - `@Video1` → `[视频1]`
   - `@Video2` → `[视频2]`, etc.
   - `@Audio1` → `[音频1]`
   - `@Audio2` → `[音频2]`, etc.

2. Translation style:
   - Use formal, technical Chinese (not colloquial)
   - Preserve the operation intent (replace / edit / extend / generate / etc.)
   - Keep visual descriptors precise (clothing, motion, lighting, camera, etc.)
   - Use Seedance-native edit verbs: "替换" (replace), "保持...不变" (keep unchanged), "严格...一致" (strictly consistent with)
   - For preservation constraints, use positive framing: "严格保持原视频的X不变" (strictly preserve X from original video unchanged)

3. Preserve all parameter values verbatim if present (--rs 720p, --duration 5, etc.)

4. Output ONLY the translated prompt. No explanation, no commentary, no quotation marks around the output, no "Translation:" prefix. Just the prompt."""


# To English + @ tag syntax (reverse / passthrough).
_TO_ENGLISH_SYSTEM_PROMPT = """You are a precise translator. Convert the user's Chinese Seedance prompt to natural English.

Apply these rules:

1. Convert tag syntax:
   - `[图1]` → `@Image1`
   - `[图2]` → `@Image2`, etc.
   - `[视频1]` → `@Video1`
   - `[音频1]` → `@Audio1`

2. Translation style:
   - Natural cinematic English
   - Preserve operation intent and all visual descriptors

3. Output ONLY the translated prompt. No explanation, no commentary."""


# Just convert syntax without translating.
_BRACKET_CONVERT_SYSTEM_PROMPT = """You are a precise text editor. Apply ONLY these tag substitutions to the input. Do not translate or rewrite anything else:

- `@Image1`, `@Image 1`, `@image1`, `@image 1` → `[图1]`
- `@Image2`, etc. → `[图2]` etc.
- `@Video1`, `@video1` → `[视频1]`
- `@Audio1`, `@audio1` → `[音频1]`
- Preserve all numbering

Output ONLY the modified prompt. No explanation."""

_AT_TAG_CONVERT_SYSTEM_PROMPT = """You are a precise text editor. Apply ONLY these tag substitutions to the input. Do not translate or rewrite anything else:

- `[图1]` → `@Image1`
- `[图2]` → `@Image2`, etc.
- `[视频1]` → `@Video1`
- `[音频1]` → `@Audio1`
- Preserve all numbering

Output ONLY the modified prompt. No explanation."""


# Cheap, fast models for translation. User can override.
_TRANSLATION_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-pro",
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite-preview",
    "gemini-3-pro-preview",
    "gemini-3.1-pro-preview",
]


# ---------------------------------------------------------------------------
# Local fallback — pure-regex syntax conversion (no API call)
# ---------------------------------------------------------------------------

def _local_at_to_brackets(text: str) -> str:
    """Pure-regex EN-tag → CN-bracket conversion. No translation."""
    text = re.sub(r"@Image\s?(\d+)", r"[图\1]", text, flags=re.IGNORECASE)
    text = re.sub(r"@Video\s?(\d+)", r"[视频\1]", text, flags=re.IGNORECASE)
    text = re.sub(r"@Audio\s?(\d+)", r"[音频\1]", text, flags=re.IGNORECASE)
    return text


def _local_brackets_to_at(text: str) -> str:
    """Pure-regex CN-bracket → EN-tag conversion. No translation."""
    text = re.sub(r"\[图\s?(\d+)\]", r"@Image\1", text)
    text = re.sub(r"\[视频\s?(\d+)\]", r"@Video\1", text)
    text = re.sub(r"\[音频\s?(\d+)\]", r"@Audio\1", text)
    return text


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class NV_SeedancePromptOptimizer:
    """Optimize a Seedance prompt for the Doubao text encoder.

    Three operations selectable:
      - `translate_to_chinese`: full Gemini translation to formal CN + bracket tags
      - `translate_to_english`: reverse direction (CN+brackets → EN+@tags)
      - `bracket_convert_only`: ONLY swap tag syntax, no translation (pure regex,
                                no API call, free)
      - `at_tag_convert_only`: reverse direction (brackets → @ tags), pure regex
      - `passthrough`: no-op
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Source prompt. Will be optimized for Seedance 2.0 per the mode selected.",
                }),
                "operation": ([
                    "translate_to_chinese",
                    "translate_to_english",
                    "bracket_convert_only",
                    "at_tag_convert_only",
                    "passthrough",
                ], {
                    "default": "translate_to_chinese",
                    "tooltip": (
                        "translate_to_chinese (recommended for face-swap shots): full Gemini "
                        "translation EN→CN + @Image1→[图1] bracket conversion.\n"
                        "translate_to_english: reverse direction.\n"
                        "bracket_convert_only: just swap @ tags to brackets, no translation (free, no API call).\n"
                        "at_tag_convert_only: reverse syntax conversion (brackets → @, free).\n"
                        "passthrough: no-op."
                    ),
                }),
            },
            "optional": {
                "model": (_TRANSLATION_MODELS, {
                    "default": "gemini-2.5-flash",
                    "tooltip": "Gemini model for translation. Flash variants are cheap and fast — recommended for this task.",
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "Optional override. Empty → env (GEMINI_API_KEY / GOOGLE_API_KEY) → .env.",
                }),
                "max_tokens": ("INT", {
                    "default": 2048, "min": 256, "max": 8192, "step": 128,
                    "tooltip": "Output cap. 2048 is plenty for typical Seedance prompts (~500-1000 chars).",
                }),
                "temperature": ("FLOAT", {
                    "default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Low (0.0-0.2) recommended for translation — deterministic, faithful to input.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("optimized_prompt", "info")
    FUNCTION = "optimize"
    CATEGORY = "NV_Utils/api"
    DESCRIPTION = (
        "Optimize a Seedance prompt for the Doubao text encoder via Gemini "
        "translation + tag-syntax conversion. CN + bracket syntax beats EN + "
        "@tags for adherence on Seedance 2.0 per multi-AI research."
    )

    def optimize(
        self,
        prompt: str,
        operation: str,
        model: str = "gemini-2.5-flash",
        api_key: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ):
        prompt = (prompt or "").strip()
        if not prompt:
            return ("", json.dumps({"operation": operation, "result": "empty_input_passthrough"}))

        if operation == "passthrough":
            return (prompt, json.dumps({"operation": "passthrough", "input_chars": len(prompt)}))

        # Pure-regex paths: no API call, free
        if operation == "bracket_convert_only":
            out = _local_at_to_brackets(prompt)
            n_changed = len(re.findall(r"@(?:Image|Video|Audio)", prompt, flags=re.IGNORECASE))
            print(f"[NV_SeedancePromptOptimizer] bracket_convert_only: {n_changed} tag(s) → brackets (no API call)")
            return (out, json.dumps({
                "operation": "bracket_convert_only",
                "tags_converted": n_changed,
                "input_chars": len(prompt),
                "output_chars": len(out),
            }, ensure_ascii=False))

        if operation == "at_tag_convert_only":
            out = _local_brackets_to_at(prompt)
            n_changed = len(re.findall(r"\[(?:图|视频|音频)\s?\d+\]", prompt))
            print(f"[NV_SeedancePromptOptimizer] at_tag_convert_only: {n_changed} bracket(s) → @ tags (no API call)")
            return (out, json.dumps({
                "operation": "at_tag_convert_only",
                "brackets_converted": n_changed,
                "input_chars": len(prompt),
                "output_chars": len(out),
            }, ensure_ascii=False))

        # Gemini-powered translation paths
        resolved_key = resolve_api_key(api_key, provider="gemini")

        if operation == "translate_to_chinese":
            system_prompt = _TO_CHINESE_SYSTEM_PROMPT
        elif operation == "translate_to_english":
            system_prompt = _TO_ENGLISH_SYSTEM_PROMPT
        else:
            raise ValueError(f"Unknown operation: {operation!r}")

        t_start = time.time()
        try:
            translated, token_info = _call_gemini_text(
                conversation_turns=[{"role": "user", "content": prompt}],
                meta_system_prompt=system_prompt,
                api_key=resolved_key,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                thinking_level="auto",
            )
        except Exception as e:
            print(f"[NV_SeedancePromptOptimizer] Gemini call failed: {e}")
            print("[NV_SeedancePromptOptimizer] Falling back to local regex conversion (no translation)")
            if operation == "translate_to_chinese":
                fallback = _local_at_to_brackets(prompt)
            else:
                fallback = _local_brackets_to_at(prompt)
            return (fallback, json.dumps({
                "operation": operation,
                "fallback": "local_regex_only",
                "error": str(e),
            }, ensure_ascii=False))

        elapsed = time.time() - t_start
        translated = (translated or "").strip()
        # Strip surrounding quotes if Gemini added them
        if (translated.startswith('"') and translated.endswith('"')) or \
           (translated.startswith("'") and translated.endswith("'")):
            translated = translated[1:-1].strip()

        info = {
            "operation": operation,
            "model": model,
            "input_chars": len(prompt),
            "output_chars": len(translated),
            "elapsed_sec": round(elapsed, 2),
            "tokens": token_info,
        }
        print(f"[NV_SeedancePromptOptimizer] {operation}: {len(prompt)} → {len(translated)} chars in {elapsed:.1f}s")
        return (translated, json.dumps(info, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "NV_SeedancePromptOptimizer": NV_SeedancePromptOptimizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_SeedancePromptOptimizer": "NV Seedance Prompt Optimizer",
}
