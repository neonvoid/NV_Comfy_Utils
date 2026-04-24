"""NV Seedance Subject Extractor — Gemini vision extracts subject description from image.

Tightly-scoped slot-filler: given a reference IMAGE, asks Gemini to describe
ONLY the subject's identity features (clothing, hair, accessories) in concrete
descriptors. Output goes into NV_SeedancePromptBuilder's `target_details` slot.

Why this exists: Seedance Mode C's primary failure mode is image-dominance —
the model reads the ref image's NON-face features (composition/scene/pose/body
silhouette) and pulls the output toward the ref image's scene even when the
prompt says to use @Video1's scene. Giving the model a strong TEXT description
of the subject provides a semantic anchor that competes with the ref image's
compositional signal, improving scene/motion adherence.

Scoped tightly to prevent hallucination:
  - ONLY describes subject features, not scene / lighting / pose / mood
  - Concrete descriptors only (colors, materials, cuts, items)
  - Hard word cap
  - Fallback on API failure → empty string → Builder degrades gracefully

Separate node (not baked into Builder) per multi-AI architectural guidance:
keeps Builder deterministic + testable, isolates API-key management + failure
handling, user can opt out by simply not wiring this.
"""

from __future__ import annotations

import base64
import io
import json
import time

import numpy as np
import torch
from PIL import Image

from .api_keys import resolve_api_key
from .prompt_refiner import _call_gemini_text


_COMMON_GUARDRAILS = (
    " If features are blurred, occluded, cropped, or uncertain, omit that "
    "detail rather than guessing. If multiple people are present, describe "
    "only the most central or largest subject. Do not name or identify any "
    "real person. Do not infer age, ethnicity, or emotional state. If no "
    "clear central subject is visible, output an empty string."
)

_FOCUS_PROMPTS = {
    "subject_full": (
        "You are a visual analyzer. Describe ONLY the central subject's "
        "identity features visible in the image — clothing, hair, accessories, "
        "and distinguishing body features. "
        "Do NOT describe: background, scene, lighting, camera angle, pose, "
        "mood, or composition. Do NOT invent features you cannot clearly see. "
        "Do NOT use full sentences — use concrete descriptor phrases separated "
        "by commas. Use concrete terms (colors, materials, cuts, items). "
        "Output a single line under {max_words} words. No preamble."
        + _COMMON_GUARDRAILS
    ),
    "wardrobe_only": (
        "You are a visual analyzer. Describe ONLY the clothing, accessories, "
        "and worn items of the central subject. Do NOT describe: body features, "
        "hair, face, skin, background, scene, lighting, pose, or mood. "
        "Output concrete descriptor phrases separated by commas — no sentences. "
        "Include colors, materials, cuts, accessories, footwear. "
        "Output a single line under {max_words} words. No preamble."
        + _COMMON_GUARDRAILS
    ),
    "hair_and_accessories_only": (
        "You are a visual analyzer. Describe ONLY the hair (length, color, "
        "style, texture) and accessories (jewelry, headwear, eyewear) of the "
        "central subject. Do NOT describe: clothing, body, face, skin, "
        "background, scene, lighting, or pose. Do not infer facial structure, "
        "ethnicity, age, or expression. "
        "Output concrete descriptor phrases separated by commas. "
        "Output a single line under {max_words} words. No preamble."
        + _COMMON_GUARDRAILS
    ),
    "body_proportions": (
        "You are a visual analyzer. Describe ONLY the body build and "
        "proportions of the central subject — height impression, build, frame. "
        "Do NOT describe: clothing, hair, face, skin, background, scene, "
        "lighting, pose, or mood. "
        "Output concrete descriptor phrases separated by commas. "
        "Output a single line under {max_words} words. No preamble."
        + _COMMON_GUARDRAILS
    ),
}


_TRANSLATION_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-pro",
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite-preview",
    "gemini-3-pro-preview",
    "gemini-3.1-pro-preview",
]


def _image_tensor_to_base64(image: torch.Tensor) -> str:
    """IMAGE tensor [1,H,W,3] or [H,W,3] or [B,H,W,3/4] → base64 PNG.

    Handles: batch dimension (warns if >1, takes frame 0), range detection
    (0-1 float vs 0-255 input — some upstream nodes emit 0-255), alpha channel
    stripping (RGBA → RGB to prevent transparent-blank vision).
    """
    if image.ndim == 4:
        if image.shape[0] > 1:
            print(
                f"[NV_SeedanceSubjectExtractor] warning: batch size {image.shape[0]}, "
                f"using frame [0] only. Wire Image Batch Select upstream if you need a different frame."
            )
        image = image[0]
    if image.ndim != 3:
        raise ValueError(
            f"[NV_SeedanceSubjectExtractor] image must be [H,W,C] or [B,H,W,C], "
            f"got {tuple(image.shape)}"
        )
    if image.shape[-1] not in (3, 4):
        raise ValueError(
            f"[NV_SeedanceSubjectExtractor] image must have 3 or 4 channels, "
            f"got {image.shape[-1]}"
        )

    arr = image.detach().cpu().numpy()
    # Range detection: if values exceed reasonable float-image range, treat as uint8-scale
    if arr.max() > 1.5:
        arr = arr.clip(0, 255).astype(np.uint8)
    else:
        arr = (arr.clip(0, 1) * 255).astype(np.uint8)
    # Strip alpha — RGBA with transparent pixels would render blank to Gemini
    arr = arr[..., :3]

    pil = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    pil.save(buf, format="PNG", optimize=False)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _post_process(text: str, max_words: int) -> str:
    """Clean up the model's output — strip markdown, preamble, quotes, enforce word cap.

    No ellipsis on truncation — video models interpret `...` as fade-out/dissolve
    and that would warp the downstream prompt's meaning. Hard trim to max_words
    and rstrip trailing punctuation instead.
    """
    import re as _re
    text = (text or "").strip()

    # Strip markdown code fences (``` or ```markdown)
    text = _re.sub(r"^```(?:\w+)?\s*\n?", "", text)
    text = _re.sub(r"\n?```\s*$", "", text).strip()

    # Strip markdown bold/italic wrappers
    text = _re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = _re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\1", text)

    # Strip leading bullet / list markers
    text = _re.sub(r"^[-*•]\s+", "", text)
    text = _re.sub(r"^\d+[.)]\s+", "", text)

    # Strip JSON-ish list wrappers
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip()
        # Strip quoted-list-item wrappers: "a", "b" → a, b
        text = _re.sub(r'["\']([^"\']+)["\']', r"\1", text)

    # Strip surrounding quotes
    if (text.startswith('"') and text.endswith('"')) or \
       (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()

    # Strip common preamble patterns
    for prefix in ("Description:", "Subject:", "Output:", "Here is", "Here's"):
        if text.lower().startswith(prefix.lower()):
            for sep in (":", ".", "\n"):
                if sep in text:
                    text = text.split(sep, 1)[1].strip()
                    break

    # Take first non-empty line (collapse accidental paragraphs)
    for line in text.splitlines():
        if line.strip():
            text = line.strip()
            break

    # Hard-enforce word cap — NO ellipsis (reads as fade-out/dissolve to video models)
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words]).rstrip(",;.!? ")

    return text


class NV_SeedanceSubjectExtractor:
    """Gemini-powered subject-description extractor from a reference image.

    Pair with NV_SeedancePromptBuilder: wire this node's `subject_description`
    output into the Builder's `target_details` input for stronger text
    anchoring in Seedance Mode C.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": (
                        "Reference image of the subject. Single frame or first frame of a "
                        "batch. Keep framing tight on the subject for best extraction."
                    ),
                }),
                "focus": (list(_FOCUS_PROMPTS.keys()), {
                    "default": "subject_full",
                    "tooltip": (
                        "subject_full (default): clothing + hair + accessories + body. "
                        "wardrobe_only: clothing/accessories only (no body/hair — useful "
                        "when subject identity comes from elsewhere). "
                        "hair_and_accessories_only: no clothing. "
                        "body_proportions: build/frame only."
                    ),
                }),
                "max_words": ("INT", {
                    "default": 20, "min": 5, "max": 100, "step": 1,
                    "tooltip": (
                        "Word cap for output. Seedance prompts have a 500-CN-char / 1000-EN-word "
                        "soft cap; this slot should be ~20-40 words to leave room for scene/motion "
                        "language."
                    ),
                }),
            },
            "optional": {
                "model": (_TRANSLATION_MODELS, {
                    "default": "gemini-2.5-flash",
                    "tooltip": "Gemini model. Flash variants are cheap + fast — recommended.",
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "Optional override. Empty → env (GEMINI_API_KEY / GOOGLE_API_KEY) → .env.",
                }),
                "extra_instructions": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": (
                        "Optional additional constraints appended to the system prompt. "
                        "Example: 'Emphasize formal wear details' or 'Ignore any costumes'."
                    ),
                }),
                "fallback_on_failure": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": (
                        "Returned as subject_description if the Gemini call fails (no API key, "
                        "rate limit, network error, malformed response). Leave empty to degrade "
                        "silently; Builder handles empty target_details gracefully."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    # RETURN_NAMES matches NV_SeedancePromptBuilder's `target_details` input slot
    # so ComfyUI's auto-wire heuristic picks the correct connection automatically.
    RETURN_NAMES = ("target_details", "info")
    FUNCTION = "extract"
    CATEGORY = "NV_Utils/api"
    DESCRIPTION = (
        "Gemini vision extracts a text description of a reference image's subject — "
        "clothing, hair, accessories — for use as text anchor in Seedance prompts. "
        "Attacks Mode C image-dominance failure mode by giving the model a strong "
        "textual identity signal. Tightly scoped to prevent hallucination. Falls "
        "back to empty string on API failure."
    )

    def extract(
        self,
        image: torch.Tensor,
        focus: str,
        max_words: int,
        model: str = "gemini-2.5-flash",
        api_key: str = "",
        extra_instructions: str = "",
        fallback_on_failure: str = "",
    ):
        t_start = time.time()

        # --- prepare image (base64-encode first/only frame) ---
        try:
            b64 = _image_tensor_to_base64(image)
        except Exception as e:
            print(f"[NV_SeedanceSubjectExtractor] image encode failed: {e}")
            return (fallback_on_failure, json.dumps({
                "status": "encode_failed", "error": str(e),
            }))

        # --- resolve API key (graceful failure) ---
        try:
            resolved_key = resolve_api_key(api_key, provider="gemini")
        except Exception as e:
            print(f"[NV_SeedanceSubjectExtractor] no Gemini key — returning fallback")
            return (fallback_on_failure, json.dumps({
                "status": "no_api_key", "error": str(e),
            }))

        # --- build system prompt ---
        system_prompt = _FOCUS_PROMPTS[focus].format(max_words=max_words)
        if extra_instructions.strip():
            system_prompt = system_prompt + " " + extra_instructions.strip()

        # --- call Gemini ---
        try:
            description, token_info = _call_gemini_text(
                conversation_turns=[{
                    "role": "user",
                    "content": "Describe the subject in this image per the instructions.",
                }],
                meta_system_prompt=system_prompt,
                api_key=resolved_key,
                model=model,
                max_tokens=256,  # short output — word cap is low
                temperature=0.1,  # deterministic extraction
                thinking_level="low",
                media_list=[(b64, "image/png", "reference subject image")],
            )
        except Exception as e:
            print(f"[NV_SeedanceSubjectExtractor] Gemini call failed: {e}")
            return (fallback_on_failure, json.dumps({
                "status": "api_error", "error": str(e), "elapsed_sec": round(time.time() - t_start, 2),
            }))

        # --- post-process ---
        cleaned = _post_process(description, max_words)
        if not cleaned:
            print(f"[NV_SeedanceSubjectExtractor] empty response — using fallback")
            return (fallback_on_failure, json.dumps({
                "status": "empty_response",
                "raw_response": description,
                "elapsed_sec": round(time.time() - t_start, 2),
            }))

        elapsed = time.time() - t_start
        info = {
            "status": "ok",
            "focus": focus,
            "model": model,
            "max_words": max_words,
            "actual_word_count": len(cleaned.split()),
            "elapsed_sec": round(elapsed, 2),
            "tokens": token_info,
        }
        print(f"[NV_SeedanceSubjectExtractor] {focus}: {cleaned!r} ({len(cleaned.split())} words, {elapsed:.1f}s)")
        return (cleaned, json.dumps(info, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "NV_SeedanceSubjectExtractor": NV_SeedanceSubjectExtractor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_SeedanceSubjectExtractor": "NV Seedance Subject Extractor",
}
