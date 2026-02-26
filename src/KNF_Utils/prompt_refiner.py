"""
NV Prompt Refiner - Iterative LLM-powered captioner/refiner for V2V pipelines.

When video/image is connected: watches the media and writes a caption guided by
the Prompt Builder's instructions (replaces GeminiVideoCaptioner in the workflow).
Without media: refines a text prompt iteratively.

Maintains conversation history across re-executions — edit user_instruction and
re-queue to iterate. Supports Gemini (direct) and OpenRouter APIs with native
multi-turn conversation for both providers.
"""

import base64
import hashlib
import json
import os
import time

import cv2
import numpy as np
import requests

import folder_paths


# ---------------------------------------------------------------------------
# Conversation state (persists across ComfyUI re-executions)
# ---------------------------------------------------------------------------

# Keyed by hash(initial_prompt) → list of {"role": str, "content": str}
_CONVERSATION_HISTORY: dict[str, list[dict]] = {}
# Keyed by hash(initial_prompt) → last refined prompt text
_LAST_REFINED: dict[str, str] = {}
# Track which initial_prompt hash we last operated on (for auto-reset detection)
_LAST_PROMPT_HASH: str = ""
# Cumulative session cost tracking
_SESSION_COST: float = 0.0
# Single-entry media cache: media_hash -> (base64_data, mime_type, info_str)
_MEDIA_CACHE: dict[str, tuple[str, str, str]] = {}


# ---------------------------------------------------------------------------
# Gemini pricing table (USD per 1M tokens, standard tier, ≤200k context)
# Source: https://ai.google.dev/pricing (Feb 2026)
# ---------------------------------------------------------------------------
_GEMINI_PRICING = {
    # model_prefix: (input_per_1M, output_per_1M)
    "gemini-2.5-flash":        (0.30,  2.50),
    "gemini-2.5-pro":          (1.25, 10.00),
    "gemini-3-flash":          (0.50,  3.00),
    "gemini-3-pro":            (2.00, 12.00),
    "gemini-3.1-pro":          (2.00, 12.00),
}


def _estimate_gemini_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate cost in USD for a Gemini API call."""
    # Match by prefix (e.g. "gemini-3.1-pro-preview" → "gemini-3.1-pro")
    pricing = None
    for prefix, rates in _GEMINI_PRICING.items():
        if model.startswith(prefix):
            pricing = rates
            break
    if pricing is None:
        return 0.0
    input_cost = (prompt_tokens / 1_000_000) * pricing[0]
    output_cost = (completion_tokens / 1_000_000) * pricing[1]
    return input_cost + output_cost


def _hash_prompt(text: str) -> str:
    """Stable hash of prompt text for conversation keying."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _hash_media(tensor) -> str:
    """Lightweight hash of a media tensor for change detection.

    Samples shape + corner pixels rather than hashing the full tensor.
    Returns empty string for None.
    """
    if tensor is None:
        return ""
    shape_str = str(tuple(tensor.shape))
    flat = tensor.reshape(-1)
    sample = f"{float(flat[0]):.4f},{float(flat[-1]):.4f},{float(flat[len(flat)//2]):.4f}"
    return hashlib.sha256(f"{shape_str}:{sample}".encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Media conversion helpers (adapted from GeminiVideoCaptioner in nodes.py)
# ---------------------------------------------------------------------------

def _tensor_to_video(video_tensor, fps=30):
    """Convert video tensor [F,H,W,C] to temporary .mp4 file path."""
    if len(video_tensor.shape) == 4:
        video_array = video_tensor.cpu().numpy()
    elif len(video_tensor.shape) == 5:
        video_array = video_tensor[0].cpu().numpy()
    else:
        raise ValueError(f"Expected 4D or 5D video tensor, got {video_tensor.shape}")

    if video_array.dtype in (np.float32, np.float64, np.float16):
        video_array = (video_array * 255).astype(np.uint8)
    else:
        video_array = video_array.astype(np.uint8)

    temp_dir = folder_paths.get_temp_directory()
    temp_path = os.path.join(temp_dir, f"refiner_video_{int(time.time())}.mp4")
    height, width = video_array.shape[1:3]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
    if not out.isOpened():
        raise RuntimeError(
            f"cv2.VideoWriter failed to open: {temp_path} "
            f"(codec=mp4v, {width}x{height} @ {fps}fps). "
            f"Check OpenCV video codec support.")
    for frame in video_array:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()
    return temp_path


def _tensor_to_image(image_tensor):
    """Convert image tensor [B,H,W,C] or [H,W,C] to temporary .jpg file path."""
    if len(image_tensor.shape) == 4:
        image_array = image_tensor[0].cpu().numpy()
    elif len(image_tensor.shape) == 3:
        image_array = image_tensor.cpu().numpy()
    else:
        raise ValueError(f"Expected 3D or 4D image tensor, got {image_tensor.shape}")

    if image_array.dtype in (np.float32, np.float64, np.float16):
        image_array = (image_array * 255).astype(np.uint8)
    else:
        image_array = image_array.astype(np.uint8)

    temp_dir = folder_paths.get_temp_directory()
    temp_path = os.path.join(temp_dir, f"refiner_image_{int(time.time())}.jpg")
    cv2.imwrite(temp_path, cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
    return temp_path


def _encode_file_to_base64(file_path):
    """Encode file to base64 string."""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _prepare_media(image_tensor, video_tensor, fps):
    """Convert tensor inputs to base64-encoded media.

    Returns (base64_data, mime_type, info_str) or None if no media provided.
    image_tensor takes priority over video_tensor.
    """
    if image_tensor is None and video_tensor is None:
        return None

    file_path = None
    try:
        if image_tensor is not None:
            file_path = _tensor_to_image(image_tensor)
            mime_type = "image/jpeg"
            h, w = image_tensor.shape[-3], image_tensor.shape[-2]
            info = f"image {w}x{h} (jpeg)"
        else:
            file_path = _tensor_to_video(video_tensor, fps)
            mime_type = "video/mp4"
            frames = video_tensor.shape[0] if len(video_tensor.shape) == 4 else video_tensor.shape[1]
            h, w = video_tensor.shape[-3], video_tensor.shape[-2]
            info = f"video {w}x{h}, {frames} frames @ {fps}fps (mp4)"

        base64_data = _encode_file_to_base64(file_path)
        size_mb = len(base64_data) / (1024 * 1024)
        info += f", {size_mb:.1f}MB base64"
        return (base64_data, mime_type, info)
    finally:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Meta-system prompt templates
# ---------------------------------------------------------------------------

_META_SYSTEM_PROMPT_TEXT = """\
You are a prompt refinement assistant for a video generation pipeline.

## Your Task
Refine the user's prompt based on their instruction. The prompt will be used \
by a vision-language model to caption a video for V2V generation.

## Pipeline Constraints
The VLM that receives this prompt is governed by these system instructions:

---
{system_instruction}
---

## Rules
- Output ONLY the refined prompt text — no explanations, preamble, or \
meta-commentary.
- Preserve the trigger word placement (must be the first word if present).
- Stay within the word count range specified in the constraints above.
- Maintain the structural priorities from the pipeline constraints (word \
budget allocation, coverage order).
- Incorporate the user's feedback naturally — do not append it as a note.
- If the user's request conflicts with the structural constraints, prioritize \
the user's request but stay as close to the constraints as possible.
- When iterating, build on the previous refined version — do not restart \
from scratch unless the user explicitly asks."""

_META_SYSTEM_PROMPT_MEDIA = """\
You are a video captioning assistant for a V2V generation pipeline.

## Your Task
Watch the provided video/image and write a detailed caption following the \
user's captioning guidelines. On subsequent turns, refine your caption based \
on the user's feedback.

## Pipeline Constraints
The V2V pipeline that receives your caption is governed by these rules:

---
{system_instruction}
---

## Rules
- Output ONLY the caption text — no explanations, preamble, or meta-commentary.
- Preserve the trigger word placement (must be the first word if present).
- Stay within the word count range specified in the constraints above.
- Maintain the structural priorities from the pipeline constraints (word \
budget allocation, coverage order).
- Ground your caption in what you observe in the video/image — do not \
hallucinate visual details that are not present.
- Incorporate the user's feedback naturally — do not append it as a note.
- If the user's request conflicts with the structural constraints, prioritize \
the user's request but stay as close to the constraints as possible.
- When iterating, build on the previous caption — do not restart from scratch \
unless the user explicitly asks."""


# ---------------------------------------------------------------------------
# Model list (text + vision capable)
# ---------------------------------------------------------------------------

_MODEL_LIST = [
    # Gemini models (direct API — all support vision)
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-3-flash-preview",
    "gemini-3-pro-preview",
    "gemini-3.1-pro-preview",
    # OpenRouter text models
    "anthropic/claude-sonnet-4",
    "anthropic/claude-haiku-4",
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "meta-llama/llama-3.3-70b-instruct",
    "google/gemini-2.5-pro",
    "google/gemini-2.5-flash",
    # OpenRouter vision models (for use with media inputs)
    "qwen/qwen2.5-vl-72b-instruct",
    "meta-llama/llama-3.2-90b-vision-instruct",
]


# ---------------------------------------------------------------------------
# API call implementations (text-only, multi-turn)
# ---------------------------------------------------------------------------

def _call_gemini_text(conversation_turns, meta_system_prompt, api_key, model,
                      max_tokens, temperature, thinking_level,
                      media=None, media_resolution="default"):
    """Call Gemini API with multi-turn conversation, optional media.

    Args:
        conversation_turns: List of {"role": "user"|"assistant", "content": str}
        meta_system_prompt: System-level instruction (includes pipeline constraints)
        api_key: Gemini API key
        model: Model name (e.g. "gemini-2.5-pro")
        max_tokens: Max output tokens
        temperature: Sampling temperature
        thinking_level: "auto", "low", "medium", or "high" (Gemini 3+ only)
        media: Optional (base64_data, mime_type) tuple — injected into first user turn
        media_resolution: Gemini 3 only — controls tokens per video frame

    Returns:
        str: The model's response text
    """
    # Strip provider prefix if present
    if "/" in model:
        model = model.split("/", 1)[1]

    is_gemini3 = any(model.startswith(p) for p in ("gemini-3", "gemini-3."))

    # Build multi-turn contents array
    # Gemini uses "user" and "model" roles
    contents = []
    for i, turn in enumerate(conversation_turns):
        role = "model" if turn["role"] == "assistant" else "user"
        parts = [{"text": turn["content"]}]

        # Inject media into the first user turn
        if i == 0 and role == "user" and media is not None:
            parts.append({
                "inline_data": {
                    "mime_type": media[1],
                    "data": media[0],
                }
            })

        contents.append({"role": role, "parts": parts})

    payload = {"contents": contents}

    # System instruction
    if meta_system_prompt and meta_system_prompt.strip():
        payload["system_instruction"] = {
            "parts": [{"text": meta_system_prompt.strip()}]
        }

    # Generation config
    gen_config = {
        "maxOutputTokens": max_tokens,
        "temperature": temperature,
    }
    if is_gemini3 and thinking_level != "auto":
        gen_config["thinkingConfig"] = {"thinkingLevel": thinking_level}

    # Media resolution (Gemini 3 only, requires v1alpha endpoint)
    use_v1alpha = False
    if is_gemini3 and media_resolution != "default" and media is not None:
        gen_config["mediaResolution"] = f"media_resolution_{media_resolution}"
        use_v1alpha = True

    payload["generationConfig"] = gen_config

    headers = {"Content-Type": "application/json"}
    api_version = "v1alpha" if use_v1alpha else "v1beta"
    api_url = f"https://generativelanguage.googleapis.com/{api_version}/models/{model}:generateContent?key={api_key}"

    media_note = " +media" if media is not None else ""
    print(f"[NV_PromptRefiner] Calling Gemini ({model}), {len(conversation_turns)} turns{media_note}...")
    request_start = time.time()
    response = requests.post(api_url, headers=headers, json=payload, timeout=300)
    request_time = time.time() - request_start
    print(f"[NV_PromptRefiner] Response: {response.status_code} in {request_time:.2f}s")

    if response.status_code == 200:
        result = response.json()

        # Extract token usage and estimate cost
        token_info = {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0}
        if "usageMetadata" in result:
            usage = result["usageMetadata"]
            token_info["prompt_tokens"] = usage.get("promptTokenCount", 0)
            token_info["completion_tokens"] = usage.get("candidatesTokenCount", 0)
            token_info["cost"] = _estimate_gemini_cost(
                model, token_info["prompt_tokens"], token_info["completion_tokens"])
            print(f"[NV_PromptRefiner] Tokens: {token_info['prompt_tokens']} prompt, "
                  f"{token_info['completion_tokens']} response "
                  f"(~${token_info['cost']:.4f})")

        if "candidates" in result and len(result["candidates"]) > 0:
            candidate = result["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                # Filter out thinking parts (Gemini 3)
                text_parts = [p["text"] for p in candidate["content"]["parts"]
                              if "text" in p and not p.get("thought", False)]
                if not text_parts:
                    text_parts = [p["text"] for p in candidate["content"]["parts"]
                                  if "text" in p]
                return "\n".join(text_parts).strip(), token_info

        raise RuntimeError("Gemini API returned no content in response")

    error_msg = response.text[:500] if len(response.text) > 500 else response.text
    raise RuntimeError(f"Gemini API error {response.status_code}: {error_msg}")


def _call_openrouter_text(conversation_turns, meta_system_prompt, api_key, model,
                          max_tokens, temperature, thinking_level,
                          media=None, media_resolution="default"):
    """Call OpenRouter API with multi-turn conversation, optional media.

    Args:
        conversation_turns: List of {"role": "user"|"assistant", "content": str}
        meta_system_prompt: System-level instruction
        api_key: OpenRouter API key
        model: Model name (e.g. "anthropic/claude-sonnet-4")
        max_tokens: Max output tokens
        temperature: Sampling temperature
        thinking_level: Unused for OpenRouter (kept for interface parity)
        media: Optional (base64_data, mime_type) tuple — injected into first user turn
        media_resolution: Unused for OpenRouter (kept for interface parity)

    Returns:
        str: The model's response text
    """
    # Auto-prefix bare Gemini model names
    if "/" not in model and model.startswith("gemini-"):
        model = f"google/{model}"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/comfyui",
        "X-Title": "ComfyUI Prompt Refiner",
    }

    # Build messages array — OpenRouter uses "assistant" role
    messages = []
    if meta_system_prompt and meta_system_prompt.strip():
        messages.append({"role": "system", "content": meta_system_prompt.strip()})

    for i, turn in enumerate(conversation_turns):
        if i == 0 and turn["role"] == "user" and media is not None:
            # First user message with media: multipart content array
            base64_data, mime_type = media[0], media[1]
            is_video = mime_type.startswith("video/")
            content_parts = [{"type": "text", "text": turn["content"]}]
            if is_video:
                content_parts.append({
                    "type": "video_url",
                    "video_url": {"url": f"data:{mime_type};base64,{base64_data}"}
                })
            else:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{base64_data}"}
                })
            messages.append({"role": turn["role"], "content": content_parts})
        else:
            messages.append({"role": turn["role"], "content": turn["content"]})

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    media_note = " +media" if media is not None else ""
    print(f"[NV_PromptRefiner] Calling OpenRouter ({model}), {len(conversation_turns)} turns{media_note}...")
    request_start = time.time()

    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions",
                                 headers=headers, json=payload, timeout=300)
    except Exception as e:
        raise RuntimeError(f"OpenRouter API exception: {str(e)}")

    request_time = time.time() - request_start
    print(f"[NV_PromptRefiner] Response: {response.status_code} in {request_time:.2f}s")

    if response.status_code == 200:
        result = response.json()

        # Extract token usage — OpenRouter includes cost directly
        token_info = {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0}
        if "usage" in result:
            usage = result["usage"]
            token_info["prompt_tokens"] = usage.get("prompt_tokens", 0)
            token_info["completion_tokens"] = usage.get("completion_tokens", 0)
            token_info["cost"] = usage.get("cost", 0.0) or 0.0
            print(f"[NV_PromptRefiner] Tokens: {token_info['prompt_tokens']} prompt, "
                  f"{token_info['completion_tokens']} completion "
                  f"(${token_info['cost']:.4f})")

        if "choices" in result and len(result["choices"]) > 0:
            text = result["choices"][0]["message"]["content"]
            if not text or not text.strip():
                raise RuntimeError("OpenRouter model returned empty response")
            return text.strip(), token_info

        raise RuntimeError("OpenRouter API returned no choices in response")

    error_msg = response.text[:500] if len(response.text) > 500 else response.text
    raise RuntimeError(f"OpenRouter API error {response.status_code}: {error_msg}")


# ---------------------------------------------------------------------------
# Conversation log formatting
# ---------------------------------------------------------------------------

def _format_conversation_log(turns):
    """Format conversation turns into a readable log string."""
    if not turns:
        return "(no conversation history)"

    lines = []
    turn_num = 0
    for turn in turns:
        if turn["role"] == "user":
            turn_num += 1
            lines.append(f"--- Turn {turn_num} (user) ---")
        else:
            lines.append(f"--- Turn {turn_num} (refined) ---")
        lines.append(turn["content"])
        lines.append("")
    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Node class
# ---------------------------------------------------------------------------

class NV_PromptRefiner:
    """Iterative LLM captioner/refiner for V2V pipelines."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "system_instruction": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "From Prompt Builder — provides task context, word budgets, and structural constraints."
                }),
                "initial_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "From Prompt Builder's prompt_text output. With media: captioning guidelines. Without media: text to refine."
                }),
                "user_instruction": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Your refinement request. Edit this and re-queue to iterate. Examples: 'Add more armor detail', 'Make the cape description shorter', 'Focus on the character's face'."
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "API key for Gemini (direct) or OpenRouter."
                }),
                "provider": (["Gemini", "OpenRouter"], {
                    "default": "Gemini",
                    "tooltip": "API provider. Gemini models work with both; OpenRouter models (Claude, GPT, LLaMA) require OpenRouter."
                }),
                "model": (_MODEL_LIST, {
                    "default": "gemini-2.5-flash",
                    "tooltip": "LLM model. Gemini models work with Gemini provider; prefixed models (anthropic/, openai/) require OpenRouter. Use vision-capable models when media is connected."
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Sampling temperature. Lower = more deterministic. Higher = more creative."
                }),
                "max_tokens": ("INT", {
                    "default": 1500,
                    "min": 100,
                    "max": 4000,
                    "step": 100,
                    "tooltip": "Maximum response length in tokens."
                }),
                "thinking_level": (["auto", "low", "medium", "high"], {
                    "default": "auto",
                    "tooltip": "Gemini 3+ only. Controls reasoning depth. Ignored for other models and OpenRouter."
                }),
                "clear_history": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Toggle on to clear conversation history and start fresh. Toggle back off for the next iteration."
                }),
                "use_cached": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "When enabled, returns the last successful result without calling the API. Saves cost when re-queuing a workflow that already has a good caption."
                }),
            },
            "optional": {
                "video_tensor": ("IMAGE", {
                    "tooltip": "Optional video/image sequence. When connected, the LLM watches the video and writes a caption (captioner mode)."
                }),
                "image_tensor": ("IMAGE", {
                    "tooltip": "Optional single image. Takes priority over video_tensor. Enables captioner mode."
                }),
                "fps": ("FLOAT", {
                    "default": 30.0,
                    "min": 1.0,
                    "max": 120.0,
                    "step": 0.1,
                    "tooltip": "Frames per second for video encoding. Only used when video_tensor is provided."
                }),
                "media_resolution": (["default", "low", "medium", "high", "ultra_high"], {
                    "default": "default",
                    "tooltip": "Gemini 3 only. Controls tokens per video frame. Higher = better detail but more tokens."
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("refined_prompt", "system_instruction", "conversation_log", "debug_info")
    FUNCTION = "refine"
    CATEGORY = "NV_Utils/Prompt"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Iterative LLM captioner/refiner for V2V pipelines. Connect a video or "
        "image and the LLM writes a caption guided by the Prompt Builder's "
        "instructions. Edit user_instruction and re-queue to iterate. "
        "Works text-only too (without media) for prompt refinement. "
        "Supports Gemini (direct) and OpenRouter (Claude, GPT, LLaMA, etc.)."
    )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always re-execute — iteration is the intent
        return float("nan")

    def refine(self, system_instruction, initial_prompt, user_instruction,
               api_key, provider, model, temperature, max_tokens,
               thinking_level, clear_history, use_cached,
               video_tensor=None, image_tensor=None, fps=30.0,
               media_resolution="default"):
        global _CONVERSATION_HISTORY, _LAST_REFINED, _LAST_PROMPT_HASH, _SESSION_COST

        system_instruction = system_instruction.strip()
        initial_prompt = initial_prompt.strip()
        user_instruction = user_instruction.strip()

        # --- Validate initial_prompt (needed for conversation key) ---
        if not initial_prompt:
            return ("(no initial prompt provided)", system_instruction,
                    "(no conversation)", "Error: initial_prompt is empty")

        # --- Conversation key (computed early for use_cached + auto-reset) ---
        prompt_hash = _hash_prompt(initial_prompt)
        media_hash = ""
        if image_tensor is not None or video_tensor is not None:
            media_hash = _hash_media(image_tensor if image_tensor is not None else video_tensor)
            conversation_key = _hash_prompt(initial_prompt + media_hash)
        else:
            conversation_key = prompt_hash

        # Auto-reset: if context changed, clear old conversation
        if conversation_key != _LAST_PROMPT_HASH:
            _CONVERSATION_HISTORY.clear()
            _LAST_REFINED.clear()
            _MEDIA_CACHE.clear()
            _LAST_PROMPT_HASH = conversation_key

        # Manual reset
        if clear_history and conversation_key in _CONVERSATION_HISTORY:
            del _CONVERSATION_HISTORY[conversation_key]
            if conversation_key in _LAST_REFINED:
                del _LAST_REFINED[conversation_key]
            print("[NV_PromptRefiner] Conversation history cleared")

        # --- Use cached result (skip API call entirely) ---
        if use_cached and conversation_key in _LAST_REFINED:
            cached = _LAST_REFINED[conversation_key]
            print(f"[NV_PromptRefiner] Using cached result ({len(cached)} chars)")
            return (cached, system_instruction,
                    _format_conversation_log(_CONVERSATION_HISTORY.get(conversation_key, [])),
                    f"Cached: {len(cached)} chars / ~{len(cached.split())} words")

        # --- Validate remaining inputs ---
        if not user_instruction:
            return (initial_prompt, system_instruction,
                    "(no instruction — passing through initial prompt)",
                    "Pass-through: no user_instruction provided")
        if not api_key:
            return ("(no API key)", system_instruction,
                    "(no conversation)", "Error: api_key is empty")

        # --- Prepare media (if provided, with caching) ---
        media_tuple = None
        media_info = "none"
        if image_tensor is not None or video_tensor is not None:
            if media_hash in _MEDIA_CACHE:
                cached_b64, cached_mime, cached_info = _MEDIA_CACHE[media_hash]
                media_tuple = (cached_b64, cached_mime)
                media_info = cached_info + " (cached)"
                print(f"[NV_PromptRefiner] Media cache hit: {media_info}")
            else:
                try:
                    result = _prepare_media(image_tensor, video_tensor, fps)
                    if result is not None:
                        media_tuple = (result[0], result[1])
                        media_info = result[2]
                        _MEDIA_CACHE.clear()
                        _MEDIA_CACHE[media_hash] = (result[0], result[1], result[2])
                        print(f"[NV_PromptRefiner] Media prepared and cached: {media_info}")
                except Exception as e:
                    print(f"[NV_PromptRefiner] Warning: media preparation failed: {e}")
                    media_info = f"failed: {e}"

        has_media = media_tuple is not None
        mode = "captioner" if has_media else "refiner"

        # Get or create conversation for this context
        history = _CONVERSATION_HISTORY.get(conversation_key, [])

        # --- Build the new user message ---
        if not history:
            if has_media:
                # Captioner mode: video attached, initial_prompt = guidelines
                user_message = (
                    f"Captioning guidelines:\n\n"
                    f"---\n{initial_prompt}\n---\n\n"
                    f"Additional instruction: {user_instruction}"
                )
            else:
                # Refiner mode: text-only, initial_prompt = text to refine
                user_message = (
                    f"Here is the initial prompt to refine:\n\n"
                    f"---\n{initial_prompt}\n---\n\n"
                    f"Refinement instruction: {user_instruction}"
                )
        else:
            # Continuation: just the new instruction
            user_message = user_instruction

        # Append user turn
        history.append({"role": "user", "content": user_message})

        # --- Build meta-system prompt ---
        sys_inst = system_instruction or "(no system instruction provided)"
        if has_media:
            meta_system = _META_SYSTEM_PROMPT_MEDIA.format(system_instruction=sys_inst)
        else:
            meta_system = _META_SYSTEM_PROMPT_TEXT.format(system_instruction=sys_inst)

        # --- Call LLM ---
        turn_number = (len(history) + 1) // 2  # user/assistant pairs
        # Pass a snapshot so the API function receives an immutable view
        turns_snapshot = list(history)
        try:
            if provider == "Gemini":
                refined, token_info = _call_gemini_text(
                    turns_snapshot, meta_system, api_key, model,
                    max_tokens, temperature, thinking_level,
                    media=media_tuple, media_resolution=media_resolution
                )
            elif provider == "OpenRouter":
                refined, token_info = _call_openrouter_text(
                    turns_snapshot, meta_system, api_key, model,
                    max_tokens, temperature, thinking_level,
                    media=media_tuple, media_resolution=media_resolution
                )
            else:
                raise RuntimeError(f"Unknown provider: {provider}")
        except Exception as e:
            # Remove the failed user turn so history stays clean
            history.pop()
            _CONVERSATION_HISTORY[conversation_key] = history
            error_msg = str(e)
            print(f"[NV_PromptRefiner] Error: {error_msg}")
            # Return last refined if available, otherwise initial
            fallback = _LAST_REFINED.get(conversation_key, initial_prompt)
            return (fallback, system_instruction,
                    _format_conversation_log(history),
                    f"Error on turn {turn_number}: {error_msg}")

        # --- Track cost ---
        turn_cost = token_info.get("cost", 0.0)
        _SESSION_COST += turn_cost

        # --- Store assistant response ---
        history.append({"role": "assistant", "content": refined})
        _CONVERSATION_HISTORY[conversation_key] = history
        _LAST_REFINED[conversation_key] = refined

        # --- Build outputs ---
        conversation_log = _format_conversation_log(history)

        debug_info = (
            f"=== NV_PromptRefiner Debug ===\n"
            f"Mode: {mode}\n"
            f"Turn: {turn_number}\n"
            f"Provider: {provider} | Model: {model}\n"
            f"Temperature: {temperature} | Max Tokens: {max_tokens}\n"
            f"Tokens: {token_info['prompt_tokens']} in / {token_info['completion_tokens']} out\n"
            f"Cost: ~${turn_cost:.4f} this turn | ~${_SESSION_COST:.4f} session total\n"
            f"Prompt Hash: {conversation_key}\n"
            f"History Turns: {len(history)} messages\n"
            f"Refined Length: {len(refined)} chars / ~{len(refined.split())} words\n"
            f"Media: {media_info}\n"
            f"Clear History: {'yes' if clear_history else 'no'}"
        )

        print(f"[NV_PromptRefiner] Turn {turn_number} complete "
              f"({len(refined)} chars, ~{len(refined.split())} words, "
              f"~${turn_cost:.4f} / ${_SESSION_COST:.4f} session)")

        return (refined, system_instruction, conversation_log, debug_info)


NODE_CLASS_MAPPINGS = {
    "NV_PromptRefiner": NV_PromptRefiner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_PromptRefiner": "NV Prompt Refiner",
}
