"""
NV Agent Param Planner - LLM-assisted parameter recommendation for VACE pipelines.

Phase 1.0 MVP: a "Suggester" node. Reads sampled input frames, optional mask,
optional manifest summary from a previous render, plus a free-form user intent
describing the shot type and desired outcome. Calls a vision-capable LLM
(Claude Opus/Sonnet, GPT, Gemini, Qwen) via OpenRouter and returns a markdown
recommendation. NO automatic param mutation — user reads the recommendation
and adjusts widgets manually. This is the lowest-risk shape to ship and
validate before investing in Phase 1.1's structured override consumption.

Design intent:
- Reuses the existing OpenRouter-via-requests pattern from prompt_refiner.py
  (no new SDK dependency).
- Reuses api_keys.resolve_api_key for env-var + .env fallback.
- Vision input via base64 JPEG (matches prompt_refiner's media-injection shape).
- N-frame sampling caps token cost — agent doesn't need 277 frames; 5-8 evenly
  spaced ones convey the shot.
- System prompt encodes the agent's role (param tuner for VACE face-swap),
  references the Mysterious Params Guide (D-132) as the canonical knowledge,
  and prescribes the output format (markdown table + reasoning).

Wiring (typical):
- images = sampled frames from VHS_LoadVideo or InpaintCrop2's cropped output
- mask = optional, helps agent see what region is being inpainted
- manifest_summary = paste from a previous NV_RenderManifest's manifest_json
- user_intent = free-form description ("matte face on dark BG", "head jitter
  near frame 80", "VAE fizz too aggressive")
- system_prompt_extra = paste relevant excerpts from MYSTERIOUS_PARAMS_GUIDE.md
  (the agent doesn't have the full guide built in — too long for every call)

Phase 1.1 (deferred): adds structured `param_overrides_json` output + an
`NV_ApplyOverrides` consumer node that injects overrides into a config bus.
Ship that only after Phase 1.0 proves agent suggestions are useful in
practice.
"""

import base64
import hashlib
import io
import json
import time

import numpy as np
import requests
import torch
from PIL import Image

from .api_keys import resolve_api_key


# ---------------------------------------------------------------------------
# Models — vision-capable, accessible via OpenRouter (matches prompt_refiner.py)
# ---------------------------------------------------------------------------

_MODELS = [
    # Anthropic — strongest for structured analytical output, vision-capable
    "anthropic/claude-opus-4.7",
    "anthropic/claude-opus-4.6",
    "anthropic/claude-sonnet-4.6",
    "anthropic/claude-haiku-4.5",
    # OpenAI — strong vision, reasoning
    "openai/gpt-5.4",
    "openai/gpt-5.4-mini",
    "openai/gpt-5.5",
    # Google — cheaper, large context
    "google/gemini-3.1-pro-preview",
    "google/gemini-3-flash-preview",
]


# ---------------------------------------------------------------------------
# OpenRouter pricing snapshot (USD per 1M tokens) — for cost diagnostic only
# Prices change; verify at openrouter.ai/models. Approximate values 2026-04.
# ---------------------------------------------------------------------------

_OR_PRICING = {
    "anthropic/claude-opus-4.7":         (15.00, 75.00),
    "anthropic/claude-opus-4.6":         (15.00, 75.00),
    "anthropic/claude-sonnet-4.6":       ( 3.00, 15.00),
    "anthropic/claude-haiku-4.5":        ( 0.80,  4.00),
    "openai/gpt-5.4":                    ( 5.00, 20.00),
    "openai/gpt-5.4-mini":               ( 0.60,  2.40),
    "openai/gpt-5.5":                    (10.00, 40.00),
    "google/gemini-3.1-pro-preview":     ( 2.00, 12.00),
    "google/gemini-3-flash-preview":     ( 0.50,  3.00),
}


def _estimate_or_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    rates = _OR_PRICING.get(model)
    if rates is None:
        return 0.0
    return (prompt_tokens / 1_000_000.0) * rates[0] + (completion_tokens / 1_000_000.0) * rates[1]


# ---------------------------------------------------------------------------
# Default system prompt — encodes agent role + output format
# Kept compact; user can extend via `system_prompt_extra` widget. The full
# Mysterious Params Guide (D-132, ~860 lines) is intentionally NOT inlined
# here because every call would pay its tokens. Users paste relevant
# excerpts via the `system_prompt_extra` input when needed.
# ---------------------------------------------------------------------------

_DEFAULT_SYSTEM_PROMPT = """You are a parameter-tuning consultant for a ComfyUI VACE face-swap pipeline.

Your job: look at the sampled input frames and the user's stated intent, then \
recommend specific parameter values from the pipeline's known nodes. You are a \
SUGGESTER, not an executor — the user reviews your output and applies changes \
manually.

Pipeline context (high level):
- Input: stitched output from a Kling face-swap, then VACE refines via inpainting
- Key post-decode nodes: NV_SimpleColorMatch (mean-shift color anchor), \
NV_TextureHarmonize (sharpness/grain matching to plate), NV_CropColorFix (legacy \
multi-stage color, opt-in only), NV_InpaintStitch2 (composites crop back to canvas)
- Tracking: NV_PointDrivenBBox + NV_CoTrackerBridge drive crop window
- VACE prep: NV_VaceControlVideoPrep + NV_VacePrePassReference + NV_StaticVaceMask

The canonical reference is `MYSTERIOUS_PARAMS_GUIDE.md` (~860 lines, in node_notes/guides/). \
Relevant excerpts may be appended below in 'EXTRA GUIDANCE'. If a parameter you \
want to recommend is not covered in your context, say so explicitly rather than \
guess — the user can paste the relevant section.

Output format (REQUIRED):

## Diagnosis
1-3 sentences identifying what the input + intent suggest as the likely root cause.

## Recommended parameter overrides

| Node | Parameter | Current (if known) | Recommended | Reasoning |
|------|-----------|--------------------|-------------|-----------|
| ... | ... | ... | ... | ... |

## Confidence
HIGH / MEDIUM / LOW + one sentence on what could falsify your diagnosis.

## What to verify after re-rendering
1-3 specific things the user should look for in the next output to confirm or \
reject your suggestions.

Constraints on your output:
- Keep total length under 600 words.
- If your only honest answer is 'I cannot tell from this input', say so. \
Do not invent parameter recommendations to fill the table.
- Do not suggest changes outside the named pipeline nodes unless explicitly \
asked. Do not suggest workflow rewiring.
- When the user mentions a SPECIFIC SYMPTOM, prefer one targeted recommendation \
over a sweeping multi-param overhaul.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hash_intent(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _sample_frames(images, max_n):
    """Uniformly sample up to max_n frames from a [B, H, W, C] tensor.

    Returns a list of (frame_idx, frame_tensor) tuples. Always includes
    first and last frame so the agent sees temporal extremes.
    """
    B = images.shape[0]
    if B <= max_n:
        return [(i, images[i]) for i in range(B)]
    # Always include first and last; uniformly distribute the rest
    if max_n == 1:
        idxs = [0]
    elif max_n == 2:
        idxs = [0, B - 1]
    else:
        # B-1 to land last index inside range; np.linspace handles the spacing
        idxs = list(np.linspace(0, B - 1, num=max_n).round().astype(int))
        # Dedup while preserving order (in case of small B + large max_n)
        seen = set()
        uniq = []
        for i in idxs:
            if i not in seen:
                seen.add(i)
                uniq.append(int(i))
        idxs = uniq
    return [(i, images[i]) for i in idxs]


def _frame_to_base64_jpeg(frame_tensor, quality=85):
    """Convert a [H, W, C] fp tensor in [0, 1] to base64 JPEG.

    JPEG (not PNG) keeps token cost down — at quality=85 the visual difference
    is imperceptible for parameter-diagnosis purposes, and payload is ~5x
    smaller than PNG. Returns (base64_str, mime_type).
    """
    arr = frame_tensor.detach().cpu().numpy()
    if arr.dtype in (np.float16, np.float32, np.float64):
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]  # drop alpha if present
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return b64, "image/jpeg"


def _mask_to_base64_jpeg(mask_tensor, quality=85):
    """Convert a [H, W] mask tensor in [0, 1] to base64 JPEG (grayscale)."""
    arr = mask_tensor.detach().cpu().numpy()
    if arr.dtype in (np.float16, np.float32, np.float64):
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr.squeeze(-1)
    img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return b64, "image/jpeg"


def _build_user_message(user_intent, manifest_summary, sampled_image_b64s,
                        sampled_mask_b64s, frame_indices, total_frames):
    """Build the OpenRouter user message: text + multipart images.

    Text section enumerates frame indices so the agent can reason about temporal
    structure. Mask images (if provided) are interleaved with image frames so
    agent can see "what region is being inpainted at frame N".
    """
    lines = []
    lines.append(f"USER INTENT:\n{user_intent.strip()}\n")

    if manifest_summary and manifest_summary.strip():
        # Truncate aggressive — manifest can be huge; a summary is what we want
        ms = manifest_summary.strip()
        if len(ms) > 6000:
            ms = ms[:6000] + "\n[...truncated]"
        lines.append(f"PRIOR RENDER MANIFEST:\n{ms}\n")

    lines.append(
        f"INPUT FRAMES:\nSampled {len(frame_indices)} frames (out of {total_frames} total). "
        f"Frame indices in order: {frame_indices}.\n"
    )
    if sampled_mask_b64s:
        lines.append(
            "Each frame is followed by its corresponding mask "
            "(white = inpaint region, black = preserved).\n"
        )

    lines.append(
        "Please analyze the inputs and produce a recommendation in the format "
        "specified by the system prompt. Be specific about parameter values."
    )
    text_part = "\n".join(lines)

    content_parts = [{"type": "text", "text": text_part}]
    for i, img_b64 in enumerate(sampled_image_b64s):
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
        })
        if sampled_mask_b64s and i < len(sampled_mask_b64s):
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{sampled_mask_b64s[i]}"},
            })

    return content_parts


def _call_openrouter(api_key, model, system_prompt, user_content_parts,
                     max_tokens, temperature):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/comfyui",
        "X-Title": "NV Agent Param Planner",
    }
    messages = []
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append({"role": "user", "content": user_content_parts})

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    print(f"[NV_AgentParamPlanner] Calling OpenRouter ({model}) with "
          f"{sum(1 for p in user_content_parts if p.get('type') == 'image_url')} images...")
    t0 = time.time()
    r = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers, json=payload, timeout=300,
    )
    dt = time.time() - t0
    print(f"[NV_AgentParamPlanner] Response: {r.status_code} in {dt:.2f}s")

    if r.status_code != 200:
        body_preview = r.text[:600] if r.text else "<empty>"
        raise RuntimeError(f"OpenRouter API error {r.status_code}: {body_preview}")

    result = r.json()
    if "choices" not in result or not result["choices"]:
        raise RuntimeError(f"OpenRouter returned no choices: {result}")

    choice = result["choices"][0]
    msg = choice.get("message", {})
    text = msg.get("content", "") or ""
    text = text.strip()

    # Extract token usage if present
    usage = result.get("usage", {}) or {}
    pt = int(usage.get("prompt_tokens", 0))
    ct = int(usage.get("completion_tokens", 0))
    cost = _estimate_or_cost(model, pt, ct)

    return text, {"prompt_tokens": pt, "completion_tokens": ct, "cost_usd": cost,
                  "request_time_s": dt, "finish_reason": choice.get("finish_reason", "")}


# ---------------------------------------------------------------------------
# Node class
# ---------------------------------------------------------------------------

class NV_AgentParamPlanner:
    """Phase 1.0 MVP: LLM-assisted parameter suggester for VACE face-swap pipelines.

    Reads sampled frames + intent + optional manifest, returns a markdown
    recommendation. User reviews and applies changes manually. No automatic
    mutation of any other node's behavior.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Input frames for the agent to analyze. Sampled down to "
                               "max_input_frames evenly-spaced frames; first and last always "
                               "included. Typically wired from VHS_LoadVideo, an InpaintCrop2 "
                               "output, or a NV_VaceDebugPreview."
                }),
                "user_intent": ("STRING", {
                    "default": "Describe the shot type and the symptom you want analyzed. "
                               "Example: 'walking talking-head shot, dark BG, AI face looks "
                               "matte/flat compared to plate. What params to tune?'",
                    "multiline": True,
                    "tooltip": "Free-form description of the shot type, symptoms, and what "
                               "you want the agent to focus on. The more specific the symptom, "
                               "the more targeted the recommendation. Vague intent → vague "
                               "recommendation."
                }),
            },
            "optional": {
                "mask": ("MASK", {
                    "tooltip": "Optional inpaint mask (1 = generated region, 0 = preserved). "
                               "When wired, the agent sees what region is being inpainted at "
                               "each sampled frame, which sharpens diagnosis on mask-edge or "
                               "context-scope issues."
                }),
                "manifest_summary": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Paste the JSON output of a prior render's NV_RenderManifest "
                               "here, OR a hand-written summary of the current node settings. "
                               "Gives the agent ground truth on 'current values' so it can "
                               "recommend deltas instead of absolutes. Truncated at 6000 chars."
                }),
                "system_prompt_extra": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Additional system-prompt content APPENDED to the default. Use "
                               "this to paste relevant excerpts from MYSTERIOUS_PARAMS_GUIDE.md "
                               "(the full guide is too long to inline every call). When you have "
                               "a known-tricky parameter in scope, paste its section here."
                }),
                "model": (_MODELS, {
                    "default": "anthropic/claude-opus-4.7",
                    "tooltip": "OpenRouter model. Claude Opus 4.7 = strongest analytical output "
                               "(default). Sonnet 4.6 = ~5x cheaper, ~95% as good for typical "
                               "tuning tasks. Haiku 4.5 = fastest/cheapest, use for quick checks. "
                               "Vision-capable across all listed models."
                }),
                "max_input_frames": ("INT", {
                    "default": 5, "min": 1, "max": 20, "step": 1,
                    "tooltip": "How many frames to sample from the input video for the agent "
                               "to look at. 5 is usually sufficient (first/mid/last + 2 mid-points). "
                               "More frames = higher token cost. Image content cost dominates."
                }),
                "max_tokens": ("INT", {
                    "default": 1500, "min": 200, "max": 8000, "step": 100,
                    "tooltip": "Max output tokens. Default 1500 fits the prescribed output "
                               "format (Diagnosis + table + Confidence + Verify list). Raise "
                               "for verbose models or longer system_prompt_extra reasoning."
                }),
                "temperature": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 1.5, "step": 0.05,
                    "tooltip": "Sampling temperature. 0.3 keeps recommendations stable across "
                               "re-runs. 0.0 = deterministic but may overcommit; >0.7 = "
                               "creative/exploratory, less reliable."
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "OpenRouter API key. LEAVE EMPTY to use OPENROUTER_API_KEY env "
                               "var or .env file (recommended — same pattern as NV_PromptRefiner). "
                               "Only paste here if you specifically want to override per-node."
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("recommendation", "info")
    FUNCTION = "execute"
    CATEGORY = "NV_Utils/Agent"
    DESCRIPTION = (
        "Phase 1.0 MVP: LLM-assisted parameter suggester for VACE face-swap. "
        "Reads sampled frames + intent + optional manifest summary, returns a markdown "
        "recommendation with parameter overrides + reasoning + confidence. NO auto-apply — "
        "user reviews and adjusts widgets manually. Routes through OpenRouter (reuses "
        "OPENROUTER_API_KEY). Default model = Claude Opus 4.7."
    )

    def execute(self, images, user_intent,
                mask=None, manifest_summary="", system_prompt_extra="",
                model="anthropic/claude-opus-4.7", max_input_frames=5,
                max_tokens=1500, temperature=0.3, api_key=""):
        TAG = "[NV_AgentParamPlanner]"
        intent = (user_intent or "").strip()
        if not intent:
            err = f"{TAG} user_intent is empty — cannot make recommendations without a description."
            print(err)
            return ("", err)

        # Resolve key (explicit input wins, then env vars, then .env)
        try:
            key = resolve_api_key(api_key, provider="openrouter")
        except RuntimeError as e:
            err = f"{TAG} API key resolution failed: {e}"
            print(err)
            return ("", err)

        # Sample frames evenly
        B = int(images.shape[0])
        sampled = _sample_frames(images, max_input_frames)
        frame_indices = [idx for idx, _ in sampled]
        sampled_image_b64s = []
        for _, frame in sampled:
            b64, _mime = _frame_to_base64_jpeg(frame)
            sampled_image_b64s.append(b64)

        # Mask, if provided — match sampled frame indices, broadcast single-frame mask
        sampled_mask_b64s = []
        if mask is not None:
            m = mask
            if m.dim() == 2:
                m = m.unsqueeze(0)
            for idx in frame_indices:
                fi = min(idx, m.shape[0] - 1)
                b64, _mime = _mask_to_base64_jpeg(m[fi])
                sampled_mask_b64s.append(b64)

        # Compose system prompt
        system_prompt = _DEFAULT_SYSTEM_PROMPT
        if system_prompt_extra and system_prompt_extra.strip():
            system_prompt = (
                system_prompt
                + "\n\nEXTRA GUIDANCE (user-supplied excerpts from "
                "MYSTERIOUS_PARAMS_GUIDE.md or other notes):\n\n"
                + system_prompt_extra.strip()
            )

        # Build user message + call
        user_content = _build_user_message(
            user_intent=intent,
            manifest_summary=manifest_summary,
            sampled_image_b64s=sampled_image_b64s,
            sampled_mask_b64s=sampled_mask_b64s,
            frame_indices=frame_indices,
            total_frames=B,
        )

        try:
            text, usage = _call_openrouter(
                api_key=key, model=model, system_prompt=system_prompt,
                user_content_parts=user_content,
                max_tokens=max_tokens, temperature=temperature,
            )
        except Exception as e:
            err = f"{TAG} API call failed: {e}"
            print(err)
            return ("", err)

        # Compose audit info (separate from the recommendation itself)
        info_lines = [
            f"{TAG} model={model}",
            f"  Frames sampled: {len(frame_indices)} of {B} (indices {frame_indices})",
            f"  Mask provided: {'yes' if mask is not None else 'no'}",
            f"  Manifest summary: {len(manifest_summary)} chars" if manifest_summary else
            "  Manifest summary: (none)",
            f"  System prompt extra: {len(system_prompt_extra)} chars" if system_prompt_extra else
            "  System prompt extra: (none)",
            f"  Tokens: {usage['prompt_tokens']} prompt + {usage['completion_tokens']} completion",
            f"  Estimated cost: ${usage['cost_usd']:.4f} USD",
            f"  Request time: {usage['request_time_s']:.2f}s",
            f"  Finish reason: {usage['finish_reason']}",
            f"  Intent hash: {_hash_intent(intent)}",
        ]
        info = "\n".join(info_lines)
        print(info)

        if usage["finish_reason"] == "length":
            warn = ("\n\n[WARNING: response was truncated by max_tokens. "
                    "Raise max_tokens or shorten system_prompt_extra.]")
            text = text + warn

        return (text, info)


NODE_CLASS_MAPPINGS = {
    "NV_AgentParamPlanner": NV_AgentParamPlanner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_AgentParamPlanner": "NV Agent Param Planner",
}
