"""NV Gemini Video Course — Extract structured data from tutorial videos and synthesize course plans.

Two-node pipeline for turning tutorial/course video series into structured learning material:

  NV_GeminiVideoExtractor — Upload a single video to the Gemini Files API, extract
    structured JSON with segments, learning objectives, skills, and timestamps.
    Saves extraction to disk for later synthesis.

  NV_GeminiCourseSynthesizer — Combine multiple extraction JSONs into a cohesive
    course plan with modules, lessons, exercises, and gap analysis.

Uses Gemini REST API with the Files API for video upload (supports up to 2GB / ~3 hours).
Auth: api_key input, or GEMINI_API_KEY / GOOGLE_API_KEY env vars.

Pricing reference (per 20-min video):
  - Flash: ~$0.01  |  Pro: ~$0.22
  - Default resolution: ~300 tok/s (~360K tokens for 20 min)
  - Low resolution: ~100 tok/s (~120K tokens for 20 min)
"""

import json
import os
import re
import time

import requests
from pathlib import Path

import folder_paths

from .api_keys import resolve_api_key


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_API_BASE = "https://generativelanguage.googleapis.com"
_UPLOAD_URL = f"{_API_BASE}/upload/v1beta/files"
_FILES_URL = f"{_API_BASE}/v1beta/files"

_SUPPORTED_VIDEO_EXTS = {".mp4", ".mpeg", ".mov", ".avi", ".flv", ".webm", ".wmv", ".3gpp", ".3gp"}

_MIME_MAP = {
    ".mp4": "video/mp4",
    ".mpeg": "video/mpeg",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".flv": "video/x-flv",
    ".webm": "video/webm",
    ".wmv": "video/x-ms-wmv",
    ".3gpp": "video/3gpp",
    ".3gp": "video/3gpp",
}

# Safety settings: BLOCK_NONE for all categories to prevent false positives
# on educational content (coding "hacks", medical training, etc.)
_SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

# Max retries for transient API errors (429, 5xx)
_MAX_RETRIES = 2
_RETRY_BASE_DELAY = 5  # seconds, doubles each retry

# Pricing per 1M tokens (USD, standard tier, <=200k context)
# Source: https://ai.google.dev/pricing
_PRICING = {
    "gemini-2.0-flash":        {"input": 0.10,  "output": 0.40},
    "gemini-2.5-flash":        {"input": 0.15,  "output": 0.60},
    "gemini-2.5-pro":          {"input": 1.25,  "output": 10.00},
    "gemini-3-flash-preview":  {"input": 0.15,  "output": 0.60},
    "gemini-3-pro-preview":    {"input": 1.25,  "output": 10.00},
    "gemini-3.1-pro-preview":  {"input": 1.25,  "output": 10.00},
}


def _estimate_cost(model, prompt_tokens, response_tokens):
    """Estimate API call cost in USD from token counts."""
    pricing = _PRICING.get(model)
    if not pricing or not isinstance(prompt_tokens, (int, float)):
        return None
    input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
    output_cost = (response_tokens / 1_000_000) * pricing["output"]
    return round(input_cost + output_cost, 4)


def _get_video_duration(file_path):
    """Get video duration in seconds using ffprobe. Returns None on failure."""
    import subprocess
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(file_path)],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception:
        pass
    return None


def _format_duration(seconds):
    """Format seconds as MM:SS or HH:MM:SS."""
    if seconds is None:
        return "??:??"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _estimate_video_cost(duration_s, model, media_resolution="default"):
    """Estimate extraction cost from video duration and model pricing.

    Approximation: default ~300 tok/s, low ~100 tok/s input.
    Output estimated at ~3K tokens per extraction.
    """
    pricing = _PRICING.get(model)
    if not pricing or duration_s is None:
        return None
    tok_per_sec = 100 if media_resolution == "low" else 300
    input_tokens = duration_s * tok_per_sec
    output_tokens = 3000  # typical extraction output
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return round(input_cost + output_cost, 4)


# ---------------------------------------------------------------------------
# Extraction JSON Schema (for Gemini responseSchema enforcement)
# ---------------------------------------------------------------------------

_EXTRACTION_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "video_title_guess": {"type": "STRING", "nullable": True},
        "overall_topic": {"type": "STRING"},
        "video_summary": {"type": "STRING"},
        "content_type": {
            "type": "STRING",
            "enum": ["theory", "demo", "guided_exercise", "walkthrough", "mixed"],
        },
        "theory_demo_ratio": {
            "type": "STRING",
            "enum": ["all_theory", "mostly_theory", "balanced", "mostly_demo", "all_demo"],
        },
        "target_level": {
            "type": "STRING",
            "enum": ["beginner", "beginner_intermediate", "intermediate", "advanced", "mixed", "unclear"],
        },
        "prerequisites": {"type": "ARRAY", "items": {"type": "STRING"}},
        "learning_objectives": {"type": "ARRAY", "items": {"type": "STRING"}},
        "segments": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "start_mmss": {"type": "STRING"},
                    "end_mmss": {"type": "STRING"},
                    "segment_title": {"type": "STRING"},
                    "segment_type": {
                        "type": "STRING",
                        "enum": [
                            "concept_explanation", "demo", "setup",
                            "recap_transition", "exercise", "q_and_a", "other",
                        ],
                    },
                    "pedagogical_role": {"type": "STRING"},
                    "core_points": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "visible_artifacts": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "skills_or_actions": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "evidence_confidence": {
                        "type": "STRING",
                        "enum": ["high", "medium", "low"],
                    },
                },
                "required": [
                    "start_mmss", "end_mmss", "segment_title", "segment_type",
                    "core_points", "evidence_confidence",
                ],
            },
        },
        "examples_and_demos": {"type": "ARRAY", "items": {"type": "STRING"}},
        "tools_platforms_or_frameworks": {"type": "ARRAY", "items": {"type": "STRING"}},
        "likely_assignments_or_practice": {"type": "ARRAY", "items": {"type": "STRING"}},
        "key_terms": {"type": "ARRAY", "items": {"type": "STRING"}},
        "uncertainties": {"type": "ARRAY", "items": {"type": "STRING"}},
        "topic_coverage": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "topic": {"type": "STRING"},
                    "depth": {
                        "type": "STRING",
                        "enum": ["mentioned", "explained", "demonstrated"],
                    },
                },
                "required": ["topic", "depth"],
            },
        },
    },
    "required": [
        "overall_topic", "video_summary", "content_type", "theory_demo_ratio",
        "target_level", "learning_objectives", "segments",
        "key_terms", "uncertainties", "topic_coverage",
    ],
}


# ---------------------------------------------------------------------------
# Phase 1: Per-video extraction prompt
# ---------------------------------------------------------------------------

_EXTRACTION_PROMPT = """Analyze this tutorial/course video and extract structured learning content for curriculum design.

Return a single JSON object with these fields:

TOP-LEVEL CLASSIFICATION:
- video_title_guess: best guess at the video title from intro/slides/context (null if unclear)
- overall_topic: main subject of this video
- video_summary: 2-4 sentence free-text summary capturing the video's narrative arc and key takeaways — preserve nuance that structured fields might lose
- content_type: classify the WHOLE video as one of: theory | demo | guided_exercise | walkthrough | mixed
  - theory = slides, lecture, conceptual explanation with no software interaction
  - demo = instructor shows something in software, learner watches
  - guided_exercise = instructor walks learner through steps they should follow along with
  - walkthrough = end-to-end workflow demonstration
  - mixed = significant portions of both theory and hands-on
- theory_demo_ratio: all_theory | mostly_theory | balanced | mostly_demo | all_demo
- target_level: beginner | beginner_intermediate | intermediate | advanced | mixed | unclear
- prerequisites: knowledge assumed by the instructor
- learning_objectives: what a student should be able to do after watching

SEGMENTS (time-based breakdown):
- segments: array of time-segmented content, each with:
  - start_mmss / end_mmss (MM:SS timestamps)
  - segment_title
  - segment_type: concept_explanation | demo | setup | recap_transition | exercise | q_and_a | other
  - pedagogical_role: how this segment fits into the learning arc
  - core_points: key concepts or facts taught
  - visible_artifacts: describe what's ON screen with specificity — "slide with comparison table of SOPs vs LOPs" not just "presentation slide". Note if it's a slide, diagram, node graph, terminal, UI screenshot, website, or talking head.
  - skills_or_actions: specific things demonstrated or practiced (leave empty for pure theory segments)
  - evidence_confidence: high | medium | low

TOPIC COVERAGE (for redundancy detection across videos):
- topic_coverage: for each major topic in the video, classify coverage depth:
  - topic: the topic name
  - depth: mentioned (named but not explained) | explained (conceptually taught) | demonstrated (shown hands-on in software)

SUPPORTING FIELDS:
- examples_and_demos: specific examples or demonstrations shown
- tools_platforms_or_frameworks: software, tools, or frameworks used or mentioned
- likely_assignments_or_practice: exercises the instructor suggests or that naturally follow
- key_terms: domain vocabulary introduced or used
- uncertainties: anything you're not sure about — mark rather than guess

RULES:
- Ground every claim in spoken audio or visible frames. Do NOT infer content not present.
- If something is unclear, add it to "uncertainties" instead of guessing.
- Distinguish concept explanation vs procedural demonstration vs setup vs recap.
- Do NOT produce a transcript or verbatim quotes.
- Do NOT extract exact code from the screen — describe code logic and patterns instead.
- Use MM:SS timestamps. Prefer approximate ranges over false precision.
- Segment boundaries should follow natural topic shifts, not arbitrary time divisions.
- For visible_artifacts: describe content specifics, not just "presentation slide" — what is ON the slide?
- The video_summary should capture the pedagogical narrative, not just list topics.
- For topic_coverage: distinguish "mentioned" (named in passing), "explained" (taught conceptually), and "demonstrated" (shown hands-on). This is critical for detecting when two videos both cover the same topic at different depths."""

# ---------------------------------------------------------------------------
# Phase 2: Course plan synthesis prompt
# ---------------------------------------------------------------------------

_SYNTHESIS_PROMPT = """You are a curriculum designer. Given the following JSON extractions from a series of tutorial videos, create a structured course plan.

VIDEO EXTRACTIONS:
{extractions}

TASK:
Synthesize these video extractions into a comprehensive course plan. Group videos into thematic modules (not necessarily by original video order). Consider pedagogical flow — concepts should build on each other.

OUTPUT FORMAT: Return well-structured Markdown with these sections:

# [Course Title]

## Course Overview
[2-3 paragraph description of what the course covers and who it's for]

## Learning Outcomes
[Bulleted list of what students will be able to do after completing the course]

## Prerequisites
[What students should know before starting]

## Course Structure

### Module 1: [Module Title]
**Description:** [What this module covers]
**Videos:** [Which source videos map to this module]

#### Lesson 1.1: [Lesson Title]
- **Source:** [Video title/filename]
- **Topics:** [Key topics covered]
- **Skills:** [Specific skills developed]
- **Exercises:** [Suggested practice activities]

[... repeat for each lesson ...]

[... repeat for each module ...]

## Gap Analysis

### Missing Topics
[Topics that would strengthen the course but aren't covered in any video]

### Bridge Topics Needed
[Concepts needed to connect videos that currently have gaps between them]

### Redundancy Notes
[Topics covered in multiple videos — note which coverage is most thorough]

## Recommended Additions
[Suggested supplementary content, readings, or exercises to fill gaps]

## Tools & Platforms Referenced
[Consolidated list across all videos]

RULES:
- Group by thematic relationship, not just video order.
- If two videos cover similar ground, note the overlap and recommend which to use as primary.
- Flag any logical gaps where a student would struggle moving from one topic to the next.
- Suggest practical exercises for modules that are explanation-heavy.
- Keep lesson descriptions concise — reference the source video for details."""


# ---------------------------------------------------------------------------
# Helpers: API key & MIME type
# ---------------------------------------------------------------------------

def _get_mime_type(file_path):
    """Get MIME type from file extension."""
    ext = Path(file_path).suffix.lower()
    return _MIME_MAP.get(ext, "application/octet-stream")


# _resolve_api_key removed — now imported from .api_keys


# ---------------------------------------------------------------------------
# Helpers: Gemini Files API (resumable upload, poll, delete)
# ---------------------------------------------------------------------------

def _upload_file(file_path, api_key, display_name=None):
    """Upload a file to the Gemini Files API using resumable upload protocol.

    Returns file metadata dict with 'name', 'uri', 'state', 'mimeType'.
    """
    file_size = os.path.getsize(file_path)
    mime = _get_mime_type(file_path)
    name = display_name or Path(file_path).stem

    print(f"[NV_GeminiVideoExtractor] Uploading {Path(file_path).name} "
          f"({file_size / 1024 / 1024:.1f} MB, {mime})")

    # Step 1: Start resumable upload
    start_headers = {
        "X-Goog-Upload-Protocol": "resumable",
        "X-Goog-Upload-Command": "start",
        "X-Goog-Upload-Header-Content-Length": str(file_size),
        "X-Goog-Upload-Header-Content-Type": mime,
        "Content-Type": "application/json",
    }
    resp = requests.post(
        f"{_UPLOAD_URL}?key={api_key}",
        headers=start_headers,
        json={"file": {"display_name": name}},
        timeout=30,
    )
    if resp.status_code != 200:
        raise RuntimeError(
            f"Files API upload start failed ({resp.status_code}): {resp.text[:500]}"
        )

    upload_url = resp.headers.get("X-Goog-Upload-URL") or resp.headers.get("x-goog-upload-url")
    if not upload_url:
        raise RuntimeError("Files API did not return an upload URL")

    # Step 2: Upload file data (streamed to avoid loading entire file into memory)
    upload_start = time.time()
    upload_headers = {
        "Content-Length": str(file_size),
        "X-Goog-Upload-Offset": "0",
        "X-Goog-Upload-Command": "upload, finalize",
    }
    with open(file_path, "rb") as f:
        resp = requests.post(upload_url, headers=upload_headers, data=f, timeout=600)

    upload_time = time.time() - upload_start
    if resp.status_code != 200:
        raise RuntimeError(
            f"Files API upload failed ({resp.status_code}): {resp.text[:500]}"
        )

    try:
        resp_data = resp.json()
    except (json.JSONDecodeError, ValueError):
        raise RuntimeError(
            f"Files API upload returned non-JSON response (status {resp.status_code}): "
            f"{resp.text[:500]}"
        )
    file_info = resp_data.get("file", resp_data)
    print(f"[NV_GeminiVideoExtractor] Upload complete in {upload_time:.1f}s "
          f"— {file_info.get('name', 'unknown')}")

    return file_info


def _poll_file_active(file_name, api_key, timeout=300, interval=5):
    """Poll Gemini Files API until file state is ACTIVE.

    Raises RuntimeError on timeout or FAILED state.
    """
    start = time.time()
    state = "UNKNOWN"
    while time.time() - start < timeout:
        # file_name is the full resource name (e.g. "files/abc123"), so use _API_BASE/v1beta/
        resp = requests.get(f"{_API_BASE}/v1beta/{file_name}?key={api_key}", timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(
                f"Files API status check failed ({resp.status_code}): {resp.text[:300]}"
            )

        try:
            info = resp.json()
        except (json.JSONDecodeError, ValueError):
            raise RuntimeError(
                f"Files API returned non-JSON response during poll: {resp.text[:300]}"
            )
        state = info.get("state", "UNKNOWN")

        if state == "ACTIVE":
            print(f"[NV_GeminiVideoExtractor] File ready ({time.time() - start:.0f}s)")
            return info
        if state == "FAILED":
            error_msg = info.get("error", {}).get("message", "unknown error")
            raise RuntimeError(f"File processing failed: {error_msg}")

        print(f"[NV_GeminiVideoExtractor] File state: {state}, waiting {interval}s...")
        time.sleep(interval)

    raise RuntimeError(f"File processing timed out after {timeout}s (last state: {state})")


def _delete_file(file_name, api_key):
    """Delete a file from the Gemini Files API. Silently ignores errors."""
    try:
        # file_name is the full resource name (e.g. "files/abc123")
        resp = requests.delete(f"{_API_BASE}/v1beta/{file_name}?key={api_key}", timeout=15)
        if resp.status_code < 300:
            print(f"[NV_GeminiVideoExtractor] Cleaned up remote file: {file_name}")
        else:
            print(f"[NV_GeminiVideoExtractor] Warning: delete returned {resp.status_code} "
                  f"for {file_name} — file may remain in Gemini storage")
    except Exception as e:
        print(f"[NV_GeminiVideoExtractor] Warning: cleanup failed for {file_name}: {e}")


# ---------------------------------------------------------------------------
# Helpers: Gemini generateContent (with retry for transient errors)
# ---------------------------------------------------------------------------

def _generate_content(api_key, model, contents, system_instruction=None,
                      temperature=1.0, max_tokens=8192, json_mode=False,
                      response_schema=None, media_resolution="default"):
    """Call Gemini generateContent API with retry for transient errors.

    Args:
        api_key: Gemini API key.
        model: Model name (e.g. "gemini-2.5-pro").
        contents: List of content part dicts for the request.
        system_instruction: Optional system-level instruction text.
        temperature: Sampling temperature.
        max_tokens: Maximum output tokens.
        json_mode: If True, request JSON output via responseMimeType.
        response_schema: Optional JSON schema dict for structured output enforcement.
        media_resolution: "default" or "low" — controls tokens per video frame.

    Returns:
        Response text string.
    """
    # Strip provider prefix if present (e.g. "google/gemini-2.5-pro" -> "gemini-2.5-pro")
    if "/" in model:
        model = model.split("/", 1)[1]

    payload = {"contents": [{"parts": contents}]}

    gen_config = {
        "maxOutputTokens": max_tokens,
        "temperature": temperature,
    }
    if json_mode:
        gen_config["responseMimeType"] = "application/json"
        if response_schema:
            gen_config["responseSchema"] = response_schema

    # Detect Gemini 3+ models (support thinkingConfig, mediaResolution via v1alpha)
    is_gemini3 = any(model.startswith(p) for p in ("gemini-3", "gemini-3."))

    # Gemini 3+ uses v1alpha endpoint for thinking and mediaResolution
    use_v1alpha = is_gemini3

    # Media resolution: controls tokens per video frame (uppercase enum required)
    # "low" = ~100 tok/s (cheaper, good for talking-head), "default" = ~300 tok/s
    _MEDIA_RES_MAP = {"low": "MEDIA_RESOLUTION_LOW"}
    if media_resolution and media_resolution != "default":
        mapped = _MEDIA_RES_MAP.get(media_resolution, f"MEDIA_RESOLUTION_{media_resolution.upper()}")
        gen_config["mediaResolution"] = mapped

    # Gemini 3+ thinking config — BUT thinking is incompatible with JSON mode
    # (responseMimeType + thinkingConfig = 400 error). Only enable for non-JSON requests.
    use_thinking = is_gemini3 and not json_mode
    if use_thinking:
        gen_config["thinkingConfig"] = {"thinkingLevel": "medium"}

    payload["generationConfig"] = gen_config

    # Safety settings: BLOCK_NONE for educational content
    payload["safetySettings"] = _SAFETY_SETTINGS

    if system_instruction and system_instruction.strip():
        payload["systemInstruction"] = {
            "parts": [{"text": system_instruction.strip()}]
        }

    api_version = "v1alpha" if use_v1alpha else "v1beta"
    url = f"{_API_BASE}/{api_version}/models/{model}:generateContent?key={api_key}"

    print(f"[Gemini] Calling {model} (max_tokens={max_tokens}, temp={temperature}, "
          f"json={json_mode}, schema={'yes' if response_schema else 'no'}, "
          f"api={api_version}{', thinking=medium' if use_thinking else ''})")

    # Retry loop for transient errors (429, 5xx)
    last_error = None
    for attempt in range(_MAX_RETRIES + 1):
        if attempt > 0:
            delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
            print(f"[Gemini] Retry {attempt}/{_MAX_RETRIES} after {delay}s...")
            time.sleep(delay)

        req_start = time.time()
        try:
            resp = requests.post(
                url, json=payload, headers={"Content-Type": "application/json"}, timeout=600
            )
        except requests.exceptions.RequestException as e:
            last_error = e
            print(f"[Gemini] Network error: {e}")
            continue

        req_time = time.time() - req_start

        # Retry on transient errors
        if resp.status_code == 429 or resp.status_code >= 500:
            last_error = RuntimeError(
                f"Gemini API error {resp.status_code}: {resp.text[:300]}"
            )
            print(f"[Gemini] Transient error {resp.status_code} ({req_time:.1f}s)")
            continue

        # Non-retryable error
        if resp.status_code != 200:
            raise RuntimeError(f"Gemini API error {resp.status_code}: {resp.text[:500]}")

        # Success
        result = resp.json()

        # Log usage and estimate cost (include thinking tokens for Gemini 3+)
        usage = result.get("usageMetadata", {})
        prompt_tokens = usage.get("promptTokenCount", 0)
        response_tokens = usage.get("candidatesTokenCount", 0)
        thinking_tokens = usage.get("thoughtsTokenCount", 0)
        # Thinking tokens are billed at output rate
        billable_output = response_tokens + thinking_tokens
        cost = _estimate_cost(model, prompt_tokens, billable_output)
        cost_str = f", est ${cost:.4f}" if cost is not None else ""
        thinking_str = f", {thinking_tokens} thinking" if thinking_tokens else ""

        print(f"[Gemini] Response in {req_time:.1f}s — "
              f"{prompt_tokens or '?'} prompt tokens, "
              f"{response_tokens or '?'} response tokens{thinking_str}{cost_str}")

        # Extract text from response (filter out thinking parts from Gemini 3+)
        candidates = result.get("candidates", [])
        if not candidates:
            raise RuntimeError("Gemini API returned no candidates")

        parts = candidates[0].get("content", {}).get("parts", [])
        text_parts = [p["text"] for p in parts if "text" in p and not p.get("thought", False)]
        if not text_parts:
            text_parts = [p["text"] for p in parts if "text" in p]
        if not text_parts:
            raise RuntimeError("Gemini API returned no text content")

        text = "\n".join(text_parts).strip()

        # Return text + usage metadata as a tuple
        return text, {
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens,
            "thinking_tokens": thinking_tokens,
            "estimated_cost_usd": cost,
            "response_time_s": round(req_time, 1),
        }

    # All retries exhausted
    raise RuntimeError(f"Gemini API failed after {_MAX_RETRIES + 1} attempts: {last_error}")


# ---------------------------------------------------------------------------
# Helpers: OpenRouter API (for multi-provider synthesis)
# ---------------------------------------------------------------------------

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Synthesis model list: Gemini (direct) + OpenRouter providers
_SYNTHESIS_MODELS = [
    # Gemini (direct API)
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-3-pro-preview",
    "gemini-3.1-pro-preview",
    "gemini-3-flash-preview",
    "gemini-2.0-flash",
    # OpenRouter — Anthropic
    "anthropic/claude-sonnet-4",
    "anthropic/claude-opus-4",
    # OpenRouter — OpenAI
    "openai/gpt-4o",
    "openai/o3",
    # OpenRouter — Google via OpenRouter
    "google/gemini-2.5-pro",
    "google/gemini-2.5-flash",
    # OpenRouter — Open models
    "meta-llama/llama-3.3-70b-instruct",
    "deepseek/deepseek-r1",
]


def _call_openrouter(api_key, model, prompt, system_instruction=None,
                     temperature=0.7, max_tokens=16384):
    """Call OpenRouter chat completions API.

    Returns (text, usage_meta) tuple matching _generate_content's signature.
    """
    # Auto-prefix bare Gemini model names for OpenRouter
    if "/" not in model and model.startswith("gemini-"):
        model = f"google/{model}"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/comfyui",
        "X-Title": "NV Gemini Course Synthesizer",
    }

    messages = []
    if system_instruction and system_instruction.strip():
        messages.append({"role": "system", "content": system_instruction.strip()})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    print(f"[OpenRouter] Calling {model} (max_tokens={max_tokens}, temp={temperature})")

    # Retry loop (same pattern as Gemini)
    last_error = None
    for attempt in range(_MAX_RETRIES + 1):
        if attempt > 0:
            delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
            print(f"[OpenRouter] Retry {attempt}/{_MAX_RETRIES} after {delay}s...")
            time.sleep(delay)

        req_start = time.time()
        try:
            resp = requests.post(_OPENROUTER_URL, headers=headers, json=payload, timeout=600)
        except requests.exceptions.RequestException as e:
            last_error = e
            print(f"[OpenRouter] Network error: {e}")
            continue

        req_time = time.time() - req_start

        if resp.status_code == 429 or resp.status_code >= 500:
            last_error = RuntimeError(f"OpenRouter error {resp.status_code}: {resp.text[:300]}")
            print(f"[OpenRouter] Transient error {resp.status_code} ({req_time:.1f}s)")
            continue

        if resp.status_code != 200:
            raise RuntimeError(f"OpenRouter error {resp.status_code}: {resp.text[:500]}")

        result = resp.json()

        # Extract usage — OpenRouter includes cost directly
        usage = result.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        response_tokens = usage.get("completion_tokens", 0)
        cost = usage.get("cost", 0.0) or 0.0
        if isinstance(cost, str):
            try:
                cost = float(cost)
            except ValueError:
                cost = 0.0
        cost = round(cost, 4)

        print(f"[OpenRouter] Response in {req_time:.1f}s — "
              f"{prompt_tokens or '?'} prompt tokens, "
              f"{response_tokens or '?'} response tokens, "
              f"cost ${cost:.4f}")

        choices = result.get("choices", [])
        if not choices:
            raise RuntimeError("OpenRouter returned no choices")

        text = choices[0].get("message", {}).get("content", "")
        if not text or not text.strip():
            raise RuntimeError("OpenRouter returned empty response")

        return text.strip(), {
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens,
            "thinking_tokens": 0,
            "estimated_cost_usd": cost,
            "response_time_s": round(req_time, 1),
        }

    raise RuntimeError(f"OpenRouter failed after {_MAX_RETRIES + 1} attempts: {last_error}")


def _is_openrouter_model(model):
    """Check if a model name is for OpenRouter (contains a / provider prefix)."""
    return "/" in model


# ---------------------------------------------------------------------------
# Helpers: OpenRouter video path — cache dir, optimization, schema conversion
# ---------------------------------------------------------------------------

# OR video path uses longer timeouts: provider decodes base64 before token #1,
# and must be large enough for network upload of a 30MB+ body at modest bandwidth.
_OR_VIDEO_READ_TIMEOUT = 300
_OR_VIDEO_RETRY_BASE_DELAY = 15  # OR video rate-limits heavier; start backoff higher

# Auto-optimize triggers (see _should_optimize)
_OR_AUTO_OPTIMIZE_BYTES = 48 * 1024 * 1024   # est. base64 >= 48MB → optimize
_OR_AUTO_OPTIMIZE_HEIGHT = 1080              # source height > 1080 → optimize
_OR_AUTO_OPTIMIZE_DURATION = 20 * 60         # source > 20min → optimize

# Absolute hard-fail ceiling on OR base64 path (avoids doomed Cloudflare POSTs)
_OR_BASE64_HARD_CEILING_BYTES = 64 * 1024 * 1024  # 64MB raw (~88MB base64)


def _get_nv_video_cache_dir():
    """Return ComfyUI/temp/nv_video_cache/, creating if missing."""
    base = folder_paths.get_temp_directory()
    cache = os.path.join(base, "nv_video_cache")
    os.makedirs(cache, exist_ok=True)
    return cache


def _estimate_base64_bytes(raw_bytes):
    """Base64 expansion is ~4/3 + padding. Ignore JSON framing overhead here."""
    return int(raw_bytes * 1.37)


def _ffprobe_streams(file_path):
    """Return (width, height, duration_s, has_audio). Best-effort; None fields on failure."""
    import subprocess
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", "-show_format", str(file_path)],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0:
            return (None, None, None, False)
        data = json.loads(result.stdout or "{}")
        width = height = None
        has_audio = False
        for s in data.get("streams", []):
            if s.get("codec_type") == "video" and width is None:
                width = s.get("width")
                height = s.get("height")
            elif s.get("codec_type") == "audio":
                has_audio = True
        dur = data.get("format", {}).get("duration")
        duration = float(dur) if dur else None
        return (width, height, duration, has_audio)
    except Exception:
        return (None, None, None, False)


def _should_optimize(mode, file_size, width, height, duration):
    """Decide whether to transcode on the OR path.

    mode: "auto" | "always" | "never"
    Returns (should_optimize: bool, reason: str).
    """
    if mode == "never":
        return (False, "mode=never")
    if mode == "always":
        return (True, "mode=always")
    # auto
    est = _estimate_base64_bytes(file_size)
    if est >= _OR_AUTO_OPTIMIZE_BYTES:
        return (True, f"est_base64 {est / 1024 / 1024:.0f}MB >= 48MB")
    if height and height > _OR_AUTO_OPTIMIZE_HEIGHT:
        return (True, f"resolution {width}x{height} > 1080p")
    if duration and duration > _OR_AUTO_OPTIMIZE_DURATION:
        return (True, f"duration {duration / 60:.1f}min > 20min")
    return (False, f"under thresholds ({file_size / 1024 / 1024:.1f}MB, {height}p, {duration}s)")


# Transcode profile tuned for screen-recording tutorials with voiceover:
#  - 1fps: VLMs sample at ~1fps anyway, near-static screen content compresses 90%+
#  - GOP=1: every frame a keyframe → no P-frame decode glitches on provider side
#  - CRF 26: preserves small IDE/terminal font edges (text legibility floor for OCR)
#  - max_height 1080: keeps full-screen code readable (720p smears tabs/line numbers)
#  - 80 kbps AAC stereo: voiceover clarity without bloat (speech is the lesson)
_OR_TRANSCODE_PROFILE = {
    "fps": 1,
    "gop": 1,
    "max_height": 1080,
    "crf": 26,
    "preset": "medium",
    "audio_kbps": 80,
}


def _transcode_cache_key(src_path, profile):
    """Hash(src_abspath + mtime + profile) → cache filename."""
    import hashlib
    src_abspath = os.path.abspath(src_path)
    mtime = int(os.path.getmtime(src_path))
    profile_str = json.dumps(profile, sort_keys=True)
    raw = f"{src_abspath}|{mtime}|{profile_str}".encode("utf-8")
    return hashlib.md5(raw).hexdigest()


def _optimize_video_for_openrouter(src_path, mode="auto"):
    """Transcode (or pass-through) a video for the OpenRouter base64 path.

    Returns (path_to_use, info_dict).
    path_to_use is either the original path (no-op) or a cached optimized mp4.
    info_dict contains: optimized, reason, src_bytes, out_bytes, duration_before, duration_after.
    """
    import subprocess

    src_bytes = os.path.getsize(src_path)
    width, height, duration, has_audio = _ffprobe_streams(src_path)
    should, reason = _should_optimize(mode, src_bytes, width, height, duration)

    info = {
        "optimized": False,
        "reason": reason,
        "src_bytes": src_bytes,
        "out_bytes": src_bytes,
        "src_resolution": f"{width}x{height}" if width else "unknown",
        "duration_s": duration,
        "had_audio": has_audio,
    }

    if not should:
        return (src_path, info)

    cache_dir = _get_nv_video_cache_dir()
    cache_key = _transcode_cache_key(src_path, _OR_TRANSCODE_PROFILE)
    out_path = os.path.join(cache_dir, f"{cache_key}.mp4")

    if os.path.isfile(out_path):
        out_bytes = os.path.getsize(out_path)
        info.update({"optimized": True, "out_bytes": out_bytes, "reason": f"{reason} (cache hit)"})
        print(f"[NV_GeminiVideoExtractor] Cache hit: {Path(out_path).name} "
              f"({src_bytes / 1024 / 1024:.1f}MB → {out_bytes / 1024 / 1024:.1f}MB)")
        return (out_path, info)

    # Build ffmpeg command — FPS reduction + keyframe-every-frame + CRF
    p = _OR_TRANSCODE_PROFILE
    # Downscale only if source exceeds max_height; preserve aspect ratio; even dims
    vf = f"scale='min(iw,trunc(iw*{p['max_height']}/ih/2)*2)':'min(ih,{p['max_height']})':flags=bicubic"
    cmd = [
        "ffmpeg", "-y", "-i", str(src_path),
        "-vf", vf,
        "-r", str(p["fps"]),
        "-g", str(p["gop"]),
        "-c:v", "libx264",
        "-preset", p["preset"],
        "-crf", str(p["crf"]),
        "-pix_fmt", "yuv420p",
        "-profile:v", "high",
    ]
    if has_audio:
        cmd += ["-c:a", "aac", "-b:a", f"{p['audio_kbps']}k", "-ac", "2"]
    else:
        cmd += ["-an"]
    cmd += ["-movflags", "+faststart", "-loglevel", "error", str(out_path)]

    # Give ffmpeg generous runtime budget: ~1s per source second, min 60s, max 1200s
    budget = max(60, min(1200, int((duration or 60) * 1.5)))
    print(f"[NV_GeminiVideoExtractor] Optimizing for OpenRouter ({reason})...")
    print(f"  Profile: {p['fps']}fps, keyframe-every-frame, <={p['max_height']}p, CRF {p['crf']}")

    t0 = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=budget)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"ffmpeg transcode timed out after {budget}s on {Path(src_path).name}")

    if result.returncode != 0:
        # Surface ffmpeg stderr so failures are actionable
        raise RuntimeError(f"ffmpeg transcode failed: {result.stderr.strip()[:500]}")

    out_bytes = os.path.getsize(out_path)
    _, _, out_duration, _ = _ffprobe_streams(out_path)

    # Guard against >2s drift between source and transcode duration
    if duration and out_duration and abs(duration - out_duration) > 2.0:
        print(f"[NV_GeminiVideoExtractor] Warning: duration drift "
              f"({duration:.1f}s → {out_duration:.1f}s) — timestamps may be off")

    info.update({
        "optimized": True,
        "out_bytes": out_bytes,
        "duration_after": out_duration,
        "reason": reason,
    })
    print(f"[NV_GeminiVideoExtractor] Optimized in {time.time() - t0:.1f}s: "
          f"{src_bytes / 1024 / 1024:.1f}MB → {out_bytes / 1024 / 1024:.1f}MB "
          f"({(1 - out_bytes / src_bytes) * 100:.0f}% smaller)")
    return (out_path, info)


def _gemini_schema_to_openapi(schema):
    """Convert Gemini's UPPERCASE type enums to OpenAPI lowercase for OpenRouter json_schema.

    Gemini uses {"type": "OBJECT"} but OpenAI/OR structured-output expects {"type": "object"}.
    Also drops Gemini-specific "nullable" (translate to ["string","null"] list).
    """
    if not isinstance(schema, dict):
        return schema

    out = {}
    for k, v in schema.items():
        if k == "type" and isinstance(v, str):
            out[k] = v.lower()
        elif k == "properties" and isinstance(v, dict):
            out[k] = {pk: _gemini_schema_to_openapi(pv) for pk, pv in v.items()}
        elif k == "items":
            out[k] = _gemini_schema_to_openapi(v)
        elif k == "nullable":
            # Translate nullable:true by promoting type to [type,"null"] on parent
            continue
        else:
            out[k] = v

    # Handle nullable: if original had nullable:true, make type an array
    if isinstance(schema, dict) and schema.get("nullable") and "type" in out:
        if isinstance(out["type"], str):
            out["type"] = [out["type"], "null"]
    return out


def _b64encode_file(path):
    """Read a file and return its base64-encoded string (no data URL prefix)."""
    import base64
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def _generate_content_openrouter_video(api_key, model, prompt, video_path=None, video_url=None,
                                        mime="video/mp4", system_instruction=None,
                                        temperature=1.0, max_tokens=16384,
                                        response_schema=None):
    """Call OpenRouter chat completions with a video content part.

    Exactly one of video_path (→ base64 data URL) or video_url (→ public URL) must be set.

    Mirrors _generate_content's return shape: (text, usage_meta) where usage_meta has
    prompt_tokens, response_tokens, thinking_tokens, estimated_cost_usd, response_time_s.
    """
    import random

    if not (bool(video_path) ^ bool(video_url)):
        raise RuntimeError("_generate_content_openrouter_video: pass exactly one of video_path or video_url")

    # Build video content part
    if video_url:
        video_part = {"type": "video_url", "video_url": {"url": video_url}}
        print(f"[OpenRouter] Video via URL: {video_url}")
    else:
        raw_bytes = os.path.getsize(video_path)
        if raw_bytes > _OR_BASE64_HARD_CEILING_BYTES:
            raise RuntimeError(
                f"Video is {raw_bytes / 1024 / 1024:.0f}MB — exceeds OpenRouter base64 ceiling "
                f"({_OR_BASE64_HARD_CEILING_BYTES / 1024 / 1024:.0f}MB). "
                f"Enable optimize=auto, set optimize=always, or use the video_url input."
            )
        b64 = _b64encode_file(video_path)
        video_part = {"type": "video_url", "video_url": {"url": f"data:{mime};base64,{b64}"}}
        print(f"[OpenRouter] Video via base64: {raw_bytes / 1024 / 1024:.1f}MB → ~{len(b64) / 1024 / 1024:.1f}MB encoded")

    messages = []
    if system_instruction and system_instruction.strip():
        messages.append({"role": "system", "content": system_instruction.strip()})
    messages.append({"role": "user", "content": [
        {"type": "text", "text": prompt},
        video_part,
    ]})

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    # Prefer json_schema structured output when a schema is provided; fall back on retry.
    if response_schema:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "video_extraction",
                "strict": True,
                "schema": _gemini_schema_to_openapi(response_schema),
            },
        }
    else:
        payload["response_format"] = {"type": "json_object"}

    # Provider routing: pin Vertex/Google for google/gemini-* (Vertex handles base64 video
    # natively; AI Studio demands YouTube URLs). require_parameters prevents silent drop of
    # response_format. data_collection=deny opts out of training-data logging.
    provider_cfg = {
        "require_parameters": True,
        "data_collection": "deny",
    }
    if model.startswith("google/"):
        provider_cfg["order"] = ["Vertex", "Google"]
    payload["provider"] = provider_cfg

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/comfyui",
        "X-Title": "NV Gemini Video Extractor",
    }

    print(f"[OpenRouter] Calling {model} (max_tokens={max_tokens}, temp={temperature}, "
          f"schema={'json_schema' if response_schema else 'json_object'})")

    last_error = None
    for attempt in range(_MAX_RETRIES + 1):
        if attempt > 0:
            # 15s → 30s → 60s with ±20% jitter
            base = _OR_VIDEO_RETRY_BASE_DELAY * (2 ** (attempt - 1))
            delay = base * (0.8 + random.random() * 0.4)
            print(f"[OpenRouter] Retry {attempt}/{_MAX_RETRIES} after {delay:.1f}s...")
            time.sleep(delay)

            # If json_schema was rejected on first attempt, downgrade to json_object
            if response_schema and payload.get("response_format", {}).get("type") == "json_schema" \
                    and last_error and "response_format" in str(last_error).lower():
                print("[OpenRouter] Downgrading response_format: json_schema → json_object")
                payload["response_format"] = {"type": "json_object"}

        req_start = time.time()
        try:
            resp = requests.post(
                _OPENROUTER_URL, headers=headers, json=payload, timeout=_OR_VIDEO_READ_TIMEOUT
            )
        except requests.exceptions.RequestException as e:
            last_error = e
            print(f"[OpenRouter] Network error: {e}")
            continue

        req_time = time.time() - req_start

        if resp.status_code == 429 or resp.status_code >= 500:
            last_error = RuntimeError(f"OpenRouter error {resp.status_code}: {resp.text[:300]}")
            print(f"[OpenRouter] Transient error {resp.status_code} ({req_time:.1f}s)")
            continue

        if resp.status_code != 200:
            raise RuntimeError(f"OpenRouter error {resp.status_code}: {resp.text[:500]}")

        result = resp.json()
        usage = result.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        response_tokens = usage.get("completion_tokens", 0)
        cost = usage.get("cost", 0.0) or 0.0
        if isinstance(cost, str):
            try:
                cost = float(cost)
            except ValueError:
                cost = 0.0
        cost = round(cost, 4)

        choices = result.get("choices", [])
        if not choices:
            raise RuntimeError("OpenRouter returned no choices")

        finish_reason = choices[0].get("finish_reason", "")
        if finish_reason == "length":
            print(f"[OpenRouter] Warning: finish_reason=length — output truncated at max_tokens ({max_tokens}). "
                  f"JSON will likely fail to parse. Increase max_tokens.")

        text = choices[0].get("message", {}).get("content", "")
        if not text or not text.strip():
            raise RuntimeError("OpenRouter returned empty response")

        print(f"[OpenRouter] Response in {req_time:.1f}s — "
              f"{prompt_tokens or '?'} prompt tokens, {response_tokens or '?'} response tokens, "
              f"cost ${cost:.4f}")

        return text.strip(), {
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens,
            "thinking_tokens": 0,
            "estimated_cost_usd": cost,
            "response_time_s": round(req_time, 1),
        }

    raise RuntimeError(f"OpenRouter video call failed after {_MAX_RETRIES + 1} attempts: {last_error}")


# ---------------------------------------------------------------------------
# Helpers: Atomic file write
# ---------------------------------------------------------------------------

def _parse_extraction_json(raw_response):
    """Parse a raw model response to a dict. Falls back through markdown-code-block salvage."""
    try:
        return json.loads(raw_response)
    except json.JSONDecodeError as e:
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw_response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        print(f"[NV_GeminiVideoExtractor] Warning: not valid JSON: {e}")
        print(f"[NV_GeminiVideoExtractor] Raw (first 500): {raw_response[:500]}")
        return {"raw_response": raw_response, "parse_error": str(e)}


def _atomic_write_json(path, data):
    """Write JSON to a file atomically (write .tmp then rename)."""
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    # os.replace is atomic on the same filesystem
    os.replace(tmp_path, path)


def _atomic_write_text(path, text):
    """Write text to a file atomically (write .tmp then rename)."""
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp_path, path)


# ---------------------------------------------------------------------------
# Node 1: Video Extractor
# ---------------------------------------------------------------------------

class NV_GeminiVideoExtractor:
    """Extract structured learning content from a tutorial video using the Gemini API."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {
                    "default": "",
                    "tooltip": "Absolute path to a video file (MP4, MOV, WebM, AVI, etc.). "
                               "Direct Gemini: up to 2GB / ~3hr via Files API. "
                               "OpenRouter: size depends on optimize mode (see optimize tooltip).",
                }),
            },
            "optional": {
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "Optional. Leave blank to use GEMINI_API_KEY / GOOGLE_API_KEY "
                               "(direct-Gemini models) or OPENROUTER_API_KEY (provider/model dropdown entries).",
                }),
                "model": ([
                    # Direct Gemini (Files API, large videos)
                    "gemini-2.5-pro",
                    "gemini-2.5-flash",
                    "gemini-3-pro-preview",
                    "gemini-3.1-pro-preview",
                    "gemini-3-flash-preview",
                    "gemini-2.0-flash",
                    # OpenRouter (base64 inline, smaller videos — use optimize=auto)
                    "google/gemini-2.5-pro",
                    "google/gemini-2.5-flash",
                    "google/gemini-3-pro-preview",
                    "qwen/qwen3-vl-plus",
                ], {
                    "default": "gemini-2.5-pro",
                    "tooltip": "Bare 'gemini-*' = direct Gemini Files API (best for long videos, rate-limited per your Google project). "
                               "'provider/model' = OpenRouter (rate-limit escape valve, but base64-inline transport — auto-optimization helps). "
                               "qwen/qwen3-vl-plus: note Qwen VL often ignores audio — not ideal for voiceover-carried lessons.",
                }),
                "media_resolution": (["default", "low"], {
                    "default": "default",
                    "tooltip": "Frame token density (Gemini direct only, ignored on OpenRouter). "
                               "'default' (~300 tok/s) for code/slides/text on screen. "
                               "'low' (~100 tok/s) for talking-head content only.",
                }),
                "optimize": (["auto", "always", "never"], {
                    "default": "auto",
                    "tooltip": "Video optimization before sending (OpenRouter path only — skipped for direct Gemini). "
                               "'auto' transcodes if base64 size > 48MB, resolution > 1080p, or duration > 20min. "
                               "Transcode profile is tuned for screen-recording tutorials with voiceover: "
                               "1fps (VLMs sample at ~1fps anyway), every frame a keyframe, CRF 26 (preserves code/terminal font edges), "
                               "80kbps AAC audio (voiceover clarity). Cached to ComfyUI/temp/nv_video_cache/. "
                               "'always' = always transcode. 'never' = send original (may hit gateway size limits).",
                }),
                "video_url": ("STRING", {
                    "default": "",
                    "tooltip": "Optional public URL to the video (OpenRouter path only). If set, bypasses base64 encoding entirely. "
                               "For google/gemini-* this must be a YouTube URL if OpenRouter routes to AI Studio. "
                               "Leave empty to use video_path instead.",
                }),
                "skip_existing": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If True and an extraction JSON already exists for this video filename, "
                               "return the cached result without calling the API. Saves cost on re-queues.",
                }),
                "output_dir": ("STRING", {
                    "default": "",
                    "tooltip": "Directory to save extraction JSON. "
                               "Empty = ComfyUI output/gemini_course/extractions/",
                }),
                "prompt_override": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Custom extraction prompt. Leave empty to use the built-in "
                               "structured extraction prompt.",
                }),
                "max_tokens": ("INT", {
                    "default": 16384,
                    "min": 1024,
                    "max": 65536,
                    "step": 1024,
                    "tooltip": "Maximum output tokens. Dense 20-min videos with many segments "
                               "can need 8K-12K tokens; 16384 provides headroom. "
                               "If you see 'finish_reason=length' in logs, bump this.",
                }),
                "trigger": ("*", {
                    "tooltip": "Optional trigger input for sequencing multiple extractions.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("extraction_json", "json_file_path", "video_filename")
    FUNCTION = "extract"
    CATEGORY = "NV_Utils/Course"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Upload a tutorial video to the Gemini Files API and extract structured learning "
        "content as JSON. Returns segments with timestamps, learning objectives, skills, "
        "tools, and uncertainties. Saves extraction JSON to disk for later synthesis with "
        "NV Gemini Course Synthesizer."
    )

    def extract(self, video_path, api_key="", model="gemini-2.5-pro",
                media_resolution="default", optimize="auto", video_url="",
                skip_existing=True, output_dir="", prompt_override="",
                max_tokens=16384, trigger=None):
        use_openrouter = _is_openrouter_model(model)
        api_key = resolve_api_key(api_key, provider="openrouter" if use_openrouter else "gemini")

        video_url = (video_url or "").strip()
        video_path = (video_path or "").strip()

        # Validate inputs — URL path requires OpenRouter and no file
        if video_url:
            if not use_openrouter:
                raise RuntimeError("video_url is only supported on the OpenRouter path. Pick a 'provider/model' entry.")
            video_name = Path(video_url).stem or "url_video"
            mime = "video/mp4"  # best-guess; OpenRouter infers from URL
        else:
            if not video_path:
                raise RuntimeError("video_path is empty (and no video_url provided)")
            if not os.path.isfile(video_path):
                raise RuntimeError(f"Video file not found: {video_path}")
            ext = Path(video_path).suffix.lower()
            if ext not in _SUPPORTED_VIDEO_EXTS:
                raise RuntimeError(
                    f"Unsupported video format: {ext}. "
                    f"Supported: {', '.join(sorted(_SUPPORTED_VIDEO_EXTS))}"
                )
            video_name = Path(video_path).stem
            mime = _get_mime_type(video_path)

        # Resolve output directory
        if not output_dir or not output_dir.strip():
            output_dir = os.path.join(
                folder_paths.get_output_directory(), "gemini_course", "extractions"
            )
        os.makedirs(output_dir, exist_ok=True)

        # Cache key: include model so direct-Gemini and OR extractions don't collide
        json_path = os.path.join(output_dir, f"{video_name}.json")
        if skip_existing and os.path.isfile(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                src = cached.get("_source", {})
                cached_path_ref = src.get("video_path") or src.get("video_url")
                expected_ref = video_url or video_path
                config_match = (
                    src.get("model") == model
                    and src.get("media_resolution") == media_resolution
                    and cached_path_ref == expected_ref
                )
                if config_match:
                    print(f"[NV_GeminiVideoExtractor] Cached extraction found: {json_path}")
                    extraction_str = json.dumps(cached, indent=2, ensure_ascii=False)
                    return (extraction_str, json_path, video_name)
                else:
                    print(f"[NV_GeminiVideoExtractor] Cache stale (config changed), re-extracting")
            except (json.JSONDecodeError, ValueError, OSError):
                print(f"[NV_GeminiVideoExtractor] Cache corrupt, re-extracting")

        # Build prompt (shared across both paths). Append explicit full-coverage instruction
        # because OR-routed providers have been observed to silently summarize only the first
        # portion of long videos without throwing.
        base_prompt = (prompt_override.strip()
                       if prompt_override and prompt_override.strip()
                       else _EXTRACTION_PROMPT)
        prompt = base_prompt + (
            "\n\nCOVERAGE REQUIREMENT: Analyze the ENTIRE video from 00:00 through the final second. "
            "If any portion was not fully processed, add an entry to 'uncertainties' naming the timestamp range."
        )

        print(f"\n{'=' * 60}")
        print(f"[NV_GeminiVideoExtractor] Processing: {video_name}")
        print(f"  Model: {model} ({'OpenRouter' if use_openrouter else 'Gemini direct'})")
        print(f"  Max tokens: {max_tokens}")
        print(f"{'=' * 60}")

        if use_openrouter:
            extraction, usage_meta, extra_source = self._extract_via_openrouter(
                video_path=video_path if not video_url else None,
                video_url=video_url or None,
                mime=mime,
                model=model,
                api_key=api_key,
                prompt=prompt,
                max_tokens=max_tokens,
                optimize=optimize,
            )
        else:
            extraction, usage_meta, extra_source = self._extract_via_gemini(
                video_path=video_path,
                video_name=video_name,
                mime=mime,
                model=model,
                api_key=api_key,
                prompt=prompt,
                max_tokens=max_tokens,
                media_resolution=media_resolution,
            )

        # Source metadata for traceability
        source = {
            "video_file": video_name,
            "model": model,
            "media_resolution": media_resolution,
            "extracted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "prompt_tokens": usage_meta.get("prompt_tokens"),
            "response_tokens": usage_meta.get("response_tokens"),
            "estimated_cost_usd": usage_meta.get("estimated_cost_usd"),
            "response_time_s": usage_meta.get("response_time_s"),
            "transport": "openrouter" if use_openrouter else "gemini_files_api",
        }
        if video_path:
            source["video_path"] = video_path
        if video_url:
            source["video_url"] = video_url
        source.update(extra_source)
        extraction["_source"] = source

        _atomic_write_json(json_path, extraction)

        extraction_str = json.dumps(extraction, indent=2, ensure_ascii=False)

        segments = extraction.get("segments", [])
        cost = usage_meta.get("estimated_cost_usd")
        cost_str = f" | Est. cost: ${cost:.4f}" if cost is not None else ""
        print(f"\n[NV_GeminiVideoExtractor] Extraction complete!")
        print(f"  Segments: {len(segments)} | Type: {extraction.get('content_type', 'N/A')}")
        print(f"  Topic: {extraction.get('overall_topic', 'N/A')}")
        print(f"  Level: {extraction.get('target_level', 'N/A')}{cost_str}")
        print(f"  Saved: {json_path}")
        print(f"{'=' * 60}\n")

        return (extraction_str, json_path, video_name)

    # -- Transport branches ---------------------------------------------------

    def _extract_via_gemini(self, video_path, video_name, mime, model, api_key,
                             prompt, max_tokens, media_resolution):
        """Direct Gemini Files API path — unchanged from original behavior."""
        file_info = _upload_file(video_path, api_key, display_name=video_name)
        file_name = file_info.get("name", "")
        file_uri = file_info.get("uri", "")

        try:
            if file_info.get("state") != "ACTIVE":
                file_info = _poll_file_active(file_name, api_key, timeout=300)
                file_uri = file_info.get("uri", file_uri)

            contents = [
                {"file_data": {"file_uri": file_uri, "mime_type": mime}},
                {"text": prompt},
            ]
            raw_response, usage_meta = _generate_content(
                api_key=api_key, model=model, contents=contents,
                temperature=1.0, max_tokens=max_tokens,
                json_mode=True, response_schema=_EXTRACTION_SCHEMA,
                media_resolution=media_resolution,
            )
            extraction = _parse_extraction_json(raw_response)
            return (extraction, usage_meta, {})
        finally:
            if file_name:
                _delete_file(file_name, api_key)

    def _extract_via_openrouter(self, video_path, video_url, mime, model, api_key,
                                 prompt, max_tokens, optimize):
        """OpenRouter path — optional ffmpeg optimization, base64 OR URL transport."""
        extra_source = {"openrouter_optimize_mode": optimize}
        send_path = None

        if video_path:
            if model.startswith("qwen/"):
                print(f"[NV_GeminiVideoExtractor] NOTE: {model} often ignores audio — "
                      f"voiceover content may not be captured. Prefer google/gemini-* for "
                      f"voiceover-heavy tutorials.")
            send_path, opt_info = _optimize_video_for_openrouter(video_path, mode=optimize)
            extra_source["optimization"] = opt_info

        raw_response, usage_meta = _generate_content_openrouter_video(
            api_key=api_key, model=model, prompt=prompt,
            video_path=send_path, video_url=video_url,
            mime=mime, temperature=1.0, max_tokens=max_tokens,
            response_schema=_EXTRACTION_SCHEMA,
        )
        extraction = _parse_extraction_json(raw_response)
        return (extraction, usage_meta, extra_source)


# ---------------------------------------------------------------------------
# Node 2: Course Synthesizer
# ---------------------------------------------------------------------------

class NV_GeminiCourseSynthesizer:
    """Synthesize course plans from video extraction JSONs using one or more LLMs."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_folder": ("STRING", {
                    "default": "",
                    "tooltip": "Folder containing extraction JSON files from "
                               "NV Gemini Video Extractor. "
                               "Empty = ComfyUI output/gemini_course/extractions/",
                }),
            },
            "optional": {
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "Optional. Leave blank to use GEMINI_API_KEY / GOOGLE_API_KEY (Gemini) or OPENROUTER_API_KEY (OpenRouter) from environment or .env file.",
                }),
                "model": (_SYNTHESIS_MODELS, {
                    "default": "gemini-2.5-pro",
                    "tooltip": "Primary model for synthesis. Bare names (gemini-*) use Gemini API. "
                               "Prefixed names (anthropic/claude-*, openai/gpt-*) use OpenRouter.",
                }),
                "additional_models": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Run synthesis with multiple models for comparison. "
                               "One model per line. Each produces its own course_plan_{model}.md. "
                               "Example:\nanthropic/claude-sonnet-4\nopenai/gpt-4o\ngemini-3.1-pro-preview",
                }),
                "course_title": ("STRING", {
                    "default": "",
                    "tooltip": "Course title override. Leave empty for auto-generated title.",
                }),
                "custom_instructions": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Additional synthesis instructions (e.g., 'target audience is "
                               "ML engineers', 'group by difficulty level', "
                               "'emphasize hands-on exercises').",
                }),
                "output_dir": ("STRING", {
                    "default": "",
                    "tooltip": "Directory for output markdown files. "
                               "Empty = output/gemini_course/",
                }),
                "max_tokens": ("INT", {
                    "default": 16384,
                    "min": 2048,
                    "max": 65536,
                    "step": 1024,
                    "tooltip": "Maximum output tokens per model.",
                }),
                "trigger": ("*", {
                    "tooltip": "Optional trigger input for sequencing after extraction.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("course_plan_md", "output_dir", "summary_json")
    FUNCTION = "synthesize"
    CATEGORY = "NV_Utils/Course"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Combine video extraction JSONs into course plans using one or more LLMs. "
        "Supports Gemini (direct) and any OpenRouter model (Claude, GPT, LLaMA, etc.). "
        "When multiple models are specified, each produces a separate course plan for comparison."
    )

    def _call_model(self, model_name, api_key, prompt, sys_instr, max_tokens):
        """Route a synthesis call to the right provider. Returns (text, usage_meta)."""
        if _is_openrouter_model(model_name):
            resolved_key = resolve_api_key(api_key, "openrouter")
            return _call_openrouter(
                api_key=resolved_key, model=model_name, prompt=prompt,
                system_instruction=sys_instr, temperature=0.7, max_tokens=max_tokens,
            )
        else:
            resolved_key = resolve_api_key(api_key, "gemini")
            return _generate_content(
                api_key=resolved_key, model=model_name,
                contents=[{"text": prompt}],
                system_instruction=sys_instr, temperature=0.7,
                max_tokens=max_tokens, json_mode=False,
            )

    def synthesize(self, json_folder, api_key="", model="gemini-2.5-pro",
                   additional_models="", course_title="", custom_instructions="",
                   output_dir="", max_tokens=16384, trigger=None):

        # Resolve json_folder default
        if not json_folder or not json_folder.strip():
            json_folder = os.path.join(
                folder_paths.get_output_directory(), "gemini_course", "extractions"
            )

        if not os.path.isdir(json_folder):
            raise RuntimeError(f"JSON folder not found: {json_folder}")

        # Load all extraction JSONs
        json_files = sorted(Path(json_folder).glob("*.json"))
        if not json_files:
            raise RuntimeError(f"No JSON files found in: {json_folder}")

        print(f"\n{'=' * 60}")
        print(f"[NV_GeminiCourseSynthesizer] Loading {len(json_files)} extraction(s)")

        extractions = []
        for jf in json_files:
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if "_source" not in data:
                    data["_source"] = {"video_file": jf.stem}
                extractions.append(data)
                topic = data.get("overall_topic", "N/A")
                segments = len(data.get("segments", []))
                print(f"  [{jf.name}] {topic} ({segments} segments)")
            except Exception as e:
                print(f"  [{jf.name}] Warning: failed to load — {e}")

        if not extractions:
            raise RuntimeError("No valid extraction JSONs could be loaded")

        print(f"[NV_GeminiCourseSynthesizer] {len(extractions)} valid extractions loaded")
        print(f"{'=' * 60}")

        # Build synthesis prompt
        extractions_text = json.dumps(extractions, indent=2, ensure_ascii=False)
        prompt = _SYNTHESIS_PROMPT.format(extractions=extractions_text)

        if course_title and course_title.strip():
            prompt += f"\n\nIMPORTANT: The course title must be: {course_title.strip()}"
        if custom_instructions and custom_instructions.strip():
            prompt += f"\n\nADDITIONAL INSTRUCTIONS:\n{custom_instructions.strip()}"

        sys_instr = (
            "You are an experienced curriculum designer creating a structured course "
            "plan from video analysis data. Output clean, well-organized Markdown."
        )

        # Resolve output directory
        if not output_dir or not output_dir.strip():
            output_dir = os.path.join(
                folder_paths.get_output_directory(), "gemini_course"
            )
        os.makedirs(output_dir, exist_ok=True)

        # Build model list: primary + additional
        all_models = [model]
        if additional_models and additional_models.strip():
            for line in additional_models.strip().splitlines():
                m = line.strip().rstrip(",")
                if m and m not in all_models:
                    all_models.append(m)

        total_models = len(all_models)
        print(f"\n[NV_GeminiCourseSynthesizer] Running synthesis with {total_models} model(s): "
              f"{', '.join(all_models)}")

        # Run each model
        results = []
        primary_plan = ""
        total_cost = 0.0

        for idx, m in enumerate(all_models, 1):
            # Filename: course_plan.md for single model, course_plan_{slug}.md for multi
            slug = m.replace("/", "_").replace(".", "-")
            if total_models == 1:
                md_path = os.path.join(output_dir, "course_plan.md")
            else:
                md_path = os.path.join(output_dir, f"course_plan_{slug}.md")

            print(f"\n  [{idx}/{total_models}] {m}")
            try:
                plan_text, usage_meta = self._call_model(m, api_key, prompt, sys_instr, max_tokens)

                _atomic_write_text(md_path, plan_text)

                cost = usage_meta.get("estimated_cost_usd") or 0
                total_cost += cost
                lines = plan_text.count("\n")

                print(f"  [{idx}/{total_models}] {m} — {lines} lines, ${cost:.4f}")

                results.append({
                    "model": m,
                    "status": "ok",
                    "output_path": md_path,
                    "lines": lines,
                    "cost_usd": cost,
                    "prompt_tokens": usage_meta.get("prompt_tokens"),
                    "response_tokens": usage_meta.get("response_tokens"),
                })

                if idx == 1:
                    primary_plan = plan_text

            except Exception as e:
                print(f"  [{idx}/{total_models}] {m} — FAILED: {e}")
                results.append({
                    "model": m,
                    "status": "failed",
                    "error": str(e),
                })

        # Summary
        succeeded = sum(1 for r in results if r["status"] == "ok")
        failed = sum(1 for r in results if r["status"] == "failed")

        summary = {
            "models_run": total_models,
            "succeeded": succeeded,
            "failed": failed,
            "total_cost_usd": round(total_cost, 4),
            "extractions_used": len(extractions),
            "output_dir": output_dir,
            "results": results,
        }
        summary_str = json.dumps(summary, indent=2, ensure_ascii=False)

        print(f"\n{'=' * 60}")
        print(f"[NV_GeminiCourseSynthesizer] Complete!")
        print(f"  Models: {succeeded} succeeded, {failed} failed | Total cost: ${total_cost:.4f}")
        for r in results:
            status = "ok" if r["status"] == "ok" else "FAILED"
            print(f"    {r['model']}: {status}" +
                  (f" — {r.get('lines', 0)} lines, ${r.get('cost_usd', 0):.4f}" if status == "ok" else ""))
        print(f"  Output: {output_dir}")
        print(f"{'=' * 60}\n")

        return (primary_plan, output_dir, summary_str)


# ---------------------------------------------------------------------------
# Node 3: Batch Extractor
# ---------------------------------------------------------------------------

class NV_GeminiBatchExtractor:
    """Extract structured learning content from all videos in a folder."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_folder": ("STRING", {
                    "default": "",
                    "tooltip": "Folder containing video files (MP4, MOV, WebM, etc.). "
                               "All supported video formats will be processed.",
                }),
            },
            "optional": {
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "Optional. Leave blank to use GEMINI_API_KEY / GOOGLE_API_KEY "
                               "(direct-Gemini models) or OPENROUTER_API_KEY (provider/model entries).",
                }),
                "model": ([
                    "gemini-2.5-pro",
                    "gemini-2.5-flash",
                    "gemini-3-pro-preview",
                    "gemini-3.1-pro-preview",
                    "gemini-3-flash-preview",
                    "gemini-2.0-flash",
                    "google/gemini-2.5-pro",
                    "google/gemini-2.5-flash",
                    "google/gemini-3-pro-preview",
                    "qwen/qwen3-vl-plus",
                ], {
                    "default": "gemini-2.5-pro",
                    "tooltip": "Bare 'gemini-*' = direct Gemini Files API (large videos, your project's quota). "
                               "'provider/model' = OpenRouter (base64-inline, auto-optimized).",
                }),
                "media_resolution": (["default", "low"], {
                    "default": "default",
                    "tooltip": "Frame token density (Gemini direct only). 'default' for code/slides, "
                               "'low' for talking-head content.",
                }),
                "optimize": (["auto", "always", "never"], {
                    "default": "auto",
                    "tooltip": "Video optimization for OpenRouter path (skipped for direct Gemini). "
                               "See NV Gemini Video Extractor tooltip for profile details.",
                }),
                "dry_run": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If True, list all videos with durations and estimated cost — "
                               "no API calls made. Use to preview before committing.",
                }),
                "skip_existing": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Skip videos that already have an extraction JSON on disk.",
                }),
                "output_dir": ("STRING", {
                    "default": "",
                    "tooltip": "Directory to save extraction JSONs. "
                               "Empty = ComfyUI output/gemini_course/extractions/",
                }),
                "max_tokens": ("INT", {
                    "default": 16384,
                    "min": 1024,
                    "max": 65536,
                    "step": 1024,
                    "tooltip": "Maximum output tokens per extraction.",
                }),
                "trigger": ("*", {
                    "tooltip": "Optional trigger input for sequencing.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("summary_json", "output_folder", "videos_processed")
    FUNCTION = "extract_batch"
    CATEGORY = "NV_Utils/Course"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Process all videos in a folder through the Gemini Video Extractor. "
        "Skips already-extracted files when skip_existing is enabled. "
        "Wire output_folder directly to NV Gemini Course Synthesizer's json_folder input."
    )

    def extract_batch(self, video_folder, api_key="", model="gemini-2.5-pro",
                      media_resolution="default", optimize="auto", dry_run=False,
                      skip_existing=True, output_dir="", max_tokens=16384, trigger=None):
        use_openrouter = _is_openrouter_model(model)
        video_folder = video_folder.strip()
        if not video_folder:
            raise RuntimeError("video_folder is empty")
        if not os.path.isdir(video_folder):
            raise RuntimeError(f"Video folder not found: {video_folder}")

        # Find all supported video files
        video_files = []
        for f in sorted(Path(video_folder).iterdir()):
            if f.is_file() and f.suffix.lower() in _SUPPORTED_VIDEO_EXTS:
                video_files.append(f)

        if not video_files:
            raise RuntimeError(
                f"No supported video files found in: {video_folder}\n"
                f"Supported: {', '.join(sorted(_SUPPORTED_VIDEO_EXTS))}"
            )

        # Resolve output directory
        if not output_dir or not output_dir.strip():
            output_dir = os.path.join(
                folder_paths.get_output_directory(), "gemini_course", "extractions"
            )

        total = len(video_files)

        # --- DRY RUN MODE ---
        if dry_run:
            print(f"\n{'=' * 60}")
            print(f"[NV_GeminiBatchExtractor] DRY RUN — {total} videos found")
            print(f"  Model: {model} | Resolution: {media_resolution}")
            print(f"{'=' * 60}")

            total_duration = 0.0
            total_est_cost = 0.0
            plan = []

            for idx, vf in enumerate(video_files, 1):
                dur = _get_video_duration(str(vf))
                dur_str = _format_duration(dur)
                size_mb = vf.stat().st_size / (1024 * 1024)
                # OR path has no pricing table — cost is reported post-hoc by OR in usage.cost
                est_cost = None if use_openrouter else (_estimate_video_cost(dur, model, media_resolution) or 0)
                cached = os.path.isfile(os.path.join(output_dir, f"{vf.stem}.json"))
                status = "cached (skip)" if cached and skip_existing else "will extract"

                if dur:
                    total_duration += dur
                if not (cached and skip_existing) and est_cost is not None:
                    total_est_cost += est_cost

                cost_cell = "unknown" if est_cost is None else f"~${est_cost:.4f}"
                line = f"  [{idx:2d}/{total}] {vf.name:<45s} {dur_str:>7s}  {size_mb:6.1f} MB  {cost_cell:>10s}  {status}"
                print(line)
                plan.append({
                    "video": vf.name,
                    "duration_s": round(dur, 1) if dur else None,
                    "duration_str": dur_str,
                    "size_mb": round(size_mb, 1),
                    "estimated_cost_usd": est_cost,
                    "status": status,
                })

            print(f"\n  {'─' * 50}")
            print(f"  Total duration: {_format_duration(total_duration)}")
            cost_summary = "unknown (reported post-hoc by OpenRouter)" if use_openrouter else f"${total_est_cost:.4f}"
            print(f"  Estimated cost: {cost_summary} ({model})")
            to_extract = sum(1 for p in plan if p["status"] == "will extract")
            to_skip = sum(1 for p in plan if "cached" in p["status"])
            print(f"  Will extract: {to_extract} | Will skip: {to_skip}")
            print(f"{'=' * 60}\n")

            summary = {
                "dry_run": True,
                "total_videos": total,
                "will_extract": to_extract,
                "will_skip": to_skip,
                "total_duration_s": round(total_duration, 1),
                "estimated_total_cost_usd": round(total_est_cost, 4),
                "model": model,
                "media_resolution": media_resolution,
                "videos": plan,
            }
            return (json.dumps(summary, indent=2, ensure_ascii=False), output_dir, 0)

        # --- EXTRACTION MODE ---
        api_key = resolve_api_key(api_key, provider="openrouter" if use_openrouter else "gemini")
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"[NV_GeminiBatchExtractor] Found {total} videos in {video_folder}")
        print(f"  Model: {model} | Resolution: {media_resolution}")
        print(f"  Output: {output_dir}")
        print(f"{'=' * 60}")

        results = []
        total_cost = 0.0
        cached_count = 0
        extracted_count = 0
        failed_count = 0

        for idx, vf in enumerate(video_files, 1):
            video_name = vf.stem
            json_path = os.path.join(output_dir, f"{video_name}.json")

            # Check cache (validate config matches before reuse)
            if skip_existing and os.path.isfile(json_path):
                cache_valid = False
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        cached = json.load(f)
                    src = cached.get("_source", {})
                    cache_valid = (
                        src.get("model") == model
                        and src.get("media_resolution") == media_resolution
                        and src.get("video_path") == str(vf)
                    )
                except (json.JSONDecodeError, ValueError, OSError):
                    pass  # corrupt cache — re-extract
                if cache_valid:
                    print(f"  [{idx}/{total}] {vf.name} — cached")
                    cached_count += 1
                    results.append({
                        "video": vf.name, "status": "cached",
                        "json_path": json_path,
                    })
                    continue
                print(f"  [{idx}/{total}] {vf.name} — cache stale, re-extracting")

            # Process video — dispatch on model prefix
            print(f"\n  [{idx}/{total}] {vf.name} — extracting...")
            try:
                mime = _get_mime_type(str(vf))
                prompt = _EXTRACTION_PROMPT + (
                    "\n\nCOVERAGE REQUIREMENT: Analyze the ENTIRE video from 00:00 through the final second. "
                    "If any portion was not fully processed, add an entry to 'uncertainties' naming the timestamp range."
                )

                extra_source = {}
                if use_openrouter:
                    if model.startswith("qwen/"):
                        print(f"    NOTE: {model} often ignores audio — voiceover may not be captured.")
                    send_path, opt_info = _optimize_video_for_openrouter(str(vf), mode=optimize)
                    extra_source["openrouter_optimize_mode"] = optimize
                    extra_source["optimization"] = opt_info
                    raw_response, usage_meta = _generate_content_openrouter_video(
                        api_key=api_key, model=model, prompt=prompt,
                        video_path=send_path, mime=mime,
                        temperature=1.0, max_tokens=max_tokens,
                        response_schema=_EXTRACTION_SCHEMA,
                    )
                    extraction = _parse_extraction_json(raw_response)
                else:
                    file_info = _upload_file(str(vf), api_key, display_name=video_name)
                    file_name = file_info.get("name", "")
                    file_uri = file_info.get("uri", "")
                    try:
                        if file_info.get("state") != "ACTIVE":
                            file_info = _poll_file_active(file_name, api_key, timeout=300)
                            file_uri = file_info.get("uri", file_uri)
                        contents = [
                            {"file_data": {"file_uri": file_uri, "mime_type": mime}},
                            {"text": prompt},
                        ]
                        raw_response, usage_meta = _generate_content(
                            api_key=api_key, model=model, contents=contents,
                            temperature=1.0, max_tokens=max_tokens,
                            json_mode=True, response_schema=_EXTRACTION_SCHEMA,
                            media_resolution=media_resolution,
                        )
                        extraction = _parse_extraction_json(raw_response)
                    finally:
                        if file_name:
                            _delete_file(file_name, api_key)

                cost = usage_meta.get("estimated_cost_usd") or 0
                source = {
                    "video_file": vf.name,
                    "video_path": str(vf),
                    "model": model,
                    "media_resolution": media_resolution,
                    "extracted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "prompt_tokens": usage_meta.get("prompt_tokens"),
                    "response_tokens": usage_meta.get("response_tokens"),
                    "estimated_cost_usd": usage_meta.get("estimated_cost_usd"),
                    "response_time_s": usage_meta.get("response_time_s"),
                    "transport": "openrouter" if use_openrouter else "gemini_files_api",
                }
                source.update(extra_source)
                extraction["_source"] = source

                _atomic_write_json(json_path, extraction)

                total_cost += cost
                extracted_count += 1
                segs = len(extraction.get("segments", []))
                ctype = extraction.get("content_type", "?")
                print(f"  [{idx}/{total}] {vf.name} — {segs} segments, {ctype}, ${cost:.4f}")

                results.append({
                    "video": vf.name, "status": "extracted",
                    "json_path": json_path,
                    "segments": segs,
                    "content_type": ctype,
                    "cost_usd": cost,
                })

            except Exception as e:
                failed_count += 1
                print(f"  [{idx}/{total}] {vf.name} — FAILED: {e}")
                results.append({
                    "video": vf.name, "status": "failed", "error": str(e),
                })

        # Summary
        summary = {
            "total_videos": total,
            "extracted": extracted_count,
            "cached": cached_count,
            "failed": failed_count,
            "total_cost_usd": round(total_cost, 4),
            "model": model,
            "output_dir": output_dir,
            "results": results,
        }
        summary_str = json.dumps(summary, indent=2, ensure_ascii=False)

        print(f"\n{'=' * 60}")
        print(f"[NV_GeminiBatchExtractor] Complete!")
        print(f"  Extracted: {extracted_count} | Cached: {cached_count} | Failed: {failed_count}")
        print(f"  Total cost: ${total_cost:.4f}")
        print(f"  Output: {output_dir}")
        print(f"{'=' * 60}\n")

        return (summary_str, output_dir, extracted_count + cached_count)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "NV_GeminiVideoExtractor": NV_GeminiVideoExtractor,
    "NV_GeminiCourseSynthesizer": NV_GeminiCourseSynthesizer,
    "NV_GeminiBatchExtractor": NV_GeminiBatchExtractor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_GeminiVideoExtractor": "NV Gemini Video Extractor",
    "NV_GeminiCourseSynthesizer": "NV Gemini Course Synthesizer",
    "NV_GeminiBatchExtractor": "NV Gemini Batch Extractor",
}
