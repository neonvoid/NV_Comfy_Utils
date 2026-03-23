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


# ---------------------------------------------------------------------------
# Extraction JSON Schema (for Gemini responseSchema enforcement)
# ---------------------------------------------------------------------------

_EXTRACTION_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "video_title_guess": {"type": "STRING", "nullable": True},
        "overall_topic": {"type": "STRING"},
        "video_summary": {"type": "STRING"},
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
    },
    "required": [
        "overall_topic", "video_summary", "target_level", "learning_objectives", "segments",
        "key_terms", "uncertainties",
    ],
}


# ---------------------------------------------------------------------------
# Phase 1: Per-video extraction prompt
# ---------------------------------------------------------------------------

_EXTRACTION_PROMPT = """Analyze this tutorial/course video and extract structured learning content.

Return a single JSON object with these fields:

- video_title_guess: best guess at the video title from intro/slides/context (null if unclear)
- overall_topic: main subject of this video
- video_summary: 2-4 sentence free-text summary capturing the video's narrative arc and key takeaways — preserve nuance that structured fields might lose
- target_level: beginner | beginner_intermediate | intermediate | advanced | mixed | unclear
- prerequisites: knowledge assumed by the instructor
- learning_objectives: what a student should be able to do after watching
- segments: array of time-segmented content, each with:
  - start_mmss / end_mmss (MM:SS timestamps)
  - segment_title
  - segment_type: concept_explanation | demo | setup | recap_transition | exercise | q_and_a | other
  - pedagogical_role: how this segment fits into the learning arc
  - core_points: key concepts or facts taught
  - visible_artifacts: slides, code on screen, diagrams, terminal output, UI shown
  - skills_or_actions: specific things demonstrated or practiced
  - evidence_confidence: high | medium | low
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
- For visible_artifacts: describe what's shown (e.g., "VS Code with Python file") rather than transcribing text.
- The video_summary should capture the pedagogical narrative, not just list topics."""

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


def _resolve_api_key(api_key):
    """Resolve API key from input or environment variables."""
    if api_key and api_key.strip():
        return api_key.strip()
    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        val = os.environ.get(var)
        if val:
            return val.strip()
    raise RuntimeError(
        "No API key provided. Set the api_key input, or GEMINI_API_KEY / GOOGLE_API_KEY env var."
    )


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

        info = resp.json()
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
        requests.delete(f"{_API_BASE}/v1beta/{file_name}?key={api_key}", timeout=15)
        print(f"[NV_GeminiVideoExtractor] Cleaned up remote file: {file_name}")
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

    # Media resolution: controls tokens per video frame
    # "low" = ~100 tok/s (cheaper, good for talking-head), "default" = ~300 tok/s
    # Gemini 3+ requires v1alpha endpoint for mediaResolution
    use_v1alpha = False
    if media_resolution and media_resolution != "default":
        gen_config["mediaResolution"] = f"media_resolution_{media_resolution}"
        if is_gemini3:
            use_v1alpha = True

    # Gemini 3+ thinking config — use medium for structured extraction
    if is_gemini3:
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
          f"api={api_version}{', thinking=medium' if is_gemini3 else ''})")

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

        # Log usage
        usage = result.get("usageMetadata", {})
        print(f"[Gemini] Response in {req_time:.1f}s — "
              f"{usage.get('promptTokenCount', '?')} prompt tokens, "
              f"{usage.get('candidatesTokenCount', '?')} response tokens")

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

        return "\n".join(text_parts).strip()

    # All retries exhausted
    raise RuntimeError(f"Gemini API failed after {_MAX_RETRIES + 1} attempts: {last_error}")


# ---------------------------------------------------------------------------
# Helpers: Atomic file write
# ---------------------------------------------------------------------------

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
                               "Supports files up to 2GB / ~3 hours.",
                }),
            },
            "optional": {
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "Gemini API key from AI Studio. Leave empty to use "
                               "GEMINI_API_KEY or GOOGLE_API_KEY env vars.",
                }),
                "model": (["gemini-2.5-pro", "gemini-2.5-flash", "gemini-3-pro-preview", "gemini-3.1-pro-preview", "gemini-3-flash-preview", "gemini-2.0-flash"], {
                    "default": "gemini-2.5-pro",
                    "tooltip": "Gemini model. Pro has best temporal/visual reasoning for complex content "
                               "(slides, code, diagrams). Flash is faster and cheaper for simpler videos.",
                }),
                "media_resolution": (["default", "low"], {
                    "default": "default",
                    "tooltip": "Frame token density. 'default' (~300 tok/s) for code/slides/text on screen. "
                               "'low' (~100 tok/s) for talking-head content only.",
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
                    "default": 8192,
                    "min": 1024,
                    "max": 65536,
                    "step": 1024,
                    "tooltip": "Maximum output tokens. 8192 is usually sufficient for a "
                               "20-minute video extraction.",
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
                media_resolution="default", skip_existing=True, output_dir="",
                prompt_override="", max_tokens=8192, trigger=None):
        api_key = _resolve_api_key(api_key)

        # Validate video path
        video_path = video_path.strip()
        if not video_path:
            raise RuntimeError("video_path is empty")
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
        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)

        # Resolve output directory
        if not output_dir or not output_dir.strip():
            output_dir = os.path.join(
                folder_paths.get_output_directory(), "gemini_course", "extractions"
            )
        os.makedirs(output_dir, exist_ok=True)

        # Check for cached result
        json_path = os.path.join(output_dir, f"{video_name}.json")
        if skip_existing and os.path.isfile(json_path):
            print(f"[NV_GeminiVideoExtractor] Cached extraction found: {json_path}")
            with open(json_path, "r", encoding="utf-8") as f:
                extraction = json.load(f)
            extraction_str = json.dumps(extraction, indent=2, ensure_ascii=False)
            return (extraction_str, json_path, Path(video_path).name)

        print(f"\n{'=' * 60}")
        print(f"[NV_GeminiVideoExtractor] Processing: {Path(video_path).name}")
        print(f"  Size: {file_size_mb:.1f} MB | Format: {ext} | Model: {model}")
        print(f"  Resolution: {media_resolution} | Max tokens: {max_tokens}")
        print(f"{'=' * 60}")

        # Upload to Gemini Files API
        file_info = _upload_file(video_path, api_key, display_name=video_name)
        file_name = file_info.get("name", "")
        file_uri = file_info.get("uri", "")

        try:
            # Poll until file is processed and ready
            if file_info.get("state") != "ACTIVE":
                file_info = _poll_file_active(file_name, api_key, timeout=300)
                file_uri = file_info.get("uri", file_uri)

            # Build prompt
            prompt = (prompt_override.strip()
                      if prompt_override and prompt_override.strip()
                      else _EXTRACTION_PROMPT)

            # Content parts — video BEFORE text (Gemini best practice for video)
            contents = [
                {"file_data": {"file_uri": file_uri, "mime_type": mime}},
                {"text": prompt},
            ]

            # Call Gemini generateContent with schema enforcement
            raw_response = _generate_content(
                api_key=api_key,
                model=model,
                contents=contents,
                temperature=1.0,
                max_tokens=max_tokens,
                json_mode=True,
                response_schema=_EXTRACTION_SCHEMA,
                media_resolution=media_resolution,
            )

            # Parse JSON response
            try:
                extraction = json.loads(raw_response)
            except json.JSONDecodeError as e:
                # Fallback: try to extract JSON from markdown code block
                match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw_response, re.DOTALL)
                if match:
                    extraction = json.loads(match.group(1))
                else:
                    print(f"[NV_GeminiVideoExtractor] Warning: not valid JSON: {e}")
                    print(f"[NV_GeminiVideoExtractor] Raw (first 500): {raw_response[:500]}")
                    extraction = {"raw_response": raw_response, "parse_error": str(e)}

            # Add source metadata for traceability
            extraction["_source"] = {
                "video_file": Path(video_path).name,
                "video_path": video_path,
                "model": model,
                "media_resolution": media_resolution,
                "extracted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }

            # Atomic write to disk
            _atomic_write_json(json_path, extraction)

            extraction_str = json.dumps(extraction, indent=2, ensure_ascii=False)

            segments = extraction.get("segments", [])
            print(f"\n[NV_GeminiVideoExtractor] Extraction complete!")
            print(f"  Segments: {len(segments)}")
            print(f"  Topic: {extraction.get('overall_topic', 'N/A')}")
            print(f"  Level: {extraction.get('target_level', 'N/A')}")
            print(f"  Saved: {json_path}")
            print(f"{'=' * 60}\n")

            return (extraction_str, json_path, Path(video_path).name)

        finally:
            # Always clean up the uploaded file from Gemini servers
            if file_name:
                _delete_file(file_name, api_key)


# ---------------------------------------------------------------------------
# Node 2: Course Synthesizer
# ---------------------------------------------------------------------------

class NV_GeminiCourseSynthesizer:
    """Synthesize a structured course plan from multiple video extraction JSONs."""

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
                    "tooltip": "Gemini API key. Leave empty to use "
                               "GEMINI_API_KEY or GOOGLE_API_KEY env vars.",
                }),
                "model": (["gemini-2.5-pro", "gemini-2.5-flash", "gemini-3-pro-preview", "gemini-3.1-pro-preview", "gemini-3-flash-preview", "gemini-2.0-flash"], {
                    "default": "gemini-2.5-pro",
                    "tooltip": "Model for synthesis. Pro recommended for coherent "
                               "cross-video thematic analysis.",
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
                "output_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path for the output markdown file. "
                               "Empty = output/gemini_course/course_plan.md",
                }),
                "max_tokens": ("INT", {
                    "default": 16384,
                    "min": 2048,
                    "max": 65536,
                    "step": 1024,
                    "tooltip": "Maximum output tokens. Course plans for ~20 videos "
                               "typically need 8K-16K tokens.",
                }),
                "trigger": ("*", {
                    "tooltip": "Optional trigger input for sequencing after extraction.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("course_plan_md", "output_file_path")
    FUNCTION = "synthesize"
    CATEGORY = "NV_Utils/Course"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Combine multiple video extraction JSONs (from NV Gemini Video Extractor) into "
        "a cohesive course plan with modules, lessons, exercises, gap analysis, and "
        "redundancy detection. Outputs Markdown."
    )

    def synthesize(self, json_folder, api_key="", model="gemini-2.5-pro",
                   course_title="", custom_instructions="", output_path="",
                   max_tokens=16384, trigger=None):
        api_key = _resolve_api_key(api_key)

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

        # Build synthesis prompt with all extractions embedded
        extractions_text = json.dumps(extractions, indent=2, ensure_ascii=False)
        prompt = _SYNTHESIS_PROMPT.format(extractions=extractions_text)

        if course_title and course_title.strip():
            prompt += f"\n\nIMPORTANT: The course title must be: {course_title.strip()}"
        if custom_instructions and custom_instructions.strip():
            prompt += f"\n\nADDITIONAL INSTRUCTIONS:\n{custom_instructions.strip()}"

        # Call Gemini for synthesis (text output, not JSON)
        contents = [{"text": prompt}]

        course_plan = _generate_content(
            api_key=api_key,
            model=model,
            contents=contents,
            system_instruction=(
                "You are an experienced curriculum designer creating a structured course "
                "plan from video analysis data. Output clean, well-organized Markdown."
            ),
            temperature=0.7,
            max_tokens=max_tokens,
            json_mode=False,
        )

        # Resolve output path
        if not output_path or not output_path.strip():
            out_dir = os.path.join(
                folder_paths.get_output_directory(), "gemini_course"
            )
            os.makedirs(out_dir, exist_ok=True)
            output_path = os.path.join(out_dir, "course_plan.md")
        else:
            parent = os.path.dirname(output_path)
            if parent:
                os.makedirs(parent, exist_ok=True)

        # Atomic write course plan
        _atomic_write_text(output_path, course_plan)

        plan_lines = course_plan.count("\n")
        print(f"\n[NV_GeminiCourseSynthesizer] Course plan generated!")
        print(f"  Lines: {plan_lines} | Videos: {len(extractions)}")
        print(f"  Saved: {output_path}")
        print(f"{'=' * 60}\n")

        return (course_plan, output_path)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "NV_GeminiVideoExtractor": NV_GeminiVideoExtractor,
    "NV_GeminiCourseSynthesizer": NV_GeminiCourseSynthesizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_GeminiVideoExtractor": "NV Gemini Video Extractor",
    "NV_GeminiCourseSynthesizer": "NV Gemini Course Synthesizer",
}
