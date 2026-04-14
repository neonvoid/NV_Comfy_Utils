"""NV Seedance Ref Video — slim API caller for Seedance 2.0 ref-to-video.

Pairs with NV_SeedancePrep in a two-node pattern (mirrors the Kling fork):

  NV_SeedancePrep        → uploads refs, emits SEEDANCE_UPLOAD_CONFIG
  NV_SeedanceRefVideo    → consumes config + final_prompt, calls the API

This node does NO uploading or tensor manipulation. It reads pre-uploaded
asset URLs from the config, assembles the Seedance 2.0 request, polls the
task endpoint, and downloads the result.

`@Image1..N / @Video1` auto-injection is preserved as a safety net in case
the upstream prompt lacks tags (e.g. user bypassed the refiner).
"""

from __future__ import annotations

import json
import re
import time

from comfy_api.latest import IO
from comfy_api_nodes.apis.bytedance import (
    Seedance2TaskCreationRequest,
    TaskCreationResponse,
    TaskImageContent,
    TaskImageContentUrl,
    TaskStatusResponse,
    TaskTextContent,
    TaskVideoContent,
    TaskVideoContentUrl,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    download_url_to_video_output,
    poll_op,
    sync_op,
    validate_string,
)

from .seedance_prep import SEEDANCE_UPLOAD_CONFIG


SEEDANCE_MODELS = {
    "Seedance 2.0": "dreamina-seedance-2-0-260128",
    "Seedance 2.0 Fast": "dreamina-seedance-2-0-fast-260128",
}

BYTEPLUS_TASK_ENDPOINT = "/proxy/byteplus/api/v3/contents/generations/tasks"
BYTEPLUS_SEEDANCE2_TASK_STATUS_ENDPOINT = "/proxy/byteplus-seedance2/api/v3/contents/generations/tasks"


def _auto_inject_image_tags(prompt: str, n_images: int, has_video: bool) -> str:
    """Safety-net tag injection. Clean space-prefix (no "Using X:" text)."""
    if not prompt:
        return prompt

    has_image_tag = bool(re.search(r"@Image\s?\d+", prompt))
    has_video_tag = bool(re.search(r"@Video\s?\d+", prompt))

    parts: list[str] = []
    if has_video and not has_video_tag:
        parts.append("@Video1")
    if n_images > 0 and not has_image_tag:
        parts.extend(f"@Image{i}" for i in range(1, n_images + 1))

    if not parts:
        return prompt
    return f"{' '.join(parts)} {prompt}"


def _analyze_prompt_tags(prompt: str, n_images: int, has_video: bool) -> dict:
    """Extract @Image / @Video / @Audio tag usage for probe-mode diagnostics.

    Returns per-tag counts + mismatch warnings. Useful when the model's
    adherence to a specific tag is being tested.
    """
    image_tags = re.findall(r"@Image\s?(\d+)", prompt)
    video_tags = re.findall(r"@Video\s?(\d+)", prompt)
    audio_tags = re.findall(r"@Audio\s?(\d+)", prompt)

    image_indices = sorted({int(t) for t in image_tags})
    video_indices = sorted({int(t) for t in video_tags})
    audio_indices = sorted({int(t) for t in audio_tags})

    warnings: list[str] = []
    if image_indices and max(image_indices) > n_images:
        warnings.append(
            f"prompt references @Image{max(image_indices)} but only {n_images} image(s) uploaded"
        )
    if video_indices and max(video_indices) > (1 if has_video else 0):
        warnings.append(
            f"prompt references @Video{max(video_indices)} but "
            f"{'only 1 video uploaded' if has_video else 'no video uploaded'}"
        )
    if audio_indices:
        warnings.append(
            f"prompt references @Audio{audio_indices} but this node does not support audio refs yet"
        )
    if n_images > 0 and not image_indices:
        warnings.append(f"{n_images} image(s) uploaded but no @ImageN tags in prompt")
    if has_video and not video_indices:
        warnings.append("reference video uploaded but no @Video1 tag in prompt")

    return {
        "image_tag_indices": image_indices,
        "video_tag_indices": video_indices,
        "audio_tag_indices": audio_indices,
        "n_image_tags": len(image_tags),
        "n_video_tags": len(video_tags),
        "n_audio_tags": len(audio_tags),
        "warnings": warnings,
    }


def _summarize_content(content: list) -> list[dict]:
    """Compact summary of the request content array for debug logs."""
    summary = []
    for i, item in enumerate(content):
        cls_name = type(item).__name__
        entry: dict = {"index": i, "type": cls_name}
        if cls_name == "TaskTextContent":
            text = getattr(item, "text", "")
            entry["text_length"] = len(text)
            entry["text_head"] = text[:120] + ("…" if len(text) > 120 else "")
        elif cls_name == "TaskImageContent":
            entry["role"] = getattr(item, "role", None)
            url = getattr(getattr(item, "image_url", None), "url", "") or ""
            entry["url_tail"] = "..." + url[-40:] if url else None
        elif cls_name == "TaskVideoContent":
            entry["role"] = getattr(item, "role", None)
            url = getattr(getattr(item, "video_url", None), "url", "") or ""
            entry["url_tail"] = "..." + url[-40:] if url else None
        summary.append(entry)
    return summary


def _scout_raw_response(response) -> dict:
    """Dump top-level response keys for schema-evolution scouting.

    ByteDance may add fields (prompt_refined, moderation_notes, seed_used,
    etc.) that aren't in our Pydantic model. This surfaces them so we know
    to update the model if new data appears.
    """
    scout: dict = {"top_level_keys": None, "content_keys": None, "extra_notes": []}
    try:
        raw = response.model_dump() if hasattr(response, "model_dump") else None
        if isinstance(raw, dict):
            scout["top_level_keys"] = sorted(raw.keys())
            content = raw.get("content")
            if isinstance(content, dict):
                scout["content_keys"] = sorted(content.keys())
                # Flag any content keys we don't model
                modeled = {"video_url"}
                extra = sorted(set(content.keys()) - modeled)
                if extra:
                    scout["extra_notes"].append(f"unmodeled content keys: {extra}")
    except Exception as e:
        scout["extra_notes"].append(f"model_dump failed: {e}")
    return scout


class NV_SeedanceRefVideo(IO.ComfyNode):
    """Seedance 2.0 reference-to-video — slim API caller. Consumes SEEDANCE_UPLOAD_CONFIG."""

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="NV_SeedanceRefVideo",
            display_name="NV Seedance Ref Video",
            category="NV_Utils/api",
            description=(
                "Seedance 2.0 ref-to-video API caller. Takes an upload config "
                "from NV Seedance Prep + a final_prompt (from NV Prompt Refiner) "
                "and calls the BytePlus Seedance 2.0 endpoint."
            ),
            inputs=[
                SEEDANCE_UPLOAD_CONFIG.Input(
                    "config",
                    tooltip="Upload config from NV Seedance Prep.",
                ),
                IO.String.Input(
                    "final_prompt",
                    tooltip=(
                        "Final Seedance prompt (typically from NV Prompt Refiner "
                        "in seedance_ref mode). @Image/@Video tags are auto-injected "
                        "as a safety net if absent."
                    ),
                    force_input=True,
                ),
                IO.Combo.Input(
                    "model",
                    options=list(SEEDANCE_MODELS.keys()),
                    default="Seedance 2.0 Fast",
                    tooltip="Seedance 2.0 = quality. Seedance 2.0 Fast = ~20% cheaper.",
                ),
                IO.Combo.Input(
                    "resolution",
                    options=["480p", "720p"],
                    default="720p",
                    tooltip="Output resolution. Seedance 2.0 does not support 1080p.",
                ),
                IO.Combo.Input(
                    "ratio",
                    options=["16:9", "4:3", "1:1", "3:4", "9:16", "21:9", "adaptive"],
                    default="adaptive",
                    tooltip="Output aspect ratio. 'adaptive' lets the model pick based on refs.",
                ),
                IO.Combo.Input(
                    "duration_mode",
                    options=["auto", "manual"],
                    default="auto",
                    tooltip=(
                        "'auto' = match the ref video's duration (ceil, clamped 4-15s). "
                        "Falls back to the manual duration if no ref video is in the config.\n"
                        "'manual' = always use the duration slider below."
                    ),
                ),
                IO.Int.Input(
                    "duration",
                    default=7,
                    min=4,
                    max=15,
                    step=1,
                    tooltip="Output duration in seconds (4-15). Used when duration_mode='manual' or no ref video.",
                    display_mode=IO.NumberDisplay.slider,
                ),
                IO.Boolean.Input(
                    "generate_audio",
                    default=True,
                    tooltip="Enable audio generation for the output video.",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed controls re-run; results are non-deterministic.",
                    optional=True,
                ),
                IO.Boolean.Input(
                    "watermark",
                    default=False,
                    tooltip="Add ByteDance watermark to output video.",
                    optional=True,
                ),
            ],
            outputs=[
                IO.Video.Output(display_name="video"),
                IO.Image.Output(display_name="images"),
                IO.Float.Output(display_name="output_fps"),
                IO.Int.Output(display_name="output_frames"),
                IO.String.Output(display_name="final_prompt"),
                IO.String.Output(display_name="api_metadata"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
        )

    @classmethod
    async def execute(
        cls,
        config: dict,
        final_prompt: str,
        model: str,
        resolution: str,
        ratio: str,
        duration_mode: str,
        duration: int,
        generate_audio: bool,
        seed: int = 0,
        watermark: bool = False,
    ) -> IO.NodeOutput:
        t_start = time.time()

        validate_string(final_prompt, strip_whitespace=True, min_length=1)
        model_id = SEEDANCE_MODELS[model]

        uploaded_image_urls: list[str] = config.get("uploaded_image_urls", [])
        uploaded_video_url: str | None = config.get("uploaded_video_url")
        n_images = len(uploaded_image_urls)
        has_video = uploaded_video_url is not None

        if n_images == 0 and not has_video:
            raise ValueError(
                "SEEDANCE_UPLOAD_CONFIG has no uploaded refs. "
                "Wire at least one image or video into NV Seedance Prep."
            )

        # --- resolve duration ---
        import math
        ref_dur = config.get("ref_video_duration_s")
        if duration_mode == "auto" and ref_dur:
            api_duration = max(4, min(15, math.ceil(float(ref_dur))))
            duration_source = f"auto (from ref video {ref_dur:.2f}s → ceil → {api_duration}s, clamped 4-15)"
        else:
            api_duration = duration
            if duration_mode == "auto":
                duration_source = f"auto-fallback to manual ({duration}s) — no ref video in config"
            else:
                duration_source = f"manual ({duration}s)"

        # --- safety-net tag injection ---
        prompt_before_inject = final_prompt.strip()
        final_prompt = _auto_inject_image_tags(prompt_before_inject, n_images, has_video)
        tag_injected = final_prompt != prompt_before_inject

        # --- probe-mode prompt tag analysis ---
        tag_analysis = _analyze_prompt_tags(final_prompt, n_images, has_video)

        print(f"[NV_SeedanceRefVideo] Model: {model_id} | res={resolution} ratio={ratio} dur={api_duration}s [{duration_source}]")
        print(f"[NV_SeedanceRefVideo] Refs: images={n_images} ({config.get('image_sampling_note', '?')}), "
              f"video={'yes' if has_video else 'no'}")
        print(f"[NV_SeedanceRefVideo] Seed: {seed} | generate_audio={generate_audio} | watermark={watermark}")
        print(f"[NV_SeedanceRefVideo] Tag analysis: "
              f"@Image{tag_analysis['image_tag_indices'] or '[]'}, "
              f"@Video{tag_analysis['video_tag_indices'] or '[]'}, "
              f"@Audio{tag_analysis['audio_tag_indices'] or '[]'}"
              f"{' (auto-injected)' if tag_injected else ''}")
        for w in tag_analysis["warnings"]:
            print(f"[NV_SeedanceRefVideo] ⚠ tag warning: {w}")
        print(f"[NV_SeedanceRefVideo] Final prompt ({len(final_prompt)} chars):\n{final_prompt}")

        # --- build content array from config ---
        content: list[TaskTextContent | TaskImageContent | TaskVideoContent] = [
            TaskTextContent(text=final_prompt),
        ]
        for url in uploaded_image_urls:
            content.append(
                TaskImageContent(
                    image_url=TaskImageContentUrl(url=url),
                    role="reference_image",
                )
            )
        if uploaded_video_url is not None:
            content.append(
                TaskVideoContent(
                    video_url=TaskVideoContentUrl(url=uploaded_video_url),
                    role="reference_video",
                )
            )

        # --- probe: dump request content array structure ---
        content_summary = _summarize_content(content)
        print(f"[NV_SeedanceRefVideo] Content array ({len(content)} items):")
        for entry in content_summary:
            print(f"  [{entry['index']}] {entry['type']} "
                  f"role={entry.get('role', '-')} "
                  f"{'url=' + entry['url_tail'] if entry.get('url_tail') else ''}"
                  f"{'text=' + repr(entry['text_head']) if 'text_head' in entry else ''}")

        t_submit = time.time()

        # --- submit task ---
        initial_response = await sync_op(
            cls,
            ApiEndpoint(path=BYTEPLUS_TASK_ENDPOINT, method="POST"),
            data=Seedance2TaskCreationRequest(
                model=model_id,
                content=content,
                generate_audio=generate_audio,
                resolution=resolution,
                ratio=ratio,
                duration=api_duration,
                seed=seed,
                watermark=watermark,
            ),
            response_model=TaskCreationResponse,
        )

        task_id = initial_response.id
        print(f"[NV_SeedanceRefVideo] Task submitted: {task_id}")

        # --- poll ---
        response = await poll_op(
            cls,
            ApiEndpoint(path=f"{BYTEPLUS_SEEDANCE2_TASK_STATUS_ENDPOINT}/{task_id}"),
            response_model=TaskStatusResponse,
            status_extractor=lambda r: r.status,
            poll_interval=9,
        )

        t_done = time.time()

        # --- probe: scout raw response for unmodeled fields ---
        response_scout = _scout_raw_response(response)
        print(f"[NV_SeedanceRefVideo] Response scout: "
              f"status={response.status!r} top_keys={response_scout.get('top_level_keys')}")
        if response_scout.get("content_keys"):
            print(f"  content keys: {response_scout['content_keys']}")
        for note in response_scout.get("extra_notes", []):
            print(f"  ⚠ {note}")

        # --- probe: token usage + cost estimate ---
        token_usage: dict = {
            "completion_tokens": None,
            "total_tokens": None,
            "cost_estimate_usd": None,
            "cost_formula": None,
        }
        try:
            if response.usage is not None:
                token_usage["completion_tokens"] = int(response.usage.completion_tokens)
                token_usage["total_tokens"] = int(response.usage.total_tokens)
                # Pricing: total_tokens × 1.43 × rate_per_1K / 1000
                # rate depends on model + has_video_input
                from comfy_api_nodes.apis.bytedance import SEEDANCE2_PRICE_PER_1K_TOKENS
                rate = SEEDANCE2_PRICE_PER_1K_TOKENS.get((model_id, has_video))
                if rate is not None:
                    cost = token_usage["total_tokens"] * 1.43 * rate / 1000.0
                    token_usage["cost_estimate_usd"] = round(cost, 4)
                    token_usage["cost_formula"] = (
                        f"{token_usage['total_tokens']} × 1.43 × ${rate}/1K = ${cost:.4f}"
                    )
                print(f"[NV_SeedanceRefVideo] Tokens: total={token_usage['total_tokens']}, "
                      f"completion={token_usage['completion_tokens']} | "
                      f"cost≈${token_usage['cost_estimate_usd']}")
        except Exception as e:
            print(f"[NV_SeedanceRefVideo] Warning: token usage capture failed: {e}")

        # --- surface terminal failures before dereferencing content ---
        if response.status != "succeeded":
            err = response.error
            err_msg = f"code={err.code!r}, message={err.message!r}" if err else "no error detail"
            raise RuntimeError(
                f"Seedance task did not succeed (status={response.status}). {err_msg}"
            )
        if response.content is None or not response.content.video_url:
            raise RuntimeError(
                f"Seedance task status={response.status} but response.content.video_url is empty. "
                f"Raw error: {response.error}"
            )

        # --- download ---
        video_url = response.content.video_url
        output_video = await download_url_to_video_output(video_url)

        import torch
        try:
            components = output_video.get_components()
            out_images = components.images
            out_fps = float(components.frame_rate)
            out_frames = int(out_images.shape[0])
        except Exception as e:
            print(f"[NV_SeedanceRefVideo] Warning: failed to decode frames from returned video: {e}")
            out_images = torch.zeros(1, 64, 64, 3)
            out_fps = 0.0
            out_frames = 0

        t_end = time.time()

        metadata = {
            "request": {
                "model": model_id,
                "resolution": resolution,
                "ratio": ratio,
                "duration": api_duration,
                "duration_source": duration_source,
                "duration_manual_input": duration,
                "generate_audio": generate_audio,
                "seed": seed,
                "watermark": watermark,
                "n_reference_images": n_images,
                "has_reference_video": has_video,
                "ref_video_dimensions": config.get("ref_video_dimensions"),
                "ref_video_duration_s": config.get("ref_video_duration_s"),
                "prompt_length": len(final_prompt),
                "prompt_tags_auto_injected": tag_injected,
                "tag_analysis": tag_analysis,
                "content_array": content_summary,
            },
            "response": {
                "task_id": task_id,
                "status": response.status,
                "video_url_tail": video_url[-60:] if video_url else None,
                "output_fps": out_fps,
                "output_frames": out_frames,
                "output_duration_s": round(out_frames / out_fps, 3) if out_fps else None,
                "scout": response_scout,
                "token_usage": token_usage,
            },
            "timing": {
                "api_submit_sec": round(t_submit - t_start, 1),
                "api_processing_sec": round(t_done - t_submit, 1),
                "download_sec": round(t_end - t_done, 1),
                "total_sec": round(t_end - t_start, 1),
            },
        }

        return IO.NodeOutput(
            output_video,
            out_images,
            out_fps,
            out_frames,
            final_prompt,
            json.dumps(metadata, indent=2),
        )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "NV_SeedanceRefVideo": NV_SeedanceRefVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_SeedanceRefVideo": "NV Seedance Ref Video",
}
