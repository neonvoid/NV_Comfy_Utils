"""NV Wan 2.7 Video Edit — slim API caller for Alibaba Tongyi Wan 2.7 Video Edit.

Pairs with NV_Wan27VideoEditPrep in a two-node pattern (mirrors the Kling and
Seedance forks):

  NV_Wan27VideoEditPrep  → uploads input video + refs, emits WAN27_UPLOAD_CONFIG
  NV_Wan27VideoEdit      → consumes config + final_prompt, calls the API

No uploading or tensor manipulation in this node. It reads pre-uploaded URLs
from the config, assembles the Wan 2.7 Video Edit request, polls the task
endpoint, and downloads the result. Matches the failure-handling pattern from
NV_SeedanceRefVideo: explicit terminal-status check + code/message surfacing
for moderation rejections that land with populated `output` but
`task_status="failed"`.

No `negative_prompt` exposed — Video Edit schema genuinely lacks it.
No `prompt_extend` exposed — Video Edit schema lacks it.
"""

from __future__ import annotations

import json
import math
import time

from comfy_api.latest import IO
from comfy_api_nodes.apis.wan import (
    TaskCreationResponse,
    VideoTaskStatusResponse,
    Wan27MediaItem,
    Wan27VideoEditInputField,
    Wan27VideoEditParametersField,
    Wan27VideoEditTaskCreationRequest,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    download_url_to_video_output,
    poll_op,
    sync_op,
    validate_string,
)

from .wan27_prep import WAN27_UPLOAD_CONFIG


WAN27_EDIT_MODEL_ID = "wan2.7-videoedit"
WAN27_TASK_ENDPOINT = "/proxy/wan/api/v1/services/aigc/video-generation/video-synthesis"
WAN27_TASK_STATUS_ENDPOINT = "/proxy/wan/api/v1/tasks"


class NV_Wan27VideoEdit(IO.ComfyNode):
    """Wan 2.7 Video Edit — slim API caller. Consumes WAN27_UPLOAD_CONFIG."""

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="NV_Wan27VideoEdit",
            display_name="NV Wan 2.7 Video Edit",
            category="NV_Utils/api",
            description=(
                "Wan 2.7 Video Edit API caller. Takes an upload config from "
                "NV Wan 2.7 Video Edit Prep + a final_prompt (from NV Prompt "
                "Refiner in wan27_edit mode) and calls the Alibaba Tongyi "
                "Wan 2.7 Video Edit endpoint."
            ),
            inputs=[
                WAN27_UPLOAD_CONFIG.Input(
                    "config",
                    tooltip="Upload config from NV Wan 2.7 Video Edit Prep.",
                ),
                IO.String.Input(
                    "final_prompt",
                    tooltip=(
                        "Final edit instruction (typically from NV Prompt Refiner "
                        "in wan27_edit mode). Describe the EDIT to make, not the "
                        "full scene. Wan 2.7 Video Edit has no negative_prompt "
                        "field — express avoidances as positive language."
                    ),
                    force_input=True,
                ),
                IO.Combo.Input(
                    "model",
                    options=[WAN27_EDIT_MODEL_ID],
                    default=WAN27_EDIT_MODEL_ID,
                    tooltip="Wan 2.7 Video Edit model variant.",
                ),
                IO.Combo.Input(
                    "resolution",
                    options=["720P", "1080P"],
                    default="720P",
                    tooltip=(
                        "Output resolution. 720P is $0.10/sec (input+output combined), "
                        "1080P is $0.15/sec."
                    ),
                ),
                IO.Combo.Input(
                    "ratio",
                    options=["auto", "16:9", "9:16", "1:1", "4:3", "3:4"],
                    default="auto",
                    tooltip=(
                        "Output aspect ratio. 'auto' omits the field from the payload "
                        "so the API infers from the input video."
                    ),
                ),
                IO.Combo.Input(
                    "duration_mode",
                    options=["auto", "manual"],
                    default="auto",
                    tooltip=(
                        "'auto' sends duration=0 to the API (server matches input video). "
                        "'manual' uses the slider below."
                    ),
                ),
                IO.Int.Input(
                    "duration",
                    default=5,
                    min=2,
                    max=10,
                    step=1,
                    tooltip="Output duration in seconds (2-10). Used when duration_mode='manual'.",
                    display_mode=IO.NumberDisplay.slider,
                ),
                IO.Combo.Input(
                    "audio_setting",
                    options=["auto", "origin"],
                    default="auto",
                    tooltip=(
                        "'auto' = model decides whether to regenerate audio based on prompt. "
                        "'origin' = preserve original audio. Coerced to 'auto' if the input "
                        "was assembled from frames (no audio track)."
                    ),
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
                    tooltip="Add Alibaba AI-generated watermark to output.",
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
        audio_setting: str,
        seed: int = 0,
        watermark: bool = False,
    ) -> IO.NodeOutput:
        t_start = time.time()

        validate_string(final_prompt, strip_whitespace=True, min_length=1)

        input_video_url = config.get("input_video_url")
        if not input_video_url:
            raise ValueError(
                "WAN27_UPLOAD_CONFIG has no input_video_url. Wire NV Wan 2.7 Video Edit Prep "
                "with a valid input video."
            )
        reference_image_urls: list[str] = config.get("reference_image_urls", [])
        input_video_dur = config.get("input_video_duration_s") or 0.0
        encode_source = config.get("encode_source", "unknown")
        n_refs = len(reference_image_urls)

        # --- resolve duration ---
        # auto → send 0 (API-native), manual → send slider value.
        if duration_mode == "auto":
            api_duration = 0
            duration_source = "auto (API matches input, sent as duration=0)"
        else:
            api_duration = duration
            duration_source = f"manual ({duration}s)"

        # --- guard audio_setting for frames-encoded source ---
        effective_audio_setting = audio_setting
        if audio_setting == "origin" and encode_source == "frames_encoded":
            effective_audio_setting = "auto"
            print(
                "[NV_Wan27VideoEdit] ⚠ audio_setting='origin' requested but input_video was "
                "assembled from frames (no audio track) — coercing to 'auto'."
            )

        # --- log resolved parameters ---
        print(f"[NV_Wan27VideoEdit] Model: {model} | res={resolution} ratio={ratio} "
              f"duration={api_duration} [{duration_source}]")
        print(f"[NV_Wan27VideoEdit] Input video: {encode_source}, {input_video_dur:.2f}s | "
              f"refs: {n_refs} ({config.get('image_sampling_note', '?')})")
        print(f"[NV_Wan27VideoEdit] Seed: {seed} | audio_setting={effective_audio_setting} "
              f"| watermark={watermark}")
        print(f"[NV_Wan27VideoEdit] Final prompt ({len(final_prompt)} chars):\n{final_prompt}")

        # --- build media list ---
        media: list[Wan27MediaItem] = [
            Wan27MediaItem(type="video", url=input_video_url),
        ]
        for url in reference_image_urls:
            media.append(Wan27MediaItem(type="reference_image", url=url))

        # --- probe: dump request content ---
        print(f"[NV_Wan27VideoEdit] Media array ({len(media)} items):")
        for i, item in enumerate(media):
            print(f"  [{i}] type={item.type!r} url=...{item.url[-40:]}")

        # --- build parameters, omitting ratio when "auto" ---
        params_kwargs: dict = {
            "resolution": resolution,
            "duration": api_duration,
            "audio_setting": effective_audio_setting,
            "watermark": watermark,
            "seed": seed,
        }
        if ratio != "auto":
            params_kwargs["ratio"] = ratio
        # When ratio == "auto" we leave it absent; the Pydantic field defaults to None
        # which we want the serializer to drop (stock uses exclude_none upstream).

        t_submit = time.time()

        # --- submit task ---
        initial_response = await sync_op(
            cls,
            ApiEndpoint(path=WAN27_TASK_ENDPOINT, method="POST"),
            response_model=TaskCreationResponse,
            data=Wan27VideoEditTaskCreationRequest(
                model=model,
                input=Wan27VideoEditInputField(prompt=final_prompt, media=media),
                parameters=Wan27VideoEditParametersField(**params_kwargs),
            ),
        )

        if not initial_response.output:
            raise RuntimeError(
                f"Wan 2.7 task creation failed. code={initial_response.code!r} "
                f"message={initial_response.message!r}"
            )

        task_id = initial_response.output.task_id
        print(f"[NV_Wan27VideoEdit] Task submitted: {task_id}")

        # --- poll ---
        # status_extractor returns a safe placeholder string when r.output is
        # transiently missing, so poll_op's status-comparison logic doesn't
        # crash on None during a rate-limit / gateway blip. The terminal-status
        # check below handles actual failures.
        response = await poll_op(
            cls,
            ApiEndpoint(path=f"{WAN27_TASK_STATUS_ENDPOINT}/{task_id}"),
            response_model=VideoTaskStatusResponse,
            status_extractor=lambda r: r.output.task_status if r.output else "pending",
            poll_interval=7,
        )

        t_done = time.time()

        # --- probe: response scout ---
        response_scout: dict = {}
        try:
            raw = response.model_dump() if hasattr(response, "model_dump") else {}
            if isinstance(raw, dict):
                response_scout["top_level_keys"] = sorted(raw.keys())
                if isinstance(raw.get("output"), dict):
                    response_scout["output_keys"] = sorted(raw["output"].keys())
                    modeled = {"task_id", "task_status", "video_url", "code", "message"}
                    extra = sorted(set(raw["output"].keys()) - modeled)
                    if extra:
                        response_scout["unmodeled_output_keys"] = extra
        except Exception as e:
            response_scout["scout_error"] = str(e)
        print(f"[NV_Wan27VideoEdit] Response scout: {response_scout}")

        # --- surface terminal failures BEFORE dereferencing video_url ---
        # Moderation rejections and server-side failures come back here with
        # populated `output` but task_status in {"failed", "canceled"} — same
        # pattern we handle in the Seedance fork.
        if response.output is None:
            raise RuntimeError(
                f"Wan 2.7 task returned no output. code={getattr(response, 'code', None)!r} "
                f"message={getattr(response, 'message', None)!r}"
            )
        task_status = response.output.task_status or ""
        if task_status.lower() not in ("succeeded", "success"):
            raise RuntimeError(
                f"Wan 2.7 task did not succeed (status={task_status!r}). "
                f"code={response.output.code!r} message={response.output.message!r}"
            )
        video_url = response.output.video_url
        if not video_url:
            raise RuntimeError(
                f"Wan 2.7 task status={task_status!r} but output.video_url is empty. "
                f"code={response.output.code!r} message={response.output.message!r}"
            )

        # --- download ---
        output_video = await download_url_to_video_output(video_url)

        import torch
        try:
            components = output_video.get_components()
            out_images = components.images
            out_fps = float(components.frame_rate)
            out_frames = int(out_images.shape[0])
        except Exception as e:
            print(f"[NV_Wan27VideoEdit] Warning: failed to decode frames from returned video: {e}")
            out_images = torch.zeros(1, 64, 64, 3)
            out_fps = 0.0
            out_frames = 0

        t_end = time.time()

        metadata = {
            "request": {
                "model": model,
                "resolution": resolution,
                "ratio": ratio,
                "duration_param": api_duration,
                "duration_source": duration_source,
                "duration_manual_input": duration,
                "audio_setting_requested": audio_setting,
                "audio_setting_effective": effective_audio_setting,
                "seed": seed,
                "watermark": watermark,
                "n_reference_images": n_refs,
                "input_video": {
                    "encode_source": encode_source,
                    "dimensions": config.get("input_video_dimensions"),
                    "duration_s": input_video_dur,
                },
                "prompt_length": len(final_prompt),
            },
            "response": {
                "task_id": task_id,
                "task_status": task_status,
                "video_url_tail": video_url[-60:] if video_url else None,
                "output_fps": out_fps,
                "output_frames": out_frames,
                "output_duration_s": round(out_frames / out_fps, 3) if out_fps else None,
                "scout": response_scout,
            },
            "timing": {
                "api_submit_sec": round(t_submit - t_start, 1),
                "api_processing_sec": round(t_done - t_submit, 1),
                "download_sec": round(t_end - t_done, 1),
                "total_sec": round(t_end - t_start, 1),
            },
        }

        # --- simple cost print (Wan 2.7 bills input+output seconds combined) ---
        rate_per_sec = 0.15 if resolution == "1080P" else 0.10
        billed_seconds_out = max(0.0, out_frames / out_fps) if out_fps else 0.0
        total_billed = (input_video_dur or 0.0) + billed_seconds_out
        est_usd = round(total_billed * rate_per_sec, 4)
        print(f"[NV_Wan27VideoEdit] Est. cost: {total_billed:.2f}s × ${rate_per_sec}/s = ${est_usd}")

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
    "NV_Wan27VideoEdit": NV_Wan27VideoEdit,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_Wan27VideoEdit": "NV Wan 2.7 Video Edit",
}
