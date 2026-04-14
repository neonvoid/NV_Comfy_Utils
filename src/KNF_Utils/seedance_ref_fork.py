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
                IO.Int.Input(
                    "duration",
                    default=7,
                    min=4,
                    max=15,
                    step=1,
                    tooltip="Output duration in seconds (4-15).",
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

        # --- safety-net tag injection ---
        final_prompt = _auto_inject_image_tags(final_prompt.strip(), n_images, has_video)

        print(f"[NV_SeedanceRefVideo] Model: {model_id} | res={resolution} ratio={ratio} dur={duration}s")
        print(f"[NV_SeedanceRefVideo] Refs: images={n_images} ({config.get('image_sampling_note', '?')}), "
              f"video={'yes' if has_video else 'no'}")
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
                duration=duration,
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
                "duration": duration,
                "generate_audio": generate_audio,
                "seed": seed,
                "watermark": watermark,
                "n_reference_images": n_images,
                "has_reference_video": has_video,
                "ref_video_dimensions": config.get("ref_video_dimensions"),
                "ref_video_duration_s": config.get("ref_video_duration_s"),
                "prompt_length": len(final_prompt),
            },
            "response": {
                "task_id": task_id,
                "video_url_tail": video_url[-60:] if video_url else None,
                "output_fps": out_fps,
                "output_frames": out_frames,
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
