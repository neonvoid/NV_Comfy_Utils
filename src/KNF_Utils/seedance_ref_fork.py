"""NV Seedance Ref Video (Fork)

MVP single-node fork of ComfyUI's ByteDance2ReferenceNode. Fixes the input
surface for programmatic pipeline wiring:

  - `prompt` is a top-level STRING (force_input), not nested in DynamicCombo.
    Enables chaining with NV_V2VPromptBuilder / PromptRefiner.
  - `reference_images` accepts a batched IMAGE tensor [N,H,W,C] and unpacks
    internally (uniform-sampled down to 9 if N > 9). No Autogrow slots.
  - `reference_video` is a single top-level VIDEO input (stock node allows up
    to 3; MVP supports 1 — multi-video can be added later if needed).
  - Auto-injects @Image1..N handles into the prompt if the prompt does not
    already contain them. The documented Seedance 2.0 convention is
    `@Image1 / @Image 1` (ByteDance Seed blog + BytePlus ModelArk playground).
    `@Video1` / `@Audio1` follow by inference.

Calls the same BytePlus Seedance 2.0 endpoint as the stock node and reuses
the stock request/response models from comfy_api_nodes.apis.bytedance.
"""

from __future__ import annotations

import json
import re
import time

import torch
from comfy_api.latest import IO, Input
from comfy_api_nodes.apis.bytedance import (
    SEEDANCE2_REF_VIDEO_PIXEL_LIMITS,
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
    upload_image_to_comfyapi,
    upload_video_to_comfyapi,
    validate_string,
)


SEEDANCE_MODELS = {
    "Seedance 2.0": "dreamina-seedance-2-0-260128",
    "Seedance 2.0 Fast": "dreamina-seedance-2-0-fast-260128",
}

BYTEPLUS_TASK_ENDPOINT = "/proxy/byteplus/api/v3/contents/generations/tasks"
BYTEPLUS_SEEDANCE2_TASK_STATUS_ENDPOINT = "/proxy/byteplus-seedance2/api/v3/contents/generations/tasks"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MAX_REF_IMAGES = 9


def _uniform_sample_indices(total: int, take: int) -> list[int]:
    """Return `take` evenly spaced indices from [0, total). total >= take >= 1."""
    if take >= total:
        return list(range(total))
    if take == 1:
        return [0]
    step = (total - 1) / (take - 1)
    return [round(i * step) for i in range(take)]


def _slice_image_batch(images: torch.Tensor, max_images: int) -> tuple[list[torch.Tensor], str]:
    """Slice batched IMAGE tensor [N,H,W,C] → list of [1,H,W,C] frames.

    If N > max_images, uniform-sample `max_images` frames across the batch.
    Returns (frames, sampling_note).
    """
    if images is None or images.shape[0] == 0:
        return [], "no images"

    n = images.shape[0]
    if n <= max_images:
        frames = [images[i:i + 1] for i in range(n)]
        return frames, f"{n}/{n} (all)"

    idx = _uniform_sample_indices(n, max_images)
    frames = [images[i:i + 1] for i in idx]
    return frames, f"uniform-sampled {max_images}/{n} (indices {idx})"


def _auto_inject_image_tags(prompt: str, n_images: int, has_video: bool) -> str:
    """Prepend @Image1..N and @Video1 if the prompt lacks any reference tags."""
    if not prompt:
        return prompt

    has_image_tag = bool(re.search(r"@Image\s?\d+", prompt))
    has_video_tag = bool(re.search(r"@Video\s?\d+", prompt))

    parts = []
    if has_video and not has_video_tag:
        parts.append("@Video1")
    if n_images > 0 and not has_image_tag:
        parts.extend(f"@Image{i}" for i in range(1, n_images + 1))

    if not parts:
        return prompt
    # Clean space-join — never "Using X: prompt" since video models can render
    # literal instructional text into the output frames.
    return f"{' '.join(parts)} {prompt}"


def _validate_ref_video(video: Input.Video, model_id: str) -> tuple[int, int, float]:
    """Check ref video dimensions and duration. Returns (w, h, duration_seconds).

    Raises ValueError with actionable messages on out-of-bounds.
    """
    w, h = video.get_dimensions()
    pixels = w * h

    limits = SEEDANCE2_REF_VIDEO_PIXEL_LIMITS.get(model_id, {})
    min_px = limits.get("min")
    max_px = limits.get("max")

    if min_px and pixels < min_px:
        raise ValueError(
            f"Reference video too small: {w}x{h} = {pixels:,}px. "
            f"Minimum {min_px:,}px (~{int(min_px ** 0.5)}x{int(min_px ** 0.5)}). "
            f"Upscale the ref video before wiring."
        )
    if max_px and pixels > max_px:
        raise ValueError(
            f"Reference video too large: {w}x{h} = {pixels:,}px. "
            f"Maximum {max_px:,}px (~{int(max_px ** 0.5)}x{int(max_px ** 0.5)}). "
            f"Downscale the ref video before wiring."
        )

    try:
        dur = float(video.get_duration())
    except Exception:
        dur = 0.0

    if dur and dur < 1.8:
        raise ValueError(f"Reference video too short: {dur:.2f}s. Minimum 1.8s.")
    if dur and dur > 15.1:
        raise ValueError(f"Reference video too long: {dur:.2f}s. Maximum 15.1s.")

    return w, h, dur


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class NV_SeedanceRefVideo(IO.ComfyNode):
    """Seedance 2.0 reference-to-video — MVP fork with pipeline-friendly inputs.

    Top-level `prompt` (force_input), batched IMAGE tensor, single VIDEO.
    Auto-injects @Image1..N handles if missing. Calls the same BytePlus
    Seedance 2.0 endpoint as the stock node.
    """

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="NV_SeedanceRefVideo",
            display_name="NV Seedance Ref Video",
            category="NV_Utils/api",
            description=(
                "MVP Seedance 2.0 reference-to-video caller. Accepts top-level "
                "prompt (force_input), batched IMAGE tensor (auto-sampled to 9), "
                "and single VIDEO reference. Use with NV V2V Prompt Builder "
                "in 'seedance_ref' mode for best results."
            ),
            inputs=[
                IO.String.Input(
                    "prompt",
                    tooltip=(
                        "Final Seedance prompt. Weave @Image1..N and @Video1 inline. "
                        "If missing, @Image / @Video tags are auto-prepended."
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
                IO.Image.Input(
                    "reference_images",
                    tooltip=(
                        "Batched IMAGE tensor [N,H,W,C]. Up to 9 images — if N > 9, "
                        "uniform-sampled. Each image becomes a role='reference_image' "
                        "entry in the API content array."
                    ),
                    optional=True,
                ),
                IO.Video.Input(
                    "reference_video",
                    tooltip=(
                        "Single reference video. Pixel budget ~409K-927K per frame "
                        "(~640x640 to ~960x960). Duration 1.8-15.1s. Becomes "
                        "role='reference_video' in the API content array."
                    ),
                    optional=True,
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
        prompt: str,
        model: str,
        resolution: str,
        ratio: str,
        duration: int,
        generate_audio: bool,
        reference_images: Input.Image | None = None,
        reference_video: Input.Video | None = None,
        seed: int = 0,
        watermark: bool = False,
    ) -> IO.NodeOutput:
        t_start = time.time()

        validate_string(prompt, strip_whitespace=True, min_length=1)
        model_id = SEEDANCE_MODELS[model]

        # --- unpack image batch ---
        image_frames, image_sampling_note = _slice_image_batch(
            reference_images, _MAX_REF_IMAGES
        ) if reference_images is not None else ([], "none")
        n_images = len(image_frames)

        # --- validate ref video ---
        ref_w = ref_h = 0
        ref_dur = 0.0
        if reference_video is not None:
            ref_w, ref_h, ref_dur = _validate_ref_video(reference_video, model_id)

        if n_images == 0 and reference_video is None:
            raise ValueError(
                "At least one reference_images frame or reference_video is required "
                "for the Seedance 2.0 Reference API."
            )

        # --- prepare final prompt ---
        final_prompt = _auto_inject_image_tags(
            prompt.strip(), n_images, has_video=reference_video is not None
        )

        print(f"[NV_SeedanceRefVideo] Model: {model_id} | res={resolution} ratio={ratio} dur={duration}s")
        print(f"[NV_SeedanceRefVideo] Refs: images={image_sampling_note}, "
              f"video={'yes ' + str(ref_w) + 'x' + str(ref_h) + ' ' + f'{ref_dur:.2f}s' if reference_video else 'no'}")
        print(f"[NV_SeedanceRefVideo] Final prompt ({len(final_prompt)} chars):\n{final_prompt}")

        # --- upload refs + build content array ---
        content: list[TaskTextContent | TaskImageContent | TaskVideoContent] = [
            TaskTextContent(text=final_prompt),
        ]

        for i, frame in enumerate(image_frames, 1):
            url = await upload_image_to_comfyapi(
                cls, image=frame, wait_label=f"Uploading @Image{i}"
            )
            content.append(
                TaskImageContent(
                    image_url=TaskImageContentUrl(url=url),
                    role="reference_image",
                )
            )
            print(f"[NV_SeedanceRefVideo] @Image{i} uploaded → ...{url[-40:]}")

        has_video_input = reference_video is not None
        if reference_video is not None:
            url = await upload_video_to_comfyapi(
                cls, reference_video, wait_label="Uploading @Video1"
            )
            content.append(
                TaskVideoContent(
                    video_url=TaskVideoContentUrl(url=url),
                    role="reference_video",
                )
            )
            print(f"[NV_SeedanceRefVideo] @Video1 uploaded → ...{url[-40:]}")

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

        try:
            components = output_video.get_components()
            out_fps = float(components.frame_rate)
            out_frames = int(components.images.shape[0])
        except Exception:
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
                "image_sampling": image_sampling_note,
                "has_reference_video": has_video_input,
                "ref_video_dimensions": f"{ref_w}x{ref_h}" if has_video_input else None,
                "ref_video_duration_s": round(ref_dur, 3) if has_video_input else None,
                "prompt_length": len(final_prompt),
            },
            "response": {
                "task_id": task_id,
                "video_url_tail": video_url[-60:] if video_url else None,
                "output_fps": out_fps,
                "output_frames": out_frames,
            },
            "timing": {
                "upload_sec": round(t_submit - t_start, 1),
                "api_processing_sec": round(t_done - t_submit, 1),
                "download_sec": round(t_end - t_done, 1),
                "total_sec": round(t_end - t_start, 1),
            },
        }

        return IO.NodeOutput(
            output_video,
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
