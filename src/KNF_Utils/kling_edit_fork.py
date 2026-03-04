"""NV Kling Edit Video (Fork)

Fork of ComfyUI's built-in Kling 3.0 Omni Edit/V2V nodes with:
  - IMAGE sequence I/O (not VIDEO) — plugs directly into ComfyUI pipelines
  - refer_type toggle: edit (base) vs reference (feature) in one node
  - Auto-duration from input frame count + fps
  - Auto-inject @image/@video reference tags when not manually specified
  - Separate negative_prompt field (appended as "Avoid: ...")
  - Exposed API params: duration, aspect_ratio, sound
  - Full API metadata output — request payload, response details, timing

Uses the same /proxy/kling/v1/videos/omni-video endpoint and
OmniProReferences2VideoRequest model as the original.
"""

import json
import math
import re
import time

from fractions import Fraction

from comfy_api.latest import IO, Input, InputImpl
from comfy_api.latest._util.video_types import VideoComponents
from comfy_api_nodes.apis.kling import (
    OmniParamImage,
    OmniParamVideo,
    OmniProReferences2VideoRequest,
    TaskStatusResponse,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    download_url_to_video_output,
    get_number_of_images,
    poll_op,
    sync_op,
    upload_images_to_comfyapi,
    upload_video_to_comfyapi,
    validate_image_aspect_ratio,
    validate_image_dimensions,
    validate_string,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_omni_prompt_references(prompt: str) -> str:
    """Rewrite @image/@video placeholders to <<<image_N>>>/<<<video_N>>> API form."""
    if not prompt:
        return prompt

    def _image_repl(m):
        return f"<<<image_{m.group('idx') or '1'}>>>"

    def _video_repl(m):
        return f"<<<video_{m.group('idx') or '1'}>>>"

    prompt = re.sub(r"(?<!\w)@image(?P<idx>\d*)(?!\w)", _image_repl, prompt)
    return re.sub(r"(?<!\w)@video(?P<idx>\d*)(?!\w)", _video_repl, prompt)


def _auto_inject_reference_tags(
    prompt: str,
    num_ref_images: int,
    is_feature: bool,
) -> str:
    """Auto-prepend @image/@video tags if user didn't include them manually.

    Skips injection if the prompt already contains any @image/<<<image or
    @video/<<<video tags — respects manual tag placement.
    """
    if not prompt:
        return prompt

    has_image_tags = bool(re.search(r"(?:@image|<<<image_)\d*", prompt, re.IGNORECASE))
    has_video_tags = bool(re.search(r"(?:@video|<<<video_)\d*", prompt, re.IGNORECASE))

    parts = []

    # Auto-inject @video tag for the input video in feature/reference mode
    if is_feature and not has_video_tags:
        parts.append("@video")

    # Auto-inject @image tags for each connected reference image
    if num_ref_images > 0 and not has_image_tags:
        for i in range(1, num_ref_images + 1):
            parts.append(f"@image{i}")

    if not parts:
        return prompt

    tag_str = ", ".join(parts)
    return f"Using {tag_str}: {prompt}"


def _infer_aspect_ratio(w: int, h: int) -> str:
    """Pick the nearest standard Kling aspect ratio from input dimensions."""
    ratio = w / h
    options = [
        (abs(ratio - 16 / 9), "16:9"),
        (abs(ratio - 9 / 16), "9:16"),
        (abs(ratio - 1.0), "1:1"),
    ]
    return min(options, key=lambda x: x[0])[1]


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class NV_KlingEditVideo(IO.ComfyNode):
    """Kling 3.0 Omni Edit Video — IMAGE I/O, auto-duration, full API transparency."""

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="NV_KlingEditVideo",
            display_name="NV Kling Edit Video",
            category="NV_Utils/api",
            description=(
                "Fork of the built-in Kling Edit Video node. "
                "Accepts/outputs IMAGE sequences (not VIDEO), auto-calculates "
                "duration from frame count, and exposes all API parameters."
            ),
            inputs=[
                IO.Combo.Input(
                    "model_name",
                    options=["kling-v3-omni", "kling-video-o1"],
                    tooltip="Kling model to use. v3-omni is latest, o1 is previous gen.",
                ),
                IO.Combo.Input(
                    "refer_type",
                    options=["edit (base)", "reference (feature)"],
                    tooltip=(
                        "'edit (base)': Direct video editing — modifies the input video "
                        "per prompt. Aspect ratio and duration are inferred from the "
                        "input (sent as None to API).\n"
                        "'reference (feature)': Style/motion reference — generates new "
                        "content using the input as a visual/motion guide. Aspect ratio "
                        "and duration are user-specified."
                    ),
                ),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip=(
                        "Text prompt. In edit mode: describe what to change. "
                        "In reference mode: describe the new scene to generate.\n"
                        "Supports @image/@video placeholders for reference images/video. "
                        "Tags are auto-injected if references are connected but no tags "
                        "are in the prompt. Max 2500 chars."
                    ),
                ),
                IO.String.Input(
                    "negative_prompt",
                    multiline=True,
                    default="",
                    tooltip=(
                        "Describe what to avoid in the output. Appended to the main "
                        "prompt as 'Avoid: {text}'. Leave empty to skip.\n"
                        "The Kling Omni API has no separate negative prompt field, "
                        "so this is concatenated into the main prompt."
                    ),
                    optional=True,
                ),
                IO.Image.Input(
                    "images",
                    tooltip=(
                        "Input frame sequence [B,H,W,C]. Encoded to video at the "
                        "specified fps for upload. Each dimension must be 720-2160px."
                    ),
                ),
                IO.Int.Input(
                    "fps",
                    default=30,
                    min=1,
                    max=60,
                    display_mode=IO.NumberDisplay.number,
                    tooltip=(
                        "Frame rate for encoding input images to video. "
                        "Also used with 'auto' duration to calculate seconds. "
                        "Kling API accepts 24-60fps input."
                    ),
                ),
                IO.Combo.Input(
                    "duration_mode",
                    options=["auto", "manual"],
                    tooltip=(
                        "'auto': duration = ceil(num_frames / fps), clamped to mode max. "
                        "'manual': use the duration slider value.\n"
                        "In edit mode, duration is sent as None regardless (API infers "
                        "from input video). The calculated value is still shown in metadata."
                    ),
                ),
                IO.Int.Input(
                    "duration",
                    default=5,
                    min=3,
                    max=15,
                    display_mode=IO.NumberDisplay.slider,
                    tooltip=(
                        "Output video duration in seconds (manual mode). "
                        "Edit mode: 3-10s (API-inferred). "
                        "Reference mode: 3-15s (v3-omni) or 3-10s (o1)."
                    ),
                ),
                IO.Combo.Input(
                    "aspect_ratio",
                    options=["auto", "16:9", "9:16", "1:1"],
                    tooltip=(
                        "In edit mode: always sent as None — API infers from input. "
                        "This setting is ignored.\n"
                        "In reference mode: 'auto' infers from input dimensions "
                        "(nearest standard ratio). Other values set output aspect ratio."
                    ),
                ),
                IO.Boolean.Input(
                    "keep_original_sound",
                    default=True,
                    tooltip="Preserve audio from the input video.",
                ),
                IO.Boolean.Input(
                    "generate_sound",
                    default=False,
                    tooltip=(
                        "Enable Kling's AI audio generation (sound='on'). "
                        "When off, the sound param is omitted from the API request."
                    ),
                ),
                IO.Image.Input(
                    "reference_images",
                    tooltip="Up to 4 additional reference images for style/identity.",
                    optional=True,
                ),
                IO.Combo.Input(
                    "resolution",
                    options=["1080p", "720p"],
                    tooltip="'1080p' = pro mode ($0.168/s), '720p' = std mode ($0.126/s).",
                    optional=True,
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip=(
                        "Seed controls whether the node should re-run; "
                        "results are non-deterministic regardless of seed."
                    ),
                    optional=True,
                ),
            ],
            outputs=[
                IO.Image.Output(display_name="images"),
                IO.Float.Output(display_name="output_fps"),
                IO.Int.Output(display_name="output_frames"),
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
        model_name: str,
        refer_type: str,
        prompt: str,
        images: Input.Image,
        fps: int,
        duration_mode: str,
        duration: int,
        aspect_ratio: str,
        keep_original_sound: bool,
        generate_sound: bool,
        reference_images: Input.Image | None = None,
        resolution: str = "1080p",
        seed: int = 0,
        negative_prompt: str = "",
    ) -> IO.NodeOutput:
        _ = seed
        t_start = time.time()

        # --- mode ---
        is_feature = refer_type == "reference (feature)"
        api_refer_type = "feature" if is_feature else "base"

        # --- input analysis + auto-upscale video if undersized ---
        num_frames = images.shape[0]
        h, w = images.shape[1], images.shape[2]

        if h < 720 or w < 720:
            images = _ensure_min_size(images, min_px=720)
            new_h, new_w = images.shape[1], images.shape[2]
            print(f"[NV_KlingEditVideo] Input video {w}x{h} below 720px minimum, "
                  f"upscaled to {new_w}x{new_h}")
            h, w = new_h, new_w

        input_duration_exact = num_frames / fps

        # --- duration calculation ---
        max_duration = 15 if is_feature else 10
        if duration_mode == "auto":
            api_duration = max(3, min(max_duration, math.ceil(input_duration_exact)))
        else:
            api_duration = min(duration, max_duration)

        # --- prompt pipeline: auto-inject → negative → normalize ---
        num_ref_images = get_number_of_images(reference_images) if reference_images is not None else 0
        original_prompt = prompt
        prompt = _auto_inject_reference_tags(prompt, num_ref_images, is_feature)

        if negative_prompt and negative_prompt.strip():
            prompt = f"{prompt}\nAvoid: {negative_prompt.strip()}"

        prompt = _normalize_omni_prompt_references(prompt)
        validate_string(prompt, min_length=1, max_length=2500)

        print(f"[NV_KlingEditVideo] Final prompt ({len(prompt)} chars):\n{prompt}")

        # --- validation ---
        if input_duration_exact < 2.5:
            raise ValueError(
                f"Input video too short: {num_frames} frames at {fps}fps = "
                f"{input_duration_exact:.2f}s (minimum ~3s required)."
            )
        if input_duration_exact > 10.5:
            raise ValueError(
                f"Input video too long: {num_frames} frames at {fps}fps = "
                f"{input_duration_exact:.2f}s (maximum ~10s allowed)."
            )
        if w > 2160 or h > 2160:
            raise ValueError(
                f"Input dimensions {w}x{h} too large (maximum 2160x2160)."
            )

        # --- encode images to video for upload ---
        video = InputImpl.VideoFromComponents(
            VideoComponents(
                images=images,
                frame_rate=Fraction(fps),
            )
        )

        # --- reference images (auto-upscale undersized) ---
        image_list: list[OmniParamImage] = []
        if reference_images is not None:
            if num_ref_images > 4:
                raise ValueError(
                    "The maximum number of reference images allowed with a video input is 4."
                )
            ref_h, ref_w = reference_images.shape[1], reference_images.shape[2]
            reference_images = _ensure_min_size(reference_images, min_px=_KLING_REF_MIN_PX)
            new_rh, new_rw = reference_images.shape[1], reference_images.shape[2]
            if (new_rh, new_rw) != (ref_h, ref_w):
                print(f"[NV_KlingEditVideo] Ref images {ref_w}x{ref_h} upscaled to "
                      f"{new_rw}x{new_rh}")
            validate_image_aspect_ratio(reference_images, (1, 2.5), (2.5, 1))
            for url in await upload_images_to_comfyapi(
                cls, reference_images, wait_label="Uploading reference image"
            ):
                image_list.append(OmniParamImage(image_url=url))

        # --- video upload ---
        upload_label = "Uploading reference video" if is_feature else "Uploading base video"
        video_url = await upload_video_to_comfyapi(
            cls, video, wait_label=upload_label
        )
        video_list = [
            OmniParamVideo(
                video_url=video_url,
                refer_type=api_refer_type,
                keep_original_sound="yes" if keep_original_sound else "no",
            )
        ]

        # --- build request (mode-dependent) ---
        api_sound = "on" if generate_sound else None
        api_mode = "pro" if resolution == "1080p" else "std"

        if is_feature:
            # Reference mode: user controls aspect_ratio and duration
            api_aspect = aspect_ratio if aspect_ratio != "auto" else _infer_aspect_ratio(w, h)
            api_duration_str = str(api_duration)
        else:
            # Edit mode: API infers aspect_ratio and duration from input video
            api_aspect = None
            api_duration_str = None

        request_data = OmniProReferences2VideoRequest(
            model_name=model_name,
            prompt=prompt,
            aspect_ratio=api_aspect,
            duration=api_duration_str,
            image_list=image_list if image_list else None,
            video_list=video_list,
            mode=api_mode,
            sound=api_sound,
        )

        t_submit = time.time()

        # --- submit to API ---
        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/kling/v1/videos/omni-video", method="POST"),
            response_model=TaskStatusResponse,
            data=request_data,
        )

        if response.code:
            raise RuntimeError(
                f"Kling request failed. Code: {response.code}, "
                f"Message: {response.message}, Data: {response.data}"
            )

        task_id = response.data.task_id

        # --- poll for completion ---
        final = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/kling/v1/videos/omni-video/{task_id}"),
            response_model=TaskStatusResponse,
            status_extractor=lambda r: (r.data.task_status if r.data else None),
        )

        t_done = time.time()

        # --- download result video and extract frames ---
        result_video_data = final.data.task_result.videos[0]
        result_video = await download_url_to_video_output(result_video_data.url)
        components = result_video.get_components()
        output_images = components.images
        output_fps = float(components.frame_rate)
        output_frames = output_images.shape[0]
        out_h, out_w = output_images.shape[1], output_images.shape[2]

        t_end = time.time()

        # --- build metadata ---
        metadata = {
            "input": {
                "frames": num_frames,
                "fps": fps,
                "resolution": f"{w}x{h}",
                "duration_exact": round(input_duration_exact, 3),
            },
            "api_request": {
                "endpoint": "/proxy/kling/v1/videos/omni-video",
                "model_name": model_name,
                "mode": api_mode,
                "refer_type": api_refer_type,
                "duration": api_duration_str,
                "duration_calculated": str(api_duration),
                "duration_mode": duration_mode,
                "aspect_ratio": api_aspect,
                "sound": api_sound,
                "keep_original_sound": "yes" if keep_original_sound else "no",
                "prompt_length": len(prompt),
                "prompt_had_auto_tags": prompt != _normalize_omni_prompt_references(original_prompt),
                "negative_prompt_length": len(negative_prompt.strip()) if negative_prompt else 0,
                "reference_images": len(image_list),
            },
            "api_response": {
                "task_id": task_id,
                "task_status": final.data.task_status if final.data else None,
                "result_duration": result_video_data.duration,
                "result_video_id": result_video_data.id,
            },
            "output": {
                "frames": output_frames,
                "fps": output_fps,
                "resolution": f"{out_w}x{out_h}",
                "duration_exact": round(output_frames / output_fps, 3) if output_fps > 0 else None,
            },
            "timing": {
                "upload_sec": round(t_submit - t_start, 1),
                "api_processing_sec": round(t_done - t_submit, 1),
                "download_sec": round(t_end - t_done, 1),
                "total_sec": round(t_end - t_start, 1),
            },
        }

        return IO.NodeOutput(
            output_images,
            output_fps,
            output_frames,
            json.dumps(metadata, indent=2),
        )


_KLING_REF_MIN_PX = 300


def _ensure_min_size(img: "torch.Tensor", min_px: int = _KLING_REF_MIN_PX) -> "torch.Tensor":
    """Upscale an IMAGE tensor [B,H,W,C] so both sides are >= min_px.

    Preserves aspect ratio, no crop. Returns the tensor unchanged if already large enough.
    """
    import torch.nn.functional as F

    h, w = img.shape[1], img.shape[2]
    if h >= min_px and w >= min_px:
        return img

    scale = max(min_px / h, min_px / w)
    new_h = math.ceil(h * scale)
    new_w = math.ceil(w * scale)

    # [B,H,W,C] → [B,C,H,W] for interpolate → back
    x = img.permute(0, 3, 1, 2)
    x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
    return x.permute(0, 2, 3, 1)


def _build_alias_legend(
    aliases: list[str | None],
    connected: list[bool],
) -> str:
    """Build a reference legend from aliases for connected image slots.

    Returns a string like:
      [References: @image1 = side profile of character, @image2 = lighting ref]
    or empty string if no aliases are provided.
    """
    parts = []
    for i, (alias, is_connected) in enumerate(zip(aliases, connected), 1):
        if is_connected and alias and alias.strip():
            parts.append(f"@image{i} = {alias.strip()}")
    if not parts:
        return ""
    return f"[References: {', '.join(parts)}]"


class NV_KlingPromptPreview(IO.ComfyNode):
    """Preview the final Kling prompt and batch reference images in one node.

    Combines prompt preview (auto-tags, aliases, negative prompt, normalization)
    with explicit @image1–@image4 reference image batching. Outputs the batched
    images ready to wire into NV Kling Edit Video, plus the fully-processed
    prompt text — all without making an API call.
    """

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="NV_KlingPromptPreview",
            display_name="NV Kling Prompt Preview",
            category="NV_Utils/api",
            description=(
                "Preview the fully-processed Kling prompt and batch reference "
                "images with explicit @image1–@image4 slot mapping. "
                "No API call — iterate on prompts for free."
            ),
            inputs=[
                IO.Combo.Input(
                    "refer_type",
                    options=["edit (base)", "reference (feature)"],
                    tooltip="Same as NV Kling Edit Video — affects @video auto-injection.",
                ),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip=(
                        "Text prompt. Use @image1–@image4 to reference specific images, "
                        "or leave tags out and they'll be auto-injected.\n"
                        "If aliases are set, a [References: ...] legend is prepended."
                    ),
                ),
                IO.String.Input(
                    "negative_prompt",
                    multiline=True,
                    default="",
                    tooltip="Appended as 'Avoid: {text}'. Leave empty to skip.",
                    optional=True,
                ),
                # --- Reference images (explicit slot → tag mapping) ---
                IO.Image.Input(
                    "image_1",
                    tooltip="Reference image for @image1.",
                    optional=True,
                ),
                IO.String.Input(
                    "alias_1",
                    default="",
                    tooltip="Describe what @image1 represents (e.g. 'side profile of character').",
                    optional=True,
                ),
                IO.Image.Input(
                    "image_2",
                    tooltip="Reference image for @image2.",
                    optional=True,
                ),
                IO.String.Input(
                    "alias_2",
                    default="",
                    tooltip="Describe what @image2 represents (e.g. 'lighting reference').",
                    optional=True,
                ),
                IO.Image.Input(
                    "image_3",
                    tooltip="Reference image for @image3.",
                    optional=True,
                ),
                IO.String.Input(
                    "alias_3",
                    default="",
                    tooltip="Describe what @image3 represents.",
                    optional=True,
                ),
                IO.Image.Input(
                    "image_4",
                    tooltip="Reference image for @image4.",
                    optional=True,
                ),
                IO.String.Input(
                    "alias_4",
                    default="",
                    tooltip="Describe what @image4 represents.",
                    optional=True,
                ),
            ],
            outputs=[
                IO.Image.Output(display_name="images"),
                IO.Int.Output(display_name="count"),
                IO.String.Output(display_name="final_prompt"),
                IO.Int.Output(display_name="char_count"),
            ],
        )

    @classmethod
    def execute(
        cls,
        refer_type: str,
        prompt: str,
        negative_prompt: str = "",
        image_1: Input.Image | None = None,
        alias_1: str = "",
        image_2: Input.Image | None = None,
        alias_2: str = "",
        image_3: Input.Image | None = None,
        alias_3: str = "",
        image_4: Input.Image | None = None,
        alias_4: str = "",
    ) -> IO.NodeOutput:
        import torch

        is_feature = refer_type == "reference (feature)"

        # --- batch reference images ---
        slots = [image_1, image_2, image_3, image_4]
        aliases = [alias_1, alias_2, alias_3, alias_4]
        connected = [img is not None for img in slots]
        images = [img for img in slots if img is not None]
        num_ref = len(images)

        if images:
            # Log mapping + auto-upscale undersized images
            tag_map = []
            single_frames = []
            for i, (img, alias) in enumerate(zip(slots, aliases), 1):
                if img is not None:
                    frame = img[0:1]  # first frame only
                    h, w = frame.shape[1], frame.shape[2]
                    frame = _ensure_min_size(frame)
                    new_h, new_w = frame.shape[1], frame.shape[2]

                    label = f' "{alias.strip()}"' if alias and alias.strip() else ""
                    if (new_h, new_w) != (h, w):
                        tag_map.append(f"  @image{i}{label} -> {w}x{h} (upscaled to {new_w}x{new_h})")
                    else:
                        tag_map.append(f"  @image{i}{label} -> {w}x{h}")
                    single_frames.append(frame)
            print(f"[NV_KlingPromptPreview] {num_ref} ref image(s):\n" + "\n".join(tag_map))

            batched = torch.cat(single_frames, dim=0)
        else:
            # No images — output a 1x1 black placeholder (won't be wired anywhere)
            batched = torch.zeros(1, 1, 1, 3)

        # --- prompt pipeline ---
        # 1. Alias legend (if any aliases provided for connected slots)
        legend = _build_alias_legend(aliases, connected)
        if legend:
            result = f"{legend}\n{prompt}"
        else:
            result = prompt

        # 2. Auto-inject tags (skipped if legend already contains @imageN tokens)
        result = _auto_inject_reference_tags(result, num_ref, is_feature)

        # 3. Negative prompt
        if negative_prompt and negative_prompt.strip():
            result = f"{result}\nAvoid: {negative_prompt.strip()}"

        # 4. Normalize to API wire format
        result = _normalize_omni_prompt_references(result)

        print(f"[NV_KlingPromptPreview] Final prompt ({len(result)} chars):\n{result}")

        return IO.NodeOutput(batched, num_ref, result, len(result))


NODE_CLASS_MAPPINGS = {
    "NV_KlingEditVideo": NV_KlingEditVideo,
    "NV_KlingPromptPreview": NV_KlingPromptPreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_KlingEditVideo": "NV Kling Edit Video",
    "NV_KlingPromptPreview": "NV Kling Prompt Preview",
}
