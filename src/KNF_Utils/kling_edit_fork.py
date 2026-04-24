"""NV Kling Edit Video (Fork)

Two-node architecture for Kling 3.0 Omni Edit/V2V:

  NV_KlingUploadPreview — unified preprocessing + preview
    All conditioning variables live here: images, fps, upload_fps, prompt,
    negative prompt, reference images (with aliases), duration, aspect ratio,
    refer_type. Does auto-upscale, fps encoding resolution, prompt pipeline,
    ref image batching, validation. Outputs everything pre-processed + a
    KLING_UPLOAD_CONFIG dict for the API node.

  NV_KlingEditVideo — slim API caller
    Takes pre-processed images + config from the preview node, encodes to
    video, uploads, calls the Kling API, downloads result, returns frames.

Uses the same /proxy/kling/v1/videos/omni-video endpoint and
OmniProReferences2VideoRequest model as the built-in Kling nodes.
"""

import json
import math
import re
import time

from fractions import Fraction

from comfy_api.latest import IO, Input, InputImpl
from comfy_api.latest._io import Custom as _IOCustom
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
    poll_op,
    sync_op,
    upload_images_to_comfyapi,
    upload_video_to_comfyapi,
    validate_image_aspect_ratio,
    validate_string,
)


# ---------------------------------------------------------------------------
# Custom type for preview → API node config handoff
# ---------------------------------------------------------------------------

KLING_UPLOAD_CONFIG = _IOCustom("KLING_UPLOAD_CONFIG")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_KLING_REF_MIN_PX = 300
_KLING_VIDEO_MIN_PX = 720
_KLING_VIDEO_MAX_PX = 2160


def _fit_to_kling(img, min_px=_KLING_VIDEO_MIN_PX, max_px=_KLING_VIDEO_MAX_PX):
    """Resize IMAGE tensor [B,H,W,C] so both sides are in [min_px, max_px] and even.

    Preserves aspect ratio.  Handles small InpaintCrop outputs (e.g. 384x512)
    by upscaling, and large inputs by downscaling.  Ensures even dimensions
    (required by most video codecs / APIs).

    Returns (img, did_resize) — did_resize is True if dimensions changed.
    """
    import torch.nn.functional as F

    h, w = img.shape[1], img.shape[2]

    # Compute scale to bring smallest side to min_px
    scale = 1.0
    if h < min_px or w < min_px:
        scale = max(min_px / h, min_px / w)

    new_h = round(h * scale)
    new_w = round(w * scale)

    # Clamp if largest side exceeds max
    if new_h > max_px or new_w > max_px:
        down = min(max_px / new_h, max_px / new_w)
        new_h = round(new_h * down)
        new_w = round(new_w * down)

    # Snap to even (video codec requirement)
    new_h = new_h + (new_h % 2)
    new_w = new_w + (new_w % 2)

    # Final clamp
    new_h = max(min_px, min(max_px, new_h))
    new_w = max(min_px, min(max_px, new_w))

    did_resize = (new_h != h or new_w != w)
    if did_resize:
        x = img.permute(0, 3, 1, 2)
        x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
        img = x.permute(0, 2, 3, 1)

    return img, did_resize


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
    """Auto-prepend @image/@video tags if user didn't include them manually."""
    if not prompt:
        return prompt

    has_image_tags = bool(re.search(r"(?:@image|<<<image_)\d*", prompt, re.IGNORECASE))
    has_video_tags = bool(re.search(r"(?:@video|<<<video_)\d*", prompt, re.IGNORECASE))

    parts = []
    if is_feature and not has_video_tags:
        parts.append("@video")
    if num_ref_images > 0 and not has_image_tags:
        for i in range(1, num_ref_images + 1):
            parts.append(f"@image{i}")

    if not parts:
        return prompt
    return f"Using {', '.join(parts)}: {prompt}"


def _infer_aspect_ratio(w: int, h: int) -> str:
    """Pick the nearest standard Kling aspect ratio from input dimensions."""
    ratio = w / h
    options = [
        (abs(ratio - 16 / 9), "16:9"),
        (abs(ratio - 9 / 16), "9:16"),
        (abs(ratio - 1.0), "1:1"),
    ]
    return min(options, key=lambda x: x[0])[1]


def _ensure_min_size(img, min_px: int = _KLING_REF_MIN_PX):
    """Upscale IMAGE tensor [B,H,W,C] so both sides >= min_px. Preserves aspect ratio."""
    import torch.nn.functional as F

    h, w = img.shape[1], img.shape[2]
    if h >= min_px and w >= min_px:
        return img

    scale = max(min_px / h, min_px / w)
    new_h = math.ceil(h * scale)
    new_w = math.ceil(w * scale)

    x = img.permute(0, 3, 1, 2)
    x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
    return x.permute(0, 2, 3, 1)


def _snap_to_kling_temporal(num_frames: int) -> int:
    """Return the next valid Kling frame count (8k+1 rule).

    Kling uses 8x temporal compression internally, requiring frame counts
    of the form T = 8k + 1 (i.e., 9, 17, 25, ..., 89, 97, 105, ...).
    If num_frames is already valid, returns it unchanged.
    """
    remainder = (num_frames - 1) % 8
    if remainder == 0:
        return num_frames
    return num_frames + (8 - remainder)


def _build_alias_legend(aliases: list[str | None], connected: list[bool]) -> str:
    """Build a [References: @image1 = desc, ...] legend for connected slots."""
    parts = []
    for i, (alias, is_connected) in enumerate(zip(aliases, connected), 1):
        if is_connected and alias and alias.strip():
            parts.append(f"@image{i} = {alias.strip()}")
    if not parts:
        return ""
    return f"[References: {', '.join(parts)}]"


# ---------------------------------------------------------------------------
# Node 1: Unified Upload Preview
# ---------------------------------------------------------------------------

class NV_KlingUploadPreview(IO.ComfyNode):
    """Unified Kling preprocessing + preview. All conditioning variables live here.

    Does auto-upscale, fps encoding, prompt pipeline, ref image batching,
    validation — and outputs everything pre-processed for the API node.
    No API call, runs instantly.
    """

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="NV_KlingUploadPreview",
            display_name="NV Kling Upload Preview",
            category="NV_Utils/api",
            description=(
                "Unified Kling preprocessing + preview. Processes video frames, "
                "prompt, reference images, and all API parameters. Wire outputs "
                "to NV Kling Edit Video for the actual API call."
            ),
            inputs=[
                IO.Combo.Input(
                    "refer_type",
                    options=["edit (base)", "reference (feature)"],
                    tooltip=(
                        "'edit (base)': Direct video editing — modifies the input video "
                        "per prompt. Duration and aspect ratio sent as None (API infers).\n"
                        "'reference (feature)': Style/motion reference — generates new "
                        "content. Duration and aspect ratio are user-specified."
                    ),
                ),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip=(
                        "Text prompt. Use @image1–@image4 for reference images, "
                        "@video for the input video (auto-injected if not present). "
                        "Max 2500 chars."
                    ),
                ),
                IO.String.Input(
                    "negative_prompt",
                    multiline=True,
                    default="",
                    tooltip="Appended as 'Avoid: {text}'. Leave empty to skip.",
                    optional=True,
                ),
                IO.Image.Input(
                    "images",
                    tooltip="Input video frames [B,H,W,C]. Auto-upscaled if < 720px.",
                ),
                IO.Int.Input(
                    "fps",
                    default=30,
                    min=1,
                    max=60,
                    display_mode=IO.NumberDisplay.number,
                    tooltip=(
                        "Source frame rate of the input images. Used for "
                        "auto-duration calculation (frames / fps = seconds)."
                    ),
                ),
                IO.Combo.Input(
                    "upload_fps",
                    options=["match input", "24 (kling native)", "30"],
                    default="24 (kling native)",
                    tooltip=(
                        "FPS to encode the uploaded video. Kling outputs at "
                        "24fps regardless of input.\n"
                        "'match input': encode at the fps value above.\n"
                        "'24 (kling native)': match Kling's native rate "
                        "(recommended). All frames preserved, duration extends.\n"
                        "'30': encode at 30fps (output may have fewer frames)."
                    ),
                ),
                IO.Combo.Input(
                    "duration_mode",
                    options=["auto", "manual"],
                    tooltip=(
                        "'auto': duration = ceil(num_frames / fps), clamped to max. "
                        "'manual': use the duration slider.\n"
                        "In edit mode, duration is sent as None regardless."
                    ),
                ),
                IO.Int.Input(
                    "duration",
                    default=5,
                    min=3,
                    max=15,
                    display_mode=IO.NumberDisplay.slider,
                    tooltip="Output duration in seconds (manual mode). 3-15s.",
                ),
                IO.Combo.Input(
                    "aspect_ratio",
                    options=["auto", "16:9", "9:16", "1:1"],
                    tooltip=(
                        "In edit mode: ignored (API infers from input).\n"
                        "In reference mode: 'auto' infers from input dimensions."
                    ),
                ),
                # --- Sequential chunking ---
                IO.Boolean.Input(
                    "chunk_mode",
                    default=False,
                    tooltip=(
                        "Enable sequential chunking for long videos. "
                        "Auto-truncates input to ~10s per chunk, extracts tail "
                        "frames from previous chunk as references, and outputs "
                        "next_chunk_start for wiring into the next pass."
                    ),
                ),
                IO.Int.Input(
                    "chunk_start_frame",
                    default=0,
                    min=0,
                    max=100000,
                    display_mode=IO.NumberDisplay.number,
                    tooltip=(
                        "Frame index to start this chunk from. "
                        "First chunk = 0. Subsequent chunks = next_chunk_start "
                        "output from the previous pass."
                    ),
                ),
                IO.Image.Input(
                    "prev_chunk_output",
                    tooltip=(
                        "Output frames from the previous Kling chunk. "
                        "Tail frames are auto-extracted as reference images "
                        "for visual consistency across chunks."
                    ),
                    optional=True,
                ),
                IO.Int.Input(
                    "tail_ref_count",
                    default=1,
                    min=0,
                    max=3,
                    display_mode=IO.NumberDisplay.number,
                    tooltip=(
                        "How many tail frames to extract from prev_chunk_output "
                        "as reference images (fills first unused image_N slots). "
                        "0 = disabled."
                    ),
                ),
                # --- Reference images (explicit slot → tag mapping) ---
                IO.Image.Input("image_1", tooltip="Reference image for @image1.", optional=True),
                IO.String.Input("alias_1", default="", tooltip="Describe @image1.", optional=True),
                IO.Image.Input("image_2", tooltip="Reference image for @image2.", optional=True),
                IO.String.Input("alias_2", default="", tooltip="Describe @image2.", optional=True),
                IO.Image.Input("image_3", tooltip="Reference image for @image3.", optional=True),
                IO.String.Input("alias_3", default="", tooltip="Describe @image3.", optional=True),
                IO.Image.Input("image_4", tooltip="Reference image for @image4.", optional=True),
                IO.String.Input("alias_4", default="", tooltip="Describe @image4.", optional=True),
            ],
            outputs=[
                IO.Image.Output(display_name="images"),
                IO.Image.Output(display_name="ref_preview"),
                IO.String.Output(display_name="final_prompt"),
                KLING_UPLOAD_CONFIG.Output(display_name="upload_config"),
                IO.Int.Output(display_name="ref_count"),
                IO.Int.Output(display_name="char_count"),
                IO.String.Output(display_name="metadata"),
                IO.Int.Output(display_name="next_chunk_start"),
            ],
        )

    @classmethod
    def execute(
        cls,
        refer_type: str,
        prompt: str,
        images: Input.Image,
        fps: int,
        upload_fps: str,
        duration_mode: str,
        duration: int,
        aspect_ratio: str,
        negative_prompt: str = "",
        chunk_mode: bool = False,
        chunk_start_frame: int = 0,
        prev_chunk_output: Input.Image | None = None,
        tail_ref_count: int = 1,
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

        # --- resolve encode fps (once, used by both chunk mode and later) ---
        if upload_fps == "match input":
            encode_fps = fps
        else:
            encode_fps = int(upload_fps.split()[0])

        # --- chunk mode: slice + truncate ---
        total_input_frames = images.shape[0]
        chunk_truncated = False
        next_chunk_start = -1  # -1 = no more chunks

        if chunk_mode:
            # Max frames that fit in ~10s, snapped DOWN to largest 8k+1 value
            max_chunk_frames = ((int(encode_fps * 10) - 1) // 8) * 8 + 1

            # Slice from chunk_start_frame
            if chunk_start_frame > 0:
                if chunk_start_frame >= total_input_frames:
                    raise ValueError(
                        f"chunk_start_frame ({chunk_start_frame}) >= total frames "
                        f"({total_input_frames}). No frames to process."
                    )
                images = images[chunk_start_frame:]
                print(
                    f"[NV_KlingUploadPreview] Chunk mode: sliced from frame "
                    f"{chunk_start_frame}, {images.shape[0]} frames remaining"
                )

            # Truncate if too long
            if images.shape[0] > max_chunk_frames:
                chunk_truncated = True
                next_chunk_start = chunk_start_frame + max_chunk_frames
                images = images[:max_chunk_frames]
                remaining = total_input_frames - next_chunk_start
                print(
                    f"[NV_KlingUploadPreview] Chunk mode: truncated to {max_chunk_frames} "
                    f"frames (~10s at {encode_fps}fps). "
                    f"Next chunk: set chunk_start_frame={next_chunk_start} "
                    f"({remaining} frames remaining)"
                )
            else:
                next_chunk_start = -1
                print(
                    f"[NV_KlingUploadPreview] Chunk mode: {images.shape[0]} frames "
                    f"fits within limit — this is the final chunk."
                )

        # --- auto-fill ref slots from previous chunk tail ---
        auto_filled_slots = []
        if prev_chunk_output is not None and tail_ref_count > 0:
            slots_list = [image_1, image_2, image_3, image_4]
            aliases_list = [alias_1, alias_2, alias_3, alias_4]
            prev_frames = prev_chunk_output.shape[0]
            count = min(tail_ref_count, prev_frames)

            for ti in range(count):
                # Find first unused slot
                free_idx = None
                for si, s in enumerate(slots_list):
                    if s is None:
                        free_idx = si
                        break
                if free_idx is None:
                    print(f"[NV_KlingUploadPreview] All 4 ref slots occupied — skipping tail frame {ti + 1}")
                    break

                # Extract frame from tail (last, second-to-last, etc.)
                frame_idx = prev_frames - count + ti
                tail_frame = prev_chunk_output[frame_idx:frame_idx + 1]
                slots_list[free_idx] = tail_frame
                aliases_list[free_idx] = "previous chunk — maintain visual consistency"
                auto_filled_slots.append(free_idx + 1)  # 1-indexed for @imageN

            image_1, image_2, image_3, image_4 = slots_list
            alias_1, alias_2, alias_3, alias_4 = aliases_list
            print(
                f"[NV_KlingUploadPreview] Sequential chunk: extracted {len(auto_filled_slots)} "
                f"tail frame(s) from prev output ({prev_frames} frames) → "
                f"slot(s) {', '.join(f'@image{i}' for i in auto_filled_slots)}"
            )

        # --- video frame processing ---
        num_frames = images.shape[0]
        input_h, input_w = images.shape[1], images.shape[2]

        images, did_resize = _fit_to_kling(images)
        h, w = images.shape[1], images.shape[2]
        if did_resize:
            print(f"[NV_KlingUploadPreview] Video {input_w}x{input_h} → fitted to {w}x{h} for Kling")

        # --- 8k+1 temporal alignment ---
        # Kling uses 8x temporal compression: valid frame counts are 8k+1.
        # Pad with repeated last frame if needed; trim back in NV_KlingEditVideo.
        original_num_frames = num_frames
        valid_frames = _snap_to_kling_temporal(num_frames)
        if valid_frames != num_frames:
            pad_count = valid_frames - num_frames
            last_frame = images[-1:].expand(pad_count, -1, -1, -1)
            images = torch.cat([images, last_frame], dim=0)
            num_frames = valid_frames
            print(
                f"[NV_KlingUploadPreview] WARNING: Frame count {original_num_frames} is not 8k+1 aligned. "
                f"Padded to {valid_frames} (+{pad_count} repeated last frame). "
                f"Output will be trimmed back to {original_num_frames}. "
                f"Recommend adjusting your source to a valid frame count: "
                f"...{_snap_to_kling_temporal(original_num_frames - 8)}, {original_num_frames - (original_num_frames - 1) % 8}, "
                f"{valid_frames}, {valid_frames + 8}..."
            )

        input_duration_exact = num_frames / fps
        upload_duration_exact = num_frames / encode_fps

        # --- duration calculation ---
        max_duration = 15 if is_feature else 10
        if duration_mode == "auto":
            api_duration_val = max(3, min(max_duration, math.ceil(input_duration_exact)))
        else:
            api_duration_val = min(duration, max_duration)

        if is_feature:
            api_duration = str(api_duration_val)
            api_aspect = aspect_ratio if aspect_ratio != "auto" else _infer_aspect_ratio(w, h)
        else:
            api_duration = None
            api_aspect = None

        # --- reference image batching ---
        slots = [image_1, image_2, image_3, image_4]
        aliases = [alias_1, alias_2, alias_3, alias_4]
        connected = [img is not None for img in slots]
        # The last tail frame (highest index in auto_filled_slots) is closest to
        # this chunk's start — tag it as "first_frame" so Kling anchors continuity.
        last_tail_slot = max(auto_filled_slots) if auto_filled_slots else None

        ref_frames = []
        ref_frame_types: list[str | None] = []  # parallel to ref_frames: OmniParamImage.type
        tag_map = []
        for i, (img, alias) in enumerate(zip(slots, aliases), 1):
            if img is not None:
                frame = img[0:1]  # first frame only
                orig_h, orig_w = frame.shape[1], frame.shape[2]
                frame = _ensure_min_size(frame, min_px=_KLING_REF_MIN_PX)
                new_h, new_w = frame.shape[1], frame.shape[2]

                # Determine OmniParamImage.type for this ref
                if i == last_tail_slot:
                    frame_type = "first_frame"
                else:
                    frame_type = None

                label = f' "{alias.strip()}"' if alias and alias.strip() else ""
                type_tag = f" [type={frame_type}]" if frame_type else ""
                if (new_h, new_w) != (orig_h, orig_w):
                    tag_map.append(f"  @image{i}{label} → {orig_w}x{orig_h} (upscaled to {new_w}x{new_h}){type_tag}")
                else:
                    tag_map.append(f"  @image{i}{label} → {orig_w}x{orig_h}{type_tag}")
                ref_frames.append(frame)
                ref_frame_types.append(frame_type)

        num_ref = len(ref_frames)
        if ref_frames:
            # Resize all refs to matching height for preview (display only)
            target_h = max(f.shape[1] for f in ref_frames)
            preview_list = []
            for f in ref_frames:
                fh, fw = f.shape[1], f.shape[2]
                if fh != target_h:
                    scale = target_h / fh
                    new_w = max(1, round(fw * scale))
                    x = f.permute(0, 3, 1, 2)
                    x = torch.nn.functional.interpolate(
                        x, size=(target_h, new_w), mode="bilinear", align_corners=False
                    )
                    preview_list.append(x.permute(0, 2, 3, 1))
                else:
                    preview_list.append(f)
            # Pad widths to max so they can be batched
            max_w = max(p.shape[2] for p in preview_list)
            batched = []
            for p in preview_list:
                pw = p.shape[2]
                if pw != max_w:
                    pad = torch.zeros(1, target_h, max_w, 3)
                    x_off = (max_w - pw) // 2
                    pad[:, :, x_off:x_off + pw, :] = p
                    batched.append(pad)
                else:
                    batched.append(p)
            preview_refs = torch.cat(batched, dim=0)
            print(f"[NV_KlingUploadPreview] {num_ref} ref image(s):\n" + "\n".join(tag_map))
        else:
            preview_refs = torch.zeros(1, 1, 1, 3)

        # --- prompt pipeline ---
        original_prompt = prompt

        legend = _build_alias_legend(aliases, connected)
        if legend:
            prompt = f"{legend}\n{prompt}"

        prompt = _auto_inject_reference_tags(prompt, num_ref, is_feature)

        if negative_prompt and negative_prompt.strip():
            prompt = f"{prompt}\nAvoid: {negative_prompt.strip()}"

        # Auto-append consistency hint for sequential chunks
        if auto_filled_slots:
            refs = ", ".join(f"@image{i}" for i in auto_filled_slots)
            prompt = f"{prompt}\nMaintain visual consistency with {refs}."

        prompt = _normalize_omni_prompt_references(prompt)

        # --- validation ---
        validate_string(prompt, min_length=1, max_length=2500)

        if num_ref > 4:
            raise ValueError("Maximum 4 reference images allowed.")
        if upload_duration_exact < 2.5:
            raise ValueError(
                f"Upload video too short: {num_frames} frames at {encode_fps}fps = "
                f"{upload_duration_exact:.2f}s (minimum ~3s)."
            )
        if upload_duration_exact > 10.5 and not chunk_mode:
            raise ValueError(
                f"Upload video too long: {num_frames} frames at {encode_fps}fps = "
                f"{upload_duration_exact:.2f}s. Base refer_type caps at 10s. Options: "
                f"(a) switch refer_type to 'reference (feature)' for 15s cap; "
                f"(b) enable chunk_mode to auto-truncate; "
                f"(c) shorten source to ≤240 frames at 24fps (or 8k+1 aligned: 233/241)."
            )
        if w > 2160 or h > 2160:
            raise ValueError(f"Dimensions {w}x{h} too large (maximum 2160x2160).")

        if ref_frames:
            validate_image_aspect_ratio(preview_refs, (1, 2.5), (2.5, 1))

        # --- console summary ---
        print(f"[NV_KlingUploadPreview] Upload: {encode_fps}fps, "
              f"{num_frames} frames = {upload_duration_exact:.2f}s "
              f"(source: {fps}fps = {input_duration_exact:.2f}s)")
        print(f"[NV_KlingUploadPreview] Final prompt ({len(prompt)} chars):\n{prompt}")

        # --- build config dict for API node ---
        config = {
            "encode_fps": encode_fps,
            "api_refer_type": "feature" if is_feature else "base",
            "api_duration": api_duration,
            "api_aspect_ratio": api_aspect,
            "num_ref_images": num_ref,
            "ref_frames": ref_frames,  # original tensors at native resolution
            "ref_frame_types": ref_frame_types,  # parallel list: OmniParamImage.type per ref
            "input_resolution": (input_w, input_h),  # pre-Kling-fit crop resolution
            "original_num_frames": original_num_frames,  # pre-padding count for trim
        }

        # --- build metadata preview ---
        metadata = {
            "input": {
                "frames": num_frames,
                "original_frames": original_num_frames,
                "padded": num_frames != original_num_frames,
                "fps": fps,
                "upload_fps": encode_fps,
                "original_resolution": f"{input_w}x{input_h}",
                "upload_resolution": f"{w}x{h}",
                "duration_exact": round(input_duration_exact, 3),
                "upload_duration_exact": round(upload_duration_exact, 3),
            },
            "api_params": {
                "refer_type": config["api_refer_type"],
                "duration": api_duration,
                "duration_calculated": str(api_duration_val),
                "duration_mode": duration_mode,
                "aspect_ratio": api_aspect,
                "prompt_length": len(prompt),
                "prompt_had_auto_tags": prompt != _normalize_omni_prompt_references(original_prompt),
                "negative_prompt_length": len(negative_prompt.strip()) if negative_prompt else 0,
                "reference_images": num_ref,
            },
        }

        if chunk_mode:
            chunk_info = {
                "total_input_frames": total_input_frames,
                "chunk_start_frame": chunk_start_frame,
                "chunk_frames": images.shape[0],
                "truncated": chunk_truncated,
                "next_chunk_start": next_chunk_start,
                "is_final_chunk": next_chunk_start == -1,
            }
            if next_chunk_start > 0:
                remaining = total_input_frames - next_chunk_start
                chunk_info["remaining_frames"] = remaining
                chunk_info["estimated_remaining_chunks"] = math.ceil(remaining / images.shape[0])
                chunk_info["guidance"] = (
                    f"Set chunk_start_frame={next_chunk_start} for the next pass. "
                    f"{remaining} frames remaining (~{chunk_info['estimated_remaining_chunks']} more chunk(s))."
                )
            else:
                chunk_info["guidance"] = "This is the final chunk — no more passes needed."
            metadata["chunk"] = chunk_info

        if auto_filled_slots:
            metadata.setdefault("chunk", {})["tail_refs"] = {
                "prev_chunk_frames": prev_chunk_output.shape[0],
                "tail_refs_extracted": len(auto_filled_slots),
                "auto_filled_slots": [f"@image{i}" for i in auto_filled_slots],
            }

        if original_num_frames != num_frames:
            prev_valid = original_num_frames - ((original_num_frames - 1) % 8)
            next_valid = valid_frames
            metadata["warning"] = (
                f"Frame count {original_num_frames} is not Kling-aligned (8k+1). "
                f"Padded to {num_frames}, output will be auto-trimmed to {original_num_frames}. "
                f"To avoid padding, use a frame count like {prev_valid}, {next_valid}, or {next_valid + 8}."
            )

        return IO.NodeOutput(
            images,
            preview_refs,
            prompt,
            config,
            num_ref,
            len(prompt),
            json.dumps(metadata, indent=2),
            next_chunk_start,
        )


# ---------------------------------------------------------------------------
# Node 2: Slim API Caller
# ---------------------------------------------------------------------------

class NV_KlingEditVideo(IO.ComfyNode):
    """Kling 3.0 Omni Edit Video — slim API caller.

    Takes pre-processed images + config from NV_KlingUploadPreview,
    encodes to video, uploads, calls the API, downloads the result.
    """

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="NV_KlingEditVideo",
            display_name="NV Kling Edit Video",
            category="NV_Utils/api",
            description=(
                "Slim Kling API caller. Wire from NV Kling Upload Preview — "
                "encodes video, uploads, calls API, downloads result frames."
            ),
            inputs=[
                IO.Combo.Input(
                    "model_name",
                    options=["kling-v3-omni", "kling-video-o1"],
                    tooltip="Kling model to use.",
                ),
                IO.Image.Input(
                    "images",
                    tooltip="Pre-processed video frames from NV Kling Upload Preview.",
                ),
                KLING_UPLOAD_CONFIG.Input(
                    "upload_config",
                    tooltip="Config from NV Kling Upload Preview.",
                ),
                IO.String.Input(
                    "final_prompt",
                    tooltip="Fully processed prompt from NV Kling Upload Preview.",
                    force_input=True,
                ),
                IO.Combo.Input(
                    "resolution",
                    options=["1080p", "720p"],
                    tooltip="'1080p' = pro mode ($0.168/s), '720p' = std mode ($0.126/s).",
                ),
                IO.Boolean.Input(
                    "keep_original_sound",
                    default=True,
                    tooltip="Preserve audio from the input video.",
                ),
                IO.Boolean.Input(
                    "generate_sound",
                    default=False,
                    tooltip="Enable Kling's AI audio generation (sound='on').",
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
            ],
            outputs=[
                IO.Image.Output(display_name="images"),
                IO.Float.Output(display_name="output_fps"),
                IO.Int.Output(display_name="output_frames"),
                IO.String.Output(display_name="api_metadata"),
                IO.Float.Output(display_name="estimated_credits"),
                IO.Float.Output(display_name="estimated_usd"),
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
        images: Input.Image,
        upload_config: dict,
        final_prompt: str,
        resolution: str,
        keep_original_sound: bool,
        generate_sound: bool,
        seed: int = 0,
    ) -> IO.NodeOutput:
        _ = seed
        t_start = time.time()

        # --- unpack config from preview node ---
        encode_fps = upload_config["encode_fps"]
        api_refer_type = upload_config["api_refer_type"]
        api_duration_str = upload_config["api_duration"]
        api_aspect = upload_config["api_aspect_ratio"]
        num_ref_images = upload_config["num_ref_images"]

        is_feature = api_refer_type == "feature"
        api_mode = "pro" if resolution == "1080p" else "std"
        api_sound = "on" if generate_sound else None

        num_frames = images.shape[0]
        h, w = images.shape[1], images.shape[2]

        # --- cost estimate (Kling Omni Edit pricing) ---
        # Rates from comfy_api_nodes/nodes_kling.py:1500-1509 (OmniProEditVideoNode price_badge).
        # Comfy USD→credits ratio is 211 (comfy_api_nodes/util/client.py:443).
        # Kling Edit bills on the input video duration (output length == input length).
        billed_seconds = num_frames / encode_fps if encode_fps > 0 else 0.0
        usd_per_sec = 0.168 if api_mode == "pro" else 0.126
        estimated_usd = round(billed_seconds * usd_per_sec, 4)
        estimated_credits = round(estimated_usd * 211, 1)

        print(f"[NV_KlingEditVideo] Encoding {num_frames} frames at {encode_fps}fps, "
              f"{w}x{h}, mode={api_mode}, refer_type={api_refer_type}")
        print(f"[NV_KlingEditVideo] Cost estimate: {billed_seconds:.2f}s × ${usd_per_sec}/s "
              f"= ${estimated_usd} ≈ {estimated_credits} credits ({api_mode} mode)")

        # --- encode images to video ---
        video = InputImpl.VideoFromComponents(
            VideoComponents(
                images=images,
                frame_rate=Fraction(encode_fps),
            )
        )

        # --- upload reference images (from config, native resolution each) ---
        image_list: list[OmniParamImage] = []
        ref_frames = upload_config.get("ref_frames", [])
        ref_frame_types = upload_config.get("ref_frame_types")
        if not isinstance(ref_frame_types, list):
            ref_frame_types = [None] * len(ref_frames)
        for idx, ref_frame in enumerate(ref_frames):
            frame_type = ref_frame_types[idx] if idx < len(ref_frame_types) else None
            for url in await upload_images_to_comfyapi(
                cls, ref_frame, wait_label="Uploading reference image"
            ):
                image_list.append(OmniParamImage(image_url=url, type=frame_type))
                print(f"[NV_KlingEditVideo] Ref image {idx + 1}: type={frame_type!r}, url=...{url[-40:]}")

        # --- upload video ---
        upload_label = "Uploading reference video" if is_feature else "Uploading base video"
        video_url = await upload_video_to_comfyapi(cls, video, wait_label=upload_label)
        video_list = [
            OmniParamVideo(
                video_url=video_url,
                refer_type=api_refer_type,
                keep_original_sound="yes" if keep_original_sound else "no",
            )
        ]

        # --- build request ---
        request_data = OmniProReferences2VideoRequest(
            model_name=model_name,
            prompt=final_prompt,
            aspect_ratio=api_aspect,
            duration=api_duration_str,
            image_list=image_list if image_list else None,
            video_list=video_list,
            mode=api_mode,
            sound=api_sound,
        )

        # --- debug: log the image_list types being sent ---
        type_summary = [f"@image{i+1}:type={img.type!r}" for i, img in enumerate(image_list)]
        print(f"[NV_KlingEditVideo] image_list types → [{', '.join(type_summary) or 'empty'}]")
        print(f"[NV_KlingEditVideo] video_list → refer_type={api_refer_type!r}, "
              f"has_video={bool(video_list)}, has_images={bool(image_list)}")

        t_submit = time.time()

        # --- submit ---
        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/kling/v1/videos/omni-video", method="POST"),
            response_model=TaskStatusResponse,
            data=request_data,
        )

        # --- debug: log raw API response ---
        print(f"[NV_KlingEditVideo] API response → code={response.code!r}, "
              f"message={response.message!r}, "
              f"task_status={response.data.task_status if response.data else 'N/A'}")

        # --- billing-field scout: dump raw payload to look for credits/cost fields ---
        # If the Kling proxy ever exposes a billing field in the response, we can wire
        # it into a price_extractor on sync_op/poll_op to override the estimate with
        # the actual deduction. Until then, this print verifies what's in the payload.
        try:
            raw_dump = response.model_dump()
            print(f"[NV_KlingEditVideo] raw response keys: {sorted(raw_dump.keys())}")
            if raw_dump.get("data"):
                print(f"[NV_KlingEditVideo] raw data keys: {sorted(raw_dump['data'].keys())}")
        except Exception as _e:
            print(f"[NV_KlingEditVideo] raw payload dump failed: {_e}")

        if response.code:
            raise RuntimeError(
                f"Kling request failed. Code: {response.code}, "
                f"Message: {response.message}, Data: {response.data}"
            )

        task_id = response.data.task_id

        # --- poll ---
        final = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/kling/v1/videos/omni-video/{task_id}"),
            response_model=TaskStatusResponse,
            status_extractor=lambda r: (r.data.task_status if r.data else None),
        )

        t_done = time.time()

        # --- download result ---
        result_video_data = final.data.task_result.videos[0]
        result_video = await download_url_to_video_output(result_video_data.url)
        components = result_video.get_components()
        output_images = components.images
        output_fps = float(components.frame_rate)
        raw_output_frames = output_images.shape[0]
        output_frames = raw_output_frames
        out_h, out_w = output_images.shape[1], output_images.shape[2]

        # --- trim padded frames back to original count ---
        original_num_frames = upload_config.get("original_num_frames", num_frames)
        trimmed = False
        if output_frames > original_num_frames:
            output_images = output_images[:original_num_frames]
            trimmed = True
            print(
                f"[NV_KlingEditVideo] Trimmed {output_frames} → {original_num_frames} frames "
                f"(removing {output_frames - original_num_frames} padded frames)"
            )
            output_frames = original_num_frames

        t_end = time.time()

        # --- metadata ---
        metadata = {
            "input": {
                "frames": num_frames,
                "encode_fps": encode_fps,
                "resolution": f"{w}x{h}",
            },
            "api_request": {
                "endpoint": "/proxy/kling/v1/videos/omni-video",
                "model_name": model_name,
                "mode": api_mode,
                "refer_type": api_refer_type,
                "duration": api_duration_str,
                "aspect_ratio": api_aspect,
                "sound": api_sound,
                "keep_original_sound": "yes" if keep_original_sound else "no",
                "prompt_length": len(final_prompt),
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
                "frames_from_api": raw_output_frames,
                "trimmed": trimmed,
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
            "cost_estimate": {
                "billed_seconds": round(billed_seconds, 3),
                "mode": api_mode,
                "usd_per_sec": usd_per_sec,
                "estimated_usd": estimated_usd,
                "estimated_credits": estimated_credits,
                "credits_per_usd": 211,
                "note": (
                    "Local estimate from price_badge rates in nodes_kling.py:1500-1509. "
                    "Actual deduction may differ — proxy server is the source of truth."
                ),
            },
        }

        return IO.NodeOutput(
            output_images,
            output_fps,
            output_frames,
            json.dumps(metadata, indent=2),
            estimated_credits,
            estimated_usd,
        )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "NV_KlingUploadPreview": NV_KlingUploadPreview,
    "NV_KlingEditVideo": NV_KlingEditVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_KlingUploadPreview": "NV Kling Upload Preview",
    "NV_KlingEditVideo": "NV Kling Edit Video",
}
