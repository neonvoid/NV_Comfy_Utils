"""NV Wan 2.7 Video Edit Prep — preprocessing + upload for Wan 2.7 Video Edit API.

Mirrors the Seedance fork pattern: uploads input video + optional reference
images, emits a WAN27_UPLOAD_CONFIG for the slim API caller. Accepts either
an already-encoded VIDEO or a batched IMAGE frames tensor + fps (auto-encoded
via VideoFromComponents).

Calls the same BytePlus/DashScope Wan 2.7 endpoint as the stock node and
reuses stock request/response helpers.
"""

from __future__ import annotations

import json
from fractions import Fraction

import torch
import torch.nn.functional as F
from comfy_api.latest import IO, Input, InputImpl
from comfy_api.latest._io import Custom as _IOCustom
from comfy_api.latest._util.video_types import VideoComponents
from comfy_api_nodes.util import (
    upload_image_to_comfyapi,
    upload_video_to_comfyapi,
)


# ---------------------------------------------------------------------------
# Shared custom type
# ---------------------------------------------------------------------------

WAN27_UPLOAD_CONFIG = _IOCustom("WAN27_UPLOAD_CONFIG")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_REF_IMAGES = 4  # Wan 2.7 Video Edit Autogrow cap in stock node


# ---------------------------------------------------------------------------
# Helpers (lifted from seedance_prep pattern — kept independent to avoid
# cross-fork coupling)
# ---------------------------------------------------------------------------

def _uniform_sample_indices(total: int, take: int) -> list[int]:
    if take >= total:
        return list(range(total))
    if take == 1:
        return [0]
    step = (total - 1) / (take - 1)
    return [round(i * step) for i in range(take)]


def _sample_image_batch(images: torch.Tensor, mode: str, max_images: int) -> tuple[list[torch.Tensor], str]:
    """Slice batched IMAGE [N,H,W,C] → list of [1,H,W,C] frames with sampling mode."""
    if images is None or images.shape[0] == 0:
        return [], "no images"
    n = images.shape[0]
    if n <= max_images:
        return [images[i:i + 1] for i in range(n)], f"{n}/{n} (all)"
    if mode == "first":
        return [images[i:i + 1] for i in range(max_images)], f"first {max_images}/{n}"
    if mode == "last":
        return [images[i:i + 1] for i in range(n - max_images, n)], f"last {max_images}/{n}"
    idx = _uniform_sample_indices(n, max_images)
    return [images[i:i + 1] for i in idx], f"uniform {max_images}/{n} (idx {idx})"


def _even_dims(h: int, w: int) -> tuple[int, int]:
    return h + (h % 2), w + (w % 2)


def _snap_even(frames: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
    """Snap [F,H,W,C] frames to even H,W (codec requirement)."""
    h, w = frames.shape[1], frames.shape[2]
    new_h, new_w = _even_dims(h, w)
    if (new_h, new_w) == (h, w):
        return frames, (h, w)
    x = frames.permute(0, 3, 1, 2)
    x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
    return x.permute(0, 2, 3, 1), (new_h, new_w)


def _build_preview_grid(ref_frames: list[torch.Tensor], video_first_frame: torch.Tensor | None) -> torch.Tensor:
    """Concat ref images + video first frame into a single [N,H,W,C] preview batch."""
    tiles: list[torch.Tensor] = list(ref_frames)
    if video_first_frame is not None:
        tiles.append(video_first_frame)

    if not tiles:
        return torch.zeros(1, 64, 64, 3)

    target_h = max(t.shape[1] for t in tiles)
    resized: list[torch.Tensor] = []
    for t in tiles:
        h, w = t.shape[1], t.shape[2]
        if h != target_h:
            scale = target_h / h
            new_w = max(1, round(w * scale))
            x = t.permute(0, 3, 1, 2)
            x = F.interpolate(x, size=(target_h, new_w), mode="bilinear", align_corners=False)
            resized.append(x.permute(0, 2, 3, 1))
        else:
            resized.append(t)

    max_w = max(r.shape[2] for r in resized)
    batched: list[torch.Tensor] = []
    for r in resized:
        rw = r.shape[2]
        if rw != max_w:
            pad = torch.zeros(1, target_h, max_w, 3)
            x_off = (max_w - rw) // 2
            pad[:, :, x_off:x_off + rw, :] = r
            batched.append(pad)
        else:
            batched.append(r)

    return torch.cat(batched, dim=0)


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class NV_Wan27VideoEditPrep(IO.ComfyNode):
    """Wan 2.7 Video Edit preprocessing + upload. Emits WAN27_UPLOAD_CONFIG.

    Accepts the input video (either as VIDEO or frames + fps) and up to 4
    reference images as a batched IMAGE tensor. Validates duration (2-10s),
    uploads assets to comfy API, emits a config for the slim
    NV_Wan27VideoEdit caller. The draft prompt passes through unchanged so
    users can route it through NV_PromptRefiner (mode=wan27_edit) before the
    API node.
    """

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="NV_Wan27VideoEditPrep",
            display_name="NV Wan 2.7 Video Edit Prep",
            category="NV_Utils/api",
            description=(
                "Upload + preview for Wan 2.7 Video Edit. Accepts input video "
                "(or frames+fps) and optional reference images. Emits "
                "WAN27_UPLOAD_CONFIG for NV Wan 2.7 Video Edit."
            ),
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip=(
                        "Draft edit instruction. Wire through NV Prompt Refiner "
                        "(mode=wan27_edit) to produce the final_prompt for the API "
                        "node. Passed through unchanged by this node."
                    ),
                ),
                IO.Video.Input(
                    "input_video",
                    tooltip=(
                        "The video to edit. 2-10 seconds. Takes precedence over "
                        "input_video_frames if both wired."
                    ),
                    optional=True,
                ),
                IO.Image.Input(
                    "input_video_frames",
                    tooltip=(
                        "Frames tensor [F,H,W,C] to encode into a VIDEO. Used only "
                        "if input_video is not wired. Audio track will be empty — "
                        "audio_setting='origin' will be coerced to 'auto' downstream."
                    ),
                    optional=True,
                ),
                IO.Float.Input(
                    "input_video_fps",
                    default=24.0,
                    min=8.0,
                    max=60.0,
                    step=1.0,
                    tooltip="FPS for encoding input_video_frames → VIDEO.",
                ),
                IO.Image.Input(
                    "reference_images",
                    tooltip=(
                        "Batched IMAGE tensor [N,H,W,C]. Up to 4 images — if N>4, "
                        "sampled per sample_mode. Each becomes a "
                        "Wan27MediaItem(type='reference_image')."
                    ),
                    optional=True,
                ),
                IO.Combo.Input(
                    "sample_mode",
                    options=["uniform", "first", "last"],
                    default="uniform",
                    tooltip="How to pick 4 frames when N>4 reference images are provided.",
                ),
            ],
            outputs=[
                WAN27_UPLOAD_CONFIG.Output(display_name="config"),
                IO.Image.Output(display_name="preview"),
                IO.String.Output(display_name="prompt"),
                IO.Float.Output(display_name="input_video_duration_s"),
                IO.String.Output(display_name="info"),
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
        input_video_fps: float,
        sample_mode: str,
        input_video: Input.Video | None = None,
        input_video_frames: Input.Image | None = None,
        reference_images: Input.Image | None = None,
    ) -> IO.NodeOutput:
        # --- probe: log raw input shapes ---
        if reference_images is not None:
            ri_shape = tuple(reference_images.shape)
            print(f"[NV_Wan27VideoEditPrep] reference_images: shape={ri_shape}")
        if input_video_frames is not None:
            rvf_shape = tuple(input_video_frames.shape)
            print(f"[NV_Wan27VideoEditPrep] input_video_frames: shape={rvf_shape}")

        # --- sample reference images ---
        image_frames, image_sampling_note = (
            _sample_image_batch(reference_images, sample_mode, _MAX_REF_IMAGES)
            if reference_images is not None
            else ([], "none")
        )
        n_images = len(image_frames)
        if n_images > 0:
            ref_dims = [(int(f.shape[2]), int(f.shape[1])) for f in image_frames]
            print(f"[NV_Wan27VideoEditPrep] ref image dims (WxH): {ref_dims} | sampling: {image_sampling_note}")

        # --- prepare input video (either encoded or from frames) ---
        video_obj: Input.Video | None = None
        video_w = video_h = 0
        video_dur = 0.0
        video_first_frame: torch.Tensor | None = None
        encode_source = "none"

        if input_video is not None:
            video_obj = input_video
            try:
                video_w, video_h = input_video.get_dimensions()
                video_dur = float(input_video.get_duration())
            except Exception as e:
                raise ValueError(
                    f"Could not read duration/dimensions from input_video: {e}. "
                    "Wan 2.7 Video Edit requires 2-10s duration — the prep node cannot "
                    "validate an unreadable video. Re-encode upstream or use "
                    "input_video_frames + input_video_fps instead."
                ) from e
            encode_source = "video_input"

        elif input_video_frames is not None and input_video_frames.shape[0] > 0:
            frames, (video_h, video_w) = _snap_even(input_video_frames)
            n_frames = frames.shape[0]
            video_dur = n_frames / input_video_fps
            video_obj = InputImpl.VideoFromComponents(
                VideoComponents(
                    images=frames,
                    frame_rate=Fraction(int(round(input_video_fps))),
                )
            )
            video_first_frame = frames[0:1]
            encode_source = "frames_encoded"

        if video_obj is None:
            raise ValueError(
                "NV_Wan27VideoEditPrep requires either input_video or input_video_frames."
            )

        # Validate duration envelope (Wan 2.7 Video Edit: 2-10s)
        if video_dur and (video_dur < 2.0 or video_dur > 10.0):
            raise ValueError(
                f"Input video duration {video_dur:.2f}s outside Wan 2.7 Video Edit range [2.0s, 10.0s]."
            )

        # --- upload video ---
        uploaded_video_url = await upload_video_to_comfyapi(
            cls, video_obj, wait_label="Uploading input video"
        )
        print(f"[NV_Wan27VideoEditPrep] Input video uploaded ({encode_source}) → ...{uploaded_video_url[-40:]}")

        # --- upload reference images ---
        uploaded_image_urls: list[str] = []
        for i, frame in enumerate(image_frames, 1):
            url = await upload_image_to_comfyapi(
                cls, image=frame, wait_label=f"Uploading reference image {i}"
            )
            uploaded_image_urls.append(url)
            print(f"[NV_Wan27VideoEditPrep] reference_image[{i}] uploaded → ...{url[-40:]}")

        # --- preview grid ---
        preview = _build_preview_grid(image_frames, video_first_frame)

        # --- config payload ---
        config = {
            "input_video_url": uploaded_video_url,
            "input_video_duration_s": round(video_dur, 3) if video_dur else None,
            "input_video_dimensions": (video_w, video_h) if video_w else None,
            "reference_image_urls": uploaded_image_urls,
            "n_reference_images": n_images,
            "image_sampling_note": image_sampling_note,
            "encode_source": encode_source,
        }

        info = json.dumps(
            {
                "input_video": {
                    "source": encode_source,
                    "dimensions": f"{video_w}x{video_h}" if video_w else None,
                    "duration_s": round(video_dur, 3) if video_dur else None,
                },
                "n_reference_images": n_images,
                "image_sampling": image_sampling_note,
                "prompt_length": len(prompt),
            },
            indent=2,
        )

        print(f"[NV_Wan27VideoEditPrep] Ready: input_video={video_w}x{video_h} {video_dur:.2f}s, "
              f"{n_images} ref image(s)")

        return IO.NodeOutput(config, preview, prompt, float(video_dur or 0.0), info)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "NV_Wan27VideoEditPrep": NV_Wan27VideoEditPrep,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_Wan27VideoEditPrep": "NV Wan 2.7 Video Edit Prep",
}
