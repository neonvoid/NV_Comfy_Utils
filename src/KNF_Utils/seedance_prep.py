"""NV Seedance Prep — preprocessing + upload for Seedance 2.0 API workflows.

Mirrors the two-node pattern used by the Kling edit fork:

  NV_SeedancePrep         — validates + uploads reference assets, emits config
                            + preview + draft prompt. Runs once per set of refs
                            (cached across prompt iterations).
  NV_SeedanceRefVideo     — slim API caller. Reads config, calls Seedance API.

Accepts a reference video as either an already-encoded VIDEO input, or as a
batched IMAGE tensor + fps (auto-encoded via VideoFromComponents). Auto-clamps
ref-video resolution to the Seedance pixel budget (409,600-927,408 per frame).
Validates duration (1.8-15.1s). Handles N>9 reference images by sampling.
"""

from __future__ import annotations

import json
from fractions import Fraction

import torch
import torch.nn.functional as F
from comfy_api.latest import IO, Input, InputImpl
from comfy_api.latest._io import Custom as _IOCustom
from comfy_api.latest._util.video_types import VideoComponents
from comfy_api_nodes.apis.bytedance import SEEDANCE2_REF_VIDEO_PIXEL_LIMITS
from comfy_api_nodes.util import (
    upload_image_to_comfyapi,
    upload_video_to_comfyapi,
)


# ---------------------------------------------------------------------------
# Shared custom type
# ---------------------------------------------------------------------------

SEEDANCE_UPLOAD_CONFIG = _IOCustom("SEEDANCE_UPLOAD_CONFIG")


# ---------------------------------------------------------------------------
# Pixel budget (same for both Seedance 2.0 variants as of 2026-04)
# ---------------------------------------------------------------------------

_DEFAULT_PIXEL_LIMITS = {"min": 409_600, "max": 927_408}
_MAX_REF_IMAGES = 9


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pixel_limits(model_id: str | None) -> dict:
    if model_id and model_id in SEEDANCE2_REF_VIDEO_PIXEL_LIMITS:
        return SEEDANCE2_REF_VIDEO_PIXEL_LIMITS[model_id]
    return _DEFAULT_PIXEL_LIMITS


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


def _clamp_frames_to_budget(frames: torch.Tensor, limits: dict) -> tuple[torch.Tensor, bool, tuple[int, int]]:
    """Bilinear-downscale frames [F,H,W,C] so pixels-per-frame fit in [min, max].

    Returns (frames, did_resize, (new_h, new_w)).
    """
    h, w = frames.shape[1], frames.shape[2]
    pixels = h * w
    max_px = limits.get("max")
    min_px = limits.get("min")

    if max_px and pixels > max_px:
        scale = (max_px / pixels) ** 0.5
        new_h = max(2, int(h * scale))
        new_w = max(2, int(w * scale))
        new_h, new_w = _even_dims(new_h, new_w)
        x = frames.permute(0, 3, 1, 2)
        x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
        return x.permute(0, 2, 3, 1), True, (new_h, new_w)

    if min_px and pixels < min_px:
        raise ValueError(
            f"Reference video too small: {w}x{h} = {pixels:,}px. "
            f"Minimum {min_px:,}px (~{int(min_px ** 0.5)}x{int(min_px ** 0.5)}). "
            f"Upscale upstream — auto-upscale not supported (would add blur)."
        )

    new_h, new_w = _even_dims(h, w)
    if (new_h, new_w) != (h, w):
        x = frames.permute(0, 3, 1, 2)
        x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
        return x.permute(0, 2, 3, 1), True, (new_h, new_w)

    return frames, False, (h, w)


def _build_preview_grid(ref_frames: list[torch.Tensor], video_first_frame: torch.Tensor | None) -> torch.Tensor:
    """Concat ref images + video first frame into a single [N,H,W,C] preview batch.

    Pads each to the max height in the set (keeps aspect intact width-wise).
    Returns a single-image fallback if no refs.
    """
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

class NV_SeedancePrep(IO.ComfyNode):
    """Seedance 2.0 preprocessing + upload. Emits SEEDANCE_UPLOAD_CONFIG.

    Accepts reference images (batched IMAGE), a reference video (as VIDEO or
    as frames + fps), and a draft prompt. Validates, auto-clamps pixel budget,
    uploads to comfy API, and emits a config for the slim NV_SeedanceRefVideo
    caller. The draft prompt is passed through unchanged so users can route it
    through NV_PromptRefiner before the API node.
    """

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="NV_SeedancePrep",
            display_name="NV Seedance Prep",
            category="NV_Utils/api",
            description=(
                "Upload + preview for Seedance 2.0. Accepts ref images, ref video "
                "(or frames+fps), and a draft prompt. Emits SEEDANCE_UPLOAD_CONFIG "
                "for NV Seedance Ref Video."
            ),
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip=(
                        "Draft prompt. Wire through NV Prompt Refiner (mode=seedance_ref) "
                        "to produce the final_prompt for the API node. Passed through unchanged "
                        "by this node."
                    ),
                ),
                IO.Image.Input(
                    "reference_images",
                    tooltip=(
                        "Batched IMAGE tensor [N,H,W,C]. Up to 9 images — if N>9, "
                        "sampled per sample_mode."
                    ),
                    optional=True,
                ),
                IO.Video.Input(
                    "reference_video",
                    tooltip="Already-encoded VIDEO. Takes precedence over reference_video_frames.",
                    optional=True,
                ),
                IO.Image.Input(
                    "reference_video_frames",
                    tooltip=(
                        "Frames tensor [F,H,W,C] to encode into a reference VIDEO. "
                        "Ignored if reference_video is wired."
                    ),
                    optional=True,
                ),
                IO.Float.Input(
                    "reference_video_fps",
                    default=24.0,
                    min=8.0,
                    max=60.0,
                    step=1.0,
                    tooltip="FPS for encoding reference_video_frames → VIDEO.",
                ),
                IO.Combo.Input(
                    "sample_mode",
                    options=["uniform", "first", "last"],
                    default="uniform",
                    tooltip="How to pick 9 frames when N>9 reference images are provided.",
                ),
                IO.Boolean.Input(
                    "auto_downscale",
                    default=True,
                    tooltip=(
                        "Bilinear-downscale reference video to fit Seedance pixel budget "
                        "(max 927,408px/frame). Upscale is never automatic — undersize ref "
                        "videos raise an error."
                    ),
                ),
            ],
            outputs=[
                SEEDANCE_UPLOAD_CONFIG.Output(display_name="config"),
                IO.Image.Output(display_name="preview"),
                IO.String.Output(display_name="prompt"),
                IO.Int.Output(display_name="num_refs"),
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
        reference_video_fps: float,
        sample_mode: str,
        auto_downscale: bool,
        reference_images: Input.Image | None = None,
        reference_video: Input.Video | None = None,
        reference_video_frames: Input.Image | None = None,
    ) -> IO.NodeOutput:
        # --- probe: log raw input shapes ---
        if reference_images is not None:
            ri_shape = tuple(reference_images.shape)
            ri_range = (float(reference_images.min().item()), float(reference_images.max().item()))
            print(f"[NV_SeedancePrep] reference_images: shape={ri_shape} range=({ri_range[0]:.3f}..{ri_range[1]:.3f})")
        if reference_video_frames is not None:
            rvf_shape = tuple(reference_video_frames.shape)
            rvf_range = (float(reference_video_frames.min().item()), float(reference_video_frames.max().item()))
            print(f"[NV_SeedancePrep] reference_video_frames: shape={rvf_shape} range=({rvf_range[0]:.3f}..{rvf_range[1]:.3f})")

        # --- sample reference images ---
        image_frames, image_sampling_note = (
            _sample_image_batch(reference_images, sample_mode, _MAX_REF_IMAGES)
            if reference_images is not None
            else ([], "none")
        )
        n_images = len(image_frames)
        if n_images > 0:
            ref_dims = [(int(f.shape[2]), int(f.shape[1])) for f in image_frames]
            print(f"[NV_SeedancePrep] ref image dims (WxH): {ref_dims} | sampling: {image_sampling_note}")

        # --- prepare ref video (either encoded or from frames) ---
        ref_video_obj: Input.Video | None = None
        ref_w = ref_h = 0
        ref_dur = 0.0
        did_resize = False
        video_first_frame: torch.Tensor | None = None
        encode_source = "none"

        if reference_video is not None:
            ref_video_obj = reference_video
            try:
                ref_w, ref_h = reference_video.get_dimensions()
                ref_dur = float(reference_video.get_duration())
            except Exception:
                ref_w = ref_h = 0
                ref_dur = 0.0
            encode_source = "video_input"

        elif reference_video_frames is not None and reference_video_frames.shape[0] > 0:
            frames = reference_video_frames
            if auto_downscale:
                frames, did_resize, (ref_h, ref_w) = _clamp_frames_to_budget(
                    frames, _DEFAULT_PIXEL_LIMITS
                )
            else:
                # Still snap to even dims for codec
                frames, did_resize, (ref_h, ref_w) = _clamp_frames_to_budget(
                    frames, {"min": None, "max": None}
                )

            n_frames = frames.shape[0]
            ref_dur = n_frames / reference_video_fps
            if ref_dur < 1.8:
                raise ValueError(
                    f"Reference video frames too short: {n_frames} frames at "
                    f"{reference_video_fps:.1f}fps = {ref_dur:.2f}s. Minimum 1.8s."
                )
            if ref_dur > 15.1:
                # Trim trailing frames down to 15s exactly
                keep = int(15.0 * reference_video_fps)
                frames = frames[:keep]
                n_frames = keep
                ref_dur = n_frames / reference_video_fps

            ref_video_obj = InputImpl.VideoFromComponents(
                VideoComponents(
                    images=frames,
                    frame_rate=Fraction(int(round(reference_video_fps))),
                )
            )
            video_first_frame = frames[0:1]
            encode_source = "frames_encoded"

        if n_images == 0 and ref_video_obj is None:
            raise ValueError(
                "At least one reference_images frame or a reference_video / "
                "reference_video_frames input is required."
            )

        # Pixel-budget guard for already-encoded video (we can't resize it here —
        # just warn loudly so the API call fails fast instead of silently).
        if reference_video is not None and ref_w and ref_h:
            pixels = ref_w * ref_h
            limits = _DEFAULT_PIXEL_LIMITS
            if pixels > limits["max"]:
                raise ValueError(
                    f"Already-encoded reference_video is {ref_w}x{ref_h} = {pixels:,}px, "
                    f"over the {limits['max']:,}px budget. Re-encode upstream, or use "
                    f"reference_video_frames + auto_downscale=True instead."
                )
            if pixels < limits["min"]:
                raise ValueError(
                    f"Already-encoded reference_video is {ref_w}x{ref_h} = {pixels:,}px, "
                    f"under the {limits['min']:,}px minimum. Use a larger source."
                )

        if ref_video_obj is not None and ref_dur and (ref_dur < 1.8 or ref_dur > 15.1):
            raise ValueError(
                f"Reference video duration {ref_dur:.2f}s outside allowed range [1.8s, 15.1s]."
            )

        # --- upload images ---
        uploaded_image_urls: list[str] = []
        for i, frame in enumerate(image_frames, 1):
            url = await upload_image_to_comfyapi(
                cls, image=frame, wait_label=f"Uploading @Image{i}"
            )
            uploaded_image_urls.append(url)
            print(f"[NV_SeedancePrep] @Image{i} uploaded → ...{url[-40:]}")

        # --- upload video ---
        uploaded_video_url: str | None = None
        if ref_video_obj is not None:
            uploaded_video_url = await upload_video_to_comfyapi(
                cls, ref_video_obj, wait_label="Uploading @Video1"
            )
            print(f"[NV_SeedancePrep] @Video1 uploaded ({encode_source}) → ...{uploaded_video_url[-40:]}")

        # --- preview grid ---
        preview = _build_preview_grid(image_frames, video_first_frame)

        # --- config payload ---
        config = {
            "uploaded_image_urls": uploaded_image_urls,
            "uploaded_video_url": uploaded_video_url,
            "n_images": n_images,
            "image_sampling_note": image_sampling_note,
            "has_video": uploaded_video_url is not None,
            "ref_video_dimensions": (ref_w, ref_h) if uploaded_video_url else None,
            "ref_video_duration_s": round(ref_dur, 3) if uploaded_video_url else None,
            "ref_video_did_resize": did_resize,
            "encode_source": encode_source,
        }

        info = json.dumps(
            {
                "n_reference_images": n_images,
                "image_sampling": image_sampling_note,
                "reference_video": {
                    "source": encode_source,
                    "dimensions": f"{ref_w}x{ref_h}" if uploaded_video_url else None,
                    "duration_s": round(ref_dur, 3) if uploaded_video_url else None,
                    "did_resize": did_resize,
                },
                "prompt_length": len(prompt),
            },
            indent=2,
        )

        print(f"[NV_SeedancePrep] Ready: {n_images} image(s), video={'yes' if uploaded_video_url else 'no'}")

        return IO.NodeOutput(config, preview, prompt, n_images, info)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "NV_SeedancePrep": NV_SeedancePrep,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_SeedancePrep": "NV Seedance Prep",
}
