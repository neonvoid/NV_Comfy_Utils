"""NV Video Frame Sampler — extract N curated frames from a VIDEO or IMAGE batch.

Primary use case: pre-process Kling output for Seedance reference_images. Take
a long Kling-edited video, sample 3-5 representative frames covering the
subject from different action moments, hand them off as a multi-view identity
reference batch.

Also useful for general workflows: "give me 5 frames evenly spaced across this
clip" without needing to wire a frame index calculator manually.

Accepts EITHER a VIDEO or an IMAGE batch as input. VIDEO input gets decoded
to frames first via .get_components(). IMAGE input is used as-is.
"""

from __future__ import annotations

import torch

from comfy_api.latest import IO, Input


def _uniform_sample_indices(total: int, take: int) -> list[int]:
    """Evenly-spaced indices across [0, total-1]."""
    if take >= total:
        return list(range(total))
    if take == 1:
        return [0]
    step = (total - 1) / (take - 1)
    return [round(i * step) for i in range(take)]


def _endpoints_plus_evenly_spaced(total: int, take: int) -> list[int]:
    """First + last + evenly spaced between. Maximizes pose diversity for short batches."""
    if take >= total:
        return list(range(total))
    if take == 1:
        return [0]
    if take == 2:
        return [0, total - 1]
    # Reserve first + last, evenly space the rest in between
    interior_count = take - 2
    interior_step = (total - 1) / (interior_count + 1)
    interior = [round((i + 1) * interior_step) for i in range(interior_count)]
    return [0] + interior + [total - 1]


def _endpoints_plus_middle(total: int, take: int) -> list[int]:
    """First + middle + last + evenly fill remaining slots. Pose-anchored coverage."""
    if take >= total:
        return list(range(total))
    if take == 1:
        return [0]
    if take == 2:
        return [0, total - 1]
    if take == 3:
        return [0, total // 2, total - 1]
    # Take >= 4: anchor at 0, total/2, total-1, fill the rest evenly
    anchors = sorted({0, total // 2, total - 1})
    remaining = take - len(anchors)
    if remaining <= 0:
        return anchors[:take]
    # Fill remaining slots evenly across the full range, dedup against anchors
    extras = []
    step = (total - 1) / (remaining + 1)
    for i in range(remaining):
        candidate = round((i + 1) * step)
        if candidate not in anchors and candidate not in extras:
            extras.append(candidate)
    combined = sorted(set(anchors + extras))
    if len(combined) < take:
        # Fallback to uniform if dedup left us short
        return _uniform_sample_indices(total, take)
    return combined[:take]


_STRATEGIES = {
    "uniform": _uniform_sample_indices,
    "endpoints_plus_evenly_spaced": _endpoints_plus_evenly_spaced,
    "endpoints_plus_middle": _endpoints_plus_middle,
}


class NV_VideoFrameSampler(IO.ComfyNode):
    """Sample N representative frames from a VIDEO or IMAGE batch."""

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="NV_VideoFrameSampler",
            display_name="NV Video Frame Sampler",
            category="NV_Utils/image",
            description=(
                "Extract N curated frames from a VIDEO or IMAGE batch. Strategies: uniform "
                "(evenly spaced), endpoints_plus_evenly_spaced (first+last+spaced between — "
                "maximizes pose diversity), endpoints_plus_middle (first+middle+last anchors). "
                "Use to feed Kling output frames into Seedance Prep V2 as multi-view refs."
            ),
            inputs=[
                IO.Video.Input(
                    "video",
                    tooltip="VIDEO input — gets decoded to frames via .get_components(). Optional if `images` is wired.",
                    optional=True,
                ),
                IO.Image.Input(
                    "images",
                    tooltip="Pre-decoded IMAGE batch [F,H,W,C]. Optional alternative to VIDEO input. Takes priority if both wired.",
                    optional=True,
                ),
                IO.Int.Input(
                    "target_count",
                    default=5,
                    min=1,
                    max=20,
                    step=1,
                    tooltip="Number of frames to sample. For Seedance reference_images: 3-5 is the sweet spot per multi-AI research.",
                ),
                IO.Combo.Input(
                    "strategy",
                    options=list(_STRATEGIES.keys()),
                    default="endpoints_plus_evenly_spaced",
                    tooltip=(
                        "uniform = evenly spaced across the clip.\n"
                        "endpoints_plus_evenly_spaced (recommended) = first, last, plus N-2 evenly between. "
                        "Maximizes pose/expression diversity for identity refs.\n"
                        "endpoints_plus_middle = first, middle, last + fill — pose-anchored coverage."
                    ),
                ),
                IO.Int.Input(
                    "skip_first_frames",
                    default=0,
                    min=0,
                    max=1000,
                    step=1,
                    tooltip="Skip this many frames at the start before sampling. Useful for trimming Kling warm-up frames.",
                ),
                IO.Int.Input(
                    "skip_last_frames",
                    default=0,
                    min=0,
                    max=1000,
                    step=1,
                    tooltip="Skip this many frames at the end before sampling.",
                ),
            ],
            outputs=[
                IO.Image.Output(display_name="sampled_images"),
                IO.String.Output(display_name="frame_indices"),
                IO.Float.Output(display_name="source_fps"),
                IO.Int.Output(display_name="total_frames_in"),
            ],
        )

    @classmethod
    async def execute(
        cls,
        target_count: int,
        strategy: str,
        skip_first_frames: int,
        skip_last_frames: int,
        video: Input.Video | None = None,
        images: Input.Image | None = None,
    ) -> IO.NodeOutput:
        # --- resolve source: images input takes priority over video ---
        source_fps = 24.0
        if images is not None:
            frames = images
            if frames.ndim == 3:
                frames = frames.unsqueeze(0)
            if frames.ndim != 4:
                raise ValueError(
                    f"[NV_VideoFrameSampler] images must be [B,H,W,C] or [H,W,C], got {tuple(frames.shape)}"
                )
            print(f"[NV_VideoFrameSampler] Source: IMAGE batch {tuple(frames.shape)}")
        elif video is not None:
            try:
                components = video.get_components()
                frames = components.images
                source_fps = float(components.frame_rate)
            except Exception as e:
                raise ValueError(f"[NV_VideoFrameSampler] Failed to decode VIDEO: {e}")
            print(f"[NV_VideoFrameSampler] Source: VIDEO decoded → IMAGE batch {tuple(frames.shape)} @ {source_fps:.2f}fps")
        else:
            raise ValueError(
                "[NV_VideoFrameSampler] Wire either `video` (VIDEO) or `images` (IMAGE batch)."
            )

        total_frames = int(frames.shape[0])

        # --- skip windowing ---
        start = max(0, min(skip_first_frames, total_frames))
        end = max(start, total_frames - max(0, skip_last_frames))
        windowed = frames[start:end]
        windowed_total = int(windowed.shape[0])
        if windowed_total == 0:
            raise ValueError(
                f"[NV_VideoFrameSampler] Skip windowing left 0 frames (skip_first={skip_first_frames}, "
                f"skip_last={skip_last_frames}, total={total_frames})."
            )

        # --- sample ---
        sampler = _STRATEGIES[strategy]
        indices = sampler(windowed_total, target_count)
        sampled = torch.stack([windowed[i] for i in indices], dim=0)
        # Add the skip offset back so the indices reflect the original frame numbers
        absolute_indices = [i + start for i in indices]

        print(
            f"[NV_VideoFrameSampler] Sampled {len(indices)} frames from {windowed_total} "
            f"({total_frames} total): strategy={strategy}, indices={absolute_indices}"
        )

        return IO.NodeOutput(
            sampled,
            str(absolute_indices),
            source_fps,
            total_frames,
        )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "NV_VideoFrameSampler": NV_VideoFrameSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_VideoFrameSampler": "NV Video Frame Sampler",
}
