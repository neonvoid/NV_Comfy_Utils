"""
Latent Utility Nodes

Helper nodes for working with video latents, particularly for VACE workflows
where reference images need to be prepended to match conditioning dimensions.
"""

import torch
import comfy.latent_formats
import comfy.sample

from .latent_constants import NV_CASCADED_CONFIG_KEY, LATENT_SAFE_KEYS


class NV_PrependReferenceLatent:
    """
    Prepend reference latent frame(s) to a video latent.

    Use this when:
    - You have an encoded video latent (e.g., for V2V with denoise < 1.0)
    - You're using WanVaceToVideo with reference_image (1 frame)
    - You're using NV_VacePrePassReference with multi-frame references (N frames)
    - You need the latent to have the same frame count as VACE conditioning

    The reference frames will be generated along with your video, then you can
    trim them with TrimVideoLatent after generation.

    Workflow:
        [VAE Encode Video] ─┬─► [NV Prepend Reference Latent] ─► [Sampler] ─► [TrimVideoLatent] ─► [VAE Decode]
        [ref_latent output] ┘
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_latent": ("LATENT",),
                "reference_latent": ("LATENT", {
                    "tooltip": "Reference latent to prepend. Single frame (from VAE Encode) "
                               "or multi-frame (from NV_VacePrePassReference ref_latent output)."
                }),
            },
            "optional": {
                "model": ("MODEL", {
                    "tooltip": "Required in cascaded mode (pre-noised latent). "
                               "Used to auto-noise the reference to match the video's noise level. "
                               "Wire from the same model used in Stage 3 sampling."
                }),
            },
        }

    RETURN_TYPES = ("LATENT", "INT")
    RETURN_NAMES = ("latent", "trim_amount")
    FUNCTION = "prepend"
    CATEGORY = "NV_Utils/latent"
    DESCRIPTION = (
        "Prepend reference frame(s) to video latent for VACE workflows. "
        "In cascaded mode (pre-noised latent), auto-noises the reference to match. "
        "Wire MODEL in cascaded pipelines. Use TrimVideoLatent with trim_amount after generation."
    )

    def prepend(self, video_latent, reference_latent, model=None):
        video_samples = video_latent["samples"]  # [B, C, T_video, H, W]
        ref_samples = reference_latent["samples"]  # [B, C, T_ref, H, W]

        # Validate spatial dimensions match
        if ref_samples.shape[3:] != video_samples.shape[3:]:
            raise ValueError(
                f"Reference latent spatial size {ref_samples.shape[3:]} doesn't match "
                f"video latent spatial size {video_samples.shape[3:]}. "
                f"Ensure both are encoded at the same resolution."
            )

        # --- Auto-noise reference in cascaded mode ---
        cascaded_config = video_latent.get(NV_CASCADED_CONFIG_KEY, None)
        if cascaded_config is not None:
            if model is None:
                raise ValueError(
                    "[NV_PrependReferenceLatent] Cascaded mode detected (video latent has "
                    "nv_cascaded_config from NV_PreNoiseLatent), but no MODEL is connected.\n"
                    "The reference latent is CLEAN (sigma=0) while the video latent is "
                    f"PRE-NOISED (sigma={cascaded_config.get('start_sigma', '?')}).\n"
                    "Without matching noise levels, the model sees a massive signal cliff "
                    "between reference and video frames, producing garbage.\n\n"
                    "FIX: Wire the Stage 3 MODEL → NV_PrependReferenceLatent 'model' input."
                )

            start_sigma = cascaded_config["start_sigma"]
            prenoise_seed = cascaded_config.get("prenoise_seed", 0)
            # Offset seed so reference noise is independent from video noise
            ref_seed = prenoise_seed + 7

            # Generate noise for reference
            noise = comfy.sample.prepare_noise(ref_samples, ref_seed)

            # Apply noise using model's formula (same as PreNoiseLatent)
            model_sampling = model.get_model_object("model_sampling")
            process_latent_in = model.get_model_object("process_latent_in")
            process_latent_out = model.get_model_object("process_latent_out")

            ref_in = process_latent_in(ref_samples)
            sigma_tensor = torch.tensor(
                [start_sigma], dtype=ref_in.dtype, device=ref_in.device
            )
            noised_ref = model_sampling.noise_scaling(
                sigma_tensor, noise.to(ref_in.device), ref_in
            )
            noised_ref = process_latent_out(noised_ref)
            noised_ref = torch.nan_to_num(noised_ref, nan=0.0, posinf=0.0, neginf=0.0)

            ref_samples = noised_ref
            print(f"[NV_PrependReferenceLatent] Cascaded mode: auto-noised reference "
                  f"to sigma={start_sigma:.4f} (seed={ref_seed})")

        # Concatenate along temporal dimension
        combined = torch.cat([ref_samples, video_samples], dim=2)

        trim_amount = reference_latent["samples"].shape[2]

        print(f"[NV_PrependReferenceLatent] Prepended {trim_amount} reference frame(s)")
        print(f"  Video latent: {list(video_samples.shape)} -> Combined: {list(combined.shape)}")
        print(f"  Use TrimVideoLatent with trim_amount={trim_amount} after generation")

        # Build clean output latent — only carry forward keys that are safe
        # after changing the temporal dimension. noise_mask and batch_index
        # are tied to the original temporal size and become stale/misaligned
        # after prepending, causing noise bleed artifacts at frame boundaries.
        out_latent = {"samples": combined}

        for key in LATENT_SAFE_KEYS:
            if key in video_latent and key != "samples":
                out_latent[key] = video_latent[key]

        # Log any dropped keys from the original
        for key in video_latent:
            if key == "samples":
                continue
            if key not in LATENT_SAFE_KEYS:
                print(f"  [NV_PrependReferenceLatent] Dropped stale key '{key}' "
                      f"(temporal dimension changed)")

        return (out_latent, trim_amount)


class NV_LatentTemporalConcat:
    """
    Concatenate two latents along the temporal dimension.

    Useful for combining video segments or prepending/appending frames.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent_a": ("LATENT",),
                "latent_b": ("LATENT",),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "concat"
    CATEGORY = "NV_Utils/latent"
    DESCRIPTION = "Concatenate two latents along the temporal dimension (latent_a first, then latent_b)."

    def concat(self, latent_a, latent_b):
        samples_a = latent_a["samples"]
        samples_b = latent_b["samples"]

        # Validate spatial dimensions match
        if samples_a.shape[3:] != samples_b.shape[3:]:
            raise ValueError(
                f"Latent A spatial size {samples_a.shape[3:]} doesn't match "
                f"Latent B spatial size {samples_b.shape[3:]}."
            )

        combined = torch.cat([samples_a, samples_b], dim=2)

        print(f"[NV_LatentTemporalConcat] {list(samples_a.shape)} + {list(samples_b.shape)} -> {list(combined.shape)}")

        # Build clean output — drop temporal-dependent metadata (noise_mask,
        # batch_index) since the temporal dimension has changed.
        out_latent = {"samples": combined}
        for key in LATENT_SAFE_KEYS:
            if key in latent_a and key != "samples":
                out_latent[key] = latent_a[key]

        for key in latent_a:
            if key == "samples":
                continue
            if key not in LATENT_SAFE_KEYS:
                print(f"  [NV_LatentTemporalConcat] Dropped stale key '{key}' "
                      f"(temporal dimension changed)")

        return (out_latent,)


class NV_LatentInfo:
    """
    Inspect a latent tensor and output its metadata.

    Reports raw tensor shape and pixel-space dimensions using configurable
    compression factors. Useful for debugging shape mismatches (e.g., VACE
    conditioning with different frame counts).

    Temporal formula (video): pixel_frames = (latent_frames - 1) * temporal_compression + 1
    This matches WAN / HunyuanVideo VAE behavior where the first frame is not temporally compressed.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "spatial_compression": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 64,
                    "step": 1,
                    "tooltip": "Spatial downscale ratio of the VAE. 8 for WAN 2.1 / HunyuanVideo, 16 for WAN 2.2 / LTXV / Flux2."
                }),
                "temporal_compression": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 16,
                    "step": 1,
                    "tooltip": "Temporal downscale ratio of the VAE. 4 for WAN / HunyuanVideo, 8 for Cosmos."
                }),
            },
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "INT", "BOOLEAN", "STRING")
    RETURN_NAMES = ("width", "height", "frames", "batch_size", "channels", "is_video", "info_string")
    FUNCTION = "inspect"
    CATEGORY = "NV_Utils/latent"
    DESCRIPTION = "Inspect a latent tensor and output pixel-space resolution, frame count, and shape metadata."

    def inspect(self, latent, spatial_compression, temporal_compression):
        samples = latent["samples"]
        shape = list(samples.shape)
        ndim = len(shape)

        is_video = ndim == 5  # [B, C, T, H, W]
        batch_size = shape[0]
        channels = shape[1]

        if is_video:
            latent_t, latent_h, latent_w = shape[2], shape[3], shape[4]
            pixel_w = latent_w * spatial_compression
            pixel_h = latent_h * spatial_compression
            pixel_frames = (latent_t - 1) * temporal_compression + 1

            info = (
                f"Video Latent: {shape}\n"
                f"Pixel: {pixel_w}x{pixel_h}, {pixel_frames} frames\n"
                f"Compression: {spatial_compression}x spatial, {temporal_compression}x temporal"
            )
        else:
            latent_h, latent_w = shape[-2], shape[-1]
            pixel_w = latent_w * spatial_compression
            pixel_h = latent_h * spatial_compression
            pixel_frames = 0

            info = (
                f"Image Latent: {shape}\n"
                f"Pixel: {pixel_w}x{pixel_h}\n"
                f"Compression: {spatial_compression}x spatial"
            )

        print(f"[NV_LatentInfo] {info}")

        return (pixel_w, pixel_h, pixel_frames, batch_size, channels, is_video, info)


class NV_LatentTemporalSlice:
    """
    Slice a range of frames from a video latent.

    Specify skip and cap in pixel frames (0-indexed, matching VHS conventions).
    The node converts to latent indices and snaps to valid VAE temporal
    boundaries (tc*n + 1).

    Use case: you have a 321-frame encoded latent but want to test your workflow
    with just the first 81 frames — set frame_load_cap=81.

    Valid frame counts for tc=4: 1, 5, 9, 13, 17, 21, 25, ... (4n+1)
    The snap option controls rounding when your request doesn't land on a boundary.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "skip_first_frames": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100000,
                    "step": 1,
                    "tooltip": "Number of pixel frames to skip from the start (0-indexed, same as VHS Load Video)."
                }),
                "frame_load_cap": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100000,
                    "step": 1,
                    "tooltip": "Maximum pixel frames to keep. 0 = all remaining frames (same as VHS Load Video)."
                }),
                "temporal_compression": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 16,
                    "step": 1,
                    "tooltip": "Temporal compression factor of the VAE. 4 for WAN / HunyuanVideo, 8 for Cosmos."
                }),
                "snap": (["down (fewer frames)", "up (more frames)", "nearest"], {
                    "default": "down (fewer frames)",
                    "tooltip": "How to round when the requested range doesn't land on a valid tc*n+1 boundary."
                }),
            },
        }

    RETURN_TYPES = ("LATENT", "INT", "STRING")
    RETURN_NAMES = ("latent", "pixel_frame_count", "info")
    FUNCTION = "slice_temporal"
    CATEGORY = "NV_Utils/latent"
    DESCRIPTION = "Slice a frame range from a video latent. Uses VHS-style skip/cap in pixel frames; auto-snaps to VAE temporal boundaries (tc*n+1)."

    def slice_temporal(self, latent, skip_first_frames, frame_load_cap, temporal_compression, snap):
        samples = latent["samples"]
        tc = temporal_compression

        if samples.ndim != 5:
            raise ValueError(
                f"Expected 5D video latent [B,C,T,H,W], got {samples.ndim}D: {list(samples.shape)}"
            )

        latent_t = samples.shape[2]
        total_pixel_frames = (latent_t - 1) * tc + 1

        # Clamp skip to valid range
        skip_first_frames = max(0, min(skip_first_frames, total_pixel_frames - 1))

        # Snap skip to latent boundary (round to the latent frame containing the skip point)
        start_latent = skip_first_frames // tc
        actual_skip = start_latent * tc

        # Remaining pixel frames after skip
        remaining_latent = latent_t - start_latent
        remaining_pixel = (remaining_latent - 1) * tc + 1

        # Desired count: frame_load_cap=0 means all remaining
        if frame_load_cap <= 0:
            desired_count = remaining_pixel
        else:
            desired_count = min(frame_load_cap, remaining_pixel)

        # Find the valid counts bracketing desired_count
        # Valid counts: k*tc + 1 for k = 0, 1, 2, ...
        k_down = max(0, (desired_count - 1) // tc)
        count_down = k_down * tc + 1
        count_up = count_down if count_down == desired_count else (k_down + 1) * tc + 1

        # Pick based on snap mode
        if "down" in snap:
            chosen_count = count_down
        elif "up" in snap:
            chosen_count = count_up
        else:  # nearest
            if (desired_count - count_down) <= (count_up - desired_count):
                chosen_count = count_down
            else:
                chosen_count = count_up

        # Convert to latent frame count and end index
        num_latent = (chosen_count - 1) // tc + 1
        end_latent = start_latent + num_latent

        # Clamp to available latent frames
        if end_latent > latent_t:
            end_latent = latent_t
            num_latent = end_latent - start_latent
            chosen_count = (num_latent - 1) * tc + 1

        sliced = samples[:, :, start_latent:end_latent, :, :]

        snapped = desired_count != chosen_count

        # Build info string
        info_lines = [
            f"Input: {total_pixel_frames} pixel frames ({latent_t} latent)",
            f"Skip: {skip_first_frames} requested \u2192 {actual_skip} actual (latent boundary)",
            f"Cap: {frame_load_cap if frame_load_cap > 0 else 'all'} requested \u2192 {chosen_count} actual",
        ]
        if snapped:
            snap_dir = snap.split(" ")[0]
            info_lines.append(
                f"Snapped {snap_dir}: {desired_count} \u2192 {chosen_count} frames "
                f"(valid counts are {tc}n+1: ...{count_down}, {count_up}...)"
            )
        info_lines.extend([
            f"Latent slice: [{start_latent}:{end_latent}] of {latent_t}",
            f"Output: {chosen_count} pixel frames, shape {list(sliced.shape)}",
        ])
        info = "\n".join(info_lines)

        print(f"[NV_LatentTemporalSlice] {info}")

        # Build clean output — drop temporal-dependent metadata (noise_mask,
        # batch_index) since the temporal dimension has changed.
        out_latent = {"samples": sliced}
        for key in LATENT_SAFE_KEYS:
            if key in latent and key != "samples":
                out_latent[key] = latent[key]

        for key in latent:
            if key == "samples":
                continue
            if key not in LATENT_SAFE_KEYS:
                print(f"  [NV_LatentTemporalSlice] Dropped stale key '{key}' "
                      f"(temporal dimension changed)")

        return (out_latent, chosen_count, info)


# Node registration
NODE_CLASS_MAPPINGS = {
    "NV_PrependReferenceLatent": NV_PrependReferenceLatent,
    "NV_LatentTemporalConcat": NV_LatentTemporalConcat,
    "NV_LatentInfo": NV_LatentInfo,
    "NV_LatentTemporalSlice": NV_LatentTemporalSlice,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_PrependReferenceLatent": "NV Prepend Reference Latent",
    "NV_LatentTemporalConcat": "NV Latent Temporal Concat",
    "NV_LatentInfo": "NV Latent Info",
    "NV_LatentTemporalSlice": "NV Latent Temporal Slice",
}
