"""
Latent Utility Nodes

Helper nodes for working with video latents, particularly for VACE workflows
where reference images need to be prepended to match conditioning dimensions.
"""

import torch
import comfy.latent_formats


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
        }

    RETURN_TYPES = ("LATENT", "INT")
    RETURN_NAMES = ("latent", "trim_amount")
    FUNCTION = "prepend"
    CATEGORY = "NV_Utils/latent"
    DESCRIPTION = "Prepend reference frame(s) to video latent for VACE workflows. Use TrimVideoLatent with trim_amount after generation."

    def prepend(self, video_latent, reference_latent):
        video_samples = video_latent["samples"]  # [B, C, T_video, H, W]
        ref_samples = reference_latent["samples"]  # [B, C, T_ref, H, W]

        # Validate spatial dimensions match
        if ref_samples.shape[3:] != video_samples.shape[3:]:
            raise ValueError(
                f"Reference latent spatial size {ref_samples.shape[3:]} doesn't match "
                f"video latent spatial size {video_samples.shape[3:]}. "
                f"Ensure both are encoded at the same resolution."
            )

        # Concatenate along temporal dimension
        combined = torch.cat([ref_samples, video_samples], dim=2)

        trim_amount = ref_samples.shape[2]

        print(f"[NV_PrependReferenceLatent] Prepended {trim_amount} reference frame(s)")
        print(f"  Video latent: {list(video_samples.shape)} -> Combined: {list(combined.shape)}")
        print(f"  Use TrimVideoLatent with trim_amount={trim_amount} after generation")

        out_latent = video_latent.copy()
        out_latent["samples"] = combined

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

        out_latent = latent_a.copy()
        out_latent["samples"] = combined

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


# Node registration
NODE_CLASS_MAPPINGS = {
    "NV_PrependReferenceLatent": NV_PrependReferenceLatent,
    "NV_LatentTemporalConcat": NV_LatentTemporalConcat,
    "NV_LatentInfo": NV_LatentInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_PrependReferenceLatent": "NV Prepend Reference Latent",
    "NV_LatentTemporalConcat": "NV Latent Temporal Concat",
    "NV_LatentInfo": "NV Latent Info",
}
