"""
NV Streaming VAE Encode

Encodes video frames to latents in a streaming fashion, moving each latent chunk
to CPU immediately. This frees GPU memory before downstream nodes (like WAN models)
load, preventing OOM on long videos.

Memory Comparison (473 frames @ 1440x960 after 2x upscale):
- Pixel frames on GPU: ~7.8 GB
- Latent output: ~82 MB (95x smaller!)
- Streaming: Frees pixel memory progressively, outputs latents on CPU

The WAN VAE encoder processes frames in groups:
- First iteration: frame 0 alone
- Subsequent iterations: groups of 4 frames (1-4, 5-8, etc.)

This node replicates that pattern but streams each chunk's latent to CPU immediately.
"""

import torch
import comfy.model_management as model_management

from .streaming_vace_to_video import streaming_vae_encode


class NV_StreamingVAEEncode:
    """
    Streaming VAE Encode for long videos.

    Encodes pixel frames chunk-by-chunk following WAN VAE's temporal pattern,
    moving each latent chunk to CPU immediately. This frees GPU memory before
    downstream nodes (like WAN diffusion models) need to load.

    Use Case:
    After upscaling a long video, the pixel frames can consume 8+ GB of GPU memory.
    Standard VAE encode keeps everything on GPU. This streaming version:
    1. Processes frames in WAN's chunk pattern (frame 0, then groups of 4)
    2. Moves each latent chunk to CPU immediately
    3. Frees the corresponding pixel frames from GPU
    4. Outputs compact latents (~95x smaller than pixels) on CPU

    This ensures GPU is clear when WAN models try to load.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pixels": ("IMAGE", {"tooltip": "Video frames [T, H, W, C] to encode."}),
                "vae": ("VAE", {"tooltip": "The WAN VAE model."}),
            },
            "optional": {
                "cache_clear_interval": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 100,
                    "tooltip": "Clear CUDA cache every N chunks. Lower = more cache clears, higher = faster but uses more transient memory."
                }),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "encode_streaming"
    CATEGORY = "NV_Utils"
    DESCRIPTION = "Encodes video frames to latents in streaming fashion, freeing GPU memory progressively. Use before WAN sampling on long upscaled videos."

    def encode_streaming(self, vae, pixels, cache_clear_interval=4):
        """Encode pixel frames to latents via shared streaming_vae_encode helper.

        Delegates the core encode loop to `streaming_vace_to_video.streaming_vae_encode`
        so this node and VacePrePassReference / NV_WanVaceToVideoStreaming / ReferenceFramePrep
        all share a single implementation.
        """
        total_pixel_frames = pixels.shape[0]
        print(f"[NV_StreamingVAEEncode] Encoding {total_pixel_frames} pixel frames (shape {tuple(pixels.shape)})")

        pixel_bytes = pixels.numel() * pixels.element_size()
        output = streaming_vae_encode(vae, pixels, cache_clear_interval=cache_clear_interval)

        latent_bytes = output.numel() * output.element_size()
        savings_ratio = pixel_bytes / latent_bytes if latent_bytes > 0 else 0
        print(f"[NV_StreamingVAEEncode] Done. Output latent shape: {tuple(output.shape)}")
        print(f"[NV_StreamingVAEEncode] Memory: {pixel_bytes / 1024**3:.2f} GB pixels -> "
              f"{latent_bytes / 1024**2:.1f} MB latents ({savings_ratio:.0f}x reduction)")

        return ({"samples": output},)


# Node registration
NODE_CLASS_MAPPINGS = {
    "NV_StreamingVAEEncode": NV_StreamingVAEEncode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_StreamingVAEEncode": "NV Streaming VAE Encode",
}
