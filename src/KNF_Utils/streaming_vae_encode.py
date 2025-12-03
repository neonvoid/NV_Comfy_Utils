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


def count_conv3d_encoder(model):
    """Count CausalConv3d modules in encoder for feat_cache initialization."""
    from comfy.ldm.wan.vae import CausalConv3d
    count = 0
    for m in model.modules():
        if isinstance(m, CausalConv3d):
            count += 1
    return count


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
        """
        Encode pixel frames to latents in streaming fashion.

        Follows WAN VAE's temporal chunking pattern:
        - Chunk 0: frame 0 alone
        - Chunk 1: frames 1-4
        - Chunk 2: frames 5-8
        - etc.
        """
        total_pixel_frames = pixels.shape[0]

        print(f"[NV_StreamingVAEEncode] Starting encode of {total_pixel_frames} pixel frames")
        print(f"[NV_StreamingVAEEncode] Input shape: {pixels.shape}")

        # Get the actual WAN VAE model from ComfyUI's wrapper
        wan_vae = vae.first_stage_model

        # Get device and dtype
        device = vae.device
        vae_dtype = vae.vae_dtype

        # Load the VAE model to GPU
        model_management.load_model_gpu(vae.patcher)

        # Preprocess pixels: [T, H, W, C] -> [1, C, T, H, W]
        # First crop to valid encode size
        pixels = vae.vae_encode_crop_pixels(pixels)

        # Convert format: [T, H, W, C] -> [T, C, H, W] -> [1, C, T, H, W]
        x = pixels.movedim(-1, 1)  # [T, H, W, C] -> [T, C, H, W]
        x = x.movedim(1, 0).unsqueeze(0)  # [T, C, H, W] -> [1, C, T, H, W]

        # Move to VAE device and dtype
        x = x.to(vae_dtype).to(device)

        t = x.shape[2]  # Number of frames
        num_chunks = 1 + (t - 1) // 4  # WAN pattern: 1 + ceil((t-1)/4)

        print(f"[NV_StreamingVAEEncode] Processing in {num_chunks} chunks (WAN temporal pattern)")

        # Initialize feat_cache for encoder's causal convolutions
        feat_map = [None] * count_conv3d_encoder(wan_vae.encoder)

        encoded_chunks = []  # Store latent chunks on CPU

        for i in range(num_chunks):
            conv_idx = [0]

            if i == 0:
                # First chunk: frame 0 alone
                chunk_input = x[:, :, :1, :, :]
                frame_range = "0"
            else:
                # Subsequent chunks: groups of 4 frames
                start_idx = 1 + 4 * (i - 1)
                end_idx = min(1 + 4 * i, t)
                chunk_input = x[:, :, start_idx:end_idx, :, :]
                frame_range = f"{start_idx}-{end_idx-1}"

            # Encode this chunk
            chunk_out = wan_vae.encoder(
                chunk_input,
                feat_cache=feat_map,
                feat_idx=conv_idx
            )

            # Apply conv1 to get mu (latent mean) - works per-chunk since conv1 is typically 1x1
            chunk_latent, _ = wan_vae.conv1(chunk_out).chunk(2, dim=1)

            # Move to CPU immediately - this is the key memory savings
            encoded_chunks.append(chunk_latent.cpu())

            # Free GPU memory from this chunk
            del chunk_out
            del chunk_latent
            del chunk_input

            # Periodic cache clear
            if (i + 1) % cache_clear_interval == 0:
                torch.cuda.empty_cache()
                print(f"[NV_StreamingVAEEncode] Encoded chunk {i+1}/{num_chunks} (frames {frame_range})")

        # Final progress message
        if num_chunks % cache_clear_interval != 0:
            print(f"[NV_StreamingVAEEncode] Encoded {num_chunks}/{num_chunks} chunks")

        # Free the input tensor from GPU
        del x
        torch.cuda.empty_cache()

        # Concatenate latent chunks on CPU
        print(f"[NV_StreamingVAEEncode] Concatenating {num_chunks} latent chunks on CPU...")
        # IMPORTANT: Convert to float32 to match ComfyUI's standard VAE.encode() behavior
        # ComfyUI's sd.py line 782 calls .float() on encoder output
        # Without this, downstream nodes (sampler sigmas) get dtype mismatches
        output = torch.cat(encoded_chunks, dim=2).float()

        # Clean up
        del encoded_chunks

        print(f"[NV_StreamingVAEEncode] Done. Output latent shape: {output.shape}")

        # Calculate memory savings
        pixel_bytes = pixels.numel() * pixels.element_size()
        latent_bytes = output.numel() * output.element_size()
        savings_ratio = pixel_bytes / latent_bytes if latent_bytes > 0 else 0
        print(f"[NV_StreamingVAEEncode] Memory: {pixel_bytes / 1024**3:.2f} GB pixels -> {latent_bytes / 1024**2:.1f} MB latents ({savings_ratio:.0f}x reduction)")

        return ({"samples": output},)


# Node registration
NODE_CLASS_MAPPINGS = {
    "NV_StreamingVAEEncode": NV_StreamingVAEEncode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_StreamingVAEEncode": "NV Streaming VAE Encode",
}
