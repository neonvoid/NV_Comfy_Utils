"""
NV Streaming VAE Decode

Decodes video latents frame-by-frame, streaming output to CPU to avoid OOM.
Maintains the causal feat_cache for temporal coherence (required by WAN VAE's
causal convolutions), but moves decoded frames to CPU immediately instead
of accumulating them on GPU.

Memory Comparison (200 frames @ 1080p):
- Standard decode: ~2.5 GB GPU (accumulates all frames)
- Streaming decode: ~200 MB GPU + 2.5 GB CPU (constant GPU usage)
"""

import torch
import comfy.model_management as model_management


def count_conv3d(model):
    """Count CausalConv3d modules in model for feat_cache initialization."""
    from comfy.ldm.wan.vae import CausalConv3d
    count = 0
    for m in model.modules():
        if isinstance(m, CausalConv3d):
            count += 1
    return count


class NV_StreamingVAEDecode:
    """
    Streaming VAE Decode for long videos.

    Decodes video latents frame-by-frame, moving each decoded frame to CPU
    immediately. This keeps GPU memory constant regardless of video length.

    The WAN VAE uses causal convolutions with a feat_cache that maintains
    temporal context between frames. This node preserves that cache (for
    temporal coherence) while streaming the OUTPUT to CPU (to avoid OOM).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT", {"tooltip": "The latent video to decode."}),
                "vae": ("VAE", {"tooltip": "The WAN VAE model."}),
            },
            "optional": {
                "cache_clear_interval": ("INT", {
                    "default": 16,
                    "min": 1,
                    "max": 100,
                    "tooltip": "Clear CUDA cache every N frames. Lower = more cache clears, higher = faster but uses more transient memory."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "decode_streaming"
    CATEGORY = "NV_Utils"
    DESCRIPTION = "Decodes video latents frame-by-frame with streaming to CPU. Use for long videos that OOM with standard VAEDecode."

    def decode_streaming(self, vae, samples, cache_clear_interval=16):
        """
        Decode latents frame-by-frame, streaming to CPU.

        The key insight: WAN VAE already processes frame-by-frame internally,
        but accumulates output on GPU with torch.cat. We replicate the loop
        but move each frame to CPU immediately.
        """
        z = samples["samples"]
        total_frames = z.shape[2]

        print(f"[NV_StreamingVAEDecode] Starting decode of {total_frames} latent frames")
        print(f"[NV_StreamingVAEDecode] Input shape: {z.shape}")

        # Get the actual WAN VAE model from ComfyUI's wrapper
        wan_vae = vae.first_stage_model

        # Move latent to VAE device and dtype
        device = vae.device
        vae_dtype = vae.vae_dtype
        z = z.to(vae_dtype).to(device)

        # Load the VAE model to GPU
        model_management.load_model_gpu(vae.patcher)

        # Initialize the causal convolution cache
        # This is critical for temporal coherence - each CausalConv3d needs cache state
        feat_map = [None] * count_conv3d(wan_vae.decoder)

        # Pre-process with conv2 (same as original)
        x = wan_vae.conv2(z)

        decoded_frames = []  # Store on CPU

        print(f"[NV_StreamingVAEDecode] Decoding frames...")

        for i in range(total_frames):
            # Reset conv index for each frame (same as original)
            conv_idx = [0]

            # Decode single frame with causal cache
            # The feat_cache maintains temporal context between frames
            frame = wan_vae.decoder(
                x[:, :, i:i+1, :, :],
                feat_cache=feat_map,
                feat_idx=conv_idx
            )

            # Move to CPU immediately - this is the key memory savings
            decoded_frames.append(frame.cpu())

            # Free GPU memory from this frame
            del frame

            # Periodic cache clear to prevent memory fragmentation
            if (i + 1) % cache_clear_interval == 0:
                torch.cuda.empty_cache()
                print(f"[NV_StreamingVAEDecode] Decoded {i+1}/{total_frames} frames")

        # Final progress message
        if total_frames % cache_clear_interval != 0:
            print(f"[NV_StreamingVAEDecode] Decoded {total_frames}/{total_frames} frames")

        # Concatenate on CPU (RAM is typically much larger than VRAM)
        print(f"[NV_StreamingVAEDecode] Concatenating {total_frames} frames on CPU...")
        output = torch.cat(decoded_frames, dim=2)

        # Apply VAE's output processing (clamp to [0,1])
        output = vae.process_output(output.float())

        # Convert to ComfyUI image format: [B, C, T, H, W] -> [B, T, H, W, C]
        # Standard VAEDecode does .movedim(1, -1) which moves C to the end
        output = output.movedim(1, -1)

        # If batch dim exists with temporal, reshape to [B*T, H, W, C]
        if len(output.shape) == 5:
            output = output.reshape(-1, output.shape[-3], output.shape[-2], output.shape[-1])

        print(f"[NV_StreamingVAEDecode] Done. Output shape: {output.shape}")

        # Clean up
        del decoded_frames
        del x
        torch.cuda.empty_cache()

        return (output,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "NV_StreamingVAEDecode": NV_StreamingVAEDecode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_StreamingVAEDecode": "NV Streaming VAE Decode",
}
