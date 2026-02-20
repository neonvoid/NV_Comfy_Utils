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

        # Validate input latent shape
        if len(z.shape) != 5:
            raise ValueError(
                f"[NV_StreamingVAEDecode] Expected 5D latent [B, C, T, H, W], "
                f"but got shape: {z.shape}. "
                f"Check that the upstream sampler produced valid WAN video latents."
            )

        total_frames = z.shape[2]

        # Validate temporal dimension
        if total_frames == 0:
            raise ValueError(
                f"[NV_StreamingVAEDecode] Input latent has 0 temporal frames! "
                f"Shape: {z.shape}. "
                f"The upstream sampler produced empty output. "
                f"This may indicate a context window or sampling issue."
            )

        print(f"[NV_StreamingVAEDecode] Starting decode of {total_frames} latent frames")
        print(f"[NV_StreamingVAEDecode] Input shape: {z.shape}")

        # Get the actual WAN VAE model from ComfyUI's wrapper
        wan_vae = vae.first_stage_model

        # Get device and dtype
        device = vae.device
        vae_dtype = vae.vae_dtype

        # CRITICAL: Clear GPU memory before starting
        # After sampling, there may be residual memory that needs to be freed
        torch.cuda.empty_cache()

        # Load the VAE model to GPU
        model_management.load_model_gpu(vae.patcher)

        # ============================================================
        # STREAMING DECODE PATH (uses manual conv2 + decoder with feat_cache)
        # ============================================================

        # Initialize the causal convolution cache
        # This is critical for temporal coherence - each CausalConv3d needs cache state
        feat_map = [None] * count_conv3d(wan_vae.decoder)

        # ============================================================
        # MEMORY-EFFICIENT conv2 PREPROCESSING
        # ============================================================
        # conv2 is CausalConv3d(z_dim, z_dim, 1) with kernel size 1.
        # Kernel size 1 means _padding[4] = 0, so NO temporal cache is used.
        # Therefore, we can safely apply conv2 per-frame for memory efficiency.
        # This produces mathematically identical results to all-at-once.
        # ============================================================

        print(f"[NV_StreamingVAEDecode] Applying conv2 per-frame...")
        conv2_outputs = []  # Store on CPU

        for i in range(total_frames):
            # Load single latent frame to GPU
            z_frame = z[:, :, i:i+1, :, :].to(vae_dtype).to(device)

            # Apply conv2 (1x1 conv, no temporal dependencies)
            x_frame = wan_vae.conv2(z_frame)

            # Store on CPU immediately
            conv2_outputs.append(x_frame.cpu())

            # Free GPU memory
            del z_frame, x_frame

            # Periodic cache clear
            if (i + 1) % cache_clear_interval == 0:
                torch.cuda.empty_cache()

        print(f"[NV_StreamingVAEDecode] conv2 complete. Decoding frames...")

        # ============================================================
        # FRAME-BY-FRAME DECODING
        # ============================================================

        decoded_frames = []  # Store on CPU

        for i in range(total_frames):
            # Reset conv index for each frame (same as original)
            conv_idx = [0]

            # Load conv2 output for this frame
            x_frame = conv2_outputs[i].to(device)

            # Decode single frame with causal cache
            # The feat_cache maintains temporal context between frames
            frame = wan_vae.decoder(
                x_frame,
                feat_cache=feat_map,
                feat_idx=conv_idx
            )

            # Move to CPU immediately - this is the key memory savings
            decoded_frames.append(frame.cpu())

            # Free GPU memory from this frame
            del frame, x_frame

            # Periodic cache clear to prevent memory fragmentation
            if (i + 1) % cache_clear_interval == 0:
                torch.cuda.empty_cache()
                print(f"[NV_StreamingVAEDecode] Decoded {i+1}/{total_frames} frames")

        # Final progress message
        if total_frames % cache_clear_interval != 0:
            print(f"[NV_StreamingVAEDecode] Decoded {total_frames}/{total_frames} frames")

        # Clean up conv2 outputs
        del conv2_outputs

        # Concatenate on CPU (RAM is typically much larger than VRAM)
        print(f"[NV_StreamingVAEDecode] Concatenating {total_frames} frames on CPU...")
        output = torch.cat(decoded_frames, dim=2)

        # Free decoded_frames BEFORE creating float32 copy to reduce peak RAM
        del decoded_frames

        # Apply VAE's output processing in-place to avoid ~8GB of temporaries
        # Standard process_output is: torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)
        # Using in-place ops avoids 3 intermediate full-size allocations
        output = output.float()
        output.add_(1.0).div_(2.0).clamp_(0.0, 1.0)

        # Convert to ComfyUI image format: [B, C, T, H, W] -> [B*T, H, W, C]
        # Use permute + contiguous instead of movedim + reshape to do it in one copy
        if len(output.shape) == 5:
            B, C, T, H, W = output.shape
            output = output.permute(0, 2, 3, 4, 1).contiguous().reshape(B * T, H, W, C)
        else:
            output = output.movedim(1, -1)

        print(f"[NV_StreamingVAEDecode] Done. Output shape: {output.shape}")

        torch.cuda.empty_cache()

        return (output,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "NV_StreamingVAEDecode": NV_StreamingVAEDecode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_StreamingVAEDecode": "NV Streaming VAE Decode",
}
