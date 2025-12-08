"""
NV Streaming VACE To Video

A streaming version of WanVaceToVideo that uses chunk-by-chunk VAE encoding
to prevent CPU/GPU OOM on long videos.

The native WanVaceToVideo node (nodes_wan.py) calls vae.encode() on the full
control video at once, which allocates all output latents at once. For long
videos (400+ frames), this can exhaust CPU RAM.

This node is an exact copy of WanVaceToVideo, except:
- Lines 340-341: vae.encode() → streaming_vae_encode()
- Streaming encode processes frames in WAN's temporal chunk pattern

Key implementation details (matching streaming_vae_encode.py):
- process_input: Scales pixels from [0,1] to [-1,1] before encoding
- feat_map count: Uses DECODER count (not encoder) per WAN VAE behavior
- conv1 application: Applied ONCE to full tensor, not per-chunk
  (conv1 is CausalConv3d requiring full temporal context for correct color)

Output format is identical to WanVaceToVideo:
- vace_frames: [1, 32, T_latent, H/8, W/8] (inactive 16ch + reactive 16ch)
- vace_mask: [1, 64, T_latent, H/8, W/8]
- vace_strength: scalar float
"""

import torch
import comfy.utils
import comfy.model_management
import comfy.latent_formats
import node_helpers


def count_conv3d(model):
    """Count CausalConv3d modules in model for feat_cache initialization.

    Note: Native WAN VAE uses decoder count for BOTH encode and decode operations.
    This matches the behavior in comfy/ldm/wan/vae.py line 472.
    """
    from comfy.ldm.wan.vae import CausalConv3d
    count = 0
    for m in model.modules():
        if isinstance(m, CausalConv3d):
            count += 1
    return count


def streaming_vae_encode(vae, pixels, cache_clear_interval=4):
    """
    Encode pixels using streaming pattern.

    Exactly replicates vae.encode() output format but processes frames in chunks
    to avoid allocating all output latents at once.

    Args:
        vae: ComfyUI VAE wrapper
        pixels: Tensor [T, H, W, C] in range [0, 1]
        cache_clear_interval: Clear CUDA cache every N chunks

    Returns:
        Tensor matching vae.encode() output format [1, C, T_latent, H/8, W/8]
    """
    total_pixel_frames = pixels.shape[0]

    # Get the actual WAN VAE model from ComfyUI's wrapper
    wan_vae = vae.first_stage_model

    # Get device and dtype
    device = vae.device
    vae_dtype = vae.vae_dtype

    # Load the VAE model to GPU if not already
    comfy.model_management.load_model_gpu(vae.patcher)

    # Preprocess pixels: [T, H, W, C] -> [1, C, T, H, W]
    # First crop to valid encode size
    pixels = vae.vae_encode_crop_pixels(pixels)

    # Convert format: [T, H, W, C] -> [T, C, H, W] -> [1, C, T, H, W]
    x = pixels.movedim(-1, 1)  # [T, H, W, C] -> [T, C, H, W]
    x = x.movedim(1, 0).unsqueeze(0)  # [T, C, H, W] -> [1, C, T, H, W]

    # CRITICAL: Apply process_input to scale pixels from [0,1] to [-1,1]
    # Native encode (comfy/sd.py line 781) does: self.process_input(pixel_samples)
    # Default process_input is: image * 2.0 - 1.0
    # Without this, colors are completely wrong (washed out)!
    x = vae.process_input(x)

    # Move to VAE device and dtype
    x = x.to(vae_dtype).to(device)

    t = x.shape[2]  # Number of frames
    num_chunks = 1 + (t - 1) // 4  # WAN pattern: 1 + ceil((t-1)/4)

    # Initialize feat_cache for causal convolutions
    # IMPORTANT: Native WAN VAE uses DECODER count for both encode and decode!
    # See comfy/ldm/wan/vae.py line 472: feat_map = [None] * count_conv3d(self.decoder)
    feat_map = [None] * count_conv3d(wan_vae.decoder)

    # Store ENCODER outputs (not latents!) on CPU
    # We must apply conv1 to the FULL accumulated tensor, not per-chunk
    # conv1 is CausalConv3d which needs temporal context for correct color
    encoder_outputs = []

    for i in range(num_chunks):
        conv_idx = [0]

        if i == 0:
            # First chunk: frame 0 alone
            chunk_input = x[:, :, :1, :, :]
        else:
            # Subsequent chunks: groups of 4 frames
            start_idx = 1 + 4 * (i - 1)
            end_idx = min(1 + 4 * i, t)
            chunk_input = x[:, :, start_idx:end_idx, :, :]

        # Encode this chunk
        chunk_out = wan_vae.encoder(
            chunk_input,
            feat_cache=feat_map,
            feat_idx=conv_idx
        )

        # Move encoder output to CPU (NOT latent - we apply conv1 later)
        encoder_outputs.append(chunk_out.cpu())

        # Free GPU memory from this chunk
        del chunk_out
        del chunk_input

        # Periodic cache clear
        if (i + 1) % cache_clear_interval == 0:
            torch.cuda.empty_cache()

    # Free the input tensor from GPU
    del x
    torch.cuda.empty_cache()

    # Concatenate encoder outputs on CPU
    full_encoder_out = torch.cat(encoder_outputs, dim=2)
    del encoder_outputs

    # Move back to GPU for conv1 (must apply to full temporal tensor for correct color)
    full_encoder_out = full_encoder_out.to(vae_dtype).to(device)

    # Apply conv1 ONCE to full tensor (matches native WAN VAE behavior)
    # This is critical - conv1 is CausalConv3d which needs full temporal context
    mu, _ = wan_vae.conv1(full_encoder_out).chunk(2, dim=1)
    del full_encoder_out
    torch.cuda.empty_cache()

    # IMPORTANT: Convert to float32 to match ComfyUI's standard VAE.encode() behavior
    output = mu.cpu().float()
    del mu

    # CRITICAL: Clean up feat_map to free GPU memory from CausalConv3d caching
    # Without this, cached tensors accumulate across multiple encode calls
    # (Each VACE node does 2 encodes: inactive + reactive)
    # Note: Set to None first (deleting by index shifts list), then clear
    for i in range(len(feat_map)):
        feat_map[i] = None
    feat_map.clear()
    del feat_map
    torch.cuda.empty_cache()

    # Move to VAE's output_device to match native VAE.encode() behavior
    # Native VAE.encode() uses: out.to(self.output_device).float()
    # Without this, VACE conditioning contains CPU tensors which causes hangs
    # when used with context windows (device mismatch during sampling)
    return output.to(vae.output_device)


class NV_WanVaceToVideoStreaming:
    """
    Streaming version of WanVaceToVideo for long videos.

    This is an exact copy of ComfyUI's WanVaceToVideo node, except the VAE encode
    calls use streaming (chunk-by-chunk) encoding to prevent CPU OOM.

    Use this instead of WanVaceToVideo when processing video chunks > 200 frames.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "width": ("INT", {"default": 832, "min": 16, "max": 8192, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": 8192, "step": 16}),
                "length": ("INT", {"default": 81, "min": 1, "max": 8192, "step": 4,
                           "tooltip": "Number of output video frames (pixel space)"}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
            },
            "optional": {
                "control_video": ("IMAGE", {
                    "tooltip": "Control video for VACE conditioning. Will be sliced to 'length' frames."
                }),
                "control_masks": ("MASK", {
                    "tooltip": "Optional mask for control video regions."
                }),
                "reference_image": ("IMAGE", {
                    "tooltip": "Optional reference image to prepend to conditioning."
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "INT",)
    RETURN_NAMES = ("positive", "negative", "latent", "trim_latent",)
    FUNCTION = "execute"
    CATEGORY = "NV_Utils"
    DESCRIPTION = "Streaming version of WanVaceToVideo. Use for long videos (400+ frames) to prevent CPU OOM."

    def execute(self, positive, negative, vae, width, height, length, batch_size, strength,
                control_video=None, control_masks=None, reference_image=None):
        """
        Execute VACE conditioning with streaming VAE encode.

        This is an exact copy of WanVaceToVideo.execute() from nodes_wan.py,
        with only lines 340-341 changed to use streaming_vae_encode().
        """

        # Line 313: Calculate latent length
        latent_length = ((length - 1) // 4) + 1

        # Lines 314-319: Control video preprocessing
        if control_video is not None:
            control_video = comfy.utils.common_upscale(
                control_video[:length].movedim(-1, 1),
                width, height, "bilinear", "center"
            ).movedim(1, -1)
            if control_video.shape[0] < length:
                control_video = torch.nn.functional.pad(
                    control_video,
                    (0, 0, 0, 0, 0, 0, 0, length - control_video.shape[0]),
                    value=0.5
                )
        else:
            control_video = torch.ones((length, height, width, 3)) * 0.5

        # Lines 321-324: Reference image handling
        if reference_image is not None:
            reference_image = comfy.utils.common_upscale(
                reference_image[:1].movedim(-1, 1),
                width, height, "bilinear", "center"
            ).movedim(1, -1)
            reference_image = vae.encode(reference_image[:, :, :, :3])
            reference_image = torch.cat([
                reference_image,
                comfy.latent_formats.Wan21().process_out(torch.zeros_like(reference_image))
            ], dim=1)

        # Lines 326-334: Mask preprocessing
        if control_masks is None:
            mask = torch.ones((length, height, width, 1))
        else:
            mask = control_masks
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            mask = comfy.utils.common_upscale(
                mask[:length],
                width, height, "bilinear", "center"
            ).movedim(1, -1)
            if mask.shape[0] < length:
                mask = torch.nn.functional.pad(
                    mask,
                    (0, 0, 0, 0, 0, 0, 0, length - mask.shape[0]),
                    value=1.0
                )

        # Lines 336-338: Split into inactive/reactive
        control_video = control_video - 0.5
        inactive = (control_video * (1 - mask)) + 0.5
        reactive = (control_video * mask) + 0.5

        # Lines 340-341: VAE ENCODE - THIS IS THE ONLY CHANGE!
        # Original: inactive = vae.encode(inactive[:, :, :, :3])
        # Original: reactive = vae.encode(reactive[:, :, :, :3])
        print(f"[NV_WanVaceToVideoStreaming] Streaming encode of {length} frames (inactive)...")
        inactive = streaming_vae_encode(vae, inactive[:, :, :, :3])
        print(f"[NV_WanVaceToVideoStreaming] Streaming encode of {length} frames (reactive)...")
        reactive = streaming_vae_encode(vae, reactive[:, :, :, :3])

        # Line 342: Concatenate latents
        control_video_latent = torch.cat((inactive, reactive), dim=1)

        # Lines 343-344: Add reference image if present
        if reference_image is not None:
            control_video_latent = torch.cat((reference_image, control_video_latent), dim=2)

        # Lines 346-352: Mask downscaling to latent space
        vae_stride = 8
        height_mask = height // vae_stride
        width_mask = width // vae_stride
        mask = mask.view(length, height_mask, vae_stride, width_mask, vae_stride)
        mask = mask.permute(2, 4, 0, 1, 3)
        mask = mask.reshape(vae_stride * vae_stride, length, height_mask, width_mask)
        mask = torch.nn.functional.interpolate(
            mask.unsqueeze(0),
            size=(latent_length, height_mask, width_mask),
            mode='nearest-exact'
        ).squeeze(0)

        # Lines 354-359: Reference image mask padding
        trim_latent = 0
        if reference_image is not None:
            mask_pad = torch.zeros_like(mask[:, :reference_image.shape[2], :, :])
            mask = torch.cat((mask_pad, mask), dim=1)
            latent_length += reference_image.shape[2]
            trim_latent = reference_image.shape[2]

        # Line 361: Unsqueeze mask
        mask = mask.unsqueeze(0)

        # Lines 363-364: Apply to conditioning
        positive = node_helpers.conditioning_set_values(
            positive,
            {"vace_frames": [control_video_latent], "vace_mask": [mask], "vace_strength": [strength]},
            append=True
        )
        negative = node_helpers.conditioning_set_values(
            negative,
            {"vace_frames": [control_video_latent], "vace_mask": [mask], "vace_strength": [strength]},
            append=True
        )

        # Lines 366-368: Create output latent
        latent = torch.zeros(
            [batch_size, 16, latent_length, height // 8, width // 8],
            device=comfy.model_management.intermediate_device()
        )
        out_latent = {}
        out_latent["samples"] = latent

        print(f"[NV_WanVaceToVideoStreaming] Done. VACE conditioning ready.")
        print(f"  control_video_latent shape: {control_video_latent.shape}")
        print(f"  vace_mask shape: {mask.shape}")

        # Line 369: Return
        return (positive, negative, out_latent, trim_latent)


# Node registration
NODE_CLASS_MAPPINGS = {
    "NV_WanVaceToVideoStreaming": NV_WanVaceToVideoStreaming,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_WanVaceToVideoStreaming": "NV VACE To Video (Streaming)",
}
