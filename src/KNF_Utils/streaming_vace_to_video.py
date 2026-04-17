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


def _cache_layer_count(model):
    """Count layers that contribute a slot to WAN VAE's feat_cache.

    Mirrors `count_cache_layers()` in comfy/ldm/wan/vae.py — CausalConv3d layers
    plus Resample layers running in downsample3d mode. Falls back to a scan over
    CausalConv3d only if the newer helper is unavailable (older cores).
    """
    from comfy.ldm.wan.vae import CausalConv3d, Resample
    count = 0
    for m in model.modules():
        if isinstance(m, CausalConv3d):
            count += 1
        elif isinstance(m, Resample) and getattr(m, "mode", None) == 'downsample3d':
            count += 1
    return count


def streaming_vae_encode(vae, pixels, cache_clear_interval=4):
    """Streaming WAN VAE encode that mirrors native _encode while spilling to CPU.

    Matches the exact chunking pattern from comfy/ldm/wan/vae.py's _encode:
    - Frame-count clamp to 4k+1, then split into iter_ = 1 + (t-1)//2 chunks
    - Chunk 0 encodes frame[0]; chunk i>0 encodes frames [1+2(i-1) : 1+2i]
    - `final=True` only on the last iteration (gates the encoder's internal flush)
    - Intermediate chunks may legitimately return None while the causal cache
      warms up — skip those, same as native
    - feat_map sized from count_cache_layers(encoder) (NOT decoder as old code did)
    - conv1 + chunk(2) applied once on concatenated encoder outputs to yield mu

    Memory story vs. native: all encoder chunk outputs accumulate on CPU and only
    move back to GPU for the final conv1 projection, so peak VRAM stays flat as
    input length grows.

    Args:
        vae: ComfyUI VAE wrapper (expects .first_stage_model = WanVAE)
        pixels: Tensor [T, H, W, C] in range [0, 1]
        cache_clear_interval: torch.cuda.empty_cache() every N chunks

    Returns:
        Tensor [1, C_z, T_latent, H/8, W/8] on vae.output_device, float32 —
        bit-identical to vae.encode() up to CPU↔GPU round-trip ordering.
    """
    wan_vae = vae.first_stage_model
    device = vae.device
    vae_dtype = vae.vae_dtype

    comfy.model_management.load_model_gpu(vae.patcher)

    pixels = vae.vae_encode_crop_pixels(pixels)
    x = pixels.movedim(-1, 1)            # [T, H, W, C] -> [T, C, H, W]
    x = x.movedim(1, 0).unsqueeze(0)     # -> [1, C, T, H, W]
    x = vae.process_input(x)             # [0, 1] -> [-1, 1]
    x = x.to(vae_dtype).to(device)

    # Native encode clamps input length to 4k+1 before chunking.
    # See comfy/ldm/wan/vae.py _encode: t = 1 + ((t - 1) // 4) * 4
    t_raw = x.shape[2]
    t = 1 + ((t_raw - 1) // 4) * 4 if t_raw > 0 else 0
    if t <= 0:
        raise ValueError(f"streaming_vae_encode: empty input (T={t_raw})")

    iter_ = 1 + (t - 1) // 2  # 2-frame chunks after first 1-frame primer
    feat_map = [None] * _cache_layer_count(wan_vae.encoder) if iter_ > 1 else None

    encoder_outputs = []  # accumulated on CPU between iterations

    for i in range(iter_):
        conv_idx = [0]
        if i == 0:
            chunk = x[:, :, :1, :, :]
            out = wan_vae.encoder(chunk, feat_cache=feat_map, feat_idx=conv_idx)
        else:
            start = 1 + 2 * (i - 1)
            end = 1 + 2 * i
            chunk = x[:, :, start:end, :, :]
            out = wan_vae.encoder(
                chunk,
                feat_cache=feat_map,
                feat_idx=conv_idx,
                final=(i == iter_ - 1),
            )
            if out is None:
                # Causal cache still warming up — native _encode does `continue` here.
                del chunk
                continue

        encoder_outputs.append(out.cpu())
        del out, chunk

        if (i + 1) % cache_clear_interval == 0:
            torch.cuda.empty_cache()

    del x
    torch.cuda.empty_cache()

    if not encoder_outputs:
        raise RuntimeError(
            "streaming_vae_encode: encoder produced no chunks — "
            "input may be too short for WAN VAE streaming path"
        )

    # conv1 applied once over the full temporal tensor — CausalConv3d needs full context
    full = torch.cat(encoder_outputs, dim=2).to(vae_dtype).to(device)
    del encoder_outputs

    mu, _ = wan_vae.conv1(full).chunk(2, dim=1)
    del full
    torch.cuda.empty_cache()

    output = mu.cpu().float()
    del mu

    # Clear cached tensors from feat_map so they don't survive to the next call
    # (each VACE node does 2 encodes: inactive + reactive)
    if feat_map is not None:
        for j in range(len(feat_map)):
            feat_map[j] = None
        feat_map.clear()
    torch.cuda.empty_cache()

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
