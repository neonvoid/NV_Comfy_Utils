"""
NV VACE Pre-Pass Reference

Multi-frame VACE reference conditioning for cascaded video generation.

Accepts multiple pre-pass reference frames (instead of WanVaceToVideo's single
reference_image) and composes them as R2V conditioning alongside the V2V control
video.

Key differences from native WanVaceToVideo:
- Multiple reference frames (not just 1)
- Frame repeat (4x default) for 3D VAE temporal compression preservation
- Uniform sampling of N reference frames from input
- Internal upscaling of reference frames to target resolution

Based on research validation:
- SkyReels-A2 (arXiv 2504.02436): frame_repeat=4 prevents 6.5% identity drop
- FlashVideo (arXiv 2502.05179): cascaded is HIGHER quality than single-stage
- HiStream (arXiv 2512.21338): cascaded validated on Wan 2.1
- VINs (arXiv 2503.17539): parallel chunks converge with shared global signal

Output format is identical to WanVaceToVideo:
- vace_frames: [1, 32, T_ref+T_ctrl, H/8, W/8]
- vace_mask: [1, 64, T_ref+T_ctrl, H/8, W/8]
- vace_strength: scalar float
"""

import torch
import comfy.utils
import comfy.model_management
import comfy.latent_formats
import node_helpers
from .streaming_vace_to_video import streaming_vae_encode


class NV_VacePrePassReference:
    """
    Multi-frame VACE reference conditioning for pre-pass cascaded generation.

    Accepts pre-pass reference frames and a control video, composing them
    into VACE conditioning where:
    - Reference frames: mask=0 (preserve), encoded with neutral reactive channel
    - Control video: mask=1 (generate), encoded with concept decoupling

    Use this for cascaded workflows where a low-res pre-pass pins style,
    and parallel high-res chunks reference those frames for consistency.
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
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01,
                            "tooltip": "VACE conditioning strength for the control video. "
                                       "This is the entry-level vace_strength applied to both references and control."}),
                "ref_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01,
                                "tooltip": "Effective strength for reference frames. When different from 'strength', "
                                           "reference latents are pre-scaled so their effective influence equals this value "
                                           "after the model applies vace_strength. E.g., strength=0.35 + ref_strength=1.0 "
                                           "means beauty control at 0.35, references at 1.0."}),
                "reference_frames": ("IMAGE", {
                    "tooltip": "Pre-pass reference frames (any resolution, will be upscaled to target). "
                               "num_refs_per_chunk frames will be uniformly sampled from these."
                }),
                "num_refs_per_chunk": ("INT", {"default": 5, "min": 1, "max": 20, "step": 1,
                                      "tooltip": "How many reference frames to sample from reference_frames"}),
                "frame_repeat": ("INT", {"default": 4, "min": 1, "max": 8, "step": 1,
                                "tooltip": "Repeat each reference N times for 3D VAE temporal compression "
                                           "(4 recommended for Wan 2.1, prevents 6.5%% identity drop per SkyReels-A2)"}),
            },
            "optional": {
                "control_video": ("IMAGE", {
                    "tooltip": "Control video for VACE V2V conditioning (e.g., 3D render). "
                               "Will be sliced to 'length' frames."
                }),
                "control_masks": ("MASK", {
                    "tooltip": "Optional mask for control video regions."
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "LATENT", "INT",)
    RETURN_NAMES = ("positive", "negative", "latent", "ref_latent", "trim_latent",)
    FUNCTION = "execute"
    CATEGORY = "NV_Utils"
    DESCRIPTION = (
        "Multi-frame VACE reference for cascaded pre-pass workflows. "
        "Samples reference frames from a pre-pass video and composes them "
        "with V2V control conditioning. Connect pre-pass output to reference_frames "
        "and 3D render to control_video."
    )

    def execute(self, positive, negative, vae, width, height, length, batch_size, strength,
                ref_strength, reference_frames, num_refs_per_chunk, frame_repeat,
                control_video=None, control_masks=None):

        latent_length = ((length - 1) // 4) + 1

        # === Step 1: Sample and repeat reference frames ===
        total_refs = reference_frames.shape[0]
        num_refs = min(num_refs_per_chunk, total_refs)

        # Uniformly sample frame indices for temporal coverage
        if num_refs >= total_refs:
            ref_indices = list(range(total_refs))
        else:
            ref_indices = torch.linspace(0, total_refs - 1, num_refs).long().tolist()

        sampled_refs = reference_frames[ref_indices]  # [num_refs, H_in, W_in, C]

        # Frame repeat: duplicate each reference frame_repeat times in pixel space
        # [r1,r1,r1,r1, r2,r2,r2,r2, ...] for 3D VAE temporal compression
        # SkyReels-A2: "before VAE" assembly with 4x repeat is critical
        if frame_repeat > 1:
            repeated = sampled_refs.unsqueeze(1).expand(-1, frame_repeat, -1, -1, -1)
            repeated = repeated.reshape(-1, *sampled_refs.shape[1:])
        else:
            repeated = sampled_refs

        total_ref_pixels = repeated.shape[0]
        ref_latent_length = ((total_ref_pixels - 1) // 4) + 1

        print(f"[NV_VacePrePassReference] Sampled {num_refs} reference frames "
              f"(indices: {ref_indices}) from {total_refs} available")
        print(f"[NV_VacePrePassReference] Frame repeat {frame_repeat}x -> "
              f"{total_ref_pixels} pixel frames -> {ref_latent_length} latent frames")

        # Upscale reference frames to target resolution
        repeated = comfy.utils.common_upscale(
            repeated.movedim(-1, 1), width, height, "bilinear", "center"
        ).movedim(1, -1)

        # VAE encode reference frames (streaming for OOM safety)
        print(f"[NV_VacePrePassReference] Encoding {total_ref_pixels} reference frames...")
        ref_latent = streaming_vae_encode(vae, repeated[:, :, :, :3])
        # ref_latent shape: [1, 16, ref_latent_length, H/8, W/8]

        # Free pixel-space reference tensors
        del repeated, sampled_refs

        # Build 32ch reference: 16ch encoded + 16ch neutral reactive
        # This matches native WanVaceToVideo reference_image encoding:
        #   cat([ref_latent, process_out(zeros_like(ref_latent))], dim=1)
        # The neutral reactive channel signals "nothing to generate here"
        ref_latent = torch.cat([
            ref_latent,
            comfy.latent_formats.Wan21().process_out(torch.zeros_like(ref_latent))
        ], dim=1)
        # ref_latent shape: [1, 32, ref_latent_length, H/8, W/8]

        # === Step 1b: Pre-scale reference latent for independent strength control ===
        # The entry-level vace_strength applies uniformly to the entire VACE entry
        # (both references and control frames). To give references a different effective
        # strength, we pre-scale the reference latent so that:
        #   effective_ref = ref_latent * ref_scale * vace_strength = ref_latent * ref_strength
        # This allows e.g. beauty control at 0.35 with references at 1.0
        if strength > 0 and abs(ref_strength - strength) > 1e-6:
            ref_scale = ref_strength / strength
            ref_latent = ref_latent * ref_scale
            print(f"[NV_VacePrePassReference] Pre-scaled reference latent by {ref_scale:.2f}x "
                  f"(ref_strength={ref_strength}, vace_strength={strength}, "
                  f"effective reference influence={ref_strength})")

        # === Step 2: Process control video (same as native WanVaceToVideo) ===
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

        # === Step 3: Process masks ===
        if control_masks is None:
            mask = torch.ones((length, height, width, 1))
        else:
            mask = control_masks
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            mask = comfy.utils.common_upscale(
                mask[:length], width, height, "bilinear", "center"
            ).movedim(1, -1)
            if mask.shape[0] < length:
                mask = torch.nn.functional.pad(
                    mask,
                    (0, 0, 0, 0, 0, 0, 0, length - mask.shape[0]),
                    value=1.0
                )

        # === Step 4: Concept decoupling for control video ===
        # Split into inactive (preserved regions) and reactive (generated regions)
        control_video = control_video - 0.5
        inactive = (control_video * (1 - mask)) + 0.5
        reactive = (control_video * mask) + 0.5

        del control_video  # Free pixel-space control video

        # VAE encode control video (streaming for long videos)
        print(f"[NV_VacePrePassReference] Encoding {length} control frames (inactive)...")
        inactive = streaming_vae_encode(vae, inactive[:, :, :, :3])
        print(f"[NV_VacePrePassReference] Encoding {length} control frames (reactive)...")
        reactive = streaming_vae_encode(vae, reactive[:, :, :, :3])

        control_video_latent = torch.cat((inactive, reactive), dim=1)
        # control_video_latent shape: [1, 32, latent_length, H/8, W/8]

        del inactive, reactive  # Free intermediate latents

        # === Step 5: Prepend reference latent to control latent ===
        control_video_latent = torch.cat((ref_latent, control_video_latent), dim=2)
        # combined shape: [1, 32, ref_latent_length + latent_length, H/8, W/8]

        del ref_latent  # Free standalone reference latent

        # === Step 6: Mask downscaling to latent space ===
        # Same logic as native WanVaceToVideo (nodes_wan.py lines 346-352)
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

        # Prepend zero-mask for reference frames (mask=0 = preserve)
        mask_pad = torch.zeros_like(mask[:, :ref_latent_length, :, :])
        mask = torch.cat((mask_pad, mask), dim=1)

        # Update total latent length
        total_latent_length = latent_length + ref_latent_length
        trim_latent = ref_latent_length

        mask = mask.unsqueeze(0)

        # === Step 7: Set conditioning ===
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

        # === Step 8: Create output latents ===
        intermediate_device = comfy.model_management.intermediate_device()
        latent = torch.zeros(
            [batch_size, 16, total_latent_length, height // 8, width // 8],
            device=intermediate_device
        )
        out_latent = {"samples": latent}

        # Reference-only latent: zero frames matching the prepended reference region.
        # Use this with NV_LatentTemporalConcat to prepend to a custom denoised latent
        # when bypassing the main latent output (e.g., low-denoise input latent workflows).
        ref_only_latent = torch.zeros(
            [batch_size, 16, ref_latent_length, height // 8, width // 8],
            device=intermediate_device
        )
        out_ref_latent = {"samples": ref_only_latent}

        print(f"[NV_VacePrePassReference] Done.")
        print(f"  VACE latent shape: {control_video_latent.shape}")
        print(f"  VACE mask shape: {mask.shape}")
        print(f"  Output latent shape: {latent.shape}")
        print(f"  ref_latent shape: {ref_only_latent.shape} (prepend to custom latent input)")
        print(f"  trim_latent: {trim_latent} (reference latent frames to trim from output)")

        return (positive, negative, out_latent, out_ref_latent, trim_latent)


# Node registration
NODE_CLASS_MAPPINGS = {
    "NV_VacePrePassReference": NV_VacePrePassReference,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_VacePrePassReference": "NV VACE Pre-Pass Reference",
}
