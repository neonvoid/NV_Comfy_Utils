"""
NV RoPE Utilities

Nodes for manipulating RoPE (Rotary Position Embedding) settings,
particularly temporal position offsets for chunked video processing.

When processing video in chunks, the model needs to know the absolute
temporal position of each chunk. Without shift_t, every chunk starts
at position 0, causing the model to treat each chunk as the "beginning"
of the video.
"""

from .chunk_utils import video_to_latent_frames


class NV_ApplyRoPEShiftT:
    """
    Apply RoPE temporal shift to model for chunk position awareness.

    When processing video chunks, the model needs to know the absolute
    temporal position of each chunk. shift_t tells the model where in
    the overall video this chunk starts.

    Example:
    - Chunk 0 (frames 0-80): shift_t = 0
    - Chunk 1 (frames 65-145): shift_t = 16 (latent frame position)
    - Chunk 2 (frames 130-200): shift_t = 32

    This helps maintain temporal coherence across chunk boundaries.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "Model to apply RoPE shift to"
                }),
                "shift_t_frames": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "tooltip": "Temporal offset in VIDEO frames (auto-converted to latent frames)"
                }),
            },
            "optional": {
                "scale_t": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Temporal scale factor (1.0 = normal speed, 2.0 = half speed perception)"
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "NV_Utils/model_patches"
    DESCRIPTION = "Apply RoPE temporal shift for chunk position awareness. Converts video frames to latent frames internally."

    def apply(self, model, shift_t_frames: int, scale_t: float = 1.0):
        # Convert video frames to latent frames
        shift_t_latent = video_to_latent_frames(shift_t_frames) if shift_t_frames > 0 else 0.0

        # Clone the model to avoid affecting other uses
        m = model.clone()

        # Apply RoPE options
        m.set_model_rope_options(
            scale_x=1.0, shift_x=0.0,
            scale_y=1.0, shift_y=0.0,
            scale_t=scale_t, shift_t=float(shift_t_latent)
        )

        print(f"[NV_ApplyRoPEShiftT] Applied shift_t={shift_t_latent} latent frames "
              f"(from {shift_t_frames} video frames), scale_t={scale_t}")

        return (m,)


class NV_ApplyRoPEShiftTLatent:
    """
    Apply RoPE temporal shift using LATENT frame position directly.

    Use this when you already know the latent frame position (e.g., from
    chunk metadata that's already in latent space).

    For most users, NV_ApplyRoPEShiftT (which takes video frames) is easier.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "Model to apply RoPE shift to"
                }),
                "shift_t_latent": ("FLOAT", {
                    "default": 0.0,
                    "min": -256.0,
                    "max": 1000.0,
                    "step": 1.0,
                    "tooltip": "Temporal offset in LATENT frames (no conversion)"
                }),
            },
            "optional": {
                "scale_t": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Temporal scale factor"
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "NV_Utils/model_patches"
    DESCRIPTION = "Apply RoPE temporal shift using latent frame position directly."

    def apply(self, model, shift_t_latent: float, scale_t: float = 1.0):
        m = model.clone()
        m.set_model_rope_options(
            scale_x=1.0, shift_x=0.0,
            scale_y=1.0, shift_y=0.0,
            scale_t=scale_t, shift_t=shift_t_latent
        )

        print(f"[NV_ApplyRoPEShiftTLatent] Applied shift_t={shift_t_latent} latent frames, scale_t={scale_t}")

        return (m,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "NV_ApplyRoPEShiftT": NV_ApplyRoPEShiftT,
    "NV_ApplyRoPEShiftTLatent": NV_ApplyRoPEShiftTLatent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_ApplyRoPEShiftT": "NV Apply RoPE Shift T (Video Frames)",
    "NV_ApplyRoPEShiftTLatent": "NV Apply RoPE Shift T (Latent Frames)",
}
