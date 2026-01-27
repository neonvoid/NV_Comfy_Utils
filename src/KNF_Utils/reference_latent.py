"""
NV Reference Latent Injection (Phase 2)

Provides attention context from neighboring chunks during sampling.
This allows each chunk to "see" frames from other chunks during
self-attention, improving temporal consistency.

How it works:
1. Reference latent is concatenated to the attention sequence
2. Current chunk attends to both its own frames AND reference frames
3. Only current chunk frames are output (reference is stripped)

The Wan model applies RoPE with temporal offset to reference frames
(t_start=max(30, time+9)) so they don't interfere with main sequence
position encoding.
"""

import torch
import node_helpers


class NV_ApplyReferenceLatent:
    """
    Inject reference latent into conditioning for attention context.

    Use this to provide neighboring chunk context during sampling.
    The reference latent participates in self-attention but is not
    included in the output.

    Example workflow for chunk consistency:
    1. Encode neighboring chunks to latent
    2. Apply this node to inject reference
    3. Sample current chunk with enriched attention context
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING", {
                    "tooltip": "Positive conditioning to modify"
                }),
                "negative": ("CONDITIONING", {
                    "tooltip": "Negative conditioning to modify"
                }),
                "reference_latent": ("LATENT", {
                    "tooltip": "Latent frames to use as attention context (from neighboring chunks)"
                }),
            },
            "optional": {
                "apply_to_negative": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Also apply reference to negative conditioning (recommended)"
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "apply"
    CATEGORY = "NV_Utils/conditioning"
    DESCRIPTION = "Inject reference latent for attention context. Reference frames participate in self-attention but are not output."

    def apply(
        self,
        positive,
        negative,
        reference_latent: dict,
        apply_to_negative: bool = True
    ):
        ref_latent = reference_latent["samples"]

        # Log reference shape
        if ref_latent.ndim == 5:
            ref_frames = ref_latent.shape[2]
            print(f"[NV_ApplyReferenceLatent] Injecting reference: {ref_frames} latent frames, "
                  f"shape {list(ref_latent.shape)}")
        else:
            print(f"[NV_ApplyReferenceLatent] Injecting reference: shape {list(ref_latent.shape)}")

        # Apply reference_latents to conditioning
        # Note: reference_latents is a LIST of latents (model uses [-1])
        positive_out = node_helpers.conditioning_set_values(
            positive,
            {"reference_latents": [ref_latent]},
            append=True
        )

        if apply_to_negative:
            negative_out = node_helpers.conditioning_set_values(
                negative,
                {"reference_latents": [ref_latent]},
                append=True
            )
        else:
            negative_out = negative

        return (positive_out, negative_out)


class NV_ApplyReferenceLatentZero:
    """
    Apply zero reference latent to conditioning.

    Some models require reference_latents to be set even when no reference
    is desired. This node creates a zero-filled reference of the correct shape.

    Use when:
    - Model expects reference_latents but you don't have a reference
    - Matching conditioning structure between batches with/without reference
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING", {
                    "tooltip": "Positive conditioning to modify"
                }),
                "negative": ("CONDITIONING", {
                    "tooltip": "Negative conditioning to modify"
                }),
                "latent": ("LATENT", {
                    "tooltip": "Reference latent for shape (will create zeros of same shape)"
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "apply"
    CATEGORY = "NV_Utils/conditioning"
    DESCRIPTION = "Apply zero reference latent (for models that require reference_latents to be set)."

    def apply(self, positive, negative, latent: dict):
        ref_latent = latent["samples"]
        zero_ref = torch.zeros_like(ref_latent)

        print(f"[NV_ApplyReferenceLatentZero] Applying zero reference: shape {list(zero_ref.shape)}")

        positive_out = node_helpers.conditioning_set_values(
            positive,
            {"reference_latents": [zero_ref]},
            append=True
        )
        negative_out = node_helpers.conditioning_set_values(
            negative,
            {"reference_latents": [zero_ref]},
            append=True
        )

        return (positive_out, negative_out)


class NV_CombineReferenceLatents:
    """
    Combine multiple latents into a single reference for richer context.

    Use when you want to provide context from multiple neighboring chunks.
    Latents are concatenated along the temporal dimension.

    Example:
    - Chunk processing frames 80-160
    - Reference 1: frames 0-80 (previous chunk)
    - Reference 2: frames 160-200 (next chunk)
    - Combined reference provides bidirectional context
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent_1": ("LATENT", {
                    "tooltip": "First reference latent"
                }),
            },
            "optional": {
                "latent_2": ("LATENT", {
                    "tooltip": "Second reference latent (optional)"
                }),
                "latent_3": ("LATENT", {
                    "tooltip": "Third reference latent (optional)"
                }),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("combined_latent",)
    FUNCTION = "combine"
    CATEGORY = "NV_Utils/latent"
    DESCRIPTION = "Combine multiple latents along temporal dimension for multi-chunk reference context."

    def combine(self, latent_1: dict, latent_2: dict = None, latent_3: dict = None):
        latents = [latent_1["samples"]]

        if latent_2 is not None:
            latents.append(latent_2["samples"])
        if latent_3 is not None:
            latents.append(latent_3["samples"])

        # Concatenate along temporal dimension (dim=2 for [B, C, T, H, W])
        combined = torch.cat(latents, dim=2)

        total_frames = sum(l.shape[2] for l in latents)
        print(f"[NV_CombineReferenceLatents] Combined {len(latents)} latents: "
              f"{total_frames} total latent frames, shape {list(combined.shape)}")

        return ({"samples": combined},)


# Node registration
NODE_CLASS_MAPPINGS = {
    "NV_ApplyReferenceLatent": NV_ApplyReferenceLatent,
    "NV_ApplyReferenceLatentZero": NV_ApplyReferenceLatentZero,
    "NV_CombineReferenceLatents": NV_CombineReferenceLatents,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_ApplyReferenceLatent": "NV Apply Reference Latent",
    "NV_ApplyReferenceLatentZero": "NV Apply Reference Latent (Zero)",
    "NV_CombineReferenceLatents": "NV Combine Reference Latents",
}
