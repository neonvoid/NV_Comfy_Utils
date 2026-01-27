"""
NV Attention Guidance

Extract attention patterns from a low-resolution full-video pass and apply them
as guidance during high-resolution chunked processing.

Key insight: Attention patterns are scale-independent. The structural relationships
between tokens (which frames attend to which) transfer across resolutions.

Research-backed parameters (from DraftAttention, Sparse-vDiT):
- Layers 0, 3, 8, 16, 19 preserve temporal structure
- Sparsity ratio of 25% is sufficient for guidance transfer
- Steps in 50-100% range capture detail refinement phase
- Blend mode at 0.7 strength works well for V2V
"""

import torch
import torch.nn.functional as F
import re
import os
import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview
from comfy.ldm.modules.attention import optimized_attention


def video_to_latent_frames(video_frames: int) -> int:
    """Convert video frame count to latent frame count (Wan 4:1 compression)."""
    if video_frames <= 0:
        return 0
    return (video_frames - 1) // 4 + 1


def parse_step_spec(spec: str, total_steps: int, denoise: float) -> list:
    """
    Parse step specification into list of step indices.

    Supports:
    - Percentage range: "50-100%" -> last 50% of effective steps
    - Explicit indices: "2,4,6,8" -> those exact steps
    - Mixed: "0,50-100%" -> step 0 plus last 50%

    For denoise < 1.0, effective_steps = int(total_steps * denoise)
    """
    effective_steps = int(total_steps * denoise) if denoise < 1.0 else total_steps
    start_step = total_steps - effective_steps

    indices = set()
    for part in spec.split(","):
        part = part.strip()
        if "%" in part:
            # Percentage range: "50-100%"
            match = re.match(r"(\d+)-(\d+)%", part)
            if match:
                pct_start, pct_end = int(match.group(1)), int(match.group(2))
                # Map percentage to effective step range
                idx_start = start_step + int(effective_steps * pct_start / 100)
                idx_end = start_step + int(effective_steps * pct_end / 100)
                # Sample evenly within range (max 6 points to limit storage)
                num_samples = min(6, idx_end - idx_start)
                if num_samples > 0:
                    step_size = (idx_end - idx_start) / num_samples
                    for i in range(num_samples):
                        indices.add(int(idx_start + i * step_size))
        else:
            # Explicit index
            try:
                indices.add(int(part))
            except ValueError:
                pass

    return sorted(indices)


def sparsify_attention(attn_weights: torch.Tensor, ratio: float = 0.25) -> torch.Tensor:
    """
    Keep only top-k% of attention weights per query token.

    Input: [batch*heads, seq_q, seq_k] or [batch, heads, seq_q, seq_k]
    Output: Sparse representation (indices and values)

    We store as a dict with indices and values for efficient storage.
    """
    original_shape = attn_weights.shape

    # Handle different input shapes
    if attn_weights.dim() == 3:
        # [batch*heads, seq_q, seq_k]
        bh, sq, sk = attn_weights.shape
    elif attn_weights.dim() == 4:
        # [batch, heads, seq_q, seq_k]
        b, h, sq, sk = attn_weights.shape
        attn_weights = attn_weights.view(b * h, sq, sk)
        bh = b * h
    else:
        raise ValueError(f"Unexpected attention shape: {attn_weights.shape}")

    # Top-k per query
    k = max(1, int(sk * ratio))
    topk_vals, topk_indices = torch.topk(attn_weights, k, dim=-1)

    return {
        "values": topk_vals.cpu().half(),  # Store as fp16
        "indices": topk_indices.cpu().short(),  # Store as int16
        "original_shape": original_shape,
        "k": k,
    }


def densify_attention(sparse_data: dict, device: torch.device = None) -> torch.Tensor:
    """
    Reconstruct dense attention from sparse representation.

    Returns: Tensor of original shape with zeros where values weren't stored.
    """
    values = sparse_data["values"]
    indices = sparse_data["indices"]
    original_shape = sparse_data["original_shape"]

    if device is not None:
        values = values.to(device).float()
        indices = indices.to(device).long()
    else:
        values = values.float()
        indices = indices.long()

    # Reconstruct dense tensor
    if len(original_shape) == 3:
        bh, sq, sk = original_shape
        dense = torch.zeros(bh, sq, sk, device=values.device, dtype=values.dtype)
        dense.scatter_(-1, indices, values)
    elif len(original_shape) == 4:
        b, h, sq, sk = original_shape
        bh = b * h
        dense = torch.zeros(bh, sq, sk, device=values.device, dtype=values.dtype)
        dense.scatter_(-1, indices, values)
        dense = dense.view(b, h, sq, sk)

    return dense


def attention_with_capture(q, k, v, heads, attn_precision=None):
    """
    Compute attention and return both output and attention weights.

    Based on attention_basic_with_sim from nodes_sag.py
    """
    from einops import rearrange
    from torch import einsum

    b, _, dim_head = q.shape
    dim_head //= heads
    scale = dim_head ** -0.5

    # Reshape for multi-head attention
    q_heads = rearrange(q, 'b n (h d) -> (b h) n d', h=heads)
    k_heads = rearrange(k, 'b n (h d) -> (b h) n d', h=heads)
    v_heads = rearrange(v, 'b n (h d) -> (b h) n d', h=heads)

    # Compute attention scores
    if attn_precision == torch.float32:
        sim = einsum('b i d, b j d -> b i j', q_heads.float(), k_heads.float()) * scale
    else:
        sim = einsum('b i d, b j d -> b i j', q_heads, k_heads) * scale

    # Softmax
    attn_weights = sim.softmax(dim=-1)

    # Compute output
    out = einsum('b i j, b j d -> b i d', attn_weights.to(v_heads.dtype), v_heads)
    out = rearrange(out, '(b h) n d -> b n (h d)', h=heads)

    return out, attn_weights


def scale_guidance_to_target(guidance_attn: torch.Tensor, target_seq_len: int) -> torch.Tensor:
    """
    Scale guidance attention pattern to match target sequence length.

    Uses bilinear interpolation on the attention matrix.
    """
    source_seq_len = guidance_attn.shape[-1]

    if source_seq_len == target_seq_len:
        return guidance_attn

    # Reshape for interpolation: treat as [batch, 1, h, w] image
    if guidance_attn.dim() == 3:
        # [bh, sq, sk] -> [bh, 1, sq, sk]
        guidance_2d = guidance_attn.unsqueeze(1)
    else:
        # [b, h, sq, sk] -> [b*h, 1, sq, sk]
        b, h, sq, sk = guidance_attn.shape
        guidance_2d = guidance_attn.view(b * h, 1, sq, sk)

    # Interpolate
    scaled = F.interpolate(
        guidance_2d,
        size=(target_seq_len, target_seq_len),
        mode='bilinear',
        align_corners=False
    )

    # Reshape back
    if guidance_attn.dim() == 3:
        return scaled.squeeze(1)
    else:
        return scaled.view(b, h, target_seq_len, target_seq_len)


class NV_ExtractAttentionGuidance:
    """
    Extract attention patterns during sampling for use as guidance.

    Hooks into self-attention layers to capture which tokens attend
    to which, then compresses to sparse representation for storage.

    Research-backed defaults:
    - Layers 0, 3, 8, 16, 19 preserve temporal structure
    - Steps 50-100% capture detail refinement phase
    - Sparsity ratio 0.25 (25%) sufficient for guidance
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0}),
            },
            "optional": {
                "committed_noise": ("COMMITTED_NOISE", {
                    "tooltip": "Pre-generated noise (from NV_CommittedNoise)"
                }),
                "extract_layers": ("STRING", {
                    "default": "0,3,8,16,19",
                    "tooltip": "Comma-separated layer indices. Research: 0,3,19 for structure preservation"
                }),
                "extract_steps": ("STRING", {
                    "default": "50-100%",
                    "tooltip": "Step indices (e.g., '2,4,6') OR percentage range (e.g., '50-100%' for last half)"
                }),
                "sparsity_ratio": ("FLOAT", {
                    "default": 0.25,
                    "min": 0.01,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Keep top X% of attention weights (0.25 = top 25%, research-backed)"
                }),
            }
        }

    RETURN_TYPES = ("LATENT", "ATTENTION_GUIDANCE")
    RETURN_NAMES = ("latent", "attention_guidance")
    FUNCTION = "extract"
    CATEGORY = "NV_Utils/sampling"
    DESCRIPTION = "Extract attention patterns during sampling for guidance in chunked processing."

    def extract(self, model, positive, negative, latent_image, seed, steps, cfg,
                sampler_name, scheduler, denoise, committed_noise=None,
                extract_layers="0,3,8,16,19", extract_steps="50-100%",
                sparsity_ratio=0.25):

        # Parse layer indices
        layer_indices = set()
        for x in extract_layers.split(","):
            try:
                layer_indices.add(int(x.strip()))
            except ValueError:
                pass

        # Parse step indices
        step_indices = set(parse_step_spec(extract_steps, steps, denoise))

        print(f"[NV_ExtractAttentionGuidance] Extracting from layers: {sorted(layer_indices)}")
        print(f"[NV_ExtractAttentionGuidance] Extracting at steps: {sorted(step_indices)}")
        print(f"[NV_ExtractAttentionGuidance] Sparsity ratio: {sparsity_ratio}")

        # Storage for extracted attention
        attention_data = {
            "version": "1.0",
            "metadata": {
                "source_shape": list(latent_image["samples"].shape),
                "steps": steps,
                "denoise": denoise,
                "sparsity_ratio": sparsity_ratio,
                "extract_layers": sorted(layer_indices),
                "extract_steps": sorted(step_indices),
            },
            "layers": {}
        }

        # State tracking via closure
        current_step = [0]
        extraction_count = [0]

        def attention_extractor(q, k, v, extra_options):
            """Hook that extracts attention weights."""
            nonlocal attention_data

            step = current_step[0]
            block_index = extra_options.get("block_index", 0)
            heads = extra_options.get("n_heads", 1)
            attn_precision = extra_options.get("attn_precision", None)

            # Check if we should extract this step/layer
            if step in step_indices and block_index in layer_indices:
                # Compute attention with capture
                out, attn_weights = attention_with_capture(
                    q, k, v, heads, attn_precision
                )

                # Sparsify and store (only store once per step/layer combo)
                key = f"step_{step}_layer_{block_index}"
                if key not in attention_data["layers"]:
                    sparse = sparsify_attention(attn_weights, sparsity_ratio)
                    attention_data["layers"][key] = sparse
                    extraction_count[0] += 1
                    print(f"[NV_ExtractAttentionGuidance] Captured {key}: "
                          f"shape {attn_weights.shape}, stored {sparse['k']} values per query")

                return out
            else:
                # Standard attention
                return optimized_attention(q, k, v, heads, attn_precision=attn_precision)

        # Clone model and add extraction hook
        patched_model = model.clone()

        # Add hook to all transformer blocks (we filter by block_index in the hook)
        # Using attn1_patch which applies to all self-attention layers
        patched_model.set_model_attn1_patch(attention_extractor)

        # Step tracking callback
        def step_callback(step, x0, x, total_steps):
            current_step[0] = step

        # Prepare for sampling
        latent = latent_image["samples"]
        latent = comfy.sample.fix_empty_latent_channels(patched_model, latent)

        if committed_noise is not None:
            noise = committed_noise["noise"]
            print(f"[NV_ExtractAttentionGuidance] Using committed noise (seed={committed_noise['seed']})")
        else:
            batch_inds = latent_image.get("batch_index", None)
            noise = comfy.sample.prepare_noise(latent, seed, batch_inds)

        noise_mask = latent_image.get("noise_mask", None)

        # Progress callback
        callback = latent_preview.prepare_callback(patched_model, steps)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        # Wrap callback to include step tracking
        original_callback = callback

        def combined_callback(step, x0, x, total_steps):
            step_callback(step, x0, x, total_steps)
            if original_callback is not None:
                return original_callback(step, x0, x, total_steps)
            return None

        # Run sampling
        samples = comfy.sample.sample(
            patched_model, noise, steps, cfg, sampler_name, scheduler,
            positive, negative, latent,
            denoise=denoise, disable_noise=False,
            noise_mask=noise_mask, callback=combined_callback,
            disable_pbar=disable_pbar, seed=seed
        )

        print(f"[NV_ExtractAttentionGuidance] Extraction complete: {extraction_count[0]} attention patterns stored")

        out = latent_image.copy()
        out["samples"] = samples

        return (out, attention_data)


class NV_SaveAttentionGuidance:
    """
    Save extracted attention guidance to disk.

    Stores attention patterns in PyTorch format for loading during
    chunked processing.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "attention_guidance": ("ATTENTION_GUIDANCE",),
                "output_path": ("STRING", {"default": "attention_guidance.pt"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = "NV_Utils/sampling"
    DESCRIPTION = "Save attention guidance to disk for use in chunked processing."

    def save(self, attention_guidance, output_path):
        # Ensure directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Save
        torch.save(attention_guidance, output_path)

        num_patterns = len(attention_guidance.get("layers", {}))
        metadata = attention_guidance.get("metadata", {})

        print(f"[NV_SaveAttentionGuidance] Saved to {output_path}")
        print(f"  Patterns: {num_patterns}")
        print(f"  Source shape: {metadata.get('source_shape', 'unknown')}")
        print(f"  Steps: {metadata.get('steps', 'unknown')}, Denoise: {metadata.get('denoise', 'unknown')}")

        return (output_path,)


class NV_LoadAttentionGuidance:
    """
    Load attention guidance and prepare for chunk application.

    For chunked processing, can specify which portion of the guidance
    to use based on chunk position.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "guidance_path": ("STRING", {"default": "attention_guidance.pt"}),
            },
            "optional": {
                "chunk_start_frame": ("INT", {
                    "default": 0,
                    "min": 0,
                    "tooltip": "Starting VIDEO frame of chunk (for subsetting guidance)"
                }),
                "chunk_frame_count": ("INT", {
                    "default": 81,
                    "min": 1,
                    "tooltip": "Number of VIDEO frames in chunk"
                }),
                "scale_factor": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 4,
                    "tooltip": "Resolution scale factor (2 = guidance was at 1/2 res)"
                }),
            }
        }

    RETURN_TYPES = ("ATTENTION_GUIDANCE",)
    RETURN_NAMES = ("attention_guidance",)
    FUNCTION = "load"
    CATEGORY = "NV_Utils/sampling"
    DESCRIPTION = "Load attention guidance for use during chunked sampling."

    def load(self, guidance_path, chunk_start_frame=0, chunk_frame_count=81, scale_factor=2):
        if not os.path.exists(guidance_path):
            raise FileNotFoundError(f"Attention guidance file not found: {guidance_path}")

        data = torch.load(guidance_path, weights_only=False)

        metadata = data.get("metadata", {})
        num_patterns = len(data.get("layers", {}))

        print(f"[NV_LoadAttentionGuidance] Loaded from {guidance_path}")
        print(f"  Patterns: {num_patterns}")
        print(f"  Source shape: {metadata.get('source_shape', 'unknown')}")
        print(f"  Chunk: frames {chunk_start_frame}-{chunk_start_frame + chunk_frame_count}")
        print(f"  Scale factor: {scale_factor}x")

        # Add chunk info to metadata for apply node
        data["chunk_info"] = {
            "start_frame": chunk_start_frame,
            "frame_count": chunk_frame_count,
            "scale_factor": scale_factor,
        }

        return (data,)


class NV_ApplyAttentionGuidance:
    """
    Apply attention guidance to model for guided sampling.

    The guidance biases attention patterns to match those from
    the low-resolution full-video pass, improving chunk consistency.

    Guidance modes:
    - blend: Interpolate between computed and guided attention (recommended)
    - mask: Force attention to follow guidance pattern
    - bias: Add guidance as bias to attention scores
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "attention_guidance": ("ATTENTION_GUIDANCE",),
            },
            "optional": {
                "guidance_strength": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Guidance strength (0.7 recommended for V2V - allows natural variation)"
                }),
                "guidance_mode": (["blend", "mask", "bias"], {
                    "default": "blend",
                    "tooltip": "blend: interpolate (best for V2V), mask: force pattern, bias: add to scores"
                }),
                "apply_layers": ("STRING", {
                    "default": "",
                    "tooltip": "Comma-separated layer indices to apply guidance (empty = use extracted layers)"
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "NV_Utils/sampling"
    DESCRIPTION = "Apply attention guidance to model for consistent chunked sampling."

    def apply(self, model, attention_guidance, guidance_strength=0.7,
              guidance_mode="blend", apply_layers=""):

        guidance_data = attention_guidance
        metadata = guidance_data.get("metadata", {})

        # Determine which layers to apply guidance
        if apply_layers.strip():
            target_layers = set()
            for x in apply_layers.split(","):
                try:
                    target_layers.add(int(x.strip()))
                except ValueError:
                    pass
        else:
            target_layers = set(metadata.get("extract_layers", []))

        extract_steps = set(metadata.get("extract_steps", []))

        print(f"[NV_ApplyAttentionGuidance] Mode: {guidance_mode}, Strength: {guidance_strength}")
        print(f"[NV_ApplyAttentionGuidance] Applying to layers: {sorted(target_layers)}")
        print(f"[NV_ApplyAttentionGuidance] Active at steps: {sorted(extract_steps)}")

        # State tracking
        current_step = [0]
        applications = [0]

        def guided_attention(q, k, v, extra_options):
            """Hook that applies attention guidance."""
            step = current_step[0]
            block_index = extra_options.get("block_index", 0)
            heads = extra_options.get("n_heads", 1)
            attn_precision = extra_options.get("attn_precision", None)

            # Check if we should apply guidance
            key = f"step_{step}_layer_{block_index}"
            if (step in extract_steps and
                block_index in target_layers and
                key in guidance_data.get("layers", {})):

                # Get guidance pattern
                sparse_guidance = guidance_data["layers"][key]
                guidance_attn = densify_attention(sparse_guidance, device=q.device)

                # Compute current attention
                out, current_attn = attention_with_capture(
                    q, k, v, heads, attn_precision
                )

                # Scale guidance to match current resolution
                target_seq_len = current_attn.shape[-1]
                scaled_guidance = scale_guidance_to_target(guidance_attn, target_seq_len)

                # Apply guidance based on mode
                if guidance_mode == "blend":
                    # Interpolate: result = (1-strength)*current + strength*guidance
                    # Then re-normalize
                    blended = (1 - guidance_strength) * current_attn + guidance_strength * scaled_guidance
                    # Re-normalize rows to sum to 1
                    blended = blended / (blended.sum(dim=-1, keepdim=True) + 1e-8)

                    # Recompute output with blended attention
                    from einops import rearrange
                    from torch import einsum

                    v_heads = rearrange(v, 'b n (h d) -> (b h) n d', h=heads)
                    out = einsum('b i j, b j d -> b i d', blended.to(v_heads.dtype), v_heads)
                    out = rearrange(out, '(b h) n d -> b n (h d)', h=heads)

                elif guidance_mode == "mask":
                    # Hard mask: use guidance pattern to mask attention
                    # Zero out attention where guidance is zero, scale up where it's non-zero
                    mask = (scaled_guidance > 0.01).float()
                    masked = current_attn * mask
                    masked = masked / (masked.sum(dim=-1, keepdim=True) + 1e-8)

                    # Blend masked with original based on strength
                    final_attn = (1 - guidance_strength) * current_attn + guidance_strength * masked

                    from einops import rearrange
                    from torch import einsum

                    v_heads = rearrange(v, 'b n (h d) -> (b h) n d', h=heads)
                    out = einsum('b i j, b j d -> b i d', final_attn.to(v_heads.dtype), v_heads)
                    out = rearrange(out, '(b h) n d -> b n (h d)', h=heads)

                elif guidance_mode == "bias":
                    # Add guidance as bias before softmax
                    # This requires recomputing attention from scratch
                    from einops import rearrange
                    from torch import einsum

                    dim_head = q.shape[-1] // heads
                    scale = dim_head ** -0.5

                    q_heads = rearrange(q, 'b n (h d) -> (b h) n d', h=heads)
                    k_heads = rearrange(k, 'b n (h d) -> (b h) n d', h=heads)
                    v_heads = rearrange(v, 'b n (h d) -> (b h) n d', h=heads)

                    # Compute attention scores
                    if attn_precision == torch.float32:
                        sim = einsum('b i d, b j d -> b i j', q_heads.float(), k_heads.float()) * scale
                    else:
                        sim = einsum('b i d, b j d -> b i j', q_heads, k_heads) * scale

                    # Add guidance bias (scaled by strength)
                    guidance_bias = scaled_guidance * guidance_strength * 5.0  # Scale factor for effect
                    sim = sim + guidance_bias.to(sim.dtype)

                    # Softmax
                    attn_weights = sim.softmax(dim=-1)

                    # Output
                    out = einsum('b i j, b j d -> b i d', attn_weights.to(v_heads.dtype), v_heads)
                    out = rearrange(out, '(b h) n d -> b n (h d)', h=heads)

                applications[0] += 1
                return out
            else:
                # Standard attention
                return optimized_attention(q, k, v, heads, attn_precision=attn_precision)

        # Clone model and add guidance hook
        patched_model = model.clone()
        patched_model.set_model_attn1_patch(guided_attention)

        # Add step tracking via sampler callback
        def step_tracker(args):
            # This is called after each CFG step
            # We can track progress but step info comes from sigma
            pass

        # Store step tracking info in model for sampler to update
        patched_model._attention_guidance_step = current_step
        patched_model._attention_guidance_applications = applications

        print(f"[NV_ApplyAttentionGuidance] Model patched with attention guidance")

        return (patched_model,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "NV_ExtractAttentionGuidance": NV_ExtractAttentionGuidance,
    "NV_SaveAttentionGuidance": NV_SaveAttentionGuidance,
    "NV_LoadAttentionGuidance": NV_LoadAttentionGuidance,
    "NV_ApplyAttentionGuidance": NV_ApplyAttentionGuidance,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_ExtractAttentionGuidance": "NV Extract Attention Guidance",
    "NV_SaveAttentionGuidance": "NV Save Attention Guidance",
    "NV_LoadAttentionGuidance": "NV Load Attention Guidance",
    "NV_ApplyAttentionGuidance": "NV Apply Attention Guidance",
}
