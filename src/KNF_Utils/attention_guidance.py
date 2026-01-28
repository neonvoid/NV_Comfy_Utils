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
from nodes import common_ksampler


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


def attention_with_capture(q, k, v, heads, attn_precision=None, skip_reshape=False):
    """
    Compute attention and return both output and attention weights.

    Based on attention_basic_with_sim from nodes_sag.py
    Handles both standard [B, S, H*D] and skip_reshape [B, H, S, D] formats.
    """
    from einops import rearrange
    from torch import einsum

    if skip_reshape:
        # Already in [B, H, S, D] format
        b, h, s, d = q.shape
        heads = h
        scale = d ** -0.5

        q_heads = q.view(b * h, s, d)
        k_heads = k.view(b * h, s, d)
        v_heads = v.view(b * h, s, d)
    else:
        # Standard [B, S, H*D] format
        b, _, dim_total = q.shape
        dim_head = dim_total // heads
        scale = dim_head ** -0.5

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

    if skip_reshape:
        out = out.view(b, heads, -1, out.shape[-1])
    else:
        out = rearrange(out, '(b h) n d -> b n (h d)', h=heads)

    return out, attn_weights


def create_attention_capture_override(storage, target_steps, target_layers, sparsity_ratio,
                                       current_step_ref, extraction_count_ref):
    """
    Create an attention override function that captures patterns.

    This uses the optimized_attention_override mechanism which intercepts
    ALL attention calls via the @wrap_attn decorator.

    Args:
        storage: Dict to store captured attention patterns (layers key)
        target_steps: Set of step indices to capture
        target_layers: Set of block indices to capture
        sparsity_ratio: Top-k ratio for sparse storage (0.25 = 25%)
        current_step_ref: List with single element [step] for mutable step tracking
        extraction_count_ref: List with single element [count] for tracking captures
    """
    def attention_capture_override(original_func, *args, **kwargs):
        """
        Override function called for every attention computation.

        Signature matches wrap_attn: (original_func, q, k, v, heads, ...)
        Note: Wan models pass heads as kwarg, UNet models pass as positional arg.
        """
        # Extract arguments - handle both positional and keyword args for heads
        q, k, v = args[0], args[1], args[2]
        # heads can be positional (arg 4) or keyword
        if len(args) > 3:
            heads = args[3]
        else:
            heads = kwargs.get("heads", 1)

        t_opts = kwargs.get("transformer_options", {})
        block_idx = t_opts.get("block_index", -1)
        step = current_step_ref[0]
        attn_precision = kwargs.get("attn_precision", None)
        skip_reshape = kwargs.get("skip_reshape", False)

        # Check if we should capture at this step and layer
        if step in target_steps and block_idx in target_layers:
            key = f"step_{step}_layer_{block_idx}"

            # Only capture once per key (avoid duplicates from multiple attention calls)
            if key not in storage:
                # Compute attention with weights capture
                output, attn_weights = attention_with_capture(
                    q, k, v, heads,
                    attn_precision=attn_precision,
                    skip_reshape=skip_reshape
                )

                # Store sparsified weights
                storage[key] = sparsify_attention(attn_weights, sparsity_ratio)
                extraction_count_ref[0] += 1

                print(f"[AttentionCapture] Captured {key}: shape {list(attn_weights.shape)}, "
                      f"sparse vals {storage[key]['values'].shape}")

                return output

        # Normal attention - call original function
        return original_func(*args, **kwargs)

    return attention_capture_override


def create_attention_guidance_override(guidance_data, target_steps, target_layers,
                                         strength, mode, current_step_ref,
                                         applications_ref, original_override=None):
    """
    Create an attention override that applies guidance patterns.

    Args:
        guidance_data: Dict with stored attention patterns (layers key)
        target_steps: Set of step indices where guidance applies
        target_layers: Set of block indices where guidance applies
        strength: Guidance strength (0.0 to 1.0+)
        mode: "blend", "mask", or "bias"
        current_step_ref: List with single element [step]
        applications_ref: List with single element [count] for tracking applications
        original_override: Previous override to chain (or None)

    Note: If step tracking is not available (step stays at 0), falls back to
          using any available pattern for the target layer.
    """
    # Pre-build a lookup for layer-only fallback
    layers_data = guidance_data.get("layers", {})
    layer_patterns = {}  # block_idx -> first available pattern key for that layer
    for key in layers_data.keys():
        # Keys are formatted as "step_X_layer_Y"
        parts = key.split("_")
        if len(parts) >= 4 and parts[2] == "layer":
            try:
                layer_idx = int(parts[3])
                if layer_idx not in layer_patterns:
                    layer_patterns[layer_idx] = key
            except ValueError:
                pass

    def attention_guidance_override(original_func, *args, **kwargs):
        """
        Override that applies guidance patterns to attention.
        Note: Wan models pass heads as kwarg, UNet models pass as positional arg.
        """
        from einops import rearrange
        from torch import einsum

        # Extract arguments - handle both positional and keyword args for heads
        q, k, v = args[0], args[1], args[2]
        if len(args) > 3:
            heads = args[3]
        else:
            heads = kwargs.get("heads", 1)

        t_opts = kwargs.get("transformer_options", {})
        block_idx = t_opts.get("block_index", -1)
        step = current_step_ref[0]
        attn_precision = kwargs.get("attn_precision", None)
        skip_reshape = kwargs.get("skip_reshape", False)

        # Try exact step+layer match first
        key = f"step_{step}_layer_{block_idx}"

        # Fallback to any pattern for this layer if exact match not found
        if key not in layers_data and block_idx in layer_patterns:
            key = layer_patterns[block_idx]

        # Check if we should apply guidance
        # Apply if: (step matches OR fallback used) AND layer is in targets AND we have data
        should_apply = block_idx in target_layers and key in layers_data
        if should_apply:
            # Get guidance pattern and densify it
            guidance_attn = densify_attention(layers_data[key], device=q.device)

            # Compute current attention
            output, current_attn = attention_with_capture(
                q, k, v, heads,
                attn_precision=attn_precision,
                skip_reshape=skip_reshape
            )

            # Scale guidance to match current resolution if needed
            target_seq_len = current_attn.shape[-1]
            if guidance_attn.shape[-1] != target_seq_len:
                guidance_attn = scale_guidance_to_target(guidance_attn, target_seq_len)

            # Apply guidance based on mode
            if mode == "blend":
                # Interpolate: result = (1-strength)*current + strength*guidance
                blended = (1 - strength) * current_attn + strength * guidance_attn
                # Re-normalize rows to sum to 1
                blended = blended / (blended.sum(dim=-1, keepdim=True) + 1e-8)

                # Recompute output with blended attention
                if skip_reshape:
                    b, h, s, d = v.shape
                    v_heads = v.view(b * h, s, d)
                else:
                    v_heads = rearrange(v, 'b n (h d) -> (b h) n d', h=heads)

                out = einsum('b i j, b j d -> b i d', blended.to(v_heads.dtype), v_heads)

                if skip_reshape:
                    out = out.view(b, h, -1, out.shape[-1])
                else:
                    out = rearrange(out, '(b h) n d -> b n (h d)', h=heads)

            elif mode == "mask":
                # Hard mask: use guidance pattern to mask attention
                mask = (guidance_attn > 0.01).float()
                masked = current_attn * mask
                masked = masked / (masked.sum(dim=-1, keepdim=True) + 1e-8)

                # Blend masked with original based on strength
                final_attn = (1 - strength) * current_attn + strength * masked

                if skip_reshape:
                    b, h, s, d = v.shape
                    v_heads = v.view(b * h, s, d)
                else:
                    v_heads = rearrange(v, 'b n (h d) -> (b h) n d', h=heads)

                out = einsum('b i j, b j d -> b i d', final_attn.to(v_heads.dtype), v_heads)

                if skip_reshape:
                    out = out.view(b, h, -1, out.shape[-1])
                else:
                    out = rearrange(out, '(b h) n d -> b n (h d)', h=heads)

            elif mode == "bias":
                # Add guidance as bias before softmax
                if skip_reshape:
                    b, h, s, d = q.shape
                    scale = d ** -0.5
                    q_heads = q.view(b * h, s, d)
                    k_heads = k.view(b * h, s, d)
                    v_heads = v.view(b * h, s, d)
                else:
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
                guidance_bias = guidance_attn * strength * 5.0
                sim = sim + guidance_bias.to(sim.dtype)

                # Softmax
                attn_weights = sim.softmax(dim=-1)

                out = einsum('b i j, b j d -> b i d', attn_weights.to(v_heads.dtype), v_heads)

                if skip_reshape:
                    out = out.view(b, h, -1, out.shape[-1])
                else:
                    out = rearrange(out, '(b h) n d -> b n (h d)', h=heads)

            else:
                out = output

            applications_ref[0] += 1
            return out

        # No guidance - use original or chain
        if original_override is not None:
            return original_override(original_func, *args, **kwargs)
        return original_func(*args, **kwargs)

    return attention_guidance_override


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

    Supports multi-model sampling (sequential/boundary modes) to match
    production workflow configuration.

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
                # Multi-model support
                "model_2": ("MODEL",),
                "model_3": ("MODEL",),
                "mode": (["single", "sequential", "boundary"], {
                    "default": "single",
                    "tooltip": "single: use model only, sequential: chain models by steps, boundary: switch by sigma"
                }),
                "model_steps": ("STRING", {
                    "default": "",
                    "tooltip": "Sequential mode: comma-separated steps per model (e.g., '7,7,6'). Empty = auto-divide."
                }),
                "model_boundaries": ("STRING", {
                    "default": "0.875,0.5",
                    "tooltip": "Boundary mode: sigma thresholds (0-1). Model switches when sigma drops below threshold."
                }),
                "model_cfg_scales": ("STRING", {
                    "default": "",
                    "tooltip": "Per-model CFG overrides (e.g., '7.0,5.0,3.0'). Empty = use main cfg."
                }),
                # Committed noise
                "committed_noise": ("COMMITTED_NOISE", {
                    "tooltip": "Pre-generated noise (from NV_CommittedNoise)"
                }),
                # Extraction settings
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
    DESCRIPTION = "Extract attention patterns during sampling for guidance in chunked processing. Supports multi-model workflows."

    def extract(self, model, positive, negative, latent_image, seed, steps, cfg,
                sampler_name, scheduler, denoise, model_2=None, model_3=None,
                mode="single", model_steps="", model_boundaries="0.875,0.5",
                model_cfg_scales="", committed_noise=None,
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

        # Collect models
        models = [model]
        if model_2 is not None:
            models.append(model_2)
        if model_3 is not None:
            models.append(model_3)

        # Parse cfg scales
        cfg_scales = self._parse_cfg_scales(model_cfg_scales, cfg, len(models))

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
                "mode": mode,
                "num_models": len(models),
            },
            "layers": {}
        }

        # State tracking via closure (mutable references)
        current_step = [0]
        extraction_count = [0]

        # Create the attention capture override
        capture_override = create_attention_capture_override(
            storage=attention_data["layers"],
            target_steps=step_indices,
            target_layers=layer_indices,
            sparsity_ratio=sparsity_ratio,
            current_step_ref=current_step,
            extraction_count_ref=extraction_count
        )

        # Clone models and apply the override to each
        patched_models = []
        for m in models:
            pm = m.clone()
            # Initialize transformer_options if it doesn't exist
            if "transformer_options" not in pm.model_options:
                pm.model_options["transformer_options"] = {}
            # Set the attention override
            pm.model_options["transformer_options"]["optimized_attention_override"] = capture_override
            patched_models.append(pm)

        print(f"[NV_ExtractAttentionGuidance] Applied attention capture override to {len(patched_models)} model(s)")
        print(f"[NV_ExtractAttentionGuidance] Target layers: {sorted(layer_indices)}, steps: {sorted(step_indices)}")

        # Get committed noise tensor if provided
        noise_tensor = committed_noise["noise"] if committed_noise is not None else None
        if committed_noise is not None:
            print(f"[NV_ExtractAttentionGuidance] Using committed noise (seed={committed_noise['seed']})")

        # Route to appropriate sampling method with step tracking
        if mode == "single" or len(patched_models) == 1:
            print(f"[NV_ExtractAttentionGuidance] Single model mode")
            result = self._sample_single(
                patched_models[0], positive, negative, latent_image,
                seed, steps, cfg, sampler_name, scheduler, denoise,
                noise_tensor, step_ref=current_step
            )
        elif mode == "sequential":
            print(f"[NV_ExtractAttentionGuidance] Sequential mode: {len(patched_models)} models")
            result = self._sample_sequential(
                patched_models, positive, negative, latent_image,
                seed, steps, cfg_scales, sampler_name, scheduler, denoise,
                model_steps, noise_tensor, step_ref=current_step
            )
        elif mode == "boundary":
            print(f"[NV_ExtractAttentionGuidance] Boundary mode: {len(patched_models)} models")
            result = self._sample_boundary(
                patched_models, positive, negative, latent_image,
                seed, steps, cfg_scales, sampler_name, scheduler, denoise,
                model_boundaries, noise_tensor, step_ref=current_step
            )
        else:
            # Fallback to single
            result = self._sample_single(
                patched_models[0], positive, negative, latent_image,
                seed, steps, cfg, sampler_name, scheduler, denoise,
                noise_tensor, step_ref=current_step
            )

        print(f"[NV_ExtractAttentionGuidance] Extraction complete: {extraction_count[0]} attention patterns stored")

        return (result, attention_data)

    def _create_step_callback(self, model, steps, step_ref):
        """Create a callback that tracks the current step and updates step_ref."""
        preview_callback = latent_preview.prepare_callback(model, steps)

        def step_tracking_callback(step, x0, x, total_steps):
            step_ref[0] = step
            if preview_callback is not None:
                return preview_callback(step, x0, x, total_steps)
        return step_tracking_callback

    def _sample_single(self, model, positive, negative, latent_image,
                       seed, steps, cfg, sampler_name, scheduler, denoise,
                       noise_tensor=None, step_ref=None):
        """Single model sampling with step tracking."""
        # Use direct sampling to inject step callback
        return self._ksampler_direct(
            model, seed, steps, cfg, sampler_name, scheduler,
            positive, negative, latent_image, denoise=denoise,
            noise_tensor=noise_tensor, step_ref=step_ref
        )

    def _sample_sequential(self, models, positive, negative, latent_image,
                           seed, steps, cfg_scales, sampler_name, scheduler, denoise,
                           model_steps_str, noise_tensor=None, step_ref=None):
        """Sequential multi-model sampling with step tracking."""
        step_distribution = self._parse_step_distribution(model_steps_str, steps, len(models))
        print(f"[NV_ExtractAttentionGuidance] Step distribution: {step_distribution}")

        current_latent = latent_image
        current_step_idx = 0

        for i, (model, model_steps, cfg) in enumerate(zip(models, step_distribution, cfg_scales)):
            if model_steps <= 0:
                continue

            start_step = current_step_idx
            end_step = current_step_idx + model_steps
            disable_noise = (i > 0)

            print(f"[NV_ExtractAttentionGuidance] Model {i+1}: steps {start_step}-{end_step}, cfg={cfg}, disable_noise={disable_noise}")

            result = self._ksampler_direct(
                model, seed, steps, cfg, sampler_name, scheduler,
                positive, negative, current_latent,
                denoise=denoise,
                noise_tensor=noise_tensor if not disable_noise else None,
                disable_noise=disable_noise,
                start_step=start_step,
                last_step=end_step,
                force_full_denoise=(i == len(models) - 1),
                step_ref=step_ref
            )

            current_latent = result
            current_step_idx = end_step

        return current_latent

    def _sample_boundary(self, models, positive, negative, latent_image,
                         seed, steps, cfg_scales, sampler_name, scheduler, denoise,
                         model_boundaries_str, noise_tensor=None, step_ref=None):
        """Boundary-based multi-model sampling with step tracking."""
        boundaries = self._parse_boundaries(model_boundaries_str, len(models))
        step_ranges = self._boundaries_to_steps(boundaries, steps, denoise)
        print(f"[NV_ExtractAttentionGuidance] Sigma boundaries: {boundaries}")
        print(f"[NV_ExtractAttentionGuidance] Step ranges: {step_ranges}")

        current_latent = latent_image

        for i, (model, (start_step, end_step), cfg) in enumerate(zip(models, step_ranges, cfg_scales)):
            if start_step >= end_step:
                continue

            disable_noise = (i > 0)

            print(f"[NV_ExtractAttentionGuidance] Model {i+1}: steps {start_step}-{end_step}, cfg={cfg}")

            result = self._ksampler_direct(
                model, seed, steps, cfg, sampler_name, scheduler,
                positive, negative, current_latent,
                denoise=denoise,
                noise_tensor=noise_tensor if not disable_noise else None,
                disable_noise=disable_noise,
                start_step=start_step,
                last_step=end_step,
                force_full_denoise=(i == len(models) - 1),
                step_ref=step_ref
            )

            current_latent = result

        return current_latent

    def _ksampler_direct(self, model, seed, steps, cfg, sampler_name, scheduler,
                          positive, negative, latent, denoise=1.0, noise_tensor=None,
                          disable_noise=False, start_step=None, last_step=None,
                          force_full_denoise=False, step_ref=None):
        """
        Unified sampling function with step tracking support.

        Combines functionality of common_ksampler and committed noise support,
        while also tracking the current step for attention capture.
        """
        latent_image = latent["samples"]
        latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)

        # Handle noise
        if noise_tensor is not None:
            # Use provided committed noise
            if noise_tensor.shape != latent_image.shape:
                raise ValueError(
                    f"Committed noise shape {list(noise_tensor.shape)} doesn't match "
                    f"latent shape {list(latent_image.shape)}"
                )
            noise = noise_tensor
            disable_noise = False  # We have noise, don't disable
        elif disable_noise:
            # Create empty noise (all zeros) when continuing from previous model
            noise = torch.zeros_like(latent_image)
        else:
            # Generate noise from seed
            batch_inds = latent.get("batch_index", None)
            noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

        noise_mask = latent.get("noise_mask", None)

        # Create callback with step tracking
        if step_ref is not None:
            callback = self._create_step_callback(model, steps, step_ref)
        else:
            callback = latent_preview.prepare_callback(model, steps)

        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        samples = comfy.sample.sample(
            model, noise, steps, cfg, sampler_name, scheduler,
            positive, negative, latent_image,
            denoise=denoise, disable_noise=disable_noise,
            start_step=start_step, last_step=last_step,
            force_full_denoise=force_full_denoise,
            noise_mask=noise_mask, callback=callback,
            disable_pbar=disable_pbar, seed=seed
        )

        out = latent.copy()
        out["samples"] = samples
        return out

    def _parse_step_distribution(self, model_steps_str, total_steps, num_models):
        """Parse step distribution string or auto-divide steps."""
        if model_steps_str.strip():
            try:
                parts = [int(x.strip()) for x in model_steps_str.split(",") if x.strip()]
                while len(parts) < num_models:
                    parts.append(0)
                return parts[:num_models]
            except ValueError:
                pass

        # Auto-divide
        base_steps = total_steps // num_models
        remainder = total_steps % num_models
        distribution = []
        for i in range(num_models):
            extra = 1 if i < remainder else 0
            distribution.append(base_steps + extra)
        return distribution

    def _parse_boundaries(self, boundaries_str, num_models):
        """Parse sigma boundary thresholds."""
        boundaries = [1.0]

        if boundaries_str.strip():
            try:
                parts = [float(x.strip()) for x in boundaries_str.split(",") if x.strip()]
                boundaries.extend(parts)
            except ValueError:
                pass

        while len(boundaries) < num_models:
            last = boundaries[-1] if boundaries else 1.0
            boundaries.append(last / 2)

        boundaries.append(0.0)
        return boundaries[:num_models + 1]

    def _boundaries_to_steps(self, boundaries, total_steps, denoise):
        """Convert sigma boundaries to step ranges."""
        step_ranges = []

        for i in range(len(boundaries) - 1):
            upper_sigma = boundaries[i]
            lower_sigma = boundaries[i + 1]

            start_step = int((1.0 - upper_sigma) * total_steps)
            end_step = int((1.0 - lower_sigma) * total_steps)

            start_step = max(0, min(start_step, total_steps))
            end_step = max(0, min(end_step, total_steps))

            step_ranges.append((start_step, end_step))

        return step_ranges

    def _parse_cfg_scales(self, cfg_scales_str, default_cfg, num_models):
        """Parse per-model CFG scales or use default."""
        if cfg_scales_str.strip():
            try:
                parts = [float(x.strip()) for x in cfg_scales_str.split(",") if x.strip()]
                while len(parts) < num_models:
                    parts.append(default_cfg)
                return parts[:num_models]
            except ValueError:
                pass

        return [default_cfg] * num_models


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
        print(f"[NV_ApplyAttentionGuidance] Available patterns: {len(guidance_data.get('layers', {}))} keys")

        # Check if guidance data is empty
        if not guidance_data.get("layers"):
            print(f"[NV_ApplyAttentionGuidance] WARNING: No attention patterns in guidance data!")
            print(f"[NV_ApplyAttentionGuidance] Returning unmodified model")
            return (model.clone(),)

        # State tracking (mutable references)
        current_step = [0]
        applications = [0]

        # Get any existing override to chain
        existing_override = None
        if "transformer_options" in model.model_options:
            existing_override = model.model_options["transformer_options"].get("optimized_attention_override")

        # Create the guidance override
        guidance_override = create_attention_guidance_override(
            guidance_data=guidance_data,
            target_steps=extract_steps,
            target_layers=target_layers,
            strength=guidance_strength,
            mode=guidance_mode,
            current_step_ref=current_step,
            applications_ref=applications,
            original_override=existing_override
        )

        # Clone model and apply the override
        patched_model = model.clone()
        if "transformer_options" not in patched_model.model_options:
            patched_model.model_options["transformer_options"] = {}
        patched_model.model_options["transformer_options"]["optimized_attention_override"] = guidance_override

        # Store step tracking info for external access (e.g., custom samplers)
        patched_model._attention_guidance_step = current_step
        patched_model._attention_guidance_applications = applications

        print(f"[NV_ApplyAttentionGuidance] Model patched with attention guidance override")

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
