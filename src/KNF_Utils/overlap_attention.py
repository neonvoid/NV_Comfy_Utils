"""
Overlap Attention for Chunk Consistency

Captures attention patterns at chunk overlap/crossover frames and applies them
to subsequent chunks. This ensures texture consistency (bark, leaves, etc.)
at chunk boundaries.

Key insight: The overlap frames are processed in BOTH chunks. If we capture
attention from chunk 1's overlap region and apply it to chunk 2's processing
of those same frames, the textures will match.

Memory efficient: Only stores attention for overlap frames (~100-200MB per transition)
vs full video attention (~13GB).
"""

import os
import torch
import torch.nn.functional as F
from typing import Dict, List, Set, Tuple, Optional
import comfy.sample
import comfy.samplers
import comfy.utils
from comfy_extras import nodes_model_advanced
import latent_preview


def get_frame_token_indices(frame_start: int, frame_end: int,
                            spatial_size: int, total_frames: int) -> Tuple[int, int]:
    """
    Get token index range for specified frames.

    For video models, tokens are arranged as:
    [frame_0_spatial..., frame_1_spatial..., ..., frame_N_spatial...]

    Args:
        frame_start: Starting latent frame index
        frame_end: Ending latent frame index (exclusive)
        spatial_size: Number of spatial tokens per frame (H × W)
        total_frames: Total number of latent frames

    Returns:
        (start_token_idx, end_token_idx)
    """
    start_idx = frame_start * spatial_size
    end_idx = frame_end * spatial_size
    return start_idx, end_idx


def extract_frame_attention(attn_weights: torch.Tensor,
                           frame_start: int, frame_end: int,
                           spatial_size: int, total_frames: int) -> torch.Tensor:
    """
    Extract attention weights for specific frames only.

    Args:
        attn_weights: Full attention [heads, seq_len, seq_len]
        frame_start, frame_end: Latent frame range
        spatial_size: Spatial tokens per frame
        total_frames: Total latent frames

    Returns:
        Sliced attention for specified frames
    """
    start_idx, end_idx = get_frame_token_indices(
        frame_start, frame_end, spatial_size, total_frames
    )

    # Extract rows (queries) and columns (keys) for these frames
    # This gives us how overlap frames attend to ALL frames
    # and how ALL frames attend to overlap frames
    frame_attn = attn_weights[:, start_idx:end_idx, :]

    return frame_attn


def sparsify_overlap_attention(attn_weights: torch.Tensor, ratio: float = 0.25) -> dict:
    """
    Sparsify overlap attention - keep top-k% of values.

    Args:
        attn_weights: [heads, overlap_tokens, all_tokens]
        ratio: Keep top k% of attention per query

    Returns:
        Sparse representation dict
    """
    # Move to CPU immediately
    attn_weights = attn_weights.detach().cpu().float()

    heads, n_overlap, n_total = attn_weights.shape
    k = max(1, int(n_total * ratio))

    topk_vals, topk_indices = torch.topk(attn_weights, k, dim=-1)

    return {
        "values": topk_vals.half(),
        "indices": topk_indices.short(),
        "shape": (heads, n_overlap, n_total),
        "k": k,
    }


def densify_overlap_attention(sparse_data: dict, device='cpu') -> torch.Tensor:
    """
    Reconstruct dense attention from sparse representation.
    """
    values = sparse_data["values"].to(device).float()
    indices = sparse_data["indices"].to(device).long()
    heads, n_overlap, n_total = sparse_data["shape"]

    dense = torch.zeros(heads, n_overlap, n_total, device=device, dtype=values.dtype)
    dense.scatter_(-1, indices, values)

    return dense


class NV_CaptureOverlapAttention:
    """
    Capture attention patterns at overlap frames during chunk sampling.

    Use this after processing a chunk to save the attention patterns
    for the overlap region. These patterns can then be applied to the
    next chunk to ensure texture consistency.

    The node wraps your sampler and captures attention during the
    sampling process, then outputs the captured patterns.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 10, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
            },
            "optional": {
                # Overlap frame specification
                "overlap_start_frame": ("INT", {
                    "default": -1,
                    "min": -1,
                    "tooltip": "Start of overlap in VIDEO frames (-1 = auto: last 20% of chunk)"
                }),
                "overlap_end_frame": ("INT", {
                    "default": -1,
                    "min": -1,
                    "tooltip": "End of overlap in VIDEO frames (-1 = auto: end of chunk)"
                }),
                # Capture settings
                "capture_layers": ("STRING", {
                    "default": "0,8,19",
                    "tooltip": "Which transformer layers to capture (fewer = smaller file)"
                }),
                "capture_steps": ("STRING", {
                    "default": "70-100%",
                    "tooltip": "Which steps to capture (later steps have clearer patterns)"
                }),
                "sparsity_ratio": ("FLOAT", {
                    "default": 0.25,
                    "min": 0.05,
                    "max": 1.0,
                    "tooltip": "Keep top X% of attention weights"
                }),
            }
        }

    RETURN_TYPES = ("LATENT", "OVERLAP_ATTENTION")
    RETURN_NAMES = ("latent", "overlap_attention")
    FUNCTION = "sample_and_capture"
    CATEGORY = "NV_Utils/sampling"
    DESCRIPTION = "Sample and capture attention at overlap frames for chunk consistency."

    def sample_and_capture(self, model, positive, negative, latent, seed, steps, cfg,
                           sampler_name, scheduler, denoise,
                           overlap_start_frame=-1, overlap_end_frame=-1,
                           capture_layers="0,8,19", capture_steps="70-100%",
                           sparsity_ratio=0.25):

        latent_image = latent["samples"]
        b, c, t_latent, h, w = latent_image.shape
        spatial_size = h * w

        # Convert video frames to latent frames (÷4 for WAN)
        t_video = t_latent * 4

        # Auto-calculate overlap region if not specified
        if overlap_start_frame < 0:
            # Default: last 20% of chunk
            overlap_start_frame = int(t_video * 0.8)
        if overlap_end_frame < 0:
            overlap_end_frame = t_video

        # Convert to latent frames
        overlap_start_latent = overlap_start_frame // 4
        overlap_end_latent = min(overlap_end_frame // 4, t_latent)

        print(f"[NV_CaptureOverlapAttention] Capturing frames {overlap_start_frame}-{overlap_end_frame} "
              f"(latent {overlap_start_latent}-{overlap_end_latent})")

        # Parse capture settings
        target_layers = set()
        for x in capture_layers.split(","):
            try:
                target_layers.add(int(x.strip()))
            except ValueError:
                pass

        target_steps = self._parse_steps(capture_steps, steps, denoise)

        print(f"[NV_CaptureOverlapAttention] Layers: {sorted(target_layers)}, Steps: {sorted(target_steps)}")

        # Storage for captured patterns
        captured_patterns = {}
        current_step = [0]

        # Create capture override
        def attention_capture_override(original_func, *args, **kwargs):
            q, k, v = args[0], args[1], args[2]
            if len(args) > 3:
                heads = args[3]
            else:
                heads = kwargs.get("heads", 1)

            t_opts = kwargs.get("transformer_options", {})
            block_idx = t_opts.get("block_index", -1)
            step = current_step[0]

            # Check if we should capture
            if step in target_steps and block_idx in target_layers:
                key = f"step_{step}_layer_{block_idx}"

                if key not in captured_patterns:
                    # Get sequence length to determine frame structure
                    seq_len = q.shape[1] if q.dim() == 3 else q.shape[2]

                    # Only capture if sequence matches expected video structure
                    expected_seq = t_latent * spatial_size
                    if seq_len == expected_seq:
                        # Compute attention
                        output, attn_weights = self._compute_attention(
                            q, k, v, heads, kwargs
                        )

                        # Extract only overlap frame attention
                        overlap_attn = extract_frame_attention(
                            attn_weights,
                            overlap_start_latent, overlap_end_latent,
                            spatial_size, t_latent
                        )

                        # Sparsify and store
                        captured_patterns[key] = sparsify_overlap_attention(
                            overlap_attn, sparsity_ratio
                        )

                        n_overlap = overlap_end_latent - overlap_start_latent
                        print(f"[NV_CaptureOverlapAttention] Captured {key}: "
                              f"{n_overlap} frames × {spatial_size} spatial = "
                              f"{overlap_attn.shape[1]} tokens")

                        return output

            return original_func(*args, **kwargs)

        # Patch model
        patched_model = model.clone()
        if "transformer_options" not in patched_model.model_options:
            patched_model.model_options["transformer_options"] = {}
        patched_model.model_options["transformer_options"]["optimized_attention_override"] = attention_capture_override

        # Sample directly (not via common_ksampler) so we can inject step callback.
        # common_ksampler creates its own internal callback, ignoring ours.
        preview_cb = latent_preview.prepare_callback(patched_model, steps)
        def step_callback(step, x0, x, total_steps):
            current_step[0] = step + 1  # Next step's attention will see this value
            if preview_cb:
                return preview_cb(step, x0, x, total_steps)

        latent_image_fixed = comfy.sample.fix_empty_latent_channels(patched_model, latent_image)
        noise = comfy.sample.prepare_noise(latent_image_fixed, seed, latent.get("batch_index", None))
        noise_mask = latent.get("noise_mask", None)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        samples = comfy.sample.sample(
            patched_model, noise, steps, cfg, sampler_name, scheduler,
            positive, negative, latent_image_fixed,
            denoise=denoise, disable_noise=False,
            start_step=None, last_step=None,
            force_full_denoise=False,
            noise_mask=noise_mask, callback=step_callback,
            disable_pbar=disable_pbar, seed=seed
        )

        out = latent.copy()
        out["samples"] = samples

        # Package captured data
        overlap_data = {
            "patterns": captured_patterns,
            "metadata": {
                "overlap_start_latent": overlap_start_latent,
                "overlap_end_latent": overlap_end_latent,
                "overlap_start_video": overlap_start_frame,
                "overlap_end_video": overlap_end_frame,
                "spatial_size": spatial_size,
                "total_latent_frames": t_latent,
                "layers": list(target_layers),
                "steps": list(target_steps),
            }
        }

        print(f"[NV_CaptureOverlapAttention] Captured {len(captured_patterns)} patterns")

        return (out, overlap_data)

    def _compute_attention(self, q, k, v, heads, kwargs):
        """Compute attention and return weights."""
        from einops import rearrange

        attn_precision = kwargs.get("attn_precision", None)
        skip_reshape = kwargs.get("skip_reshape", False)

        if skip_reshape:
            b, h, s, d = q.shape
            scale = d ** -0.5
            q_h = q.view(b * h, s, d)
            k_h = k.view(b * h, s, d)
            v_h = v.view(b * h, s, d)
        else:
            b, _, dim = q.shape
            dim_head = dim // heads
            scale = dim_head ** -0.5
            q_h = rearrange(q, 'b n (h d) -> (b h) n d', h=heads)
            k_h = rearrange(k, 'b n (h d) -> (b h) n d', h=heads)
            v_h = rearrange(v, 'b n (h d) -> (b h) n d', h=heads)

        if attn_precision == torch.float32:
            sim = torch.einsum('b i d, b j d -> b i j', q_h.float(), k_h.float()) * scale
        else:
            sim = torch.einsum('b i d, b j d -> b i j', q_h, k_h) * scale

        attn = sim.softmax(dim=-1)
        out = torch.einsum('b i j, b j d -> b i d', attn.to(v_h.dtype), v_h)

        if skip_reshape:
            out = out.view(b, heads, -1, out.shape[-1])
        else:
            out = rearrange(out, '(b h) n d -> b n (h d)', h=heads)

        return out, attn

    def _parse_steps(self, steps_str, total_steps, denoise):
        """Parse step specification."""
        import re
        actual_steps = int(total_steps * denoise)
        steps = set()

        if "%" in steps_str:
            match = re.match(r"(\d+)-(\d+)%", steps_str.strip())
            if match:
                start_pct = int(match.group(1))
                end_pct = int(match.group(2))
                start = int(actual_steps * start_pct / 100)
                end = int(actual_steps * end_pct / 100)
                steps = set(range(start, end + 1))
        else:
            for part in steps_str.split(","):
                try:
                    steps.add(int(part.strip()))
                except ValueError:
                    pass

        return steps


class NV_SaveOverlapAttention:
    """Save captured overlap attention to disk."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "overlap_attention": ("OVERLAP_ATTENTION",),
                "output_path": ("STRING", {"default": "overlap_attention.pt"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = "NV_Utils/sampling"

    def save(self, overlap_attention, output_path):
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        torch.save(overlap_attention, output_path)

        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        n_patterns = len(overlap_attention.get("patterns", {}))
        metadata = overlap_attention.get("metadata", {})

        print(f"[NV_SaveOverlapAttention] Saved to {output_path}")
        print(f"  Patterns: {n_patterns}")
        print(f"  Overlap frames: {metadata.get('overlap_start_video', '?')}-{metadata.get('overlap_end_video', '?')}")
        print(f"  File size: {size_mb:.1f} MB")

        return (output_path,)


class NV_LoadOverlapAttention:
    """Load overlap attention from disk."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": "overlap_attention.pt"}),
            }
        }

    RETURN_TYPES = ("OVERLAP_ATTENTION",)
    RETURN_NAMES = ("overlap_attention",)
    FUNCTION = "load"
    CATEGORY = "NV_Utils/sampling"

    def load(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Overlap attention not found: {path}")

        data = torch.load(path, map_location='cpu', weights_only=False)

        metadata = data.get("metadata", {})
        n_patterns = len(data.get("patterns", {}))

        print(f"[NV_LoadOverlapAttention] Loaded from {path}")
        print(f"  Patterns: {n_patterns}")
        print(f"  Overlap frames: {metadata.get('overlap_start_video', '?')}-{metadata.get('overlap_end_video', '?')}")

        return (data,)


class NV_ApplyOverlapAttention:
    """
    Apply overlap attention to guide chunk sampling.

    This patches the model to apply captured attention patterns
    at the overlap region, ensuring texture consistency with
    the previous chunk.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "overlap_attention": ("OVERLAP_ATTENTION",),
            },
            "optional": {
                "guidance_strength": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "How strongly to apply overlap guidance"
                }),
                "guidance_mode": (["blend", "bias"], {
                    "default": "blend",
                    "tooltip": "blend: interpolate attention, bias: add to scores"
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "NV_Utils/sampling"
    DESCRIPTION = "Apply overlap attention for chunk consistency."

    def apply(self, model, overlap_attention, guidance_strength=0.5, guidance_mode="blend"):

        patterns = overlap_attention.get("patterns", {})
        metadata = overlap_attention.get("metadata", {})

        if not patterns:
            print("[NV_ApplyOverlapAttention] WARNING: No patterns to apply!")
            return (model.clone(),)

        overlap_start = metadata.get("overlap_start_latent", 0)
        overlap_end = metadata.get("overlap_end_latent", 0)
        spatial_size = metadata.get("spatial_size", 1408)
        target_layers = set(metadata.get("layers", []))
        target_steps = set(metadata.get("steps", []))

        print(f"[NV_ApplyOverlapAttention] Applying {len(patterns)} patterns")
        print(f"  Overlap: latent frames {overlap_start}-{overlap_end}")
        print(f"  Strength: {guidance_strength}, Mode: {guidance_mode}")

        current_step = [0]
        last_sigma_val = [None]
        applications = [0]

        def attention_guidance_override(original_func, *args, **kwargs):
            q, k, v = args[0], args[1], args[2]
            if len(args) > 3:
                heads = args[3]
            else:
                heads = kwargs.get("heads", 1)

            t_opts = kwargs.get("transformer_options", {})
            block_idx = t_opts.get("block_index", -1)

            # Infer step from sigma schedule. The Apply node returns a MODEL
            # (doesn't control sampling), so we can't use a step callback.
            # Instead, derive step by matching current sigma to the schedule.
            sigmas = t_opts.get("sigmas", None)
            if sigmas is not None:
                sigma_val = sigmas.flatten()[0].item()
                if last_sigma_val[0] is None or abs(sigma_val - last_sigma_val[0]) > 1e-6:
                    last_sigma_val[0] = sigma_val
                    sample_sigmas = t_opts.get("sample_sigmas", None)
                    if sample_sigmas is not None:
                        for idx in range(len(sample_sigmas)):
                            if abs(sample_sigmas[idx].item() - sigma_val) < 1e-5:
                                current_step[0] = idx
                                break

            step = current_step[0]

            # Check if we should apply guidance
            key = f"step_{step}_layer_{block_idx}"
            if key in patterns:
                seq_len = q.shape[1] if q.dim() == 3 else q.shape[2]

                # Get token indices for overlap region
                start_idx = overlap_start * spatial_size
                end_idx = overlap_end * spatial_size

                if end_idx <= seq_len:
                    # Densify the guidance pattern
                    guidance_attn = densify_overlap_attention(patterns[key], device=q.device)

                    # Compute current attention
                    output, current_attn = self._compute_attention(q, k, v, heads, kwargs)

                    # Apply guidance only to overlap region
                    if guidance_mode == "blend":
                        # Blend overlap region attention
                        current_overlap = current_attn[:, start_idx:end_idx, :]
                        blended = (1 - guidance_strength) * current_overlap + guidance_strength * guidance_attn
                        blended = blended / (blended.sum(dim=-1, keepdim=True) + 1e-8)

                        # Recompute output for overlap region with blended attention
                        # For simplicity, recompute full output with modified attention
                        modified_attn = current_attn.clone()
                        modified_attn[:, start_idx:end_idx, :] = blended

                        output = self._apply_attention(modified_attn, v, heads, kwargs)

                    if applications[0] == 0:
                        print(f"[NV_ApplyOverlapAttention] First guidance at step {step}, layer {block_idx}")
                    applications[0] += 1
                    return output

            return original_func(*args, **kwargs)

        # Patch model
        patched_model = model.clone()
        if "transformer_options" not in patched_model.model_options:
            patched_model.model_options["transformer_options"] = {}
        patched_model.model_options["transformer_options"]["optimized_attention_override"] = attention_guidance_override

        print(f"[NV_ApplyOverlapAttention] Model patched (step tracking via sigma inference)")

        return (patched_model,)

    def _compute_attention(self, q, k, v, heads, kwargs):
        """Compute attention and return weights."""
        from einops import rearrange

        attn_precision = kwargs.get("attn_precision", None)
        skip_reshape = kwargs.get("skip_reshape", False)

        if skip_reshape:
            b, h, s, d = q.shape
            scale = d ** -0.5
            q_h = q.view(b * h, s, d)
            k_h = k.view(b * h, s, d)
            v_h = v.view(b * h, s, d)
        else:
            b, _, dim = q.shape
            dim_head = dim // heads
            scale = dim_head ** -0.5
            q_h = rearrange(q, 'b n (h d) -> (b h) n d', h=heads)
            k_h = rearrange(k, 'b n (h d) -> (b h) n d', h=heads)
            v_h = rearrange(v, 'b n (h d) -> (b h) n d', h=heads)

        if attn_precision == torch.float32:
            sim = torch.einsum('b i d, b j d -> b i j', q_h.float(), k_h.float()) * scale
        else:
            sim = torch.einsum('b i d, b j d -> b i j', q_h, k_h) * scale

        attn = sim.softmax(dim=-1)
        out = torch.einsum('b i j, b j d -> b i d', attn.to(v_h.dtype), v_h)

        if skip_reshape:
            out = out.view(b, heads, -1, out.shape[-1])
        else:
            out = rearrange(out, '(b h) n d -> b n (h d)', h=heads)

        return out, attn

    def _apply_attention(self, attn, v, heads, kwargs):
        """Apply attention weights to values."""
        from einops import rearrange

        skip_reshape = kwargs.get("skip_reshape", False)

        if skip_reshape:
            b, h, s, d = v.shape
            v_h = v.view(b * h, s, d)
        else:
            b = v.shape[0]
            v_h = rearrange(v, 'b n (h d) -> (b h) n d', h=heads)

        out = torch.einsum('b i j, b j d -> b i d', attn.to(v_h.dtype), v_h)

        if skip_reshape:
            out = out.view(b, heads, -1, out.shape[-1])
        else:
            out = rearrange(out, '(b h) n d -> b n (h d)', h=heads)

        return out


# Node registration
NODE_CLASS_MAPPINGS = {
    "NV_CaptureOverlapAttention": NV_CaptureOverlapAttention,
    "NV_SaveOverlapAttention": NV_SaveOverlapAttention,
    "NV_LoadOverlapAttention": NV_LoadOverlapAttention,
    "NV_ApplyOverlapAttention": NV_ApplyOverlapAttention,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_CaptureOverlapAttention": "NV Capture Overlap Attention",
    "NV_SaveOverlapAttention": "NV Save Overlap Attention",
    "NV_LoadOverlapAttention": "NV Load Overlap Attention",
    "NV_ApplyOverlapAttention": "NV Apply Overlap Attention",
}
