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

The Capture node mirrors the NV_MultiModelSampler interface (multi-model,
committed noise, RoPE shift_t) so it can be used as a drop-in replacement.
"""

import os
import re
import torch
import torch.nn.functional as F
from typing import Dict, List, Set, Tuple, Optional
import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview


def video_to_latent_frames(video_frames: int) -> int:
    """Convert video frame count to latent frame count (Wan 4:1 compression)."""
    if video_frames <= 0:
        return 0
    return (video_frames - 1) // 4 + 1


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

    Mirrors the NV_MultiModelSampler interface: supports multi-model sequential/
    boundary sampling, committed noise, and RoPE shift_t. Captures attention at
    overlap frames for applying to subsequent chunks via NV_ApplyOverlapAttention.
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
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                # Multi-model support
                "model_2": ("MODEL",),
                "model_3": ("MODEL",),
                "mode": (["single", "sequential", "boundary"], {"default": "single"}),
                "model_steps": ("STRING", {"default": "", "multiline": False,
                    "tooltip": "Sequential mode: comma-separated steps per model (e.g., '7,7,6'). Empty = auto-divide."}),
                "model_boundaries": ("STRING", {"default": "0.875,0.5", "multiline": False,
                    "tooltip": "Boundary mode: sigma thresholds (0-1). Model switches when sigma drops below threshold."}),
                "model_cfg_scales": ("STRING", {"default": "", "multiline": False,
                    "tooltip": "Per-model CFG overrides (e.g., '7.0,5.0,3.0'). Empty = use main cfg."}),
                # Chunk consistency
                "committed_noise": ("COMMITTED_NOISE", {
                    "tooltip": "Pre-generated noise with FreeNoise correlation (from NV_CommittedNoise). If provided, seed is ignored."}),
                "enable_shift_t": ("BOOLEAN", {"default": False,
                    "tooltip": "Apply RoPE temporal offset so model knows absolute position in video."}),
                "shift_t_frames": ("INT", {"default": 0, "min": 0, "max": 10000,
                    "tooltip": "Temporal position in VIDEO frames (auto-converted to latent frames)."}),
                # Overlap capture settings
                "overlap_start_frame": ("INT", {
                    "default": -1, "min": -1,
                    "tooltip": "Start of overlap in VIDEO frames (-1 = auto: last 20% of chunk)"}),
                "overlap_end_frame": ("INT", {
                    "default": -1, "min": -1,
                    "tooltip": "End of overlap in VIDEO frames (-1 = auto: end of chunk)"}),
                "capture_layers": ("STRING", {
                    "default": "0,8,19",
                    "tooltip": "Which transformer layers to capture (fewer = smaller file)"}),
                "capture_steps": ("STRING", {
                    "default": "70-100%",
                    "tooltip": "Which steps to capture (later steps have clearer patterns)"}),
                "sparsity_ratio": ("FLOAT", {
                    "default": 0.25, "min": 0.05, "max": 1.0,
                    "tooltip": "Keep top X% of attention weights"}),
            }
        }

    RETURN_TYPES = ("LATENT", "OVERLAP_ATTENTION")
    RETURN_NAMES = ("latent", "overlap_attention")
    FUNCTION = "sample_and_capture"
    CATEGORY = "NV_Utils/sampling"
    DESCRIPTION = "Multi-model sampler that captures attention at overlap frames for chunk consistency."

    def sample_and_capture(self, model, positive, negative, latent, seed, steps, cfg,
                           sampler_name, scheduler, denoise,
                           model_2=None, model_3=None,
                           mode="single", model_steps="", model_boundaries="0.875,0.5",
                           model_cfg_scales="", committed_noise=None,
                           enable_shift_t=False, shift_t_frames=0,
                           overlap_start_frame=-1, overlap_end_frame=-1,
                           capture_layers="0,8,19", capture_steps="70-100%",
                           sparsity_ratio=0.25):

        latent_image = latent["samples"]
        b, c, t_latent, h, w = latent_image.shape
        # WAN uses 2x2 patch embedding before attention, so each frame has
        # (h/2)*(w/2) spatial tokens in the attention layer, not h*w.
        spatial_size = (h // 2) * (w // 2)

        # Convert video frames to latent frames (÷4 for WAN)
        t_video = t_latent * 4

        # Auto-calculate overlap region if not specified
        if overlap_start_frame < 0:
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

        # ============= RoPE Temporal Position Alignment (shift_t) =============
        if enable_shift_t and shift_t_frames > 0:
            shift_t_latent = video_to_latent_frames(shift_t_frames)
            model = model.clone()
            model.set_model_rope_options(
                scale_x=1.0, shift_x=0.0,
                scale_y=1.0, shift_y=0.0,
                scale_t=1.0, shift_t=float(shift_t_latent)
            )
            if model_2 is not None:
                model_2 = model_2.clone()
                model_2.set_model_rope_options(1.0, 0.0, 1.0, 0.0, 1.0, float(shift_t_latent))
            if model_3 is not None:
                model_3 = model_3.clone()
                model_3.set_model_rope_options(1.0, 0.0, 1.0, 0.0, 1.0, float(shift_t_latent))
            print(f"[NV_CaptureOverlapAttention] Applied RoPE shift_t={shift_t_latent} latent frames "
                  f"(from {shift_t_frames} video frames)")

        # Collect models
        models = [model]
        if model_2 is not None:
            models.append(model_2)
        if model_3 is not None:
            models.append(model_3)

        cfg_scales = self._parse_cfg_scales(model_cfg_scales, cfg, len(models))

        # Storage and state tracking
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
                    seq_len = q.shape[1] if q.dim() == 3 else q.shape[2]

                    # Context windows split the full sequence into smaller windows.
                    # Each attention call processes only one window's frames.
                    context_window = t_opts.get("context_window", None)
                    if context_window is not None:
                        window_indices = context_window.index_list
                        window_frames = len(window_indices)
                    else:
                        window_indices = list(range(t_latent))
                        window_frames = t_latent

                    expected_seq = window_frames * spatial_size
                    if seq_len != expected_seq:
                        return original_func(*args, **kwargs)

                    # Find which overlap frames are in this window (local indices)
                    local_overlap = []
                    for local_idx, global_idx in enumerate(window_indices):
                        if overlap_start_latent <= global_idx < overlap_end_latent:
                            local_overlap.append(local_idx)

                    if not local_overlap:
                        # No overlap frames in this window - skip, don't block key
                        return original_func(*args, **kwargs)

                    local_start = min(local_overlap)
                    local_end = max(local_overlap) + 1

                    # Compute attention
                    output, attn_weights = self._compute_attention(
                        q, k, v, heads, kwargs
                    )

                    # Extract only overlap frame attention (using local window indices)
                    overlap_attn = extract_frame_attention(
                        attn_weights,
                        local_start, local_end,
                        spatial_size, window_frames
                    )

                    # Sparsify and store
                    captured_patterns[key] = sparsify_overlap_attention(
                        overlap_attn, sparsity_ratio
                    )

                    n_overlap = local_end - local_start
                    global_start = window_indices[local_start]
                    global_end = window_indices[local_end - 1]
                    print(f"[NV_CaptureOverlapAttention] Captured {key}: "
                          f"{n_overlap}/{window_frames} window frames "
                          f"(global {global_start}-{global_end})")

                    return output

            return original_func(*args, **kwargs)

        # Patch all models with capture override
        patched_models = []
        for idx, m in enumerate(models):
            pm = m.clone()
            if "transformer_options" not in pm.model_options:
                pm.model_options["transformer_options"] = {}
            pm.model_options["transformer_options"]["optimized_attention_override"] = attention_capture_override
            patched_models.append(pm)

        print(f"[NV_CaptureOverlapAttention] Patched {len(patched_models)} model(s) with capture override")

        # Get committed noise tensor if provided
        noise_tensor = committed_noise["noise"] if committed_noise is not None else None
        if committed_noise is not None:
            print(f"[NV_CaptureOverlapAttention] Using committed noise (seed={committed_noise['seed']})")

        # Route to appropriate sampling method with step tracking
        if mode == "single" or len(patched_models) == 1:
            result = self._sample_single(
                patched_models[0], positive, negative, latent,
                seed, steps, cfg, sampler_name, scheduler, denoise,
                noise_tensor, step_ref=current_step
            )
        elif mode == "sequential":
            print(f"[NV_CaptureOverlapAttention] Sequential mode: {len(patched_models)} models")
            result = self._sample_sequential(
                patched_models, positive, negative, latent,
                seed, steps, cfg_scales, sampler_name, scheduler, denoise,
                model_steps, noise_tensor, step_ref=current_step
            )
        elif mode == "boundary":
            print(f"[NV_CaptureOverlapAttention] Boundary mode: {len(patched_models)} models")
            result = self._sample_boundary(
                patched_models, positive, negative, latent,
                seed, steps, cfg_scales, sampler_name, scheduler, denoise,
                model_boundaries, noise_tensor, step_ref=current_step
            )
        else:
            result = self._sample_single(
                patched_models[0], positive, negative, latent,
                seed, steps, cfg, sampler_name, scheduler, denoise,
                noise_tensor, step_ref=current_step
            )

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

        return (result, overlap_data)

    # =========================================================================
    # Sampling infrastructure (mirrors NV_ExtractAttentionGuidance)
    # =========================================================================

    def _create_step_callback(self, model, steps, step_ref, start_step_offset=0):
        """
        Create a callback that tracks the current step and updates step_ref.

        Args:
            start_step_offset: Offset to add to step index (for multi-model sampling
                               where step indices may be relative to current model's range)
        """
        preview_callback = latent_preview.prepare_callback(model, steps)

        def step_tracking_callback(step, x0, x, total_steps):
            # The callback is called AFTER the step completes with the step index.
            # Add offset for multi-model sampling and add 1 since we want the
            # NEXT step's index (since attention happens during, not after)
            step_ref[0] = step + start_step_offset + 1
            if preview_callback is not None:
                return preview_callback(step, x0, x, total_steps)
        return step_tracking_callback

    def _sample_single(self, model, positive, negative, latent_image,
                       seed, steps, cfg, sampler_name, scheduler, denoise,
                       noise_tensor=None, step_ref=None):
        """Single model sampling with step tracking."""
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
        print(f"[NV_CaptureOverlapAttention] Step distribution: {step_distribution}")

        current_latent = latent_image
        current_step_idx = 0

        for i, (model, model_steps, cfg) in enumerate(zip(models, step_distribution, cfg_scales)):
            if model_steps <= 0:
                continue

            start_step = current_step_idx
            end_step = current_step_idx + model_steps
            disable_noise = (i > 0)

            print(f"[NV_CaptureOverlapAttention] Model {i+1}: steps {start_step}-{end_step}, "
                  f"cfg={cfg}, disable_noise={disable_noise}")

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
        print(f"[NV_CaptureOverlapAttention] Sigma boundaries: {boundaries}")
        print(f"[NV_CaptureOverlapAttention] Step ranges: {step_ranges}")

        current_latent = latent_image

        for i, (model, (start_step, end_step), cfg) in enumerate(zip(models, step_ranges, cfg_scales)):
            if start_step >= end_step:
                continue

            disable_noise = (i > 0)

            print(f"[NV_CaptureOverlapAttention] Model {i+1}: steps {start_step}-{end_step}, cfg={cfg}")

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
            if noise_tensor.shape != latent_image.shape:
                raise ValueError(
                    f"Committed noise shape {list(noise_tensor.shape)} doesn't match "
                    f"latent shape {list(latent_image.shape)}"
                )
            noise = noise_tensor
            disable_noise = False
        elif disable_noise:
            noise = torch.zeros_like(latent_image)
        else:
            batch_inds = latent.get("batch_index", None)
            noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

        noise_mask = latent.get("noise_mask", None)

        # Create callback with step tracking
        if step_ref is not None:
            offset = start_step if start_step is not None else 0
            step_ref[0] = offset
            callback = self._create_step_callback(model, steps, step_ref, start_step_offset=offset)
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

    # =========================================================================
    # Attention computation
    # =========================================================================

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

    # =========================================================================
    # Parsing helpers
    # =========================================================================

    def _parse_steps(self, steps_str, total_steps, denoise):
        """Parse step specification (e.g., '70-100%' or '7,8,9')."""
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
                "target_overlap_start": ("INT", {
                    "default": 0, "min": 0,
                    "tooltip": "Start of overlap region in target chunk (latent frames). "
                               "Default 0 = head of chunk (most common for sequential chunks)."
                }),
                "target_overlap_end": ("INT", {
                    "default": -1, "min": -1,
                    "tooltip": "End of overlap region in target chunk (latent frames). "
                               "-1 = auto: same number of frames as captured overlap."
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "NV_Utils/sampling"
    DESCRIPTION = "Apply overlap attention for chunk consistency."

    def apply(self, model, overlap_attention, guidance_strength=0.5, guidance_mode="blend",
              target_overlap_start=0, target_overlap_end=-1):

        patterns = overlap_attention.get("patterns", {})
        metadata = overlap_attention.get("metadata", {})

        if not patterns:
            print("[NV_ApplyOverlapAttention] WARNING: No patterns to apply!")
            return (model.clone(),)

        src_overlap_start = metadata.get("overlap_start_latent", 0)
        src_overlap_end = metadata.get("overlap_end_latent", 0)
        # spatial_size is in attention tokens (post-patch-embedding), not latent pixels
        spatial_size = metadata.get("spatial_size", 220)
        n_overlap_frames = src_overlap_end - src_overlap_start

        # Target overlap region in the receiving chunk.
        # Default: head of chunk (frames 0 to N) for sequential chunk processing.
        apply_start = target_overlap_start
        apply_end = target_overlap_end if target_overlap_end > 0 else n_overlap_frames

        print(f"[NV_ApplyOverlapAttention] Applying {len(patterns)} patterns")
        print(f"  Source overlap: latent frames {src_overlap_start}-{src_overlap_end} ({n_overlap_frames} frames)")
        print(f"  Target overlap: latent frames {apply_start}-{apply_end}")
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

                # Context windows: find overlap frames in this window
                context_window = t_opts.get("context_window", None)
                if context_window is not None:
                    window_indices = context_window.index_list
                    window_frames = len(window_indices)
                else:
                    window_indices = None
                    window_frames = seq_len // spatial_size if spatial_size > 0 else 0

                expected_seq = window_frames * spatial_size
                if seq_len != expected_seq or window_frames == 0:
                    return original_func(*args, **kwargs)

                # Find target overlap frames in this window (local indices)
                if window_indices is not None:
                    local_overlap = []
                    for local_idx, global_idx in enumerate(window_indices):
                        if apply_start <= global_idx < apply_end:
                            local_overlap.append(local_idx)
                    if not local_overlap:
                        return original_func(*args, **kwargs)
                    local_start = min(local_overlap)
                    local_end = max(local_overlap) + 1
                else:
                    local_start = apply_start
                    local_end = apply_end
                    if local_end * spatial_size > seq_len:
                        return original_func(*args, **kwargs)

                start_token = local_start * spatial_size
                end_token = local_end * spatial_size
                n_local_overlap = local_end - local_start

                # Densify the guidance pattern
                guidance_attn = densify_overlap_attention(patterns[key], device=q.device)
                g_heads, g_tokens, g_total = guidance_attn.shape

                # Shape check: guidance overlap tokens must match local overlap tokens
                expected_tokens = n_local_overlap * spatial_size
                if g_tokens != expected_tokens:
                    # Mismatch - captured and target have different overlap frame counts
                    # in this window. Skip gracefully.
                    return original_func(*args, **kwargs)

                # Column dimension check: guidance columns span the capture window,
                # current attention columns span this apply window. They must match
                # (same total tokens = same window frame count) for direct blending.
                if g_total != seq_len:
                    return original_func(*args, **kwargs)

                # Compute current attention
                output, current_attn = self._compute_attention(q, k, v, heads, kwargs)

                # Apply guidance only to overlap region
                if guidance_mode == "blend":
                    current_overlap = current_attn[:, start_token:end_token, :]
                    # Both capture and apply windows have the same frame count,
                    # so guidance columns align with current attention columns.
                    blended = (1 - guidance_strength) * current_overlap + guidance_strength * guidance_attn
                    blended = blended / (blended.sum(dim=-1, keepdim=True) + 1e-8)

                    modified_attn = current_attn.clone()
                    modified_attn[:, start_token:end_token, :] = blended

                    output = self._apply_attention(modified_attn, v, heads, kwargs)

                if applications[0] == 0:
                    print(f"[NV_ApplyOverlapAttention] First guidance at step {step}, layer {block_idx}, "
                          f"local frames {local_start}-{local_end}")
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
