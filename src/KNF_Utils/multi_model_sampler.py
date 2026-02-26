"""
NV Multi-Model Sampler v2

A clean, KSampler-like node that supports multi-model sequential/boundary sampling.
Designed to work alongside ComfyUI native nodes (WAN Context Windows, WanVaceToVideo).

Modes:
- single: Standard KSampler behavior (uses model only)
- sequential: Divides steps among models (e.g., model_1 does steps 0-6, model_2 does 7-13, etc.)
- boundary: Switches models based on sigma thresholds (noise level)

Chunk Consistency Features:
- committed_noise: Pre-generated noise for consistent chunked processing
- enable_shift_t: RoPE temporal offset for position awareness
"""

import torch
import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview
from nodes import common_ksampler

from .chunk_utils import video_to_latent_frames


class NV_MultiModelSampler:
    """
    Multi-model sampler with KSampler-compatible interface.

    Supports three sampling modes:
    - single: Identical to KSampler (ignores model_2, model_3)
    - sequential: Chains models by step count
    - boundary: Switches models by sigma (noise) thresholds
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
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "model_2": ("MODEL",),
                "model_3": ("MODEL",),
                "sampler_name_override": ("STRING", {"forceInput": True,
                    "tooltip": "Overrides sampler dropdown when connected (e.g., from Sweep Loader). Must be a valid sampler name."}),
                "scheduler_override": ("STRING", {"forceInput": True,
                    "tooltip": "Overrides scheduler dropdown when connected (e.g., from Sweep Loader). Must be a valid scheduler name."}),
                "mode": (["single", "sequential", "boundary"], {"default": "single"}),
                "model_steps": ("STRING", {"default": "", "multiline": False,
                    "tooltip": "Sequential mode: comma-separated steps per model (e.g., '7,7,6' for 20 steps). Empty = auto-divide."}),
                "model_boundaries": ("STRING", {"default": "0.875,0.5", "multiline": False,
                    "tooltip": "Boundary mode: sigma thresholds (0-1). Model switches when sigma drops below threshold."}),
                "model_cfg_scales": ("STRING", {"default": "", "multiline": False,
                    "tooltip": "Per-model CFG overrides (e.g., '7.0,5.0,3.0'). Empty = use main cfg."}),
                # Chunk consistency options
                "committed_noise": ("COMMITTED_NOISE", {
                    "tooltip": "Pre-generated noise with FreeNoise correlation (from NV_CommittedNoise). If provided, seed is ignored."
                }),
                "enable_shift_t": ("BOOLEAN", {"default": False,
                    "tooltip": "Apply RoPE temporal offset so model knows absolute position in video."}),
                "shift_t_frames": ("INT", {"default": 0, "min": 0, "max": 10000,
                    "tooltip": "Temporal position in VIDEO frames (auto-converted to latent frames). Use when processing a specific chunk."}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "sample"
    CATEGORY = "NV_Utils/sampling"
    DESCRIPTION = "KSampler with multi-model support. Use 'single' for standard sampling, 'sequential' to chain models by steps, 'boundary' to switch by noise level."

    def sample(self, model, positive, negative, latent_image, seed, steps, cfg,
               sampler_name, scheduler, denoise, model_2=None, model_3=None,
               sampler_name_override=None, scheduler_override=None,
               mode="single", model_steps="", model_boundaries="0.875,0.5",
               model_cfg_scales="", committed_noise=None, enable_shift_t=False,
               shift_t_frames=0):

        # ============= Override sampler/scheduler from STRING inputs =============
        if sampler_name_override is not None and sampler_name_override.strip():
            sampler_name_override = sampler_name_override.strip()
            if sampler_name_override not in comfy.samplers.KSampler.SAMPLERS:
                raise ValueError(
                    f"Invalid sampler_name_override '{sampler_name_override}'. "
                    f"Valid samplers: {', '.join(comfy.samplers.KSampler.SAMPLERS)}"
                )
            print(f"[NV_MultiModelSampler] Sampler override: {sampler_name} → {sampler_name_override}")
            sampler_name = sampler_name_override

        if scheduler_override is not None and scheduler_override.strip():
            scheduler_override = scheduler_override.strip()
            if scheduler_override not in comfy.samplers.KSampler.SCHEDULERS:
                raise ValueError(
                    f"Invalid scheduler_override '{scheduler_override}'. "
                    f"Valid schedulers: {', '.join(comfy.samplers.KSampler.SCHEDULERS)}"
                )
            print(f"[NV_MultiModelSampler] Scheduler override: {scheduler} → {scheduler_override}")
            scheduler = scheduler_override

        # ============= RoPE Temporal Position Alignment (shift_t) =============
        if enable_shift_t and shift_t_frames > 0:
            shift_t_latent = video_to_latent_frames(shift_t_frames)
            model = model.clone()
            model.set_model_rope_options(
                scale_x=1.0, shift_x=0.0,
                scale_y=1.0, shift_y=0.0,
                scale_t=1.0, shift_t=float(shift_t_latent)
            )
            print(f"[NV_MultiModelSampler] Applied RoPE shift_t={shift_t_latent} latent frames "
                  f"(from {shift_t_frames} video frames)")

            # Also apply to secondary models if provided
            if model_2 is not None:
                model_2 = model_2.clone()
                model_2.set_model_rope_options(1.0, 0.0, 1.0, 0.0, 1.0, float(shift_t_latent))
            if model_3 is not None:
                model_3 = model_3.clone()
                model_3.set_model_rope_options(1.0, 0.0, 1.0, 0.0, 1.0, float(shift_t_latent))

        # Collect available models
        models = [model]
        if model_2 is not None:
            models.append(model_2)
        if model_3 is not None:
            models.append(model_3)

        # Parse cfg scales
        cfg_scales = self._parse_cfg_scales(model_cfg_scales, cfg, len(models))

        # Route to appropriate sampling method
        if mode == "single" or len(models) == 1:
            return self._sample_single(
                model, positive, negative, latent_image,
                seed, steps, cfg, sampler_name, scheduler, denoise,
                committed_noise=committed_noise
            )
        elif mode == "sequential":
            return self._sample_sequential(
                models, positive, negative, latent_image,
                seed, steps, cfg_scales, sampler_name, scheduler, denoise,
                model_steps, committed_noise=committed_noise
            )
        elif mode == "boundary":
            return self._sample_boundary(
                models, positive, negative, latent_image,
                seed, steps, cfg_scales, sampler_name, scheduler, denoise,
                model_boundaries, committed_noise=committed_noise
            )
        else:
            # Fallback to single
            return self._sample_single(
                model, positive, negative, latent_image,
                seed, steps, cfg, sampler_name, scheduler, denoise,
                committed_noise=committed_noise
            )

    def _sample_single(self, model, positive, negative, latent_image,
                       seed, steps, cfg, sampler_name, scheduler, denoise,
                       committed_noise=None):
        """Standard KSampler behavior - with optional committed noise support."""
        if committed_noise is not None:
            return self._ksampler_with_noise(
                model, committed_noise["noise"], seed, steps, cfg, sampler_name, scheduler,
                positive, negative, latent_image, denoise=denoise
            )
        return common_ksampler(
            model, seed, steps, cfg, sampler_name, scheduler,
            positive, negative, latent_image, denoise=denoise
        )

    def _sample_sequential(self, models, positive, negative, latent_image,
                           seed, steps, cfg_scales, sampler_name, scheduler,
                           denoise, model_steps_str, committed_noise=None):
        """
        Chain models by step count.

        Each model processes a portion of the total steps, passing the latent
        to the next model. Later models use disable_noise=True to continue
        from the previous model's output.
        """
        # Parse step distribution
        step_distribution = self._parse_step_distribution(model_steps_str, steps, len(models))

        print(f"[NV_MultiModelSampler] Sequential mode: {len(models)} models")
        print(f"[NV_MultiModelSampler] Step distribution: {step_distribution}")
        if committed_noise is not None:
            print(f"[NV_MultiModelSampler] Using committed noise (seed={committed_noise['seed']})")

        current_latent = latent_image
        current_step = 0

        # Get committed noise tensor if provided
        noise_tensor = committed_noise["noise"] if committed_noise is not None else None

        for i, (model, model_steps, cfg) in enumerate(zip(models, step_distribution, cfg_scales)):
            if model_steps <= 0:
                continue

            start_step = current_step
            end_step = current_step + model_steps

            # First model gets noise, subsequent models continue without noise
            disable_noise = (i > 0)

            print(f"[NV_MultiModelSampler] Model {i+1}: steps {start_step}-{end_step}, cfg={cfg}, disable_noise={disable_noise}")

            if noise_tensor is not None and not disable_noise:
                # Use committed noise for first model
                result = self._ksampler_with_noise(
                    model, noise_tensor, seed, steps, cfg, sampler_name, scheduler,
                    positive, negative, current_latent,
                    denoise=denoise,
                    start_step=start_step,
                    last_step=end_step,
                    force_full_denoise=(i == len(models) - 1)
                )
            else:
                result = common_ksampler(
                    model, seed, steps, cfg, sampler_name, scheduler,
                    positive, negative, current_latent,
                    denoise=denoise,
                    disable_noise=disable_noise,
                    start_step=start_step,
                    last_step=end_step,
                    force_full_denoise=(i == len(models) - 1)
                )

            current_latent = result[0]
            current_step = end_step

        return (current_latent,)

    def _sample_boundary(self, models, positive, negative, latent_image,
                         seed, steps, cfg_scales, sampler_name, scheduler,
                         denoise, model_boundaries_str, committed_noise=None):
        """
        Switch models based on sigma (noise level) thresholds.

        Sigma starts at 1.0 (full noise) and decreases toward 0.0 (clean image).
        When sigma drops below a threshold, the next model takes over.

        Example: boundaries="0.875,0.5" with 3 models
        - Model 1: sigma 1.0 -> 0.875 (initial high-noise denoising)
        - Model 2: sigma 0.875 -> 0.5 (mid-range refinement)
        - Model 3: sigma 0.5 -> 0.0 (final detail pass)
        """
        # Parse boundaries
        boundaries = self._parse_boundaries(model_boundaries_str, len(models))

        print(f"[NV_MultiModelSampler] Boundary mode: {len(models)} models")
        print(f"[NV_MultiModelSampler] Sigma boundaries: {boundaries}")
        if committed_noise is not None:
            print(f"[NV_MultiModelSampler] Using committed noise (seed={committed_noise['seed']})")

        # Calculate step ranges from boundaries
        # We need to map sigma thresholds to step indices
        step_ranges = self._boundaries_to_steps(boundaries, steps, denoise)

        print(f"[NV_MultiModelSampler] Calculated step ranges: {step_ranges}")

        # Get committed noise tensor if provided
        noise_tensor = committed_noise["noise"] if committed_noise is not None else None

        current_latent = latent_image

        for i, (model, (start_step, end_step), cfg) in enumerate(zip(models, step_ranges, cfg_scales)):
            if start_step >= end_step:
                continue

            disable_noise = (i > 0)

            print(f"[NV_MultiModelSampler] Model {i+1}: steps {start_step}-{end_step}, cfg={cfg}")

            if noise_tensor is not None and not disable_noise:
                # Use committed noise for first model
                result = self._ksampler_with_noise(
                    model, noise_tensor, seed, steps, cfg, sampler_name, scheduler,
                    positive, negative, current_latent,
                    denoise=denoise,
                    start_step=start_step,
                    last_step=end_step,
                    force_full_denoise=(i == len(models) - 1)
                )
            else:
                result = common_ksampler(
                    model, seed, steps, cfg, sampler_name, scheduler,
                    positive, negative, current_latent,
                    denoise=denoise,
                    disable_noise=disable_noise,
                    start_step=start_step,
                    last_step=end_step,
                    force_full_denoise=(i == len(models) - 1)
                )

            current_latent = result[0]

        return (current_latent,)

    def _ksampler_with_noise(self, model, noise, seed, steps, cfg, sampler_name, scheduler,
                              positive, negative, latent, denoise=1.0, start_step=None,
                              last_step=None, force_full_denoise=False):
        """
        KSampler-like function that accepts pre-generated noise.

        This mirrors common_ksampler but uses provided noise tensor instead of
        generating new noise from seed.
        """
        latent_image = latent["samples"]
        latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image,
                                                               latent.get("downscale_ratio_spacial", None))

        # Validate noise shape matches latent
        if noise.shape != latent_image.shape:
            raise ValueError(
                f"Committed noise shape {list(noise.shape)} doesn't match "
                f"latent shape {list(latent_image.shape)}"
            )

        noise_mask = latent.get("noise_mask", None)

        callback = latent_preview.prepare_callback(model, steps)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        samples = comfy.sample.sample(
            model, noise, steps, cfg, sampler_name, scheduler,
            positive, negative, latent_image,
            denoise=denoise, disable_noise=False,
            start_step=start_step, last_step=last_step,
            force_full_denoise=force_full_denoise,
            noise_mask=noise_mask, callback=callback,
            disable_pbar=disable_pbar, seed=seed
        )

        out = latent.copy()
        out.pop("downscale_ratio_spacial", None)
        out["samples"] = samples
        return (out,)

    def _parse_step_distribution(self, model_steps_str, total_steps, num_models):
        """Parse step distribution string or auto-divide steps."""
        if model_steps_str.strip():
            try:
                parts = [int(x.strip()) for x in model_steps_str.split(",") if x.strip()]
                # Pad with zeros if not enough values
                while len(parts) < num_models:
                    parts.append(0)
                return parts[:num_models]
            except ValueError:
                pass

        # Auto-divide: split steps as evenly as possible
        base_steps = total_steps // num_models
        remainder = total_steps % num_models
        distribution = []
        for i in range(num_models):
            # Distribute remainder across first models
            extra = 1 if i < remainder else 0
            distribution.append(base_steps + extra)
        return distribution

    def _parse_boundaries(self, boundaries_str, num_models):
        """Parse sigma boundary thresholds."""
        boundaries = [1.0]  # Always start at 1.0

        if boundaries_str.strip():
            try:
                parts = [float(x.strip()) for x in boundaries_str.split(",") if x.strip()]
                boundaries.extend(parts)
            except ValueError:
                pass

        # Ensure we have enough boundaries (one fewer than models, plus implicit 0.0 at end)
        # For N models, we need N-1 explicit boundaries
        while len(boundaries) < num_models:
            # Auto-generate evenly spaced boundaries
            last = boundaries[-1] if boundaries else 1.0
            boundaries.append(last / 2)

        boundaries.append(0.0)  # Always end at 0.0
        return boundaries[:num_models + 1]

    def _boundaries_to_steps(self, boundaries, total_steps, denoise):
        """
        Convert sigma boundaries to step ranges.

        Sigma decreases approximately linearly with steps (scheduler-dependent).
        This is a simplified approximation - actual sigma schedule varies by scheduler.
        """
        step_ranges = []

        for i in range(len(boundaries) - 1):
            upper_sigma = boundaries[i]
            lower_sigma = boundaries[i + 1]

            # Map sigma to step (linear approximation)
            # sigma=1.0 -> step=0, sigma=0.0 -> step=total_steps
            start_step = int((1.0 - upper_sigma) * total_steps)
            end_step = int((1.0 - lower_sigma) * total_steps)

            # Clamp to valid range
            start_step = max(0, min(start_step, total_steps))
            end_step = max(0, min(end_step, total_steps))

            step_ranges.append((start_step, end_step))

        return step_ranges

    def _parse_cfg_scales(self, cfg_scales_str, default_cfg, num_models):
        """Parse per-model CFG scales or use default."""
        if cfg_scales_str.strip():
            try:
                parts = [float(x.strip()) for x in cfg_scales_str.split(",") if x.strip()]
                # Pad with default if not enough values
                while len(parts) < num_models:
                    parts.append(default_cfg)
                return parts[:num_models]
            except ValueError:
                pass

        return [default_cfg] * num_models


# Node mappings for registration
NODE_CLASS_MAPPINGS = {
    "NV_MultiModelSampler": NV_MultiModelSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_MultiModelSampler": "NV Multi-Model Sampler",
}
