"""
NV Multi-Model Sampler v3

A KSampler-like node that supports multi-model sequential sampling and
cascaded pipeline integration (pre-noised latents, shift override, step ranges).

Modes:
- single: Standard KSampler behavior (uses model only)
- sequential: Divides steps among models (e.g., model_1 does steps 0-6, model_2 does 7-13)

Cascaded Pipeline Features:
- shift_override: Patches model sigma schedule (wire from NV_PreNoiseLatent)
- add_noise: Disable for pre-noised latents
- start_at_step / end_at_step: Sub-range execution for partial denoising
- committed_noise: Pre-generated noise for consistent chunked processing
"""

import comfy.sample
import comfy.samplers
import comfy.model_sampling
import comfy.utils
import latent_preview
from nodes import common_ksampler


class NV_MultiModelSampler:
    """
    Multi-model sampler with KSampler-compatible interface.

    Supports two sampling modes:
    - single: Identical to KSampler (ignores model_2, model_3)
    - sequential: Chains models by step count
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
                "mode": (["single", "sequential"], {"default": "single"}),
                "model_steps": ("STRING", {"default": "", "multiline": False,
                    "tooltip": "Sequential mode: comma-separated steps per model (e.g., '7,7,6' for 20 steps). Empty = auto-divide."}),
                # Cascaded pipeline options (wire from NV_PreNoiseLatent outputs)
                "shift_override": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 20.0, "step": 0.5,
                    "tooltip": (
                        "Override model shift for sigma schedule. 0=use model default. "
                        "Wire from NV_PreNoiseLatent shift_used output to keep in sync. "
                        "LUVE recommends: Stage 1 shift=7-8, Stage 3 shift=3-4."
                    )}),
                "add_noise": (["enable", "disable"], {"default": "enable",
                    "tooltip": (
                        "Disable noise injection when latent is already pre-noised "
                        "(e.g., from NV_PreNoiseLatent in cascaded pipeline)."
                    )}),
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000,
                    "tooltip": (
                        "Skip to this step in the schedule (0=start from beginning). "
                        "Wire from NV_PreNoiseLatent start_at_step output for cascaded pipeline."
                    )}),
                "end_at_step": ("INT", {"default": 0, "min": 0, "max": 10000,
                    "tooltip": "Stop at this step (0=run to end). For cascaded: wire expanded_steps here."}),
                # Chunk consistency
                "committed_noise": ("COMMITTED_NOISE", {
                    "tooltip": "Pre-generated noise with FreeNoise correlation (from NV_CommittedNoise). If provided, seed is ignored."
                }),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "sample"
    CATEGORY = "NV_Utils/sampling"
    DESCRIPTION = "KSampler with multi-model and cascaded pipeline support. Use 'single' for standard sampling, 'sequential' to chain models by steps."

    def sample(self, model, positive, negative, latent_image, seed, steps, cfg,
               sampler_name, scheduler, denoise, model_2=None, model_3=None,
               sampler_name_override=None, scheduler_override=None,
               mode="single", model_steps="", shift_override=0.0,
               add_noise="enable", start_at_step=0, end_at_step=0,
               committed_noise=None):

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

        # ============= Shift Override (patches model_sampling sigma schedule) =============
        if shift_override > 0.0:
            model = self._apply_shift_override(model, shift_override)
            if model_2 is not None:
                model_2 = self._apply_shift_override(model_2, shift_override)
            if model_3 is not None:
                model_3 = self._apply_shift_override(model_3, shift_override)

        # Collect available models
        models = [model]
        if model_2 is not None:
            models.append(model_2)
        if model_3 is not None:
            models.append(model_3)

        # Resolve cascaded pipeline params
        disable_noise = (add_noise == "disable")
        start_step = start_at_step if start_at_step > 0 else None
        last_step = end_at_step if end_at_step > 0 else None

        if disable_noise or start_step is not None:
            print(f"[NV_MultiModelSampler] Cascaded mode: add_noise={add_noise}, "
                  f"start_at_step={start_step}, end_at_step={last_step}, "
                  f"shift_override={shift_override if shift_override > 0 else 'model default'}")

        # Route to appropriate sampling method
        if mode == "single" or len(models) == 1:
            return self._sample_single(
                model, positive, negative, latent_image,
                seed, steps, cfg, sampler_name, scheduler, denoise,
                committed_noise=committed_noise,
                disable_noise=disable_noise,
                start_step=start_step, last_step=last_step
            )
        elif mode == "sequential":
            return self._sample_sequential(
                models, positive, negative, latent_image,
                seed, steps, cfg, sampler_name, scheduler, denoise,
                model_steps, committed_noise=committed_noise,
                disable_noise=disable_noise,
                start_step=start_step, last_step=last_step
            )
        else:
            # Fallback to single
            return self._sample_single(
                model, positive, negative, latent_image,
                seed, steps, cfg, sampler_name, scheduler, denoise,
                committed_noise=committed_noise,
                disable_noise=disable_noise,
                start_step=start_step, last_step=last_step
            )

    def _sample_single(self, model, positive, negative, latent_image,
                       seed, steps, cfg, sampler_name, scheduler, denoise,
                       committed_noise=None, disable_noise=False,
                       start_step=None, last_step=None):
        """Standard KSampler behavior - with optional committed noise and cascaded pipeline support."""
        if committed_noise is not None and not disable_noise:
            return self._ksampler_with_noise(
                model, committed_noise["noise"], seed, steps, cfg, sampler_name, scheduler,
                positive, negative, latent_image, denoise=denoise,
                start_step=start_step, last_step=last_step
            )
        return common_ksampler(
            model, seed, steps, cfg, sampler_name, scheduler,
            positive, negative, latent_image, denoise=denoise,
            disable_noise=disable_noise,
            start_step=start_step, last_step=last_step
        )

    def _sample_sequential(self, models, positive, negative, latent_image,
                           seed, steps, cfg, sampler_name, scheduler,
                           denoise, model_steps_str, committed_noise=None,
                           disable_noise=False, start_step=None, last_step=None):
        """
        Chain models by step count.

        Each model processes a portion of the total steps, passing the latent
        to the next model. Later models use disable_noise=True to continue
        from the previous model's output.

        When start_step/last_step are provided (cascaded pipeline), the step
        distribution is applied within that sub-range.
        """
        # Determine effective step range
        effective_start = start_step if start_step is not None else 0
        effective_end = last_step if last_step is not None else steps
        effective_steps = effective_end - effective_start

        # Parse step distribution over the effective range
        step_distribution = self._parse_step_distribution(model_steps_str, effective_steps, len(models))

        print(f"[NV_MultiModelSampler] Sequential mode: {len(models)} models, "
              f"range [{effective_start}..{effective_end}]")
        print(f"[NV_MultiModelSampler] Step distribution: {step_distribution}")
        if committed_noise is not None:
            print(f"[NV_MultiModelSampler] Using committed noise (seed={committed_noise['seed']})")

        current_latent = latent_image
        current_step = effective_start

        # Get committed noise tensor if provided
        noise_tensor = committed_noise["noise"] if committed_noise is not None else None

        for i, (model, model_steps) in enumerate(zip(models, step_distribution)):
            if model_steps <= 0:
                continue

            seg_start = current_step
            seg_end = current_step + model_steps

            # First model: respect caller's disable_noise; subsequent always disable
            seg_disable_noise = disable_noise if i == 0 else True

            print(f"[NV_MultiModelSampler] Model {i+1}: steps {seg_start}-{seg_end}, cfg={cfg}, disable_noise={seg_disable_noise}")

            if noise_tensor is not None and not seg_disable_noise:
                result = self._ksampler_with_noise(
                    model, noise_tensor, seed, steps, cfg, sampler_name, scheduler,
                    positive, negative, current_latent,
                    denoise=denoise,
                    start_step=seg_start,
                    last_step=seg_end,
                    force_full_denoise=(i == len(models) - 1)
                )
            else:
                result = common_ksampler(
                    model, seed, steps, cfg, sampler_name, scheduler,
                    positive, negative, current_latent,
                    denoise=denoise,
                    disable_noise=seg_disable_noise,
                    start_step=seg_start,
                    last_step=seg_end,
                    force_full_denoise=(i == len(models) - 1)
                )

            current_latent = result[0]
            current_step = seg_end

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

        # Build clean output dict (defensive: drop stale temporal-dependent keys)
        out = {"samples": samples}
        for key in ("downscale_ratio_spacial", "latent_format_version_0"):
            if key in latent:
                out[key] = latent[key]
        return (out,)

    @staticmethod
    def _apply_shift_override(model, shift):
        """
        Clone model and patch its model_sampling with a new shift value.

        Uses the same pattern as ComfyUI's ModelSamplingFlux node:
        creates a new ModelSamplingAdvanced(ModelSamplingFlux, CONST) and
        calls set_parameters(shift=...).
        """
        m = model.clone()
        sampling_base = comfy.model_sampling.ModelSamplingFlux
        sampling_type = comfy.model_sampling.CONST

        class ModelSamplingAdvanced(sampling_base, sampling_type):
            pass

        model_sampling = ModelSamplingAdvanced(model.model.model_config)
        model_sampling.set_parameters(shift=shift)
        m.add_object_patch("model_sampling", model_sampling)
        old_shift = getattr(model.get_model_object("model_sampling"), "shift", "?")
        print(f"[NV_MultiModelSampler] Shift override: {old_shift} → {shift}")
        return m

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

        # Auto-divide: split steps as evenly as possible
        base_steps = total_steps // num_models
        remainder = total_steps % num_models
        distribution = []
        for i in range(num_models):
            extra = 1 if i < remainder else 0
            distribution.append(base_steps + extra)
        return distribution


# Node mappings for registration
NODE_CLASS_MAPPINGS = {
    "NV_MultiModelSampler": NV_MultiModelSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_MultiModelSampler": "NV Multi-Model Sampler",
}
