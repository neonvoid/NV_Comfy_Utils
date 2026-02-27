"""
NV Pre-Noise Latent — Cascaded Pipeline Noise Injection

Adds calibrated noise to a clean upscaled latent BEFORE chunk slicing,
so overlapping regions between chunks receive identical noise patterns.
This is the bridge between Stage 1 (low-res full pass) and Stage 3
(chunked high-res refinement) in the cascaded pipeline.

The sigma schedule is computed using ComfyUI's actual calculate_sigmas()
function with the user-selected scheduler, ensuring the noise level
EXACTLY matches what KSampler will use at the same step index.

Flow matching formula: noised = sigma * noise + (1 - sigma) * latent

References:
  - FlashVideo (2502.05179): Stage 2 starts from upscaled+noised low-res latent
  - LUVE (2602.11564): 3-stage cascaded latent pipeline
  - comfy/model_sampling.py: noise_scaling() for flow matching CONST class
  - comfy/samplers.py: calculate_sigmas(), KSampler.set_steps()
"""

import json
import torch
import comfy.sample
import comfy.samplers
import comfy.model_sampling

from .chunk_utils import video_to_latent_frames
from .committed_noise import apply_freenoise_temporal_correlation
from .latent_constants import NV_CASCADED_CONFIG_KEY, LATENT_SAFE_KEYS


class NV_PreNoiseLatent:
    """
    Add calibrated noise to a clean latent for cascaded refinement.

    Designed for the cascaded chunked pipeline:
      Stage 1 low-res KSampler → LatentUpscale → [THIS NODE] → ChunkLoader → KSamplerAdvanced

    The noise is applied ONCE to the full latent before chunk slicing, ensuring
    overlapping regions between chunks share identical noise. This dramatically
    reduces boundary seams compared to per-chunk independent noise.

    Outputs expanded_steps and start_at_step to plug directly into KSamplerAdvanced
    with add_noise=disable.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {
                    "tooltip": "Clean upscaled latent from Stage 1 (low-res full pass)"
                }),
                "model": ("MODEL", {
                    "tooltip": "Model for noise scaling formula and shift detection"
                }),
                "denoise": ("FLOAT", {
                    "default": 0.15, "min": 0.05, "max": 0.95, "step": 0.05,
                    "tooltip": (
                        "Noise amount. At WAN shift=8: "
                        "0.10→51% signal, 0.15→42% signal, 0.30→23% signal. "
                        "Lower shift (3-4) at Stage 3 preserves more. "
                        "Start at 0.15, increase if output is too blurry"
                    )
                }),
                "steps": ("INT", {
                    "default": 6, "min": 1, "max": 100,
                    "tooltip": "Refinement steps for chunked Stage 3 (fewer = faster)"
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "control_after_generate": True,
                    "tooltip": "Noise seed for reproducibility"
                }),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {
                    "default": "normal",
                    "tooltip": (
                        "MUST match the scheduler used by the Stage 3 KSampler. "
                        "Different schedulers produce different sigma values at the same step, "
                        "causing noise level mismatch if they don't agree."
                    )
                }),
            },
            "optional": {
                "plan_json_path": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Path to chunk plan JSON from NV_ParallelChunkPlanner. "
                        "If provided, injects cascaded_config (shift, steps, start_at_step) "
                        "into the plan for downstream nodes to read automatically."
                    )
                }),
                "shift_override": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 20.0, "step": 0.5,
                    "tooltip": (
                        "Override model shift for sigma computation. 0=use model default. "
                        "MUST match the shift used by Stage 3 KSampler. "
                        "LUVE recommends: Stage 1 shift=7-8, Stage 3 shift=3-4. "
                        "At shift=4, denoise=0.15→27% signal vs shift=8→42% signal"
                    )
                }),
                "enable_freenoise": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply FreeNoise temporal correlation for chunk consistency"
                }),
                "context_length": ("INT", {
                    "default": 81, "min": 5, "max": 513,
                    "tooltip": "Context window size in VIDEO frames (for FreeNoise)"
                }),
                "context_overlap": ("INT", {
                    "default": 16, "min": 1, "max": 128,
                    "tooltip": "Context overlap in VIDEO frames (for FreeNoise)"
                }),
            }
        }

    RETURN_TYPES = ("LATENT", "INT", "INT", "FLOAT", "FLOAT", "FLOAT", "STRING", "STRING")
    RETURN_NAMES = ("latent", "expanded_steps", "start_at_step",
                    "shift_used", "start_sigma", "signal_preserved_pct", "info",
                    "plan_json_path")
    FUNCTION = "execute"
    CATEGORY = "NV_Utils/Chunked Pipeline"
    DESCRIPTION = (
        "Add calibrated noise to a clean upscaled latent for cascaded refinement. "
        "Ensures consistent noise across chunk boundaries. "
        "If plan_json_path is provided, injects cascaded_config into the plan "
        "so NV_ChunkLoaderVACE can output shift/steps/start_at_step automatically."
    )

    def execute(self, latent, model, denoise, steps, seed, scheduler="normal",
                plan_json_path="", shift_override=0.0, enable_freenoise=True,
                context_length=81, context_overlap=16):
        samples = latent["samples"]  # [B, C, T, H, W]

        # --- 1. Compute sigma schedule using ComfyUI's actual calculate_sigmas ---
        # This MUST match the KSampler's schedule exactly. Using the same function
        # guarantees consistency regardless of scheduler type (normal, simple, beta, etc.).
        model_sampling = model.get_model_object("model_sampling")
        model_shift = getattr(model_sampling, "shift", 1.0)
        shift = shift_override if shift_override > 0.0 else model_shift

        expanded_steps = int(steps / denoise)
        start_at_step = expanded_steps - steps

        # Build model_sampling with the correct shift for sigma computation.
        # CRITICAL: Must use the model's ACTUAL model_sampling class, not a hardcoded
        # one. Different classes use different sigma functions:
        #   - ModelSamplingDiscreteFlow (WAN): time_snr_shift (Möbius transform)
        #   - ModelSamplingFlux (Flux):        flux_time_shift (exponential)
        # Using the wrong class produces completely wrong sigmas.
        # Example: shift=4, t=0.303 → DiscreteFlow gives σ=0.635, Flux gives σ=0.96!
        if shift_override > 0.0 and abs(shift_override - model_shift) > 1e-4:
            ms_cls = type(model_sampling)
            sigma_model_sampling = ms_cls(model.model.model_config)
            kwargs = {"shift": shift}
            if hasattr(model_sampling, "multiplier"):
                kwargs["multiplier"] = model_sampling.multiplier
            sigma_model_sampling.set_parameters(**kwargs)
            print(f"[NV_PreNoiseLatent] Created {ms_cls.__name__} with shift={shift} "
                  f"(model has {model_shift})")
        else:
            sigma_model_sampling = model_sampling
            if shift_override > 0.0:
                print(f"[NV_PreNoiseLatent] shift_override={shift_override} matches model "
                      f"shift={model_shift}, using model's own sampling")

        # Compute the FULL expanded schedule (same as KSampler with denoise=1.0)
        full_sigmas = comfy.samplers.calculate_sigmas(sigma_model_sampling, scheduler, expanded_steps)
        # The sigma at start_at_step is what the KSampler will see
        start_sigma = float(full_sigmas[start_at_step])

        # --- 2. Generate noise for the FULL latent ---
        batch_inds = latent.get("batch_index", None)
        noise = comfy.sample.prepare_noise(samples, seed, batch_inds)

        # --- 3. Optional FreeNoise temporal correlation ---
        total_latent_frames = samples.shape[2] if samples.ndim >= 5 else 1
        context_length_latent = video_to_latent_frames(context_length)
        freenoise_applied = False

        if enable_freenoise and total_latent_frames > context_length_latent:
            context_overlap_latent = video_to_latent_frames(context_overlap)
            noise = apply_freenoise_temporal_correlation(
                noise, seed,
                context_length=context_length_latent,
                context_overlap=context_overlap_latent,
                temporal_dim=2
            )
            freenoise_applied = True

        # --- 4. Apply noise using model's formula ---
        # Flow matching CONST: noised = sigma * noise + (1 - sigma) * latent
        process_latent_in = model.get_model_object("process_latent_in")
        process_latent_out = model.get_model_object("process_latent_out")

        samples_in = process_latent_in(samples)
        sigma_tensor = torch.tensor([start_sigma], dtype=samples_in.dtype, device=samples_in.device)
        noised = model_sampling.noise_scaling(sigma_tensor, noise.to(samples_in.device), samples_in)
        noised = process_latent_out(noised)
        noised = torch.nan_to_num(noised, nan=0.0, posinf=0.0, neginf=0.0)

        # --- 5. Build clean output dict (drop stale temporal keys) ---
        out = {"samples": noised}
        for key in LATENT_SAFE_KEYS:
            if key in latent and key != NV_CASCADED_CONFIG_KEY:
                out[key] = latent[key]
        # Deliberately drop noise_mask, batch_index — stale after noise injection

        # --- 6. Compute diagnostics + embed config in latent ---
        signal_preserved = (1.0 - start_sigma) * 100.0

        cascaded_config = {
            "shift_override": shift,
            "expanded_steps": expanded_steps,
            "start_at_step": start_at_step,
            "add_noise": "disable",
            "start_sigma": round(start_sigma, 6),
            "signal_preserved_pct": round(signal_preserved, 2),
            "prenoise_denoise": denoise,
            "prenoise_steps": steps,
            "prenoise_seed": seed,
            "scheduler": scheduler,
            "freenoise_applied": freenoise_applied,
        }

        # Embed config IN the latent dict — travels with data through save/load/slice
        out[NV_CASCADED_CONFIG_KEY] = cascaded_config

        info = json.dumps({
            **cascaded_config,
            "model_shift": model_shift,
            "shift_overridden": shift_override > 0.0,
            "latent_shape": list(samples.shape),
            "usage": (
                "Wire expanded_steps → NV_MultiModelSampler 'steps' (and 'end_at_step'), "
                "start_at_step → 'start_at_step', "
                "shift_used → 'shift_override', set add_noise=disable"
            ),
        }, indent=2)

        # --- 7. Inject cascaded_config into plan JSON if provided ---
        plan_out_path = plan_json_path
        if plan_json_path and plan_json_path.strip():
            try:
                with open(plan_json_path, 'r') as f:
                    plan = json.load(f)
                plan["cascaded_config"] = cascaded_config
                with open(plan_json_path, 'w') as f:
                    json.dump(plan, f, indent=2)
                print(f"[NV_PreNoiseLatent] Injected cascaded_config into {plan_json_path}")
            except FileNotFoundError:
                print(f"[NV_PreNoiseLatent] Warning: plan not found at {plan_json_path}, "
                      f"skipping injection. Wire outputs manually.")
            except (json.JSONDecodeError, OSError) as e:
                print(f"[NV_PreNoiseLatent] Warning: could not update plan: {e}")

        shift_src = f"{shift} (override)" if shift_override > 0.0 else f"{shift} (model)"
        print(f"[NV_PreNoiseLatent] shift={shift_src}, scheduler={scheduler}, "
              f"denoise={denoise}, steps={steps}→expanded={expanded_steps}, "
              f"start_step={start_at_step}, sigma={start_sigma:.4f}, "
              f"signal_preserved={signal_preserved:.1f}%, "
              f"freenoise={'yes' if freenoise_applied else 'no'}")

        return (out, expanded_steps, start_at_step,
                shift, start_sigma, signal_preserved, info, plan_out_path)


NODE_CLASS_MAPPINGS = {
    "NV_PreNoiseLatent": NV_PreNoiseLatent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_PreNoiseLatent": "NV Pre-Noise Latent (Cascaded)",
}
