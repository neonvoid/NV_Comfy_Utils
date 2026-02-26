"""
NV Pre-Noise Latent — Cascaded Pipeline Noise Injection

Adds calibrated noise to a clean upscaled latent BEFORE chunk slicing,
so overlapping regions between chunks receive identical noise patterns.
This is the bridge between Stage 1 (low-res full pass) and Stage 3
(chunked high-res refinement) in the cascaded pipeline.

The noise level is computed from denoise/steps/shift using the same
schedule math as KSampler, and the node outputs the exact expanded_steps
and start_at_step values needed for KSamplerAdvanced(add_noise=disable).

Flow matching formula: noised = sigma * noise + (1 - sigma) * latent

References:
  - FlashVideo (2502.05179): Stage 2 starts from upscaled+noised low-res latent
  - LUVE (2602.11564): 3-stage cascaded latent pipeline
  - comfy/model_sampling.py: noise_scaling() for flow matching CONST class
  - comfy/samplers.py: set_steps() denoise truncation logic
"""

import json
import torch
import comfy.sample

from .chunk_utils import video_to_latent_frames
from .committed_noise import apply_freenoise_temporal_correlation
from .sigma_schedule_visualizer import compute_schedule


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
            },
            "optional": {
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

    RETURN_TYPES = ("LATENT", "INT", "INT", "FLOAT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("latent", "expanded_steps", "start_at_step",
                    "shift_used", "start_sigma", "signal_preserved_pct", "info")
    FUNCTION = "execute"
    CATEGORY = "NV_Utils/Chunked Pipeline"
    DESCRIPTION = (
        "Add calibrated noise to a clean upscaled latent for cascaded refinement. "
        "Ensures consistent noise across chunk boundaries. "
        "Wire expanded_steps, start_at_step, and shift_used to NV_MultiModelSampler."
    )

    def execute(self, latent, model, denoise, steps, seed,
                shift_override=0.0, enable_freenoise=True,
                context_length=81, context_overlap=16):
        samples = latent["samples"]  # [B, C, T, H, W]

        # --- 1. Compute sigma schedule (same math as KSampler) ---
        model_sampling = model.get_model_object("model_sampling")
        model_shift = getattr(model_sampling, "shift", 1.0)
        shift = shift_override if shift_override > 0.0 else model_shift

        expanded_steps = int(steps / denoise)
        start_at_step = expanded_steps - steps

        _, used_sigmas, _ = compute_schedule(shift, steps, denoise)
        start_sigma = used_sigmas[0]

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
        for key in ("downscale_ratio_spacial", "latent_format_version_0"):
            if key in latent:
                out[key] = latent[key]
        # Deliberately drop noise_mask, batch_index — stale after noise injection

        # --- 6. Compute diagnostics ---
        signal_preserved = (1.0 - start_sigma) * 100.0

        info = json.dumps({
            "shift": shift,
            "model_shift": model_shift,
            "shift_overridden": shift_override > 0.0,
            "denoise": denoise,
            "steps_requested": steps,
            "expanded_steps": expanded_steps,
            "start_at_step": start_at_step,
            "start_sigma": round(start_sigma, 6),
            "signal_preserved_pct": round(signal_preserved, 2),
            "seed": seed,
            "freenoise_applied": freenoise_applied,
            "latent_shape": list(samples.shape),
            "usage": (
                "Wire expanded_steps → NV_MultiModelSampler 'steps' (and 'end_at_step'), "
                "start_at_step → 'start_at_step', "
                "shift_used → 'shift_override', set add_noise=disable"
            ),
        }, indent=2)

        shift_src = f"{shift} (override)" if shift_override > 0.0 else f"{shift} (model)"
        print(f"[NV_PreNoiseLatent] shift={shift_src}, denoise={denoise}, "
              f"steps={steps}→expanded={expanded_steps}, "
              f"start_step={start_at_step}, sigma={start_sigma:.4f}, "
              f"signal_preserved={signal_preserved:.1f}%, "
              f"freenoise={'yes' if freenoise_applied else 'no'}")

        return (out, expanded_steps, start_at_step,
                shift, start_sigma, signal_preserved, info)


NODE_CLASS_MAPPINGS = {
    "NV_PreNoiseLatent": NV_PreNoiseLatent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_PreNoiseLatent": "NV Pre-Noise Latent (Cascaded)",
}
