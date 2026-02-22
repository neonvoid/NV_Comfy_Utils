"""
Context Window Multi-Stage Sampler Patch

Patches the context window handler's set_step() method to support samplers that
evaluate at intermediate sigma values (midpoints between scheduled steps).

Problem: ComfyUI's context_windows.py:set_step() uses torch.isclose() to match
the current timestep against sample_sigmas. Multi-stage samplers (res_2s, res_3s,
dpm_2, dpmpp_2s_ancestral, dpmpp_sde, seeds_2, seeds_3) evaluate at intermediate
sigmas not in the schedule, causing set_step() to crash with:
    "No sample_sigmas matched current timestep; something went wrong."

Fix: Instance-patch set_step() to fall back to the nearest sigma in the schedule
when no exact match is found. The intermediate evaluation uses the same context
window arrangement as the nearest scheduled step, which is correct since the
midpoint is between the current and next step.

This is an instance patch (not a class patch), so it only affects the specific
model clone for the current sampling run. No core ComfyUI files are modified.

Usage:
    [WAN Context Windows] -> [NV Context Window Sampler Patch] -> [KSampler]
"""

import torch


class NV_ContextWindowSamplerPatch:
    """Patches context window set_step() to support multi-stage samplers
    (res_2s, res_3s, dpm_2, dpmpp_2s, seeds_2, etc.) that evaluate at
    intermediate sigma values not in the schedule."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "NV_Utils/context_windows"
    DESCRIPTION = (
        "Patches context windows to work with multi-stage samplers "
        "(res_2s, res_3s, dpm_2, dpmpp_2s, seeds_2, etc.) that evaluate "
        "at intermediate sigma values. Place after WAN Context Windows, "
        "before KSampler."
    )

    def patch(self, model):
        model = model.clone()

        context_handler = model.model_options.get("context_handler", None)

        if context_handler is None:
            print("[NV_ContextWindowSamplerPatch] Warning: No context handler found on model.")
            print("  Make sure to connect WAN Context Windows node before this patcher.")
            return (model,)

        def patched_set_step(timestep, model_options):
            sample_sigmas = model_options["transformer_options"]["sample_sigmas"]
            mask = torch.isclose(sample_sigmas, timestep[0], rtol=0.0001)
            matches = torch.nonzero(mask)
            if matches.numel() == 0:
                # Nearest-sigma fallback for intermediate evaluations
                diffs = (sample_sigmas - timestep[0]).abs()
                context_handler._step = int(diffs.argmin().item())
            else:
                context_handler._step = int(matches[0].item())

        context_handler.set_step = patched_set_step
        print("[NV_ContextWindowSamplerPatch] Patched set_step() for multi-stage sampler support")

        return (model,)


NODE_CLASS_MAPPINGS = {
    "NV_ContextWindowSamplerPatch": NV_ContextWindowSamplerPatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_ContextWindowSamplerPatch": "NV Context Window Sampler Patch",
}
