"""
NV Co-Denoise Chunk Sampler - Phase 1 Prototype

Per-step co-denoising for parallel video chunks with overlap blending.
Instead of sampling chunks independently (causing cross-chunk drift),
processes all chunks one step at a time and blends overlap regions between
steps -- replicating at the chunk level what native context windows do
within a single chunk.

Euler-only. Single model. Supports denoise masks (VACE) and context windows.
"""

import torch
import math
import comfy.samplers
import comfy.sampler_helpers
import comfy.model_patcher
import comfy.patcher_extension
import comfy.hooks
import comfy.sample
import comfy.utils
import comfy.model_management
import latent_preview


# ============================================================================
# Blend helpers
# ============================================================================

def _hann_weights(length, device=None):
    if length <= 1:
        return torch.ones(max(1, length), device=device)
    t = torch.linspace(0, 1, length, device=device)
    return 0.5 * (1.0 - torch.cos(math.pi * t))


def _linear_weights(length, device=None):
    if length <= 1:
        return torch.ones(max(1, length), device=device)
    return torch.linspace(0, 1, length, device=device)


_BLEND_FNS = {
    "hann": _hann_weights,
    "cosine": _hann_weights,
    "linear": _linear_weights,
}


def _blend_overlaps(chunks_x, overlap, blend_mode, tdim=2):
    """Blend overlap regions between adjacent chunks in-place."""
    if overlap <= 0 or len(chunks_x) < 2:
        return

    wfn = _BLEND_FNS.get(blend_mode, _hann_weights)

    for i in range(len(chunks_x) - 1):
        t_a = chunks_x[i].shape[tdim]
        t_b = chunks_x[i + 1].shape[tdim]
        eff = min(overlap, t_a, t_b)
        if eff <= 0:
            continue

        w = wfn(eff, chunks_x[0].device)
        shape = [1] * chunks_x[0].ndim
        shape[tdim] = eff
        w = w.view(shape)

        sl_a = [slice(None)] * chunks_x[i].ndim
        sl_a[tdim] = slice(-eff, None)
        sl_b = [slice(None)] * chunks_x[i + 1].ndim
        sl_b[tdim] = slice(None, eff)

        region_a = chunks_x[i][tuple(sl_a)]
        region_b = chunks_x[i + 1][tuple(sl_b)]
        blended = (1.0 - w) * region_a + w * region_b

        chunks_x[i][tuple(sl_a)] = blended
        chunks_x[i + 1][tuple(sl_b)] = blended


# ============================================================================
# Node
# ============================================================================

class NV_CoDenoiseChunkSampler:
    """
    Per-step co-denoising sampler for chunked video.
    Euler-only prototype. Supports denoise masks (VACE) and context windows.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "overlap_latent_frames": ("INT", {
                    "default": 4, "min": 0, "max": 64,
                    "tooltip": "Overlapping LATENT frames between adjacent chunks. "
                               "For Wan 4:1 compression: 4 latent = ~13 video frames."
                }),
                "blend_mode": (["hann", "linear", "cosine"], {"default": "hann"}),
                # Chunk 0 (required)
                "positive_0": ("CONDITIONING",),
                "negative_0": ("CONDITIONING",),
                "latent_0": ("LATENT",),
            },
            "optional": {
                "noise_0": ("LATENT", {"tooltip": "Committed noise for chunk 0"}),
                "positive_1": ("CONDITIONING",),
                "negative_1": ("CONDITIONING",),
                "latent_1": ("LATENT",),
                "noise_1": ("LATENT",),
                "positive_2": ("CONDITIONING",),
                "negative_2": ("CONDITIONING",),
                "latent_2": ("LATENT",),
                "noise_2": ("LATENT",),
                "positive_3": ("CONDITIONING",),
                "negative_3": ("CONDITIONING",),
                "latent_3": ("LATENT",),
                "noise_3": ("LATENT",),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT", "LATENT", "LATENT")
    RETURN_NAMES = ("latent_0", "latent_1", "latent_2", "latent_3")
    FUNCTION = "sample"
    CATEGORY = "NV_Utils/sampling"
    DESCRIPTION = (
        "Co-denoises overlapping video chunks one step at a time with "
        "overlap blending. Prevents cross-chunk drift. Euler-only prototype."
    )

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, denoise,
               overlap_latent_frames, blend_mode,
               positive_0, negative_0, latent_0,
               noise_0=None, positive_1=None, negative_1=None, latent_1=None, noise_1=None,
               positive_2=None, negative_2=None, latent_2=None, noise_2=None,
               positive_3=None, negative_3=None, latent_3=None, noise_3=None):

        # -- Collect chunks --
        raw = []
        for pos, neg, lat, noi in [
            (positive_0, negative_0, latent_0, noise_0),
            (positive_1, negative_1, latent_1, noise_1),
            (positive_2, negative_2, latent_2, noise_2),
            (positive_3, negative_3, latent_3, noise_3),
        ]:
            if pos is not None and neg is not None and lat is not None:
                raw.append((pos, neg, lat, noi))

        N = len(raw)
        if N == 0:
            raise ValueError("At least chunk_0 is required.")

        print(f"[CoDenoiseChunkSampler] {N} chunks, {steps} steps, "
              f"overlap={overlap_latent_frames} latent frames, blend={blend_mode}")

        # -- Sigmas --
        device = model.load_device
        ks_temp = comfy.samplers.KSampler(
            model, steps=steps, device=device,
            sampler=sampler_name, scheduler=scheduler, denoise=denoise
        )
        sigmas = ks_temp.sigmas

        if sigmas.shape[-1] == 0:
            return self._pass_through(raw, N)

        # -- Per-chunk data --
        latent_images = []
        noises = []
        denoise_masks = []
        chunk_conds = []

        for i, (pos, neg, lat, noi) in enumerate(raw):
            li = lat["samples"]
            li = comfy.sample.fix_empty_latent_channels(
                model, li, lat.get("downscale_ratio_spacial", None)
            )
            noise = noi["samples"] if noi is not None else comfy.sample.prepare_noise(li, seed + i)
            dm = lat.get("noise_mask", None)
            conds = {
                "positive": comfy.sampler_helpers.convert_cond(pos),
                "negative": comfy.sampler_helpers.convert_cond(neg),
            }
            latent_images.append(li)
            noises.append(noise)
            denoise_masks.append(dm)
            chunk_conds.append(conds)

        # -- Preprocess hooks --
        for conds in chunk_conds:
            comfy.samplers.preprocess_conds_hooks(conds)

        # Merge conds so hooks from ALL chunks get registered
        merged = {"positive": [], "negative": []}
        for conds in chunk_conds:
            for k in merged:
                merged[k].extend(conds[k])

        # -- Model setup (mirrors CFGGuider.sample + outer_sample) --
        guider = comfy.samplers.CFGGuider(model)
        guider.set_cfg(cfg)

        orig_model_options = guider.model_options
        guider.model_options = comfy.model_patcher.create_model_options_clone(orig_model_options)

        orig_hook_mode = guider.model_patcher.hook_mode
        if comfy.samplers.get_total_hook_groups_in_conds(merged) <= 1:
            guider.model_patcher.hook_mode = comfy.hooks.EnumHookMode.MinVram

        comfy.sampler_helpers.prepare_model_patcher(
            guider.model_patcher, merged, guider.model_options
        )
        for conds in chunk_conds:
            comfy.samplers.filter_registered_hooks_on_conds(conds, guider.model_options)

        # Load model to GPU
        real_model, _, loaded_models = comfy.sampler_helpers.prepare_sampling(
            guider.model_patcher, noises[0].shape, merged, guider.model_options
        )
        guider.inner_model = real_model

        sigmas = sigmas.to(device)
        comfy.samplers.cast_to_load_options(
            guider.model_options, device=device, dtype=guider.model_patcher.model_dtype()
        )

        try:
            guider.model_patcher.pre_run()

            # -- Per-chunk: move to device, process_latent_in, process_conds --
            for i in range(N):
                latent_images[i] = latent_images[i].to(device)
                noises[i] = noises[i].to(device)
                if denoise_masks[i] is not None:
                    denoise_masks[i] = comfy.sampler_helpers.prepare_mask(
                        denoise_masks[i], latent_images[i].shape, device
                    )
                if torch.count_nonzero(latent_images[i]) > 0:
                    latent_images[i] = real_model.process_latent_in(latent_images[i])

                chunk_conds[i] = comfy.samplers.process_conds(
                    real_model, noises[i], chunk_conds[i], device,
                    latent_images[i], denoise_masks[i], seed,
                    latent_shapes=[latent_images[i].shape]
                )

            # -- sample_sigmas for context window compatibility --
            extra_opts = comfy.model_patcher.create_model_options_clone(guider.model_options)
            extra_opts.setdefault("transformer_options", {})["sample_sigmas"] = sigmas

            # -- Initialize x per chunk (noise scaling) --
            ms = real_model.model_sampling
            is_max_denoise = (
                math.isclose(float(ms.sigma_max), float(sigmas[0]), rel_tol=1e-05)
                or float(sigmas[0]) > float(ms.sigma_max)
            )
            chunks_x = []
            for i in range(N):
                x = ms.noise_scaling(sigmas[0], noises[i], latent_images[i], is_max_denoise)
                chunks_x.append(x)

            # -- Euler co-denoise loop --
            results = self._euler_co_denoise(
                guider, real_model, chunks_x, chunk_conds, noises,
                latent_images, denoise_masks, sigmas, extra_opts,
                overlap_latent_frames, blend_mode, seed, model
            )

            # -- Finalize outputs --
            outputs = []
            for i in range(N):
                samples = ms.inverse_noise_scaling(sigmas[-1], results[i])
                samples = real_model.process_latent_out(samples.to(torch.float32))
                samples = samples.to(comfy.model_management.intermediate_device())
                out = raw[i][2].copy()
                out["samples"] = samples
                out.pop("noise_mask", None)
                out.pop("downscale_ratio_spacial", None)
                outputs.append(out)

        finally:
            guider.model_patcher.cleanup()
            comfy.sampler_helpers.cleanup_models(merged, loaded_models)
            comfy.samplers.cast_to_load_options(
                guider.model_options, device=guider.model_patcher.offload_device
            )
            guider.model_options = orig_model_options
            guider.model_patcher.hook_mode = orig_hook_mode
            guider.model_patcher.restore_hook_patches()

        while len(outputs) < 4:
            outputs.append({"samples": torch.zeros(1)})

        print(f"[CoDenoiseChunkSampler] Done. {N} chunks processed.")
        return tuple(outputs)

    def _euler_co_denoise(self, guider, real_model, chunks_x, chunk_conds,
                          noises, latent_images, denoise_masks, sigmas,
                          extra_opts, overlap, blend_mode, seed, model_patcher):
        """Core Euler step loop with per-step overlap blending."""
        N = len(chunks_x)
        total_steps = len(sigmas) - 1
        callback = latent_preview.prepare_callback(model_patcher, total_steps)

        print(f"[CoDenoiseChunkSampler] Starting euler co-denoise: "
              f"{total_steps} steps x {N} chunks")

        for step in range(total_steps):
            sigma = sigmas[step]
            sigma_next = sigmas[step + 1]
            dt = sigma_next - sigma
            s_in = chunks_x[0].new_ones([chunks_x[0].shape[0]])

            comfy.model_management.throw_exception_if_processing_interrupted()

            for ci in range(N):
                # Swap conditioning for this chunk
                guider.conds = chunk_conds[ci]

                x = chunks_x[ci]

                # Denoise mask handling (replicates KSamplerX0Inpaint)
                dm = denoise_masks[ci]
                if dm is not None:
                    dm_step = dm
                    if "denoise_mask_function" in extra_opts:
                        dm_step = extra_opts["denoise_mask_function"](
                            sigma, dm_step,
                            extra_options={"model": guider, "sigmas": sigmas}
                        )
                    latent_mask = 1.0 - dm_step
                    if hasattr(real_model, 'scale_latent_inpaint'):
                        inpaint_fill = real_model.scale_latent_inpaint(
                            x=x, sigma=sigma, noise=noises[ci],
                            latent_image=latent_images[ci]
                        )
                    else:
                        inpaint_fill = sigma * noises[ci] + latent_images[ci]
                    x_in = x * dm_step + inpaint_fill * latent_mask
                else:
                    x_in = x

                # Forward pass through guider -> sampling_function ->
                # calc_cond_batch -> context windows
                denoised = guider(
                    x_in, sigma * s_in, model_options=extra_opts, seed=seed
                )

                # Post-mask: preserve reference regions
                if dm is not None:
                    denoised = denoised * dm_step + latent_images[ci] * latent_mask

                # Euler step: d = (x - denoised) / sigma; x_new = x + d * dt
                d = (x - denoised) / sigma
                chunks_x[ci] = x + d * dt

            # Blend overlaps between adjacent chunks
            _blend_overlaps(chunks_x, overlap, blend_mode)

            if callback is not None:
                callback(step, chunks_x[0], chunks_x[0], total_steps)

        return chunks_x

    def _pass_through(self, raw, N):
        results = [r[2] for r in raw]
        while len(results) < 4:
            results.append({"samples": torch.zeros(1)})
        return tuple(results)


# Node registration
NODE_CLASS_MAPPINGS = {
    "NV_CoDenoiseChunkSampler": NV_CoDenoiseChunkSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_CoDenoiseChunkSampler": "NV Co-Denoise Chunk Sampler",
}
