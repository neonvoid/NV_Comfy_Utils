"""
DEPRECATED: NV Co-Denoise Chunk Sampler

This node has been deprecated. It hardcodes Euler stepping regardless of the
sampler_name input, and has not produced competitive results compared to the
latent-space chunk stitching pipeline.

Use instead:
  [NV_SaveChunkLatent] -> [NV_LatentChunkStitcher] -> [NV_BoundaryNoiseMask]
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

from .chunk_utils import video_to_latent_frames, compute_blend_weights


# ============================================================================
# Blend helpers (kept for internal use, delegates to chunk_utils)
# ============================================================================

def _hann_weights(length, device=None):
    return compute_blend_weights(length, "hann", device)


def _linear_weights(length, device=None):
    return compute_blend_weights(length, "linear", device)


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
# Co-Denoise Chunk Sampler
# ============================================================================

class NV_CoDenoiseChunkSampler:
    """
    Per-step co-denoising sampler for chunked video.

    Accepts full-length conditioning + latent from a single WanVaceToVideo
    call. Internally computes chunk boundaries, slices VACE conditioning
    per chunk, runs per-step Euler co-denoising with overlap blending,
    and outputs a single merged full-length latent.

    Supports multi-model sequential/boundary sampling, committed noise,
    and automatic per-chunk RoPE shift_t.
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
                "chunk_length_latent": ("INT", {
                    "default": 21, "min": 1, "max": 256,
                    "tooltip": "Latent frames per chunk. "
                               "21 latent = 81 video frames (Wan default context window)."
                }),
                "overlap_latent_frames": ("INT", {
                    "default": 4, "min": 0, "max": 64,
                    "tooltip": "Overlapping latent frames between adjacent chunks. "
                               "4 latent = ~13 video frames."
                }),
                "blend_mode": (["hann", "linear", "cosine"], {"default": "hann"}),
            },
            "optional": {
                "model_2": ("MODEL",),
                "model_3": ("MODEL",),
                "mode": (["single", "sequential", "boundary"], {
                    "default": "single",
                    "tooltip": "single: one model. sequential: chain by step count. "
                               "boundary: switch by sigma threshold."
                }),
                "model_steps": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "Sequential mode: comma-separated steps per model "
                               "(e.g., '7,7,6'). Empty = auto-divide."
                }),
                "model_boundaries": ("STRING", {
                    "default": "0.875,0.5", "multiline": False,
                    "tooltip": "Boundary mode: sigma thresholds (0-1). "
                               "Model switches when sigma drops below."
                }),
                "model_cfg_scales": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "Per-model CFG overrides (e.g., '7.0,5.0,3.0'). "
                               "Empty = use main cfg."
                }),
                "committed_noise": ("COMMITTED_NOISE", {
                    "tooltip": "Full-length pre-generated noise. Sliced per chunk "
                               "internally. Ensures identical noise in overlap regions."
                }),
                "enable_shift_t": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Auto-apply RoPE temporal offset per chunk based on "
                               "its position in the full video."
                }),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "sample"
    CATEGORY = "NV_Utils/sampling"
    DESCRIPTION = (
        "Co-denoises overlapping video chunks one step at a time with "
        "per-step overlap blending. Accepts full-length conditioning from "
        "a single WanVaceToVideo -- no per-chunk VACE nodes needed. "
        "Supports multi-model sequential/boundary sampling."
    )

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def sample(self, model, positive, negative, latent_image, seed, steps, cfg,
               sampler_name, scheduler, denoise, chunk_length_latent,
               overlap_latent_frames, blend_mode,
               model_2=None, model_3=None, mode="single",
               model_steps="", model_boundaries="0.875,0.5",
               model_cfg_scales="", committed_noise=None,
               enable_shift_t=False):

        # -- Extract full latent --
        full_samples = latent_image["samples"]
        full_samples = comfy.sample.fix_empty_latent_channels(
            model, full_samples, latent_image.get("downscale_ratio_spacial", None)
        )
        T_total = full_samples.shape[2]
        full_noise_mask = latent_image.get("noise_mask", None)

        # -- Compute chunk boundaries --
        boundaries = self._compute_chunk_boundaries(
            T_total, chunk_length_latent, overlap_latent_frames
        )
        N = len(boundaries)

        # -- Generate or accept noise --
        if committed_noise is not None:
            full_noise = committed_noise["noise"]
            if full_noise.shape != full_samples.shape:
                full_noise = comfy.sample.prepare_noise(full_samples, seed)
                print("[CoDenoiseChunkSampler] Warning: committed_noise shape "
                      "mismatch, falling back to seed-based noise.")
        else:
            full_noise = comfy.sample.prepare_noise(full_samples, seed)

        # -- Collect models --
        models = [model]
        if model_2 is not None:
            models.append(model_2)
        if model_3 is not None:
            models.append(model_3)

        cfg_scales = self._parse_cfg_scales(model_cfg_scales, cfg, len(models))

        # -- Compute sigmas --
        device = model.load_device
        ks_temp = comfy.samplers.KSampler(
            model, steps=steps, device=device,
            sampler=sampler_name, scheduler=scheduler, denoise=denoise
        )
        sigmas = ks_temp.sigmas
        total_steps = len(sigmas) - 1

        if total_steps == 0:
            return (latent_image,)

        # -- Compute model step groups --
        step_groups = self._compute_step_groups(
            mode, len(models), total_steps, model_steps, model_boundaries
        )

        print(f"[CoDenoiseChunkSampler] {N} chunks from {T_total} latent frames "
              f"(chunk_len={chunk_length_latent}, overlap={overlap_latent_frames})")
        for i, (s, e) in enumerate(boundaries):
            print(f"  Chunk {i}: latent frames [{s}:{e}] ({e - s} frames)")
        print(f"  {total_steps} steps, blend={blend_mode}")
        for mi, s, e in step_groups:
            print(f"  Model {mi + 1}: steps {s}-{e}, cfg={cfg_scales[mi]}")

        # -- Slice per-chunk data --
        latent_images = []
        noises = []
        denoise_masks = []
        chunk_shift_t = []
        chunk_conds = []

        for i, (start, end) in enumerate(boundaries):
            # Slice latent
            li = full_samples[:, :, start:end, :, :].clone()

            # Slice noise (from same full-length source = identical overlap noise)
            noise = full_noise[:, :, start:end, :, :].clone()

            # Slice noise_mask
            dm = self._slice_noise_mask(full_noise_mask, start, end, T_total)

            # Slice conditioning (VACE frames + mask along temporal dim)
            chunk_pos = self._slice_cond_temporal(positive, start, end)
            chunk_neg = self._slice_cond_temporal(negative, start, end)

            # Auto shift_t from chunk position
            shift_t = start if enable_shift_t else 0

            # Convert conditioning to internal format
            conds = {
                "positive": comfy.sampler_helpers.convert_cond(chunk_pos),
                "negative": comfy.sampler_helpers.convert_cond(chunk_neg),
            }

            latent_images.append(li)
            noises.append(noise)
            denoise_masks.append(dm)
            chunk_shift_t.append(shift_t)
            chunk_conds.append(conds)

        # -- Preprocess hooks --
        for conds in chunk_conds:
            comfy.samplers.preprocess_conds_hooks(conds)

        # Merge conds for model setup (hooks from ALL chunks)
        merged = {"positive": [], "negative": []}
        for conds in chunk_conds:
            for k in merged:
                merged[k].extend(conds[k])

        # Save original hooks for re-filtering across model groups
        _saved_hooks = []
        for conds in chunk_conds:
            saved = {}
            for k in conds:
                saved[k] = [c.get("hooks", None) for c in conds[k]]
            _saved_hooks.append(saved)

        # -- Process model groups --
        sigmas = sigmas.to(device)
        chunks_x = None
        model_sampling = None
        last_real_model = None
        callback = latent_preview.prepare_callback(model, total_steps)

        for group_idx, (model_idx, start_step, end_step) in enumerate(step_groups):
            if start_step >= end_step:
                continue

            current_model = models[model_idx]
            current_cfg = cfg_scales[model_idx]

            # Restore original hooks before filtering for this model group
            if group_idx > 0:
                for ci, conds in enumerate(chunk_conds):
                    for k in conds:
                        for j, c in enumerate(conds[k]):
                            orig_hook = _saved_hooks[ci][k][j]
                            if orig_hook is not None:
                                c["hooks"] = orig_hook

            # -- Setup guider for this model --
            guider, real_model, loaded_models, orig_opts, orig_hook_mode = \
                self._setup_model(current_model, current_cfg, merged,
                                  chunk_conds, noises[0].shape, device)

            try:
                guider.model_patcher.pre_run()

                # First model group: initialize chunk data
                if chunks_x is None:
                    model_sampling = real_model.model_sampling
                    last_real_model = real_model

                    for i in range(N):
                        latent_images[i] = latent_images[i].to(device)
                        noises[i] = noises[i].to(device)
                        if denoise_masks[i] is not None:
                            denoise_masks[i] = comfy.sampler_helpers.prepare_mask(
                                denoise_masks[i], latent_images[i].shape, device
                            )
                        if torch.count_nonzero(latent_images[i]) > 0:
                            latent_images[i] = real_model.process_latent_in(
                                latent_images[i]
                            )

                        chunk_conds[i] = comfy.samplers.process_conds(
                            real_model, noises[i], chunk_conds[i], device,
                            latent_images[i], denoise_masks[i], seed,
                            latent_shapes=[latent_images[i].shape]
                        )

                    # Initialize x per chunk (noise scaling)
                    is_max_denoise = (
                        math.isclose(float(model_sampling.sigma_max),
                                     float(sigmas[0]), rel_tol=1e-05)
                        or float(sigmas[0]) > float(model_sampling.sigma_max)
                    )
                    chunks_x = []
                    for i in range(N):
                        x = model_sampling.noise_scaling(
                            sigmas[0], noises[i], latent_images[i], is_max_denoise
                        )
                        chunks_x.append(x)
                else:
                    last_real_model = real_model

                # -- Step loop for this model group --
                extra_opts = comfy.model_patcher.create_model_options_clone(
                    guider.model_options
                )
                extra_opts.setdefault(
                    "transformer_options", {}
                )["sample_sigmas"] = sigmas

                self._run_steps(
                    guider, real_model, chunks_x, chunk_conds, noises,
                    latent_images, denoise_masks, sigmas,
                    start_step, end_step, extra_opts,
                    overlap_latent_frames, blend_mode,
                    enable_shift_t, chunk_shift_t,
                    seed, callback, total_steps, N
                )

            finally:
                self._cleanup_model(guider, merged, loaded_models,
                                    orig_opts, orig_hook_mode)

        # -- Merge chunks into single full-length output --
        merged_x = self._merge_chunks(chunks_x, boundaries, T_total)

        merged_samples = model_sampling.inverse_noise_scaling(
            sigmas[-1], merged_x
        )
        merged_samples = last_real_model.process_latent_out(
            merged_samples.to(torch.float32)
        )
        merged_samples = merged_samples.to(
            comfy.model_management.intermediate_device()
        )

        out = latent_image.copy()
        out["samples"] = merged_samples
        out.pop("noise_mask", None)
        out.pop("downscale_ratio_spacial", None)

        print(f"[CoDenoiseChunkSampler] Done. {N} chunks merged to "
              f"{T_total} latent frames.")
        return (out,)

    # ------------------------------------------------------------------
    # Chunk computation
    # ------------------------------------------------------------------

    def _compute_chunk_boundaries(self, total_frames, chunk_length, overlap):
        """Compute (start, end) boundaries for each chunk."""
        if total_frames <= chunk_length:
            return [(0, total_frames)]

        stride = chunk_length - overlap
        if stride <= 0:
            raise ValueError(
                f"overlap_latent_frames ({overlap}) must be less than "
                f"chunk_length_latent ({chunk_length})"
            )

        boundaries = []
        start = 0
        while start < total_frames:
            end = min(start + chunk_length, total_frames)
            boundaries.append((start, end))
            if end >= total_frames:
                break
            start += stride

        return boundaries

    # ------------------------------------------------------------------
    # Conditioning slicing
    # ------------------------------------------------------------------

    def _slice_cond_temporal(self, cond_list, start_lat, end_lat):
        """Create a copy of conditioning with VACE tensors sliced temporally."""
        sliced = []
        for entry in cond_list:
            embedding = entry[0]
            meta = dict(entry[1])

            if "vace_frames" in meta:
                meta["vace_frames"] = [
                    f[:, :, start_lat:min(end_lat, f.shape[2]), :, :]
                    for f in meta["vace_frames"]
                ]
            if "vace_mask" in meta:
                meta["vace_mask"] = [
                    m[:, :, start_lat:min(end_lat, m.shape[2]), :, :]
                    for m in meta["vace_mask"]
                ]

            sliced.append([embedding, meta])
        return sliced

    def _slice_noise_mask(self, noise_mask, start, end, total_t):
        """Slice noise_mask along its temporal dimension."""
        if noise_mask is None:
            return None
        for d in range(noise_mask.ndim):
            if noise_mask.shape[d] == total_t:
                sl = [slice(None)] * noise_mask.ndim
                sl[d] = slice(start, end)
                return noise_mask[tuple(sl)].clone()
        return noise_mask

    # ------------------------------------------------------------------
    # Chunk merging
    # ------------------------------------------------------------------

    def _merge_chunks(self, chunks_x, boundaries, total_frames):
        """Merge co-denoised chunks into a single full-length latent.
        Overlap regions are averaged (after per-step blending, both chunks
        hold the same values in overlap regions, so averaging is identity)."""
        if len(chunks_x) == 1:
            return chunks_x[0]

        B, C, _, H, W = chunks_x[0].shape
        dtype = chunks_x[0].dtype
        device = chunks_x[0].device

        result = torch.zeros(B, C, total_frames, H, W,
                             device=device, dtype=dtype)
        counts = torch.zeros(1, 1, total_frames, 1, 1,
                             device=device, dtype=dtype)

        for i, (start, end) in enumerate(boundaries):
            chunk_T = chunks_x[i].shape[2]
            use_T = min(chunk_T, end - start)
            result[:, :, start:start + use_T, :, :] += \
                chunks_x[i][:, :, :use_T, :, :]
            counts[:, :, start:start + use_T, :, :] += 1.0

        counts = counts.clamp(min=1.0)
        return result / counts

    # ------------------------------------------------------------------
    # Model setup / teardown
    # ------------------------------------------------------------------

    def _setup_model(self, current_model, cfg, merged_conds, chunk_conds,
                     noise_shape, device):
        """Create guider, register hooks, load model to GPU."""
        guider = comfy.samplers.CFGGuider(current_model)
        guider.set_cfg(cfg)

        orig_model_options = guider.model_options
        guider.model_options = comfy.model_patcher.create_model_options_clone(
            orig_model_options
        )

        orig_hook_mode = guider.model_patcher.hook_mode
        if comfy.samplers.get_total_hook_groups_in_conds(merged_conds) <= 1:
            guider.model_patcher.hook_mode = comfy.hooks.EnumHookMode.MinVram

        comfy.sampler_helpers.prepare_model_patcher(
            guider.model_patcher, merged_conds, guider.model_options
        )
        for conds in chunk_conds:
            comfy.samplers.filter_registered_hooks_on_conds(
                conds, guider.model_options
            )

        real_model, _, loaded_models = comfy.sampler_helpers.prepare_sampling(
            guider.model_patcher, noise_shape, merged_conds, guider.model_options
        )
        guider.inner_model = real_model

        comfy.samplers.cast_to_load_options(
            guider.model_options, device=device,
            dtype=guider.model_patcher.model_dtype()
        )

        return guider, real_model, loaded_models, orig_model_options, orig_hook_mode

    def _cleanup_model(self, guider, merged_conds, loaded_models,
                       orig_model_options, orig_hook_mode):
        """Clean up after a model group finishes."""
        guider.model_patcher.cleanup()
        comfy.sampler_helpers.cleanup_models(merged_conds, loaded_models)
        comfy.samplers.cast_to_load_options(
            guider.model_options, device=guider.model_patcher.offload_device
        )
        guider.model_options = orig_model_options
        guider.model_patcher.hook_mode = orig_hook_mode
        guider.model_patcher.restore_hook_patches()

    # ------------------------------------------------------------------
    # Step loop
    # ------------------------------------------------------------------

    def _run_steps(self, guider, real_model, chunks_x, chunk_conds, noises,
                   latent_images, denoise_masks, sigmas,
                   start_step, end_step, extra_opts,
                   overlap, blend_mode,
                   enable_shift_t, chunk_shift_t,
                   seed, callback, total_steps, N):
        """Run euler steps for a model group, blending overlaps each step."""

        for step in range(start_step, end_step):
            sigma = sigmas[step]
            sigma_next = sigmas[step + 1]
            dt = sigma_next - sigma
            s_in = chunks_x[0].new_ones([chunks_x[0].shape[0]])

            comfy.model_management.throw_exception_if_processing_interrupted()

            for ci in range(N):
                guider.conds = chunk_conds[ci]

                # Per-chunk RoPE shift_t (dynamic patch via extra_opts)
                if enable_shift_t and chunk_shift_t[ci] > 0:
                    extra_opts.setdefault(
                        "transformer_options", {}
                    )["rope_options"] = {
                        "scale_x": 1.0, "shift_x": 0.0,
                        "scale_y": 1.0, "shift_y": 0.0,
                        "scale_t": 1.0, "shift_t": float(chunk_shift_t[ci]),
                    }
                elif enable_shift_t:
                    extra_opts.get("transformer_options", {}).pop(
                        "rope_options", None
                    )

                x = chunks_x[ci]

                # Denoise mask handling (KSamplerX0Inpaint pattern)
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

                # Forward pass
                denoised = guider(
                    x_in, sigma * s_in, model_options=extra_opts, seed=seed
                )

                # Post-mask: preserve reference regions
                if dm is not None:
                    denoised = denoised * dm_step + latent_images[ci] * latent_mask

                # Euler step: d = (x - denoised) / sigma; x_new = x + d * dt
                d = (x - denoised) / sigma
                chunks_x[ci] = x + d * dt

            # Per-step overlap blending
            _blend_overlaps(chunks_x, overlap, blend_mode)

            if callback is not None:
                callback(step, chunks_x[0], chunks_x[0], total_steps)

    # ------------------------------------------------------------------
    # Step group computation
    # ------------------------------------------------------------------

    def _compute_step_groups(self, mode, num_models, total_steps,
                             model_steps_str, model_boundaries_str):
        """Compute (model_idx, start_step, end_step) groups."""
        if mode == "single" or num_models == 1:
            return [(0, 0, total_steps)]

        if mode == "sequential":
            distribution = self._parse_step_distribution(
                model_steps_str, total_steps, num_models
            )
            groups = []
            current = 0
            for i, count in enumerate(distribution):
                if count > 0:
                    groups.append((i, current, current + count))
                    current += count
            return groups

        if mode == "boundary":
            boundaries = self._parse_boundaries(
                model_boundaries_str, num_models
            )
            groups = []
            for i in range(len(boundaries) - 1):
                upper = boundaries[i]
                lower = boundaries[i + 1]
                start = max(0, min(
                    int((1.0 - upper) * total_steps), total_steps
                ))
                end = max(0, min(
                    int((1.0 - lower) * total_steps), total_steps
                ))
                if start < end:
                    groups.append((i, start, end))
            return groups if groups else [(0, 0, total_steps)]

        return [(0, 0, total_steps)]

    # ------------------------------------------------------------------
    # Parsing helpers (ported from NV_MultiModelSampler)
    # ------------------------------------------------------------------

    def _parse_step_distribution(self, model_steps_str, total_steps, num_models):
        if model_steps_str.strip():
            try:
                parts = [int(x.strip()) for x in model_steps_str.split(",")
                         if x.strip()]
                while len(parts) < num_models:
                    parts.append(0)
                return parts[:num_models]
            except ValueError:
                pass
        base = total_steps // num_models
        remainder = total_steps % num_models
        return [base + (1 if i < remainder else 0) for i in range(num_models)]

    def _parse_boundaries(self, boundaries_str, num_models):
        boundaries = [1.0]
        if boundaries_str.strip():
            try:
                parts = [float(x.strip()) for x in boundaries_str.split(",")
                         if x.strip()]
                boundaries.extend(parts)
            except ValueError:
                pass
        while len(boundaries) < num_models:
            last = boundaries[-1] if boundaries else 1.0
            boundaries.append(last / 2)
        boundaries.append(0.0)
        return boundaries[:num_models + 1]

    def _parse_cfg_scales(self, cfg_scales_str, default_cfg, num_models):
        if cfg_scales_str.strip():
            try:
                parts = [float(x.strip()) for x in cfg_scales_str.split(",")
                         if x.strip()]
                while len(parts) < num_models:
                    parts.append(default_cfg)
                return parts[:num_models]
            except ValueError:
                pass
        return [default_cfg] * num_models


# ============================================================================
# Node registration
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "NV_CoDenoiseChunkSampler": NV_CoDenoiseChunkSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_CoDenoiseChunkSampler": "NV Co-Denoise Chunk Sampler",
}
