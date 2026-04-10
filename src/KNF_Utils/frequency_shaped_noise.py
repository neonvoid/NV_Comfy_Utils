"""
NV Frequency-Shaped Noise — SHELVED 2026-04-10 (D-029)

Status: SHELVED — kept for reference and potential future use with non-flow-matching models.

Original idea: shape the starting noise of the diffusion sampler to match the source
footage's radial power spectrum, biasing the model toward generating at the source's
quality level (softness, grain, micro-contrast) instead of the typical "AI-clean" look.

Why it failed for WAN/VACE:
- WAN uses flow-matching with a DiT backbone. Flow-matching ODEs solve a deterministic
  trajectory from N(0, I) to data. There is no SDE drift to correct off-distribution
  starts. Any covariance perturbation in the init noise produces a bad trajectory.
- The model reads spatially correlated noise as content to denoise, not as a stylistic
  prior. Even bounded shaping (gain range [0.5, 2.0]) produced rainbow checkerboard
  artifacts in the inpainted region.
- Modern DiT/flow-matching architectures are MORE brittle to init-noise structure than
  older U-Net DDPM models, not less.

Empirical evidence:
- Weak shaping (strength=0.5): no measurable effect on output (post-process sharpness
  measurements identical to baseline).
- Strong shaping (strength=1.0): catastrophic structured artifacts.
- All bug fixes applied (fftshift alignment, DC removal, log-domain blending, bounded
  clamp, deterministic linear filter, per-channel renormalization). Still failed.
- Multi-AI consensus from 2 review rounds: no usable operating window between "ignored"
  and "destabilizing".

Decision: Texture matching is owned by post-processing (NV_TextureHarmonize). For aesthetic
control INSIDE flow-matching DiTs, the recommended approach is FreeU-style feature
manipulation in late blocks (30-39), not init-noise shaping.

The node is not unregistered — it's preserved as a research artifact and may be useful
for non-flow-matching architectures (older U-Net DDPM models). Do NOT wire it into
production WAN/VACE workflows.

Implementation notes (for the curious):
- Pixel-space FFT of full source frame, radial bin averaging, DC excluded
- Power spectrum analysis with sqrt for amplitude filter
- Deterministic linear filter F_out = H * F_white preserves Gaussianity
- Log-domain blending with bounded clamp prevents extreme gain values
- Per-channel renormalization to preserve WAN's 16-channel statistics
"""

import torch
import numpy as np
import comfy.sample


class _FrequencyShapedNoiseObj:
    """NOISE-protocol object that generates frequency-shaped noise."""

    def __init__(self, seed, radial_profile, strength):
        self.seed = seed
        self._radial_profile = radial_profile  # 1D tensor: amplitude per radial frequency bin
        self._strength = strength

    def generate_noise(self, input_latent):
        latent_image = input_latent["samples"]
        batch_inds = input_latent.get("batch_index", None)

        # Generate standard white noise (deterministic from seed)
        white = comfy.sample.prepare_noise(latent_image, self.seed, batch_inds)

        if self._strength <= 0 or self._radial_profile is None:
            return white

        # Shape each spatial slice independently (preserve temporal dimension for 5D)
        return self._apply_shaping(white)

    def _apply_shaping(self, noise):
        """Apply radial frequency shaping to noise tensor.

        Works on both 4D [B, C, H, W] and 5D [B, C, T, H, W] latents.
        Shapes each spatial (H, W) slice independently using batched FFT.

        Algorithm (post-bugfix):
        1. FFT noise → fftshift to put DC at center (matches profile extraction)
        2. Build smooth gain curve H(k) from target/white power ratio (log domain)
        3. Multiply: F_out = H * F_white (linear filter, preserves Gaussianity)
        4. ifftshift → IFFT back to spatial
        """
        shape = noise.shape
        device = noise.device
        profile = self._radial_profile.to(device)
        strength = self._strength

        if noise.ndim == 5:
            B, C, T, H, W = shape
            flat = noise.reshape(-1, H, W)
        elif noise.ndim == 4:
            B, C, H, W = shape
            flat = noise.reshape(-1, H, W)
        else:
            return noise

        # Build 2D radial frequency map (DC at center to match fftshift convention)
        radial_map = _build_radial_map(H, W, len(profile), device)  # [H, W]

        # Look up target amplitude per pixel from radial profile
        target_amp_2d = profile[radial_map]  # [H, W]

        # Compute deterministic gain H(k) from target amplitude.
        # Profile is normalized to mean=1, so a "white" profile would give H=1 everywhere.
        # gain = (target / mean_target) ** strength interpolates in log space:
        #   strength=0 → H=1 (no shaping)
        #   strength=1 → H = target/mean (full shaping)
        eps = 1e-8
        target_norm = target_amp_2d / (target_amp_2d.mean() + eps)
        log_gain = torch.log(target_norm.clamp(min=eps))
        # Clamp log gain to bound the effective gain range (~0.5x to 2x at strength=1)
        log_gain = log_gain.clamp(min=-0.7, max=0.7)
        gain = torch.exp(log_gain * strength)  # [H, W]

        # Batched FFT — much faster than per-slice loop
        f = torch.fft.fft2(flat)                   # [N, H, W]
        f_shifted = torch.fft.fftshift(f, dim=(-2, -1))  # DC at center

        # Apply deterministic linear filter (preserves Gaussianity)
        f_shaped = f_shifted * gain                # broadcast [N, H, W] * [H, W]

        # Unshift and inverse FFT
        f_unshifted = torch.fft.ifftshift(f_shaped, dim=(-2, -1))
        shaped = torch.fft.ifft2(f_unshifted).real

        result = shaped.reshape(shape)

        # Per-channel renormalization to preserve original noise statistics.
        # WAN latents have per-channel mean/std that the sampler relies on.
        if noise.ndim == 5:
            # [B, C, T, H, W] — normalize per channel
            for c in range(C):
                orig_std_c = noise[:, c].std()
                res_std_c = result[:, c].std()
                if res_std_c > eps:
                    result[:, c] = result[:, c] * (orig_std_c / res_std_c)
                # Match mean too (filter shouldn't shift mean but be safe)
                result[:, c] = result[:, c] - result[:, c].mean() + noise[:, c].mean()
        elif noise.ndim == 4:
            for c in range(C):
                orig_std_c = noise[:, c].std()
                res_std_c = result[:, c].std()
                if res_std_c > eps:
                    result[:, c] = result[:, c] * (orig_std_c / res_std_c)
                result[:, c] = result[:, c] - result[:, c].mean() + noise[:, c].mean()

        # Debug: log what the gain curve actually looks like
        gain_min = gain.min().item()
        gain_max = gain.max().item()
        gain_mean = gain.mean().item()
        print(f"[NV_FrequencyShapedNoise] Applied gain: range=[{gain_min:.3f}, {gain_max:.3f}], "
              f"mean={gain_mean:.3f} (strength={strength})")
        print(f"[NV_FrequencyShapedNoise] Noise stats: white std={noise.std().item():.4f}, "
              f"shaped std={result.std().item():.4f}")

        return result


def _build_radial_map(H, W, num_bins, device):
    """Build a 2D map where each pixel contains its radial frequency bin index.

    Returns [H, W] long tensor with values in [0, num_bins-1].
    """
    cy, cx = H // 2, W // 2
    y = torch.arange(H, device=device).float() - cy
    x = torch.arange(W, device=device).float() - cx
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    dist = torch.sqrt(xx ** 2 + yy ** 2)

    # Normalize to [0, num_bins-1]
    max_dist = max(cy, cx)
    if max_dist > 0:
        dist = dist / max_dist * (num_bins - 1)
    return dist.long().clamp(0, num_bins - 1)


def _compute_radial_profile(image, num_bins=64, skip_dc_bins=2):
    """Compute radial power spectrum profile from pixel-space image.

    Takes [B, H, W, C] IMAGE tensor, returns [num_bins] AMPLITUDE profile.
    Profile is sqrt of radially-averaged power, with mean removed and DC excluded.

    Returns the profile normalized so the average gain is 1.0 (so it can be used
    directly as a multiplicative filter, with 1.0 = pass-through).
    """
    # Convert to grayscale for frequency analysis
    gray = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
    B, H, W = gray.shape

    # Subtract per-frame mean to suppress DC (large image-wide brightness bias)
    gray = gray - gray.mean(dim=(1, 2), keepdim=True)

    # Batched FFT
    f = torch.fft.fft2(gray)                                  # [B, H, W]
    f_shifted = torch.fft.fftshift(f, dim=(-2, -1))           # DC at center
    power = (f_shifted.real ** 2 + f_shifted.imag ** 2)       # [B, H, W]

    # Average across batch
    avg_power = power.mean(dim=0)  # [H, W]

    # Build radial bin map
    radial_map = _build_radial_map(H, W, num_bins, gray.device)  # [H, W]

    # Radial average using scatter
    profile_power = torch.zeros(num_bins, device=gray.device)
    counts = torch.zeros(num_bins, device=gray.device)
    flat_power = avg_power.flatten()
    flat_bins = radial_map.flatten()
    profile_power.scatter_add_(0, flat_bins, flat_power)
    counts.scatter_add_(0, flat_bins, torch.ones_like(flat_power))
    profile_power = profile_power / counts.clamp(min=1)

    # Convert power to amplitude (sqrt) — this is what we'll use as a filter
    profile = torch.sqrt(profile_power.clamp(min=1e-12))

    # Zero out DC and first few bins (scene brightness, gross composition)
    if skip_dc_bins > 0:
        profile[:skip_dc_bins] = 0.0

    # Replace zeroed bins with the value of the first valid bin (avoids
    # division-by-zero when computing log gain — these frequencies just
    # become "no shaping" at strength=1)
    if skip_dc_bins > 0 and skip_dc_bins < num_bins:
        first_valid = profile[skip_dc_bins]
        profile[:skip_dc_bins] = first_valid

    # Normalize so the MEAN gain is 1.0 (filter that doesn't change total energy)
    profile = profile / (profile.mean() + 1e-12)

    return profile


class NV_FrequencyShapedNoise:
    """Generate frequency-shaped noise matching a source image's spectral profile."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_image": ("IMAGE", {
                    "tooltip": "Full-frame source video/image. Its frequency profile (lens softness, "
                               "grain, compression) is extracted and transferred to the starting noise. "
                               "Can be any resolution — only the radial spectrum shape matters."
                }),
                "noise_seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "control_after_generate": True,
                    "tooltip": "Random seed for noise generation. Same seed = same spatial pattern."
                }),
            },
            "optional": {
                "strength": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "How much to shape the noise toward the source profile. "
                               "0.0 = pure white noise (standard). 1.0 = fully shaped. "
                               "0.3-0.5 recommended for first tests."
                }),
                "num_bins": ("INT", {
                    "default": 64, "min": 16, "max": 256, "step": 16,
                    "tooltip": "Radial frequency bins for profile extraction. "
                               "64 = good balance of resolution vs smoothness."
                }),
                "sample_frames": ("INT", {
                    "default": 8, "min": 1, "max": 32, "step": 1,
                    "tooltip": "Number of frames to sample from reference for profile averaging. "
                               "More = smoother profile, slower analysis."
                }),
            },
        }

    RETURN_TYPES = ("NOISE",)
    RETURN_NAMES = ("noise",)
    FUNCTION = "execute"
    CATEGORY = "NV_Utils/sampling"
    DESCRIPTION = (
        "Generates frequency-shaped starting noise for SamplerCustomAdvanced. "
        "Analyzes the full-frame source image's radial power spectrum and shapes "
        "the random noise to match — biasing the model toward generating at the "
        "source's quality level (softness, grain, micro-contrast). "
        "Drop-in replacement for RandomNoise."
    )

    def execute(self, reference_image, noise_seed, strength=0.5, num_bins=64, sample_frames=8):
        TAG = "[NV_FrequencyShapedNoise]"
        B = reference_image.shape[0]

        # Sample frames for profile computation (avoid processing entire video)
        if B > sample_frames:
            indices = torch.linspace(0, B - 1, sample_frames).long()
            sampled = reference_image[indices]
        else:
            sampled = reference_image

        print(f"{TAG} Analyzing {sampled.shape[0]} frames at {sampled.shape[1]}x{sampled.shape[2]} "
              f"({num_bins} bins, strength={strength})")

        # Compute radial amplitude profile (DC removed, normalized to mean=1)
        profile = _compute_radial_profile(sampled, num_bins, skip_dc_bins=2)

        # Log profile shape — now meaningful since DC is removed and mean is 1.0
        # Values < 1.0 = below average, > 1.0 = above average
        # A "soft" source has high low-freq values and low high-freq values
        low_avg = profile[2:num_bins // 4].mean().item()  # skip DC bins
        mid_avg = profile[num_bins // 4:num_bins * 3 // 4].mean().item()
        high_avg = profile[num_bins * 3 // 4:].mean().item()
        slope = profile[2].item() / max(profile[-1].item(), 1e-8)
        print(f"{TAG} Profile (mean=1.0): low={low_avg:.3f}, mid={mid_avg:.3f}, high={high_avg:.3f}, "
              f"low/high slope={slope:.1f}x")

        noise_obj = _FrequencyShapedNoiseObj(noise_seed, profile, strength)
        return (noise_obj,)


NODE_CLASS_MAPPINGS = {
    "NV_FrequencyShapedNoise": NV_FrequencyShapedNoise,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_FrequencyShapedNoise": "NV Frequency-Shaped Noise",
}
