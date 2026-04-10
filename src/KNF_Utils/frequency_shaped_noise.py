"""
NV Frequency-Shaped Noise — Shape starting noise to match source footage's frequency profile.

Standard diffusion starts from white noise (equal energy at all frequencies). This biases
the model toward generating "clean" AI-perfect output. By shaping the starting noise to
match the source footage's frequency profile, we bias the model toward generating at the
same quality level as the source.

The node analyzes a full-frame reference image's radial power spectrum (how much energy
at each spatial frequency) and transfers that profile onto the standard random noise.
Phase (spatial arrangement) stays random — we're not copying the image, just its
"frequency fingerprint."

Uses pixel-space FFT on the full frame for maximum fidelity — captures lens softness,
sensor grain, compression artifacts that the VAE would destroy.

Outputs a NOISE object compatible with SamplerCustomAdvanced / SamplerCustom.
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
        Shapes each spatial (H, W) slice independently.
        """
        shape = noise.shape
        device = noise.device
        profile = self._radial_profile.to(device)
        strength = self._strength

        if noise.ndim == 5:
            B, C, T, H, W = shape
            # Reshape to [B*C*T, H, W] for per-slice FFT
            flat = noise.reshape(-1, H, W)
        elif noise.ndim == 4:
            B, C, H, W = shape
            flat = noise.reshape(-1, H, W)
        else:
            return noise

        # Build 2D radial frequency map for this spatial size
        radial_map = _build_radial_map(H, W, len(profile), device)  # [H, W]

        # Look up target amplitude per pixel from radial profile
        target_amp = profile[radial_map]  # [H, W]

        # FFT each slice
        shaped_slices = []
        for i in range(flat.shape[0]):
            f = torch.fft.fft2(flat[i])
            amp = torch.abs(f)
            phase = f / (amp + 1e-8)

            # Blend amplitude toward target profile
            blended_amp = amp * (1.0 - strength) + target_amp * strength

            # Reconstruct
            shaped = torch.fft.ifft2(blended_amp * phase).real
            shaped_slices.append(shaped)

        result = torch.stack(shaped_slices, dim=0).reshape(shape)

        # Normalize to match original noise statistics (preserve sigma scale)
        orig_std = noise.std()
        result_std = result.std()
        if result_std > 1e-8:
            result = result * (orig_std / result_std)

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


def _compute_radial_profile(image, num_bins=64):
    """Compute radial power spectrum profile from pixel-space image.

    Takes [B, H, W, C] IMAGE tensor, returns [num_bins] amplitude profile (averaged
    across batch, channels, normalized).
    """
    # Convert to grayscale for frequency analysis
    gray = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
    # gray shape: [B, H, W]

    B, H, W = gray.shape
    profiles = []

    for b in range(B):
        f = torch.fft.fft2(gray[b])
        f_shifted = torch.fft.fftshift(f)
        amp = torch.abs(f_shifted)

        # Build radial bin map
        radial_map = _build_radial_map(H, W, num_bins, gray.device)

        # Average amplitude per radial bin
        profile = torch.zeros(num_bins, device=gray.device)
        counts = torch.zeros(num_bins, device=gray.device)
        for bin_idx in range(num_bins):
            mask = radial_map == bin_idx
            if mask.any():
                profile[bin_idx] = amp[mask].mean()
                counts[bin_idx] = mask.float().sum()

        profiles.append(profile)

    # Average across batch
    avg_profile = torch.stack(profiles, dim=0).mean(dim=0)

    # Normalize so total energy = 1
    total = avg_profile.sum()
    if total > 0:
        avg_profile = avg_profile / total

    return avg_profile


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

        # Compute radial power spectrum profile
        profile = _compute_radial_profile(sampled, num_bins)

        # Log profile shape (low vs high frequency energy)
        low_energy = profile[:num_bins // 4].sum().item()
        high_energy = profile[num_bins * 3 // 4:].sum().item()
        ratio = low_energy / max(high_energy, 1e-8)
        print(f"{TAG} Profile: low-freq energy={low_energy:.4f}, high-freq={high_energy:.4f}, "
              f"ratio={ratio:.1f}x (higher = softer source)")

        noise_obj = _FrequencyShapedNoiseObj(noise_seed, profile, strength)
        return (noise_obj,)


NODE_CLASS_MAPPINGS = {
    "NV_FrequencyShapedNoise": NV_FrequencyShapedNoise,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_FrequencyShapedNoise": "NV Frequency-Shaped Noise",
}
