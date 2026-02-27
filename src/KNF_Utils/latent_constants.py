"""
Shared constants for LATENT dict handling across NV_Comfy_Utils nodes.

Centralizes:
- NV_CASCADED_CONFIG_KEY: dict key for cascaded pipeline config embedded in LATENT
- LATENT_SAFE_KEYS: keys safe to copy when building new LATENT dicts
  (excludes temporal-dependent keys like noise_mask and batch_index)

Import from node modules:
    from .latent_constants import NV_CASCADED_CONFIG_KEY, LATENT_SAFE_KEYS
"""

# Key name for cascaded pipeline config embedded in LATENT dicts.
# Contains: shift_override, expanded_steps, start_at_step, start_sigma,
# signal_preserved_pct, prenoise_denoise, prenoise_steps, prenoise_seed, etc.
NV_CASCADED_CONFIG_KEY = "nv_cascaded_config"

# Keys safe to carry forward when building new LATENT dicts from temporal operations
# (concat, slice, prepend). Excludes noise_mask and batch_index which are
# temporal-dimension-dependent and become stale after shape changes.
LATENT_SAFE_KEYS = frozenset({
    "downscale_ratio_spacial",
    "latent_format_version_0",
    NV_CASCADED_CONFIG_KEY,
})
