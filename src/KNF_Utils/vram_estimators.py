"""WAN/VACE VRAM estimator formulas — extracted from vace_benchmark.py.

Heuristic predictors of feat_cache, VACE tensor, and forward-activation memory
given workflow params. Useful for capacity planning and pre-flight checks
(e.g. NV_WanMemoryEstimator / NV_WanVramEstimator could call into these).

These are *predictions*, not measurements — the actual measured values come
from `shot_runtime.get_memory_snapshot()`. Keep these around because they
let workflows estimate VRAM cost BEFORE running, which the measurement nodes
can't do by definition.
"""


def estimate_feat_cache(H, W):
    """Calculate exact feat_cache size for WAN VAE encoder, in GB.

    Cache holds 2 frames per stage at fp32. Five stages with halving spatial
    resolution + middle block + head.
    """
    if H == 0 or W == 0:
        return 0.0

    CACHE_T = 2
    dtype_bytes = 4  # fp32

    conv1 = 3 * CACHE_T * H * W
    stage0 = 4 * 128 * CACHE_T * H * W + 128 * 1 * (H // 2) * (W // 2)
    stage1 = 4 * 256 * CACHE_T * (H // 2) * (W // 2) + 256 * 1 * (H // 4) * (W // 4)
    stage2 = 4 * 512 * CACHE_T * (H // 4) * (W // 4)
    stage3 = 4 * 512 * CACHE_T * (H // 8) * (W // 8)
    middle = 4 * 512 * CACHE_T * (H // 8) * (W // 8)
    head = 512 * CACHE_T * (H // 8) * (W // 8)

    total_elements = conv1 + stage0 + stage1 + stage2 + stage3 + middle + head
    total_bytes = total_elements * dtype_bytes
    return round(total_bytes / 1024**3, 3)


def estimate_vace_tensors(latent_frames, H, W):
    """Calculate VACE control tensor + packed mask size, in GB.

    VACE control = 32-channel residual at latent resolution.
    VACE mask    = 64-channel packed (8x8 pixel-shuffle) at latent resolution.
    Both fp32, single batch.
    """
    if latent_frames == 0 or H == 0 or W == 0:
        return 0.0

    latent_H = H // 8
    latent_W = W // 8
    dtype_bytes = 4  # fp32

    control = 1 * 32 * latent_frames * latent_H * latent_W * dtype_bytes
    mask    = 1 * 64 * latent_frames * latent_H * latent_W * dtype_bytes
    return round((control + mask) / 1024**3, 3)


def estimate_forward_activations(H, W, chunk_frames=4):
    """Estimate peak forward-pass activation memory, in GB.

    Peak occurs at stage 0 (full spatial resolution × 128 channels), with
    a 1.5× safety multiplier for transient buffers (kernel launches,
    intermediate tensors, autograd shadows on graph re-trace).
    """
    if H == 0 or W == 0:
        return 0.0

    dtype_bytes = 4
    stage0_peak = 2 * 3 * 128 * chunk_frames * H * W * dtype_bytes
    return round(stage0_peak * 1.5 / 1024**3, 3)
