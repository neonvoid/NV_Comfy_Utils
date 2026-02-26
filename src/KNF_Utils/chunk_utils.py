"""
Shared utilities for chunked video pipeline.

Pure functions — no node classes, no NODE_CLASS_MAPPINGS.
Import from node modules:
    from .chunk_utils import video_to_latent_frames, is_wan_aligned, ...

This module is a leaf dependency: it imports nothing from other NV_Comfy_Utils
node modules. All node modules may import from it without creating cycles.
"""

import math
import torch
from typing import Optional, Tuple, List, Callable


# ============================================================================
# Frame conversion (WAN VAE temporal compression 4:1)
# ============================================================================

def video_to_latent_frames(video_frames: int) -> int:
    """Convert video (pixel) frame count to latent frame count.

    WAN VAE temporal compression: latent = (video - 1) // 4 + 1
    The first frame gets its own latent slot (the "first-frame bonus").

    Examples: 1->1, 5->2, 9->3, 13->4, 17->5, 81->21
    """
    if video_frames <= 0:
        return 0
    return (video_frames - 1) // 4 + 1


def latent_to_video_frames(latent_frames: int) -> int:
    """Convert latent frame count to video (pixel) frame count.

    Inverse of video_to_latent_frames:
      video = (latent - 1) * 4 + 1

    Examples: 1->1, 2->5, 3->9, 5->17, 21->81
    """
    if latent_frames <= 0:
        return 0
    return (latent_frames - 1) * 4 + 1


# ============================================================================
# WAN frame alignment
# ============================================================================

def is_wan_aligned(frames: int) -> bool:
    """Check if frame count satisfies WAN constraint: (frames % 4) == 1.

    Valid counts: 1, 5, 9, 13, 17, 21, 25, 29, 33, ...
    """
    return (frames % 4) == 1


def nearest_wan_aligned(frames: int, round_up: bool = False) -> int:
    """Return the nearest valid WAN frame count.

    Args:
        frames: Input frame count (must be > 0).
        round_up: If True, round up to next valid count. Default rounds down.
    """
    if frames <= 0:
        return 1  # Minimum valid WAN frame count
    if is_wan_aligned(frames):
        return frames
    if round_up:
        return ((frames - 1) // 4 + 1) * 4 + 1
    return ((frames - 1) // 4) * 4 + 1


def validate_wan_alignment(frames: int) -> Tuple[bool, int, int]:
    """Check WAN alignment and return (is_valid, nearest_valid, difference)."""
    is_valid = is_wan_aligned(frames)
    nearest = nearest_wan_aligned(frames)
    diff = frames - nearest
    return (is_valid, nearest, diff)


def adjust_for_wan_alignment(frame_count: int, min_frames: int = 5) -> int:
    """Round down to nearest WAN-aligned count, with minimum floor."""
    aligned = nearest_wan_aligned(frame_count)
    return max(aligned, min_frames)


# ============================================================================
# Blend weight functions
# ============================================================================

def compute_blend_weights(
    num_frames: int,
    mode: str = "linear",
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Compute crossfade blend weights from 0.0 (keep A) to 1.0 (keep B).

    Supports: "linear", "cosine"/"hann", "hamming".
    Returns tensor of shape [num_frames].
    """
    if num_frames <= 1:
        return torch.ones(max(1, num_frames), device=device)

    t = torch.linspace(0, 1, num_frames, device=device)

    if mode == "linear":
        return t
    elif mode in ("cosine", "hann"):
        return 0.5 * (1.0 - torch.cos(math.pi * t))
    elif mode == "hamming":
        return 0.54 - 0.46 * torch.cos(math.pi * t)
    else:
        return t  # Fallback to linear


def compute_ramp_weights(num_frames: int, mode: str = "cosine") -> torch.Tensor:
    """Compute ramp weights with exclusive endpoints (no 0.0 or 1.0).

    For boundary noise masks: ensures every ramp frame has a truly
    intermediate value. The 0.0 and 1.0 belong to context and core zones.

    For 4 frames: ~[0.1, 0.3, 0.7, 0.9] (cosine) or [0.2, 0.4, 0.6, 0.8] (linear).
    """
    if num_frames <= 0:
        return torch.tensor([])
    if num_frames == 1:
        return torch.tensor([0.5])

    t = torch.linspace(0, 1, num_frames + 2)[1:-1]  # Exclusive endpoints
    if mode == "cosine":
        return 0.5 * (1.0 - torch.cos(math.pi * t))
    else:
        return t


# ============================================================================
# Latent overlap computation
# ============================================================================

def compute_latent_overlap(
    video_a_frames: int,
    video_b_frames: int,
    overlap_video_frames: int,
    adjust: int = 0,
) -> int:
    """Compute precise latent overlap between two chunks.

    Accounts for WAN VAE's first-frame encoding asymmetry:
      combined_video = video_A + video_B - overlap_video
      combined_latent = video_to_latent_frames(combined_video)
      overlap_latent = latent_A + latent_B - combined_latent + adjust
    """
    latent_a = video_to_latent_frames(video_a_frames)
    latent_b = video_to_latent_frames(video_b_frames)
    combined_video = video_a_frames + video_b_frames - overlap_video_frames
    combined_latent = video_to_latent_frames(combined_video)
    overlap_latent = latent_a + latent_b - combined_latent + adjust
    return max(1, min(overlap_latent, latent_a, latent_b))


# ============================================================================
# VRAM-aware max frames computation
# ============================================================================

def estimate_max_inference_frames(
    estimate_fn: Callable[[int], int],
    available_vram_bytes: int,
    low: int = 5,
    high: int = 2001,
) -> int:
    """Binary search for the maximum pixel frame count that fits in VRAM.

    Args:
        estimate_fn: Callable(pixel_frames) -> peak_bytes.
            Wrap estimate_inference_peak() with functools.partial() binding
            all params except total_pixel_frames.
        available_vram_bytes: Target VRAM budget in bytes.
        low, high: Search bounds in pixel frames.

    Returns: Maximum WAN-aligned pixel frame count that fits.
    """
    low = max(1, nearest_wan_aligned(low, round_up=True))
    high = nearest_wan_aligned(high)

    result = low
    while low <= high:
        mid = nearest_wan_aligned((low + high) // 2)
        if mid < low:
            break

        peak = estimate_fn(mid)
        if peak <= available_vram_bytes:
            result = mid
            low = mid + 4  # Next WAN-aligned candidate
        else:
            high = mid - 4

    return result
