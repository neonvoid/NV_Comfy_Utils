"""
NV SAM3Preprocess — Grade-the-plate preprocessing for SAM3 segmentation.

When SAM3 fails to segment a region (wispy hair on dark background, skin blending
with warm backgrounds, semi-transparent fabric, low-contrast wardrobe), the problem
is often in the image signal itself, not the model. Same VFX principle as keying:
when a key fails, grade the plate, don't buy a better keyer.

This node applies conservative, evidence-backed image operations to improve SAM3's
signal-to-noise ratio on degraded inputs. The processed image goes to SAM3; the
resulting mask is applied to the ORIGINAL pixels (spatial alignment preserved —
preprocessing doesn't warp).

Operation order (applied only when each is enabled):
  1. Guided filter denoise (optional) — edge-preserving smoothing that avoids
     the staircasing that bilateral filtering induces. SAM would latch onto
     bilateral staircases as false boundaries.
  2. Gamma correction on luminance — global shadow lift for dark-on-dark cases.
     Often more predictable than CLAHE alone.
  3. CLAHE on Lab L channel — local contrast enhancement. Only the L channel
     is quantized for the OpenCV CLAHE call; a/b stay float throughout to avoid
     chroma drift.

Evidence base:
  - Huang et al. 2024 (Bioengineering): preprocessing before MedSAM improved
    IoU +0.081, Dice +0.050 on liver segmentation.
  - Medical imaging literature: CLAHE on L-channel of Lab at clipLimit 2.0-4.0,
    tileGridSize 8x8, consistently improves Dice on low-contrast inputs.
  - RobustSAM (CVPR 2024): confirms SAM degrades on blur/haze/lowlight but
    fixes it architecturally — preprocessing alone is insufficient for the
    hardest cases.

Known limits:
  - Well-exposed in-distribution photos: preprocessing gains are small to
    negative. This is a targeted fix for degraded inputs, not always-on.
  - Global histogram equalization is explicitly avoided (degrades edges per
    Huang 2024).
  - Per-frame CLAHE recomputes local histograms per frame; on video this can
    introduce mild temporal flicker if clip_limit is aggressive. Keep
    clip_limit low (≤1.5) for video.
  - Does not solve true VNS (Visually Non-Salient) failure cases. For those,
    consider MatAnyone / BiRefNet as a second-opinion segmentor rather than
    more aggressive preprocessing.

Workflow pattern:
  IMAGE ─┬─► NV_SAM3Preprocess ──► SAM3 ──► mask
         │
         └─► (branch original) ──► InpaintCrop2 / compositor (apply mask to original)
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

from .guided_filter import guided_filter_color, guided_filter_fast


# =============================================================================
# Core operations
# =============================================================================


def _apply_guided_filter_denoise(image_bchw: torch.Tensor, radius: int, eps: float) -> torch.Tensor:
    """Edge-preserving smoothing via guided filter (He et al. 2013).

    Uses the image as both guide and source — self-guided filtering acts as
    an edge-aware denoiser. Unlike bilateral filtering, this does not produce
    staircase artifacts that SAM would interpret as boundaries.

    Args:
        image_bchw: [B, 3, H, W] float32 in [0, 1]
        radius: filter window radius
        eps: regularization (smaller = sharper edges, larger = more smoothing)
    Returns:
        [B, 3, H, W] float32 in [0, 1]
    """
    b, _, h, w = image_bchw.shape

    # Process each channel independently against the color guide to preserve color edges.
    # We use the color guided filter per-channel: guide = full RGB, src = single channel.
    result = torch.empty_like(image_bchw)
    use_fast = h * w >= 1280 * 720

    for c in range(3):
        src = image_bchw[:, c:c + 1]  # [B, 1, H, W]
        if use_fast:
            filtered = guided_filter_fast(image_bchw, src, radius, eps, subsample=4)
        else:
            filtered = guided_filter_color(image_bchw, src, radius, eps)
        result[:, c:c + 1] = filtered

    return result.clamp(0.0, 1.0)


def _apply_gamma_on_luminance(image_bhwc: torch.Tensor, gamma: float) -> torch.Tensor:
    """Apply gamma to luminance only, preserving hue/saturation.

    Uses Rec. 601 luma weights. gamma > 1.0 lifts shadows, gamma < 1.0 lifts
    highlights. Operates in RGB space by scaling all channels uniformly per
    pixel based on the luminance ratio, which preserves chroma.

    Args:
        image_bhwc: [B, H, W, 3] float32 in [0, 1]
        gamma: gamma value (1.0 = identity)
    Returns:
        [B, H, W, 3] float32 in [0, 1]
    """
    if abs(gamma - 1.0) < 1e-6:
        return image_bhwc

    # Rec. 601 luma
    weights = image_bhwc.new_tensor([0.299, 0.587, 0.114]).view(1, 1, 1, 3)
    luma = (image_bhwc * weights).sum(dim=-1, keepdim=True).clamp(1e-6, 1.0)  # [B, H, W, 1]

    # Gamma-adjusted luminance. gamma > 1.0 lifts shadows.
    new_luma = luma.pow(1.0 / gamma)

    # Scale each pixel's RGB by the luminance ratio (preserves chroma).
    ratio = new_luma / luma
    out = image_bhwc * ratio
    return out.clamp(0.0, 1.0)


def _apply_clahe_on_lab_L(
    image_bhwc: torch.Tensor,
    clip_limit: float,
    tile_grid: int,
) -> torch.Tensor:
    """Apply CLAHE to the L channel of Lab color space, per-frame.

    Critical implementation detail: keeps a/b channels in float throughout.
    Only L is quantized to uint16 for the OpenCV CLAHE call (OpenCV's CLAHE
    accepts CV_8UC1 or CV_16UC1 — uint16 preserves tonal precision on 10-bit
    sources).

    Args:
        image_bhwc: [B, H, W, 3] float32 in [0, 1]
        clip_limit: CLAHE clipLimit. 0.0 disables (no-op).
        tile_grid: CLAHE tileGridSize (NxN tiles).
    Returns:
        [B, H, W, 3] float32 in [0, 1]
    """
    if clip_limit <= 0.0:
        return image_bhwc

    device = image_bhwc.device
    dtype = image_bhwc.dtype
    b = image_bhwc.shape[0]

    # OpenCV expects numpy. Iterate frames. Per-frame CPU cost for 1080p CLAHE
    # is ~5-10ms — acceptable preprocessing overhead below SAM's per-frame cost.
    img_np = image_bhwc.detach().cpu().numpy().astype(np.float32, copy=False)  # [B, H, W, 3]

    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile_grid), int(tile_grid)))
    out_np = np.empty_like(img_np)

    # L range in float Lab: [0, 100]. Normalize to [0, 65535] uint16 for CLAHE precision.
    L_SCALE = 65535.0 / 100.0
    L_INV_SCALE = 100.0 / 65535.0

    for i in range(b):
        rgb_f = img_np[i]  # [H, W, 3] float32 in [0, 1]
        lab = cv2.cvtColor(rgb_f, cv2.COLOR_RGB2Lab)  # L in [0, 100], a/b in [-128, 127]
        L = lab[..., 0]

        # Quantize only L to uint16 for CLAHE; a/b stay float.
        L_u16 = np.clip(L * L_SCALE, 0.0, 65535.0).astype(np.uint16)
        L_eq_u16 = clahe.apply(L_u16)
        L_eq = L_eq_u16.astype(np.float32) * L_INV_SCALE  # back to [0, 100]

        lab_out = lab.copy()
        lab_out[..., 0] = L_eq
        rgb_out = cv2.cvtColor(lab_out, cv2.COLOR_Lab2RGB)
        out_np[i] = np.clip(rgb_out, 0.0, 1.0)

    return torch.from_numpy(out_np).to(device=device, dtype=dtype)


# =============================================================================
# Node
# =============================================================================


class NV_SAM3Preprocess:
    """Grade-the-plate preprocessing for SAM3 on degraded inputs."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Source image(s) to preprocess before SAM3. [B, H, W, C] float in [0, 1]. "
                               "Feed the processed output to SAM3, and branch the ORIGINAL image to "
                               "downstream compositing nodes — preprocessing does not warp pixels, so "
                               "masks from SAM3 stay spatially aligned with the original."
                }),
            },
            "optional": {
                "gamma": ("FLOAT", {
                    "default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05,
                    "tooltip": "Shadow/highlight lift applied to luminance (Rec. 601) before CLAHE. "
                               ">1.0 lifts shadows (good for dark hair on dark backgrounds). "
                               "<1.0 lifts highlights (rare — use for blown-out subjects). "
                               "1.0 = disabled. Often more predictable than pushing CLAHE hard."
                }),
                "clahe_clip_limit": ("FLOAT", {
                    "default": 1.5, "min": 0.0, "max": 3.0, "step": 0.1,
                    "tooltip": "CLAHE clipLimit applied to the L channel of Lab color space. "
                               "0.0 = disabled. 1.0-1.5 = conservative (safe on in-distribution). "
                               "2.0-3.0 = medical imaging norm (stronger, may introduce video flicker). "
                               "Capped at 3.0 — higher values are off-manifold for SAM's image encoder."
                }),
                "clahe_tile_grid": ("INT", {
                    "default": 8, "min": 4, "max": 16, "step": 1,
                    "tooltip": "CLAHE tileGridSize (NxN tiles). Default 8 = OpenCV standard. "
                               "Smaller tiles = more aggressive local contrast but risk of spatial "
                               "instability across tile boundaries."
                }),
                "guided_filter_radius": ("INT", {
                    "default": 0, "min": 0, "max": 16, "step": 1,
                    "tooltip": "Edge-preserving denoise via guided filter (He et al. 2013) applied BEFORE "
                               "gamma+CLAHE. 0 = disabled. 2-4 suppresses CLAHE-amplified sensor noise "
                               "without staircasing. Unlike bilateral filter, does not create false edges "
                               "that SAM would latch onto."
                }),
                "guided_filter_eps": ("FLOAT", {
                    "default": 0.01, "min": 0.0001, "max": 0.1, "step": 0.0001,
                    "tooltip": "Guided filter regularization. Larger = more smoothing, smaller = sharper "
                               "edge tracking. Only used when guided_filter_radius > 0. 0.01 = mild "
                               "smoothing suitable for noise suppression."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("processed",)
    FUNCTION = "execute"
    CATEGORY = "NV_Utils/Mask"
    DESCRIPTION = (
        "Grade-the-plate preprocessing for SAM3: optional guided-filter denoise, gamma on "
        "luminance, then CLAHE on Lab L. Targeted fix for wispy hair on dark backgrounds, "
        "skin-vs-warm-background, semi-transparent fabric. Not recommended as always-on — "
        "may reduce quality on well-exposed in-distribution inputs."
    )

    def execute(
        self,
        image: torch.Tensor,
        gamma: float = 1.0,
        clahe_clip_limit: float = 1.5,
        clahe_tile_grid: int = 8,
        guided_filter_radius: int = 0,
        guided_filter_eps: float = 0.01,
    ):
        TAG = "[NV_SAM3Preprocess]"

        if image.dim() != 4 or image.shape[-1] < 3:
            raise ValueError(
                f"{TAG} Expected IMAGE as [B, H, W, C>=3] tensor, got shape {list(image.shape)}."
            )

        # Work in float32 for CV ops regardless of input dtype.
        x = image[..., :3].to(dtype=torch.float32).clamp(0.0, 1.0)
        b, h, w, _ = x.shape

        ops_applied = []

        # 1) Guided filter denoise (operates in [B, C, H, W])
        if guided_filter_radius > 0:
            x_bchw = x.permute(0, 3, 1, 2).contiguous()
            x_bchw = _apply_guided_filter_denoise(x_bchw, guided_filter_radius, guided_filter_eps)
            x = x_bchw.permute(0, 2, 3, 1).contiguous()
            ops_applied.append(f"guided_filter(r={guided_filter_radius}, eps={guided_filter_eps})")

        # 2) Gamma on luminance
        if abs(gamma - 1.0) > 1e-6:
            x = _apply_gamma_on_luminance(x, gamma)
            ops_applied.append(f"gamma({gamma:.2f})")

        # 3) CLAHE on Lab L
        if clahe_clip_limit > 0.0:
            x = _apply_clahe_on_lab_L(x, clahe_clip_limit, clahe_tile_grid)
            ops_applied.append(f"clahe(clip={clahe_clip_limit:.2f}, tile={clahe_tile_grid})")

        if not ops_applied:
            print(f"{TAG} No operations enabled — passthrough. B={b}, {h}x{w}.")
        else:
            print(f"{TAG} B={b} frames {h}x{w}: {' → '.join(ops_applied)}")

        # Preserve any extra channels (e.g. alpha) from input
        if image.shape[-1] > 3:
            extra = image[..., 3:].to(dtype=torch.float32)
            x = torch.cat([x, extra], dim=-1)

        return (x.to(dtype=image.dtype),)


NODE_CLASS_MAPPINGS = {
    "NV_SAM3Preprocess": NV_SAM3Preprocess,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_SAM3Preprocess": "NV SAM3 Preprocess",
}
