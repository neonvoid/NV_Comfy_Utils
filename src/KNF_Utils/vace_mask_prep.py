"""
NV VACE Mask Prep — VAE-aware mask preprocessing for seam-free VACE inpainting.

WanVaceToVideo splits the control video into inactive/reactive channels at the
mask boundary, then VAE-encodes both. A hard mask edge creates a step-function
discontinuity that the VAE (8x8 spatial stride) can't cleanly encode within a
single block, causing latent-space ringing → visible dark seam around the
inpainted character.

This node fixes the problem by:
  1. Eroding the mask inward (~half a VAE block) to pull the edge away from any
     control-video/mask alignment mismatch.
  2. Feathering the edge (~1.5 VAE blocks) so the transition spans more than one
     full encoding block, eliminating the discontinuity.

In "auto" mode, the node analyzes the mask via distance transform to compute the
inscribed radius (minimum thickness), then derives safe erosion/feather values
that won't collapse thin mask regions.
"""

import torch
import numpy as np
from scipy.ndimage import distance_transform_edt

from .inpaint_crop import mask_erode_dilate, mask_blur


def compute_inscribed_radius(mask: torch.Tensor) -> tuple[list[float], int]:
    """Compute inscribed radius per frame via EDT. Returns (radii_list, min_radius_frame_idx)."""
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)

    radii = []
    for b in range(mask.shape[0]):
        mask_bin = (mask[b] > 0.5).cpu().numpy().astype(np.uint8)
        if np.any(mask_bin):
            dist = distance_transform_edt(mask_bin)
            radii.append(float(np.max(dist)))
        else:
            radii.append(0.0)

    min_idx = int(np.argmin(radii)) if radii else 0
    return radii, min_idx


class NV_VaceMaskPrep:
    """Preprocess masks for VACE inpainting to eliminate dark seam artifacts."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "mode": (["auto", "manual"], {
                    "default": "auto",
                    "tooltip": "auto: analyze mask geometry and derive optimal erosion/feather. "
                               "manual: use erosion_blocks and feather_blocks values directly."
                }),
                "erosion_blocks": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 4.0, "step": 0.25,
                    "tooltip": "(Manual mode) Erode mask inward by this many VAE blocks. "
                               "0.5 blocks = 4px for WAN. Fixes control-video/mask edge misalignment."
                }),
                "feather_blocks": ("FLOAT", {
                    "default": 1.5, "min": 0.0, "max": 8.0, "step": 0.25,
                    "tooltip": "(Manual mode) Feather mask edge over this many VAE blocks. "
                               "1.5 blocks = 12px for WAN. Smooths the transition so no single "
                               "8x8 VAE block sees a hard discontinuity."
                }),
                "threshold": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Binarize mask at 0.5 before processing. Enable if mask has "
                               "intermediate values from resizing that should be hard before feathering."
                }),
            },
            "optional": {
                "vae_stride": ("INT", {
                    "default": 8, "min": 4, "max": 32, "step": 4,
                    "tooltip": "VAE spatial stride in pixels (always 8 for WAN). "
                               "Controls block-to-pixel conversion."
                }),
            },
        }

    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("mask", "info")
    FUNCTION = "execute"
    CATEGORY = "NV_Utils/VACE"
    DESCRIPTION = (
        "Preprocess masks for VACE inpainting to eliminate dark seam artifacts. "
        "In auto mode, analyzes mask geometry to derive optimal erosion and feather. "
        "In manual mode, parameters are in VAE block units for intuitive control."
    )

    def execute(self, mask, mode, erosion_blocks, feather_blocks,
                threshold=False, vae_stride=8):
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        info_lines = [f"[NV_VaceMaskPrep] mode={mode} | vae_stride={vae_stride}px"]

        # --- Edge case: trivial masks ---
        mask_min = mask.min().item()
        mask_max = mask.max().item()

        if mask_max < 0.01:
            info_lines.append("  Mask is all zeros — passing through unchanged")
            info = "\n".join(info_lines)
            print(info)
            return (mask, info)

        if mask_min > 0.99:
            info_lines.append("  Mask is all ones — passing through unchanged")
            info = "\n".join(info_lines)
            print(info)
            return (mask, info)

        result = mask.clone()

        # --- Step 1: Threshold ---
        if threshold:
            result = (result > 0.5).float()
            info_lines.append("  Threshold: applied at 0.5")

        # --- Step 2: Analyze mask geometry ---
        radii, min_frame = compute_inscribed_radius(result)
        min_radius = radii[min_frame] if radii else 0.0
        radius_blocks = min_radius / vae_stride

        info_lines.append(
            f"  Inscribed radius: {min_radius:.1f}px ({radius_blocks:.1f} VAE blocks)"
            f" — min at frame {min_frame}/{len(radii)}"
        )

        # --- Step 3: Compute pixel values ---
        if mode == "auto":
            # Erosion: target half a VAE block, cap at 1/3 inscribed radius
            erosion_target = vae_stride * 0.5
            erosion_max_safe = max(1.0, min_radius / 3.0)
            erosion_px = int(round(min(erosion_target, erosion_max_safe)))

            # Feather: target 1.5 VAE blocks, cap at half inscribed radius
            #          but always allow at least 1 VAE block
            feather_target = vae_stride * 1.5
            feather_max_safe = max(float(vae_stride), min_radius / 2.0)
            feather_px = int(round(min(feather_target, feather_max_safe)))

            info_lines.append(
                f"  Erosion (auto): {erosion_px}px "
                f"— target: {erosion_target:.0f}px, max safe: {erosion_max_safe:.0f}px"
            )
            info_lines.append(
                f"  Feather (auto): {feather_px}px "
                f"— target: {feather_target:.0f}px, max safe: {feather_max_safe:.0f}px"
            )

            if min_radius < vae_stride:
                info_lines.append(
                    f"  WARNING: Mask thinner than 1 VAE block ({min_radius:.1f}px < {vae_stride}px). "
                    f"Auto values are heavily constrained — consider manual mode or a wider mask."
                )

        else:  # manual
            erosion_px = int(round(erosion_blocks * vae_stride))
            feather_px = int(round(feather_blocks * vae_stride))
            info_lines.append(
                f"  Erosion (manual): {erosion_blocks} blocks = {erosion_px}px"
            )
            info_lines.append(
                f"  Feather (manual): {feather_blocks} blocks = {feather_px}px"
            )

        # --- Step 4: Erode ---
        if erosion_px > 0:
            result = mask_erode_dilate(result, -erosion_px)

        # --- Step 5: Blur (feather) ---
        if feather_px > 0:
            # mask_blur ensures odd kernel internally
            result = mask_blur(result, feather_px)
            effective_kernel = feather_px if feather_px % 2 == 1 else feather_px + 1
            info_lines.append(f"  Blur kernel: {effective_kernel}px (odd)")

        # --- Step 6: Clamp ---
        result = result.clamp(0.0, 1.0)

        # Transition summary
        transition_px = erosion_px + feather_px
        transition_blocks = transition_px / vae_stride
        transition_patches = transition_px / (vae_stride * 2)  # patch_size (1,2,2)
        info_lines.append(
            f"  Transition zone: ~{transition_px}px "
            f"({transition_blocks:.1f} VAE blocks, {transition_patches:.1f} attention patches)"
        )

        info = "\n".join(info_lines)
        print(info)

        return (result, info)


NODE_CLASS_MAPPINGS = {
    "NV_VaceMaskPrep": NV_VaceMaskPrep,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_VaceMaskPrep": "NV VACE Mask Prep",
}
