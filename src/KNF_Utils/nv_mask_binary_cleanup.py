"""
NV Mask Binary Cleanup - Threshold + binary fill-holes + CC cleanup +
optional final morph + optional re-feather. Designed to follow soft-mask
producers (NV_MaskUnion guided refine, NV_GuidedFilterMask, raw SAM3,
post-blur masks) and convert depth-map-like soft outputs into clean
binary masks usable downstream.

Pipeline (per frame):
    threshold to binary
  → optional scipy.ndimage.binary_fill_holes (TRUE binary fill, fills
    any-sized internal holes regardless of kernel — different from
    greyscale closing)
  → optional connected-component cleanup (drop CCs < min_area, keep
    only N largest by area)
  → optional final dilate/erode (sized via mask_erode_dilate)
  → optional final Gaussian feather (mask_blur, gradient-preserving)

When NOT to use: if your downstream consumer wants soft alpha gradient
information preserved (e.g. blend feathering for InpaintStitch2 with
`alpha` mode), this node will destroy that information by binarizing.
Use NV_MaskUnion's `post_smooth_px` or NV_GuidedFilterMask instead.
"""

import cv2
import numpy as np
import scipy.ndimage
import torch

from .guided_filter import refine_mask
from .mask_ops import mask_blur, mask_erode_dilate


class NV_MaskBinaryCleanup:
    """Threshold + fill-holes + CC cleanup + optional re-feather a soft mask."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK", {
                    "tooltip": "Input mask [B,H,W] in [0,1]. 2D [H,W] auto-promoted."
                }),
                "binarize_threshold": ("FLOAT", {
                    "default": 0.5, "min": 0.01, "max": 0.99, "step": 0.01,
                    "tooltip": (
                        "Pixels above this value become foreground (1.0); below become "
                        "background (0.0). 0.5 is the standard threshold; lower (0.3) if "
                        "your soft mask is generally faint, higher (0.7) if it's bright "
                        "with noisy halos. Picks the cut for the depth-map-like grey gradient."
                    ),
                }),
                "fill_holes": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "True binary fill of internal background-blobs surrounded by "
                        "foreground (scipy.ndimage.binary_fill_holes). Different from "
                        "greyscale closing — fills ANY-sized holes regardless of kernel "
                        "radius. Eliminates 'depth map / grey gaps' inside the subject "
                        "after thresholding."
                    ),
                }),
                "min_component_area_px": ("INT", {
                    "default": 0, "min": 0, "max": 10000, "step": 10,
                    "tooltip": (
                        "Drop connected components smaller than this area (pixels). 0 = "
                        "keep all. 100-500 typical for killing speckle / hair-frizz noise; "
                        "5000+ if you want only big regions like a full subject silhouette."
                    ),
                }),
                "keep_n_largest": ("INT", {
                    "default": 0, "min": 0, "max": 10, "step": 1,
                    "tooltip": (
                        "Keep only the N largest connected components by area; 0 = keep "
                        "all components that pass min_component_area_px. Useful for "
                        "single-subject shots where there should only be one big blob "
                        "(set to 1)."
                    ),
                }),
                "final_dilate_px": ("INT", {
                    "default": 0, "min": -64, "max": 64, "step": 1,
                    "tooltip": (
                        "Final morphological adjustment in pixels: positive = dilate "
                        "(expand mask), negative = erode (shrink mask). Operates on the "
                        "cleaned binary mask. Uses GPU max_pool2d when CUDA available."
                    ),
                }),
                "final_feather_px": ("INT", {
                    "default": 0, "min": 0, "max": 32, "step": 1,
                    "tooltip": (
                        "Optional Gaussian softening on the binary output for downstream "
                        "compositing. 0 = pure binary out (clean step edge). 2-4 = subtle "
                        "feather for blend masks. 8+ = soft falloff for VACE conditioning. "
                        "Runs AFTER guided_image refinement (if wired)."
                    ),
                }),
            },
            "optional": {
                "guided_image": ("IMAGE", {
                    "tooltip": (
                        "Optional source IMAGE [B,H,W,3] for edge-aware refinement of the "
                        "CLEANED BINARY mask. Runs guided filter (He et al. 2013) — snaps "
                        "the binary boundary to luminance gradients. More effective here "
                        "than on a soft input mask: clean binary input + image guide = sharp "
                        "image-aligned edges (especially for hair). Spatial dims must match "
                        "the mask; batch must match mask batch or be 1 (broadcast)."
                    ),
                }),
                "guided_radius": ("INT", {
                    "default": 0, "min": 0, "max": 64, "step": 1,
                    "tooltip": (
                        "Guided filter window radius. 0 = auto (max(4, min(H,W) // 64) — "
                        "~16 at 1080p, ~33 at 4K). Smaller = follows fine detail more "
                        "tightly; larger = smoother edges, may over-relax. Only used "
                        "when guided_image is wired."
                    ),
                }),
                "guided_eps": ("FLOAT", {
                    "default": 0.001, "min": 0.0001, "max": 0.1, "step": 0.0001,
                    "tooltip": (
                        "Guided filter regularization. Lower = sharper edge tracking, "
                        "higher = smoother. 0.001 default works well for hair-edge "
                        "alignment. Only used when guided_image is wired."
                    ),
                }),
                "guided_strength": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": (
                        "Lerp between cleaned binary (0.0) and guided-refined (1.0). "
                        "0.7 default avoids over-erosion when the guide is unreliable "
                        "(motion blur, low contrast). 0.0 short-circuits the refine block "
                        "entirely (saves compute). Only used when guided_image is wired."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("cleaned_mask",)
    FUNCTION = "cleanup"
    CATEGORY = "NV_Utils/Mask"
    DESCRIPTION = (
        "Binarize, fill internal holes, drop CC noise, optional final morph + feather. "
        "Designed to follow soft-mask producers (NV_MaskUnion guided refine, raw SAM3) "
        "and convert depth-map-like soft outputs into clean binary masks."
    )

    def cleanup(
        self,
        mask,
        binarize_threshold=0.5,
        fill_holes=True,
        min_component_area_px=0,
        keep_n_largest=0,
        final_dilate_px=0,
        final_feather_px=0,
        guided_image=None,
        guided_radius=0,
        guided_eps=0.001,
        guided_strength=0.7,
    ):
        # --- Normalize input -------------------------------------------------
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        elif mask.ndim != 3:
            raise ValueError(
                f"[NV_MaskBinaryCleanup] mask must be 2D [H,W] or 3D [B,H,W], "
                f"got shape {tuple(mask.shape)}."
            )
        device = mask.device
        mask_clean = torch.nan_to_num(mask, nan=0.0, posinf=1.0, neginf=0.0)

        # --- Threshold to binary ---------------------------------------------
        binary_t = (mask_clean > float(binarize_threshold)).to(torch.uint8)

        # --- Per-frame CPU work: fill_holes + CC cleanup --------------------
        # scipy.ndimage and cv2 are CPU-only; one .cpu() round-trip, batch loop.
        do_cc = (min_component_area_px > 0) or (keep_n_largest > 0)
        if fill_holes or do_cc:
            binary_cpu = binary_t.cpu().numpy()
            cleaned_frames = np.empty_like(binary_cpu)
            for i in range(binary_cpu.shape[0]):
                frame = binary_cpu[i]

                if fill_holes:
                    frame = scipy.ndimage.binary_fill_holes(frame).astype(np.uint8)

                if do_cc:
                    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                        frame, connectivity=8, ltype=cv2.CV_32S
                    )
                    # Background is label 0; real CCs are 1..num_labels-1.
                    if num_labels > 1:
                        areas = stats[1:, cv2.CC_STAT_AREA]
                        component_ids = np.arange(1, num_labels)

                        if min_component_area_px > 0:
                            keep = areas >= int(min_component_area_px)
                            component_ids = component_ids[keep]
                            areas = areas[keep]

                        if keep_n_largest > 0 and len(component_ids) > int(keep_n_largest):
                            top_idx = np.argsort(-areas)[:int(keep_n_largest)]
                            component_ids = component_ids[top_idx]

                        if len(component_ids) > 0:
                            keep_set = component_ids.astype(np.int32)
                            frame = np.isin(labels, keep_set).astype(np.uint8)
                        else:
                            frame = np.zeros_like(frame)
                    else:
                        # Only background present after fill_holes; preserve.
                        pass

                cleaned_frames[i] = frame
            binary_t = torch.from_numpy(cleaned_frames).to(device=device)

        result = binary_t.to(dtype=torch.float32)

        # --- Final dilate/erode (coarse) ------------------------------------
        if final_dilate_px != 0:
            result = mask_erode_dilate(result, int(final_dilate_px))

        # --- Optional guided refine (fine, image-edge aligned) --------------
        # Operates on the CLEANED BINARY mask, not the original soft input — the
        # filter is sharper here because the input has crisp edges, so the
        # filter only adjusts edge POSITION instead of redistributing soft mass.
        # Short-circuit on strength=0 to skip the full guided-filter cost.
        if guided_image is not None and float(guided_strength) > 0.0:
            self._validate_guide(guided_image, result)
            guide = guided_image.to(device=result.device, dtype=result.dtype)
            if guide.shape[0] == 1 and result.shape[0] > 1:
                guide = guide.expand(result.shape[0], -1, -1, -1).contiguous()
            radius = (
                int(guided_radius) if guided_radius > 0
                else max(4, min(result.shape[1], result.shape[2]) // 64)
            )
            result = refine_mask(
                result, guide,
                radius=radius,
                eps=float(guided_eps),
                strength=float(guided_strength),
                mode="color",
            )

        # --- Final feather (Gaussian softening) -----------------------------
        if final_feather_px > 0:
            result = mask_blur(result, int(final_feather_px))

        # ComfyUI MASK convention: float32, [0,1]
        return (result.clamp(0.0, 1.0).to(dtype=torch.float32),)

    @staticmethod
    def _validate_guide(guided_image, mask):
        """Fail fast on shape/batch/spatial mismatch in the IMAGE guide."""
        if guided_image.dim() != 4 or guided_image.shape[-1] != 3:
            raise ValueError(
                f"[NV_MaskBinaryCleanup] guided_image must be IMAGE [B,H,W,3], "
                f"got shape {tuple(guided_image.shape)}."
            )
        if (guided_image.shape[1] != mask.shape[1]
                or guided_image.shape[2] != mask.shape[2]):
            raise ValueError(
                f"[NV_MaskBinaryCleanup] guided_image spatial dims "
                f"{tuple(guided_image.shape[1:3])} must match mask "
                f"{tuple(mask.shape[1:])}."
            )
        if guided_image.shape[0] != mask.shape[0] and guided_image.shape[0] != 1:
            raise ValueError(
                f"[NV_MaskBinaryCleanup] guided_image batch={guided_image.shape[0]} "
                f"must match mask batch={mask.shape[0]} or be 1 (broadcast)."
            )


NODE_CLASS_MAPPINGS = {
    "NV_MaskBinaryCleanup": NV_MaskBinaryCleanup,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_MaskBinaryCleanup": "NV Mask Binary Cleanup",
}
