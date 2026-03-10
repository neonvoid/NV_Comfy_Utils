"""
NV Pixel-to-Latent Stitcher — DEPRECATED / BROKEN

DO NOT USE. This node has a fundamental flaw: InpaintCrop2 crops per-frame at
varying sizes/positions (stabilization), then resizes each to 512x512 at different
magnifications. This node computes a static union bbox and tries to paste the
KSampler output at that union size. But the content was generated at per-frame
magnification, not union magnification → face pasted at wrong scale (~30% too large).

5D latent paste is static (same crop for all frames). Per-frame pixel crops have
varying spatial layouts. These are structurally incompatible.

CORRECT APPROACH: Use NV_LatentInpaintCrop for latent crop+stitch (static union,
consistent scale). Use InpaintCrop2 only for pixel-space VACE conditioning.

Kept for reference only. Will print a deprecation warning if executed.
"""

import torch
import comfy.model_management

from .latent_inpaint_crop import snap_to_vae_grid, VAE_STRIDE
from .latent_constants import LATENT_SAFE_KEYS
from .mask_ops import rescale_mask


class NV_PixelToLatentStitcher:
    """Convert pixel STITCHER to LATENT_STITCHER for latent-space compositing.

    Takes the crop coordinates from NV_InpaintCrop2 (per-frame, stabilized)
    and a full-scene latent, computes a union bbox, and builds a LATENT_STITCHER
    compatible with NV_LatentInpaintStitch.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stitcher": ("STITCHER", {
                    "tooltip": "Pixel-space stitcher from NV Inpaint Crop v2."
                }),
                "latent": ("LATENT", {
                    "tooltip": "Full-scene latent [B,C,T,H,W] from VAE encode. "
                               "Must cover the entire original image, not a crop."
                }),
            },
        }

    RETURN_TYPES = ("LATENT_STITCHER", "STRING")
    RETURN_NAMES = ("stitcher", "info")
    FUNCTION = "execute"
    CATEGORY = "NV_Utils/Inpaint"
    DESCRIPTION = (
        "Convert pixel STITCHER (from NV Inpaint Crop v2) to LATENT_STITCHER "
        "for latent-space compositing with NV Latent Inpaint Stitch. "
        "Uses pixel crop coordinates as the single source of truth."
    )

    def execute(self, stitcher, latent):
        import warnings
        warnings.warn(
            "[NV_PixelToLatentStitcher] DEPRECATED — this node produces wrong scale/position. "
            "Use NV_LatentInpaintCrop for latent crop+stitch instead. "
            "InpaintCrop2 should only be used for pixel-space VACE conditioning.",
            DeprecationWarning, stacklevel=2
        )
        print(
            "\n⚠️  [NV_PixelToLatentStitcher] WARNING: This node is BROKEN and DEPRECATED.\n"
            "    Per-frame pixel crops have varying magnification that cannot map to static\n"
            "    latent union paste → face will be pasted at WRONG SCALE.\n"
            "    Use NV_LatentInpaintCrop for latent crop+stitch instead.\n"
        )

        intermediate = comfy.model_management.intermediate_device()
        samples = latent["samples"]
        info_lines = []

        if samples.ndim != 5:
            raise ValueError(
                f"[NV_PixelToLatentStitcher] Expected 5D latent [B,C,T,H,W], "
                f"got {samples.ndim}D {list(samples.shape)}"
            )

        spatial_h_px = samples.shape[-2] * VAE_STRIDE
        spatial_w_px = samples.shape[-1] * VAE_STRIDE
        info_lines.append(
            f"Full latent: {list(samples.shape)} "
            f"({spatial_w_px}x{spatial_h_px}px)"
        )

        # --- Compute per-frame crop-in-original coordinates ---
        total_frames = stitcher['total_frames']
        ctc_x = stitcher['cropped_to_canvas_x']
        ctc_y = stitcher['cropped_to_canvas_y']
        ctc_w = stitcher['cropped_to_canvas_w']
        ctc_h = stitcher['cropped_to_canvas_h']
        cto_x = stitcher['canvas_to_orig_x']
        cto_y = stitcher['canvas_to_orig_y']

        # Transform canvas-space coords to original-image-space
        orig_xs = [ctc_x[i] - cto_x[i] for i in range(total_frames)]
        orig_ys = [ctc_y[i] - cto_y[i] for i in range(total_frames)]
        orig_ws = [ctc_w[i] for i in range(total_frames)]
        orig_hs = [ctc_h[i] for i in range(total_frames)]

        # --- Union bbox across all frames ---
        union_x = min(orig_xs)
        union_y = min(orig_ys)
        union_r = max(orig_xs[i] + orig_ws[i] for i in range(total_frames))
        union_b = max(orig_ys[i] + orig_hs[i] for i in range(total_frames))
        union_w = union_r - union_x
        union_h = union_b - union_y

        info_lines.append(
            f"Union bbox (px): ({union_x},{union_y}) {union_w}x{union_h}"
        )

        # --- Derive target dims from blend mask shape ---
        # Must happen BEFORE snap so we can grow the union bbox to match target aspect.
        # The blend mask is at InpaintCrop2's output resolution (target size).
        target_w_latent = None
        target_h_latent = None
        target_w_px = None
        target_h_px = None
        resize_method = None

        if stitcher.get('cropped_mask_for_blend') and len(stitcher['cropped_mask_for_blend']) > 0:
            mask_sample = stitcher['cropped_mask_for_blend'][0]
            target_h_px = mask_sample.shape[-2]
            target_w_px = mask_sample.shape[-1]

            # Snap target to VAE grid
            target_w_px = max(VAE_STRIDE, ((target_w_px + VAE_STRIDE - 1) // VAE_STRIDE) * VAE_STRIDE)
            target_h_px = max(VAE_STRIDE, ((target_h_px + VAE_STRIDE - 1) // VAE_STRIDE) * VAE_STRIDE)

            target_w_latent = target_w_px // VAE_STRIDE
            target_h_latent = target_h_px // VAE_STRIDE
            resize_method = stitcher.get('resize_algorithm', 'bislerp')

            info_lines.append(
                f"Target (from blend mask): {target_w_px}x{target_h_px}px | "
                f"latent: {target_w_latent}x{target_h_latent} ({resize_method})"
            )

        # --- Grow union bbox to match target aspect ratio ---
        # The pixel InpaintCrop2 may have grown its bbox (crop_aspect) before cropping,
        # but ctc-cto coordinates reflect the raw bbox. Without matching that growth here,
        # the latent crop gets resized to the target at the wrong aspect ratio, squishing
        # the content during KSampler denoising.
        if target_w_px is not None and target_h_px is not None:
            target_aspect = target_w_px / target_h_px
            bbox_aspect = union_w / union_h

            if abs(bbox_aspect - target_aspect) > 0.01:  # only grow if meaningfully different
                if bbox_aspect < target_aspect:
                    # Need wider — grow width
                    new_w = union_h * target_aspect
                    union_x -= (new_w - union_w) / 2.0
                    union_w = new_w
                else:
                    # Need taller — grow height
                    new_h = union_w / target_aspect
                    union_y -= (new_h - union_h) / 2.0
                    union_h = new_h

                # Shift to keep in bounds (no canvas expansion in latent space)
                if union_x < 0:
                    union_x = 0.0
                elif union_x + union_w > spatial_w_px:
                    union_x = float(spatial_w_px) - union_w
                if union_y < 0:
                    union_y = 0.0
                elif union_y + union_h > spatial_h_px:
                    union_y = float(spatial_h_px) - union_h

                # Hard clamp if grown bbox exceeds frame
                union_x = max(0.0, union_x)
                union_y = max(0.0, union_y)
                union_w = min(union_w, float(spatial_w_px) - union_x)
                union_h = min(union_h, float(spatial_h_px) - union_y)

                info_lines.append(
                    f"Aspect growth: target ar={target_aspect:.3f}, "
                    f"bbox ar={bbox_aspect:.3f} -> grown ({union_x:.0f},{union_y:.0f}) "
                    f"{union_w:.0f}x{union_h:.0f}px"
                )

        # --- Snap to VAE grid ---
        cx_px, cy_px, cw_px, ch_px = snap_to_vae_grid(
            union_x, union_y, union_w, union_h,
            spatial_h_px, spatial_w_px
        )
        cx_l = cx_px // VAE_STRIDE
        cy_l = cy_px // VAE_STRIDE
        cw_l = cw_px // VAE_STRIDE
        ch_l = ch_px // VAE_STRIDE

        info_lines.append(
            f"Snapped crop (px): ({cx_px},{cy_px}) {cw_px}x{ch_px} | "
            f"latent: ({cx_l},{cy_l}) {cw_l}x{ch_l}"
        )

        # --- Build blend mask (union of per-frame masks) ---
        blend_mask = None
        if stitcher.get('cropped_mask_for_blend') and len(stitcher['cropped_mask_for_blend']) > 0:
            # Stack per-frame masks and take max (union)
            masks = torch.stack([m for m in stitcher['cropped_mask_for_blend']])
            union_mask = masks.max(dim=0).values  # [H_target, W_target]
            union_mask = union_mask.unsqueeze(0)   # [1, H_target, W_target]

            # Resize from target resolution to native crop pixel resolution
            # (LatentInpaintStitch will downscale to latent res internally)
            if cw_px != union_mask.shape[-1] or ch_px != union_mask.shape[-2]:
                blend_mask = rescale_mask(union_mask, cw_px, ch_px, 'bilinear')
            else:
                blend_mask = union_mask

            info_lines.append(
                f"Blend mask: union of {len(stitcher['cropped_mask_for_blend'])} frames "
                f"-> [{blend_mask.shape[-2]}x{blend_mask.shape[-1]}]"
            )

        # --- Extract safe_keys from input latent ---
        safe_keys = {}
        for key in LATENT_SAFE_KEYS:
            if key in latent:
                safe_keys[key] = latent[key]

        # --- Build LATENT_STITCHER ---
        latent_stitcher = {
            'original_samples': samples.to(intermediate),
            'crop_x_latent': cx_l,
            'crop_y_latent': cy_l,
            'crop_w_latent': cw_l,
            'crop_h_latent': ch_l,
            'crop_x_pixel': cx_px,
            'crop_y_pixel': cy_px,
            'crop_w_pixel': cw_px,
            'crop_h_pixel': ch_px,
            'safe_keys': safe_keys,
            'target_w_latent': target_w_latent,
            'target_h_latent': target_h_latent,
            'resize_method': resize_method,
        }

        if blend_mask is not None:
            latent_stitcher['blend_mask'] = blend_mask.to(intermediate)

        info_str = "\n".join(info_lines)
        print(f"[NV_PixelToLatentStitcher] {info_str}")

        return (latent_stitcher, info_str)


# =============================================================================
# Node Registration
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "NV_PixelToLatentStitcher": NV_PixelToLatentStitcher,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_PixelToLatentStitcher": "NV Pixel-to-Latent Stitcher",
}
