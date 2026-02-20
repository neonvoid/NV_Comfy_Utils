"""
Latent-Space Spatial Tools

5D-safe spatial crop for video latents [B, C, T, H, W].

Built-in LatentCrop is broken for 5D — it uses samples[:,:,y:to_y, x:to_x]
which confuses the temporal dimension T with spatial height H.
This node uses ellipsis indexing: samples[..., y:y+h, x:x+w]

Designed for the spatial detail enhancement workflow:
    NV_LatentSpatialCrop → LatentUpscale → SetLatentNoiseMask
        → KSampler → LatentUpscale (downscale) → LatentCompositeMasked

Supports video-length masks from 3D engines:
    - bbox_mask: bounding box mask (video-length). Bbox union across ALL frames
      handles moving subjects. Crop region = envelope of all per-frame bboxes.
    - subject_mask: character/subject silhouette (video-length). Cropped to same
      spatial region and output for downstream use as noise_mask + composite mask.

Output coordinates wire directly to LatentCompositeMasked (pixel-space x/y).

See: 2026-02-19_spatial_detail_enhancement_research.md
"""

import torch
import torch.nn.functional as F


class NV_LatentSpatialCrop:
    """
    5D-safe spatial crop for video latents.

    Crops a spatial region from a latent tensor, supporting both 4D [B,C,H,W]
    and 5D [B,C,T,H,W] video latents. Coordinates are in pixel space (÷8 internally
    for Wan's VAE stride).

    When bbox_mask is provided, the crop region is auto-derived from the bbox union
    across ALL frames (x/y/width/height inputs are ignored). This handles moving
    subjects — the crop region is the envelope covering the subject's full trajectory.

    When subject_mask is provided, it's spatially cropped to the same region and
    output for downstream use as:
      - noise_mask (via SetLatentNoiseMask) → only refine where subject is visible
      - composite mask (via LatentCompositeMasked) → only paste back subject pixels

    Temporal gating is automatic: frames where subject_mask=0 everywhere →
    KSampler preserves those frames, LatentCompositeMasked doesn't paste anything.

    Output pixel-space coordinates wire directly to LatentCompositeMasked's x/y
    inputs for the stitch-back step.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "x": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8,
                       "tooltip": "Left edge of crop region in pixel space (snapped to 8px VAE grid). "
                                  "Ignored when bbox_mask is provided."}),
                "y": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8,
                       "tooltip": "Top edge of crop region in pixel space (snapped to 8px VAE grid). "
                                  "Ignored when bbox_mask is provided."}),
                "width": ("INT", {"default": 512, "min": 8, "max": 8192, "step": 8,
                           "tooltip": "Width of crop region in pixel space. "
                                      "Ignored when bbox_mask is provided."}),
                "height": ("INT", {"default": 512, "min": 8, "max": 8192, "step": 8,
                            "tooltip": "Height of crop region in pixel space. "
                                       "Ignored when bbox_mask is provided."}),
            },
            "optional": {
                "bbox_mask": ("MASK", {
                    "tooltip": "Bounding box mask (video-length from 3D engine, segmentation, etc). "
                               "Crop region is derived from the bbox UNION across ALL frames — "
                               "the envelope covering the subject's full trajectory. "
                               "When provided, x/y/width/height inputs are ignored."
                }),
                "subject_mask": ("MASK", {
                    "tooltip": "Subject/character silhouette mask (video-length). "
                               "Cropped to the same spatial region as the latent and output "
                               "for downstream use as noise_mask (SetLatentNoiseMask) and "
                               "composite mask (LatentCompositeMasked). Frames where the subject "
                               "is absent (mask=0) are automatically preserved by KSampler."
                }),
                "padding": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 8,
                             "tooltip": "Extra padding around crop region in pixel space. "
                                        "Provides surrounding context for the subject through attention. "
                                        "Recommended: ~50%% of subject bbox size for generous context."}),
            }
        }

    RETURN_TYPES = ("LATENT", "MASK", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("latent", "cropped_mask", "crop_x", "crop_y", "crop_width", "crop_height")
    FUNCTION = "execute"
    CATEGORY = "NV_Utils"
    DESCRIPTION = (
        "5D-safe spatial crop for video latents. Built-in LatentCrop is broken "
        "for 5D video tensors (confuses temporal dim T with spatial H). "
        "Supports video-length bbox_mask (union across ALL frames for moving subjects) "
        "and subject_mask (cropped to same region for noise_mask + composite mask). "
        "Output coordinates wire directly to LatentCompositeMasked for stitching back."
    )

    def execute(self, latent, x, y, width, height, bbox_mask=None, subject_mask=None, padding=0):
        samples = latent["samples"]
        vae_stride = 8

        # Spatial dimensions in pixel space (last 2 dims of latent × stride)
        spatial_h = samples.shape[-2] * vae_stride
        spatial_w = samples.shape[-1] * vae_stride

        # Auto-derive bbox from bbox_mask union across ALL frames
        if bbox_mask is not None:
            # bbox_mask shape: [T, H, W] or [1, H, W] or [H, W]
            # Process ALL frames to get the envelope bbox
            if bbox_mask.dim() == 2:
                # Single frame [H, W] — treat as static
                m = bbox_mask
            else:
                # [T, H, W] or [B, H, W] — union across all frames
                # Max-reduce across frame dim to get the envelope
                m = bbox_mask.max(dim=0).values

            non_zero = torch.nonzero(m > 0.01)
            if non_zero.numel() > 0:
                y_min = non_zero[:, -2].min().item()
                y_max = non_zero[:, -2].max().item()
                x_min = non_zero[:, -1].min().item()
                x_max = non_zero[:, -1].max().item()

                # Scale if mask resolution differs from latent spatial resolution
                mask_h, mask_w = m.shape[-2], m.shape[-1]
                if mask_h != spatial_h or mask_w != spatial_w:
                    scale_y = spatial_h / mask_h
                    scale_x = spatial_w / mask_w
                    x_min = int(x_min * scale_x)
                    x_max = int(x_max * scale_x)
                    y_min = int(y_min * scale_y)
                    y_max = int(y_max * scale_y)

                x = x_min
                y = y_min
                width = x_max - x_min + 1
                height = y_max - y_min + 1
                num_frames = bbox_mask.shape[0] if bbox_mask.dim() == 3 else 1
                print(f"[NV_LatentSpatialCrop] Bbox union across {num_frames} frames: "
                      f"x={x}, y={y}, {width}x{height}")

        # Apply padding
        if padding > 0:
            x = max(0, x - padding)
            y = max(0, y - padding)
            # Expand width/height but clamp to image bounds
            width = min(spatial_w - x, width + 2 * padding)
            height = min(spatial_h - y, height + 2 * padding)

        # Snap to VAE grid (8px boundaries)
        x = (x // vae_stride) * vae_stride
        y = (y // vae_stride) * vae_stride
        width = max(vae_stride, ((width + vae_stride - 1) // vae_stride) * vae_stride)
        height = max(vae_stride, ((height + vae_stride - 1) // vae_stride) * vae_stride)

        # Clamp to image bounds
        if x + width > spatial_w:
            width = spatial_w - x
        if y + height > spatial_h:
            height = spatial_h - y
        x = max(0, min(x, spatial_w - vae_stride))
        y = max(0, min(y, spatial_h - vae_stride))

        # Convert to latent coordinates
        x_l = x // vae_stride
        y_l = y // vae_stride
        w_l = width // vae_stride
        h_l = height // vae_stride

        # 5D-safe spatial crop (ellipsis skips B, C, and optional T dims)
        cropped = samples[..., y_l:y_l + h_l, x_l:x_l + w_l].clone()

        print(f"[NV_LatentSpatialCrop] Crop: pixel ({x},{y}) {width}x{height} -> "
              f"latent ({x_l},{y_l}) {w_l}x{h_l}")
        print(f"[NV_LatentSpatialCrop] Input shape: {list(samples.shape)} -> "
              f"Output shape: {list(cropped.shape)}")

        out = latent.copy()
        out["samples"] = cropped

        # Also crop noise_mask if present (ellipsis handles broadcast dims safely)
        if "noise_mask" in latent:
            nm = latent["noise_mask"]
            out["noise_mask"] = nm[..., y_l:y_l + h_l, x_l:x_l + w_l].clone()

        # Crop subject_mask to same spatial region (pixel space crop)
        cropped_mask = torch.zeros(1, height, width)  # default empty mask
        if subject_mask is not None:
            # subject_mask shape: [T, H, W] or [1, H, W] or [H, W]
            sm = subject_mask
            if sm.dim() == 2:
                sm = sm.unsqueeze(0)  # [H, W] -> [1, H, W]

            # Scale subject_mask if resolution differs from pixel space
            sm_h, sm_w = sm.shape[-2], sm.shape[-1]
            if sm_h != spatial_h or sm_w != spatial_w:
                # Resize to match pixel-space resolution
                # F.interpolate needs [N, C, H, W] — add batch+channel dims
                sm_4d = sm.unsqueeze(1)  # [T, 1, H, W]
                sm_4d = F.interpolate(sm_4d, size=(spatial_h, spatial_w),
                                      mode='bilinear', align_corners=False)
                sm = sm_4d.squeeze(1)  # back to [T, H, W]
                print(f"[NV_LatentSpatialCrop] Subject mask resized: "
                      f"{sm_h}x{sm_w} -> {spatial_h}x{spatial_w}")

            # Spatial crop in pixel space
            cropped_mask = sm[..., y:y + height, x:x + width].clone()
            print(f"[NV_LatentSpatialCrop] Subject mask cropped: "
                  f"{list(sm.shape)} -> {list(cropped_mask.shape)}")

        # Output pixel-space coordinates (wire to LatentCompositeMasked x/y)
        return (out, cropped_mask, x, y, width, height)


# Node registration
NODE_CLASS_MAPPINGS = {
    "NV_LatentSpatialCrop": NV_LatentSpatialCrop,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_LatentSpatialCrop": "NV Latent Spatial Crop",
}
