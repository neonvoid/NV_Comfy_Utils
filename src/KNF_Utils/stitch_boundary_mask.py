"""
NV Stitch Boundary Mask — Generate a gradient mask along stitch boundaries for seam harmonization.

Given a LATENT_STITCHER (from NV_LatentInpaintCrop), generates a gradient mask that
straddles the crop boundary. Feed this mask to SetLatentNoiseMask, then run a low-denoise
KSampler pass to let WAN 2.2's native mask handling harmonize the seam.

Pipeline:
    NV_LatentInpaintStitch (hard paste) → NV_StitchBoundaryMask → SetLatentNoiseMask
    → KSampler (denoise 0.3-0.5) → StreamingVAEDecode
"""

import torch
import torch.nn.functional as F


VAE_STRIDE = 8


class NV_StitchBoundaryMask:
    """Generate a gradient mask along stitch boundaries for boundary diffusion seam repair.

    Reads crop coordinates from LATENT_STITCHER and creates a narrow gradient strip
    centered on the crop rectangle edge. WAN 2.2 native mask handling (scale_latent_inpaint,
    process_timestep, extra_conds) uses this to harmonize the stitch boundary.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stitcher": ("LATENT_STITCHER", {
                    "tooltip": "From NV Latent Inpaint Crop. Contains crop coordinates "
                               "that define where the stitch boundary is."
                }),
                "strip_width": ("INT", {
                    "default": 32, "min": 8, "max": 128, "step": 8,
                    "tooltip": "Total width of boundary strip in pixels. "
                               "32px = 4 latent cells — good default for character inpaint. "
                               "Wider strips give the model more context for harmonization."
                }),
                "blur_radius": ("INT", {
                    "default": 16, "min": 0, "max": 64, "step": 1,
                    "tooltip": "Gaussian blur for gradient falloff in pixels. "
                               "Creates smooth transition from 1.0 (center of strip) to 0.0 (edges). "
                               "0 = hard binary strip (not recommended)."
                }),
                "inner_ratio": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1,
                    "tooltip": "Fraction of strip_width that extends inward (into the crop region). "
                               "0.5 = symmetric (half inside, half outside crop edge). "
                               "1.0 = strip entirely inside crop. 0.0 = entirely outside."
                }),
            },
        }

    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("mask", "info")
    FUNCTION = "execute"
    CATEGORY = "NV_Utils/Inpaint"
    DESCRIPTION = (
        "Generate a gradient mask along the stitch boundary from LATENT_STITCHER coordinates. "
        "Use with SetLatentNoiseMask + low-denoise KSampler to harmonize inpaint seams. "
        "WAN 2.2 native mask handling makes this a semantic harmonization pass, not just blending."
    )

    def execute(self, stitcher, strip_width=32, blur_radius=16, inner_ratio=0.5):
        # Read crop coordinates (pixel space)
        cx = stitcher['crop_x_pixel']
        cy = stitcher['crop_y_pixel']
        cw = stitcher['crop_w_pixel']
        ch = stitcher['crop_h_pixel']

        # Full-frame dimensions from original_samples
        orig = stitcher['original_samples']
        if orig.ndim == 5:
            _, _, T, H_l, W_l = orig.shape
        else:
            _, _, H_l, W_l = orig.shape
            T = 1
        frame_h = H_l * VAE_STRIDE
        frame_w = W_l * VAE_STRIDE

        # Compute inner/outer offsets from the crop edge
        inner_px = max(1, int(strip_width * inner_ratio))
        outer_px = max(1, strip_width - inner_px)

        # Build binary strip mask at pixel resolution
        mask = torch.zeros(1, frame_h, frame_w)

        # Inner rectangle (crop edge minus inner offset)
        inner_top = max(0, cy + inner_px)
        inner_bot = min(frame_h, cy + ch - inner_px)
        inner_left = max(0, cx + inner_px)
        inner_right = min(frame_w, cx + cw - inner_px)

        # Outer rectangle (crop edge plus outer offset)
        outer_top = max(0, cy - outer_px)
        outer_bot = min(frame_h, cy + ch + outer_px)
        outer_left = max(0, cx - outer_px)
        outer_right = min(frame_w, cx + cw + outer_px)

        # Fill the outer rectangle with 1.0
        mask[0, outer_top:outer_bot, outer_left:outer_right] = 1.0

        # Clear the inner rectangle (leave only the strip)
        if inner_top < inner_bot and inner_left < inner_right:
            mask[0, inner_top:inner_bot, inner_left:inner_right] = 0.0

        # Apply Gaussian blur for gradient falloff
        if blur_radius > 0:
            kernel_size = blur_radius * 2 + 1
            # F.conv2d with Gaussian kernel
            sigma = blur_radius / 3.0
            x_coord = torch.arange(kernel_size, dtype=torch.float32) - blur_radius
            kernel_1d = torch.exp(-0.5 * (x_coord / sigma) ** 2)
            kernel_1d = kernel_1d / kernel_1d.sum()

            # Separable 2D Gaussian blur
            mask_4d = mask.unsqueeze(0)  # [1, 1, H, W]
            # Horizontal pass
            kh = kernel_1d.view(1, 1, 1, -1)
            mask_4d = F.pad(mask_4d, (blur_radius, blur_radius, 0, 0), mode='reflect')
            mask_4d = F.conv2d(mask_4d, kh)
            # Vertical pass
            kv = kernel_1d.view(1, 1, -1, 1)
            mask_4d = F.pad(mask_4d, (0, 0, blur_radius, blur_radius), mode='reflect')
            mask_4d = F.conv2d(mask_4d, kv)
            mask = mask_4d.squeeze(0).clamp(0.0, 1.0)

        # Compute coverage stats
        total_pixels = frame_h * frame_w
        active_pixels = (mask > 0.01).sum().item()
        coverage_pct = 100.0 * active_pixels / total_pixels

        info_lines = [
            f"Frame: {frame_w}x{frame_h}px ({W_l}x{H_l} latent)",
            f"Crop: ({cx},{cy}) {cw}x{ch}px",
            f"Strip: {strip_width}px total (inner={inner_px}px, outer={outer_px}px)",
            f"Blur: {blur_radius}px (sigma={blur_radius/3.0:.1f})",
            f"Coverage: {active_pixels}/{total_pixels} pixels ({coverage_pct:.1f}%)",
        ]
        info = "\n".join(info_lines)
        print(f"[NV_StitchBoundaryMask] {info}")

        return (mask, info)


NODE_CLASS_MAPPINGS = {
    "NV_StitchBoundaryMask": NV_StitchBoundaryMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_StitchBoundaryMask": "NV Stitch Boundary Mask",
}
