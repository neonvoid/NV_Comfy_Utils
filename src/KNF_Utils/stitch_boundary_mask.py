"""
NV Stitch Boundary Mask — Generate a gradient mask along stitch boundaries for seam harmonization.

Works with both pipeline types:
  - Latent path: LATENT_STITCHER from NV_LatentInpaintCrop (static union bbox)
  - Pixel path:  STITCHER from NV_InpaintCrop2 (per-frame coords → union computed here)

Feed the output mask to SetLatentNoiseMask, then run a low-denoise KSampler pass
to let WAN 2.2's native mask handling harmonize the seam.

Pixel path pipeline (e.g. Kling API output):
    InpaintStitch2 (pixel composite) → VAE Encode → NV_StitchBoundaryMask
    → SetLatentNoiseMask → KSampler (denoise 0.3-0.5) → VAE Decode

Latent path pipeline:
    NV_LatentInpaintStitch (hard paste) → NV_StitchBoundaryMask
    → SetLatentNoiseMask → KSampler (denoise 0.3-0.5) → StreamingVAEDecode
"""

import torch
import torch.nn.functional as F


VAE_STRIDE = 8


def _gaussian_blur_mask(mask, blur_radius):
    """Apply separable Gaussian blur to a [1, H, W] mask tensor."""
    kernel_size = blur_radius * 2 + 1
    sigma = blur_radius / 3.0
    x_coord = torch.arange(kernel_size, dtype=torch.float32) - blur_radius
    kernel_1d = torch.exp(-0.5 * (x_coord / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()

    mask_4d = mask.unsqueeze(0)  # [1, 1, H, W]
    # Horizontal pass
    kh = kernel_1d.view(1, 1, 1, -1)
    mask_4d = F.pad(mask_4d, (blur_radius, blur_radius, 0, 0), mode='reflect')
    mask_4d = F.conv2d(mask_4d, kh)
    # Vertical pass
    kv = kernel_1d.view(1, 1, -1, 1)
    mask_4d = F.pad(mask_4d, (0, 0, blur_radius, blur_radius), mode='reflect')
    mask_4d = F.conv2d(mask_4d, kv)
    return mask_4d.squeeze(0).clamp(0.0, 1.0)


def _build_boundary_strip(frame_h, frame_w, cx, cy, cw, ch, inner_px, outer_px):
    """Build a binary ring mask straddling a rectangle boundary.

    Returns [1, frame_h, frame_w] float tensor.
    """
    mask = torch.zeros(1, frame_h, frame_w)

    # Outer rectangle (crop edge + outer offset)
    outer_top = max(0, cy - outer_px)
    outer_bot = min(frame_h, cy + ch + outer_px)
    outer_left = max(0, cx - outer_px)
    outer_right = min(frame_w, cx + cw + outer_px)
    mask[0, outer_top:outer_bot, outer_left:outer_right] = 1.0

    # Inner rectangle (crop edge - inner offset) — carve out the hollow center
    inner_top = max(0, cy + inner_px)
    inner_bot = min(frame_h, cy + ch - inner_px)
    inner_left = max(0, cx + inner_px)
    inner_right = min(frame_w, cx + cw - inner_px)
    if inner_top < inner_bot and inner_left < inner_right:
        mask[0, inner_top:inner_bot, inner_left:inner_right] = 0.0

    return mask


def _extract_pixel_stitcher_union(stitcher):
    """Compute union bbox from pixel STITCHER per-frame coordinates.

    Returns (x, y, w, h) in pixel space covering all frames.
    """
    xs = stitcher['canvas_to_orig_x']
    ys = stitcher['canvas_to_orig_y']
    ws = stitcher['canvas_to_orig_w']
    hs = stitcher['canvas_to_orig_h']

    x_min = min(xs)
    y_min = min(ys)
    x_max = max(x + w for x, w in zip(xs, ws))
    y_max = max(y + h for y, h in zip(ys, hs))

    return int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)


class NV_StitchBoundaryMask:
    """Generate a gradient mask along stitch boundaries for boundary diffusion seam repair.

    Accepts either LATENT_STITCHER (from latent crop path) or pixel STITCHER
    (from InpaintCrop2, e.g. Kling API workflows). When using pixel STITCHER,
    connect the VAE-encoded stitched frame as the latent input for frame dimensions.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
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
            "optional": {
                "latent_stitcher": ("LATENT_STITCHER", {
                    "tooltip": "From NV Latent Inpaint Crop. Contains static crop coordinates "
                               "and original_samples for frame dimensions. Use for latent-path workflows."
                }),
                "pixel_stitcher": ("STITCHER", {
                    "tooltip": "From NV Inpaint Crop v2. Contains per-frame crop coordinates "
                               "(union computed automatically). Use for pixel-path workflows (e.g. Kling API). "
                               "Requires latent input for frame dimensions."
                }),
                "latent": ("LATENT", {
                    "tooltip": "VAE-encoded stitched frame. Required when using pixel_stitcher "
                               "to determine frame dimensions. Not needed for latent_stitcher."
                }),
            },
        }

    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("mask", "info")
    FUNCTION = "execute"
    CATEGORY = "NV_Utils/Inpaint"
    DESCRIPTION = (
        "Generate a gradient mask along stitch boundaries for boundary diffusion seam repair. "
        "Accepts LATENT_STITCHER (latent path) or pixel STITCHER (Kling/API path). "
        "Use with SetLatentNoiseMask + low-denoise KSampler to harmonize inpaint seams."
    )

    def execute(self, strip_width=32, blur_radius=16, inner_ratio=0.5,
                latent_stitcher=None, pixel_stitcher=None, latent=None):

        # --- Resolve crop coordinates and frame dimensions ---
        if latent_stitcher is not None:
            cx = latent_stitcher['crop_x_pixel']
            cy = latent_stitcher['crop_y_pixel']
            cw = latent_stitcher['crop_w_pixel']
            ch = latent_stitcher['crop_h_pixel']

            orig = latent_stitcher['original_samples']
            if orig.ndim == 5:
                H_l, W_l = orig.shape[-2], orig.shape[-1]
            else:
                H_l, W_l = orig.shape[-2], orig.shape[-1]
            frame_h = H_l * VAE_STRIDE
            frame_w = W_l * VAE_STRIDE
            source = "latent_stitcher"

        elif pixel_stitcher is not None:
            if latent is None:
                raise ValueError(
                    "[NV_StitchBoundaryMask] pixel_stitcher requires a latent input "
                    "for frame dimensions. Connect the VAE-encoded stitched frame."
                )
            cx, cy, cw, ch = _extract_pixel_stitcher_union(pixel_stitcher)

            samples = latent["samples"]
            H_l, W_l = samples.shape[-2], samples.shape[-1]
            frame_h = H_l * VAE_STRIDE
            frame_w = W_l * VAE_STRIDE
            num_frames = pixel_stitcher.get('total_frames', len(pixel_stitcher['canvas_to_orig_x']))
            source = f"pixel_stitcher (union of {num_frames} frames)"

        else:
            raise ValueError(
                "[NV_StitchBoundaryMask] Connect either latent_stitcher or pixel_stitcher."
            )

        # --- Build boundary strip ---
        inner_px = max(1, int(strip_width * inner_ratio))
        outer_px = max(1, strip_width - inner_px)

        mask = _build_boundary_strip(frame_h, frame_w, cx, cy, cw, ch, inner_px, outer_px)

        # --- Gaussian blur for gradient falloff ---
        if blur_radius > 0:
            mask = _gaussian_blur_mask(mask, blur_radius)

        # --- Info ---
        total_pixels = frame_h * frame_w
        active_pixels = (mask > 0.01).sum().item()
        coverage_pct = 100.0 * active_pixels / total_pixels

        info_lines = [
            f"Source: {source}",
            f"Frame: {frame_w}x{frame_h}px ({W_l}x{H_l} latent)",
            f"Crop: ({cx},{cy}) {cw}x{ch}px",
            f"Strip: {strip_width}px total (inner={inner_px}px, outer={outer_px}px)",
            f"Blur: {blur_radius}px (sigma={blur_radius / 3.0:.1f})",
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
