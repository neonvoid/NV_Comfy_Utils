"""
NV Stitch Boundary Mask — Per-frame gradient mask along stitch boundaries for seam harmonization.

Generates a [B, H, W] mask where each frame's boundary strip tracks the subject's
position in that frame. For video, the subject moves — the boundary must follow.

Input modes (connect one):
  1. bbox_mask + latent: Per-frame bbox from MaskTrackingBBox → per-frame boundary strips.
     Most accurate for pixel-path workflows (Kling API, etc.).
  2. latent_stitcher: Static crop from NV_LatentInpaintCrop (broadcast to all frames).
  3. pixel_stitcher + latent: Per-frame canvas_to_orig coordinates from InpaintCrop2.

Feed the output mask to SetLatentNoiseMask, then run a low-denoise KSampler pass
to let WAN 2.2's native mask handling harmonize the seam.
"""

import torch
import torch.nn.functional as F


VAE_STRIDE = 8


def _make_gaussian_kernel(blur_radius):
    """Create a 1D Gaussian kernel for separable convolution."""
    kernel_size = blur_radius * 2 + 1
    sigma = blur_radius / 3.0
    x_coord = torch.arange(kernel_size, dtype=torch.float32) - blur_radius
    kernel_1d = torch.exp(-0.5 * (x_coord / sigma) ** 2)
    return kernel_1d / kernel_1d.sum()


def _gaussian_blur_batch(mask, blur_radius):
    """Apply separable Gaussian blur to a [B, H, W] mask tensor (batched)."""
    kernel_1d = _make_gaussian_kernel(blur_radius)
    B = mask.shape[0]

    # [B, 1, H, W] — conv2d handles batch dim automatically with single-channel kernel
    mask_4d = mask.unsqueeze(1)

    # Horizontal pass
    kh = kernel_1d.view(1, 1, 1, -1)
    mask_4d = F.pad(mask_4d, (blur_radius, blur_radius, 0, 0), mode='reflect')
    mask_4d = F.conv2d(mask_4d, kh)

    kv = kernel_1d.view(1, 1, -1, 1)
    mask_4d = F.pad(mask_4d, (0, 0, blur_radius, blur_radius), mode='reflect')
    mask_4d = F.conv2d(mask_4d, kv)

    return mask_4d.squeeze(1).clamp(0.0, 1.0)


def _bbox_from_single_frame(frame_mask):
    """Compute bbox from a single [H, W] mask. Returns (x, y, w, h) or None."""
    non_zero = torch.nonzero(frame_mask > 0.01)
    if non_zero.numel() == 0:
        return None
    y_min = non_zero[:, 0].min().item()
    y_max = non_zero[:, 0].max().item()
    x_min = non_zero[:, 1].min().item()
    x_max = non_zero[:, 1].max().item()
    return int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1)


def _build_boundary_strip_frame(frame_h, frame_w, cx, cy, cw, ch, inner_px, outer_px):
    """Build a binary ring mask for one frame. Returns [H, W] float tensor."""
    mask = torch.zeros(frame_h, frame_w)

    outer_top = max(0, cy - outer_px)
    outer_bot = min(frame_h, cy + ch + outer_px)
    outer_left = max(0, cx - outer_px)
    outer_right = min(frame_w, cx + cw + outer_px)
    mask[outer_top:outer_bot, outer_left:outer_right] = 1.0

    inner_top = max(0, cy + inner_px)
    inner_bot = min(frame_h, cy + ch - inner_px)
    inner_left = max(0, cx + inner_px)
    inner_right = min(frame_w, cx + cw - inner_px)
    if inner_top < inner_bot and inner_left < inner_right:
        mask[inner_top:inner_bot, inner_left:inner_right] = 0.0

    return mask


class NV_StitchBoundaryMask:
    """Per-frame gradient mask along stitch boundaries for boundary diffusion seam repair.

    For video: the boundary strip tracks the subject's position each frame.
    Accepts bbox_mask (per-frame), latent_stitcher (static), or pixel_stitcher (per-frame).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "strip_width": ("INT", {
                    "default": 32, "min": 8, "max": 128, "step": 8,
                    "tooltip": "Total width of boundary strip in pixels. "
                               "32px = 4 latent cells — good default for character inpaint."
                }),
                "blur_radius": ("INT", {
                    "default": 16, "min": 0, "max": 64, "step": 1,
                    "tooltip": "Gaussian blur for gradient falloff in pixels. "
                               "0 = hard binary strip (not recommended)."
                }),
                "inner_ratio": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1,
                    "tooltip": "Fraction of strip_width extending inward (into crop). "
                               "0.5 = symmetric. 1.0 = entirely inside. 0.0 = entirely outside."
                }),
            },
            "optional": {
                "bbox_mask": ("MASK", {
                    "tooltip": "Per-frame bounding box mask [B,H,W] from MaskTrackingBBox. "
                               "Each frame gets its own boundary strip that tracks the subject. "
                               "Most accurate for pixel-path workflows (Kling, etc.). "
                               "Requires latent input for frame dimensions."
                }),
                "latent_stitcher": ("LATENT_STITCHER", {
                    "tooltip": "From NV Latent Inpaint Crop. Static crop — same boundary "
                               "every frame (correct for latent path since crop is static)."
                }),
                "pixel_stitcher": ("STITCHER", {
                    "tooltip": "From NV Inpaint Crop v2. Per-frame canvas_to_orig coordinates. "
                               "Includes stabilization padding — prefer bbox_mask for accuracy. "
                               "Requires latent input for frame dimensions."
                }),
                "latent": ("LATENT", {
                    "tooltip": "VAE-encoded stitched frame. Required with bbox_mask or "
                               "pixel_stitcher to determine output resolution."
                }),
            },
        }

    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("mask", "info")
    FUNCTION = "execute"
    CATEGORY = "NV_Utils/Inpaint"
    DESCRIPTION = (
        "Per-frame gradient mask along stitch boundaries for boundary diffusion. "
        "Each frame's strip tracks the subject — not a static union. "
        "Use with SetLatentNoiseMask + low-denoise KSampler to harmonize seams."
    )

    def execute(self, strip_width=32, blur_radius=16, inner_ratio=0.5,
                bbox_mask=None, latent_stitcher=None, pixel_stitcher=None, latent=None):

        inner_px = max(1, int(strip_width * inner_ratio))
        outer_px = max(1, strip_width - inner_px)

        # =================================================================
        # Mode 1: bbox_mask — per-frame bbox → per-frame boundary strip
        # =================================================================
        if bbox_mask is not None:
            if latent is None:
                raise ValueError(
                    "[NV_StitchBoundaryMask] bbox_mask requires latent for frame dimensions."
                )
            samples = latent["samples"]
            H_l, W_l = samples.shape[-2], samples.shape[-1]
            frame_h = H_l * VAE_STRIDE
            frame_w = W_l * VAE_STRIDE

            bm = bbox_mask
            if bm.dim() == 2:
                bm = bm.unsqueeze(0)
            num_frames = bm.shape[0]

            # Rescale mask to frame dimensions if needed
            mask_h, mask_w = bm.shape[-2], bm.shape[-1]
            if mask_h != frame_h or mask_w != frame_w:
                bm = F.interpolate(
                    bm.unsqueeze(1).float(),
                    size=(frame_h, frame_w),
                    mode='bilinear', align_corners=False
                ).squeeze(1)

            # Per-frame boundary strips
            frames = []
            for i in range(num_frames):
                bbox = _bbox_from_single_frame(bm[i])
                if bbox is None:
                    frames.append(torch.zeros(frame_h, frame_w))
                else:
                    cx, cy, cw, ch = bbox
                    frames.append(
                        _build_boundary_strip_frame(frame_h, frame_w, cx, cy, cw, ch, inner_px, outer_px)
                    )

            mask = torch.stack(frames, dim=0)  # [B, H, W]
            source = f"bbox_mask ({num_frames} per-frame strips)"

        # =================================================================
        # Mode 2: latent_stitcher — static crop, broadcast to T frames
        # =================================================================
        elif latent_stitcher is not None:
            cx = latent_stitcher['crop_x_pixel']
            cy = latent_stitcher['crop_y_pixel']
            cw = latent_stitcher['crop_w_pixel']
            ch = latent_stitcher['crop_h_pixel']

            orig = latent_stitcher['original_samples']
            H_l, W_l = orig.shape[-2], orig.shape[-1]
            frame_h = H_l * VAE_STRIDE
            frame_w = W_l * VAE_STRIDE
            num_frames = orig.shape[2] if orig.ndim == 5 else 1

            strip = _build_boundary_strip_frame(frame_h, frame_w, cx, cy, cw, ch, inner_px, outer_px)
            mask = strip.unsqueeze(0).expand(num_frames, -1, -1).clone()  # [T, H, W]
            source = f"latent_stitcher (static, {num_frames} frames)"

        # =================================================================
        # Mode 3: pixel_stitcher — per-frame crop coordinates in original-image space
        # =================================================================
        elif pixel_stitcher is not None:
            # Frame dimensions come from canvas_to_orig (the original image size)
            cto_ws = pixel_stitcher['canvas_to_orig_w']
            cto_hs = pixel_stitcher['canvas_to_orig_h']
            frame_h = int(cto_hs[0])
            frame_w = int(cto_ws[0])

            # Crop-in-original coords = cropped_to_canvas - canvas_to_orig offset
            cto_xs = pixel_stitcher['canvas_to_orig_x']
            cto_ys = pixel_stitcher['canvas_to_orig_y']
            ctc_xs = pixel_stitcher['cropped_to_canvas_x']
            ctc_ys = pixel_stitcher['cropped_to_canvas_y']
            ctc_ws = pixel_stitcher['cropped_to_canvas_w']
            ctc_hs = pixel_stitcher['cropped_to_canvas_h']
            num_frames = len(ctc_xs)

            frames = []
            for i in range(num_frames):
                # Convert canvas coords to original-image coords
                crop_x = int(ctc_xs[i]) - int(cto_xs[i])
                crop_y = int(ctc_ys[i]) - int(cto_ys[i])
                crop_w = int(ctc_ws[i])
                crop_h = int(ctc_hs[i])
                frames.append(
                    _build_boundary_strip_frame(
                        frame_h, frame_w,
                        crop_x, crop_y, crop_w, crop_h,
                        inner_px, outer_px
                    )
                )
            mask = torch.stack(frames, dim=0)  # [B, H, W]
            source = f"pixel_stitcher ({num_frames} per-frame strips)"

        else:
            raise ValueError(
                "[NV_StitchBoundaryMask] Connect one of: bbox_mask, latent_stitcher, or pixel_stitcher."
            )

        # --- Gaussian blur (batched) ---
        if blur_radius > 0:
            mask = _gaussian_blur_batch(mask, blur_radius)

        # --- Info ---
        total_pixels = frame_h * frame_w
        # Use first frame for coverage stats (representative)
        active_first = (mask[0] > 0.01).sum().item()
        coverage_pct = 100.0 * active_first / total_pixels

        info_lines = [
            f"Source: {source}",
            f"Frame: {frame_w}x{frame_h}px ({W_l}x{H_l} latent)",
            f"Output: [{mask.shape[0]}, {mask.shape[1]}, {mask.shape[2]}]",
            f"Strip: {strip_width}px (inner={inner_px}px, outer={outer_px}px)",
            f"Blur: {blur_radius}px (sigma={blur_radius / 3.0:.1f})",
            f"Coverage (frame 0): {active_first}/{total_pixels} ({coverage_pct:.1f}%)",
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
