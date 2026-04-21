"""
NV Mask Overlay Viz - Visualize 1-4 MASK tensors as colored overlays on an IMAGE.

Debug companion to NV_MaskUnion. Shows WHERE each mask covers and how the masks
relate spatially:
  - Each mask gets its own solid color tint
  - Overlapping regions additively blend (red + green = yellow, etc.)
  - Pixels not in any mask show the original image unmodified

Typical use:
  - wire face_mask → mask_a (color=red)
  - wire hair_mask → mask_b (color=green)
  - pipe the result into Preview Image or NV Preview Animation
  - overlap areas show yellow → spots where face and hair both segment the same pixel
"""

import torch


# Named color presets → RGB [0,1]. Keeps the UI friendly; advanced users can
# rewire slot order to get different color assignments.
_COLORS = {
    "red":     (1.0, 0.25, 0.25),
    "green":   (0.25, 0.95, 0.35),
    "blue":    (0.30, 0.50, 1.00),
    "yellow":  (1.0,  0.92, 0.25),
    "magenta": (1.0,  0.25, 0.95),
    "cyan":    (0.25, 0.95, 1.0),
    "orange":  (1.0,  0.55, 0.15),
    "white":   (1.0,  1.0,  1.0),
}
_COLOR_NAMES = list(_COLORS.keys())


class NV_MaskOverlayViz:
    """Overlay up to 4 MASKs on an IMAGE with per-slot colors for quick debug."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Base image or video frames."}),
                "mask_a": ("MASK",),
                "color_a": (_COLOR_NAMES, {"default": "red"}),
                "opacity": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Shared overlay opacity for all masks. 0 = invisible, 1 = opaque.",
                }),
            },
            "optional": {
                "mask_b": ("MASK",),
                "color_b": (_COLOR_NAMES, {"default": "green"}),
                "mask_c": ("MASK",),
                "color_c": (_COLOR_NAMES, {"default": "blue"}),
                "mask_d": ("MASK",),
                "color_d": (_COLOR_NAMES, {"default": "yellow"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("preview",)
    FUNCTION = "viz"
    CATEGORY = "NV_Utils/Mask"
    DESCRIPTION = (
        "Visualize up to 4 MASKs as colored overlays on an IMAGE. Overlapping "
        "regions mix additively (red+green=yellow). Use for debugging mask "
        "union pipelines — verify each region covers the right pixels."
    )

    @staticmethod
    def _normalize_mask(name, mask):
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        elif mask.ndim != 3:
            raise ValueError(
                f"[NV_MaskOverlayViz] mask_{name} must be 2D [H,W] or 3D [B,H,W], "
                f"got shape {tuple(mask.shape)}."
            )
        return mask

    def viz(self, image, mask_a, color_a, opacity,
            mask_b=None, color_b="green",
            mask_c=None, color_c="blue",
            mask_d=None, color_d="yellow"):
        # image: [B, H, W, 3] float32 in [0,1]
        if image.ndim != 4 or image.shape[-1] < 3:
            raise ValueError(
                f"[NV_MaskOverlayViz] image must be [B,H,W,C>=3], got {tuple(image.shape)}."
            )

        inputs = [
            ("a", mask_a, color_a),
            ("b", mask_b, color_b),
            ("c", mask_c, color_c),
            ("d", mask_d, color_d),
        ]

        # Drop unused slots; normalize shape; sanitize NaN/inf.
        pairs = []
        for name, m, color in inputs:
            if m is None:
                continue
            m = self._normalize_mask(name, m)
            m = torch.nan_to_num(m, nan=0.0, posinf=1.0, neginf=0.0)
            pairs.append((name, m, color))

        B_img, H_img, W_img, _ = image.shape
        device = image.device
        dtype = image.dtype

        # Accumulate per-pixel colored overlay + total coverage.
        # overlay: sum of (mask * color * opacity) — additive color mixing.
        # coverage: max of (mask * opacity) — how much of each pixel is tinted total.
        overlay = torch.zeros(B_img, H_img, W_img, 3, device=device, dtype=dtype)
        coverage = torch.zeros(B_img, H_img, W_img, 1, device=device, dtype=dtype)

        for name, m, color_name in pairs:
            m = m.to(device=device, dtype=dtype)
            # Broadcast mask shape to match image batch if needed (video case).
            if m.shape[0] != B_img:
                if m.shape[0] == 1:
                    m = m.expand(B_img, -1, -1)
                else:
                    raise ValueError(
                        f"[NV_MaskOverlayViz] mask_{name} batch={m.shape[0]} "
                        f"doesn't match image batch={B_img}. Masks should either "
                        f"match the image batch size or be single-frame [1,H,W]."
                    )
            if m.shape[1:] != (H_img, W_img):
                raise ValueError(
                    f"[NV_MaskOverlayViz] mask_{name} spatial={tuple(m.shape[1:])} "
                    f"doesn't match image spatial=({H_img},{W_img})."
                )
            m = m.unsqueeze(-1)  # [B, H, W, 1]
            alpha = m * opacity
            color_rgb = torch.tensor(_COLORS[color_name], device=device, dtype=dtype).view(1, 1, 1, 3)
            overlay = overlay + color_rgb * alpha
            coverage = torch.maximum(coverage, alpha)

        # Clamp overlay so additive overlaps don't produce super-saturated pixels.
        overlay = overlay.clamp(0.0, 1.0)
        coverage = coverage.clamp(0.0, 1.0)

        # Blend: image * (1 - coverage) + overlay * coverage
        # If image has alpha channel, preserve it untouched.
        if image.shape[-1] > 3:
            rgb = image[..., :3]
            extra = image[..., 3:]
            blended_rgb = rgb * (1.0 - coverage) + overlay * coverage
            result = torch.cat([blended_rgb, extra], dim=-1)
        else:
            result = image * (1.0 - coverage) + overlay * coverage

        result = result.clamp(0.0, 1.0).to(dtype=torch.float32)
        return (result,)


NODE_CLASS_MAPPINGS = {
    "NV_MaskOverlayViz": NV_MaskOverlayViz,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_MaskOverlayViz": "NV Mask Overlay Viz",
}
