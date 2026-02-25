"""
NV Image Screen Composite - Overlay foreground-on-black onto a background image.

Derives alpha from the foreground's luminance (max across RGB channels),
so black pixels become transparent and colored pixels become opaque.
Useful for compositing OpenPose skeletons, edge maps, or other
visualizations rendered on black backgrounds onto arbitrary images.
"""

import torch


class NV_ImageScreenComposite:
    """
    Composite a foreground image (colored content on black background)
    onto a background image using luminance-derived alpha.

    Black pixels in the foreground become fully transparent.
    Non-black pixels are composited at their luminance intensity.
    An optional opacity control scales the overall foreground contribution.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "background": ("IMAGE", {
                    "tooltip": "Background image (e.g., grey inpaint area, video frames)."
                }),
                "foreground": ("IMAGE", {
                    "tooltip": "Foreground image on black background "
                               "(e.g., OpenPose skeleton, edge map)."
                }),
            },
            "optional": {
                "opacity": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Overall foreground opacity. 1.0 = full strength, "
                               "0.0 = background only."
                }),
                "threshold": ("FLOAT", {
                    "default": 0.01, "min": 0.0, "max": 1.0, "step": 0.005,
                    "tooltip": "Luminance threshold below which foreground pixels "
                               "are treated as fully transparent. Helps eliminate "
                               "near-black noise. 0.0 = no threshold."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "composite"
    CATEGORY = "NV_Utils/image"
    DESCRIPTION = (
        "Overlay a foreground-on-black image onto a background using "
        "luminance-derived alpha. Black becomes transparent, colored "
        "content becomes opaque. Perfect for compositing OpenPose "
        "skeletons or edge maps onto grey inpaint areas."
    )

    def composite(self, background, foreground, opacity=1.0, threshold=0.01):
        # Handle batch dimension mismatch: broadcast if needed
        bg = background
        fg = foreground

        # If one is a single image and the other is a batch, broadcast
        if bg.shape[0] == 1 and fg.shape[0] > 1:
            bg = bg.expand(fg.shape[0], -1, -1, -1)
        elif fg.shape[0] == 1 and bg.shape[0] > 1:
            fg = fg.expand(bg.shape[0], -1, -1, -1)

        # Handle spatial size mismatch: resize foreground to match background
        if fg.shape[1] != bg.shape[1] or fg.shape[2] != bg.shape[2]:
            # IMAGE tensor is [B, H, W, C], need [B, C, H, W] for interpolate
            fg_perm = fg.permute(0, 3, 1, 2)
            fg_perm = torch.nn.functional.interpolate(
                fg_perm,
                size=(bg.shape[1], bg.shape[2]),
                mode="bilinear",
                align_corners=False,
            )
            fg = fg_perm.permute(0, 2, 3, 1)

        # Derive alpha from luminance: max across RGB channels
        # IMAGE tensors are [B, H, W, C] with C=3 (RGB), values 0-1
        alpha = fg[:, :, :, :3].max(dim=-1, keepdim=True).values  # [B, H, W, 1]

        # Apply threshold to clean up near-black noise
        if threshold > 0:
            alpha = torch.where(alpha < threshold, torch.zeros_like(alpha), alpha)

        # Apply opacity
        alpha = alpha * opacity

        # Composite: result = bg * (1 - alpha) + fg * alpha
        result = bg * (1.0 - alpha) + fg * alpha

        return (result.clamp(0.0, 1.0),)


NODE_CLASS_MAPPINGS = {
    "NV_ImageScreenComposite": NV_ImageScreenComposite,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_ImageScreenComposite": "NV Image Screen Composite",
}
