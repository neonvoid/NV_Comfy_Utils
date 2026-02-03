"""
NV Resolution Verifier - Adjust resolution to be divisible by a specified value.

Useful for ensuring resolutions are compatible with VAE/diffusion models that
require dimensions divisible by 8, 16, 32, 64, etc.
"""

import math


class NV_ResolutionVerifier:
    """
    Verify and adjust resolution to be divisible by a specified value.

    Inputs can be:
    - Aspect ratio string (e.g., "16:9") + target dimension
    - Explicit width and height
    - Width/height from upstream nodes (like NV_BBoxCreator)
    """

    ASPECT_RATIOS = ["Custom", "1:1", "4:3", "3:4", "16:9", "9:16", "3:2", "2:3", "21:9", "9:21"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["from_dimensions", "from_aspect_ratio"], {"default": "from_dimensions"}),
                "divisible_by": ("INT", {
                    "default": 64,
                    "min": 1,
                    "max": 512,
                    "step": 1,
                    "tooltip": "Output dimensions will be divisible by this value (common: 8, 16, 32, 64)"
                }),
                "round_mode": (["nearest", "up", "down"], {
                    "default": "nearest",
                    "tooltip": "How to round: nearest (closest), up (ceil), down (floor)"
                }),
            },
            "optional": {
                # For from_dimensions mode
                "width": ("INT", {"default": 512, "min": 1, "max": 16384, "step": 1}),
                "height": ("INT", {"default": 512, "min": 1, "max": 16384, "step": 1}),
                # For from_aspect_ratio mode
                "aspect_ratio": (cls.ASPECT_RATIOS, {"default": "16:9"}),
                "custom_ratio": ("STRING", {"default": "16:9", "tooltip": "Custom aspect ratio (e.g., '21:9')"}),
                "target_pixels": ("INT", {
                    "default": 1048576,
                    "min": 65536,
                    "max": 67108864,
                    "step": 1024,
                    "tooltip": "Target total pixels (e.g., 1048576 = 1024x1024 equivalent)"
                }),
                # Minimum constraints
                "min_width": ("INT", {"default": 64, "min": 1, "max": 16384}),
                "min_height": ("INT", {"default": 64, "min": 1, "max": 16384}),
            },
        }

    RETURN_TYPES = ("INT", "INT", "STRING", "FLOAT", "INT")
    RETURN_NAMES = ("width", "height", "resolution_str", "ratio_value", "total_pixels")
    FUNCTION = "verify_resolution"
    CATEGORY = "NV_Utils/utils"

    def verify_resolution(
        self,
        mode,
        divisible_by,
        round_mode,
        width=512,
        height=512,
        aspect_ratio="16:9",
        custom_ratio="16:9",
        target_pixels=1048576,
        min_width=64,
        min_height=64,
    ):
        """
        Verify and adjust resolution to be divisible by specified value.
        """

        if mode == "from_dimensions":
            # Use provided width/height directly
            out_width = width
            out_height = height
        else:
            # from_aspect_ratio mode - compute dimensions from ratio and target pixels
            ratio_str = custom_ratio if aspect_ratio == "Custom" else aspect_ratio
            ratio_value = self._parse_ratio(ratio_str)

            # Calculate dimensions from target pixels and ratio
            # width * height = target_pixels
            # width / height = ratio_value
            # So: width = sqrt(target_pixels * ratio_value)
            #     height = sqrt(target_pixels / ratio_value)
            out_width = int(math.sqrt(target_pixels * ratio_value))
            out_height = int(math.sqrt(target_pixels / ratio_value))

        # Apply divisibility constraint
        out_width = self._make_divisible(out_width, divisible_by, round_mode, min_width)
        out_height = self._make_divisible(out_height, divisible_by, round_mode, min_height)

        # Calculate outputs
        ratio_value = out_width / out_height if out_height > 0 else 1.0
        total_pixels = out_width * out_height
        resolution_str = f"{out_width}x{out_height}"

        return (out_width, out_height, resolution_str, ratio_value, total_pixels)

    def _parse_ratio(self, ratio_str):
        """Parse aspect ratio string like '16:9' into float value."""
        try:
            parts = ratio_str.split(":")
            if len(parts) == 2:
                return float(parts[0]) / float(parts[1])
        except (ValueError, ZeroDivisionError):
            pass
        return 1.0

    def _make_divisible(self, value, divisor, round_mode, minimum):
        """Adjust value to be divisible by divisor."""
        if round_mode == "nearest":
            result = round(value / divisor) * divisor
        elif round_mode == "up":
            result = math.ceil(value / divisor) * divisor
        else:  # down
            result = math.floor(value / divisor) * divisor

        # Ensure minimum
        if result < minimum:
            result = math.ceil(minimum / divisor) * divisor

        return int(result)


class NV_ResolutionFromMask:
    """
    Extract resolution from a mask tensor and optionally adjust for divisibility.
    Useful for getting inpaint canvas dimensions from a mask.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 8.0,
                    "step": 0.1,
                    "tooltip": "Scale factor applied to dimensions before divisibility check"
                }),
                "divisible_by": ("INT", {
                    "default": 64,
                    "min": 1,
                    "max": 512,
                    "step": 1,
                    "tooltip": "Output dimensions will be divisible by this value"
                }),
                "round_mode": (["nearest", "up", "down"], {"default": "nearest"}),
                "use_bbox": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If True, use bounding box of mask content instead of full mask dimensions"
                }),
                "padding": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 512,
                    "tooltip": "Padding to add around bbox (only when use_bbox=True)"
                }),
            },
        }

    RETURN_TYPES = ("INT", "INT", "STRING", "FLOAT", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("width", "height", "resolution_str", "ratio_value", "x", "y", "bbox_width", "bbox_height")
    FUNCTION = "get_resolution"
    CATEGORY = "NV_Utils/utils"

    def get_resolution(self, mask, scale, divisible_by, round_mode, use_bbox, padding):
        """
        Get resolution from mask, optionally using bbox of mask content.
        """
        import torch

        # Mask shape is [B, H, W] or [H, W]
        if len(mask.shape) == 3:
            h, w = mask.shape[1], mask.shape[2]
            mask_2d = mask[0]  # Use first mask in batch
        else:
            h, w = mask.shape
            mask_2d = mask

        # Default to full mask dimensions
        x, y = 0, 0
        bbox_w, bbox_h = w, h

        if use_bbox:
            # Find bounding box of non-zero mask content
            nonzero = torch.nonzero(mask_2d > 0.5)
            if len(nonzero) > 0:
                y_min = nonzero[:, 0].min().item()
                y_max = nonzero[:, 0].max().item()
                x_min = nonzero[:, 1].min().item()
                x_max = nonzero[:, 1].max().item()

                # Apply padding
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(w - 1, x_max + padding)
                y_max = min(h - 1, y_max + padding)

                x, y = x_min, y_min
                bbox_w = x_max - x_min + 1
                bbox_h = y_max - y_min + 1

        # Apply scale factor before divisibility check
        scaled_w = bbox_w * scale
        scaled_h = bbox_h * scale

        # Adjust for divisibility
        out_width = self._make_divisible(scaled_w, divisible_by, round_mode)
        out_height = self._make_divisible(scaled_h, divisible_by, round_mode)

        ratio_value = out_width / out_height if out_height > 0 else 1.0
        resolution_str = f"{out_width}x{out_height}"

        return (out_width, out_height, resolution_str, ratio_value, x, y, bbox_w, bbox_h)

    def _make_divisible(self, value, divisor, round_mode):
        """Adjust value to be divisible by divisor."""
        import math
        if round_mode == "nearest":
            result = round(value / divisor) * divisor
        elif round_mode == "up":
            result = math.ceil(value / divisor) * divisor
        else:  # down
            result = math.floor(value / divisor) * divisor
        return max(int(result), divisor)  # At least one divisor unit


NODE_CLASS_MAPPINGS = {
    "NV_ResolutionVerifier": NV_ResolutionVerifier,
    "NV_ResolutionFromMask": NV_ResolutionFromMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_ResolutionVerifier": "NV Resolution Verifier",
    "NV_ResolutionFromMask": "NV Resolution From Mask",
}
