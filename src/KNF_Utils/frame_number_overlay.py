"""
Frame Number Overlay Node - Simple debugging utility to overlay frame numbers on images
"""

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class NV_FrameNumberOverlay:
    """Overlays frame numbers on images for debugging video/animation workflows"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "font_size": ("INT", {"default": 32, "min": 8, "max": 256, "step": 1}),
                "position": (["top-left", "top-right", "bottom-left", "bottom-right"],),
                "padding": ("INT", {"default": 10, "min": 0, "max": 100, "step": 1}),
            },
            "optional": {
                "text_color": ("STRING", {"default": "#FFFFFF"}),
                "background": ("BOOLEAN", {"default": True}),
                "start_number": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "overlay_frame_numbers"
    CATEGORY = "NV_Utils/Debug"

    def hex_to_rgb(self, hex_color: str) -> tuple:
        """Convert hex color string to RGB tuple"""
        hex_color = hex_color.lstrip("#")
        if len(hex_color) != 6:
            return (255, 255, 255)  # Default to white
        try:
            return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
        except ValueError:
            return (255, 255, 255)

    def overlay_frame_numbers(
        self,
        images: torch.Tensor,
        font_size: int,
        position: str,
        padding: int,
        text_color: str = "#FFFFFF",
        background: bool = True,
        start_number: int = 0,
    ) -> tuple:
        """
        Overlay frame numbers on each image in the batch.

        Args:
            images: [B, H, W, C] tensor in float32 [0, 1] range
            font_size: Size of the frame number text
            position: Where to place the text
            padding: Distance from edge
            text_color: Hex color for the text
            background: Whether to add dark background behind text
            start_number: First frame number to display

        Returns:
            Annotated images tensor [B, H, W, C]
        """
        batch_size = images.shape[0]
        height = images.shape[1]
        width = images.shape[2]

        # Convert color
        rgb_color = self.hex_to_rgb(text_color)

        # Try to get a good font, fall back to default
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except (OSError, IOError):
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", font_size)
            except (OSError, IOError):
                font = ImageFont.load_default()

        result_frames = []

        for i in range(batch_size):
            frame_num = start_number + i

            # Convert tensor to PIL Image
            img_np = images[i].cpu().numpy()
            img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)

            # Create draw object
            draw = ImageDraw.Draw(pil_img)

            # Get text to draw
            text = str(frame_num)

            # Get text bounding box
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # Calculate position
            if position == "top-left":
                x = padding
                y = padding
            elif position == "top-right":
                x = width - text_width - padding
                y = padding
            elif position == "bottom-left":
                x = padding
                y = height - text_height - padding
            else:  # bottom-right
                x = width - text_width - padding
                y = height - text_height - padding

            # Draw background rectangle if enabled
            if background:
                bg_padding = 4
                draw.rectangle(
                    [
                        x - bg_padding,
                        y - bg_padding,
                        x + text_width + bg_padding,
                        y + text_height + bg_padding,
                    ],
                    fill=(0, 0, 0, 180),
                )

            # Draw text with outline for visibility
            outline_color = (0, 0, 0)
            for ox, oy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                draw.text((x + ox, y + oy), text, font=font, fill=outline_color)

            # Draw main text
            draw.text((x, y), text, font=font, fill=rgb_color)

            # Convert back to tensor
            result_np = np.array(pil_img).astype(np.float32) / 255.0
            result_frames.append(torch.from_numpy(result_np))

        # Stack all frames
        result = torch.stack(result_frames, dim=0)

        return (result,)
