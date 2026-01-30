"""
Text Overlay Node - Overlay customizable text on images

Forked from comfyui-textoverlay with bug fixes for:
- Hex color expansion (3-char hex like #F00 now works correctly)
- Image dimension handling (grayscale/batch edge cases)
"""

import os
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


class NV_TextOverlay:
    """
    Overlay text on images with customization options for font, color,
    stroke, alignment, and positioning.
    """

    _horizontal_alignments = ["left", "center", "right"]
    _vertical_alignments = ["top", "middle", "bottom"]

    def __init__(self):
        self._loaded_font = None
        self._full_text = None
        self._x = None
        self._y = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "text": ("STRING", {"multiline": True, "default": "Hello"}),
                "font_size": ("INT", {"default": 32, "min": 1, "max": 9999, "step": 1}),
                "font": ("STRING", {"default": "arial.ttf"}),
                "fill_color_hex": ("STRING", {"default": "#FFFFFF"}),
                "stroke_color_hex": ("STRING", {"default": "#000000"}),
                "stroke_thickness": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "padding": ("INT", {"default": 16, "min": 0, "max": 128, "step": 1}),
                "horizontal_alignment": (cls._horizontal_alignments, {"default": "center"}),
                "vertical_alignment": (cls._vertical_alignments, {"default": "bottom"}),
                "x_shift": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1}),
                "y_shift": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1}),
                "line_spacing": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 50.0, "step": 0.5}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "overlay_text"
    CATEGORY = "NV_Utils/Image"

    def hex_to_rgb(self, hex_color: str) -> tuple:
        """Convert hex color string to RGB tuple."""
        hex_color = hex_color.lstrip("#")
        if len(hex_color) == 3:
            # Expand 3-char hex: "F00" -> "FF0000"
            hex_color = ''.join(c * 2 for c in hex_color)
        if len(hex_color) != 6:
            return (255, 255, 255)  # Default to white on invalid input
        try:
            return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
        except ValueError:
            return (255, 255, 255)

    def _load_font(self, font: str, font_size: int) -> ImageFont.FreeTypeFont:
        """Load font from fonts directory or system, with fallback to default."""
        # Check local fonts directory first
        fonts_dir = os.path.join(os.path.dirname(__file__), "fonts")
        font_path = os.path.join(fonts_dir, font)

        if not os.path.exists(font_path):
            # Try system font path
            font_path = font

        try:
            return ImageFont.truetype(font_path, font_size)
        except Exception:
            # Fall back to default font
            try:
                return ImageFont.load_default(font_size)
            except TypeError:
                # Older PIL versions don't accept size arg
                return ImageFont.load_default()

    def _wrap_text(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        font: ImageFont.FreeTypeFont,
        max_width: int,
    ) -> str:
        """Wrap text to fit within max_width, preserving explicit newlines."""
        words = text.replace("\n", "\n ").split(" ")
        text_lines = []
        line = ""

        for word in words:
            has_newline = "\n" in word
            word = word.strip()

            if draw.textlength(line + word, font=font) < max_width:
                line += word + " "
            else:
                if line.strip():
                    text_lines.append(line.strip())
                line = word + " "

            if has_newline:
                text_lines.append(line.strip())
                line = ""

        if line.strip():
            text_lines.append(line.strip())

        return "\n".join(text_lines)

    def _calculate_position(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        font: ImageFont.FreeTypeFont,
        img_width: int,
        img_height: int,
        stroke_width: int,
        horizontal_alignment: str,
        vertical_alignment: str,
        padding: int,
        x_shift: int,
        y_shift: int,
        line_spacing: float,
    ) -> tuple:
        """Calculate text position based on alignment settings."""
        left, top, right, bottom = draw.multiline_textbbox(
            (0, 0),
            text,
            font=font,
            stroke_width=stroke_width,
            align=horizontal_alignment,
            spacing=line_spacing,
        )
        text_width = right - left
        text_height = bottom - top

        # Horizontal position
        if horizontal_alignment == "left":
            x = padding
        elif horizontal_alignment == "center":
            x = (img_width - text_width) / 2
        else:  # right
            x = img_width - text_width - padding

        # Vertical position
        if vertical_alignment == "top":
            y = padding
        elif vertical_alignment == "middle":
            y = (img_height - text_height) / 2
        else:  # bottom
            y = img_height - text_height - padding

        return x + x_shift, y + y_shift

    def overlay_text(
        self,
        images: torch.Tensor,
        text: str,
        font_size: int,
        font: str,
        fill_color_hex: str,
        stroke_color_hex: str,
        stroke_thickness: float,
        padding: int,
        horizontal_alignment: str,
        vertical_alignment: str,
        x_shift: int,
        y_shift: int,
        line_spacing: float,
    ) -> tuple:
        """
        Overlay text on each image in the batch.

        Args:
            images: [B, H, W, C] tensor in float32 [0, 1] range
            text: Text to overlay
            font_size: Size of the text
            font: Font file name (checked in fonts/ dir first, then system)
            fill_color_hex: Text fill color in hex format
            stroke_color_hex: Text stroke color in hex format
            stroke_thickness: Stroke thickness as fraction of font size
            padding: Distance from edge for alignment
            horizontal_alignment: "left", "center", or "right"
            vertical_alignment: "top", "middle", or "bottom"
            x_shift: Additional horizontal offset
            y_shift: Additional vertical offset
            line_spacing: Space between text lines

        Returns:
            Annotated images tensor [B, H, W, C]
        """
        # Ensure batch dimension exists
        if len(images.shape) == 3:
            images = images.unsqueeze(0)

        batch_size = images.shape[0]
        height = images.shape[1]
        width = images.shape[2]

        # Load font
        loaded_font = self._load_font(font, font_size)

        # Convert colors
        fill_color = self.hex_to_rgb(fill_color_hex)
        stroke_color = self.hex_to_rgb(stroke_color_hex)
        stroke_width = int(font_size * stroke_thickness * 0.5)

        # Cache for text wrapping and positioning (same for all frames)
        wrapped_text = None
        text_x, text_y = None, None

        result_frames = []

        for i in range(batch_size):
            # Convert tensor to PIL Image
            img_np = images[i].cpu().numpy()
            img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)

            draw = ImageDraw.Draw(pil_img)

            # Wrap text on first frame (cache for rest)
            if wrapped_text is None:
                max_width = width - 2 * padding
                wrapped_text = self._wrap_text(draw, text, loaded_font, max_width)

            # Calculate position on first frame (cache for rest)
            if text_x is None:
                text_x, text_y = self._calculate_position(
                    draw,
                    wrapped_text,
                    loaded_font,
                    width,
                    height,
                    stroke_width,
                    horizontal_alignment,
                    vertical_alignment,
                    padding,
                    x_shift,
                    y_shift,
                    line_spacing,
                )

            # Draw text with stroke
            draw.multiline_text(
                (text_x, text_y),
                wrapped_text,
                fill=fill_color,
                stroke_fill=stroke_color,
                stroke_width=stroke_width,
                font=loaded_font,
                align=horizontal_alignment,
                spacing=line_spacing,
            )

            # Convert back to tensor
            result_np = np.array(pil_img).astype(np.float32) / 255.0
            result_frames.append(torch.from_numpy(result_np))

        result = torch.stack(result_frames, dim=0)
        return (result,)


NODE_CLASS_MAPPINGS = {
    "NV_TextOverlay": NV_TextOverlay,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_TextOverlay": "Text Overlay",
}
