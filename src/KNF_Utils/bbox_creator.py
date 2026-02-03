"""
NV BBox Creator - Interactive bounding box creator with aspect ratio constraints.

Draw bounding boxes on images with selectable aspect ratios, output as MASK.
Uses frontend canvas for drawing, backend for mask generation.
"""

import torch
import json
import base64
from io import BytesIO
from PIL import Image
import numpy as np


class NV_BBoxCreator:
    """
    Interactive bounding box creator with aspect ratio constraints.
    Draw a bbox on the input image, get a MASK output.
    """

    ASPECT_RATIOS = ["Free", "1:1", "4:3", "3:4", "16:9", "9:16", "3:2", "2:3", "Custom"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "aspect_ratio": (cls.ASPECT_RATIOS, {"default": "Free"}),
            },
            "optional": {
                "custom_ratio": ("STRING", {"default": "16:9"}),
                # bbox_data is hidden visually by JS but needs to be optional (not hidden)
                # so it creates a widget that can be serialized and sent to backend
                "bbox_data": ("STRING", {"default": "{}"}),
            },
        }

    RETURN_TYPES = ("MASK", "SAM3_BOX_PROMPT", "SAM3_BOXES_PROMPT")
    RETURN_NAMES = ("mask", "sam3_box", "sam3_boxes")
    FUNCTION = "create_bbox"
    CATEGORY = "NV_Utils/mask"
    OUTPUT_NODE = True  # Required to send preview back to frontend

    def create_bbox(self, images, aspect_ratio, custom_ratio="16:9", bbox_data="{}"):
        """
        Create a mask from the bounding box drawn by the user.

        Args:
            images: Input image tensor [B, H, W, C]
            aspect_ratio: Selected aspect ratio preset
            custom_ratio: Custom ratio string (e.g., "21:9")
            bbox_data: JSON string with bbox coordinates from frontend

        Returns:
            dict with UI preview and mask result
        """
        B, H, W, C = images.shape

        # Parse bbox from frontend JSON
        try:
            bbox = json.loads(bbox_data) if bbox_data else {}
        except (json.JSONDecodeError, TypeError):
            bbox = {}

        # Extract coordinates (default to no bbox if not set)
        x1 = bbox.get("x1")
        y1 = bbox.get("y1")
        x2 = bbox.get("x2")
        y2 = bbox.get("y2")

        # Create mask tensor [B, H, W]
        mask = torch.zeros((B, H, W), dtype=torch.float32)

        # Only fill mask if we have valid bbox coordinates
        if all(v is not None for v in [x1, y1, x2, y2]):
            # Convert to int and clamp to image bounds
            x1 = max(0, min(int(x1), W))
            x2 = max(0, min(int(x2), W))
            y1 = max(0, min(int(y1), H))
            y2 = max(0, min(int(y2), H))

            # Ensure proper ordering
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1

            # Fill bbox region with 1.0
            if x2 > x1 and y2 > y1:
                mask[:, y1:y2, x1:x2] = 1.0

        # Build SAM3-compatible outputs (normalized center format)
        sam3_box = None
        sam3_boxes = {"boxes": [], "labels": []}

        if all(v is not None for v in [x1, y1, x2, y2]) and x2 > x1 and y2 > y1:
            # Convert pixel coords to normalized 0-1 center format
            # SAM3 expects: [center_x, center_y, width, height]
            center_x = ((x1 + x2) / 2) / W
            center_y = ((y1 + y2) / 2) / H
            width = (x2 - x1) / W
            height = (y2 - y1) / H

            # Single box format (SAM3_BOX_PROMPT)
            sam3_box = {
                "box": [center_x, center_y, width, height],
                "label": True  # Positive box by default
            }

            # Multi-box format (SAM3_BOXES_PROMPT)
            sam3_boxes = {
                "boxes": [[center_x, center_y, width, height]],
                "labels": [True]
            }

        # Send first frame as preview to frontend (for canvas display)
        preview_image = self._encode_preview(images[0])

        return {"ui": {"bg_image": [preview_image]}, "result": (mask, sam3_box, sam3_boxes)}

    def _encode_preview(self, image_tensor):
        """
        Encode image tensor to base64 JPEG for frontend display.

        Args:
            image_tensor: Single image tensor [H, W, C] in range [0, 1]

        Returns:
            Base64 encoded JPEG string
        """
        # Convert [H, W, C] tensor to numpy array
        img_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)

        # Create PIL Image
        pil_img = Image.fromarray(img_np)

        # Encode to base64 JPEG
        buffer = BytesIO()
        pil_img.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


NODE_CLASS_MAPPINGS = {
    "NV_BBoxCreator": NV_BBoxCreator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_BBoxCreator": "NV BBox Creator",
}
