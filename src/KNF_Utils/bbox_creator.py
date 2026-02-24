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
                "mask": ("MASK", {"tooltip": "Optional mask to overlay on the canvas as a semi-transparent tint, useful for visualizing existing masked areas while drawing boxes."}),
            },
        }

    RETURN_TYPES = ("MASK", "SAM3_BOXES_PROMPT", "SAM3_BOXES_PROMPT", "STRING", "FLOAT", "INT", "INT")
    RETURN_NAMES = ("mask", "positive_boxes", "negative_boxes", "aspect_ratio", "ratio_value", "width", "height")
    FUNCTION = "create_bbox"
    CATEGORY = "NV_Utils/mask"
    OUTPUT_NODE = True  # Required to send preview back to frontend

    def create_bbox(self, images, aspect_ratio, custom_ratio="16:9", bbox_data="{}", mask=None):
        """
        Create a mask from the bounding box drawn by the user.

        Args:
            images: Input image tensor [B, H, W, C]
            aspect_ratio: Selected aspect ratio preset
            custom_ratio: Custom ratio string (e.g., "21:9")
            bbox_data: JSON string with bbox coordinates from frontend
                       Format: {"positive": [...], "negative": [...]}
            mask: Optional mask tensor to overlay on the canvas preview

        Returns:
            dict with UI preview and mask result
        """
        overlay_mask = mask  # Save before local 'mask' variable shadows it
        B, H, W, C = images.shape

        # Parse bbox data from frontend JSON
        # New format: {"positive": [{x1,y1,x2,y2}, ...], "negative": [{x1,y1,x2,y2}, ...]}
        try:
            data = json.loads(bbox_data) if bbox_data else {}
        except (json.JSONDecodeError, TypeError):
            data = {}

        pos_bboxes = data.get("positive", [])
        neg_bboxes = data.get("negative", [])

        # Create mask tensor [B, H, W] - fill positive boxes, leave negative empty
        mask = torch.zeros((B, H, W), dtype=torch.float32)

        # Track the "primary" bbox for aspect ratio calculation (first positive, or first negative)
        primary_bbox = None

        # Helper to process and clamp bbox coordinates
        def process_bbox(bbox_dict):
            bx1 = bbox_dict.get("x1")
            by1 = bbox_dict.get("y1")
            bx2 = bbox_dict.get("x2")
            by2 = bbox_dict.get("y2")
            if all(v is not None for v in [bx1, by1, bx2, by2]):
                bx1 = max(0, min(int(bx1), W))
                bx2 = max(0, min(int(bx2), W))
                by1 = max(0, min(int(by1), H))
                by2 = max(0, min(int(by2), H))
                if bx1 > bx2:
                    bx1, bx2 = bx2, bx1
                if by1 > by2:
                    by1, by2 = by2, by1
                if bx2 > bx1 and by2 > by1:
                    return (bx1, by1, bx2, by2)
            return None

        # Process positive boxes - fill mask
        for bbox_dict in pos_bboxes:
            coords = process_bbox(bbox_dict)
            if coords:
                bx1, by1, bx2, by2 = coords
                mask[:, by1:by2, bx1:bx2] = 1.0
                if primary_bbox is None:
                    primary_bbox = coords

        # Track first negative for fallback primary
        for bbox_dict in neg_bboxes:
            coords = process_bbox(bbox_dict)
            if coords and primary_bbox is None:
                primary_bbox = coords

        # Build SAM3-compatible outputs (normalized center format)
        positive_boxes = {"boxes": [], "labels": []}
        negative_boxes = {"boxes": [], "labels": []}

        # Convert positive boxes to SAM3 format
        for bbox_dict in pos_bboxes:
            coords = process_bbox(bbox_dict)
            if coords:
                bx1, by1, bx2, by2 = coords
                center_x = ((bx1 + bx2) / 2) / W
                center_y = ((by1 + by2) / 2) / H
                width = (bx2 - bx1) / W
                height = (by2 - by1) / H
                positive_boxes["boxes"].append([center_x, center_y, width, height])
                positive_boxes["labels"].append(True)

        # Convert negative boxes to SAM3 format
        for bbox_dict in neg_bboxes:
            coords = process_bbox(bbox_dict)
            if coords:
                bx1, by1, bx2, by2 = coords
                center_x = ((bx1 + bx2) / 2) / W
                center_y = ((by1 + by2) / 2) / H
                width = (bx2 - bx1) / W
                height = (by2 - by1) / H
                negative_boxes["boxes"].append([center_x, center_y, width, height])
                negative_boxes["labels"].append(False)

        # Use primary_bbox for aspect ratio calculations
        x1, y1, x2, y2 = primary_bbox if primary_bbox else (None, None, None, None)

        # Compute aspect ratio outputs
        ratio_presets = {
            "1:1": 1.0, "4:3": 4/3, "3:4": 3/4, "16:9": 16/9,
            "9:16": 9/16, "3:2": 3/2, "2:3": 2/3
        }

        # Determine the ratio string and value
        if aspect_ratio == "Custom":
            ratio_str = custom_ratio
            try:
                parts = custom_ratio.split(":")
                ratio_value = float(parts[0]) / float(parts[1]) if len(parts) == 2 else 1.0
            except (ValueError, ZeroDivisionError):
                ratio_value = 1.0
        elif aspect_ratio == "Free":
            ratio_str = "Free"
            # For free mode, compute actual ratio from drawn bbox
            if all(v is not None for v in [x1, y1, x2, y2]) and x2 > x1 and y2 > y1:
                ratio_value = float(x2 - x1) / float(y2 - y1)
            else:
                ratio_value = 1.0
        else:
            ratio_str = aspect_ratio
            ratio_value = ratio_presets.get(aspect_ratio, 1.0)

        # Compute bbox dimensions in pixels
        if all(v is not None for v in [x1, y1, x2, y2]) and x2 > x1 and y2 > y1:
            bbox_width = int(x2 - x1)
            bbox_height = int(y2 - y1)
        else:
            bbox_width = 0
            bbox_height = 0

        # Send first frame as preview to frontend (for canvas display)
        preview_image = self._encode_preview(images[0], mask=overlay_mask)

        return {
            "ui": {"bg_image": [preview_image]},
            "result": (mask, positive_boxes, negative_boxes, ratio_str, ratio_value, bbox_width, bbox_height)
        }

    def _encode_preview(self, image_tensor, mask=None):
        """
        Encode image tensor to base64 JPEG for frontend display.

        Args:
            image_tensor: Single image tensor [H, W, C] in range [0, 1]
            mask: Optional mask tensor [N, H, W] or [H, W] to overlay

        Returns:
            Base64 encoded JPEG string
        """
        # Convert [H, W, C] tensor to numpy array
        img_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)

        # Overlay mask if provided
        if mask is not None:
            img_np = self._overlay_mask(img_np, mask, frame_index=0)

        # Create PIL Image
        pil_img = Image.fromarray(img_np)

        # Encode to base64 JPEG
        buffer = BytesIO()
        pil_img.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    @staticmethod
    def _overlay_mask(img_array, mask_tensor, frame_index=0, color=(0, 180, 255), alpha=0.35):
        """Composite a mask as a semi-transparent colored overlay on the image.

        Args:
            img_array: uint8 numpy array [H, W, 3]
            mask_tensor: torch tensor [N, H, W] or [H, W]
            frame_index: which frame to extract from the mask batch
            color: RGB tuple for the overlay tint
            alpha: overlay opacity (0=invisible, 1=opaque)

        Returns:
            uint8 numpy array [H, W, 3] with mask overlay composited
        """
        # Extract the correct frame from the mask batch
        if mask_tensor.dim() == 3:
            fi = min(frame_index, mask_tensor.shape[0] - 1)
            frame_mask = mask_tensor[fi].cpu().numpy()
        elif mask_tensor.dim() == 2:
            frame_mask = mask_tensor.cpu().numpy()
        else:
            return img_array

        # Normalize to 0-1 if needed
        if frame_mask.max() > 1.0:
            frame_mask = frame_mask / 255.0

        # Resize mask if dimensions don't match image
        h, w = img_array.shape[:2]
        mh, mw = frame_mask.shape[:2]
        if mh != h or mw != w:
            mask_t = torch.from_numpy(frame_mask).float().unsqueeze(0).unsqueeze(0)
            mask_t = torch.nn.functional.interpolate(mask_t, size=(h, w), mode='bilinear', align_corners=False)
            frame_mask = mask_t.squeeze().numpy()

        # Create binary mask (threshold at 0.5)
        binary = (frame_mask > 0.5).astype(np.float32)

        # Composite: img * (1 - alpha*mask) + color * (alpha*mask)
        overlay = img_array.astype(np.float32)
        for c in range(3):
            overlay[:, :, c] = overlay[:, :, c] * (1.0 - alpha * binary) + color[c] * (alpha * binary)

        return np.clip(overlay, 0, 255).astype(np.uint8)


NODE_CLASS_MAPPINGS = {
    "NV_BBoxCreator": NV_BBoxCreator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_BBoxCreator": "NV BBox Creator",
}
