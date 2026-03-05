"""
NV Point Picker - Interactive point picker for specifying tracking points.

Click on features in the image to place tracking points. Outputs a JSON string
of point coordinates that NV_CoTrackerBridge consumes for multi-point stabilization.

Workflow:
  InpaintCrop2 (stabilize=off) -> NV_PointPicker -> NV_CoTrackerBridge -> denoise -> InpaintStitch2
"""

import json
import base64
from io import BytesIO

import torch
import numpy as np
from PIL import Image


class NV_PointPicker:
    """Interactive point picker for placing tracking points on images.

    Click on features you want to stabilize. The points are passed to
    NV_CoTrackerBridge for multi-point tracking and averaging.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Input images to display for point picking (typically cropped output from InpaintCrop2)."
                }),
            },
            "optional": {
                "frame_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 99999,
                    "tooltip": "Frame index to display from video batch (0 = first frame). Pick a frame where the feature you want to track is clearly visible."
                }),
                "mask": ("MASK", {
                    "tooltip": "Optional mask overlay to help visualize the tracked region while placing points."
                }),
                "point_data": ("STRING", {
                    "default": "[]",
                    "tooltip": "JSON point data from frontend (hidden widget, managed by JS extension)."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("images", "tracking_points", "info", "frame_index", "total_frames")
    FUNCTION = "pick_points"
    CATEGORY = "NV_Utils/Inpaint"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Click on features to place tracking points for CoTracker stabilization. "
        "Left-click to add, right-click to remove nearest. "
        "Connect tracking_points output to NV_CoTrackerBridge."
    )

    def pick_points(self, images, frame_index=0, mask=None, point_data="[]"):
        B, H, W, C = images.shape
        frame_index = min(frame_index, B - 1)

        # Parse point data from frontend
        try:
            points = json.loads(point_data) if point_data else []
        except (json.JSONDecodeError, TypeError):
            points = []

        # Validate and clamp points to image bounds
        valid_points = []
        for p in points:
            if isinstance(p, dict) and "x" in p and "y" in p:
                px = max(0.0, min(float(p["x"]), float(W)))
                py = max(0.0, min(float(p["y"]), float(H)))
                valid_points.append({"x": px, "y": py})

        tracking_points = json.dumps(valid_points)

        n = len(valid_points)
        if n == 0:
            info = f"No points placed. Click on the image to add tracking points. Image: {W}x{H}, {B} frames, showing frame {frame_index}."
        else:
            coords = ", ".join(f"({p['x']:.0f}, {p['y']:.0f})" for p in valid_points)
            info = f"{n} tracking point{'s' if n != 1 else ''}: {coords}. Image: {W}x{H}, {B} frames, showing frame {frame_index}."

        print(f"[NV_PointPicker] {info}")

        # Send selected frame as preview to frontend
        preview = self._encode_preview(images[frame_index], mask=mask, frame_index=frame_index if mask is not None else 0)

        return {
            "ui": {"bg_image": [preview], "image_size": [{"width": W, "height": H}], "total_frames": [B]},
            "result": (images, tracking_points, info, frame_index, B),
        }

    def _encode_preview(self, image_tensor, mask=None, frame_index=0):
        """Encode image tensor to base64 JPEG for frontend display."""
        img_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)

        if mask is not None:
            img_np = self._overlay_mask(img_np, mask, frame_index=frame_index)

        pil_img = Image.fromarray(img_np)
        buffer = BytesIO()
        pil_img.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    @staticmethod
    def _overlay_mask(img_array, mask_tensor, frame_index=0, color=(0, 180, 255), alpha=0.25):
        """Composite a mask as a semi-transparent overlay."""
        if mask_tensor.dim() == 3:
            fi = min(frame_index, mask_tensor.shape[0] - 1)
            frame_mask = mask_tensor[fi].cpu().numpy()
        elif mask_tensor.dim() == 2:
            frame_mask = mask_tensor.cpu().numpy()
        else:
            return img_array

        if frame_mask.max() > 1.0:
            frame_mask = frame_mask / 255.0

        h, w = img_array.shape[:2]
        mh, mw = frame_mask.shape[:2]
        if mh != h or mw != w:
            mask_t = torch.from_numpy(frame_mask).float().unsqueeze(0).unsqueeze(0)
            mask_t = torch.nn.functional.interpolate(mask_t, size=(h, w), mode="bilinear", align_corners=False)
            frame_mask = mask_t.squeeze().numpy()

        binary = (frame_mask > 0.5).astype(np.float32)
        overlay = img_array.astype(np.float32)
        for c in range(3):
            overlay[:, :, c] = overlay[:, :, c] * (1.0 - alpha * binary) + color[c] * (alpha * binary)

        return np.clip(overlay, 0, 255).astype(np.uint8)


NODE_CLASS_MAPPINGS = {
    "NV_PointPicker": NV_PointPicker,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_PointPicker": "NV Point Picker",
}
