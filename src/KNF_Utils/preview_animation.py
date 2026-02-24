"""
NV Preview Animation - Fast animated preview using individual JPEG frames + JS flipbook.

Replaces the slow animated WebP approach (PIL save_all) with per-frame JPEG encoding
(~50-100x faster) and a custom frontend canvas widget that animates them.
"""

import os
import random

import numpy as np
from PIL import Image

import folder_paths


class NV_PreviewAnimation:
    """Fast animated preview: saves individual JPEG frames, JS frontend animates them."""

    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(
            random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(5)
        )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fps": ("FLOAT", {
                    "default": 8.0,
                    "min": 0.1,
                    "max": 120.0,
                    "step": 0.1,
                    "tooltip": "Playback frames per second",
                }),
            },
            "optional": {
                "images": ("IMAGE", {"tooltip": "Image batch to animate [B, H, W, C]"}),
                "masks": ("MASK", {"tooltip": "Optional mask batch to overlay as alpha"}),
                "quality": ("INT", {
                    "default": 85,
                    "min": 10,
                    "max": 100,
                    "step": 5,
                    "tooltip": "JPEG encoding quality (lower = smaller files, faster)",
                }),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "preview"
    OUTPUT_NODE = True
    CATEGORY = "NV_Utils/Preview"
    DESCRIPTION = "Fast animated preview using individual JPEG frames + JS flipbook player"

    def preview(self, fps, images=None, masks=None, quality=85):
        filename_prefix = "NVAnimPreview" + self.prefix_append
        full_output_folder, filename, counter, subfolder, _ = \
            folder_paths.get_save_image_path(filename_prefix, self.output_dir)

        results = []
        pil_images = []

        # Build PIL image list with optional mask compositing
        if images is not None and masks is not None:
            for i in range(images.shape[0]):
                img_np = (images[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                pil_img = Image.fromarray(img_np)
                if i < masks.shape[0]:
                    mask_np = (masks[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                    mask_pil = Image.fromarray(mask_np, mode='L')
                    pil_img = pil_img.convert("RGBA")
                    pil_img.putalpha(mask_pil)
                pil_images.append(pil_img)

        elif images is not None:
            for i in range(images.shape[0]):
                img_np = (images[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                pil_images.append(Image.fromarray(img_np))

        elif masks is not None:
            for i in range(masks.shape[0]):
                mask_np = (masks[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                pil_images.append(Image.fromarray(mask_np))

        else:
            return {"ui": {"frames": [], "fps": [fps], "frame_count": [0]}}

        # Save each frame as individual JPEG
        num_frames = len(pil_images)
        for i, pil_img in enumerate(pil_images):
            # JPEG doesn't support alpha — flatten RGBA onto black background
            if pil_img.mode == "RGBA":
                background = Image.new("RGB", pil_img.size, (0, 0, 0))
                background.paste(pil_img, mask=pil_img.split()[3])
                pil_img = background
            elif pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")

            frame_file = f"{filename}_{counter:05}_{i:04}.jpg"
            pil_img.save(
                os.path.join(full_output_folder, frame_file),
                quality=quality,
            )
            results.append({
                "filename": frame_file,
                "subfolder": subfolder,
                "type": self.type,
            })

        return {
            "ui": {
                "frames": results,
                "fps": [fps],
                "frame_count": [num_frames],
            }
        }


NODE_CLASS_MAPPINGS = {
    "NV_PreviewAnimation": NV_PreviewAnimation,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_PreviewAnimation": "NV Preview Animation",
}
