"""
NV Preview Animation - Fast animated preview using individual JPEG frames + JS flipbook.

Replaces the slow animated WebP approach (PIL save_all) with per-frame JPEG encoding
(~50-100x faster) and a custom frontend canvas widget that animates them.
"""

import gc
import os
import random

import numpy as np
import torch
from PIL import Image

import folder_paths


def _tensor_to_uint8_np(t):
    """Convert a [H, W, C] or [H, W] image/mask tensor in [0, 1] to a uint8
    numpy array. Conversion happens on the SOURCE device, so the only CPU
    allocation is the final uint8 buffer — avoids the 3× intermediate fp16
    numpy arrays that the legacy `(t.cpu().numpy() * 255).clip(0,255).astype`
    path produces (~36 MB per 1080p frame in transient allocations, which
    cumulatively starves the system on back-to-back queue runs).
    """
    return (t.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).cpu().numpy()


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

        if images is not None:
            num_frames = images.shape[0]
        elif masks is not None:
            num_frames = masks.shape[0]
        else:
            return {"ui": {"frames": [], "fps": [fps], "frame_count": [0]}}

        # Stream-process: convert each frame on its source device to uint8,
        # save immediately, release before processing the next frame. Avoids
        # two failure modes that surface under CPU RAM pressure on back-to-
        # back queue runs:
        #   (1) fp16 input → intermediate fp16 numpy array allocations (~12 MB
        #       per 1080p frame × 3 stages) that fail when system is starved.
        #       The on-device conversion in _tensor_to_uint8_np produces a
        #       single uint8 buffer, eliminating the fp16 numpy intermediates.
        #   (2) accumulating all PIL frames before save (~2 GB for 277f@1080p).
        #       Streaming saves keep peak memory bounded by one frame.
        results = []
        width = height = 0

        for i in range(num_frames):
            if images is not None:
                pil_img = Image.fromarray(_tensor_to_uint8_np(images[i]))
                if masks is not None and i < masks.shape[0]:
                    mask_pil = Image.fromarray(_tensor_to_uint8_np(masks[i]), mode='L')
                    pil_img = pil_img.convert("RGBA")
                    pil_img.putalpha(mask_pil)
                    del mask_pil
            else:
                pil_img = Image.fromarray(_tensor_to_uint8_np(masks[i]), mode='L')

            if i == 0:
                width, height = pil_img.size

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
            # Explicit release of per-frame intermediates so they're reclaimed
            # before the next iteration's allocations under RAM pressure.
            del pil_img

        # Hint the allocator to release after a long video; cheap on a
        # short batch, helpful when this node runs during back-to-back queue
        # cycles where the scheduler doesn't naturally trigger collection.
        gc.collect()

        return {
            "ui": {
                "frames": results,
                "fps": [fps],
                "frame_count": [num_frames],
                "width": [width],
                "height": [height],
            }
        }


NODE_CLASS_MAPPINGS = {
    "NV_PreviewAnimation": NV_PreviewAnimation,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_PreviewAnimation": "NV Preview Animation",
}
