"""
NV BBox Saver/Loader - Save and load masks with aspect ratio metadata.

Useful for persisting bbox selections from NV_BBoxCreator for reuse.
"""

import os
import json
import torch
import numpy as np
from PIL import Image
import folder_paths


class NV_BBoxSaver:
    """
    Save a mask and its associated metadata (aspect ratio, dimensions) to disk.
    Creates a PNG mask file and a JSON metadata file.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "filename_prefix": ("STRING", {"default": "bbox_mask"}),
            },
            "optional": {
                "save_directory": ("STRING", {
                    "default": "",
                    "placeholder": "Leave empty for output dir, or enter custom path",
                    "tooltip": "Custom directory to save files. Empty = ComfyUI output directory."
                }),
                "aspect_ratio": ("STRING", {"default": "Free"}),
                "ratio_value": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0}),
                "width": ("INT", {"default": 0, "min": 0, "max": 16384}),
                "height": ("INT", {"default": 0, "min": 0, "max": 16384}),
                "positive_boxes": ("SAM3_BOXES_PROMPT",),
                "negative_boxes": ("SAM3_BOXES_PROMPT",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filepath",)
    FUNCTION = "save_bbox"
    CATEGORY = "NV_Utils/io"
    OUTPUT_NODE = True

    def save_bbox(
        self,
        mask,
        filename_prefix,
        save_directory="",
        aspect_ratio="Free",
        ratio_value=1.0,
        width=0,
        height=0,
        positive_boxes=None,
        negative_boxes=None,
    ):
        """
        Save mask and metadata to specified directory.
        """
        # Get output directory - use custom path if provided, else default output
        if save_directory and save_directory.strip():
            output_dir = save_directory.strip()
            # Create directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = folder_paths.get_output_directory()

        # Generate unique filename
        counter = 1
        while True:
            base_name = f"{filename_prefix}_{counter:05d}"
            mask_path = os.path.join(output_dir, f"{base_name}.png")
            meta_path = os.path.join(output_dir, f"{base_name}.json")
            if not os.path.exists(mask_path) and not os.path.exists(meta_path):
                break
            counter += 1

        # Convert mask to PIL Image and save
        # Mask shape is [B, H, W] - take first mask
        if len(mask.shape) == 3:
            mask_np = mask[0].cpu().numpy()
        else:
            mask_np = mask.cpu().numpy()

        # Convert to uint8 (0-255)
        mask_uint8 = (mask_np * 255).astype(np.uint8)
        mask_img = Image.fromarray(mask_uint8, mode='L')
        mask_img.save(mask_path)

        # Build metadata
        metadata = {
            "aspect_ratio": aspect_ratio,
            "ratio_value": float(ratio_value),
            "width": int(width),
            "height": int(height),
            "mask_width": int(mask_np.shape[1]),
            "mask_height": int(mask_np.shape[0]),
        }

        # Include SAM3 boxes if provided
        if positive_boxes is not None:
            metadata["positive_boxes"] = positive_boxes
        if negative_boxes is not None:
            metadata["negative_boxes"] = negative_boxes

        # Save metadata
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"[NV_BBoxSaver] Saved mask to: {mask_path}")
        print(f"[NV_BBoxSaver] Saved metadata to: {meta_path}")

        return (mask_path,)


class NV_BBoxLoader:
    """
    Load a mask and its associated metadata from disk.
    Reads a PNG mask file and its JSON metadata file.

    Use dropdown to select from output directory, or provide custom_path to load from anywhere.
    """

    @classmethod
    def INPUT_TYPES(cls):
        # Get list of saved bbox masks from output directory
        output_dir = folder_paths.get_output_directory()
        mask_files = []
        if os.path.exists(output_dir):
            for f in os.listdir(output_dir):
                if f.endswith('.png'):
                    # Check if corresponding JSON exists
                    json_path = os.path.join(output_dir, f.replace('.png', '.json'))
                    if os.path.exists(json_path):
                        mask_files.append(f)

        if not mask_files:
            mask_files = ["none"]

        return {
            "required": {
                "mask_file": (sorted(mask_files), {"default": mask_files[0] if mask_files else "none"}),
            },
            "optional": {
                "custom_path": ("STRING", {
                    "default": "",
                    "placeholder": "Optional: full path to mask.png (overrides dropdown)",
                    "tooltip": "If provided, loads from this path instead of the dropdown selection."
                }),
            },
        }

    RETURN_TYPES = ("MASK", "STRING", "FLOAT", "INT", "INT", "SAM3_BOXES_PROMPT", "SAM3_BOXES_PROMPT")
    RETURN_NAMES = ("mask", "aspect_ratio", "ratio_value", "width", "height", "positive_boxes", "negative_boxes")
    FUNCTION = "load_bbox"
    CATEGORY = "NV_Utils/io"

    @classmethod
    def IS_CHANGED(cls, mask_file, custom_path=""):
        # Determine which path to check
        if custom_path and custom_path.strip():
            mask_path = custom_path.strip()
        else:
            output_dir = folder_paths.get_output_directory()
            mask_path = os.path.join(output_dir, mask_file)

        if os.path.exists(mask_path):
            return os.path.getmtime(mask_path)
        return float("nan")

    def load_bbox(self, mask_file, custom_path=""):
        """
        Load mask and metadata from output directory or custom path.
        """
        # Determine which path to use
        if custom_path and custom_path.strip():
            mask_path = custom_path.strip()
            print(f"[NV_BBoxLoader] Using custom path: {mask_path}")
        else:
            output_dir = folder_paths.get_output_directory()
            mask_path = os.path.join(output_dir, mask_file)

        meta_path = mask_path.replace('.png', '.json')

        # Load mask
        if not os.path.exists(mask_path):
            print(f"[NV_BBoxLoader] Mask file not found: {mask_path}")
            # Return empty mask
            mask = torch.zeros((1, 512, 512), dtype=torch.float32)
            return (mask, "Free", 1.0, 0, 0, {"boxes": [], "labels": []}, {"boxes": [], "labels": []})

        mask_img = Image.open(mask_path).convert('L')
        mask_np = np.array(mask_img).astype(np.float32) / 255.0
        mask = torch.from_numpy(mask_np).unsqueeze(0)  # [1, H, W]

        # Load metadata
        metadata = {}
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                metadata = json.load(f)

        aspect_ratio = metadata.get("aspect_ratio", "Free")
        ratio_value = float(metadata.get("ratio_value", 1.0))
        width = int(metadata.get("width", 0))
        height = int(metadata.get("height", 0))
        positive_boxes = metadata.get("positive_boxes", {"boxes": [], "labels": []})
        negative_boxes = metadata.get("negative_boxes", {"boxes": [], "labels": []})

        print(f"[NV_BBoxLoader] Loaded mask from: {mask_path}")
        print(f"[NV_BBoxLoader] Aspect ratio: {aspect_ratio}, Ratio: {ratio_value}, Size: {width}x{height}")

        return (mask, aspect_ratio, ratio_value, width, height, positive_boxes, negative_boxes)


class NV_BBoxLoaderPath:
    """
    Load a mask and metadata from a specific file path (not from dropdown).
    Useful for dynamic/programmatic loading.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask_path": ("STRING", {"default": "", "placeholder": "path/to/mask.png"}),
            },
        }

    RETURN_TYPES = ("MASK", "STRING", "FLOAT", "INT", "INT", "SAM3_BOXES_PROMPT", "SAM3_BOXES_PROMPT")
    RETURN_NAMES = ("mask", "aspect_ratio", "ratio_value", "width", "height", "positive_boxes", "negative_boxes")
    FUNCTION = "load_bbox"
    CATEGORY = "NV_Utils/io"

    @classmethod
    def IS_CHANGED(cls, mask_path):
        if os.path.exists(mask_path):
            return os.path.getmtime(mask_path)
        return float("nan")

    def load_bbox(self, mask_path):
        """
        Load mask and metadata from specified path.
        """
        meta_path = mask_path.replace('.png', '.json')

        # Load mask
        if not os.path.exists(mask_path):
            print(f"[NV_BBoxLoaderPath] Mask file not found: {mask_path}")
            mask = torch.zeros((1, 512, 512), dtype=torch.float32)
            return (mask, "Free", 1.0, 0, 0, {"boxes": [], "labels": []}, {"boxes": [], "labels": []})

        mask_img = Image.open(mask_path).convert('L')
        mask_np = np.array(mask_img).astype(np.float32) / 255.0
        mask = torch.from_numpy(mask_np).unsqueeze(0)  # [1, H, W]

        # Load metadata
        metadata = {}
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                metadata = json.load(f)

        aspect_ratio = metadata.get("aspect_ratio", "Free")
        ratio_value = float(metadata.get("ratio_value", 1.0))
        width = int(metadata.get("width", 0))
        height = int(metadata.get("height", 0))
        positive_boxes = metadata.get("positive_boxes", {"boxes": [], "labels": []})
        negative_boxes = metadata.get("negative_boxes", {"boxes": [], "labels": []})

        print(f"[NV_BBoxLoaderPath] Loaded mask from: {mask_path}")

        return (mask, aspect_ratio, ratio_value, width, height, positive_boxes, negative_boxes)


class NV_ImageSaver:
    """
    Save images to a custom directory with configurable format and quality.
    More flexible than the built-in SaveImage node.
    Optionally saves workflow JSON alongside images for reproducibility.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "image"}),
                "format": (["png", "jpg", "webp"], {"default": "png"}),
            },
            "optional": {
                "save_directory": ("STRING", {
                    "default": "",
                    "placeholder": "Leave empty for output dir, or enter custom path",
                    "tooltip": "Custom directory to save files. Empty = ComfyUI output directory."
                }),
                "quality": ("INT", {
                    "default": 95,
                    "min": 1,
                    "max": 100,
                    "tooltip": "Quality for JPG/WebP (1-100). Ignored for PNG."
                }),
                "png_compression": ("INT", {
                    "default": 6,
                    "min": 0,
                    "max": 9,
                    "tooltip": "PNG compression level (0=none, 9=max). Only for PNG format."
                }),
                "save_workflow": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Save workflow JSON alongside the image for reproducibility."
                }),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filepath",)
    FUNCTION = "save_image"
    CATEGORY = "NV_Utils/io"
    OUTPUT_NODE = True

    def save_image(
        self,
        images,
        filename_prefix,
        format="png",
        save_directory="",
        quality=95,
        png_compression=6,
        save_workflow=True,
        prompt=None,
        extra_pnginfo=None,
    ):
        """
        Save images to specified directory with optional workflow.
        """
        # Get output directory - use custom path if provided, else default output
        if save_directory and save_directory.strip():
            output_dir = save_directory.strip()
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = folder_paths.get_output_directory()

        saved_paths = []
        workflow_saved = False

        for i, image in enumerate(images):
            # Generate unique filename
            counter = 1
            while True:
                if len(images) > 1:
                    base_name = f"{filename_prefix}_{counter:05d}_{i:03d}"
                else:
                    base_name = f"{filename_prefix}_{counter:05d}"
                img_path = os.path.join(output_dir, f"{base_name}.{format}")
                if not os.path.exists(img_path):
                    break
                counter += 1

            # Convert tensor to PIL Image
            # Image shape is [H, W, C] in range [0, 1]
            img_np = (image.cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)

            # Save with format-specific options
            if format == "png":
                # For PNG, we can embed workflow in metadata
                from PIL import PngImagePlugin
                metadata = PngImagePlugin.PngInfo()
                if save_workflow:
                    if prompt is not None:
                        metadata.add_text("prompt", json.dumps(prompt))
                    if extra_pnginfo is not None:
                        for key, value in extra_pnginfo.items():
                            metadata.add_text(key, json.dumps(value))
                pil_img.save(img_path, format="PNG", compress_level=png_compression, pnginfo=metadata)
            elif format == "jpg":
                # Convert to RGB if necessary (remove alpha)
                if pil_img.mode == "RGBA":
                    pil_img = pil_img.convert("RGB")
                pil_img.save(img_path, format="JPEG", quality=quality)
            elif format == "webp":
                pil_img.save(img_path, format="WEBP", quality=quality)

            saved_paths.append(img_path)
            print(f"[NV_ImageSaver] Saved: {img_path}")

            # Save workflow JSON file (once per batch, for non-PNG or if explicit)
            if save_workflow and not workflow_saved:
                workflow_path = os.path.join(output_dir, f"{base_name}_workflow.json")
                workflow_data = {}
                if prompt is not None:
                    workflow_data["prompt"] = prompt
                if extra_pnginfo is not None:
                    workflow_data.update(extra_pnginfo)
                if workflow_data:
                    with open(workflow_path, 'w') as f:
                        json.dump(workflow_data, f, indent=2)
                    print(f"[NV_ImageSaver] Saved workflow: {workflow_path}")
                workflow_saved = True

        # Return first path (or last if batch)
        return (saved_paths[0] if saved_paths else "",)


NODE_CLASS_MAPPINGS = {
    "NV_BBoxSaver": NV_BBoxSaver,
    "NV_BBoxLoader": NV_BBoxLoader,
    "NV_BBoxLoaderPath": NV_BBoxLoaderPath,
    "NV_ImageSaver": NV_ImageSaver,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_BBoxSaver": "NV BBox Saver",
    "NV_BBoxLoader": "NV BBox Loader",
    "NV_BBoxLoaderPath": "NV BBox Loader (Path)",
    "NV_ImageSaver": "NV Image Saver",
}
