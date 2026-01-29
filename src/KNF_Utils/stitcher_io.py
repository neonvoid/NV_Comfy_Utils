"""
NV Stitcher Save/Load Nodes

Saves and loads STITCHER objects from the comfy-inpaint-crop-fork nodes.
This allows workflows to be split across separate runs or machines.

The STITCHER object contains:
- Metadata (algorithms, blend settings, coordinates) -> JSON file
- Tensor data (canvas_image, cropped_mask_for_blend, original_frames) -> Image/video files

File structure:
  stitcher_data/
    metadata.json          # All non-tensor data
    canvas_images/         # Original frames at canvas size
      frame_0000.png
      frame_0001.png
      ...
    blend_masks/           # Blend masks for stitching
      frame_0000.png
      ...
    original_frames/       # Original frames for skipped indices
      frame_0000.png
      ...
"""

import os
import json
import torch
import numpy as np
from PIL import Image


def save_tensor_as_image(tensor, filepath):
    """Save a single image tensor [H, W, C] or [H, W] as PNG."""
    if tensor.dim() == 2:
        # Mask: [H, W] -> grayscale
        arr = (tensor.cpu().numpy() * 255).astype(np.uint8)
        img = Image.fromarray(arr, mode='L')
    elif tensor.dim() == 3:
        # Image: [H, W, C]
        arr = (tensor.cpu().numpy() * 255).astype(np.uint8)
        if arr.shape[2] == 4:
            img = Image.fromarray(arr, mode='RGBA')
        else:
            img = Image.fromarray(arr, mode='RGB')
    else:
        raise ValueError(f"Unexpected tensor dimensions: {tensor.shape}")

    img.save(filepath, compress_level=1)  # Fast compression


def load_image_as_tensor(filepath, is_mask=False, device='cpu'):
    """Load PNG as tensor."""
    img = Image.open(filepath)
    arr = np.array(img).astype(np.float32) / 255.0

    if is_mask:
        # Grayscale mask
        if arr.ndim == 3:
            arr = arr[:, :, 0]  # Take first channel
        return torch.from_numpy(arr).to(device)
    else:
        # Color image
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.shape[2] == 4:
            arr = arr[:, :, :3]  # Drop alpha
        return torch.from_numpy(arr).to(device)


class NV_SaveStitcher:
    """
    Save a STITCHER object to disk for later use.

    This allows you to:
    - Split inpainting workflows across multiple runs
    - Process crops on one machine, stitch on another
    - Resume interrupted workflows

    The stitcher is saved as a directory containing:
    - metadata.json: Coordinates and settings
    - canvas_images/: Original frames at canvas size
    - blend_masks/: Masks for blending during stitch
    - original_frames/: Frames that were skipped (no mask)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stitcher": ("STITCHER",),
                "output_path": ("STRING", {
                    "default": "stitcher_data",
                    "tooltip": "Directory path to save the stitcher data"
                }),
            },
            "optional": {
                "save_canvas_images": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Save canvas images (required for stitching). Disable to save only metadata."
                }),
                "image_format": (["png", "jpg"], {
                    "default": "png",
                    "tooltip": "Image format for saved frames. PNG is lossless, JPG is smaller."
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("saved_path", "save_report",)
    OUTPUT_NODE = True
    FUNCTION = "save_stitcher"
    CATEGORY = "NV_Utils/Stitcher"
    DESCRIPTION = "Save a STITCHER object to disk for later use with NV_LoadStitcher."

    def save_stitcher(self, stitcher, output_path, save_canvas_images=True, image_format="png"):
        """Save the stitcher object to disk."""

        # Create output directory
        os.makedirs(output_path, exist_ok=True)

        report_lines = [
            "=" * 60,
            "STITCHER SAVE REPORT",
            "=" * 60,
            f"Output path: {output_path}",
        ]

        # Extract metadata (non-tensor fields)
        metadata = {
            'version': '1.0',
            'downscale_algorithm': stitcher.get('downscale_algorithm', 'bilinear'),
            'upscale_algorithm': stitcher.get('upscale_algorithm', 'bilinear'),
            'blend_pixels': stitcher.get('blend_pixels', 32),
            'canvas_to_orig_x': stitcher.get('canvas_to_orig_x', []),
            'canvas_to_orig_y': stitcher.get('canvas_to_orig_y', []),
            'canvas_to_orig_w': stitcher.get('canvas_to_orig_w', []),
            'canvas_to_orig_h': stitcher.get('canvas_to_orig_h', []),
            'cropped_to_canvas_x': stitcher.get('cropped_to_canvas_x', []),
            'cropped_to_canvas_y': stitcher.get('cropped_to_canvas_y', []),
            'cropped_to_canvas_w': stitcher.get('cropped_to_canvas_w', []),
            'cropped_to_canvas_h': stitcher.get('cropped_to_canvas_h', []),
            'skipped_indices': stitcher.get('skipped_indices', []),
            'total_frames': stitcher.get('total_frames', 0),
        }

        # Count frames
        num_canvas_images = len(stitcher.get('canvas_image', []))
        num_blend_masks = len(stitcher.get('cropped_mask_for_blend', []))
        num_original_frames = len(stitcher.get('original_frames', []))

        report_lines.append(f"Canvas images: {num_canvas_images}")
        report_lines.append(f"Blend masks: {num_blend_masks}")
        report_lines.append(f"Skipped frames: {num_original_frames}")
        report_lines.append(f"Total frames: {metadata['total_frames']}")

        # Save canvas images
        if save_canvas_images and num_canvas_images > 0:
            canvas_dir = os.path.join(output_path, "canvas_images")
            os.makedirs(canvas_dir, exist_ok=True)

            for i, img_tensor in enumerate(stitcher['canvas_image']):
                ext = image_format
                filepath = os.path.join(canvas_dir, f"frame_{i:04d}.{ext}")
                save_tensor_as_image(img_tensor, filepath)

            metadata['canvas_images_dir'] = "canvas_images"
            metadata['canvas_images_count'] = num_canvas_images
            metadata['canvas_images_format'] = image_format
            report_lines.append(f"Saved {num_canvas_images} canvas images to {canvas_dir}")

        # Save blend masks
        if num_blend_masks > 0:
            mask_dir = os.path.join(output_path, "blend_masks")
            os.makedirs(mask_dir, exist_ok=True)

            for i, mask_tensor in enumerate(stitcher['cropped_mask_for_blend']):
                filepath = os.path.join(mask_dir, f"frame_{i:04d}.png")
                save_tensor_as_image(mask_tensor, filepath)

            metadata['blend_masks_dir'] = "blend_masks"
            metadata['blend_masks_count'] = num_blend_masks
            report_lines.append(f"Saved {num_blend_masks} blend masks to {mask_dir}")

        # Save original frames (for skipped indices)
        if num_original_frames > 0:
            orig_dir = os.path.join(output_path, "original_frames")
            os.makedirs(orig_dir, exist_ok=True)

            for i, img_tensor in enumerate(stitcher['original_frames']):
                ext = image_format
                filepath = os.path.join(orig_dir, f"frame_{i:04d}.{ext}")
                save_tensor_as_image(img_tensor, filepath)

            metadata['original_frames_dir'] = "original_frames"
            metadata['original_frames_count'] = num_original_frames
            metadata['original_frames_format'] = image_format
            report_lines.append(f"Saved {num_original_frames} original frames to {orig_dir}")

        # Save metadata JSON
        metadata_path = os.path.join(output_path, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        report_lines.append(f"Saved metadata to {metadata_path}")
        report_lines.append("=" * 60)

        report = "\n".join(report_lines)
        print(report)

        return (output_path, report)


class NV_LoadStitcher:
    """
    Load a STITCHER object from disk.

    Use this to restore a stitcher saved with NV_SaveStitcher.
    The loaded stitcher can be used with InpaintStitch to
    composite inpainted results back into the original frames.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_path": ("STRING", {
                    "default": "stitcher_data",
                    "tooltip": "Directory path containing the saved stitcher data"
                }),
            },
        }

    RETURN_TYPES = ("STITCHER", "STRING",)
    RETURN_NAMES = ("stitcher", "load_report",)
    FUNCTION = "load_stitcher"
    CATEGORY = "NV_Utils/Stitcher"
    DESCRIPTION = "Load a STITCHER object saved with NV_SaveStitcher."

    def load_stitcher(self, input_path):
        """Load the stitcher object from disk."""

        import comfy.model_management
        intermediate = comfy.model_management.intermediate_device()

        report_lines = [
            "=" * 60,
            "STITCHER LOAD REPORT",
            "=" * 60,
            f"Input path: {input_path}",
        ]

        # Load metadata
        metadata_path = os.path.join(input_path, "metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        version = metadata.get('version', '1.0')
        report_lines.append(f"Stitcher version: {version}")

        # Reconstruct stitcher dict
        stitcher = {
            'downscale_algorithm': metadata.get('downscale_algorithm', 'bilinear'),
            'upscale_algorithm': metadata.get('upscale_algorithm', 'bilinear'),
            'blend_pixels': metadata.get('blend_pixels', 32),
            'canvas_to_orig_x': metadata.get('canvas_to_orig_x', []),
            'canvas_to_orig_y': metadata.get('canvas_to_orig_y', []),
            'canvas_to_orig_w': metadata.get('canvas_to_orig_w', []),
            'canvas_to_orig_h': metadata.get('canvas_to_orig_h', []),
            'cropped_to_canvas_x': metadata.get('cropped_to_canvas_x', []),
            'cropped_to_canvas_y': metadata.get('cropped_to_canvas_y', []),
            'cropped_to_canvas_w': metadata.get('cropped_to_canvas_w', []),
            'cropped_to_canvas_h': metadata.get('cropped_to_canvas_h', []),
            'skipped_indices': metadata.get('skipped_indices', []),
            'total_frames': metadata.get('total_frames', 0),
            'canvas_image': [],
            'cropped_mask_for_blend': [],
            'original_frames': [],
        }

        # Load canvas images
        canvas_dir = metadata.get('canvas_images_dir')
        if canvas_dir:
            canvas_path = os.path.join(input_path, canvas_dir)
            img_format = metadata.get('canvas_images_format', 'png')
            count = metadata.get('canvas_images_count', 0)

            for i in range(count):
                filepath = os.path.join(canvas_path, f"frame_{i:04d}.{img_format}")
                if os.path.exists(filepath):
                    tensor = load_image_as_tensor(filepath, is_mask=False, device=intermediate)
                    stitcher['canvas_image'].append(tensor)

            report_lines.append(f"Loaded {len(stitcher['canvas_image'])} canvas images")

        # Load blend masks
        mask_dir = metadata.get('blend_masks_dir')
        if mask_dir:
            mask_path = os.path.join(input_path, mask_dir)
            count = metadata.get('blend_masks_count', 0)

            for i in range(count):
                filepath = os.path.join(mask_path, f"frame_{i:04d}.png")
                if os.path.exists(filepath):
                    tensor = load_image_as_tensor(filepath, is_mask=True, device=intermediate)
                    stitcher['cropped_mask_for_blend'].append(tensor)

            report_lines.append(f"Loaded {len(stitcher['cropped_mask_for_blend'])} blend masks")

        # Load original frames
        orig_dir = metadata.get('original_frames_dir')
        if orig_dir:
            orig_path = os.path.join(input_path, orig_dir)
            img_format = metadata.get('original_frames_format', 'png')
            count = metadata.get('original_frames_count', 0)

            for i in range(count):
                filepath = os.path.join(orig_path, f"frame_{i:04d}.{img_format}")
                if os.path.exists(filepath):
                    tensor = load_image_as_tensor(filepath, is_mask=False, device=intermediate)
                    stitcher['original_frames'].append(tensor)

            report_lines.append(f"Loaded {len(stitcher['original_frames'])} original frames")

        report_lines.append(f"Total frames in stitcher: {stitcher['total_frames']}")
        report_lines.append(f"Skipped indices: {stitcher['skipped_indices']}")
        report_lines.append("=" * 60)

        report = "\n".join(report_lines)
        print(report)

        return (stitcher, report)


class NV_StitcherInfo:
    """
    Display information about a STITCHER object.

    Shows frame counts, coordinate ranges, and other metadata
    useful for debugging inpaint crop/stitch workflows.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stitcher": ("STITCHER",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    FUNCTION = "get_info"
    CATEGORY = "NV_Utils/Stitcher"
    DESCRIPTION = "Display information about a STITCHER object."

    def get_info(self, stitcher):
        """Extract and format stitcher info."""

        lines = [
            "=" * 60,
            "STITCHER INFO",
            "=" * 60,
        ]

        # Settings
        lines.append("SETTINGS:")
        lines.append(f"  Downscale algorithm: {stitcher.get('downscale_algorithm', 'N/A')}")
        lines.append(f"  Upscale algorithm: {stitcher.get('upscale_algorithm', 'N/A')}")
        lines.append(f"  Blend pixels: {stitcher.get('blend_pixels', 'N/A')}")

        # Frame counts
        lines.append("")
        lines.append("FRAME COUNTS:")
        lines.append(f"  Total frames: {stitcher.get('total_frames', 0)}")
        lines.append(f"  Canvas images: {len(stitcher.get('canvas_image', []))}")
        lines.append(f"  Blend masks: {len(stitcher.get('cropped_mask_for_blend', []))}")
        lines.append(f"  Skipped frames: {len(stitcher.get('skipped_indices', []))}")

        if stitcher.get('skipped_indices'):
            lines.append(f"  Skipped indices: {stitcher['skipped_indices']}")

        # Coordinate summary
        canvas_x = stitcher.get('canvas_to_orig_x', [])
        canvas_y = stitcher.get('canvas_to_orig_y', [])
        canvas_w = stitcher.get('canvas_to_orig_w', [])
        canvas_h = stitcher.get('canvas_to_orig_h', [])

        if canvas_x:
            lines.append("")
            lines.append("CANVAS TO ORIGINAL MAPPING:")
            lines.append(f"  X range: {min(canvas_x)} - {max(canvas_x)}")
            lines.append(f"  Y range: {min(canvas_y)} - {max(canvas_y)}")
            lines.append(f"  W range: {min(canvas_w)} - {max(canvas_w)}")
            lines.append(f"  H range: {min(canvas_h)} - {max(canvas_h)}")

        cropped_x = stitcher.get('cropped_to_canvas_x', [])
        cropped_y = stitcher.get('cropped_to_canvas_y', [])
        cropped_w = stitcher.get('cropped_to_canvas_w', [])
        cropped_h = stitcher.get('cropped_to_canvas_h', [])

        if cropped_x:
            lines.append("")
            lines.append("CROPPED TO CANVAS MAPPING:")
            lines.append(f"  X range: {min(cropped_x)} - {max(cropped_x)}")
            lines.append(f"  Y range: {min(cropped_y)} - {max(cropped_y)}")
            lines.append(f"  W range: {min(cropped_w)} - {max(cropped_w)}")
            lines.append(f"  H range: {min(cropped_h)} - {max(cropped_h)}")

        # Canvas image info
        canvas_images = stitcher.get('canvas_image', [])
        if canvas_images:
            first_img = canvas_images[0]
            if hasattr(first_img, 'shape'):
                lines.append("")
                lines.append("CANVAS IMAGE INFO:")
                lines.append(f"  Shape: {first_img.shape}")
                lines.append(f"  Device: {first_img.device if hasattr(first_img, 'device') else 'N/A'}")
                lines.append(f"  Dtype: {first_img.dtype}")

        lines.append("=" * 60)

        info = "\n".join(lines)
        print(info)

        return (info,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "NV_SaveStitcher": NV_SaveStitcher,
    "NV_LoadStitcher": NV_LoadStitcher,
    "NV_StitcherInfo": NV_StitcherInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_SaveStitcher": "NV Save Stitcher",
    "NV_LoadStitcher": "NV Load Stitcher",
    "NV_StitcherInfo": "NV Stitcher Info",
}
