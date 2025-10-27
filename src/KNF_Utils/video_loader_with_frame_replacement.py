"""
Video Loader with Frame Replacement
Extends VideoHelperSuite's LoadVideoFFmpegPath with first/last frame replacement capability
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Import VideoHelperSuite components
VHS_AVAILABLE = False
load_video = None
ffmpeg_frame_generator = None
get_load_formats = None
validate_path = None
hash_path = None
is_url = None
try_download_video = None
strip_path = None

try:
    # Try direct import first (if VHS is in Python path)
    from videohelpersuite.load_video_nodes import (
        load_video,
        ffmpeg_frame_generator,
        get_load_formats,
        validate_path,
        hash_path,
        is_url,
        try_download_video,
        strip_path,
        video_extensions,
        BIGMAX,
        DIMMAX,
        floatOrInt,
        imageOrLatent
    )
    VHS_AVAILABLE = True
    print("[NV_Video_Loader_Path] Successfully imported VideoHelperSuite")
except ImportError as e1:
    # Try adding to path and importing
    try:
        custom_nodes_path = Path(__file__).parent.parent.parent.parent
        videohelpersuite_path = custom_nodes_path / "comfyui-videohelpersuite"
        
        if not videohelpersuite_path.exists():
            raise ImportError(f"VideoHelperSuite path does not exist: {videohelpersuite_path}")
        
        sys.path.insert(0, str(videohelpersuite_path))
        
        from videohelpersuite.load_video_nodes import (
            load_video,
            ffmpeg_frame_generator,
            get_load_formats,
            validate_path,
            hash_path,
            is_url,
            try_download_video,
            strip_path,
            video_extensions,
            BIGMAX,
            DIMMAX,
            floatOrInt,
            imageOrLatent
        )
        VHS_AVAILABLE = True
        print("[NV_Video_Loader_Path] Successfully imported VideoHelperSuite (via path)")
    except Exception as e2:
        print("=" * 80)
        print("[NV_Video_Loader_Path] ERROR: Could not import VideoHelperSuite!")
        print(f"[NV_Video_Loader_Path] Error 1: {e1}")
        print(f"[NV_Video_Loader_Path] Error 2: {e2}")
        print(f"[NV_Video_Loader_Path] Expected path: {videohelpersuite_path if 'videohelpersuite_path' in locals() else 'unknown'}")
        print("[NV_Video_Loader_Path] Please install comfyui-videohelpersuite:")
        print("[NV_Video_Loader_Path]   cd ComfyUI/custom_nodes")
        print("[NV_Video_Loader_Path]   git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite comfyui-videohelpersuite")
        print("=" * 80)
        VHS_AVAILABLE = False

# Define fallbacks for when VHS is not available
if not VHS_AVAILABLE:
    video_extensions = ['webm', 'mp4', 'mkv', 'gif', 'mov']
    BIGMAX = 999999999
    DIMMAX = 16384
    floatOrInt = "FLOAT"
    imageOrLatent = "IMAGE"
    
    def get_load_formats():
        return (["None"], {"default": "None", "formats": {}})


class NV_Video_Loader_Path:
    """
    Video loader that extends VideoHelperSuite's LoadVideoFFmpegPath with frame replacement.
    Allows replacing the first and/or last frame of a loaded video with custom images.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        if not VHS_AVAILABLE:
            return {
                "required": {
                    "error": ("STRING", {"default": "VideoHelperSuite not found. Please install comfyui-videohelpersuite."}),
                }
            }
        
        return {
            "required": {
                "video": ("STRING", {
                    "placeholder": "X://insert/path/here.mp4", 
                    "vhs_path_extensions": video_extensions
                }),
                "force_rate": (floatOrInt, {"default": 0, "min": 0, "max": 60, "step": 1, "disable": 0}),
                "custom_width": ("INT", {"default": 0, "min": 0, "max": DIMMAX, 'disable': 0}),
                "custom_height": ("INT", {"default": 0, "min": 0, "max": DIMMAX, 'disable': 0}),
                "frame_load_cap": ("INT", {"default": 0, "min": 0, "max": BIGMAX, "step": 1, "disable": 0}),
                "start_time": ("FLOAT", {"default": 0, "min": 0, "max": BIGMAX, "step": .001, "widgetType": "VHSTIMESTAMP"}),
            },
            "optional": {
                "meta_batch": ("VHS_BatchManager",),
                "vae": ("VAE",),
                "format": get_load_formats() if VHS_AVAILABLE else (["None"],),
                "first_frame_image": ("IMAGE",),
                "last_frame_image": ("IMAGE",),
                "replace_first_frame": ("BOOLEAN", {"default": False}),
                "replace_last_frame": ("BOOLEAN", {"default": False}),
            },
            "hidden": {
                "force_size": "STRING",
                "unique_id": "UNIQUE_ID"
            },
        }
    
    CATEGORY = "KNF_Utils/Video"
    
    RETURN_TYPES = (imageOrLatent, "MASK", "AUDIO", "VHS_VIDEOINFO")
    RETURN_NAMES = ("IMAGE", "mask", "audio", "video_info")
    
    FUNCTION = "load_video_with_replacement"
    
    def load_video_with_replacement(self, video, force_rate=0, custom_width=0, custom_height=0, 
                                    frame_load_cap=0, start_time=0, meta_batch=None, vae=None, 
                                    format='None', first_frame_image=None, last_frame_image=None,
                                    replace_first_frame=False, replace_last_frame=False, 
                                    force_size=None, unique_id=None):
        """
        Load video using VideoHelperSuite and optionally replace first/last frames.
        
        Args:
            video: Path to video file
            force_rate: Force a specific frame rate (0 = use original)
            custom_width: Custom width (0 = use original)
            custom_height: Custom height (0 = use original)
            frame_load_cap: Maximum frames to load (0 = load all)
            start_time: Start time in seconds
            meta_batch: VHS BatchManager
            vae: VAE for encoding
            format: Video format preset (AnimateDiff, Mochi, etc.)
            first_frame_image: Image to replace first frame with
            last_frame_image: Image to replace last frame with
            replace_first_frame: Enable first frame replacement
            replace_last_frame: Enable last frame replacement
            
        Returns:
            Tuple of (images/latents, mask, audio, video_info)
        """
        if not VHS_AVAILABLE:
            raise RuntimeError("VideoHelperSuite is not available. Please install comfyui-videohelpersuite.")
        
        # Validate video path
        if video is None or validate_path(video) != True:
            raise Exception(f"video is not a valid path: {video}")
        
        # Handle URL downloads
        if is_url(video):
            video = try_download_video(video) or video
        
        # Load video using VideoHelperSuite's ffmpeg loader
        kwargs = {
            'video': video,
            'force_rate': force_rate,
            'custom_width': custom_width,
            'custom_height': custom_height,
            'frame_load_cap': frame_load_cap,
            'start_time': start_time,
            'meta_batch': meta_batch,
            'vae': vae,
            'format': format,
            'unique_id': unique_id,
        }
        
        image, frame_count, audio, video_info = load_video(**kwargs, generator=ffmpeg_frame_generator)
        
        # Extract mask if present
        mask = None
        if isinstance(image, dict):
            # Image is latent, can't replace frames
            if replace_first_frame or replace_last_frame:
                print("[NV_Video_Loader_Path] Warning: Frame replacement not supported with VAE encoding")
            return (image, None, audio, video_info)
        
        # Handle alpha channel for mask
        if image.size(3) == 4:
            mask = 1 - image[:,:,:,3]
            image = image[:,:,:,:3]
        else:
            mask = torch.zeros(image.size(0), 64, 64, device="cpu")
        
        # Perform frame replacements if enabled
        if (replace_first_frame and first_frame_image is not None) or \
           (replace_last_frame and last_frame_image is not None):
            image = self._replace_frames(
                image, 
                first_frame_image if replace_first_frame else None,
                last_frame_image if replace_last_frame else None,
                custom_width if custom_width > 0 else None,
                custom_height if custom_height > 0 else None
            )
        
        return (image, mask, audio, video_info)
    
    def _replace_frames(self, video_tensor, first_frame=None, last_frame=None, 
                       target_width=None, target_height=None):
        """
        Replace first and/or last frames of video tensor with provided images.
        
        Args:
            video_tensor: Video tensor (batch, height, width, channels)
            first_frame: Image tensor to replace first frame (or None)
            last_frame: Image tensor to replace last frame (or None)
            target_width: Target width to resize images to (or None for video size)
            target_height: Target height to resize images to (or None for video size)
            
        Returns:
            Modified video tensor
        """
        if video_tensor.size(0) == 0:
            print("[NV_Video_Loader_Path] Warning: Empty video tensor, cannot replace frames")
            return video_tensor
        
        # Get video dimensions
        num_frames, height, width, channels = video_tensor.shape
        target_width = target_width or width
        target_height = target_height or height
        
        # Make a copy to avoid modifying the original
        video_tensor = video_tensor.clone()
        
        # Replace first frame
        if first_frame is not None:
            try:
                processed_frame = self._prepare_frame(
                    first_frame, target_height, target_width, channels
                )
                video_tensor[0] = processed_frame
                print(f"[NV_Video_Loader_Path] Replaced first frame")
            except Exception as e:
                print(f"[NV_Video_Loader_Path] Error replacing first frame: {e}")
        
        # Replace last frame
        if last_frame is not None:
            try:
                processed_frame = self._prepare_frame(
                    last_frame, target_height, target_width, channels
                )
                video_tensor[-1] = processed_frame
                print(f"[NV_Video_Loader_Path] Replaced last frame")
            except Exception as e:
                print(f"[NV_Video_Loader_Path] Error replacing last frame: {e}")
        
        return video_tensor
    
    def _prepare_frame(self, image_tensor, target_height, target_width, target_channels):
        """
        Prepare an image tensor to match video frame requirements.
        
        Args:
            image_tensor: Input image tensor (batch, height, width, channels)
            target_height: Target height
            target_width: Target width
            target_channels: Target number of channels (3 for RGB)
            
        Returns:
            Processed frame tensor (height, width, channels)
        """
        # Handle batch dimension
        if len(image_tensor.shape) == 4:
            # Take first image from batch
            image_tensor = image_tensor[0]
        elif len(image_tensor.shape) == 3:
            # Already single image
            pass
        else:
            raise ValueError(f"Unexpected image tensor shape: {image_tensor.shape}")
        
        # Get current dimensions
        current_height, current_width, current_channels = image_tensor.shape
        
        # Handle channel mismatch
        if current_channels != target_channels:
            if current_channels == 4 and target_channels == 3:
                # RGBA to RGB - drop alpha
                image_tensor = image_tensor[:, :, :3]
            elif current_channels == 1 and target_channels == 3:
                # Grayscale to RGB - repeat channel
                image_tensor = image_tensor.repeat(1, 1, 3)
            elif current_channels == 3 and target_channels == 4:
                # RGB to RGBA - add alpha channel
                alpha = torch.ones((current_height, current_width, 1), dtype=image_tensor.dtype, device=image_tensor.device)
                image_tensor = torch.cat([image_tensor, alpha], dim=2)
            else:
                print(f"[NV_Video_Loader_Path] Warning: Channel mismatch {current_channels} -> {target_channels}, using as-is")
        
        # Resize if dimensions don't match
        if current_height != target_height or current_width != target_width:
            # Use torch resize (interpolate)
            # Need to rearrange dimensions for interpolate: (channels, height, width)
            image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
            image_tensor = torch.nn.functional.interpolate(
                image_tensor,
                size=(target_height, target_width),
                mode='bilinear',
                align_corners=False
            )
            image_tensor = image_tensor.squeeze(0).permute(1, 2, 0)  # Back to (H, W, C)
            print(f"[NV_Video_Loader_Path] Resized frame from {current_width}x{current_height} to {target_width}x{target_height}")
        
        return image_tensor
    
    @classmethod
    def IS_CHANGED(cls, video, **kwargs):
        if not VHS_AVAILABLE:
            return float("inf")
        return hash_path(video)
    
    @classmethod
    def VALIDATE_INPUTS(cls, video, **kwargs):
        if not VHS_AVAILABLE:
            return "VideoHelperSuite is not installed"
        return validate_path(video, allow_none=True)


# Node registration
NODE_CLASS_MAPPINGS = {
    "NV_Video_Loader_Path": NV_Video_Loader_Path
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_Video_Loader_Path": "NV Video Loader Path"
}

