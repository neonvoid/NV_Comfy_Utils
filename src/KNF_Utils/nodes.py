from inspect import cleandoc
import torch
import numpy as np
import os
import random
from PIL import Image
import folder_paths
import node_helpers
import json
import base64
import requests
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple
import time
from datetime import datetime
import cv2

class KNF_Organizer:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        files = folder_paths.filter_files_content_types(files,["image"])
        return {
            "required": {
                "FirstFrame":(sorted(files),{"image_upload":True}),
                "firstframeOriginal":("STRING",{"multiline":True,"default":"FF_prompt"}),
                "firstFrameRestyled":("STRING",{"multiline":True,"default":"FF_Restyled_prompt"}),
                "videoPrompt":("STRING",{"multiline":True,"default":"vid_prompt"}),
                }
        }
            
    RETURN_TYPES = ("IMAGE","STRING","STRING","STRING")
    RETURN_NAMES = ("FirstFrame","firstframeOriginal","firstFrameRestyled","videoPrompt")
    FUNCTION = "KNF_Organizer"

    def KNF_Organizer(self, FirstFrame, firstframeOriginal, firstFrameRestyled, videoPrompt):
        """Load and return the selected image along with the text prompts."""
        try:
            image_path = folder_paths.get_annotated_filepath(FirstFrame)
            image = Image.open(image_path)
            
            # ComfyUI expects images as tensors with shape (batch, height, width, channels)
            image_array = np.array(image).astype(np.float32) / 255.0
            
            # Convert to tensor and add batch dimension
            image_tensor = torch.from_numpy(image_array)[None,]
            
            # Return the image tensor and the three text strings
            return (image_tensor, firstframeOriginal, firstFrameRestyled, videoPrompt)
            
        except Exception as e:
            print(f"Error loading image in KNF_Organizer: {e}")
            black_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            return (black_image, firstframeOriginal, firstFrameRestyled, videoPrompt)

    CATEGORY = "Example"

class GeminiVideoCaptioner:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm', '.wmv', '.flv', '.m4v'))]
        files = folder_paths.filter_files_content_types(files, ["video"])
        return {
            "required": {
                "video_file": (sorted(files), {"video_upload": True}),
                "prompt_text": ("STRING", {"multiline": True, "default": "Describe this video in detail."}),
                "api_key": ("STRING", {"default": ""}),
                "model": (["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.5-pro"], {"default": "gemini-2.5-pro"}),
                "fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 120.0, "step": 0.1}),
            },
            "optional": {
                "video_tensor": ("IMAGE",)
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("caption",)
    FUNCTION = "caption_video"
    CATEGORY = "KNF_Utils"
    OUTPUT_NODE = True

    def get_video_metadata(self, video_path: str) -> Optional[Dict]:
        """Get video metadata using ffprobe."""
        try:
            cmd = [
                'ffprobe', '-v', 'error', '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height,r_frame_rate,nb_frames,duration',
                '-of', 'json', video_path
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            if result.returncode != 0:
                print(f"ffprobe error: {result.stderr}")
                return None
                
            info = json.loads(result.stdout)
            if 'streams' not in info or not info['streams']:
                return None
                
            stream = info['streams'][0]
            width = stream.get('width')
            height = stream.get('height')
            
            # Calculate FPS
            r_frame_rate = stream.get('r_frame_rate', '0/0')
            try:
                num, denom = map(int, r_frame_rate.split('/'))
                fps = num / denom if denom else 0
            except:
                fps = 0
                
            nb_frames = int(stream.get('nb_frames', 0)) if stream.get('nb_frames') else 0
            duration = float(stream.get('duration', 0)) if stream.get('duration') else 0
            
            return {
                'width': width,
                'height': height,
                'fps': fps,
                'nb_frames': nb_frames,
                'duration': duration
            }
        except Exception as e:
            print(f"Warning: Could not get video metadata for {video_path}: {e}")
            return None

    def encode_file_to_base64(self, file_path: str) -> str:
        """Encode file to base64 string."""
        with open(file_path, "rb") as file:
            return base64.b64encode(file.read()).decode('utf-8')

    def get_mime_type(self, file_path: str) -> str:
        """Determine MIME type based on file extension."""
        extension = Path(file_path).suffix.lower()
        mime_types = {
            '.mp4': 'video/mp4',
            '.avi': 'video/x-msvideo',
            '.mov': 'video/quicktime',
            '.mkv': 'video/x-matroska',
            '.webm': 'video/webm',
            '.wmv': 'video/x-ms-wmv',
            '.flv': 'video/x-flv',
            '.m4v': 'video/mp4'
        }
        return mime_types.get(extension, 'video/mp4')

    def tensor_to_video(self, video_tensor, fps=30):
        """Convert video tensor to temporary video file."""
        try:
            # Handle both 4D (frames, height, width, channels) and 5D (batch, frames, height, width, channels) tensors
            if len(video_tensor.shape) == 4:
                # 4D tensor: (frames, height, width, channels)
                video_array = video_tensor.cpu().numpy()
            elif len(video_tensor.shape) == 5:
                # 5D tensor: (batch, frames, height, width, channels)
                video_array = video_tensor[0].cpu().numpy()  # Remove batch dimension
            else:
                raise ValueError(f"Expected 4D tensor (frames, height, width, channels) or 5D tensor (batch, frames, height, width, channels), got {video_tensor.shape}")
            
            # Ensure values are in [0, 255] range
            if video_array.max() <= 1.0:
                video_array = (video_array * 255).astype(np.uint8)
            else:
                video_array = video_array.astype(np.uint8)
            
            # Create temporary video file
            temp_dir = folder_paths.get_temp_directory()
            temp_video_path = os.path.join(temp_dir, f"temp_video_{int(time.time())}.mp4")
            
            # Use OpenCV to write video
            height, width = video_array.shape[1:3]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
            
            for frame in video_array:
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            out.release()
            return temp_video_path
            
        except Exception as e:
            print(f"Error converting tensor to video: {e}")
            return None

    def caption_video(self, video_file=None, prompt_text="Describe this video in detail.", api_key="", model="gemini-2.5-pro", fps=30.0, video_tensor=None):
        """Main function to caption video using Gemini API."""
        if not api_key:
            return ("Error: API key is required",)
        
        video_path = None
        
        # Handle video tensor input (from node link)
        if video_tensor is not None:
            print(f"Processing video tensor input at {fps} FPS...")
            video_path = self.tensor_to_video(video_tensor, fps)
            if video_path is None:
                return ("Error: Failed to convert video tensor to file",)
        # Handle video file input
        elif video_file:
            # Construct full path for ComfyUI video file
            if not os.path.isabs(video_file):
                input_dir = folder_paths.get_input_directory()
                video_path = os.path.join(input_dir, video_file)
            else:
                video_path = video_file
            
            if not os.path.exists(video_path):
                return (f"Error: Video file not found: {video_path}",)
        else:
            return ("Error: Either video file or video tensor must be provided",)
        
        try:
            # Get video metadata
            metadata = self.get_video_metadata(video_path)
            
            # Encode video to base64
            print("Encoding video to base64...")
            base64_data = self.encode_file_to_base64(video_path)
            mime_type = self.get_mime_type(video_path)
            
            # Prepare API request
            headers = {
                "Content-Type": "application/json"
            }
            
            payload = {
                "contents": [{
                    "parts": [
                        {"text": prompt_text},
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": base64_data
                            }
                        }
                    ]
                }]
            }
            
            api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
            
            print("Sending request to Gemini API...")
            
            # Make API request
            response = requests.post(api_url, headers=headers, json=payload, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                
                if 'candidates' in result and len(result['candidates']) > 0:
                    candidate = result['candidates'][0]
                    if 'content' in candidate and 'parts' in candidate['content']:
                        caption = candidate['content']['parts'][0]['text']
                        # Clean up temporary video file if it was created from tensor
                        if video_tensor is not None and os.path.exists(video_path):
                            try:
                                os.remove(video_path)
                                print(f"Cleaned up temporary video file: {video_path}")
                            except Exception as cleanup_error:
                                print(f"Warning: Could not clean up temporary file {video_path}: {cleanup_error}")
                        return (caption.strip(),)
                    else:
                        return ("Error: No content in API response",)
                else:
                    return ("Error: No candidates in API response",)
            else:
                return (f"API error: {response.status_code} - {response.text}",)
                
        except Exception as e:
            # Clean up temporary video file if it was created from tensor
            if video_tensor is not None and video_path and os.path.exists(video_path):
                try:
                    os.remove(video_path)
                    print(f"Cleaned up temporary video file after error: {video_path}")
                except Exception as cleanup_error:
                    print(f"Warning: Could not clean up temporary file {video_path}: {cleanup_error}")
            return (f"Exception: {str(e)}",)


class CustomVideoSaver:
    """
    Custom video saver that allows saving videos to a user-specified directory.
    Similar to ComfyUI's native video saving but with custom directory selection.
    """
    
    def __init__(self):
        self.type = "output"
        self.output_dir = folder_paths.get_output_directory()
        self._check_codec_support()
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_tensor": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "ComfyUI_video"}),
                "custom_directory": ("STRING", {"default": ""}),  # Empty = use default output
                "video_format": (["mp4", "avi", "mov", "mkv", "webm", "wmv"], {"default": "mp4"}),
                "fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 120.0, "step": 0.1}),
                "quality": ("INT", {"default": 18, "min": 0, "max": 51, "step": 1}),  # FFmpeg CRF quality
                "preserve_colors": ("BOOLEAN", {"default": True}),  # Enable color preservation mode
            },
            "optional": {
                "subfolder": ("STRING", {"default": ""}),
                "prompt": ("PROMPT",),
                "extra_pnginfo": ("EXTRA_PNGINFO",),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("video_path", "filename", "info")
    FUNCTION = "save_video"
    CATEGORY = "KNF_Utils/Video"
    OUTPUT_NODE = True
    
    def save_video(self, video_tensor, filename_prefix="ComfyUI_video", custom_directory="", 
                   video_format="mp4", fps=30.0, quality=18, preserve_colors=True, subfolder="", prompt=None, extra_pnginfo=None):
        """
        Save video tensor to specified directory with custom naming and color preservation.
        
        Args:
            video_tensor: Video tensor (batch, frames, height, width, channels)
            filename_prefix: Base name for the video file
            custom_directory: Custom directory path (empty = use default output)
            video_format: Video format extension
            fps: Frames per second
            quality: FFmpeg CRF quality (0=lossless, 18=high, 23=medium, 51=lowest)
            preserve_colors: Enable color preservation mode (recommended: True)
            subfolder: Subfolder within the output directory
            prompt: Optional prompt information
            extra_pnginfo: Optional extra PNG info
            
        Returns:
            Tuple of (full_path, filename, info_string)
        """
        try:
            # Determine output directory
            if custom_directory and custom_directory.strip():
                # Use custom directory if provided
                if os.path.isabs(custom_directory):
                    # Absolute path
                    if not os.path.exists(custom_directory):
                        os.makedirs(custom_directory, exist_ok=True)
                    output_dir = custom_directory
                else:
                    # Relative path from ComfyUI root
                    comfy_root = os.path.dirname(folder_paths.get_output_directory())
                    output_dir = os.path.join(comfy_root, custom_directory)
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir, exist_ok=True)
            else:
                # Use default ComfyUI output directory
                output_dir = self.output_dir
            
            # Add subfolder if specified
            if subfolder and subfolder.strip():
                output_dir = os.path.join(output_dir, subfolder)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
            
            # Generate filename with counter for video files
            # Use custom implementation to properly handle video file incrementing
            video_filename, video_path = self._get_unique_video_filename(
                filename_prefix, output_dir, video_format
            )
            
            # Convert tensor to video with color preservation
            success = self._tensor_to_video_file(video_tensor, video_path, fps, quality, video_format, preserve_colors)
            
            if not success:
                return ("", "", f"Error: Failed to save video to {video_path}")
            
            # Create info string
            color_mode = "Color Preserved" if preserve_colors else "Standard"
            info = f"Video saved: {video_filename} | FPS: {fps} | Quality: {quality} | Format: {video_format.upper()} | {color_mode}"
            if custom_directory:
                info += f" | Custom dir: {custom_directory}"
            
            return (video_path, video_filename, info)
            
        except Exception as e:
            error_msg = f"Error saving video: {str(e)}"
            print(f"[CustomVideoSaver] {error_msg}")
            return ("", "", error_msg)
    
    def _tensor_to_video_file(self, video_tensor, output_path, fps, quality, video_format, preserve_colors=True):
        """
        Convert video tensor to video file using OpenCV with color data preservation.
        
        Args:
            video_tensor: Video tensor (batch, frames, height, width, channels)
            output_path: Output file path
            fps: Frames per second
            quality: Video quality (for codec-specific settings)
            video_format: Video format extension
            preserve_colors: Enable color preservation mode
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Handle tensor dimensions
            if len(video_tensor.shape) == 5:
                # (batch, frames, height, width, channels) - remove batch dimension
                video_array = video_tensor[0].cpu().numpy()
            elif len(video_tensor.shape) == 4:
                # (frames, height, width, channels)
                video_array = video_tensor.cpu().numpy()
            else:
                print(f"[CustomVideoSaver] Unexpected tensor shape: {video_tensor.shape}")
                return False
            
            # Preserve original data type and range for maximum color fidelity
            original_dtype = video_array.dtype
            original_min = video_array.min()
            original_max = video_array.max()
            
            print(f"[CustomVideoSaver] Original data: dtype={original_dtype}, range=[{original_min:.6f}, {original_max:.6f}]")
            
            # Color preservation mode: use high-precision conversion
            if preserve_colors:
                # Use high-precision conversion to preserve color accuracy
                if original_dtype == np.float32 or original_dtype == np.float64:
                    if original_max <= 1.0:
                        # Normalized float data [0,1] -> [0,255] with high precision
                        video_array = np.clip(video_array * 255.0, 0, 255)
                        # Use round() for more accurate conversion than direct casting
                        video_array = np.round(video_array).astype(np.uint8)
                    else:
                        # Float data in [0,255] range - preserve as much precision as possible
                        video_array = np.clip(video_array, 0, 255)
                        video_array = np.round(video_array).astype(np.uint8)
                elif original_dtype == np.uint8:
                    # Already in correct format - no conversion needed
                    video_array = video_array.astype(np.uint8)
                elif original_dtype in [np.uint16, np.int16]:
                    # 16-bit data - scale down to 8-bit while preserving color information
                    if original_max > 255:
                        video_array = np.clip(video_array / 256.0, 0, 255).astype(np.uint8)
                    else:
                        video_array = np.clip(video_array, 0, 255).astype(np.uint8)
                else:
                    # Other integer types - convert with clipping
                    video_array = np.clip(video_array, 0, 255).astype(np.uint8)
            else:
                # Standard conversion mode (faster but less precise)
                if original_dtype == np.float32 or original_dtype == np.float64:
                    if original_max <= 1.0:
                        video_array = np.clip(video_array * 255.0, 0, 255).astype(np.uint8)
                    else:
                        video_array = np.clip(video_array, 0, 255).astype(np.uint8)
                else:
                    video_array = np.clip(video_array, 0, 255).astype(np.uint8)
            
            # Get video dimensions
            num_frames, height, width, channels = video_array.shape
            
            print(f"[CustomVideoSaver] Processing {num_frames} frames of {width}x{height} with {channels} channels")
            
            # Set up video codec based on format with color preservation settings
            fourcc = self._get_fourcc(video_format)
            
            # Create video writer with color preservation settings
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                print(f"[CustomVideoSaver] Failed to open video writer for {output_path}")
                print(f"[CustomVideoSaver] Trying alternative codec...")
                
                # Try alternative codec as fallback
                if video_format.lower() == "mp4":
                    alternative_fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    out = cv2.VideoWriter(output_path, alternative_fourcc, fps, (width, height))
                    if out.isOpened():
                        print(f"[CustomVideoSaver] Using XVID codec as fallback")
                    else:
                        print(f"[CustomVideoSaver] All codecs failed, cannot create video")
                        return False
                else:
                    return False
            
            # Write frames with careful color space handling
            for i, frame in enumerate(video_array):
                # Ensure frame is contiguous in memory for better performance
                frame = np.ascontiguousarray(frame)
                
                # Validate color data before conversion
                if preserve_colors:
                    frame_min, frame_max = frame.min(), frame.max()
                    if frame_min < 0 or frame_max > 255:
                        print(f"[CustomVideoSaver] Warning: Frame {i} has values outside [0,255] range: [{frame_min}, {frame_max}]")
                        frame = np.clip(frame, 0, 255)
                
                # Handle different channel configurations with color preservation
                if channels == 3:
                    # RGB to BGR conversion for OpenCV (preserves all color data)
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                elif channels == 4:
                    # RGBA to BGR (drop alpha channel, preserve RGB)
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                elif channels == 1:
                    # Grayscale to BGR (preserve grayscale data)
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                else:
                    # Unknown channel count, try to preserve as-is
                    print(f"[CustomVideoSaver] Warning: Unknown channel count {channels}, using as-is")
                    frame_bgr = frame
                
                # Ensure the frame is in the correct format for the codec
                if frame_bgr.dtype != np.uint8:
                    frame_bgr = frame_bgr.astype(np.uint8)
                
                # Final validation for color preservation
                if preserve_colors:
                    bgr_min, bgr_max = frame_bgr.min(), frame_bgr.max()
                    if bgr_min < 0 or bgr_max > 255:
                        print(f"[CustomVideoSaver] Warning: BGR frame {i} has invalid values: [{bgr_min}, {bgr_max}]")
                        frame_bgr = np.clip(frame_bgr, 0, 255).astype(np.uint8)
                
                out.write(frame_bgr)
            
            # Release video writer
            out.release()
            
            # Verify file was created and has reasonable size
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                file_size = os.path.getsize(output_path)
                print(f"[CustomVideoSaver] Successfully saved video: {output_path} ({file_size} bytes)")
                return True
            else:
                print(f"[CustomVideoSaver] Video file was not created or is empty: {output_path}")
                return False
                
        except Exception as e:
            print(f"[CustomVideoSaver] Error converting tensor to video: {e}")
            return False
    
    def _get_fourcc(self, video_format):
        """Get OpenCV fourcc codec for video format with robust fallback system."""
        # Define codec preferences with fallbacks
        codec_preferences = {
            "mp4": [
                ('mp4v', 'MP4V'),  # Most compatible fallback
                ('XVID', 'XVID'),  # Alternative
                ('MJPG', 'MJPG'),  # Motion JPEG
            ],
            "avi": [
                ('XVID', 'XVID'),  # Most compatible for AVI
                ('MJPG', 'MJPG'),  # Motion JPEG
                ('mp4v', 'MP4V'),  # Alternative
            ],
            "mov": [
                ('mp4v', 'MP4V'),  # Most compatible for MOV
                ('XVID', 'XVID'),  # Alternative
                ('MJPG', 'MJPG'),  # Motion JPEG
            ],
            "mkv": [
                ('mp4v', 'MP4V'),  # Most compatible for MKV
                ('XVID', 'XVID'),  # Alternative
                ('MJPG', 'MJPG'),  # Motion JPEG
            ],
            "webm": [
                ('VP80', 'VP8'),   # VP8 for WebM
                ('mp4v', 'MP4V'),  # Fallback
            ],
            "wmv": [
                ('WMV2', 'WMV2'),  # WMV2 for WMV
                ('mp4v', 'MP4V'),  # Fallback
            ],
        }
        
        # Get codec preferences for the format
        preferences = codec_preferences.get(video_format.lower(), [('mp4v', 'MP4V')])
        
        # Try each codec in order of preference
        for codec_name, display_name in preferences:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec_name)
                # Test if codec works by creating a test writer
                test_path = f"test_{codec_name}.{video_format}"
                test_writer = cv2.VideoWriter(test_path, fourcc, 30, (640, 480))
                
                if test_writer.isOpened():
                    test_writer.release()
                    # Clean up test file if it was created
                    if os.path.exists(test_path):
                        os.remove(test_path)
                    print(f"[CustomVideoSaver] Using {display_name} codec for {video_format.upper()}")
                    return fourcc
                else:
                    print(f"[CustomVideoSaver] {display_name} codec not available, trying next...")
                    
            except Exception as e:
                print(f"[CustomVideoSaver] {display_name} codec test failed: {e}")
                continue
        
        # If all codecs fail, return the first one as last resort
        print(f"[CustomVideoSaver] All codecs failed, using fallback: {preferences[0][1]}")
        return cv2.VideoWriter_fourcc(*preferences[0][0])
    
    def _get_unique_video_filename(self, filename_prefix, output_dir, video_format):
        """
        Generate a unique video filename with proper incrementing.
        
        Args:
            filename_prefix: Base name for the video file
            output_dir: Directory where the video will be saved
            video_format: Video format extension
            
        Returns:
            Tuple of (filename, full_path)
        """
        # Ensure the output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Start with counter 1
        counter = 1
        
        while True:
            # Generate filename with 5-digit counter (e.g., video_00001.mp4)
            video_filename = f"{filename_prefix}_{counter:05}.{video_format}"
            video_path = os.path.join(output_dir, video_filename)
            
            # Check if file already exists
            if not os.path.exists(video_path):
                return video_filename, video_path
            
            # File exists, increment counter and try again
            counter += 1
            
            # Safety check to prevent infinite loop (max 99999 files)
            if counter > 99999:
                # Fallback: use timestamp
                import time
                timestamp = int(time.time())
                video_filename = f"{filename_prefix}_{timestamp}.{video_format}"
                video_path = os.path.join(output_dir, video_filename)
                return video_filename, video_path
    
    def _check_codec_support(self):
        """Check available codecs and provide helpful information."""
        print("[CustomVideoSaver] Checking codec support...")
        
        # Test common codecs
        test_codecs = [
            ('mp4v', 'MP4V'),
            ('XVID', 'XVID'),
            ('MJPG', 'Motion JPEG'),
            ('H264', 'H.264'),
        ]
        
        available_codecs = []
        for codec_name, display_name in test_codecs:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec_name)
                test_writer = cv2.VideoWriter("test_codec.mp4", fourcc, 30, (640, 480))
                if test_writer.isOpened():
                    available_codecs.append(display_name)
                    test_writer.release()
                    if os.path.exists("test_codec.mp4"):
                        os.remove("test_codec.mp4")
            except:
                pass
        
        if available_codecs:
            print(f"[CustomVideoSaver] Available codecs: {', '.join(available_codecs)}")
        else:
            print("[CustomVideoSaver] Warning: No codecs available, video saving may fail")
            print("[CustomVideoSaver] Consider installing OpenCV with additional codec support")
        
        # Check for common codec issues
        if 'H.264' not in available_codecs and 'MP4V' in available_codecs:
            print("[CustomVideoSaver] Note: H.264 not available, using MP4V (this is normal)")
        
        if not available_codecs:
            print("[CustomVideoSaver] Error: No video codecs available!")
            print("[CustomVideoSaver] Please ensure OpenCV is properly installed with video support")


# Import the video path loader
from .smart_video_loader import AutomateVideoPathLoader

# Simple Get Variable Node - Python backend
class GetVariableNode:
    """
    Python backend for Get Variable Node - passes through data from Set Variable Node.
    This is a minimal implementation that just passes data through.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dummy": ("*", {"default": None}),
            }
        }
    
    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("value",)
    FUNCTION = "get_variable"
    CATEGORY = "KNF_Utils/Variables"
    
    def get_variable(self, dummy=None):
        """Pass through the dummy input - the real data comes from the frontend connection."""
        return (dummy,)
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("inf")

# Register the nodes (NodeBypasser is frontend-only, no Python registration needed)
NODE_CLASS_MAPPINGS = {
    "KNF_Organizer": KNF_Organizer,
    "GeminiVideoCaptioner": GeminiVideoCaptioner,
    "AutomateVideoPathLoader": AutomateVideoPathLoader,
    "CustomVideoSaver": CustomVideoSaver,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KNF_Organizer": "KNF_Organizer",
    "GeminiVideoCaptioner": "Gemini Video Captioner",
    "AutomateVideoPathLoader": "Automate Video Path Loader",
    "CustomVideoSaver": "Custom Video Saver",
}