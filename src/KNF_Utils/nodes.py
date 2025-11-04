from inspect import cleandoc
import torch
import torch.nn.functional as F
import numpy as np
import copy
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
import sys
import platform
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server-side rendering
import matplotlib.pyplot as plt
from io import BytesIO

# Windows COM initialization for Media Foundation codecs
_COM_INITIALIZED = False
def _initialize_com_for_video():
    """Initialize COM in MTA mode on Windows for Media Foundation codecs"""
    global _COM_INITIALIZED
    if _COM_INITIALIZED or platform.system() != 'Windows':
        return True
    
    try:
        import pythoncom
        # Try to initialize COM in MTA mode (Multi-Threaded Apartment)
        # This is required for Media Foundation video encoders
        pythoncom.CoInitializeEx(pythoncom.COINIT_MULTITHREADED)
        _COM_INITIALIZED = True
        print("[CustomVideoSaver] COM initialized in MTA mode for Media Foundation")
        return True
    except Exception as e:
        # COM might already be initialized in STA mode by ComfyUI or another component
        print(f"[CustomVideoSaver] Warning: Could not initialize COM in MTA mode: {e}")
        print("[CustomVideoSaver] Media Foundation codecs (HEVC/H.265) may not work")
        print("[CustomVideoSaver] Will use fallback codecs (MP4V, MJPEG, XVID)")
        return False

# Import IO.ANY for proper wildcard type support
try:
    from comfy.comfy_types.node_typing import IO
except ImportError:
    # Fallback for older ComfyUI versions
    class IO:
        ANY = "*"

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
                "provider": (["Gemini", "OpenRouter"], {"default": "Gemini"}),
                "model": ([
                    # Gemini models
                    "gemini-1.5-flash", 
                    "gemini-1.5-pro", 
                    "gemini-2.5-pro",
                    # OpenRouter models (popular vision models)
                    "qwen/qwen2.5-vl-72b-instruct",
                    "qwen/qwen3-vl-235b-a22b-instruct",
                    "meta-llama/llama-3.2-90b-vision-instruct",
                ], {"default": "gemini-2.5-pro"}),
                "fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 120.0, "step": 0.1}),
            },
            "optional": {
                "video_tensor": ("IMAGE",),
                "image_tensor": ("IMAGE",),  # For single image captioning (OpenRouter compatible)
                "max_tokens": ("INT", {"default": 1000, "min": 100, "max": 4000, "step": 100}),  # For OpenRouter
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.1}),  # For OpenRouter
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
            # Video formats
            '.mp4': 'video/mp4',
            '.avi': 'video/x-msvideo',
            '.mov': 'video/quicktime',
            '.mkv': 'video/x-matroska',
            '.webm': 'video/webm',
            '.wmv': 'video/x-ms-wmv',
            '.flv': 'video/x-flv',
            '.m4v': 'video/mp4',
            # Image formats
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
            '.bmp': 'image/bmp'
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
    
    def tensor_to_image(self, image_tensor):
        """Convert image tensor to temporary image file."""
        try:
            # Handle batch dimension - take first image
            if len(image_tensor.shape) == 4:
                # (batch, height, width, channels)
                image_array = image_tensor[0].cpu().numpy()
            elif len(image_tensor.shape) == 3:
                # (height, width, channels)
                image_array = image_tensor.cpu().numpy()
            else:
                raise ValueError(f"Expected 3D or 4D tensor, got {image_tensor.shape}")
            
            # Ensure values are in [0, 255] range
            if image_array.max() <= 1.0:
                image_array = (image_array * 255).astype(np.uint8)
            else:
                image_array = image_array.astype(np.uint8)
            
            # Create temporary image file
            temp_dir = folder_paths.get_temp_directory()
            temp_image_path = os.path.join(temp_dir, f"temp_image_{int(time.time())}.jpg")
            
            # Convert RGB to BGR for OpenCV
            image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            cv2.imwrite(temp_image_path, image_bgr)
            
            return temp_image_path
            
        except Exception as e:
            print(f"Error converting tensor to image: {e}")
            return None
    
    def caption_video(self, video_file=None, prompt_text="Describe this video in detail.", api_key="", 
                     provider="Gemini", model="gemini-2.5-pro", fps=30.0, video_tensor=None, 
                     image_tensor=None, max_tokens=1000, temperature=0.7):
        """Main function to caption video/image using selected API provider."""
        if not api_key:
            return ("Error: API key is required",)
        
        file_path = None
        is_temp_file = False
        
        # Priority 1: Handle image tensor input (for OpenRouter compatibility)
        if image_tensor is not None:
            print("Processing image tensor input...")
            file_path = self.tensor_to_image(image_tensor)
            if file_path is None:
                return ("Error: Failed to convert image tensor to file",)
            is_temp_file = True
        # Priority 2: Handle video tensor input (from node link)
        elif video_tensor is not None:
            print(f"Processing video tensor input at {fps} FPS...")
            file_path = self.tensor_to_video(video_tensor, fps)
            if file_path is None:
                return ("Error: Failed to convert video tensor to file",)
            is_temp_file = True
        # Priority 3: Handle video file input
        elif video_file:
            # Construct full path for ComfyUI video file
            if not os.path.isabs(video_file):
                input_dir = folder_paths.get_input_directory()
                file_path = os.path.join(input_dir, video_file)
            else:
                file_path = video_file
            
            if not os.path.exists(file_path):
                return (f"Error: Video file not found: {file_path}",)
        else:
            return ("Error: Either video file, video tensor, or image tensor must be provided",)
        
        try:
            # Route to appropriate API provider
            if provider == "Gemini":
                result = self._call_gemini_api(file_path, prompt_text, api_key, model, is_temp_file)
            elif provider == "OpenRouter":
                result = self._call_openrouter_api(file_path, prompt_text, api_key, model, 
                                                   max_tokens, temperature, is_temp_file)
            else:
                result = (f"Error: Unknown provider: {provider}",)
            
            return result
                
        except Exception as e:
            # Clean up temporary file if it was created from tensor
            if is_temp_file and file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"Cleaned up temporary file after error: {file_path}")
                except Exception as cleanup_error:
                    print(f"Warning: Could not clean up temporary file {file_path}: {cleanup_error}")
            return (f"Exception: {str(e)}",)
    
    def _call_gemini_api(self, file_path, prompt_text, api_key, model, is_temp_file=False):
        """Call Gemini API for video/image captioning."""
        # Get file info for debug
        file_name = Path(file_path).name
        file_size_bytes = os.path.getsize(file_path)
        file_size_mb = file_size_bytes / (1024 * 1024)
        
        # Get metadata if it's a video
        metadata = self.get_video_metadata(file_path)
        mime_type = self.get_mime_type(file_path)
        
        # Print debug header
        print("\n" + "="*50)
        print(f"File: {file_name}")
        print(f"Size: {file_size_mb:.2f} MB")
        
        if metadata and 'width' in metadata:
            print(f"Resolution: {metadata['width']}x{metadata['height']}")
            print(f"FPS: {metadata['fps']:.2f}")
            print(f"Frames: {metadata['nb_frames']}")
            print(f"⏱Duration: {metadata['duration']:.2f}s")
        
        print(f"Model: {model}")
        print(f"MIME type: {mime_type}")
        print("-"*50)
        
        # Encode file to base64
        print("Encoding to base64...")
        encode_start = time.time()
        base64_data = self.encode_file_to_base64(file_path)
        encode_time = time.time() - encode_start
        base64_size_mb = len(base64_data) / (1024 * 1024)
        
        print(f"Base64 size: {base64_size_mb:.2f} MB")
        print(f"Encoding time: {encode_time:.2f}s")
        
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
        
        # Make API request
        print("Sending request to Gemini API...")
        request_start = time.time()
        response = requests.post(api_url, headers=headers, json=payload, timeout=60)
        request_time = time.time() - request_start
        
        print(f"API request time: {request_time:.2f}s")
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("API request successful!")
            
            # Show usage info if available
            if 'usageMetadata' in result:
                usage = result['usageMetadata']
                print(f"Tokens: {usage.get('promptTokenCount', 'N/A')} prompt, {usage.get('candidatesTokenCount', 'N/A')} response")
            
            print("="*50 + "\n")
            
            if 'candidates' in result and len(result['candidates']) > 0:
                candidate = result['candidates'][0]
                if 'content' in candidate and 'parts' in candidate['content']:
                    caption = candidate['content']['parts'][0]['text']
                    # Clean up temporary file if it was created from tensor
                    if is_temp_file and os.path.exists(file_path):
                        try:
                            os.remove(file_path)
                        except:
                            pass
                    return (caption.strip(),)
                else:
                    print("❌ Error: No content in API response")
                    print("="*50 + "\n")
                    return ("Error: No content in API response",)
            else:
                print("❌ Error: No candidates in API response")
                print("="*50 + "\n")
                return ("Error: No candidates in API response",)
        else:
            print(f"❌ API Error: {response.status_code}")
            print("="*50 + "\n")
            return (f"API error: {response.status_code} - {response.text}",)
    
    def _call_openrouter_api(self, file_path, prompt_text, api_key, model, 
                            max_tokens, temperature, is_temp_file=False):
        """Call OpenRouter API for image/video captioning."""
        try:
            # Get file info for debug
            file_name = Path(file_path).name
            file_size_bytes = os.path.getsize(file_path)
            file_size_mb = file_size_bytes / (1024 * 1024)
            
            # Get metadata if it's a video
            metadata = self.get_video_metadata(file_path)
            mime_type = self.get_mime_type(file_path)
            
            # Print debug header
            print("\n" + "="*50)
            print(f"File: {file_name}")
            print(f"Size: {file_size_mb:.2f} MB")
            
            if metadata and 'width' in metadata:
                print(f"Resolution: {metadata['width']}x{metadata['height']}")
                print(f"FPS: {metadata['fps']:.2f}")
                print(f"Frames: {metadata['nb_frames']}")
                print(f"⏱Duration: {metadata['duration']:.2f}s")
            
            print(f"Model: {model}")
            print(f"MIME type: {mime_type}")
            print("-"*50)
            
            # Encode file to base64
            print("Encoding to base64...")
            encode_start = time.time()
            base64_data = self.encode_file_to_base64(file_path)
            encode_time = time.time() - encode_start
            base64_size_mb = len(base64_data) / (1024 * 1024)
            
            print(f"Base64 size: {base64_size_mb:.2f} MB")
            print(f"Encoding time: {encode_time:.2f}s")
            
            # Prepare headers
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://github.com/comfyui",
                "X-Title": "ComfyUI Video Captioner"
            }
            
            # OpenRouter payload
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt_text
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{base64_data}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            # Make API request
            print("Sending request to OpenRouter API...")
            request_start = time.time()
            response = requests.post("https://openrouter.ai/api/v1/chat/completions", 
                                   headers=headers, json=payload, timeout=60)
            request_time = time.time() - request_start
            
            print(f"API request time: {request_time:.2f}s")
            print(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("API request successful!")
                
                # Show usage info if available
                if 'usage' in result:
                    usage = result['usage']
                    print(f"Tokens: {usage.get('prompt_tokens', 'N/A')} prompt, {usage.get('completion_tokens', 'N/A')} completion")
                
                print("="*50 + "\n")
                
                if 'choices' in result and len(result['choices']) > 0:
                    caption = result['choices'][0]['message']['content']
                    
                    if not caption or caption.strip() == "":
                        print("⚠️ Warning: Model returned empty response")
                        return ("Error: Model returned empty response.",)
                    
                    # Clean up temp file
                    if is_temp_file and os.path.exists(file_path):
                        try:
                            os.remove(file_path)
                        except:
                            pass
                    
                    return (caption.strip(),)
                else:
                    print("❌ Error: No caption generated by API")
                    print("="*50 + "\n")
                    return ("Error: No caption generated by API",)
            else:
                print(f"❌ API Error: {response.status_code}")
                print("="*50 + "\n")
                error_msg = response.text[:200] if len(response.text) > 200 else response.text
                return (f"API error {response.status_code}: {error_msg}",)
                
        except Exception as e:
            print(f"❌ Exception: {str(e)}")
            print("="*50 + "\n")
            return (f"Exception: {str(e)}",)


class CustomVideoSaver:
    """
    Custom video saver that allows saving videos to a user-specified directory.
    
    ENCODING METHODS:
    -----------------
    - FFmpeg (RECOMMENDED): Professional encoding, supports all codecs including H.265/HEVC
    - OpenCV: Simpler but limited codec support on Windows
    
    CODEC RECOMMENDATIONS:
    ----------------------
    - H.265/HEVC: Best compression, smallest files (FFmpeg only)
    - H.264/AVC: Good compression, widely compatible (FFmpeg recommended)
    - MP4V: Universal compatibility, works everywhere
    - MJPEG: Highest quality, largest files
    - VP9: Modern codec, good for web (FFmpeg only)
    - ProRes: Professional editing (FFmpeg only)
    
    FFmpeg will be used automatically for H.265/HEVC and H.264/AVC to avoid
    Windows COM threading issues. OpenCV is used for other codecs.
    """
    
    def __init__(self):
        self.type = "output"
        self.output_dir = folder_paths.get_output_directory()
        self.ffmpeg_available = self._check_ffmpeg_available()
        self._check_codec_support()
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_tensor": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "ComfyUI_video"}),
                "custom_directory": ("STRING", {"default": ""}),  # Empty = use default output
                "video_format": (["mp4", "mkv", "webm", "mov", "avi"], {"default": "mp4"}),
                "codec": (["H.265/HEVC", "H.264/AVC", "VP9", "ProRes", "MP4V", "MJPEG", "XVID"], {"default": "H.265/HEVC"}),
                "fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 120.0, "step": 0.1}),
                "quality": ("INT", {"default": 23, "min": 0, "max": 51, "step": 1}),  # CRF: 0=lossless, 18=visually lossless, 23=good, 28=default
                "encoding_preset": (["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"], {"default": "medium"}),
                "preserve_colors": ("BOOLEAN", {"default": True}),  # Enable color preservation mode
                "use_ffmpeg": ("BOOLEAN", {"default": True}),  # Use FFmpeg when available
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
                   video_format="mp4", codec="H.265/HEVC", fps=30.0, quality=23, encoding_preset="medium",
                   preserve_colors=True, use_ffmpeg=True, subfolder="", prompt=None, extra_pnginfo=None):
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
            
            # Determine encoding method with automatic fallback
            use_ffmpeg_encoding = False
            original_codec = codec
            
            if use_ffmpeg and self.ffmpeg_available:
                # Use FFmpeg for these codecs (better quality, no COM issues)
                if codec in ["H.265/HEVC", "H.264/AVC", "VP9", "ProRes"]:
                    use_ffmpeg_encoding = True
                    print(f"[CustomVideoSaver] Using FFmpeg for {codec} encoding")
            elif codec in ["H.265/HEVC", "H.264/AVC", "VP9", "ProRes"] and not self.ffmpeg_available:
                # FFmpeg is required but not available
                print(f"[CustomVideoSaver] Warning: {codec} requires FFmpeg but it's not available")
                print(f"[CustomVideoSaver] Falling back to MP4V with OpenCV")
                codec = "MP4V"
            
            # Convert tensor to video with automatic fallback
            success = False
            encoding_method = ""
            
            if use_ffmpeg_encoding:
                # Try FFmpeg first
                print(f"[CustomVideoSaver] Attempting FFmpeg encoding with {codec}...")
                success = self._tensor_to_video_ffmpeg(video_tensor, video_path, fps, quality, codec, 
                                                       encoding_preset, preserve_colors)
                encoding_method = "FFmpeg"
                
                # If H.265 fails, automatically try H.264
                if not success and codec == "H.265/HEVC":
                    print(f"[CustomVideoSaver] H.265/HEVC failed, trying H.264/AVC as fallback...")
                    codec = "H.264/AVC"
                    success = self._tensor_to_video_ffmpeg(video_tensor, video_path, fps, quality, codec, 
                                                           encoding_preset, preserve_colors)
                    if success:
                        encoding_method = "FFmpeg (H.264 fallback)"
                
                # If FFmpeg still fails, try OpenCV as last resort
                if not success:
                    print(f"[CustomVideoSaver] FFmpeg encoding failed, trying OpenCV with MP4V as fallback...")
                    codec = "MP4V"
                    success = self._tensor_to_video_opencv(video_tensor, video_path, fps, quality, codec, preserve_colors)
                    if success:
                        encoding_method = "OpenCV (MP4V fallback)"
            else:
                # Use OpenCV directly
                success = self._tensor_to_video_opencv(video_tensor, video_path, fps, quality, codec, preserve_colors)
                encoding_method = "OpenCV"
            
            if not success:
                return ("", "", f"Error: All encoding methods failed for video {video_path}. Check logs for details.")
            
            # Create info string
            color_mode = "Color Preserved" if preserve_colors else "Standard"
            quality_desc = "Lossless" if quality == 0 else f"CRF {quality}"
            info = f"Video saved: {video_filename} | Codec: {codec} | {quality_desc} | Preset: {encoding_preset} | FPS: {fps} | Format: {video_format.upper()} | {color_mode} | Encoder: {encoding_method}"
            if custom_directory:
                info += f" | Custom dir: {custom_directory}"
            
            return (video_path, video_filename, info)
            
        except Exception as e:
            error_msg = f"Error saving video: {str(e)}"
            print(f"[CustomVideoSaver] {error_msg}")
            return ("", "", error_msg)
    
    def _tensor_to_video_ffmpeg(self, video_tensor, output_path, fps, quality, codec, preset, preserve_colors=True):
        """
        Convert video tensor to video file using FFmpeg subprocess.
        This bypasses Windows COM issues and provides professional-grade encoding.
        
        Args:
            video_tensor: Video tensor (batch, frames, height, width, channels)
            output_path: Output file path
            fps: Frames per second
            quality: Video quality (CRF value, 0=lossless, 23=good)
            codec: Video codec to use (H.265/HEVC, H.264/AVC, VP9, ProRes)
            preset: Encoding preset (ultrafast to veryslow)
            preserve_colors: Enable color preservation mode
            
        Returns:
            bool: True if successful, False otherwise
        """
        import threading
        import queue
        
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
            
            # Convert to uint8 with color preservation
            if preserve_colors:
                if original_dtype == np.float32 or original_dtype == np.float64:
                    if original_max <= 1.0:
                        # Normalized float data [0,1] -> [0,255]
                        video_array = np.clip(video_array * 255.0 + 0.5, 0, 255).astype(np.uint8)
                    else:
                        video_array = np.clip(video_array + 0.5, 0, 255).astype(np.uint8)
                else:
                    video_array = np.clip(video_array, 0, 255).astype(np.uint8)
            else:
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
            print(f"[CustomVideoSaver] Using FFmpeg with codec: {codec} (CRF: {quality}, Preset: {preset})")
            
            # Map codec to FFmpeg codec name
            codec_map = {
                "H.265/HEVC": "libx265",
                "H.264/AVC": "libx264",
                "VP9": "libvpx-vp9",
                "ProRes": "prores_ks"
            }
            
            ffmpeg_codec = codec_map.get(codec, "libx264")
            
            # Build FFmpeg command with extra robustness options
            cmd = [
                'ffmpeg',
                '-y',  # Overwrite output file
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-s', f'{width}x{height}',
                '-pix_fmt', 'rgb24',
                '-r', str(fps),
                '-i', '-',  # Read from stdin
                '-c:v', ffmpeg_codec,
                '-movflags', '+faststart',  # Enable fast start for MP4
                '-max_muxing_queue_size', '9999',  # Prevent muxing queue overflow
            ]
            
            # Add codec-specific parameters
            if codec in ["H.265/HEVC", "H.264/AVC"]:
                cmd.extend([
                    '-crf', str(quality),
                    '-preset', preset,
                    '-pix_fmt', 'yuv420p',  # Maximum compatibility
                ])
                # Add extra params for stability with large videos
                if codec == "H.265/HEVC":
                    cmd.extend([
                        '-x265-params', 'log-level=error',  # Reduce x265 verbosity
                    ])
            elif codec == "VP9":
                cmd.extend([
                    '-crf', str(quality),
                    '-b:v', '0',  # Use CRF mode
                    '-pix_fmt', 'yuv420p',
                    '-row-mt', '1',  # Enable row-based multithreading
                ])
            elif codec == "ProRes":
                # ProRes doesn't use CRF, use profile instead
                cmd.extend([
                    '-profile:v', '3',  # ProRes HQ
                    '-pix_fmt', 'yuv422p10le',
                ])
            
            # Add output file
            cmd.append(output_path)
            
            print(f"[CustomVideoSaver] FFmpeg command: {' '.join(cmd[:12])}...")
            
            # Create a queue for stderr output
            stderr_queue = queue.Queue()
            stderr_lines = []
            
            def read_stderr(pipe, q):
                """Read stderr in a separate thread to prevent deadlock"""
                try:
                    for line in iter(pipe.readline, b''):
                        line_str = line.decode('utf-8', errors='ignore').strip()
                        q.put(line_str)
                        stderr_lines.append(line_str)
                    pipe.close()
                except Exception as e:
                    q.put(f"Error reading stderr: {e}")
            
            # Start FFmpeg process with stderr capture
            # CRITICAL: We now capture stderr in a separate thread to avoid deadlock
            # while still being able to see error messages
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,  # Capture stderr for debugging
                bufsize=10**8,
                creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == 'Windows' else 0
            )
            
            # Start stderr reading thread
            stderr_thread = threading.Thread(target=read_stderr, args=(process.stderr, stderr_queue))
            stderr_thread.daemon = True
            stderr_thread.start()
            
            # Write frames to FFmpeg
            frame_count = len(video_array)
            print(f"[CustomVideoSaver] Encoding {frame_count} frames...")
            
            write_failed = False
            last_error = None
            
            for i, frame in enumerate(video_array):
                # Progress update every 10 frames
                if i > 0 and i % 10 == 0:
                    progress = (i / frame_count) * 100
                    print(f"[CustomVideoSaver] Progress: {progress:.1f}% ({i}/{frame_count})")
                
                # Check if process is still alive
                if process.poll() is not None:
                    print(f"[CustomVideoSaver] FFmpeg process died at frame {i}")
                    write_failed = True
                    break
                
                # Ensure frame is contiguous in memory
                frame = np.ascontiguousarray(frame)
                
                # Validate frame data
                if preserve_colors:
                    frame_min, frame_max = frame.min(), frame.max()
                    if frame_min < 0 or frame_max > 255:
                        print(f"[CustomVideoSaver] Warning: Frame {i} has invalid range [{frame_min}, {frame_max}]")
                        frame = np.clip(frame, 0, 255).astype(np.uint8)
                
                # Write raw RGB data
                try:
                    process.stdin.write(frame.tobytes())
                    # Flush periodically to avoid buffer buildup
                    if i % 30 == 0:
                        process.stdin.flush()
                except BrokenPipeError:
                    print(f"[CustomVideoSaver] FFmpeg pipe broken at frame {i}")
                    write_failed = True
                    last_error = "BrokenPipeError"
                    break
                except OSError as e:
                    print(f"[CustomVideoSaver] OS error writing frame {i}: {e}")
                    write_failed = True
                    last_error = str(e)
                    break
                except Exception as e:
                    print(f"[CustomVideoSaver] Error writing frame {i}: {e}")
                    write_failed = True
                    last_error = str(e)
                    break
            
            print(f"[CustomVideoSaver] All frames written, finalizing...")
            
            # Close stdin and wait for FFmpeg to finish
            try:
                process.stdin.flush()
                process.stdin.close()
            except Exception as e:
                print(f"[CustomVideoSaver] Error closing stdin: {e}")
            
            # Wait with timeout to prevent infinite hang
            # For 4K video, allow more time (10 minutes)
            timeout = 600 if width >= 3840 or height >= 2160 else 300
            try:
                return_code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                print(f"[CustomVideoSaver] FFmpeg timed out after {timeout} seconds")
                process.kill()
                process.wait()  # Clean up
                return False
            
            # Wait for stderr thread to finish
            stderr_thread.join(timeout=5)
            
            # Check if encoding was successful
            if return_code != 0 or write_failed:
                print(f"[CustomVideoSaver] ==================== FFmpeg ERROR ====================")
                print(f"[CustomVideoSaver] FFmpeg failed with return code {return_code}")
                if last_error:
                    print(f"[CustomVideoSaver] Last error: {last_error}")
                
                # Print last 20 lines of stderr for debugging
                print(f"[CustomVideoSaver] Last FFmpeg messages:")
                for line in stderr_lines[-20:]:
                    if line.strip():  # Skip empty lines
                        print(f"[CustomVideoSaver]   {line}")
                print(f"[CustomVideoSaver] =====================================================")
                
                # Try to provide helpful suggestions
                if "Invalid argument" in ' '.join(stderr_lines):
                    print(f"[CustomVideoSaver] Suggestion: Try using H.264/AVC instead of {codec}")
                elif "not supported" in ' '.join(stderr_lines).lower():
                    print(f"[CustomVideoSaver] Suggestion: Codec {codec} may not be available")
                elif any("memory" in line.lower() for line in stderr_lines):
                    print(f"[CustomVideoSaver] Suggestion: Try reducing video resolution or using faster preset")
                
                return False
            
            # Verify file was created
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
                print(f"[CustomVideoSaver] Successfully saved video: {output_path} ({file_size:.2f} MB)")
                return True
            else:
                print(f"[CustomVideoSaver] Video file was not created or is empty: {output_path}")
                return False
                
        except Exception as e:
            print(f"[CustomVideoSaver] Error in FFmpeg encoding: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _tensor_to_video_opencv(self, video_tensor, output_path, fps, quality, codec, preserve_colors=True):
        """
        Convert video tensor to video file using OpenCV with color data preservation.
        
        Args:
            video_tensor: Video tensor (batch, frames, height, width, channels)
            output_path: Output file path
            fps: Frames per second
            quality: Video quality (CRF value, 0=lossless)
            codec: Video codec to use (H.265/HEVC, H.264/AVC, MP4V, XVID, MJPEG)
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
                        # Multiply by 255 and add 0.5 before floor (equivalent to round but faster)
                        video_array = np.clip(video_array * 255.0 + 0.5, 0, 255).astype(np.uint8)
                    else:
                        # Float data in [0,255] range - preserve as much precision as possible
                        video_array = np.clip(video_array + 0.5, 0, 255).astype(np.uint8)
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
            print(f"[CustomVideoSaver] Using codec: {codec} (CRF: {quality})")
            
            # Initialize COM for Windows Media Foundation codecs
            com_available = False
            if codec in ["H.265/HEVC", "H.264/AVC"] and platform.system() == 'Windows':
                com_available = _initialize_com_for_video()
                if not com_available:
                    print(f"[CustomVideoSaver] COM initialization failed, cannot use {codec}")
                    print(f"[CustomVideoSaver] Falling back to MP4V codec...")
                    codec = "MP4V"
            
            # Set up video codec
            fourcc, codec_name = self._get_fourcc_for_codec(codec)
            
            # Create video writer
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                print(f"[CustomVideoSaver] Failed to open video writer with {codec_name}")
                print(f"[CustomVideoSaver] Trying fallback codecs...")
                
                # Try fallback codecs in order of preference
                fallback_codecs = [
                    ('mp4v', 'MP4V'),
                    ('MJPG', 'MJPEG'),
                    ('XVID', 'XVID'),
                ]
                
                for fallback_fourcc_str, fallback_name in fallback_codecs:
                    try:
                        fallback_fourcc = cv2.VideoWriter_fourcc(*fallback_fourcc_str)
                        out = cv2.VideoWriter(output_path, fallback_fourcc, fps, (width, height))
                        if out.isOpened():
                            print(f"[CustomVideoSaver] Successfully using {fallback_name} codec")
                            break
                    except Exception as e:
                        print(f"[CustomVideoSaver] {fallback_name} codec failed: {e}")
                        continue
                
                if not out.isOpened():
                    print(f"[CustomVideoSaver] All codecs failed, cannot create video")
                    print(f"[CustomVideoSaver] This might be due to:")
                    print(f"  1. COM threading mode conflict on Windows (try restarting ComfyUI)")
                    print(f"  2. Missing video codec support in OpenCV")
                    print(f"  3. Invalid output path or permissions")
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
    
    def _get_fourcc_for_codec(self, codec):
        """
        Get OpenCV fourcc code for the selected codec.
        Returns tuple of (fourcc_code, codec_name_used).
        """
        codec_map = {
            "H.265/HEVC": ['hev1', 'HEVC', 'H265', 'x265'],  # Try multiple H.265 codes
            "H.264/AVC": ['H264', 'avc1', 'X264', 'h264'],   # Try multiple H.264 codes
            "MP4V": ['mp4v', 'MP4V'],
            "XVID": ['XVID', 'xvid'],
            "MJPEG": ['MJPG', 'mjpg']
        }
        
        # Get possible fourcc codes for this codec
        fourcc_codes = codec_map.get(codec, ['mp4v'])
        
        # Try each fourcc code
        for fourcc_str in fourcc_codes:
            try:
                fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
                print(f"[CustomVideoSaver] Trying {codec} with fourcc: {fourcc_str}")
                return (fourcc, fourcc_str)
            except Exception as e:
                print(f"[CustomVideoSaver] fourcc '{fourcc_str}' failed: {e}")
                continue
        
        # Fallback to MP4V
        print(f"[CustomVideoSaver] Using MP4V fallback")
        return (cv2.VideoWriter_fourcc(*'mp4v'), 'mp4v')
    
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
    
    def _check_ffmpeg_available(self):
        """Check if FFmpeg is available in the system."""
        try:
            result = subprocess.run(
                ['ffmpeg', '-version'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5
            )
            if result.returncode == 0:
                # Parse FFmpeg version
                output = result.stdout.decode('utf-8', errors='ignore')
                first_line = output.split('\n')[0]
                print(f"[CustomVideoSaver] {first_line}")
                print(f"[CustomVideoSaver] FFmpeg is available - using for H.265/HEVC encoding")
                return True
            else:
                print("[CustomVideoSaver] FFmpeg found but returned error")
                return False
        except FileNotFoundError:
            print("[CustomVideoSaver] FFmpeg not found in PATH")
            print("[CustomVideoSaver] H.265/HEVC and advanced codecs will not be available")
            print("[CustomVideoSaver] Install FFmpeg for better codec support")
            return False
        except Exception as e:
            print(f"[CustomVideoSaver] Error checking FFmpeg: {e}")
            return False
    
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

# Import the video loader with frame replacement
try:
    from .video_loader_with_frame_replacement import NV_Video_Loader_Path
    NV_VIDEO_LOADER_AVAILABLE = True
except Exception as e:
    print(f"[NV_Comfy_Utils] Warning: Could not import NV_Video_Loader_Path: {e}")
    import traceback
    traceback.print_exc()
    NV_VIDEO_LOADER_AVAILABLE = False
    # Create a dummy class to prevent registration errors
    class NV_Video_Loader_Path:
        pass

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
                "dummy": (IO.ANY, {"default": None}),
            }
        }
    
    RETURN_TYPES = (IO.ANY,)
    RETURN_NAMES = ("value",)
    FUNCTION = "get_variable"
    CATEGORY = "KNF_Utils/Variables"
    
    @classmethod
    def VALIDATE_INPUTS(cls, dummy=None):
        """
        Validate inputs - allow any type to pass through.
        The frontend handles the actual type matching.
        """
        return True
    
    def get_variable(self, dummy=None):
        """Pass through the dummy input - the real data comes from the frontend connection."""
        return (dummy,)
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("inf")


class LazySwitch:
    """
    Lazy Switch Node - Only evaluates the input corresponding to the boolean value.
    Unlike standard switches, this allows the workflow to run even if only the 
    relevant input is connected.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "boolean": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "on_false": (IO.ANY, {"lazy": True}),
                "on_true": (IO.ANY, {"lazy": True}),
            },
        }
    
    RETURN_TYPES = (IO.ANY,)
    RETURN_NAMES = ("output",)
    FUNCTION = "switch"
    CATEGORY = "KNF_Utils/Logic"
    DESCRIPTION = "Lazily evaluates only the input branch corresponding to the boolean value. If true, only on_true is evaluated. If false, only on_false is evaluated."
    
    @classmethod
    def VALIDATE_INPUTS(cls, boolean, on_false=None, on_true=None):
        """
        Validate inputs - allow wildcard types to pass through.
        This is more permissive than the default validation.
        """
        # Always return True to bypass strict type checking
        # The lazy evaluation will handle which input is actually needed
        return True
    
    def check_lazy_status(self, boolean, on_false=None, on_true=None):
        """
        This method tells ComfyUI which inputs need to be evaluated.
        Only the input corresponding to the boolean value will be requested.
        """
        if boolean and on_true is None:
            # Boolean is true but on_true hasn't been evaluated yet
            return ["on_true"]
        if not boolean and on_false is None:
            # Boolean is false but on_false hasn't been evaluated yet
            return ["on_false"]
        # If we get here, the required input has been evaluated
        return []
    
    def switch(self, boolean, on_false=None, on_true=None):
        """
        Return the value corresponding to the boolean.
        Only one of on_false or on_true will be evaluated.
        """
        if boolean:
            return (on_true,)
        else:
            return (on_false,)


class VideoExtensionDiagnostic:
    """
    Diagnose color drift and visual degradation in video extension workflows.
    Compares original video with up to 3 generations of extended videos.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_video": ("IMAGE",),
                "frame_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "step": 1,
                    "display": "number"
                }),
            },
            "optional": {
                "gen1_video": ("IMAGE",),
                "gen2_video": ("IMAGE",),
                "gen3_video": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("diagnostic_image", "statistics")
    FUNCTION = "diagnose"
    CATEGORY = "NV_Comfy_Utils/Diagnostic"
    
    def diagnose(self, original_video, frame_index, gen1_video=None, gen2_video=None, gen3_video=None):
        """
        Create diagnostic visualization showing color drift over generations.
        
        Args:
            original_video: Original video tensor [frames, height, width, channels]
            frame_index: Which frame to visualize
            gen1_video: First generation extended video (optional)
            gen2_video: Second generation extended video (optional)
            gen3_video: Third generation extended video (optional)
        
        Returns:
            diagnostic_image: Matplotlib visualization as IMAGE tensor
            statistics: Text statistics string
        """
        # Collect available videos
        videos = [original_video]
        titles = ['Original']
        
        if gen1_video is not None:
            videos.append(gen1_video)
            titles.append('Gen 1')
        if gen2_video is not None:
            videos.append(gen2_video)
            titles.append('Gen 2')
        if gen3_video is not None:
            videos.append(gen3_video)
            titles.append('Gen 3')
        
        num_videos = len(videos)
        
        # Validate frame index
        for i, video in enumerate(videos):
            if frame_index >= video.shape[0]:
                print(f"[VideoExtensionDiagnostic] Warning: frame_index {frame_index} >= {video.shape[0]} frames in {titles[i]}, using last frame")
                frame_index = min(frame_index, video.shape[0] - 1)
        
        # Create figure with 2 rows: sample frames and histograms
        fig, axes = plt.subplots(2, num_videos, figsize=(4 * num_videos, 8))
        
        # Handle single video case (axes won't be 2D array)
        if num_videos == 1:
            axes = axes.reshape(2, 1)
        
        # Convert tensors to numpy for visualization
        video_arrays = []
        for video in videos:
            # ComfyUI format: [frames, height, width, channels], values in [0, 1]
            video_np = video.cpu().numpy()
            video_arrays.append(video_np)
        
        # Row 1: Sample frames
        for i, (video_np, title) in enumerate(zip(video_arrays, titles)):
            frame = video_np[frame_index]
            axes[0, i].imshow(frame)
            axes[0, i].set_title(f'{title}\nFrame {frame_index}', fontsize=10, fontweight='bold')
            axes[0, i].axis('off')
        
        # Row 2: Histograms showing color distribution
        for i, (video_np, title) in enumerate(zip(video_arrays, titles)):
            # Flatten all frames to analyze overall distribution
            flattened = video_np.flatten()
            
            axes[1, i].hist(flattened, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
            axes[1, i].set_xlim([0, 1])
            axes[1, i].set_xlabel('Pixel Value', fontsize=9)
            axes[1, i].set_ylabel('Frequency', fontsize=9)
            
            mean_val = flattened.mean()
            std_val = flattened.std()
            axes[1, i].set_title(f'μ={mean_val:.3f}, σ={std_val:.3f}', fontsize=9)
            axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert matplotlib figure to IMAGE tensor
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        
        # Load image from buffer
        img_pil = Image.open(buf)
        img_np = np.array(img_pil).astype(np.float32) / 255.0
        
        # Convert to ComfyUI format: [batch, height, width, channels]
        if img_np.ndim == 3:
            img_tensor = torch.from_numpy(img_np)[None, :]
        else:
            img_tensor = torch.from_numpy(img_np[:, :, :3])[None, :]
        
        # Generate statistics text
        stats_lines = []
        stats_lines.append("=" * 60)
        stats_lines.append("VIDEO EXTENSION DIAGNOSTIC REPORT")
        stats_lines.append("=" * 60)
        stats_lines.append(f"Analysis Frame: {frame_index}")
        stats_lines.append("")
        
        for i, (video_np, title) in enumerate(zip(video_arrays, titles)):
            stats_lines.append(f"{title}:")
            stats_lines.append(f"  Frames:        {video_np.shape[0]}")
            stats_lines.append(f"  Resolution:    {video_np.shape[1]}x{video_np.shape[2]}")
            stats_lines.append(f"  Mean:          {video_np.mean():.6f}")
            stats_lines.append(f"  Std:           {video_np.std():.6f}")
            stats_lines.append(f"  Min:           {video_np.min():.6f}")
            stats_lines.append(f"  Max:           {video_np.max():.6f}")
            
            # Calculate clipping percentages
            clipped_to_0 = (video_np == 0).sum() / video_np.size * 100
            clipped_to_1 = (video_np == 1).sum() / video_np.size * 100
            stats_lines.append(f"  % clipped to 0: {clipped_to_0:.2f}%")
            stats_lines.append(f"  % clipped to 1: {clipped_to_1:.2f}%")
            
            # Calculate drift from original (if not the original)
            if i > 0:
                diff = video_np.mean() - video_arrays[0].mean()
                diff_percent = (diff / video_arrays[0].mean()) * 100
                stats_lines.append(f"  Mean drift from original: {diff:+.6f} ({diff_percent:+.2f}%)")
            
            stats_lines.append("")
        
        # Add color drift summary
        if num_videos > 1:
            stats_lines.append("=" * 60)
            stats_lines.append("COLOR DRIFT ANALYSIS")
            stats_lines.append("=" * 60)
            
            orig_mean = video_arrays[0].mean()
            for i in range(1, num_videos):
                gen_mean = video_arrays[i].mean()
                drift = gen_mean - orig_mean
                drift_percent = (drift / orig_mean) * 100
                
                if abs(drift_percent) > 5:
                    severity = "⚠️ SIGNIFICANT"
                elif abs(drift_percent) > 2:
                    severity = "⚡ MODERATE"
                else:
                    severity = "✓ MINIMAL"
                
                stats_lines.append(f"{titles[i]}: {drift:+.6f} ({drift_percent:+.2f}%) - {severity}")
            
            stats_lines.append("")
        
        statistics_text = "\n".join(stats_lines)
        print(f"\n{statistics_text}")
        
        return (img_tensor, statistics_text)


class VAE_LUT_Generator:
    """
    Generate a VAE correction LUT by analyzing videos with different VAE cycle counts.
    This node helps you build a calibration profile for your specific VAE.
    
    Now supports AUTOMATIC degradation testing - just provide a VAE and video!
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_video": ("IMAGE",),
                "vae_name": ("STRING", {"default": "my_vae", "multiline": False}),
            },
            "optional": {
                # NEW: Automated testing mode
                "vae": ("VAE",),
                
                # Simple mode: Test 1, 2, 3, ... up to max_cycles
                "max_cycles": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "display": "number"
                }),
                
                # Advanced mode: Specify exact cycles (e.g., "1,3,6,10")
                "test_cycles": ("STRING", {"default": "", "multiline": False}),
                
                # Codec degradation options (save/load between cycles)
                "include_codec_degradation": ("BOOLEAN", {"default": False}),
                "codec_quality": ("INT", {"default": 23, "min": 0, "max": 51, "step": 1}),
                "fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 120.0, "step": 0.1}),
                
                # Legacy: Manual pre-generated videos (still supported)
                "cycle_1_video": ("IMAGE",),
                "cycle_2_video": ("IMAGE",),
                "cycle_3_video": ("IMAGE",),
                "cycle_4_video": ("IMAGE",),
                "cycle_5_video": ("IMAGE",),
                "cycle_6_video": ("IMAGE",),
                "cycle_8_video": ("IMAGE",),
                "cycle_10_video": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("lut_json", "lut_report")
    FUNCTION = "generate_lut"
    CATEGORY = "NV_Comfy_Utils/Diagnostic"
    
    def _apply_vae_cycles(self, video_tensor, vae, num_cycles, include_codec_degradation=False, 
                         codec_quality=23, fps=30.0):
        """
        Apply N VAE encode/decode cycles to a video tensor.
        Uses the EXACT same logic as ComfyUI's native VAEEncode/VAEDecode nodes.
        
        Args:
            video_tensor: Input video [frames, height, width, channels]
            vae: VAE model
            num_cycles: Number of encode/decode cycles
            include_codec_degradation: If True, save/load video between cycles
            codec_quality: FFmpeg CRF quality (0=lossless, 23=default, 51=worst)
            fps: Frames per second for video encoding
        
        Returns:
            Degraded video tensor
        """
        current_pixels = video_tensor
        original_frame_count = current_pixels.shape[0]
        
        print(f"[VAE_LUT_Generator] Applying {num_cycles} VAE cycles...")
        print(f"[VAE_LUT_Generator] Input shape: {current_pixels.shape}, dtype: {current_pixels.dtype}")
        
        # Check if this is a temporal/video VAE (compresses frames)
        vae_type = type(vae.first_stage_model).__name__ if hasattr(vae, 'first_stage_model') else "Unknown"
        is_temporal_vae = 'wan' in vae_type.lower() or 'video' in vae_type.lower() or 'temporal' in vae_type.lower()
        
        if is_temporal_vae:
            print(f"[VAE_LUT_Generator] ⚠️  Detected temporal/video VAE ({vae_type})")
            print(f"[VAE_LUT_Generator] This VAE compresses the temporal dimension (frame count)")
            print(f"[VAE_LUT_Generator] Frame count may decrease with each cycle")
            print(f"[VAE_LUT_Generator] ")
            print(f"[VAE_LUT_Generator] 💡 RECOMMENDATION for temporal VAEs:")
            print(f"[VAE_LUT_Generator]    • Use videos with {original_frame_count * 2}+ frames")
            print(f"[VAE_LUT_Generator]    • Start with max_cycles: 2-3 to test")
            print(f"[VAE_LUT_Generator]    • Monitor frame count reduction")
            print(f"[VAE_LUT_Generator] ")
        
        for cycle in range(num_cycles):
            try:
                # DEBUG: Show mean BEFORE encode
                mean_before = float(current_pixels.mean())
                print(f"[VAE_LUT_Generator]   🔍 Cycle {cycle + 1} - Mean BEFORE encode: {mean_before:.6f}")
                
                # === EXACT NATIVE COMFYUI VAE ENCODE LOGIC ===
                # From: ComfyUI/nodes.py line 342 (VAEEncode)
                # t = vae.encode(pixels[:,:,:,:3])
                # 
                # NOTE: We do NOT call vae_encode_crop_pixels() because:
                # 1. Native ComfyUI VAEEncode doesn't use it
                # 2. Cropping reduces degradation by removing edge artifacts
                # 3. We want to measure the FULL degradation that users experience
                
                # Encode: ensure RGB only (first 3 channels), no cropping
                latent = vae.encode(current_pixels[:,:,:,:3])
                
                # === EXACT NATIVE COMFYUI VAE DECODE LOGIC ===
                # From: ComfyUI/nodes.py line 294 (VAEDecode)
                # images = vae.decode(samples["samples"])
                # if len(images.shape) == 5: #Combine batches
                #     images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
                
                # Decode: vae.decode returns pixels directly
                current_pixels = vae.decode(latent)
                
                # Reshape 5D to 4D if needed (for temporal VAEs)
                if len(current_pixels.shape) == 5:
                    current_pixels = current_pixels.reshape(-1, current_pixels.shape[-3], current_pixels.shape[-2], current_pixels.shape[-1])
                
                # DEBUG: Show mean AFTER decode (before clamp)
                mean_after_decode = float(current_pixels.mean())
                print(f"[VAE_LUT_Generator]   🔍 After decode: {mean_after_decode:.6f}")
                
                # Clamp to valid range [0, 1]
                current_pixels = torch.clamp(current_pixels, 0.0, 1.0)
                
                # DEBUG: Show mean AFTER clamp
                mean_after_clamp = float(current_pixels.mean())
                if abs(mean_after_clamp - mean_after_decode) > 0.0001:
                    print(f"[VAE_LUT_Generator]   🔍 After clamp: {mean_after_clamp:.6f} (clamp changed it!)")
                
                # DEBUG: Show degradation for this cycle
                degradation = mean_after_clamp - mean_before
                degradation_pct = (degradation / mean_before) * 100
                print(f"[VAE_LUT_Generator]   📉 Cycle {cycle + 1} degradation: {degradation:+.6f} ({degradation_pct:+.2f}%)")
                
                # === CODEC DEGRADATION (Optional) ===
                # If enabled, save and reload the video to include codec degradation
                # IMPORTANT: Always do this after EVERY cycle (including the last one)
                # because we need to measure the degradation state AFTER save/load
                if include_codec_degradation:
                    print(f"[VAE_LUT_Generator]   💾 Saving/loading video (codec degradation)...")
                    mean_before_codec = float(current_pixels.mean())
                    
                    try:
                        # Save video to temporary file using H.265
                        import tempfile
                        import subprocess
                        import cv2
                        
                        temp_dir = tempfile.gettempdir()
                        temp_path = os.path.join(temp_dir, f"vae_lut_temp_cycle{cycle+1}.mp4")
                        
                        # Convert tensor to numpy with CustomVideoSaver's exact rounding
                        # This matches: video_array = np.clip(video_array * 255.0 + 0.5, 0, 255)
                        video_np = np.clip(current_pixels.cpu().numpy() * 255.0 + 0.5, 0, 255).astype(np.uint8)
                        height, width = video_np.shape[1], video_np.shape[2]
                        
                        # Use FFmpeg with EXACT CustomVideoSaver settings
                        ffmpeg_cmd = [
                            'ffmpeg', '-y',
                            '-f', 'rawvideo',
                            '-vcodec', 'rawvideo',
                            '-s', f'{width}x{height}',
                            '-pix_fmt', 'rgb24',
                            '-r', str(fps),
                            '-i', '-',
                            '-c:v', 'libx264',
                            '-movflags', '+faststart',  # CustomVideoSaver uses this
                            '-max_muxing_queue_size', '9999',  # CustomVideoSaver uses this
                            '-crf', str(codec_quality),
                            '-preset', 'medium',
                            '-pix_fmt', 'yuv420p',
                            temp_path
                        ]
                        
                        # Use DEVNULL for stdout/stderr to prevent pipe deadlock
                        process = subprocess.Popen(ffmpeg_cmd, 
                                                  stdin=subprocess.PIPE, 
                                                  stdout=subprocess.DEVNULL, 
                                                  stderr=subprocess.DEVNULL)
                        
                        # Write all frames as one block
                        video_bytes = video_np.tobytes()
                        process.stdin.write(video_bytes)
                        process.stdin.close()
                        
                        # Wait for completion
                        return_code = process.wait()
                        
                        if return_code != 0:
                            print(f"[VAE_LUT_Generator]   ⚠️  FFmpeg returned error code {return_code}")
                            raise Exception(f"FFmpeg encoding failed with code {return_code}")
                        
                        # Load video back
                        cap = cv2.VideoCapture(temp_path)
                        frames = []
                        while True:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            # Convert BGR to RGB
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frames.append(frame_rgb)
                        cap.release()
                        
                        # Clean up temp file
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                        
                        # Convert back to tensor
                        if frames:
                            current_pixels = torch.from_numpy(np.array(frames)).float() / 255.0
                            current_pixels = current_pixels.to(video_tensor.device)
                            
                            mean_after_codec = float(current_pixels.mean())
                            codec_degradation = mean_after_codec - mean_before_codec
                            codec_degradation_pct = (codec_degradation / mean_before_codec) * 100
                            print(f"[VAE_LUT_Generator]   🎬 Codec degradation: {codec_degradation:+.6f} ({codec_degradation_pct:+.2f}%)")
                        else:
                            print(f"[VAE_LUT_Generator]   ⚠️  Warning: Failed to load video frames")
                            
                    except Exception as e:
                        print(f"[VAE_LUT_Generator]   ⚠️  Warning: Codec save/load failed: {e}")
                        print(f"[VAE_LUT_Generator]   Continuing without codec degradation for this cycle")
                
                # Get current frame count
                # After reshape, tensor is always 4D: [frames, height, width, channels]
                current_frames = current_pixels.shape[0]  # Frames is dimension 0
                
                if (cycle + 1) % 2 == 0 or num_cycles <= 3:
                    print(f"[VAE_LUT_Generator]   Completed cycle {cycle + 1}/{num_cycles} (shape: {current_pixels.shape}, frames: {current_frames})")
                
                # Check if temporal VAE has compressed frames too much
                if is_temporal_vae and current_frames < 3 and cycle < num_cycles - 1:
                    print(f"[VAE_LUT_Generator] ⚠️  Frame count too low ({current_frames} frames)")
                    print(f"[VAE_LUT_Generator] ⚠️  Cannot continue to cycle {cycle + 2}")
                    print(f"[VAE_LUT_Generator] ⚠️  Stopping at cycle {cycle + 1}")
                    print(f"[VAE_LUT_Generator] ⚠️  This temporal VAE requires more input frames!")
                    raise RuntimeError(
                        f"Temporal VAE compressed video to {current_frames} frames. "
                        f"Cannot perform cycle {cycle + 2}. "
                        f"Original video had {original_frame_count} frames. "
                        f"Recommendation: Use a video with {original_frame_count * 2} or more frames, "
                        f"or reduce max_cycles to {cycle + 1}."
                    )
                    
            except Exception as e:
                print(f"[VAE_LUT_Generator] ❌ Error in cycle {cycle + 1}:")
                print(f"[VAE_LUT_Generator]    Error type: {type(e).__name__}")
                print(f"[VAE_LUT_Generator]    Error message: {e}")
                print(f"[VAE_LUT_Generator]    Current tensor shape: {current_pixels.shape if hasattr(current_pixels, 'shape') else 'N/A'}")
                
                # Provide specific guidance for temporal VAE issues
                if is_temporal_vae and ('out' in str(e) or 'UnboundLocalError' in str(e) or 'Temporal VAE' in str(e)):
                    # Get frame count from current_pixels (always 4D after reshape)
                    if hasattr(current_pixels, 'shape'):
                        current_frames = current_pixels.shape[0]
                    else:
                        current_frames = 0
                    
                    print(f"[VAE_LUT_Generator] ")
                    print(f"[VAE_LUT_Generator] 🎥 TEMPORAL VAE ISSUE DETECTED:")
                    print(f"[VAE_LUT_Generator] Your video VAE compresses the frame count with each cycle.")
                    print(f"[VAE_LUT_Generator] Frame count reduced: {original_frame_count} → {current_frames}")
                    print(f"[VAE_LUT_Generator] ")
                    print(f"[VAE_LUT_Generator] SOLUTIONS:")
                    print(f"[VAE_LUT_Generator] 1. Use MORE frames: Try {original_frame_count * 3}+ frames")
                    print(f"[VAE_LUT_Generator] 2. Use FEWER cycles: Set max_cycles to {cycle} instead of {num_cycles}")
                    print(f"[VAE_LUT_Generator] 3. Use LONGER video: This VAE needs substantial temporal data")
                    print(f"[VAE_LUT_Generator] ")
                
                import traceback
                traceback.print_exc()
                raise RuntimeError(f"VAE cycle {cycle + 1} failed. See error above for details.") from e
        
        print(f"[VAE_LUT_Generator] ✓ {num_cycles} cycles complete")
        print(f"[VAE_LUT_Generator] Output shape: {current_pixels.shape}")
        
        # DEBUG: Show total degradation
        original_mean = float(video_tensor.mean())
        final_mean = float(current_pixels.mean())
        total_degradation = final_mean - original_mean
        total_degradation_pct = (total_degradation / original_mean) * 100
        print(f"[VAE_LUT_Generator] ")
        print(f"[VAE_LUT_Generator] 📊 TOTAL DEGRADATION AFTER {num_cycles} CYCLES:")
        print(f"[VAE_LUT_Generator]    Original mean: {original_mean:.6f}")
        print(f"[VAE_LUT_Generator]    Final mean:    {final_mean:.6f}")
        print(f"[VAE_LUT_Generator]    Total drift:   {total_degradation:+.6f} ({total_degradation_pct:+.2f}%)")
        print(f"[VAE_LUT_Generator] ")
        
        return current_pixels
    
    def generate_lut(self, original_video, vae_name, vae=None, max_cycles=0, test_cycles="",
                    include_codec_degradation=False, codec_quality=23, fps=30.0, **kwargs):
        """
        Analyze multiple VAE-cycled videos and generate correction LUT.
        
        Three modes:
        1. AUTOMATIC (Simple): Provide VAE + max_cycles, tests 1..max_cycles
        2. AUTOMATIC (Advanced): Provide VAE + test_cycles string for specific cycles
        3. MANUAL: Provide pre-generated cycle_N_video inputs (legacy mode)
        
        Args:
            include_codec_degradation: If True, save/load video between cycles to include H.265 codec degradation
            codec_quality: FFmpeg CRF quality (0=lossless, 23=default, 51=worst)
            fps: Frames per second for video encoding/decoding
        """
        cycle_videos = {}
        mode = "unknown"
        
        # Check if we're in automatic mode (VAE provided)
        if vae is not None:
            mode = "automatic"
            print("[VAE_LUT_Generator] ═══════════════════════════════════════════")
            print("[VAE_LUT_Generator] AUTOMATIC MODE - Generating test videos...")
            print("[VAE_LUT_Generator] ═══════════════════════════════════════════")
            
            if include_codec_degradation:
                print(f"[VAE_LUT_Generator] 🎬 Codec degradation: ENABLED (H.264, CRF {codec_quality}, {fps} fps)")
                print(f"[VAE_LUT_Generator]    Will save/load video between cycles (matches manual workflow)")
            else:
                print(f"[VAE_LUT_Generator] ⚡ Codec degradation: DISABLED (pure VAE only)")
            
            # Determine which cycles to test
            cycle_list = []
            
            # Priority 1: max_cycles (simple mode)
            if max_cycles > 0:
                cycle_list = list(range(1, max_cycles + 1))
                print(f"[VAE_LUT_Generator] Simple Mode: Testing cycles 1 through {max_cycles}")
            
            # Priority 2: test_cycles string (advanced mode)
            elif test_cycles.strip():
                try:
                    cycle_list = [int(c.strip()) for c in test_cycles.split(",") if c.strip()]
                    cycle_list = sorted(set(cycle_list))  # Remove duplicates and sort
                    print(f"[VAE_LUT_Generator] Advanced Mode: Testing specific cycles: {cycle_list}")
                except ValueError as e:
                    return ("{}", f"Error parsing test_cycles: {e}. Use format like '1,2,3,4,5,6,8,10'")
            
            # Fallback: Use default if nothing specified
            else:
                cycle_list = [1, 2, 3, 4, 5, 6, 8, 10]
                print(f"[VAE_LUT_Generator] Using default cycles: {cycle_list}")
            
            if not cycle_list:
                return ("{}", 
                       "Error: No test cycles specified.\n"
                       "Either:\n"
                       "  1. Set max_cycles (e.g., 10 tests cycles 1-10), or\n"
                       "  2. Set test_cycles (e.g., '1,3,6,10' for specific cycles)")
            
            # Generate degraded videos for each cycle count
            try:
                print(f"[VAE_LUT_Generator] Testing {len(cycle_list)} cycle counts: {cycle_list}")
                print(f"[VAE_LUT_Generator] This will take approximately {len(cycle_list) * 5} seconds...")
                print()
                
                for i, num_cycles in enumerate(cycle_list, 1):
                    print(f"[VAE_LUT_Generator] ─── Test {i}/{len(cycle_list)}: {num_cycles} cycles ───")
                    degraded = self._apply_vae_cycles(original_video, vae, num_cycles,
                                                     include_codec_degradation=include_codec_degradation,
                                                     codec_quality=codec_quality,
                                                     fps=fps)
                    cycle_videos[num_cycles] = degraded
                    print()
                
                print("[VAE_LUT_Generator] ✓ All test videos generated successfully!")
                print()
                
            except Exception as e:
                return ("{}", f"Error during automatic testing: {e}")
        
        else:
            # Manual mode: collect pre-generated videos from kwargs
            mode = "manual"
            print("[VAE_LUT_Generator] ═══════════════════════════════════════════")
            print("[VAE_LUT_Generator] MANUAL MODE - Using pre-generated videos...")
            print("[VAE_LUT_Generator] ═══════════════════════════════════════════")
            
            for key, video in kwargs.items():
                if video is not None and key.startswith("cycle_"):
                    # Extract cycle number from key (e.g., "cycle_3_video" -> 3)
                    cycle_num = int(key.split("_")[1])
                    cycle_videos[cycle_num] = video
            
            if not cycle_videos:
                return ("{}", 
                       "Error: No cycle videos provided.\n"
                       "Either:\n"
                       "  1. Connect a VAE for automatic testing (recommended), or\n"
                       "  2. Connect at least one cycle_N_video input for manual mode")
        
        # Get original statistics
        orig_np = original_video.cpu().numpy()
        orig_stats = {
            "mean": float(orig_np.mean()),
            "std": float(orig_np.std()),
            "min": float(orig_np.min()),
            "max": float(orig_np.max()),
            "clip_0": float((orig_np == 0).sum() / orig_np.size),
            "clip_1": float((orig_np == 1).sum() / orig_np.size),
        }
        
        # Analyze each cycle video
        lut_data = {
            "vae_name": vae_name,
            "generation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "calibration_video_stats": orig_stats,
            "corrections": {},
            "drift_analysis": {},
        }
        
        drift_rates = []
        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("VAE LUT GENERATION REPORT")
        report_lines.append("=" * 70)
        report_lines.append(f"VAE Name: {vae_name}")
        report_lines.append(f"Generation Mode: {mode.upper()}")
        if mode == "automatic":
            if max_cycles > 0:
                report_lines.append(f"Test Mode: SIMPLE (cycles 1-{max_cycles})")
            elif test_cycles.strip():
                report_lines.append(f"Test Mode: ADVANCED (custom cycles)")
            else:
                report_lines.append(f"Test Mode: DEFAULT")
            report_lines.append(f"Test Cycles: {sorted(cycle_videos.keys())}")
        report_lines.append(f"Calibration Video: {orig_np.shape[0]} frames, {orig_np.shape[1]}x{orig_np.shape[2]}")
        report_lines.append(f"Original Mean: {orig_stats['mean']:.6f}")
        report_lines.append(f"Original Std:  {orig_stats['std']:.6f}")
        report_lines.append("")
        report_lines.append("CYCLE ANALYSIS:")
        report_lines.append("-" * 70)
        
        for cycle_num in sorted(cycle_videos.keys()):
            video_np = cycle_videos[cycle_num].cpu().numpy()
            
            # Calculate statistics
            stats = {
                "mean": float(video_np.mean()),
                "std": float(video_np.std()),
                "min": float(video_np.min()),
                "max": float(video_np.max()),
                "clip_0": float((video_np == 0).sum() / video_np.size),
                "clip_1": float((video_np == 1).sum() / video_np.size),
            }
            
            # Calculate drift
            mean_drift = stats["mean"] - orig_stats["mean"]
            mean_drift_pct = (mean_drift / orig_stats["mean"]) * 100
            std_drift = stats["std"] - orig_stats["std"]
            std_drift_pct = (std_drift / orig_stats["std"]) * 100
            
            # Calculate correction factors (inverse of degradation)
            brightness_correction = orig_stats["mean"] / stats["mean"] if stats["mean"] > 0 else 1.0
            contrast_correction = orig_stats["std"] / stats["std"] if stats["std"] > 0 else 1.0
            
            # Calculate shadow lift needed
            clip_increase = stats["clip_0"] - orig_stats["clip_0"]
            shadow_lift = max(0.0, clip_increase * 0.05)  # Heuristic: 5% of clipping increase
            
            # Store in LUT
            lut_data["corrections"][cycle_num] = {
                "brightness_mult": round(brightness_correction, 6),
                "contrast_mult": round(contrast_correction, 6),
                "shadow_lift": round(shadow_lift, 6),
            }
            
            lut_data["drift_analysis"][cycle_num] = {
                "mean_drift": round(mean_drift, 6),
                "mean_drift_pct": round(mean_drift_pct, 2),
                "std_drift": round(std_drift, 6),
                "std_drift_pct": round(std_drift_pct, 2),
                "clip_0_increase": round(clip_increase * 100, 2),
            }
            
            drift_rates.append((cycle_num, mean_drift_pct / cycle_num))
            
            # Add to report
            report_lines.append(f"Cycle {cycle_num}:")
            report_lines.append(f"  Mean: {stats['mean']:.6f} (drift: {mean_drift_pct:+.2f}%)")
            report_lines.append(f"  Std:  {stats['std']:.6f} (drift: {std_drift_pct:+.2f}%)")
            report_lines.append(f"  Clip0: {stats['clip_0']*100:.2f}% (increase: {clip_increase*100:+.2f}%)")
            report_lines.append(f"  → Brightness correction: ×{brightness_correction:.4f}")
            report_lines.append(f"  → Contrast correction:   ×{contrast_correction:.4f}")
            report_lines.append(f"  → Shadow lift:           +{shadow_lift:.4f}")
            report_lines.append("")
        
        # Estimate drift model
        if len(drift_rates) >= 2:
            avg_drift_rate = sum(rate for _, rate in drift_rates) / len(drift_rates)
            lut_data["drift_model"] = {
                "type": "linear",
                "mean_decay_rate_per_cycle": round(abs(avg_drift_rate) / 100, 6),
            }
            
            report_lines.append("=" * 70)
            report_lines.append("DRIFT MODEL:")
            report_lines.append(f"Average drift rate: {avg_drift_rate:.2f}% per cycle")
            report_lines.append(f"Model type: Linear")
            report_lines.append("")
        
        # Usage instructions
        report_lines.append("=" * 70)
        report_lines.append("USAGE:")
        report_lines.append("1. Save the LUT JSON output to a file (e.g., 'my_vae_lut.json')")
        report_lines.append("2. Use 'VAE Correction Applier' node with this LUT")
        report_lines.append("3. Specify the number of VAE cycles your video has undergone")
        report_lines.append("")
        if mode == "automatic":
            report_lines.append("NOTE: Generated in AUTOMATIC mode - all test videos created internally!")
            report_lines.append(f"      Tested {len(cycle_videos)} cycle counts in ~{len(cycle_videos) * 5} seconds")
            if max_cycles > 0:
                report_lines.append(f"      Simple mode used: tested every cycle from 1 to {max_cycles}")
            elif test_cycles.strip():
                report_lines.append(f"      Advanced mode used: tested specific cycles {sorted(cycle_videos.keys())}")
        report_lines.append("=" * 70)
        
        lut_json = json.dumps(lut_data, indent=2)
        report_text = "\n".join(report_lines)
        
        print(f"\n{report_text}")
        print(f"\nGenerated LUT JSON:\n{lut_json}")
        
        return (lut_json, report_text)


class VAE_Correction_Applier:
    """
    Apply VAE correction using a LUT generated by VAE_LUT_Generator.
    Restores brightness, contrast, and shadow detail lost during VAE encode/decode cycles.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("IMAGE",),
                "lut_json": ("STRING", {"default": "{}", "multiline": True}),
                "num_cycles": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 50,
                    "step": 1,
                }),
                "correction_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "slider",
                }),
                "content_adaptive": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("corrected_video", "correction_report")
    FUNCTION = "apply_correction"
    CATEGORY = "NV_Comfy_Utils/Diagnostic"
    
    def apply_correction(self, video, lut_json, num_cycles, correction_strength=1.0, content_adaptive=True):
        """
        Apply VAE cycle corrections to restore video quality.
        """
        if num_cycles == 0:
            return (video, "No correction needed (0 cycles)")
        
        # Parse LUT
        try:
            lut = json.loads(lut_json)
        except json.JSONDecodeError as e:
            return (video, f"Error: Invalid LUT JSON: {e}")
        
        if "corrections" not in lut:
            return (video, "Error: LUT JSON missing 'corrections' key")
        
        corrections = lut["corrections"]
        
        # Find correction for this cycle count (or interpolate)
        if str(num_cycles) in corrections:
            corr = corrections[str(num_cycles)]
        elif num_cycles in corrections:
            corr = corrections[num_cycles]
        else:
            # Interpolate from available data
            available_cycles = sorted([int(k) for k in corrections.keys()])
            if not available_cycles:
                return (video, "Error: LUT has no correction data")
            
            if num_cycles < available_cycles[0]:
                # Extrapolate downward
                corr = corrections[str(available_cycles[0])]
                scale = num_cycles / available_cycles[0]
                corr = {
                    "brightness_mult": 1 + (corr["brightness_mult"] - 1) * scale,
                    "contrast_mult": 1 + (corr["contrast_mult"] - 1) * scale,
                    "shadow_lift": corr["shadow_lift"] * scale,
                }
            elif num_cycles > available_cycles[-1]:
                # Extrapolate upward
                corr = corrections[str(available_cycles[-1])]
                scale = num_cycles / available_cycles[-1]
                corr = {
                    "brightness_mult": 1 + (corr["brightness_mult"] - 1) * scale,
                    "contrast_mult": 1 + (corr["contrast_mult"] - 1) * scale,
                    "shadow_lift": corr["shadow_lift"] * scale,
                }
            else:
                # Interpolate between two points
                lower = max(c for c in available_cycles if c < num_cycles)
                upper = min(c for c in available_cycles if c > num_cycles)
                weight = (num_cycles - lower) / (upper - lower)
                
                corr_lower = corrections[str(lower)]
                corr_upper = corrections[str(upper)]
                
                corr = {
                    "brightness_mult": corr_lower["brightness_mult"] * (1 - weight) + corr_upper["brightness_mult"] * weight,
                    "contrast_mult": corr_lower["contrast_mult"] * (1 - weight) + corr_upper["contrast_mult"] * weight,
                    "shadow_lift": corr_lower["shadow_lift"] * (1 - weight) + corr_upper["shadow_lift"] * weight,
                }
        
        # Convert to numpy
        video_np = video.cpu().numpy()
        original_mean = video_np.mean()
        original_std = video_np.std()
        
        # Content-adaptive scaling
        if content_adaptive and "calibration_video_stats" in lut:
            calib_mean = lut["calibration_video_stats"]["mean"]
            brightness_ratio = original_mean / calib_mean if calib_mean > 0 else 1.0
            
            # Adjust corrections based on content brightness
            # Dark videos need stronger correction
            if brightness_ratio < 0.8:
                content_factor = 1.2
            elif brightness_ratio > 1.2:
                content_factor = 0.85
            else:
                content_factor = 1.0
            
            corr["brightness_mult"] = 1 + (corr["brightness_mult"] - 1) * content_factor
            corr["shadow_lift"] = corr["shadow_lift"] * content_factor
        
        # Apply correction strength multiplier
        brightness_mult = 1 + (corr["brightness_mult"] - 1) * correction_strength
        contrast_mult = 1 + (corr["contrast_mult"] - 1) * correction_strength
        shadow_lift = corr["shadow_lift"] * correction_strength
        
        # Apply corrections
        corrected = video_np.copy()
        
        # 1. Brightness correction
        corrected = corrected * brightness_mult
        
        # 2. Contrast correction (around mean)
        mean = corrected.mean()
        corrected = (corrected - mean) * contrast_mult + mean
        
        # 3. Shadow lift
        corrected = corrected + shadow_lift
        
        # 4. Clamp to valid range
        corrected = np.clip(corrected, 0.0, 1.0)
        
        # Convert back to tensor
        corrected_tensor = torch.from_numpy(corrected.astype(np.float32))
        
        # Generate report
        final_mean = corrected.mean()
        final_std = corrected.std()
        mean_change = ((final_mean - original_mean) / original_mean) * 100
        std_change = ((final_std - original_std) / original_std) * 100
        
        report = []
        report.append("VAE CORRECTION APPLIED")
        report.append("=" * 50)
        report.append(f"VAE: {lut.get('vae_name', 'Unknown')}")
        report.append(f"Cycles: {num_cycles}")
        report.append(f"Correction Strength: {correction_strength:.1%}")
        report.append(f"Content Adaptive: {'Yes' if content_adaptive else 'No'}")
        report.append("")
        report.append("CORRECTIONS APPLIED:")
        report.append(f"  Brightness: ×{brightness_mult:.4f}")
        report.append(f"  Contrast:   ×{contrast_mult:.4f}")
        report.append(f"  Shadow Lift: +{shadow_lift:.4f}")
        report.append("")
        report.append("RESULTS:")
        report.append(f"  Original mean: {original_mean:.6f}")
        report.append(f"  Corrected mean: {final_mean:.6f} ({mean_change:+.2f}%)")
        report.append(f"  Original std: {original_std:.6f}")
        report.append(f"  Corrected std: {final_std:.6f} ({std_change:+.2f}%)")
        report.append("=" * 50)
        
        report_text = "\n".join(report)
        print(f"\n{report_text}")
        
        return (corrected_tensor, report_text)


class NV_VideoSampler:
    """
    Custom video sampler with automation-friendly features + SDE support.
    
    Features:
    - SDE (Stochastic) sampling with multiple noise types
    - Brownian/Pyramid noise for temporal consistency (RES4LYF quality if available)
    - RES samplers (res_2m, res_3s, etc.) if RES4LYF installed
    - Custom schedulers (beta57, bong_tangent) if available
    - Unsample/Resample for advanced img2img
    - Dynamic CFG and automation support
    """
    
    # Try to import RES4LYF noise generators
    try:
        from RES4LYF.beta.noise_classes import (
            BrownianNoiseGenerator,
            PyramidNoiseGenerator,
            HiresPyramidNoiseGenerator
        )
        RES4LYF_NOISE_AVAILABLE = True
    except ImportError:
        RES4LYF_NOISE_AVAILABLE = False
    
    # RES4LYF samplers/schedulers availability is checked via comfy.samplers lists
    # (RES4LYF automatically registers them when loaded)
    
    @classmethod
    def INPUT_TYPES(s):
        import comfy.samplers
        
        # Build sampler list
        # RES4LYF samplers (res_2m, res_3s, etc.) are automatically registered
        # in comfy.samplers.KSampler.SAMPLERS by RES4LYF's add_samplers() function
        all_samplers = list(comfy.samplers.KSampler.SAMPLERS)
        
        # Build scheduler list
        # RES4LYF registers bong_tangent and other schedulers in comfy.samplers.SCHEDULER_NAMES
        base_schedulers = list(comfy.samplers.KSampler.SCHEDULERS)
        
        # Add beta57 as custom scheduler if not present
        if "beta57" not in base_schedulers:
            all_schedulers = sorted(base_schedulers + ["beta57"])
        else:
            all_schedulers = base_schedulers
        
        # Build noise type list
        if s.RES4LYF_NOISE_AVAILABLE:
            noise_types = ["gaussian", "brownian", "pyramid", "hires-pyramid", "pink", "blue"]
        else:
            noise_types = ["gaussian", "brownian", "pyramid", "pink", "blue"]
        
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "The model used for denoising (works with any ComfyUI model type)"
                }),
                "seed": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 0xffffffffffffffff,
                    "tooltip": "Random seed for noise generation (ensures reproducibility)"
                }),
                "steps": ("INT", {
                    "default": 30, 
                    "min": 1, 
                    "max": 200,
                    "tooltip": "Number of denoising steps (more = higher quality but slower)"
                }),
                "cfg": ("FLOAT", {
                    "default": 6.0, 
                    "min": 0.0, 
                    "max": 30.0, 
                    "step": 0.1,
                    "tooltip": "Classifier-Free Guidance scale (1=no guidance, 7-8=balanced, 15+=very strong)"
                }),
                "sampler_name": (all_samplers, {
                    "default": "euler",
                    "tooltip": "RES samplers (res_2m, res_3s, res_5s) available if RES4LYF installed"
                }),
                "scheduler": (all_schedulers, {
                    "default": "normal",
                    "tooltip": "beta57/bong_tangent recommended for video (if RES4LYF installed)"
                }),
                "sampler_mode": (["standard", "unsample", "resample"], {
                    "default": "standard",
                    "tooltip": "standard=normal, unsample=reverse to noise, resample=denoise after unsample"
                }),
            },
            "optional": {
                "latent_image": ("LATENT", {
                    "tooltip": "Input latent (canvas). Can be auto-generated from preprocessor or provided manually for img2img/vid2vid."
                }),
                "positive": ("CONDITIONING", {
                    "tooltip": "Positive conditioning (what you want). REQUIRED for standard mode. Optional for chunked mode (used as fallback)."
                }),
                "negative": ("CONDITIONING", {
                    "tooltip": "Negative conditioning (what to avoid). REQUIRED for standard mode. Optional for chunked mode (used as fallback)."
                }),
                "denoise_strength": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.01,
                    "tooltip": "Amount of denoising (1.0=full, <1.0=partial for video2video)"
                }),
                "start_step": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 10000,
                    "tooltip": "Start sampling at this step (for advanced workflows)"
                }),
                "end_step": ("INT", {
                    "default": -1, 
                    "min": -1, 
                    "max": 10000,
                    "tooltip": "End sampling at this step (-1=full sampling)"
                }),
                "cfg_end": ("FLOAT", {
                    "default": -1.0,
                    "min": -1.0,
                    "max": 30.0,
                    "step": 0.1,
                    "tooltip": "End CFG value for dynamic CFG (-1=use same as cfg throughout)"
                }),
                "add_noise": (["enable", "disable"], {
                    "default": "enable",
                    "tooltip": "Add initial noise (disable for img2img with already-noised latents)"
                }),
                "return_with_leftover_noise": (["disable", "enable"], {
                    "default": "disable",
                    "tooltip": "Return with remaining noise (enable for multi-pass sampling)"
                }),
                "eta": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": "SDE noise amount (0=ODE/deterministic, 0.25=subtle, 0.5=balanced, 1.0=full SDE)"
                }),
                "noise_type": (noise_types, {
                    "default": "gaussian",
                    "tooltip": "brownian=best for video (RES4LYF quality if available), hires-pyramid for high-res"
                }),
                "noise_mode": (["hard", "soft", "comfy"], {
                    "default": "hard",
                    "tooltip": "hard=aggressive, soft=gradual decay, comfy=balanced"
                }),
                "noise_seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Separate seed for SDE noise (-1 = use main seed + 1)"
                }),
                "per_chunk_seed_offset": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use different seed for each chunk (seed + chunk_idx). Adds variation while staying deterministic."
                }),
                "chunk_conditionings": ("CHUNK_CONDITIONING_LIST", {
                    "tooltip": "Optional: Enable context window sampling (from NV_ChunkConditioningPreprocessor)"
                }),
                "blend_transition_frames": ("INT", {
                    "default": 16,
                    "min": 4,
                    "max": 64,
                    "step": 4,
                    "tooltip": "Crossfade duration in frames. Should match VACE overlap (16=default overlap, 32=extended blend)"
                }),
                "vae": ("VAE", {
                    "tooltip": "VAE for Tier 3 VACE overlap encoding (optional, enables strong temporal consistency)"
                }),
                "enable_diffusion_refine": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable diffusion-based quality restoration for Tier 3 extension frames (fixes VAE degradation, ~6-10% overhead)"
                }),
                "refine_denoise": ("FLOAT", {
                    "default": 0.25,
                    "min": 0.1,
                    "max": 0.5,
                    "step": 0.05,
                    "tooltip": "Denoise strength for extension frame refinement (0.25=balanced, 0.1=subtle, 0.5=heavy)"
                }),
                "refine_steps": ("INT", {
                    "default": 5,
                    "min": 3,
                    "max": 15,
                    "step": 1,
                    "tooltip": "Diffusion steps for refinement (5=fast, 10=quality, 15=overkill)"
                }),
            }
        }
    
    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("samples", "info")
    FUNCTION = "sample"
    CATEGORY = "NV_Utils/sampling"
    DESCRIPTION = """
    NV Video Sampler - Wan-focused video automation sampler
    
    Features:
    • Optimized for Wan video diffusion models
    • Context window chunking for long videos
    • Per-chunk VACE control support (depth, pose, etc.)
    • Dynamic CFG for temporal consistency
    • Selector input for workflow automation
    • Latent space blending for seamless chunk transitions
    • Compatible with standard models (SD/SDXL) for testing
    
    Wan Integration:
    • VACE controls properly formatted for Wan sampling
    • Multiple controls per chunk via additional_vace_inputs
    • Chunk-specific control weights and timing
    • Video-optimized sampling parameters
    """
    
    def sample(self, model, seed, steps, cfg,
               sampler_name, scheduler, sampler_mode="standard", latent_image=None,
               positive=None, negative=None,
               denoise_strength=1.0, start_step=0, end_step=-1,
               cfg_end=-1.0, add_noise="enable",
               return_with_leftover_noise="disable", eta=0.0, noise_type="gaussian",
               noise_mode="hard", noise_seed=-1, per_chunk_seed_offset=False, chunk_conditionings=None,
               blend_transition_frames=32, vae=None,
               enable_diffusion_refine=False, refine_denoise=0.25, refine_steps=5):
        
        import comfy.sample
        import comfy.samplers
        import comfy.utils
        import time
        import math

        start_time = time.time()

        # Store sampling parameters for diffusion refinement (used in Tier 3 extension frame processing)
        self._refine_model = model
        self._refine_sampler_name = sampler_name
        self._refine_scheduler = scheduler
        self._refine_cfg = cfg
        self._refine_positive = positive
        self._refine_negative = negative
        self._refine_seed = seed
        self._refine_enable = enable_diffusion_refine
        self._refine_denoise = refine_denoise
        self._refine_steps = refine_steps

        # Validate inputs
        has_chunk_cond = chunk_conditionings is not None and len(chunk_conditionings) > 0
        has_global_cond = positive is not None and negative is not None
        has_latent = latent_image is not None
        
        # Check conditioning
        if not has_chunk_cond and not has_global_cond:
            raise ValueError(
                "Either provide chunk_conditionings (from NV_ChunkConditioningPreprocessor) "
                "OR provide both positive and negative conditioning.\n"
                "For chunked workflows: Only connect chunk_conditionings.\n"
                "For standard workflows: Connect positive and negative from CLIP encoding."
            )
        
        # Check latent
        if not has_latent:
            raise ValueError(
                "No latent_image provided!\n"
                "Connect the 'latent' output from NV_ChunkConditioningPreprocessor (auto-generated)\n"
                "OR provide an Empty Latent Video / VAE Encode output for custom dimensions."
            )
        
        # Get model sampling for sigma ranges (needed for RES4LYF noise)
        model_sampling = model.get_model_object("model_sampling")
        sigma_min = float(model_sampling.sigma_min) if hasattr(model_sampling, 'sigma_min') else 0.03
        sigma_max = float(model_sampling.sigma_max) if hasattr(model_sampling, 'sigma_max') else 14.6
        
        # SDE noise generator functions
        def generate_noise_res4lyf(noise_type, samples, noise_seed, sigma_current=None, sigma_next=None):
            """Generate noise using RES4LYF generators (best quality)"""
            if noise_type == "brownian":
                gen = self.BrownianNoiseGenerator(
                    x=samples,
                    sigma_min=sigma_min,
                    sigma_max=sigma_max,
                    seed=noise_seed
                )
                return gen(sigma=sigma_current, sigma_next=sigma_next)
            elif noise_type == "pyramid":
                gen = self.PyramidNoiseGenerator(
                    x=samples,
                    sigma_min=sigma_min,
                    sigma_max=sigma_max,
                    seed=noise_seed,
                    discount=0.8,
                    mode='nearest-exact'
                )
                return gen()
            elif noise_type == "hires-pyramid":
                gen = self.HiresPyramidNoiseGenerator(
                    x=samples,
                    sigma_min=sigma_min,
                    sigma_max=sigma_max,
                    seed=noise_seed,
                    discount=0.7,
                    mode='nearest-exact'
                )
                return gen()
            return None
        
        def generate_brownian_noise_fallback(shape, device, dtype):
            """Fallback brownian (simplified)"""
            noise = torch.randn(shape, device=device, dtype=dtype)
            if len(shape) == 5 and shape[2] > 1:  # [B, C, F, H, W]
                kernel_size = 3
                kernel = torch.ones(1, 1, kernel_size, 1, 1, device=device, dtype=dtype) / kernel_size
                noise = torch.nn.functional.conv3d(
                    noise, kernel, padding=(kernel_size//2, 0, 0)
                )
            return noise
        
        def generate_pyramid_noise_fallback(shape, device, dtype):
            """Fallback pyramid (simplified)"""
            noise = torch.zeros(shape, device=device, dtype=dtype)
            for scale in [1, 2, 4]:
                if len(shape) == 5:  # Video
                    scaled_shape = (shape[0], shape[1], 
                                  max(1, shape[2]//scale),
                                  max(1, shape[3]//scale), 
                                  max(1, shape[4]//scale))
                else:  # Image
                    scaled_shape = (shape[0], shape[1],
                                  max(1, shape[2]//scale),
                                  max(1, shape[3]//scale))
                
                scale_noise = torch.randn(scaled_shape, device=device, dtype=dtype)
                scale_noise = torch.nn.functional.interpolate(
                    scale_noise if len(shape) == 4 else scale_noise.flatten(0, 1),
                    size=shape[2:] if len(shape) == 4 else shape[3:],
                    mode='nearest'
                )
                if len(shape) == 5:
                    scale_noise = scale_noise.view(shape[0], shape[1], shape[2], shape[3], shape[4])
                noise += scale_noise / scale
            return noise / noise.std()
        
        def generate_pink_noise(shape, device, dtype):
            """Pink (1/f) noise"""
            noise = torch.randn(shape, device=device, dtype=dtype)
            # FFT-based pink noise generation
            fft_noise = torch.fft.fftn(noise, dim=list(range(2, len(shape))))
            freqs = torch.fft.fftfreq(shape[-1], device=device)
            freq_mag = torch.sqrt(freqs**2 + 1e-8)
            fft_noise = fft_noise / (freq_mag.view(*([1]*(len(shape)-1)), -1) + 1e-8)
            return torch.fft.ifftn(fft_noise, dim=list(range(2, len(shape)))).real
        
        def generate_blue_noise(shape, device, dtype):
            """Blue (high-frequency) noise"""
            noise = torch.randn(shape, device=device, dtype=dtype)
            fft_noise = torch.fft.fftn(noise, dim=list(range(2, len(shape))))
            freqs = torch.fft.fftfreq(shape[-1], device=device)
            freq_mag = torch.sqrt(freqs**2 + 1e-8)
            fft_noise = fft_noise * freq_mag.view(*([1]*(len(shape)-1)), -1)
            return torch.fft.ifftn(fft_noise, dim=list(range(2, len(shape)))).real
        
        def calculate_noise_scaling(sigma, sigma_next, noise_mode):
            """Calculate how much noise to add based on mode"""
            if noise_mode == "hard":
                return (sigma_next**2 * (sigma**2 - sigma_next**2) / sigma**2).sqrt()
            elif noise_mode == "soft":
                # Gradual decay
                progress = 1.0 - (sigma_next / sigma)
                return (sigma_next**2 * (sigma**2 - sigma_next**2) / sigma**2).sqrt() * (1.0 - progress)
            elif noise_mode == "comfy":
                # Balanced approach
                return ((sigma_next**2 * (sigma**2 - sigma_next**2)) / sigma**2).sqrt() * 0.5
            return torch.tensor(0.0, device=sigma.device)
        
        def get_beta57_sigmas(model_sampling, steps):
            """Beta schedule with alpha=0.5, beta=0.7"""
            timesteps = 1000
            beta_start = 0.00085
            beta_end = 0.012
            
            betas = (
                torch.linspace(beta_start**0.5, beta_end**0.5, timesteps, dtype=torch.float32) ** 2
            )
            
            alphas = 1.0 - betas
            alphas_cumprod = torch.cumprod(alphas, dim=0)
            
            sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
            
            # Sample evenly
            timestep_indices = torch.linspace(0, len(sigmas) - 1, steps).long()
            sigmas = sigmas[timestep_indices]
            
            return torch.cat([sigmas, torch.zeros(1)])
        
        def get_custom_scheduler_sigmas(scheduler_name, steps):
            """Get sigmas for custom schedulers using RES4LYF"""
            try:
                # Import RES4LYF scheduler function
                from RES4LYF.res4lyf import calculate_sigmas_RES4LYF
                
                # This should work with custom schedulers
                sigmas = calculate_sigmas_RES4LYF(model_sampling, scheduler_name, steps)
                return sigmas
            except Exception as e:
                info_lines.append(f"  Warning: Failed to use RES4LYF scheduler {scheduler_name}: {e}")
                return None
        
        # Noise seed handling
        if noise_seed == -1:
            noise_seed = seed + 1
        
        # ============================================================
        # STEP 1: Extract latent and prepare basic info
        # ============================================================
        latent = latent_image["samples"]
        
        # Handle both image (4D) and video (5D) latents
        is_image_latent = latent.ndim == 4
        if is_image_latent:
            # Convert image latent [B, C, H, W] to video latent [B, C, 1, H, W]
            latent = latent.unsqueeze(2)  # Add temporal dimension
        
        batch_size, channels, frames, height, width = latent.shape
        
        # Start building info output
        info_lines = []
        info_lines.append("=" * 60)
        info_lines.append("NV VIDEO SAMPLER - SAMPLING INFO")
        info_lines.append("=" * 60)
        
        # Report latent source
        latent_source = "from preprocessor (auto-generated)" if has_chunk_cond and not has_global_cond else "user-provided"
        info_lines.append(f"Latent source: {latent_source}")
        info_lines.append(f"Input Type: {'IMAGE (4D)' if is_image_latent else 'VIDEO (5D)'}")
        info_lines.append(f"Latent Shape: {list(latent.shape)} (B×C×F×H×W)")
        if not is_image_latent:
            info_lines.append(f"Video Frames: {(frames - 1) * 4 + 1} (latent: {frames})")
        info_lines.append(f"Resolution: {height * 8}×{width * 8} (latent: {height}×{width})")
        info_lines.append("")
        
        # ============================================================
        # STEP 2: Calculate sigma schedule (noise levels)
        # ============================================================
        # Calculate base sigmas
        if scheduler == "beta57":
            sigmas = get_beta57_sigmas(model_sampling, steps)
        elif scheduler not in comfy.samplers.KSampler.SCHEDULERS:
            # Try RES4LYF custom scheduler (e.g., bong_tangent)
            custom_sigmas = get_custom_scheduler_sigmas(scheduler, steps)
            if custom_sigmas is not None:
                sigmas = custom_sigmas
            else:
                # Fallback to normal
                sigmas = comfy.samplers.calculate_sigmas(model_sampling, "normal", steps)
                info_lines.append(f"  Warning: Unknown scheduler '{scheduler}', using 'normal' as fallback")
        else:
            sigmas = comfy.samplers.calculate_sigmas(
                model_sampling,
                scheduler,
                steps
            )
        
        # Handle partial denoising (for video2video workflows)
        if denoise_strength < 1.0:
            # Calculate total steps needed for this denoise strength
            total_steps = int(steps / denoise_strength)
            sigmas_full = comfy.samplers.calculate_sigmas(
                model_sampling, 
                scheduler, 
                total_steps
            )
            # Take the last N+1 sigmas (N steps + final 0)
            sigmas = sigmas_full[-(steps + 1):]
        
        # Apply start_step and end_step slicing
        if end_step >= 0 and end_step < (len(sigmas) - 1):
            sigmas = sigmas[:end_step + 1]
            if return_with_leftover_noise == "disable":
                sigmas[-1] = 0.0  # Force clean output
        
        if start_step > 0 and start_step < (len(sigmas) - 1):
            sigmas = sigmas[start_step:]
        
        info_lines.append("SAMPLING SCHEDULE:")
        info_lines.append(f"  Scheduler: {scheduler}")
        info_lines.append(f"  Sampler: {sampler_name}")
        info_lines.append(f"  Mode: {sampler_mode}")
        info_lines.append(f"  Total Steps: {len(sigmas) - 1}")
        info_lines.append(f"  Denoise Strength: {denoise_strength:.2f}")
        if cfg_end > 0 and cfg_end != cfg:
            info_lines.append(f"  CFG: {cfg:.1f} → {cfg_end:.1f} (dynamic)")
        else:
            info_lines.append(f"  CFG: {cfg:.1f} (constant)")
        info_lines.append(f"  Sigma Range: [{sigmas[0]:.4f}, {sigmas[-1]:.4f}]")
        if eta > 0:
            noise_quality = "RES4LYF" if self.RES4LYF_NOISE_AVAILABLE and noise_type in ["brownian", "pyramid", "hires-pyramid"] else "builtin"
            info_lines.append(f"  SDE: eta={eta:.2f}, noise={noise_type} ({noise_quality}), mode={noise_mode}")
        
        # Check if RES samplers are available (they would be in the sampler list)
        has_res_samplers = any(s.startswith("res_") for s in comfy.samplers.KSampler.SAMPLERS)
        if has_res_samplers:
            info_lines.append(f"  RES4LYF: Samplers available ({len([s for s in comfy.samplers.KSampler.SAMPLERS if s.startswith('res_')])} RES variants)")
        info_lines.append("")
        
        # ============================================================
        # STEP 3: Prepare noise
        # ============================================================
        # Handle unsample/resample modes
        if sampler_mode in ["unsample", "resample"]:
            disable_noise = True
            add_noise = "disable"
        else:
            disable_noise = (add_noise == "disable")
        
        if disable_noise:
            noise = torch.zeros(
                latent.size(), 
                dtype=latent.dtype, 
                layout=latent.layout, 
                device="cpu"
            )
            info_lines.append(f"NOISE: Disabled (mode={sampler_mode})")
        else:
            batch_inds = latent_image.get("batch_index", None)
            noise = comfy.sample.prepare_noise(latent, seed, batch_inds)
            info_lines.append(f"NOISE: Generated with seed {seed}")
        
        # Handle noise mask (for inpainting)
        noise_mask = latent_image.get("noise_mask", None)
        if noise_mask is not None:
            info_lines.append("MASK: Inpainting mask detected")
        
        info_lines.append("")
        
        # ============================================================
        # STEP 4: Dynamic CFG setup
        # ============================================================
        use_dynamic_cfg = (cfg_end > 0 and cfg_end != cfg)
        
        def get_cfg_for_step(step_idx, total_steps):
            """Calculate CFG value for current step"""
            if not use_dynamic_cfg:
                return cfg
            # Linear interpolation from cfg to cfg_end
            progress = step_idx / max(total_steps - 1, 1)
            return cfg + (cfg_end - cfg) * progress
        
        # ============================================================
        # STEP 5: Run the sampler
        # ============================================================
        info_lines.append("SAMPLING:")
        info_lines.append(f"  Starting sampling with {sampler_name}...")
        
        # Set up progress bar
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        
        # Determine full denoise behavior
        force_full_denoise = (return_with_leftover_noise == "disable")
        
        # Fix latent channels if needed
        latent = comfy.sample.fix_empty_latent_channels(model, latent)
        
        # For dynamic CFG, we need to wrap the model
        if use_dynamic_cfg:
            # We'll handle this through model options
            model_options = {
                "transformer_options": {
                    "dynamic_cfg": True,
                    "cfg_start": cfg,
                    "cfg_end": cfg_end,
                }
            }
        else:
            model_options = {}
        
        # Handle unsample mode (reverse sigmas)
        if sampler_mode == "unsample":
            sigmas = torch.flip(sigmas, [0])
            info_lines.append("  Mode: UNSAMPLE (reversing to noise)")
        elif sampler_mode == "resample":
            info_lines.append("  Mode: RESAMPLE (denoising after unsample)")
        
        # ============================================================
        # CONTEXT WINDOW SAMPLING (if chunk_conditionings provided)
        # ============================================================
        if chunk_conditionings is not None and len(chunk_conditionings) > 0:
            start_time = time.time()
            
            # Store VAE reference for Tier 3 (VACE overlap encoding)
            # This is needed to decode/re-encode overlap regions
            self._temp_vae = vae
            
            # Extract start_image pixels for color matching (if available)
            self._temp_start_image = None
            self._original_reference = None  # Store original to prevent accumulation
            if len(chunk_conditionings) > 0:
                first_chunk = chunk_conditionings[0]
                if "start_image_pixels" in first_chunk and first_chunk["start_image_pixels"] is not None:
                    self._temp_start_image = first_chunk["start_image_pixels"]
                    self._original_reference = self._temp_start_image.clone()  # Keep pristine copy
                    print(f"✓ Start image available for color matching: {self._temp_start_image.shape}")
                    print(f"✓ Original reference stored to prevent contrast accumulation")
                else:
                    print(f"⚠️  WARNING: No start_image_pixels found in chunk_conditionings")
                    print(f"   Available keys: {list(first_chunk.keys())}")
                    print(f"   Color matching will be SKIPPED!")
            
            print(f"\n{'='*60}")
            print(f"CONTEXT WINDOW SAMPLING MODE")
            print(f"{'='*60}")
            print(f"Total chunks: {len(chunk_conditionings)}")
            print(f"Latent shape: {latent.shape}")
            if vae is not None:
                print(f"VAE available: Tier 3 (VACE overlap) ENABLED")
            else:
                print(f"VAE not provided: Tier 3 (VACE overlap) DISABLED")
            print("")
            
            info_lines.append("=" * 60)
            info_lines.append("CONTEXT WINDOW SAMPLING ENABLED")
            info_lines.append("=" * 60)
            info_lines.append(f"Total chunks: {len(chunk_conditionings)}")
            info_lines.append("")
            
            # Initialize full output latent (preserve original as fallback)
            output_latent = latent.clone()  # Start with original latent
            weight_sum = torch.ones_like(latent)  # Weight 1.0 for original
            
            # Track which regions will be overwritten
            processed_regions = torch.zeros(latent.shape[2], dtype=torch.bool)  # Per latent frame
            for chunk_cond in chunk_conditionings:
                chunk_start = chunk_cond.get("latent_start", chunk_cond["start_frame"] // 4)
                chunk_end = chunk_cond.get("latent_end", (chunk_cond["end_frame"] - 1) // 4 + 1)
                processed_regions[chunk_start:chunk_end] = True
            
            # Warn if there are gaps
            unprocessed_frames = (~processed_regions).sum().item()
            if unprocessed_frames > 0:
                info_lines.append(f"⚠️ WARNING: {unprocessed_frames} latent frames not covered by any chunk!")
                info_lines.append(f"⚠️ These will use original/input latent (may cause artifacts)")
                info_lines.append("")
            
            # Reset weight_sum to zero for regions that will be processed
            # (so we can accumulate weighted chunks)
            for chunk_cond in chunk_conditionings:
                chunk_start = chunk_cond.get("latent_start", chunk_cond["start_frame"] // 4)
                chunk_end = chunk_cond.get("latent_end", (chunk_cond["end_frame"] - 1) // 4 + 1)
                output_latent[:, :, chunk_start:chunk_end, :, :] = 0  # Clear for accumulation
                weight_sum[:, :, chunk_start:chunk_end, :, :] = 0
            
            # Calculate chunk overlap from first two chunks (if multiple exist)
            if len(chunk_conditionings) > 1:
                chunk0_end = chunk_conditionings[0].get("latent_end", (chunk_conditionings[0]["end_frame"] - 1) // 4 + 1)
                chunk1_start = chunk_conditionings[1].get("latent_start", chunk_conditionings[1]["start_frame"] // 4)
                chunk_overlap = chunk0_end - chunk1_start if chunk1_start < chunk0_end else 0
            else:
                chunk_overlap = 0
            
            # Process each chunk
            print(f"\n{'='*60}")
            print(f"NV_VIDEOSAMPLER: Processing {len(chunk_conditionings)} chunk(s)")
            print(f"{'='*60}")
            
            # Initialize Tier 3 VACE control tracker
            prev_chunk_vace_control = None

            # Initialize VACE reference (will be updated per-chunk)
            vace_reference = None  # Will be set from chunk 0, then updated from last frame of each chunk

            for chunk_idx, chunk_cond in enumerate(chunk_conditionings):
                chunk_start_time = time.time()
                
                # Calculate latent frame indices (with fallback for image-based chunks)
                chunk_start = chunk_cond.get("latent_start", chunk_cond["start_frame"] // 4)
                chunk_end = chunk_cond.get("latent_end", (chunk_cond["end_frame"] - 1) // 4 + 1)
                chunk_frames = chunk_end - chunk_start

                # Per-chunk seed variation (if enabled)
                chunk_seed = seed + chunk_idx if per_chunk_seed_offset else seed

                # Handle problematic seeds (0, 1, 2 often produce brown noise in diffusion models)
                if chunk_seed in [0, 1, 2]:
                    safe_seed = 42422 + chunk_seed  # Use safe fallback seed with offset
                    print(f"  ⚠️ Seed {chunk_seed} detected (problematic), using fallback seed {safe_seed}")
                    chunk_seed = safe_seed

                print(f"\n[Chunk {chunk_idx + 1}/{len(chunk_conditionings)}] Starting...")
                print(f"  Frame range: {chunk_cond['start_frame']}-{chunk_cond['end_frame']}")
                print(f"  Latent range: {chunk_start}-{chunk_end} ({chunk_frames} frames)")
                
                info_lines.append(f"Processing chunk {chunk_idx + 1}/{len(chunk_conditionings)}:")
                info_lines.append(f"  Latent frames: {chunk_start}-{chunk_end} ({chunk_frames} frames)")
                
                # Extract chunk from latent and noise
                # Will be extended later if VACE reference is present
                chunk_latent = latent[:, :, chunk_start:chunk_end, :, :]
                chunk_noise = noise[:, :, chunk_start:chunk_end, :, :] if not disable_noise else torch.zeros_like(chunk_latent)
                
                # Store for later potential extension
                chunk_latent_for_sampling = chunk_latent
                chunk_noise_for_sampling = chunk_noise
                
                # Use pre-encoded conditioning from chunk (with fallback to global)
                chunk_positive = chunk_cond.get("positive")
                chunk_negative = chunk_cond.get("negative")
                
                # Fallback to global conditioning if chunk conditioning is empty
                if not chunk_positive or (isinstance(chunk_positive, list) and len(chunk_positive) == 0):
                    if positive is not None:
                        chunk_positive = positive
                        info_lines.append(f"  (Using global positive conditioning)")
                    else:
                        raise ValueError(f"Chunk {chunk_idx + 1} has no positive conditioning and no global fallback provided!")
                        
                if not chunk_negative or (isinstance(chunk_negative, list) and len(chunk_negative) == 0):
                    if negative is not None:
                        chunk_negative = negative
                        info_lines.append(f"  (Using global negative conditioning)")
                    else:
                        raise ValueError(f"Chunk {chunk_idx + 1} has no negative conditioning and no global fallback provided!")
                
                # Get VACE reference for this chunk
                # Chunk 0: User-provided reference from preprocessor
                # Chunk 1+: Updated from previous chunk's last frame (set at end of previous iteration)
                if chunk_idx == 0:
                    vace_reference = chunk_cond.get("vace_reference")  # User-provided initial reference
                # else: use vace_reference from previous iteration (already updated)

                has_vace_reference = vace_reference is not None  # Apply to ALL chunks (not just chunk 0)
                
                # NEW: Chunk-by-chunk VACE encoding with pixel-space overlap
                chunk_control_pixels_info = chunk_cond.get("control_pixels_info")
                if chunk_control_pixels_info and len(chunk_control_pixels_info) > 0:
                    print(f"  Encoding {len(chunk_control_pixels_info)} control(s) for this chunk...")
                    if has_vace_reference:
                        print(f"  VACE reference will be prepended as frame 0 to each control")
                    info_lines.append(f"  → Chunk has {len(chunk_control_pixels_info)} control(s)")

                    # Get VAE and tiled setting from chunk conditioning
                    chunk_vae = chunk_cond.get("vae")
                    chunk_tiled = chunk_cond.get("tiled_vae", False)

                    # Combine all control embeds for this chunk
                    # ComfyUI Wan VACE expects: vace_frames, vace_mask, vace_strength
                    all_vace_frames = []
                    all_vace_masks = []
                    all_vace_strengths = []

                    for ctrl_name, ctrl_info in chunk_control_pixels_info.items():
                        # Extract control pixels for this chunk's frame range
                        full_control_pixels = ctrl_info["pixels"]  # [T, H, W, C] full video

                        # Get pixel frames for this chunk
                        chunk_control_pixels = full_control_pixels[chunk_cond["start_frame"]:chunk_cond["end_frame"]].clone()  # [chunk_frames, H, W, C]

                        # NEW: Replace first 16 VIDEO frames with color-matched pixels from previous chunk
                        if chunk_idx > 0 and hasattr(self, '_prev_chunk_overlap_pixels'):
                            overlap_pixels = self._prev_chunk_overlap_pixels  # [16, H, W, C] already color-matched
                            overlap_frames = min(16, chunk_control_pixels.shape[0])

                            # Replace first 16 frames with overlap from previous chunk
                            chunk_control_pixels[:overlap_frames] = overlap_pixels[-overlap_frames:]

                            print(f"    • Replaced first {overlap_frames} frames with color-matched overlap from chunk {chunk_idx-1}")

                        # Encode this chunk's control pixels to VACE dual-channel format
                        inactive_latents, reactive_latents = self._encode_vace_dual_channel(
                            chunk_control_pixels, chunk_vae, chunk_tiled
                        )

                        # Package in ComfyUI Wan VACE format with batch dimension
                        inactive_batched = inactive_latents.unsqueeze(0)  # [1, 16, T_lat, H, W]
                        reactive_batched = reactive_latents.unsqueeze(0)  # [1, 16, T_lat, H, W]
                        chunk_vace_frames = torch.cat([inactive_batched, reactive_batched], dim=1)  # [1, 32, T_lat, H, W]

                        # Create 64-channel extended mask for this chunk
                        C, T_lat, H_lat, W_lat = inactive_latents.shape
                        vae_stride = 8
                        chunk_vace_mask = torch.ones(1, vae_stride * vae_stride, T_lat, H_lat, W_lat,
                                                     device=chunk_vace_frames.device, dtype=chunk_vace_frames.dtype)

                        # Set mask=0 for overlap frames (inactive stream)
                        if chunk_idx > 0 and hasattr(self, '_prev_chunk_overlap_pixels'):
                            # Calculate how many latent frames correspond to 16 video frames
                            # Approximately 16 video frames → 4 latent frames
                            overlap_latent_frames = min(4, T_lat)
                            chunk_vace_mask[:, :, :overlap_latent_frames, :, :] = 0.0
                            print(f"      Mask: first {overlap_latent_frames} latent frames = 0 (inactive), rest = 1 (reactive)")

                        # PREPEND VACE REFERENCE (if present)
                        if has_vace_reference:
                            # Prepend reference to control frames [1, 32, 1, H, W] + [1, 32, T, H, W]
                            chunk_vace_frames = torch.cat([vace_reference, chunk_vace_frames], dim=2)

                            # Create zero mask for reference frame [1, 64, 1, H, W]
                            zero_mask = torch.zeros((1, 64, 1, chunk_vace_mask.shape[3], chunk_vace_mask.shape[4]),
                                                   device=chunk_vace_mask.device, dtype=chunk_vace_mask.dtype)
                            chunk_vace_mask = torch.cat([zero_mask, chunk_vace_mask], dim=2)

                        all_vace_frames.append(chunk_vace_frames)
                        all_vace_masks.append(chunk_vace_mask)
                        all_vace_strengths.append(ctrl_info["weight"])

                        ref_suffix = " +ref" if has_vace_reference else ""
                        print(f"    • {ctrl_name}: weight={ctrl_info['weight']:.2f}, shape={chunk_vace_frames.shape}{ref_suffix}")
                        info_lines.append(f"    • '{ctrl_name}': weight={ctrl_info['weight']:.2f}, encoded for chunk{ref_suffix}")

                    # Inject into both positive AND negative conditioning
                    # ComfyUI conditioning format: [[cond_tensor, {"key": value}], ...]
                    if isinstance(chunk_positive, list) and len(chunk_positive) > 0:
                        # Clone the conditioning to avoid modifying original
                        chunk_positive = [[c[0], c[1].copy()] for c in chunk_positive]
                        chunk_negative = [[c[0], c[1].copy()] for c in chunk_negative]

                        # Add Wan VACE keys to the first conditioning entry (both positive and negative)
                        chunk_positive[0][1]["vace_frames"] = all_vace_frames
                        chunk_positive[0][1]["vace_mask"] = all_vace_masks
                        chunk_positive[0][1]["vace_strength"] = all_vace_strengths

                        chunk_negative[0][1]["vace_frames"] = all_vace_frames
                        chunk_negative[0][1]["vace_mask"] = all_vace_masks
                        chunk_negative[0][1]["vace_strength"] = all_vace_strengths

                        print(f"    ✓ Controls encoded and injected into conditioning")
                        info_lines.append(f"    ✓ VACE controls injected: {len(all_vace_frames)} control(s)")
                
                # VACE REFERENCE: Extend latent for first chunk if reference present
                # Per WAN VACE architecture: generate with reference, then trim
                trim_reference_frame = 0
                
                if has_vace_reference:
                    # Extend the chunk latent and noise by 1 frame to accommodate reference
                    # The reference frame will be frame 0, actual generation starts at frame 1
                    if chunk_end + 1 <= latent.shape[2]:
                        chunk_latent_for_sampling = latent[:, :, chunk_start:chunk_end+1, :, :]  # +1 frame
                        chunk_noise_for_sampling = noise[:, :, chunk_start:chunk_end+1, :, :] if not disable_noise else torch.zeros_like(chunk_latent_for_sampling)
                    else:
                        # At the end, just extend with zeros
                        extra_frame = torch.zeros((1, latent.shape[1], 1, latent.shape[3], latent.shape[4]), 
                                                  device=latent.device, dtype=latent.dtype)
                        chunk_latent_for_sampling = torch.cat([chunk_latent, extra_frame], dim=2)
                        if not disable_noise:
                            chunk_noise_for_sampling = torch.cat([chunk_noise, torch.randn_like(extra_frame)], dim=2)
                        else:
                            chunk_noise_for_sampling = torch.zeros_like(chunk_latent_for_sampling)
                    
                    trim_reference_frame = 1
                    print(f"  VACE reference: Extended latent by 1 frame for generation")
                    print(f"    Latent shape: {chunk_latent_for_sampling.shape}")
                    print(f"    Will trim {trim_reference_frame} frame after generation")
                    info_lines.append(f"  → VACE reference: Latent extended, will trim after gen")
                # TIER 2: Now handled through VACE controls - refined frames replace control frames with mask=0
                # The concat_mask approach is no longer needed since VACE handles the forcing
                # if chunk_idx > 0:
                #     print(f"  Tier 2: First 16 frames enforced through VACE controls (mask=0 for refined frames)")

                # FALLBACK: If no diffused overlap available, use old Tier 2 logic
                elif chunk_idx > 0 and 'prev_chunk_samples' in locals():
                    # Calculate overlap region
                    overlap_frames = min(chunk_overlap, chunk_frames)

                    if overlap_frames > 0:
                        # Extract overlap from the END of previous chunk's RAW output
                        prev_overlap = prev_chunk_samples[:, :, -overlap_frames:, :, :].clone()

                        # Build concat_latent_image: [overlap, new_noise]
                        new_noise_frames = chunk_frames - overlap_frames
                        if new_noise_frames > 0:
                            new_noise = torch.randn(
                                prev_overlap.shape[0], prev_overlap.shape[1], new_noise_frames,
                                prev_overlap.shape[3], prev_overlap.shape[4],
                                device=prev_overlap.device, dtype=prev_overlap.dtype
                            )
                            concat_latent = torch.cat([prev_overlap, new_noise], dim=2)
                        else:
                            concat_latent = prev_overlap

                        # Build concat_mask for I2V
                        concat_mask = torch.ones((1, 1, chunk_frames, chunk_latent.shape[3], chunk_latent.shape[4]),
                                                   device=chunk_latent.device, dtype=chunk_latent.dtype)
                        concat_mask[:, :, :overlap_frames, :, :] = 0.0

                        print(f"  Tier 2 (FALLBACK): Using {overlap_frames} overlap frames from previous chunk (not diffused)")

                        # Apply Tier 2
                        chunk_positive = [[c[0], c[1].copy()] for c in chunk_positive]
                        chunk_negative = [[c[0], c[1].copy()] for c in chunk_negative]

                        chunk_positive[0][1]["concat_latent_image"] = concat_latent
                        chunk_positive[0][1]["concat_mask"] = concat_mask
                        chunk_negative[0][1]["concat_latent_image"] = concat_latent
                        chunk_negative[0][1]["concat_mask"] = concat_mask
                
                # Sample this chunk
                print(f"  Starting sampling...")
                print(f"    Sampler: {sampler_name}, Scheduler: {scheduler}")
                print(f"    Steps: {steps}, CFG: {cfg}, Denoise: {denoise_strength}")
                print(f"    Seed: {chunk_seed}" + (f" (base seed {seed} + offset {chunk_idx})" if per_chunk_seed_offset else ""))
                
                # Create progress callback
                sample_start_time = time.time()
                last_step_time = sample_start_time
                
                def progress_callback(step, denoised, x, total_steps):
                    nonlocal last_step_time
                    current_time = time.time()
                    step_time = current_time - last_step_time
                    elapsed = current_time - sample_start_time
                    
                    if step % max(1, total_steps // 10) == 0 or step == total_steps - 1:
                        avg_step_time = elapsed / (step + 1)
                        eta = avg_step_time * (total_steps - step - 1)
                        print(f"    Step {step + 1}/{total_steps} ({(step + 1) / total_steps * 100:.0f}%) | "
                              f"Step: {step_time:.2f}s | ETA: {eta:.1f}s")
                    
                    last_step_time = current_time
                
                chunk_samples = comfy.sample.sample(
                    model,
                    chunk_noise_for_sampling,
                    steps,
                    cfg,
                    sampler_name,
                    scheduler if scheduler != "beta57" else "normal",
                    chunk_positive,
                    chunk_negative,
                    chunk_latent_for_sampling,
                    denoise=denoise_strength,
                    disable_noise=disable_noise,
                    start_step=start_step if start_step > 0 else None,
                    last_step=end_step if end_step >= 0 else None,
                    force_full_denoise=force_full_denoise,
                    noise_mask=None,
                    callback=progress_callback,
                    disable_pbar=False,  # Show progress bar
                    seed=chunk_seed  # Per-chunk seed if enabled, otherwise same seed for consistency
                )
                
                sample_time = time.time() - sample_start_time
                print(f"  ✓ Sampling complete: {sample_time:.2f}s ({sample_time/steps:.3f}s/step)")
                
                # TRIM REFERENCE FRAME if present
                # Per WAN VACE architecture: remove the reference frame from output
                if trim_reference_frame > 0:
                    print(f"  Trimming {trim_reference_frame} reference frame from output...")
                    print(f"    Before trim: {chunk_samples.shape}")
                    chunk_samples = chunk_samples[:, :, trim_reference_frame:, :, :]  # Remove first frame
                    print(f"    After trim: {chunk_samples.shape}")
                    info_lines.append(f"  → Reference frame trimmed from output")

                # REMOVED: Extension frame trimming (was breaking chunk sizes for crossfade blending)
                # The full sampled chunk is needed for proper crossfade at the correct frame positions
                # Extension frame management is handled during VACE conditioning prep, not output trimming

                # if chunk_idx > 0 and chunk_overlap > 0:
                #     # This was causing chunks to be smaller than expected (e.g., 65 instead of 81 frames)
                #     # which broke the crossfade blending dimension calculations
                #     pass

                # Decode chunk to pixels immediately for pixel-space processing
                if vae is not None:
                    try:
                        print(f"  Decoding chunk to pixels...")
                        chunk_pixels = vae.decode(chunk_samples)
                        
                        # Handle shape
                        if chunk_pixels.dim() == 5 and chunk_pixels.shape[0] == 1:
                            chunk_pixels = chunk_pixels.squeeze(0)

                        # NOTE: Pixel replacement no longer needed - VACE controls enforce the overlap
                        # The first 16 frames are forced identical through VACE mask=0
                        # if chunk_idx > 0 and hasattr(self, '_prev_diffused_overlap') and self._prev_diffused_overlap is not None:
                        #     overlap_frames = 16  # Fixed 16-frame overlap
                        #     print(f"  First {overlap_frames} frames enforced identical through VACE controls (mask=0)")

                        # PER-CHUNK COLOR MATCHING (before blending)
                        # Apply color matching to each chunk immediately after decode
                        # This ensures smooth color transitions when chunks are blended
                        # Apply to ALL chunks including chunk 0 for consistency
                        if hasattr(self, '_original_reference') and self._original_reference is not None:
                            try:
                                from color_matcher import ColorMatcher

                                print(f"  Applying color matching to chunk {chunk_idx}...")
                                # Use ORIGINAL reference to prevent accumulation
                                ref_np = self._original_reference.cpu().numpy().squeeze()
                                cm = ColorMatcher()

                                # Color match each frame
                                matched_frames = []
                                for frame_idx in range(len(chunk_pixels)):
                                    try:
                                        frame_np = chunk_pixels[frame_idx].cpu().numpy()
                                        # Full strength MVGD color matching as per user preference
                                        matched = cm.transfer(src=frame_np, ref=ref_np, method='mvgd')
                                        matched_frames.append(torch.from_numpy(matched))
                                    except Exception as e:
                                        # Fallback: use original frame
                                        matched_frames.append(chunk_pixels[frame_idx])

                                chunk_pixels = torch.stack(matched_frames, dim=0)
                                print(f"  ✓ Color matching complete for chunk {chunk_idx} (full strength, mvgd method, original ref)")
                            except ImportError:
                                print(f"  ⚠️  color-matcher not installed, skipping per-chunk color matching")
                            except Exception as e:
                                print(f"  ⚠️  Color matching failed for chunk {chunk_idx}: {e}, using unmatched")

                        # NEW: Store last 16 frames (color-matched) for next chunk's VACE overlap
                        if chunk_pixels is not None and len(chunk_pixels) >= 16:
                            self._prev_chunk_overlap_pixels = chunk_pixels[-16:].clone()
                            print(f"  ✓ Stored last 16 color-matched frames for next chunk's VACE overlap")
                        elif chunk_pixels is not None:
                            # If chunk has less than 16 frames, store what we have
                            self._prev_chunk_overlap_pixels = chunk_pixels.clone()
                            print(f"  ✓ Stored {len(chunk_pixels)} color-matched frames for next chunk's VACE overlap")

                        # ==============================================================================
                        # DIFFUSION REFINEMENT OF LAST N FRAMES (NEW LOCATION - END OF CHUNK PIPELINE)
                        # ==============================================================================
                        # This creates high-quality overlap frames that will be:
                        # 1. Part of current chunk's output
                        # 2. Forced as identical input for next chunk (via concat_mask)
                        # 3. Used to create VACE reference for next chunk
                        #
                        # KEY FIX: Work with LATENT frames first, accept whatever VIDEO frame count VAE produces

                        if hasattr(self, '_refine_enable') and self._refine_enable and 'chunk_samples' in locals():
                            try:
                                # Calculate target latent frames (aim for ~16 video frames, but accept VAE's decision)
                                target_latent_frames = min(4, chunk_frames - 1)  # ~4 latent = ~16 video frames

                                print(f"  🎨 Diffusion refinement: Processing last {target_latent_frames} latent frames...")

                                # Extract last N LATENT frames from chunk_samples for refinement
                                # We refine directly in latent space (no decode→re-encode cycle)
                                last_n_latent = chunk_samples[:, :, -target_latent_frames:, :, :]

                                print(f"    Refining last {target_latent_frames} latent frames (shape: {last_n_latent.shape})")

                                # Get conditioning for diffusion
                                refine_positive = chunk_positive if chunk_positive is not None else self._refine_positive
                                refine_negative = chunk_negative if chunk_negative is not None else self._refine_negative

                                if refine_positive is None or refine_negative is None:
                                    print(f"    ⚠️  No conditioning available, skipping diffusion refinement")
                                    raise ValueError("No conditioning for diffusion refinement")

                                # Deep copy conditioning to preserve all data (not just VACE)
                                # Then clear VACE controls and inject new ones for refinement
                                refine_positive = [[c[0], copy.deepcopy(c[1])] for c in refine_positive]
                                refine_negative = [[c[0], copy.deepcopy(c[1])] for c in refine_negative]

                                # Clear existing VACE controls (will inject new ones for last N frames)
                                refine_positive[0][1].pop("vace_frames", None)
                                refine_positive[0][1].pop("vace_mask", None)
                                refine_positive[0][1].pop("vace_strength", None)
                                refine_negative[0][1].pop("vace_frames", None)
                                refine_negative[0][1].pop("vace_mask", None)
                                refine_negative[0][1].pop("vace_strength", None)

                                # Inject VACE controls for these last N frames
                                # Extract from already-processed chunk controls (not raw full controls)
                                if hasattr(chunk_positive[0][1], '__getitem__') and "vace_frames" in chunk_positive[0][1]:
                                    # Get the VACE controls that were used for this chunk's sampling
                                    chunk_vace_frames_list = chunk_positive[0][1]["vace_frames"]  # List of controls
                                    chunk_vace_masks_list = chunk_positive[0][1]["vace_mask"]
                                    chunk_vace_strengths = chunk_positive[0][1]["vace_strength"]

                                    # Extract last N latent frames from each control
                                    refine_vace_frames = []
                                    refine_vace_masks = []

                                    for vace_frames, vace_mask in zip(chunk_vace_frames_list, chunk_vace_masks_list):
                                        # vace_frames shape: [1, 32, T_chunk, H, W] where T_chunk includes reference if present
                                        # We want the last N latent frames that correspond to our refined latent

                                        # Skip reference frame if it was prepended (first frame has mask=0)
                                        if has_vace_reference:
                                            # Remove reference frame (first frame)
                                            vace_frames_no_ref = vace_frames[:, :, 1:, :, :]
                                            vace_mask_no_ref = vace_mask[:, :, 1:, :, :]
                                        else:
                                            vace_frames_no_ref = vace_frames
                                            vace_mask_no_ref = vace_mask

                                        # Now extract last N latent frames
                                        overlap_vace_frames = vace_frames_no_ref[:, :, -target_latent_frames:, :, :]
                                        overlap_vace_mask = vace_mask_no_ref[:, :, -target_latent_frames:, :, :]

                                        refine_vace_frames.append(overlap_vace_frames)
                                        refine_vace_masks.append(overlap_vace_mask)

                                    all_vace_frames = refine_vace_frames
                                    all_vace_masks = refine_vace_masks
                                    all_vace_strengths = chunk_vace_strengths

                                    print(f"    ✓ Extracted VACE controls for last {target_latent_frames} latent frames (shape: {all_vace_frames[0].shape})")
                                else:
                                    # With chunk-by-chunk VACE encoding, controls are already in conditioning
                                    # This fallback is no longer needed
                                    print(f"    ⚠️  No VACE controls found for refinement")
                                    all_vace_frames = []
                                    all_vace_masks = []
                                    all_vace_strengths = []

                                    # Inject into refinement conditioning
                                    refine_positive[0][1]["vace_frames"] = all_vace_frames
                                    refine_positive[0][1]["vace_mask"] = all_vace_masks
                                    refine_positive[0][1]["vace_strength"] = all_vace_strengths
                                    refine_negative[0][1]["vace_frames"] = all_vace_frames
                                    refine_negative[0][1]["vace_mask"] = all_vace_masks
                                    refine_negative[0][1]["vace_strength"] = all_vace_strengths

                                    print(f"    ✓ Injected {len(all_vace_frames)} VACE control(s) for refinement")

                                # Run diffusion refinement on the original latent frames
                                # VACE controls already match (both {target_latent_frames} latent frames)
                                from nodes import common_ksampler
                                print(f"    Running {self._refine_steps} steps @ {self._refine_denoise} denoise...")
                                refined_output = common_ksampler(
                                    model=self._refine_model,
                                    seed=self._refine_seed,  # Consistent seed
                                    steps=self._refine_steps,
                                    cfg=self._refine_cfg,
                                    sampler_name=self._refine_sampler_name,
                                    scheduler=self._refine_scheduler,
                                    positive=refine_positive,
                                    negative=refine_negative,
                                    latent={"samples": last_n_latent},
                                    denoise=self._refine_denoise,
                                    force_full_denoise=True
                                )

                                # Decode refined latent to pixels
                                refined_latent = refined_output[0]["samples"]
                                refined_pixels = vae.decode(refined_latent.to(torch.float32))

                                # Handle shape
                                if refined_pixels.dim() == 5 and refined_pixels.shape[0] == 1:
                                    refined_pixels = refined_pixels.squeeze(0)

                                # Get actual frame count from decoded refined latent
                                actual_overlap_frames = len(refined_pixels)
                                print(f"  ✓ Diffusion refinement complete: refined {target_latent_frames} latent → {actual_overlap_frames} video frames")

                                # Apply color matching to refined frames using ORIGINAL reference
                                # This ensures refined frames match the original color/contrast
                                if hasattr(self, '_original_reference') and self._original_reference is not None:
                                    try:
                                        from color_matcher import ColorMatcher
                                        print(f"    Applying color match to refined frames...")
                                        ref_np = self._original_reference.cpu().numpy().squeeze()
                                        cm = ColorMatcher()

                                        matched_refined = []
                                        for frame_idx in range(len(refined_pixels)):
                                            try:
                                                frame_np = refined_pixels[frame_idx].cpu().numpy()
                                                # Full strength MVGD matching to original reference
                                                matched = cm.transfer(src=frame_np, ref=ref_np, method='mvgd')
                                                matched_refined.append(torch.from_numpy(matched))
                                            except:
                                                matched_refined.append(refined_pixels[frame_idx])

                                        refined_pixels = torch.stack(matched_refined, dim=0)
                                        print(f"    ✓ Color match complete for refined frames (full strength, mvgd, original ref)")
                                    except ImportError:
                                        print(f"    ⚠️  color-matcher not installed, skipping refined frame color matching")
                                    except Exception as e:
                                        print(f"    ⚠️  Color matching failed for refined frames: {e}")


                                # DO NOT replace frames in chunk - refined frames are ONLY for VACE conditioning
                                # The original 81 frames from chunk 1 are the actual output
                                # chunk_pixels[-actual_overlap_frames:] = refined_pixels  # REMOVED - don't modify output

                                # Store refined overlap for next chunk
                                # Store LATENT (not pixels) to avoid re-encoding dimension loss
                                if not hasattr(self, '_prev_diffused_overlap_latent'):
                                    self._prev_diffused_overlap_latent = None
                                if not hasattr(self, '_prev_diffused_overlap'):
                                    self._prev_diffused_overlap = None
                                self._prev_diffused_overlap_latent = refined_latent.clone()  # Store refined latent (4 frames)
                                self._prev_diffused_overlap = refined_pixels.clone()  # Store pixels for reference
                                self._actual_overlap_frames = actual_overlap_frames  # Use ACTUAL VAE output (typically 13)

                                # ENCODE refined frames as VACE controls for next chunk
                                if chunk_idx < len(chunk_conditionings) - 1:  # Not the last chunk
                                    print(f"  Encoding refined {actual_overlap_frames} frames as VACE controls for next chunk...")
                                    # Use all refined frames (typically 13 from VAE decode)
                                    refined_overlap_frames = refined_pixels  # Use all frames from VAE decode

                                    # Encode as VACE dual-channel format (16ch VAE + 16ch zeros)
                                    # refined_overlap_frames is [actual_overlap_frames, H, W, C]
                                    refined_overlap_frames_batch = refined_overlap_frames.unsqueeze(0)  # [1, actual_overlap_frames, H, W, C]

                                    # Encode using VAE (get 16ch latent)
                                    vae_encoded = vae.encode(refined_overlap_frames_batch.to(torch.float32))  # [1, 16, ~4, H, W]

                                    # Add zeros padding for VACE dual-channel format
                                    zeros_padding = torch.zeros_like(vae_encoded)
                                    refined_vace_controls = torch.cat([vae_encoded, zeros_padding], dim=1)  # [1, 32, ~4, H, W]

                                    # Store for next chunk
                                    if not hasattr(self, '_prev_refined_vace_controls'):
                                        self._prev_refined_vace_controls = None
                                    self._prev_refined_vace_controls = refined_vace_controls
                                    print(f"    ✓ Encoded refined frames as VACE controls: {refined_vace_controls.shape}")

                                    # CRITICAL: Decode VACE controls back to get ACTUAL forced overlap frame count
                                    # This tells us how many frames will be truly identical in the next chunk
                                    try:
                                        # Decode just the VAE portion (first 16 channels)
                                        vace_decoded = vae.decode(vae_encoded.to(torch.float32))
                                        if vace_decoded.dim() == 5 and vace_decoded.shape[0] == 1:
                                            vace_decoded = vace_decoded.squeeze(0)

                                        # Store the ACTUAL forced overlap count (typically ~8 frames from 2 latent)
                                        self._actual_vace_overlap_frames = len(vace_decoded)
                                        print(f"    ✓ VACE-forced overlap: {self._actual_vace_overlap_frames} frames (re-encoded from {actual_overlap_frames} refined frames)")
                                        print(f"      → Next chunk's first {self._actual_vace_overlap_frames} frames will be IDENTICAL to this chunk's last {self._actual_vace_overlap_frames} frames")
                                    except Exception as e:
                                        print(f"    ⚠️  Failed to decode VACE controls for overlap count: {e}")
                                        # Fallback to refinement frame count (may cause slight mismatch)
                                        self._actual_vace_overlap_frames = actual_overlap_frames

                                # UPDATE VACE REFERENCE for next chunk (last frame of refined output)
                                if chunk_idx < len(chunk_conditionings) - 1:  # Not the last chunk
                                    # Extract last LATENT frame directly (no VAE encode needed!)
                                    # This avoids the single-frame VAE encode bug
                                    last_frame_latent = refined_latent[:, :, -1:, :, :]  # [1, 16, 1, H, W]

                                    # Add zeros padding to make VACE reference (32 channels)
                                    zeros_padding = torch.zeros_like(last_frame_latent)
                                    vace_reference = torch.cat([last_frame_latent, zeros_padding], dim=1)  # [1, 32, 1, H, W]

                                    # DON'T update color matching reference - use original to prevent accumulation
                                    # self._temp_start_image = refined_pixels[-1:].clone()  # REMOVED - causes progressive enhancement

                                    print(f"  ✓ Updated VACE reference for chunk {chunk_idx + 1} (color ref stays original)")
                                    print(f"    (actual overlap: {actual_overlap_frames} frames for next chunk)")

                            except Exception as e:
                                print(f"    ⚠️  Diffusion refinement failed: {e}")
                                import traceback
                                traceback.print_exc()

                        # ==============================================================================
                        # END DIFFUSION REFINEMENT
                        # ==============================================================================

                        # Store for pixel-space blending
                        if not hasattr(self, '_chunk_pixels_list'):
                            self._chunk_pixels_list = []

                        self._chunk_pixels_list.append({
                            'pixels': chunk_pixels.cpu(),  # Move to CPU to save VRAM
                            'chunk_idx': chunk_idx,
                            'latent_start': chunk_start,
                            'latent_end': chunk_end,
                            'start_frame': chunk_cond.get("start_frame", 0),  # Video frame index for blending
                            'end_frame': chunk_cond.get("end_frame", chunk_pixels.shape[0]),
                        })
                        
                        print(f"  ✓ Chunk decoded and stored: {chunk_pixels.shape}")
                    except Exception as e:
                        print(f"  ✗ Failed to decode chunk: {e}")
                
                # SKIP LATENT-SPACE BLENDING
                # Since extension frames are trimmed from chunks 2+, latent-space blending
                # would cause dimension mismatches. We use pixel-space concatenation instead.
                print(f"  Skipping latent-space blending (using pixel-space concatenation)")

                # Note: output_latent and weight_sum are not used when doing pixel-space concatenation
                # They remain as zeros and will be replaced by the pixel-space result at the end
                
                # Store this chunk's RAW output for next chunk's Tier 2 (temporal continuity)
                # Store the actual chunk samples, not the accumulated blend
                prev_chunk_samples = chunk_samples
                prev_chunk_start = chunk_start
                prev_chunk_end = chunk_end
                
                # TIER 3: VACE Frame Extension (OLD LOCATION - NOW DISABLED)
                # MOVED: Diffusion refinement now happens at END of chunk pipeline (see line ~3305)
                # This ensures diffused frames are part of chunk output AND used for next chunk's concat_mask
                prev_chunk_vace_control = None
                if False and chunk_idx < len(chunk_conditionings) - 1:  # DISABLED - replaced by new diffusion location
                    try:
                        # Calculate overlap for NEXT chunk
                        # Extract from the END of the current chunk's RAW output
                        overlap_frames_to_encode = min(chunk_overlap, chunk_frames)
                        
                        if overlap_frames_to_encode > 0 and hasattr(self, '_temp_vae') and self._temp_vae is not None:
                            print(f"  Tier 3: Preparing VACE extension for next chunk...")
                            print(f"    Decoding {overlap_frames_to_encode} overlap frames...")
                            
                            # Extract overlap latent from END of RAW chunk output [1, 16, T, H, W]
                            overlap_latent = prev_chunk_samples[:, :, -overlap_frames_to_encode:, :, :]
                            
                            # Decode to pixels with FP32 for higher quality (reduces quantization noise)
                            original_dtype = overlap_latent.dtype
                            overlap_latent_fp32 = overlap_latent.to(torch.float32)
                            overlap_pixels = self._temp_vae.decode(overlap_latent_fp32)
                            overlap_pixels = overlap_pixels.to(original_dtype)  # Convert back for consistency
                            
                            # Handle shape: Wan VAE returns [B, T, H, W, C]
                            if overlap_pixels.dim() == 5 and overlap_pixels.shape[0] == 1:
                                overlap_pixels = overlap_pixels.squeeze(0)  # [T, H, W, C]
                            elif overlap_pixels.dim() == 4:
                                pass  # Already [T, H, W, C]
                            else:
                                raise ValueError(f"Unexpected decoded pixel shape: {overlap_pixels.shape}")
                            
                            # Just use the decoded pixels directly - no test cycle!
                            # The issue was: decode (1 cycle) + test cycle (2 cycles) + VACE encode (3 cycles) = too much degradation
                            # Now: decode (1 cycle) + VACE encode (2 cycles) = matches manual workflow
                            print(f"    Using decoded pixels directly (no test cycle to avoid extra degradation)")
                            print(f"    Pixel stats: mean={overlap_pixels.mean():.4f}, std={overlap_pixels.std():.4f}")

                            # Apply bilateral filter to reduce VAE compression noise
                            try:
                                import cv2
                                import numpy as np

                                # Convert to numpy for cv2 processing
                                overlap_np = overlap_pixels.cpu().numpy()

                                # Apply bilateral filter per frame (edge-preserving denoising)
                                filtered_frames = []
                                for frame in overlap_np:
                                    # Convert to uint8 for cv2.bilateralFilter
                                    frame_uint8 = (np.clip(frame, 0, 1) * 255).astype(np.uint8)

                                    # Parameters: d=5 (neighbor diameter), sigmaColor=5, sigmaSpace=1.5
                                    # Gentle denoising that preserves edges
                                    filtered = cv2.bilateralFilter(
                                        frame_uint8,
                                        d=5,
                                        sigmaColor=5,  # 5/255 ≈ 0.02 in [0,1] range
                                        sigmaSpace=1.5
                                    )

                                    # Convert back to float32 [0,1]
                                    filtered_frames.append(filtered.astype(np.float32) / 255.0)

                                overlap_pixels = torch.from_numpy(np.stack(filtered_frames)).to(overlap_pixels.device, original_dtype)
                                print(f"    ℹ️  Bilateral filter applied to {len(filtered_frames)} extension frames")
                            except ImportError:
                                print(f"    ⚠️  OpenCV not available, skipping bilateral filter (install with: pip install opencv-python)")
                            except Exception as e:
                                print(f"    ⚠️  Bilateral filter failed: {e}, continuing without filtering")

                            # OPTIONAL: Diffusion-based refinement to restore quality lost from VAE degradation
                            if hasattr(self, '_refine_enable') and self._refine_enable:
                                try:
                                    # Get conditioning for this chunk (needed for diffusion refinement)
                                    # In chunked mode, global positive/negative are None, so use chunk-specific conditioning
                                    chunk_positive = chunk_conditionings[chunk_idx].get("positive") if chunk_conditionings else None
                                    chunk_negative = chunk_conditionings[chunk_idx].get("negative") if chunk_conditionings else None

                                    # Fallback to global conditioning if chunk-specific not available
                                    refine_positive = chunk_positive if chunk_positive is not None else self._refine_positive
                                    refine_negative = chunk_negative if chunk_negative is not None else self._refine_negative

                                    # Validate we have conditioning
                                    if refine_positive is None or refine_negative is None:
                                        print(f"    ⚠️  Skipping diffusion refinement: no conditioning available for chunk {chunk_idx}")
                                        raise ValueError("No conditioning available for diffusion refinement")

                                    # INJECT VACE CONTROLS FOR EXTENSION FRAMES
                                    # With chunk-by-chunk VACE encoding, controls are already properly injected
                                    # The overlap pixels are already being used in the next chunk's VACE encoding
                                    # So this injection is no longer needed
                                    print(f"    ℹ️  Using chunk-by-chunk VACE encoding (overlap controls already injected)")

                                    print(f"    🎨 Applying diffusion refinement ({self._refine_steps} steps @ {self._refine_denoise} denoise)...")

                                    # Import common_ksampler from nodes.py
                                    from nodes import common_ksampler

                                    # Encode filtered pixels to latent (need batch dimension)
                                    # overlap_pixels shape: [T, H, W, C]
                                    overlap_pixels_batched = overlap_pixels.unsqueeze(0)  # [1, T, H, W, C]
                                    overlap_latent_for_refine = self._temp_vae.encode(overlap_pixels_batched.to(torch.float32))

                                    # Run low-denoise diffusion pass (img2img-style quality restoration)
                                    # Use same seed for all chunks to ensure consistent refinement style
                                    refined_output = common_ksampler(
                                        model=self._refine_model,
                                        seed=self._refine_seed,  # Consistent seed across all chunks for reproducibility
                                        steps=self._refine_steps,
                                        cfg=self._refine_cfg,
                                        sampler_name=self._refine_sampler_name,
                                        scheduler=self._refine_scheduler,
                                        positive=refine_positive,
                                        negative=refine_negative,
                                        latent={"samples": overlap_latent_for_refine},
                                        denoise=self._refine_denoise,
                                        force_full_denoise=True
                                    )

                                    # Decode refined latent back to pixels
                                    # common_ksampler returns (latent_dict,) tuple, so unpack it first
                                    refined_latent = refined_output[0]["samples"]
                                    overlap_pixels_refined = self._temp_vae.decode(refined_latent.to(torch.float32))

                                    # Handle shape: Wan VAE returns [B, T, H, W, C]
                                    if overlap_pixels_refined.dim() == 5 and overlap_pixels_refined.shape[0] == 1:
                                        overlap_pixels = overlap_pixels_refined.squeeze(0)  # [T, H, W, C]
                                    elif overlap_pixels_refined.dim() == 4:
                                        overlap_pixels = overlap_pixels_refined  # Already [T, H, W, C]
                                    else:
                                        raise ValueError(f"Unexpected refined pixel shape: {overlap_pixels_refined.shape}")

                                    overlap_pixels = overlap_pixels.to(original_dtype)  # Convert back to original dtype
                                    print(f"    ✓ Diffusion refinement complete: mean={overlap_pixels.mean():.4f}, std={overlap_pixels.std():.4f}")

                                    # Update color matching reference to refined last frame
                                    # This provides temporal color continuity from chunk to chunk
                                    # DON'T update color matching reference - causes progressive enhancement
                                    # refined_last_frame = overlap_pixels[-1:].clone()  # [1, H, W, C]
                                    # self._temp_start_image = refined_last_frame.squeeze(0).unsqueeze(0)  # REMOVED
                                    # Keep using original reference for all chunks
                                    print(f"    ✓ Keeping original color reference (prevents progressive enhancement)")

                                except Exception as e:
                                    print(f"    ⚠️  Diffusion refinement failed: {e}, using filtered pixels without refinement")
                                    import traceback
                                    traceback.print_exc()

                            # Encode with dual-channel VACE format
                            inactive_latents, reactive_latents = self._encode_vace_dual_channel_standalone(
                                overlap_pixels, self._temp_vae, False
                            )
                            control_video_latent = torch.cat([inactive_latents.unsqueeze(0), reactive_latents.unsqueeze(0)], dim=1)
                            
                            # Create 64-channel mask: 0 = no control (use as reference/extension)
                            vae_stride = 8
                            T_vace = control_video_latent.shape[2]
                            H_vace = control_video_latent.shape[3]
                            W_vace = control_video_latent.shape[4]
                            mask_64ch = torch.zeros((1, vae_stride * vae_stride, T_vace, H_vace, W_vace), 
                                                    device=control_video_latent.device, dtype=control_video_latent.dtype)
                            
                            prev_chunk_vace_control = {
                                "vace_frames": control_video_latent,  # [1, 32, overlap_frames, H, W]
                                "vace_mask": mask_64ch,               # [1, 64, overlap_frames, H, W]
                            }
                            
                            print(f"    ✓ VACE extension prepared: {control_video_latent.shape}")
                            print(f"    ℹ️  Tier 3 only applies to first transition to avoid cumulative degradation")
                            info_lines.append(f"  → Tier 3: VACE frame extension (first transition only)")
                    except Exception as e:
                        print(f"    ✗ Failed to prepare VACE extension: {e}")
                        import traceback
                        traceback.print_exc()
                
                chunk_total_time = time.time() - chunk_start_time
                print(f"  ✓ Chunk {chunk_idx + 1} complete: {chunk_total_time:.2f}s total")
                print("")
                
                info_lines.append(f"  ✓ Chunk {chunk_idx + 1} complete")
            
            # Normalize by weight sum
            print(f"Finalizing blend normalization...")
            samples = output_latent / (weight_sum + 1e-8)
            
            # PIXEL-SPACE COLOR MATCHING & OVERLAP BLENDING
            # Use stored chunk pixels, color match, blend overlaps, re-encode
            if vae is not None and hasattr(self, '_chunk_pixels_list') and hasattr(self, '_temp_start_image') and self._temp_start_image is not None:
                print(f"\n{'='*60}")
                print(f"PIXEL-SPACE OVERLAP BLENDING & COLOR MATCHING")
                print(f"{'='*60}")
                
                try:
                    chunk_pixels_list = self._chunk_pixels_list
                    print(f"Processing {len(chunk_pixels_list)} decoded chunks...")
                    
                    # Build complete video dimensions
                    first_chunk = chunk_pixels_list[0]
                    H, W, C = first_chunk['pixels'].shape[1], first_chunk['pixels'].shape[2], first_chunk['pixels'].shape[3]

                    # Calculate total frames based on ACTUAL chunk sizes and VACE overlap
                    # Use VACE overlap (typically 5 frames) NOT refinement overlap (13 frames)
                    actual_overlap = 16  # Fixed 16-frame overlap with chunk-by-chunk VACE encoding
                    num_chunks = len(chunk_pixels_list)

                    # Calculate actual total from real chunk sizes
                    total_video_frames = 0
                    for idx, chunk_data in enumerate(chunk_pixels_list):
                        chunk_frames = len(chunk_data['pixels'])
                        if idx == 0:
                            total_video_frames += chunk_frames  # First chunk: use all frames
                        else:
                            total_video_frames += chunk_frames - actual_overlap  # Subsequent: subtract overlap

                    print(f"  Total video frames: {total_video_frames} (calculated from actual chunk sizes)")
                    print(f"  Video dimensions: {H}x{W}x{C}")
                    print(f"  VACE overlap: {actual_overlap} frames (frames forced identical between chunks)")

                    # Initialize final concatenated video (in PIXEL space, not latent space!)
                    final_video = torch.zeros((total_video_frames, H, W, C), dtype=first_chunk['pixels'].dtype)

                    # CROSSFADE BLENDING (matching VACE-forced identical region)
                    # Blends the actual VACE-forced identical frames (latent space round-trip)
                    # Using linear interpolation for smooth, uniform transitions
                    # NEW: Use fixed 16-frame overlap for crossfade (matching our chunk-by-chunk VACE overlap)
                    # With chunk-by-chunk encoding, we control the exact overlap
                    overlap_frames = 16  # Fixed 16-frame crossfade as requested by user

                    try:
                        from color_matcher import ColorMatcher

                        print(f"\n  Blending chunks with {overlap_frames}-frame linear crossfade...")

                        output_position = 0  # Track position in final output

                        for idx, chunk_data in enumerate(chunk_pixels_list):
                            chunk_idx = chunk_data['chunk_idx']
                            chunk_pixels = chunk_data['pixels'].to(final_video.device)  # Move to same device
                            num_frames = len(chunk_pixels)

                            if chunk_idx == 0:
                                # First chunk: Take ALL frames, but save last overlap frames for blending
                                frames_to_use = min(81, num_frames)  # Safety check
                                final_video[output_position:output_position + frames_to_use] = chunk_pixels[:frames_to_use]
                                output_position += frames_to_use

                                # Store last overlap frames for blending with next chunk
                                if idx < len(chunk_pixels_list) - 1:  # Not the last chunk
                                    prev_overlap_frames = chunk_pixels[-overlap_frames:].clone()

                                print(f"  Chunk {chunk_idx}: placed ALL {frames_to_use} frames → position {output_position}")
                            else:
                                # Subsequent chunks: Crossfade overlap region
                                # These frames have identical content (forced by VACE mask=0)
                                # But may have slight rendering variations that crossfade smooths

                                # First, go back to where we need to blend (overwrite last overlap frames)
                                blend_start_position = output_position - overlap_frames

                                # Get current chunk's first overlap frames (VACE-forced identical content)
                                current_overlap_frames = chunk_pixels[:overlap_frames]

                                # Perform linear crossfade blending
                                print(f"  Chunk {chunk_idx}: blending {overlap_frames} frames at position {blend_start_position} (linear)...")
                                for i in range(overlap_frames):
                                    # Calculate linear t value (no easing)
                                    t = i / (overlap_frames - 1) if overlap_frames > 1 else 0.5

                                    # Calculate weights using linear interpolation
                                    weight_prev = 1.0 - t
                                    weight_next = t

                                    # Blend the frames
                                    blended_frame = (prev_overlap_frames[i] * weight_prev +
                                                   current_overlap_frames[i] * weight_next)

                                    # Place blended frame in output
                                    final_video[blend_start_position + i] = blended_frame

                                # Now add the NEW frames (frames 17-81)
                                new_frames_start = overlap_frames
                                new_frames_end = num_frames
                                new_frames_count = new_frames_end - new_frames_start

                                if new_frames_count > 0:
                                    final_video[output_position:output_position + new_frames_count] = \
                                        chunk_pixels[new_frames_start:new_frames_end]
                                    output_position += new_frames_count

                                    # Store last 16 frames for next chunk (if not last chunk)
                                    if idx < len(chunk_pixels_list) - 1:
                                        prev_overlap_frames = chunk_pixels[-overlap_frames:].clone()

                                    print(f"  Chunk {chunk_idx}: blended {overlap_frames} frames, placed {new_frames_count} NEW frames → position {output_position}")
                                else:
                                    print(f"  Chunk {chunk_idx}: ⚠️ No new frames after overlap!")

                        print(f"  ✓ All {len(chunk_pixels_list)} chunks blended with crossfade ({output_position} total frames output)")

                        # Verify frame count matches expected
                        if output_position != total_video_frames:
                            print(f"  ⚠️ WARNING: Output frames ({output_position}) != expected ({total_video_frames})")
                            # Trim or pad as needed
                            if output_position < total_video_frames:
                                final_video = final_video[:output_position]  # Trim excess zeros
                                total_video_frames = output_position
                            # If output_position > total_video_frames, some frames were overwritten (should not happen)

                        # REMOVED: Global color matching pass
                        # Color matching is now applied per-chunk before diffusion refinement
                        # This prevents brightness dips at chunk transitions
                        print(f"\n  Skipping global color matching (already applied per-chunk)")

                        info_lines.append(f"  → Chunk concatenation with {actual_overlap}-frame VACE overlap")
                        info_lines.append(f"  → {overlap_frames}-frame linear crossfade blending (VACE-forced identical region)")
                        info_lines.append("  → Per-chunk color matching (full strength, mvgd method, original reference)")
                        
                    except ImportError:
                        print(f"  ⚠️  color-matcher not installed")
                        print(f"      Install with: pip install color-matcher")
                        # Fallback: just use first chunk's pixels
                        final_video = chunk_pixels_list[0]['pixels']
                    
                    final_video = torch.clamp(final_video, 0.0, 1.0)
                    
                    # Re-encode to latents
                    print(f"\n  Encoding blended pixels back to latents...")
                    samples = vae.encode(final_video.to(vae.device)[:,:,:,:3])
                    print(f"  ✓ Re-encoded shape: {samples.shape}")
                    
                    # Clean up stored pixels
                    delattr(self, '_chunk_pixels_list')
                    
                except Exception as e:
                    print(f"  ✗ Pixel-space processing failed: {e}")
                    import traceback
                    traceback.print_exc()
                    print(f"  Continuing with latent-space result...")
            
            total_time = time.time() - start_time
            print(f"\n{'='*60}")
            print(f"✓ ALL CHUNKS COMPLETE!")
            print(f"  Total time: {total_time:.2f}s")
            print(f"  Avg per chunk: {total_time/len(chunk_conditionings):.2f}s")
            print(f"{'='*60}\n")
            
            info_lines.append("")
            info_lines.append("Context window sampling complete!")
            info_lines.append("=" * 60)
        
        # Call ComfyUI's sample function (standard mode)
        else:
            # Double-check we have conditioning for standard mode
            if positive is None or negative is None:
                raise ValueError(
                    "Standard (non-chunked) mode requires both positive and negative conditioning. "
                    "Connect CLIP Text Encode outputs to positive and negative inputs."
                )
            
            try:
                # For SDE sampling, we need to inject noise after each step
                # This requires custom sampling loop
                if eta > 0 and sampler_mode == "standard":
                    info_lines.append(f"  Using SDE sampling (eta={eta:.2f})")
                
                # We'll use ComfyUI's sampler but post-process with SDE noise
                # Note: This is a simplified approach
                # Full SDE would require hooking into the sampling loop
                samples = comfy.sample.sample(
                    model,
                    noise,
                    steps,
                    cfg,
                    sampler_name,
                    scheduler if scheduler != "beta57" else "normal",
                    positive,
                    negative,
                    latent,
                    denoise=denoise_strength,
                    disable_noise=disable_noise,
                    start_step=start_step if start_step > 0 else None,
                    last_step=end_step if end_step >= 0 else None,
                    force_full_denoise=force_full_denoise,
                    noise_mask=noise_mask,
                    disable_pbar=disable_pbar,
                    seed=seed
                )
                
                # Apply SDE noise as post-processing
                # (Note: True SDE requires per-step injection, this is an approximation)
                if eta > 0.05:  # Only apply if meaningful
                    torch.manual_seed(noise_seed)
                    
                    # Try RES4LYF noise generators first (best quality)
                    if self.RES4LYF_NOISE_AVAILABLE and noise_type in ["brownian", "pyramid", "hires-pyramid"]:
                        try:
                            sde_noise = generate_noise_res4lyf(
                                noise_type, samples, noise_seed,
                                sigma_current=sigmas[0], sigma_next=sigmas[-1]
                            )
                            info_lines.append(f"  Applied {noise_type} SDE noise (RES4LYF quality)")
                        except Exception as e:
                            info_lines.append(f"  Warning: RES4LYF noise failed, using fallback: {e}")
                            # Fallback to simple implementations
                            if noise_type in ["brownian", "hires-pyramid"]:
                                sde_noise = generate_brownian_noise_fallback(samples.shape, samples.device, samples.dtype)
                            else:
                                sde_noise = generate_pyramid_noise_fallback(samples.shape, samples.device, samples.dtype)
                            info_lines.append(f"  Applied {noise_type} SDE noise (fallback)")
                    else:
                        # Use fallback implementations
                        if noise_type == "gaussian":
                            sde_noise = torch.randn_like(samples)
                        elif noise_type == "brownian":
                            sde_noise = generate_brownian_noise_fallback(samples.shape, samples.device, samples.dtype)
                        elif noise_type == "pyramid":
                            sde_noise = generate_pyramid_noise_fallback(samples.shape, samples.device, samples.dtype)
                        elif noise_type == "pink":
                            sde_noise = generate_pink_noise(samples.shape, samples.device, samples.dtype)
                        elif noise_type == "blue":
                            sde_noise = generate_blue_noise(samples.shape, samples.device, samples.dtype)
                        
                        quality_note = " (RES4LYF)" if self.RES4LYF_NOISE_AVAILABLE else " (fallback)"
                        info_lines.append(f"  Applied {noise_type} SDE noise{quality_note}")
                    
                    # Apply noise with scaling
                    noise_scale = eta * 0.1  # Scale factor
                    samples = samples + sde_noise * noise_scale
                
                sampling_time = time.time() - start_time
                
                info_lines.append(f"  ✓ Sampling completed successfully")
                info_lines.append(f"  Time: {sampling_time:.2f}s ({sampling_time/(len(sigmas)-1):.3f}s/step)")
                
            except Exception as e:
                info_lines.append(f"  ✗ Sampling failed: {str(e)}")
                import traceback
                info_lines.append(f"  Traceback: {traceback.format_exc()}")
                samples = latent  # Return input on failure
        
        info_lines.append("")
        
        # ============================================================
        # STEP 6: Prepare output
        # ============================================================
        # Decide output format based on channels
        # Video models (Wan, etc.) have 16+ channels and need 5D format
        # Image models (SD, SDXL) have 4 channels and can use 4D
        output_channels = samples.shape[1]
        
        if is_image_latent and output_channels <= 4:
            # Standard image model: squeeze back to 4D
            samples = samples.squeeze(2)  # [B, C, 1, H, W] -> [B, C, H, W]
            info_lines.append("  Output format: 4D (image latent)")
        else:
            # Video model or multi-channel: keep as 5D
            info_lines.append("  Output format: 5D (video latent or high-channel model)")
        
        out = latent_image.copy()
        out["samples"] = samples
        
        # Final statistics
        info_lines.append("OUTPUT:")
        info_lines.append(f"  Shape: {list(samples.shape)} ({'4D IMAGE' if is_image_latent else '5D VIDEO'})")
        info_lines.append(f"  Mean: {samples.mean().item():.6f}")
        info_lines.append(f"  Std: {samples.std().item():.6f}")
        info_lines.append(f"  Min: {samples.min().item():.6f}")
        info_lines.append(f"  Max: {samples.max().item():.6f}")
        info_lines.append("=" * 60)
        
        info_text = "\n".join(info_lines)
        print(f"\n{info_text}")
        
        return (out, info_text)
    
    def _encode_vace_dual_channel_standalone(self, frames, vae, tiled=False):
        """
        Standalone method to encode frames into dual-channel VACE format.
        Used by Tier 3 to encode overlap regions during sampling.
        
        Args:
            frames: [T, H, W, C] tensor in [0, 1] range
            vae: VAE object for encoding
            tiled: Whether to use tiled VAE (unused for Wan VAE)
        
        Returns:
            inactive_latents: [16, T, H, W] tensor
            reactive_latents: [16, T, H, W] tensor
        """
        import torch
        import comfy.utils
        
        device = vae.device if hasattr(vae, 'device') else torch.device('cpu')
        # Force FP32 for Tier 3 VACE encoding to reduce quantization noise
        dtype = torch.float32  # Always use FP32 for extension frames

        # Ensure frames are on the right device and dtype
        frames = frames.to(device=device, dtype=dtype)
        
        # Frames should be [T, H, W, C] in [0, 1] range
        # Center to [-0.5, 0.5] for splitting
        centered_frames = frames - 0.5
        
        # Inactive stream: set all to 0.5 (neutral gray in [0, 1])
        inactive_frames = torch.full_like(frames, 0.5)
        
        # Reactive stream: use the actual frames [0, 1]
        reactive_frames = frames.clone()
        
        # Encode both streams
        def encode_stream(stream_frames):
            # Ensure divisible by 16
            T, H, W, C = stream_frames.shape
            H_new = ((H + 15) // 16) * 16
            W_new = ((W + 15) // 16) * 16
            
            if H != H_new or W != W_new:
                # Resize using comfy.utils.common_upscale
                # Input: [B, C, H, W] or [B, C, T, H, W]
                # We have [T, H, W, C], need to permute to [T, C, H, W]
                stream_frames_perm = stream_frames.permute(0, 3, 1, 2)  # [T, C, H, W]
                stream_frames_perm = stream_frames_perm.unsqueeze(0)  # [1, T, C, H, W]
                # Reshape to [1, T*C, H, W] for upscale
                B, T, C, H, W = stream_frames_perm.shape
                stream_frames_flat = stream_frames_perm.reshape(B, T*C, H, W)
                stream_frames_resized = comfy.utils.common_upscale(
                    stream_frames_flat, W_new, H_new, "bilinear", "disabled"
                )
                # Reshape back to [1, T, C, H, W]
                stream_frames_resized = stream_frames_resized.reshape(B, T, C, H_new, W_new)
                # Remove batch and permute back to [T, H, W, C]
                stream_frames = stream_frames_resized.squeeze(0).permute(0, 2, 3, 1)
            
            # Encode using VAE
            # VAE expects [T, H, W, C] in [0, 1] range for ComfyUI wrapper
            latents = vae.encode(stream_frames)
            
            # Handle different return formats
            if isinstance(latents, (list, tuple)):
                latents = latents[0]
            
            # Expected shape: [1, 16, T, H, W] or [16, T, H, W]
            if latents.dim() == 5 and latents.shape[0] == 1:
                latents = latents.squeeze(0)  # Remove batch dim
            
            return latents  # [16, T, H, W]
        
        inactive_latents = encode_stream(inactive_frames)
        reactive_latents = encode_stream(reactive_frames)
        
        return inactive_latents, reactive_latents


class NV_VideoChunkAnalyzer:
    """
    Node 1: Analyzes videos and creates chunk structure with JSON output.
    Pure data layer - no model dependencies. Supports control videos for WanVACE.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Input video frames to analyze and chunk"
                }),
                "chunk_size": ("INT", {
                    "default": 81,
                    "min": 9,
                    "max": 1000,
                    "step": 1,
                    "tooltip": "Number of frames per chunk"
                }),
                "chunk_overlap": ("INT", {
                    "default": 16,
                    "min": 0,
                    "max": 200,
                    "step": 1,
                    "tooltip": "Overlap between chunks in frames (for smooth blending)"
                }),
                "output_json_path": ("STRING", {
                    "default": "chunks/project.json",
                    "multiline": False,
                    "tooltip": "Path to save chunk JSON file (will be created)"
                }),
                "save_chunks": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Save each chunk as a separate video file for preview"
                }),
            },
            "optional": {
                "chunks_folder": ("STRING", {
                    "default": "chunks/videos",
                    "multiline": False,
                    "tooltip": "Folder to save chunk videos (if save_chunks is enabled)"
                }),
                "chunk_fps": ("INT", {
                    "default": 30,
                    "min": 1,
                    "max": 120,
                    "tooltip": "FPS for saved chunk videos"
                }),
                # Support for AutomateVideoPathLoader outputs
                "beauty_path": ("STRING", {"default": "", "tooltip": "Beauty/main video pass"}),
                "depth_path": ("STRING", {"default": "", "tooltip": "Depth map video"}),
                "openpose_path": ("STRING", {"default": "", "tooltip": "OpenPose video"}),
                "canny_path": ("STRING", {"default": "", "tooltip": "Canny edge video"}),
                "lineart_path": ("STRING", {"default": "", "tooltip": "Line art video"}),
                # Manual control paths (one per line)
                "control_video_paths": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Optional: Manual control paths (format: name:path per line, e.g., 'depth:D:/depth.mp4')"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("json_path", "num_chunks", "info")
    FUNCTION = "analyze"
    CATEGORY = "NV_Utils/Video/Chunking"
    
    def analyze(self, images, chunk_size, chunk_overlap, output_json_path, 
                save_chunks=False, chunks_folder="chunks/videos", chunk_fps=30,
                beauty_path="", depth_path="", openpose_path="", canny_path="", lineart_path="",
                control_video_paths=""):
        import json
        import os
        from pathlib import Path
        
        # Get image dimensions (batch, height, width, channels)
        total_frames, height, width, channels = images.shape
        
        # Parse control video paths - combine structured inputs and manual paths
        control_paths = {}
        
        # Add structured control paths (from AutomateVideoPathLoader)
        if beauty_path and beauty_path.strip():
            control_paths["beauty"] = beauty_path.strip()
        if depth_path and depth_path.strip():
            control_paths["depth"] = depth_path.strip()
        if openpose_path and openpose_path.strip():
            control_paths["openpose"] = openpose_path.strip()
        if canny_path and canny_path.strip():
            control_paths["canny"] = canny_path.strip()
        if lineart_path and lineart_path.strip():
            control_paths["lineart"] = lineart_path.strip()
        
        # Parse manual control paths (format: "name:path" per line)
        if control_video_paths.strip():
            for line in control_video_paths.split('\n'):
                line = line.strip()
                if not line:
                    continue
                if ':' in line:
                    # Format: "name:path"
                    name, path = line.split(':', 1)
                    control_paths[name.strip()] = path.strip()
                else:
                    # Just a path - use filename as name
                    control_paths[os.path.splitext(os.path.basename(line))[0]] = line
        
        # Calculate stride (no rounding needed for image-based chunking)
        stride = chunk_size - chunk_overlap
        
        if stride <= 0:
            raise ValueError(f"Invalid chunk settings: overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})")
        
        # Calculate chunks
        chunks = []
        chunk_idx = 0
        current_frame = 0
        
        info_lines = []
        info_lines.append("=" * 60)
        info_lines.append("NV VIDEO CHUNK ANALYZER")
        info_lines.append("=" * 60)
        info_lines.append(f"Input: {total_frames} frames")
        info_lines.append(f"Resolution: {width}×{height}")
        info_lines.append(f"Chunk size: {chunk_size} frames")
        info_lines.append(f"Chunk overlap: {chunk_overlap} frames")
        info_lines.append(f"Stride: {stride} frames")
        if control_paths:
            info_lines.append(f"Control videos: {len(control_paths)} detected")
            for ctrl_name, ctrl_path in control_paths.items():
                info_lines.append(f"  - {ctrl_name}: {os.path.basename(ctrl_path)}")
        if save_chunks:
            info_lines.append(f"Saving chunks to: {chunks_folder}")
        info_lines.append("")
        
        # Create chunks folder if saving
        if save_chunks:
            os.makedirs(chunks_folder, exist_ok=True)
        
        while current_frame < total_frames:
            end_frame = min(current_frame + chunk_size, total_frames)
            
            # Create per-chunk control settings (user can enable/disable per chunk)
            chunk_controls = {}
            for ctrl_name in control_paths.keys():
                chunk_controls[ctrl_name] = {
                    "enabled": True,  # Default: all controls enabled
                    "weight": 1.0,
                    "start_percent": 0.0,
                    "end_percent": 1.0,
                }
            
            chunk_info = {
                "id": chunk_idx,
                "start_frame": current_frame,
                "end_frame": end_frame,
                "num_frames": end_frame - current_frame,
                "is_first": (current_frame == 0),
                "is_last": (end_frame >= total_frames),
                # Prompts to be filled by user
                "prompt_positive": "",
                "prompt_negative": "",
                # Per-chunk control settings
                "controls": chunk_controls,
            }
            
            # Save chunk video if requested
            if save_chunks:
                chunk_frames = images[current_frame:end_frame]
                chunk_filename = f"chunk_{chunk_idx:04d}.mp4"
                chunk_path = os.path.join(chunks_folder, chunk_filename)
                
                try:
                    self._save_chunk_video(chunk_frames, chunk_path, fps=chunk_fps)
                    chunk_info["saved_video_path"] = chunk_path
                    info_lines.append(f"Chunk {chunk_idx}: frames {current_frame}-{end_frame} ({end_frame - current_frame} frames) -> {chunk_filename}")
                except Exception as e:
                    info_lines.append(f"Chunk {chunk_idx}: frames {current_frame}-{end_frame} (SAVE FAILED: {e})")
            else:
                info_lines.append(f"Chunk {chunk_idx}: frames {current_frame}-{end_frame} ({end_frame - current_frame} frames)")
            
            chunks.append(chunk_info)
            chunk_idx += 1
            current_frame += stride
            
            if chunk_idx > 10000:
                raise ValueError("Too many chunks! Check your chunk_size and overlap settings.")
        
        # Build JSON structure
        json_data = {
            "metadata": {
                "total_frames": total_frames,
                "width": width,
                "height": height,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "resolution": [width, height],  # Keep for compatibility
                "fps": chunk_fps,
            },
            "control_videos": control_paths,  # Dict of {name: path}
            "global_settings": {
                "default_negative": "blurry, low quality, distorted",
                "notes": "Edit this JSON to customize prompts and control settings per chunk",
            },
            "chunks": chunks
        }
        
        # Save JSON with debug output and error handling
        info_lines.append("")
        info_lines.append(f"Total chunks: {len(chunks)}")
        info_lines.append("")
        info_lines.append("Attempting to save JSON...")
        
        # Get absolute path
        json_absolute_path = os.path.abspath(output_json_path)
        json_dir = os.path.dirname(json_absolute_path)
        
        info_lines.append(f"  Target path: {json_absolute_path}")
        info_lines.append(f"  Target directory: {json_dir}")
        
        try:
            # Create directory if needed
            if json_dir:
                os.makedirs(json_dir, exist_ok=True)
                info_lines.append(f"  Directory exists: {os.path.exists(json_dir)}")
                info_lines.append(f"  Directory writable: {os.access(json_dir, os.W_OK)}")
            
            # Try to write JSON
            with open(json_absolute_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            info_lines.append(f"  ✓ JSON saved successfully!")
            info_lines.append(f"  File size: {os.path.getsize(json_absolute_path)} bytes")
            
        except PermissionError as e:
            info_lines.append(f"  ✗ Permission denied: {e}")
            info_lines.append(f"  Current working directory: {os.getcwd()}")
            
            # Try fallback to ComfyUI output directory
            try:
                import folder_paths
                fallback_dir = os.path.join(folder_paths.get_output_directory(), "chunks")
                os.makedirs(fallback_dir, exist_ok=True)
                fallback_path = os.path.join(fallback_dir, os.path.basename(output_json_path))
                
                info_lines.append(f"  Trying fallback location: {fallback_path}")
                
                with open(fallback_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, indent=2, ensure_ascii=False)
                
                json_absolute_path = fallback_path
                info_lines.append(f"  ✓ JSON saved to fallback location!")
                
            except Exception as fallback_error:
                info_lines.append(f"  ✗ Fallback also failed: {fallback_error}")
                raise  # Re-raise the original error
                
        except Exception as e:
            info_lines.append(f"  ✗ Unexpected error: {type(e).__name__}: {e}")
            info_lines.append(f"  Current working directory: {os.getcwd()}")
            raise
        info_lines.append("")
        info_lines.append("NEXT STEPS:")
        info_lines.append("1. Edit the JSON file to add prompts per chunk")
        if control_paths:
            info_lines.append("2. Enable/disable controls per chunk in 'controls' section")
            info_lines.append("   - Set 'enabled': false to disable a control for a chunk")
            info_lines.append("   - Adjust 'weight' to control strength (0.0-2.0)")
            info_lines.append("3. Feed JSON to NV_ChunkConditioningPreprocessor")
        else:
            info_lines.append("2. Feed JSON to NV_ChunkConditioningPreprocessor")
        info_lines.append("=" * 60)
        
        info_text = "\n".join(info_lines)
        print(info_text)
        
        return (json_absolute_path, len(chunks), info_text)
    
    def _save_chunk_video(self, frames, output_path, fps=30):
        """Helper to save a chunk of frames as a video using CustomVideoSaver"""
        # Use the existing CustomVideoSaver class
        video_saver = CustomVideoSaver()
        
        # Extract directory and filename from output_path
        output_dir = os.path.dirname(output_path)
        filename = os.path.splitext(os.path.basename(output_path))[0]
        
        # Save the video using CustomVideoSaver with good defaults
        result = video_saver.save_video(
            video_tensor=frames,
            filename_prefix=filename,
            custom_directory=output_dir,
            video_format="mp4",
            codec="H.264/AVC",  # Good compatibility and quality
            fps=float(fps),
            quality=23,  # Good quality
            encoding_preset="fast",  # Faster encoding for chunks
            preserve_colors=True,
            use_ffmpeg=True,
            subfolder=""
        )
        
        # CustomVideoSaver returns (video_path, filename, info)
        # We just need to verify it succeeded
        if result[0] != output_path:
            # Rename if the filename doesn't match exactly (due to counter increment)
            import shutil
            if os.path.exists(result[0]):
                shutil.move(result[0], output_path)


class NV_ChunkConditioningPreprocessor:
    """
    Node 2: Loads chunk JSON and encodes all conditioning (prompts + controls).
    General-purpose conditioning encoder supporting CLIP, VAE, and WanVACE control formats.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "chunk_json_path": ("STRING", {
                    "default": "chunks/project.json",
                    "multiline": False,
                    "tooltip": "Path to chunk JSON from NV_VideoChunkAnalyzer"
                }),
                "clip": ("CLIP", {
                    "tooltip": "CLIP model for encoding text prompts"
                }),
            },
            "optional": {
                "vae": ("VAE", {
                    "tooltip": "VAE for encoding control videos (required if using WanVACE controls)"
                }),
                "start_image": ("IMAGE", {
                    "tooltip": "First frame reference for ALL chunks (improves character/scene consistency)"
                }),
                "global_positive": ("CONDITIONING", {
                    "tooltip": "Fallback positive conditioning for chunks without custom prompts"
                }),
                "global_negative": ("CONDITIONING", {
                    "tooltip": "Fallback negative conditioning"
                }),
                "control_mode": (["disabled", "wan_vace", "wan_controlnet"], {
                    "default": "disabled",
                    "tooltip": "Control mode: disabled, wan_vace (VACE controls), wan_controlnet"
                }),
                "tiled_vae": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use tiled VAE encoding for large videos (saves VRAM)"
                }),
            }
        }
    
    RETURN_TYPES = ("CHUNK_CONDITIONING_LIST", "LATENT", "STRING")
    RETURN_NAMES = ("chunk_conditionings", "latent", "info")
    FUNCTION = "preprocess"
    CATEGORY = "NV_Utils/Video/Chunking"
    
    def preprocess(self, chunk_json_path, clip, vae=None, start_image=None,
                   global_positive=None, global_negative=None,
                   control_mode="disabled", tiled_vae=False):
        import json
        import os
        import cv2
        import numpy as np
        import torch
        import comfy.utils
        
        # Load JSON
        if not os.path.exists(chunk_json_path):
            # Provide helpful error message
            if os.path.isdir(chunk_json_path):
                raise FileNotFoundError(
                    f"Chunk JSON path is a directory, not a file: {chunk_json_path}\n"
                    f"Please provide the full path including .json filename (e.g., '{os.path.join(chunk_json_path, 'project.json')}')"
                )
            elif not chunk_json_path.endswith('.json'):
                raise FileNotFoundError(
                    f"Chunk JSON file not found: {chunk_json_path}\n"
                    f"Did you forget the .json extension? Try: {chunk_json_path}.json"
                )
            else:
                raise FileNotFoundError(f"Chunk JSON not found: {chunk_json_path}")
        
        with open(chunk_json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        chunks = json_data.get("chunks", [])
        metadata = json_data.get("metadata", {})
        global_settings = json_data.get("global_settings", {})
        control_videos = json_data.get("control_videos", {})
        
        info_lines = []
        info_lines.append("=" * 60)
        info_lines.append("NV CHUNK CONDITIONING PREPROCESSOR")
        info_lines.append("=" * 60)
        info_lines.append(f"Loading: {chunk_json_path}")
        info_lines.append(f"Total chunks: {len(chunks)}")
        info_lines.append(f"Control mode: {control_mode}")
        if control_mode != "disabled" and control_videos:
            info_lines.append(f"Control videos: {list(control_videos.keys())}")
        info_lines.append("")
        
        # Validate control mode setup
        if control_mode != "disabled" and not vae:
            info_lines.append("⚠️ WARNING: Control mode enabled but no VAE provided!")
            info_lines.append("⚠️ Control videos will NOT be encoded. Falling back to disabled.")
            control_mode = "disabled"
        
        # Encode start_image as VACE reference (NOT I2V concat_latent)
        # Per WAN VACE guide: reference_image is prepended to control videos as frame 0
        vace_reference_latent = None
        if start_image is not None and vae is not None and control_mode == "wan_vace":
            info_lines.append("Encoding start image as VACE reference...")
            try:
                # Resize to match video dimensions
                video_width = metadata.get("width", 832)
                video_height = metadata.get("height", 480)
                
                # Take first frame if multiple frames provided
                if start_image.shape[0] > 1:
                    start_image = start_image[0:1]
                
                # Resize to video resolution
                start_image_resized = comfy.utils.common_upscale(
                    start_image.movedim(-1, 1), 
                    video_width, video_height, 
                    "bilinear", "center"
                ).movedim(1, -1)
                
                # Encode single frame with VAE → 16 channels
                reference_encoded = vae.encode(start_image_resized[:, :, :, :3])  # [1, 16, 1, H, W]
                
                # Pad with zeros to make 32 channels (no reactive stream for reference)
                # Per guide: reference = [16ch VAE, 16ch ZEROS]
                zeros_padding = torch.zeros_like(reference_encoded)
                vace_reference_latent = torch.cat([reference_encoded, zeros_padding], dim=1)  # [1, 32, 1, H, W]
                
                print(f"  ✓ VACE reference encoded: {vace_reference_latent.shape}")
                print(f"    Structure: 16ch VAE + 16ch zeros")
                print(f"    Pixels stored for color matching: {start_image_resized.shape}")
                info_lines.append(f"  ✓ Start image encoded as VACE reference: {vace_reference_latent.shape}")
                
                # Store pixels for color matching in sampler
                self._start_image_pixels = start_image_resized
            except Exception as e:
                info_lines.append(f"  ✗ Failed to encode VACE reference: {e}")
                vace_reference_latent = None
        
        # Encode FULL control videos ONCE (not per-chunk)
        # This matches how the main latent works and avoids temporal interpolation
        # NEW: Store control videos in pixel space for chunk-by-chunk encoding
        full_control_pixels = {}
        if control_mode != "disabled" and control_videos and vae:
            info_lines.append("Loading control videos (pixel space for chunk-by-chunk encoding)...")
            for ctrl_name, ctrl_path in control_videos.items():
                if not ctrl_path or not os.path.exists(ctrl_path):
                    info_lines.append(f"  ⚠️ Control '{ctrl_name}' path not found: {ctrl_path}")
                    continue

                try:
                    # Load ENTIRE control video (not chunked)
                    total_frames = metadata.get("total_frames", 81)
                    ctrl_frames = self._load_video_frames(ctrl_path, 0, total_frames)

                    if ctrl_frames is not None:
                        # Store control frames in PIXEL space for chunk-by-chunk encoding
                        full_control_pixels[ctrl_name] = ctrl_frames  # [T, H, W, C] in range [0, 1]

                        info_lines.append(f"  ✓ Loaded '{ctrl_name}': {ctrl_frames.shape} (pixel space)")
                except Exception as e:
                    info_lines.append(f"  ✗ Failed to load control '{ctrl_name}': {e}")
            
            info_lines.append("")
        
        # Process each chunk
        chunk_conditionings = []
        
        for chunk_idx, chunk in enumerate(chunks):
            prompt_pos = chunk.get("prompt_positive", "").strip()
            prompt_neg = chunk.get("prompt_negative", "").strip()
            
            # Use default if empty
            if not prompt_neg:
                prompt_neg = global_settings.get("default_negative", "")
            
            # Encode prompts with CLIP
            if prompt_pos:
                # Use ComfyUI's CLIPTextEncode equivalent
                tokens_pos = clip.tokenize(prompt_pos)
                cond_pos, pooled_pos = clip.encode_from_tokens(tokens_pos, return_pooled=True)
                positive_cond = [[cond_pos, {"pooled_output": pooled_pos}]]
            else:
                positive_cond = global_positive if global_positive is not None else []
            
            if prompt_neg:
                tokens_neg = clip.tokenize(prompt_neg)
                cond_neg, pooled_neg = clip.encode_from_tokens(tokens_neg, return_pooled=True)
                negative_cond = [[cond_neg, {"pooled_output": pooled_neg}]]
            else:
                negative_cond = global_negative if global_negative is not None else []
            
            # Reference control videos in PIXEL space for chunk-by-chunk encoding
            control_pixels_info = {}
            if control_mode != "disabled" and full_control_pixels:
                chunk_controls = chunk.get("controls", {})
                for ctrl_name, ctrl_settings in chunk_controls.items():
                    # Skip if this control is disabled for this chunk
                    if not ctrl_settings.get("enabled", True):
                        continue

                    # Skip controls with weight=0 (no effect)
                    weight = ctrl_settings.get("weight", 1.0)
                    if weight == 0.0:
                        continue

                    # Get full control pixels
                    if ctrl_name not in full_control_pixels:
                        info_lines.append(f"  ⚠️ Control '{ctrl_name}' not found in loaded controls")
                        continue

                    # Store reference to FULL control pixels
                    # Sampler will encode the appropriate slice
                    control_pixels_info[ctrl_name] = {
                        "pixels": full_control_pixels[ctrl_name],  # [T, H, W, C] full video in pixel space
                        "weight": ctrl_settings.get("weight", 1.0),
                        "start_percent": ctrl_settings.get("start_percent", 0.0),
                        "end_percent": ctrl_settings.get("end_percent", 1.0),
                    }
            
            # Package conditioning for this chunk
            chunk_conditioning = {
                "chunk_id": chunk.get("id", chunk_idx),
                "start_frame": chunk.get("start_frame", 0),
                "end_frame": chunk.get("end_frame", 0),
                "num_frames": chunk.get("num_frames", 0),
                "positive": positive_cond,
                "negative": negative_cond,
                "control_pixels_info": control_pixels_info if control_pixels_info else None,  # Pixel-space controls
                "vace_reference": vace_reference_latent,  # VACE reference (global anchor)
                "start_image_pixels": getattr(self, '_start_image_pixels', None),  # For pixel-space color matching
                "is_first": chunk.get("is_first", False),
                "is_last": chunk.get("is_last", False),
                "vae": vae,  # Store VAE for chunk-by-chunk encoding
                "tiled_vae": tiled_vae,  # Store tiled setting
            }
            
            chunk_conditionings.append(chunk_conditioning)
            
            info_lines.append(f"Chunk {chunk_idx}:")
            if prompt_pos:
                info_lines.append(f"  Positive: \"{prompt_pos[:60]}...\"" if len(prompt_pos) > 60 else f"  Positive: \"{prompt_pos}\"")
            else:
                info_lines.append(f"  Positive: (using global)")
            info_lines.append(f"  Negative: \"{prompt_neg[:40]}...\"" if len(prompt_neg) > 40 else f"  Negative: \"{prompt_neg}\"")
        
        info_lines.append("")
        info_lines.append(f"Encoded {len(chunk_conditionings)} chunk conditionings")
        
        # Create empty latent based on video metadata
        # This automates latent creation for the sampler
        video_width = metadata.get("width", 832)
        video_height = metadata.get("height", 480)
        video_frames = metadata.get("total_frames", 81)
        
        # Wan uses 16-channel latents, but we'll use 16 as default for video models
        # Users can override by providing their own latent to the sampler
        latent_channels = 16  # Wan/video models
        
        # Convert pixel dimensions to latent dimensions (Wan VAE: ÷8 spatial, ÷4 temporal)
        latent_height = video_height // 8
        latent_width = video_width // 8
        latent_frames = (video_frames - 1) // 4 + 1  # 4n+1 formula
        
        # Create empty latent: [batch, channels, frames, height, width]
        empty_latent = torch.zeros(1, latent_channels, latent_frames, latent_height, latent_width)
        
        latent_dict = {"samples": empty_latent}
        
        info_lines.append("")
        info_lines.append(f"Generated empty latent:")
        info_lines.append(f"  Video: {video_frames}f × {video_height}h × {video_width}w")
        info_lines.append(f"  Latent: {latent_frames}f × {latent_height}h × {latent_width}w × {latent_channels}ch")
        info_lines.append("=" * 60)
        
        info_text = "\n".join(info_lines)
        print(info_text)
        
        return (chunk_conditionings, latent_dict, info_text)
    
    def _load_video_frames(self, video_path, start_frame, end_frame):
        """Load a range of frames from a video file"""
        import cv2
        import numpy as np
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Clamp to video length
        start_frame = max(0, min(start_frame, total_frames - 1))
        end_frame = max(start_frame + 1, min(end_frame, total_frames))
        
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        for frame_idx in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Normalize to [0, 1]
            frame_normalized = frame_rgb.astype(np.float32) / 255.0
            frames.append(frame_normalized)
        
        cap.release()
        
        if not frames:
            return None
        
        # Convert to tensor: [T, H, W, C]
        frames_tensor = torch.from_numpy(np.stack(frames, axis=0))
        return frames_tensor
    
    def _encode_vace_dual_channel(self, frames, vae, tiled=False):
        """
        Encode control frames using Wan VACE dual-channel format.
        
        Implements the dual-stream architecture:
        - Inactive stream (channels 0-15): Context preservation
        - Reactive stream (channels 16-31): Active control
        
        Args:
            frames: [T, H, W, C] in range [0, 1]
            vae: VAE model
            tiled: Whether to use tiled encoding
            
        Returns:
            32-channel latent: [32, T_lat, H_lat, W_lat]
        """
        import torch
        
        print(f"[VACE Dual] Encoding with dual-channel format")
        print(f"[VACE Dual] Input frames shape: {frames.shape}")
        
        # Default mask: all ones (full control everywhere)
        # mask shape: [T, H, W, 1]
        mask = torch.ones(frames.shape[0], frames.shape[1], frames.shape[2], 1)
        
        # Step 1: Center control video around zero
        # From guide: "control_video = control_video - 0.5"
        frames_centered = frames - 0.5
        print(f"[VACE Dual] Centered frames range: [{frames_centered.min():.3f}, {frames_centered.max():.3f}]")
        
        # Step 2: Split into inactive and reactive streams
        # inactive = (centered * (1 - mask)) + 0.5  → Preserves non-masked areas
        # reactive = (centered * mask) + 0.5        → Applies masked areas
        inactive = (frames_centered * (1 - mask)) + 0.5  # [T, H, W, C]
        reactive = (frames_centered * mask) + 0.5        # [T, H, W, C]
        
        print(f"[VACE Dual] Inactive range: [{inactive.min():.3f}, {inactive.max():.3f}]")
        print(f"[VACE Dual] Reactive range: [{reactive.min():.3f}, {reactive.max():.3f}]")
        
        # Step 3: Encode both streams with VAE
        inactive_latents = self._encode_control_frames(inactive, vae, tiled)  # [16, T_lat, H_lat, W_lat]
        reactive_latents = self._encode_control_frames(reactive, vae, tiled)  # [16, T_lat, H_lat, W_lat]
        
        print(f"[VACE Dual] Inactive latents: {inactive_latents.shape}")
        print(f"[VACE Dual] Reactive latents: {reactive_latents.shape}")
        
        # Step 4: Return as separate tensors (model will concatenate internally)
        # DO NOT concatenate here - model expects them separate for processing
        print(f"[VACE Dual] ✓ Encoding complete (returning as separate streams)")
        
        return inactive_latents, reactive_latents
    
    def _encode_control_frames(self, frames, vae, tiled=False):
        """Encode control frames with VAE (WanVACE format)"""
        import torch
        import comfy.utils
        
        # frames shape: [T, H, W, C] in range [0, 1]
        # Need to convert to VAE format: [C, T, H, W] in range [-1, 1]
        
        device = vae.device if hasattr(vae, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dtype = vae.dtype if hasattr(vae, 'dtype') else torch.float32
        
        # Move to device and convert dtype
        frames = frames.to(device).to(dtype)
        
        T, H, W, C = frames.shape
        
        # CRITICAL: Resize to make dimensions divisible by 16 (required by Wan VAE)
        if W % 16 != 0 or H % 16 != 0:
            new_height = (H // 16) * 16
            new_width = (W // 16) * 16
            print(f"[Control Encode] Resizing {W}x{H} -> {new_width}x{new_height} (must be divisible by 16)")
            
            # Use ComfyUI's upscale function: [T, H, W, C] -> [T, C, H, W] -> upscale -> [T, H, W, C]
            frames = frames.permute(0, 3, 1, 2)  # [T, C, H, W]
            frames = comfy.utils.common_upscale(frames, new_width, new_height, "lanczos", "disabled")
            frames = frames.permute(0, 2, 3, 1)  # [T, H, W, C]
            H, W = new_height, new_width
        
        # NOTE: Empirically, very small chunks (<9 frames) may fail with temporal VAE
        # This might be due to temporal convolution requirements, not confirmed
        # Wan VAE *can* encode single images for I2V, but control encoding may differ
        T, H, W, C = frames.shape  # Still [T, H, W, C]
        MIN_FRAMES = 9  # Empirically determined safe minimum (may be reducible)
        
        original_T = T
        if T < MIN_FRAMES:
            print(f"[Control Encode] ⚠️ Chunk has only {T} frames, padding to {MIN_FRAMES} (temporal VAE requirement)")
            # Pad by repeating the last frame
            padding_needed = MIN_FRAMES - T
            last_frame = frames[-1:, :, :, :].repeat(padding_needed, 1, 1, 1)  # Repeat last frame
            frames = torch.cat([frames, last_frame], dim=0)  # [MIN_FRAMES, H, W, C]
            T = frames.shape[0]
            print(f"[Control Encode] Padded shape: {frames.shape}")
        
        # Keep in ComfyUI video format: [T, H, W, C] with values in [0, 1]
        # DON'T normalize or permute - VAE wrapper handles this
        
        # Encode with VAE
        with torch.no_grad():
            try:
                print(f"[Control Encode] Attempting VAE encode...")
                print(f"[Control Encode] VAE type: {type(vae)}")
                print(f"[Control Encode] Input shape: {frames.shape} (format: [T, H, W, C])")
                print(f"[Control Encode] Value range: [{frames.min():.3f}, {frames.max():.3f}]")
                
                # ComfyUI VAE expects [T, H, W, C] format (video frames) just like LoadVideo output
                # Ensure only RGB channels (first 3)
                if frames.shape[-1] == 4:
                    frames = frames[:, :, :, :3]  # Remove alpha channel if present
                
                # Check if this is a ComfyUI VAE wrapper
                if hasattr(vae, 'first_stage_model'):
                    print(f"[Control Encode] Detected ComfyUI VAE wrapper")
                    # This is the comfy VAE wrapper, call encode directly
                    # Pass frames as [T, H, W, C] - same format as VAEEncode node
                    latents = vae.encode(frames[:, :, :, :3])
                elif hasattr(vae, 'model'):
                    print(f"[Control Encode] Detected raw VAE model")
                    latents = vae.encode(frames[:, :, :, :3])
                else:
                    print(f"[Control Encode] Unknown VAE type, trying direct encode")
                    latents = vae.encode(frames[:, :, :, :3])
                
                print(f"[Control Encode] Encode successful!")
                print(f"[Control Encode] Latents type: {type(latents)}")
                print(f"[Control Encode] Latents shape: {latents.shape if hasattr(latents, 'shape') else 'N/A'}")
                
                # Handle different return types
                if isinstance(latents, list):
                    print(f"[Control Encode] Latents is list, getting first element")
                    latents = latents[0]
                elif isinstance(latents, tuple):
                    print(f"[Control Encode] Latents is tuple, getting first element")
                    latents = latents[0]
                
                print(f"[Control Encode] Latents shape after unwrapping: {latents.shape}")
                
                # Determine temporal dimension based on shape
                # Could be [B, C, T, H, W] or [C, T, H, W]
                if latents.dim() == 5:
                    # Has batch dimension
                    if latents.shape[0] == 1:
                        latents = latents.squeeze(0)  # Remove batch: [1, C, T, H, W] -> [C, T, H, W]
                        print(f"[Control Encode] Removed batch dim: {latents.shape}")
                    
                    # Now should be [C, T, H, W]
                    temporal_dim_idx = 1
                elif latents.dim() == 4:
                    # No batch dimension, already [C, T, H, W]
                    temporal_dim_idx = 1
                else:
                    print(f"[Control Encode] ⚠️ Unexpected latent dimensions: {latents.dim()}")
                    temporal_dim_idx = 1
                
                # Trim back to original temporal size if we padded
                if original_T < MIN_FRAMES and latents.dim() >= 4:
                    latent_T = latents.shape[temporal_dim_idx]
                    # Wan VAE downsamples by 4 temporally
                    original_latent_T = (original_T - 1) // 4 + 1
                    if latent_T > original_latent_T:
                        print(f"[Control Encode] Trimming latents from {latent_T} to {original_latent_T} temporal frames")
                        latents = latents[:, :original_latent_T, :, :]
                
                print(f"[Control Encode] ✓ Final latents shape: {latents.shape}")
                    
            except Exception as e:
                import traceback
                print(f"[Control Encode] ✗ ENCODE FAILED!")
                print(f"[Control Encode] Exception: {e}")
                print(f"[Control Encode] Traceback:")
                traceback.print_exc()
                raise RuntimeError(f"Failed to encode control frames with VAE: {e}\n"
                                 f"Input shape: {frames.shape}, dtype: {frames.dtype}, device: {frames.device}\n"
                                 f"VAE type: {type(vae)}\n\n"
                                 f"SUGGESTION: Try setting control_mode='disabled' to test without controls first.")
        
        # latents shape should now be [C, T_lat, H_lat, W_lat]
        return latents
    
    def _encode_vace_dual_channel_standalone(self, frames, vae, tiled=False):
        """
        Standalone VACE dual-channel encoding for NV_VideoSampler Tier 3.
        This is a copy of NV_ChunkConditioningPreprocessor's method for use in the sampler.
        """
        import torch
        
        print(f"[VACE Dual] Encoding with dual-channel format")
        print(f"[VACE Dual] Input frames shape: {frames.shape}")
        
        # Step 1: Center pixel values
        frames_centered = frames - 0.5
        print(f"[VACE Dual] Centered frames range: [{frames_centered.min():.3f}, {frames_centered.max():.3f}]")
        
        # Step 2: Split into inactive and reactive (using default mask of all ones)
        mask = torch.ones_like(frames_centered[:, :, :, :1])  # All reactive by default
        inactive = (frames_centered * (1 - mask)) + 0.5
        reactive = (frames_centered * mask) + 0.5
        
        print(f"[VACE Dual] Inactive range: [{inactive.min():.3f}, {inactive.max():.3f}]")
        print(f"[VACE Dual] Reactive range: [{reactive.min():.3f}, {reactive.max():.3f}]")
        
        # Step 3: Encode both streams separately
        inactive_latents = self._encode_control_frames(inactive, vae, tiled)
        reactive_latents = self._encode_control_frames(reactive, vae, tiled)
        
        print(f"[VACE Dual] Inactive latents: {inactive_latents.shape}")
        print(f"[VACE Dual] Reactive latents: {reactive_latents.shape}")
        
        # Step 4: Return as separate tensors
        print(f"[VACE Dual] ✓ Encoding complete (returning as separate streams)")
        
        return inactive_latents, reactive_latents


# Register the nodes (NodeBypasser is frontend-only, no Python registration needed)
NODE_CLASS_MAPPINGS = {
    "KNF_Organizer": KNF_Organizer,
    "GeminiVideoCaptioner": GeminiVideoCaptioner,
    "AutomateVideoPathLoader": AutomateVideoPathLoader,
    "CustomVideoSaver": CustomVideoSaver,
    "LazySwitch": LazySwitch,
    "VideoExtensionDiagnostic": VideoExtensionDiagnostic,
    "VAE_LUT_Generator": VAE_LUT_Generator,
    "VAE_Correction_Applier": VAE_Correction_Applier,
    "NV_VideoSampler": NV_VideoSampler,
    "NV_VideoChunkAnalyzer": NV_VideoChunkAnalyzer,
    "NV_ChunkConditioningPreprocessor": NV_ChunkConditioningPreprocessor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KNF_Organizer": "KNF_Organizer",
    "GeminiVideoCaptioner": "Multi-API Video Captioner",
    "AutomateVideoPathLoader": "Automate Video Path Loader",
    "CustomVideoSaver": "Custom Video Saver",
    "LazySwitch": "Lazy Switch",
    "VideoExtensionDiagnostic": "Video Extension Diagnostic",
    "VAE_LUT_Generator": "VAE LUT Generator",
    "VAE_Correction_Applier": "VAE Correction Applier",
    "NV_VideoSampler": "NV Video Sampler",
    "NV_VideoChunkAnalyzer": "NV Video Chunk Analyzer",
    "NV_ChunkConditioningPreprocessor": "NV Chunk Conditioning Preprocessor",
}

# Conditionally add NV_Video_Loader_Path if it imported successfully
if NV_VIDEO_LOADER_AVAILABLE:
    NODE_CLASS_MAPPINGS["NV_Video_Loader_Path"] = NV_Video_Loader_Path
    NODE_DISPLAY_NAME_MAPPINGS["NV_Video_Loader_Path"] = "NV Video Loader Path"
    print("[NV_Comfy_Utils] NV_Video_Loader_Path registered successfully")
else:
    print("[NV_Comfy_Utils] NV_Video_Loader_Path NOT registered (import failed)")
