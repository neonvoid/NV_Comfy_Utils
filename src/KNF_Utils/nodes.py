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
import sys
import platform

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
            
            # Determine encoding method
            use_ffmpeg_encoding = False
            if use_ffmpeg and self.ffmpeg_available:
                # Use FFmpeg for these codecs (better quality, no COM issues)
                if codec in ["H.265/HEVC", "H.264/AVC", "VP9", "ProRes"]:
                    use_ffmpeg_encoding = True
                    print(f"[CustomVideoSaver] Using FFmpeg for {codec} encoding")
            
            # Convert tensor to video
            if use_ffmpeg_encoding:
                success = self._tensor_to_video_ffmpeg(video_tensor, video_path, fps, quality, codec, 
                                                       encoding_preset, preserve_colors)
                encoding_method = "FFmpeg"
            else:
                success = self._tensor_to_video_opencv(video_tensor, video_path, fps, quality, codec, preserve_colors)
                encoding_method = "OpenCV"
            
            if not success:
                return ("", "", f"Error: Failed to save video to {video_path}")
            
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
            
            # Build FFmpeg command
            # Use pipe input for raw RGB frames
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
            ]
            
            # Add codec-specific parameters
            if codec in ["H.265/HEVC", "H.264/AVC"]:
                cmd.extend([
                    '-crf', str(quality),
                    '-preset', preset,
                    '-pix_fmt', 'yuv420p',  # Maximum compatibility
                ])
            elif codec == "VP9":
                cmd.extend([
                    '-crf', str(quality),
                    '-b:v', '0',  # Use CRF mode
                    '-pix_fmt', 'yuv420p',
                ])
            elif codec == "ProRes":
                # ProRes doesn't use CRF, use profile instead
                cmd.extend([
                    '-profile:v', '3',  # ProRes HQ
                    '-pix_fmt', 'yuv422p10le',
                ])
            
            # Add output file
            cmd.append(output_path)
            
            print(f"[CustomVideoSaver] FFmpeg command: {' '.join(cmd[:10])}...")
            
            # Start FFmpeg process
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10**8
            )
            
            # Write frames to FFmpeg
            for i, frame in enumerate(video_array):
                # Ensure frame is contiguous in memory
                frame = np.ascontiguousarray(frame)
                
                # Validate frame data
                if preserve_colors:
                    frame_min, frame_max = frame.min(), frame.max()
                    if frame_min < 0 or frame_max > 255:
                        print(f"[CustomVideoSaver] Warning: Frame {i} has values outside [0,255]: [{frame_min}, {frame_max}]")
                        frame = np.clip(frame, 0, 255).astype(np.uint8)
                
                # Write raw RGB data
                try:
                    process.stdin.write(frame.tobytes())
                except Exception as e:
                    print(f"[CustomVideoSaver] Error writing frame {i}: {e}")
                    process.kill()
                    return False
            
            # Close stdin and wait for FFmpeg to finish
            process.stdin.close()
            stdout, stderr = process.communicate()
            
            # Check if encoding was successful
            if process.returncode != 0:
                print(f"[CustomVideoSaver] FFmpeg encoding failed with return code {process.returncode}")
                if stderr:
                    error_msg = stderr.decode('utf-8', errors='ignore')
                    print(f"[CustomVideoSaver] FFmpeg error: {error_msg[-500:]}")  # Last 500 chars
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


# Register the nodes (NodeBypasser is frontend-only, no Python registration needed)
NODE_CLASS_MAPPINGS = {
    "KNF_Organizer": KNF_Organizer,
    "GeminiVideoCaptioner": GeminiVideoCaptioner,
    "AutomateVideoPathLoader": AutomateVideoPathLoader,
    "CustomVideoSaver": CustomVideoSaver,
    "LazySwitch": LazySwitch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KNF_Organizer": "KNF_Organizer",
    "GeminiVideoCaptioner": "Multi-API Video Captioner",
    "AutomateVideoPathLoader": "Automate Video Path Loader",
    "CustomVideoSaver": "Custom Video Saver",
    "LazySwitch": "Lazy Switch",
}
