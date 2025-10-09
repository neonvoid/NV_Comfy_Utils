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


# Register the nodes (NodeBypasser is frontend-only, no Python registration needed)
NODE_CLASS_MAPPINGS = {
    "KNF_Organizer": KNF_Organizer,
    "GeminiVideoCaptioner": GeminiVideoCaptioner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KNF_Organizer": "KNF_Organizer",
    "GeminiVideoCaptioner": "Gemini Video Captioner",
}