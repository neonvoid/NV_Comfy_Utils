import os
import folder_paths
from pathlib import Path
import re
from typing import Dict, List, Tuple, Optional, Any

class AutomateVideoPathLoader:
    """
    Automatically finds and organizes video file paths based on naming patterns.
    Designed to work with your specific naming convention: LS_Poses.KnightA.beauty, etc.
    
    This node finds videos and returns their paths for use with existing LoadVideo nodes.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": ""}),
                "base_name": ("STRING", {"default": ""}),  # e.g., "KnightA" to filter specific character
            },
            "optional": {
                "video_pattern": ("STRING", {"default": "*.mp4"}),
                "video_types": ("STRING", {"default": "beauty,depth,stencil,alpha,openpose,restyled,optical_flow"}),  # Types to look for
                "auto_detect": ("BOOLEAN", {"default": True}),  # Auto-detect available types
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("beauty_path", "depth_path", "stencil_path", "alpha_path", "openpose_path", "restyled_path", "optical_flow_path", "extra_path", "info")
    FUNCTION = "load_videos"
    CATEGORY = "KNF_Utils/Video"
    
    def load_videos(self, folder_path: str, base_name: str = "", video_pattern: str = "*.mp4", 
                   video_types: str = "beauty,depth,stencil,alpha,openpose,restyled,optical_flow", auto_detect: bool = True) -> Tuple[str, ...]:
        """
        Find and organize videos based on naming patterns.
        
        Args:
            folder_path: Path to folder containing videos
            base_name: Filter by specific base name (e.g., "KnightA")
            video_pattern: File pattern to match (e.g., "*.mp4", "*.mov")
            video_types: Comma-separated list of video types to look for
            auto_detect: Whether to auto-detect available types
            
        Returns:
            Tuple of video file paths: (beauty, depth, stencil, alpha, openpose, restyled, optical_flow, extra, info)
        """
        try:
            if not folder_path or not os.path.exists(folder_path):
                return self._create_empty_result(f"Folder not found: {folder_path}")
            
            # Parse video types
            target_types = [t.strip().lower() for t in video_types.split(',') if t.strip()]
            
            # Get all video files
            path = Path(folder_path)
            video_files = sorted(list(path.glob(video_pattern)))
            
            if not video_files:
                return self._create_empty_result(f"No videos found matching pattern: {video_pattern}")
            
            # Find videos matching the pattern
            matched_videos = self._find_matching_videos(video_files, base_name, target_types, auto_detect)
            
            if not matched_videos:
                return self._create_empty_result("No videos matched the naming pattern")
            
            # Organize videos by type
            organized_videos = self._organize_videos_by_type(matched_videos, target_types)
            
            # Create info string
            info = self._create_info_string(video_files, matched_videos, organized_videos, base_name)
            
            return (
                organized_videos.get("beauty", ""),
                organized_videos.get("depth", ""),
                organized_videos.get("stencil", ""),
                organized_videos.get("alpha", ""),
                organized_videos.get("openpose", ""),
                organized_videos.get("restyled", ""),
                organized_videos.get("optical_flow", ""),
                organized_videos.get("extra", ""),
                info
            )
            
        except Exception as e:
            return self._create_empty_result(f"Error loading videos: {str(e)}")
    
    def _find_matching_videos(self, video_files: List[Path], base_name: str, 
                            target_types: List[str], auto_detect: bool) -> Dict[str, Path]:
        """
        Find videos that match the naming pattern with flexible keyword matching.
        
        Flexible patterns:
        - LS_Poses.KnightA.beauty.mp4 -> base_name="KnightA", type="beauty"
        - KnightA_beauty_video.mp4 -> base_name="KnightA", type="beauty"  
        - video_openpose_KnightA.mp4 -> base_name="KnightA", type="openpose"
        - KnightA_depth_map.mp4 -> base_name="KnightA", type="depth"
        """
        matched_videos = {}
        all_types_found = set()
        
        for video_file in video_files:
            filename = video_file.stem.lower()  # Remove extension and make lowercase
            
            # Check if base_name is in the filename (if specified)
            if base_name and base_name.lower() not in filename:
                continue
            
            # Look for any of the target video types anywhere in the filename
            matched_type = None
            for video_type in target_types:
                if video_type.lower() in filename:
                    matched_type = video_type.lower()
                    break
            
            # If no specific type found but auto-detect is on, try to find any known type
            if not matched_type and auto_detect:
                known_types = ["beauty", "depth", "stencil", "alpha", "openpose", "restyled", "optical_flow"]
                for known_type in known_types:
                    if known_type in filename:
                        matched_type = known_type
                        break
            
            if matched_type:
                # Track all types found for auto-detection
                all_types_found.add(matched_type)
                
                # Store the video (only keep the first match if multiple types found)
                if matched_type not in matched_videos:
                    matched_videos[matched_type] = video_file
        
        # If auto-detect is enabled, use all found types
        if auto_detect:
            print(f"Auto-detected video types: {sorted(all_types_found)}")
        
        return matched_videos
    
    def _organize_videos_by_type(self, matched_videos: Dict[str, Path], target_types: List[str]) -> Dict[str, str]:
        """Organize videos by type and return file paths."""
        organized = {}
        
        # Define the standard video types we support
        standard_types = ["beauty", "depth", "stencil", "alpha", "openpose", "restyled", "optical_flow"]
        
        # Process standard types first
        for video_type in standard_types:
            if video_type in matched_videos:
                organized[video_type] = str(matched_videos[video_type])
        
        # Add any remaining videos as extra
        for video_type, video_path in matched_videos.items():
            if video_type not in standard_types and video_type not in organized:
                organized["extra"] = str(video_path)
                break  # Only keep one extra video
        
        return organized
    
    def _create_empty_result(self, message: str) -> Tuple[str, ...]:
        """Create empty result with error message."""
        return ("", "", "", "", "", "", "", "", message)
    
    def _create_info_string(self, video_files: List[Path], matched_videos: Dict[str, Path], 
                           organized_videos: Dict[str, str], base_name: str) -> str:
        """Create informative status string."""
        total_files = len(video_files)
        matched_count = len(matched_videos)
        organized_count = len([v for v in organized_videos.values() if v])
        
        info_parts = [
            f"Total files: {total_files}",
            f"Matched videos: {matched_count}",
            f"Organized: {organized_count}",
        ]
        
        if base_name:
            info_parts.append(f"Base name: {base_name}")
        
        if matched_videos:
            info_parts.append(f"Types found: {', '.join(sorted(matched_videos.keys()))}")
        
        return " | ".join(info_parts)

# Register the node
NODE_CLASS_MAPPINGS = {
    "AutomateVideoPathLoader": AutomateVideoPathLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AutomateVideoPathLoader": "Automate Video Path Loader"
}
