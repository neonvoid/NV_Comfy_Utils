#!/usr/bin/env python

"""Tests for `KNF_Utils` package."""

import pytest
import torch
import numpy as np
import tempfile
import os
from src.KNF_Utils.nodes import KNF_Organizer, GeminiVideoCaptioner, CustomVideoSaver

@pytest.fixture
def knf_organizer_node():
    """Fixture to create a KNF_Organizer node instance."""
    return KNF_Organizer()

@pytest.fixture
def gemini_captioner_node():
    """Fixture to create a GeminiVideoCaptioner node instance."""
    return GeminiVideoCaptioner()

@pytest.fixture
def custom_video_saver_node():
    """Fixture to create a CustomVideoSaver node instance."""
    return CustomVideoSaver()

@pytest.fixture
def sample_video_tensor():
    """Fixture to create a sample video tensor for testing."""
    # Create a small test video: 5 frames, 64x64, RGB
    video_array = np.random.randint(0, 255, (5, 64, 64, 3), dtype=np.uint8)
    video_tensor = torch.from_numpy(video_array.astype(np.float32) / 255.0)
    return video_tensor

def test_knf_organizer_initialization(knf_organizer_node):
    """Test that the KNF_Organizer node can be instantiated."""
    assert isinstance(knf_organizer_node, KNF_Organizer)

def test_gemini_captioner_initialization(gemini_captioner_node):
    """Test that the GeminiVideoCaptioner node can be instantiated."""
    assert isinstance(gemini_captioner_node, GeminiVideoCaptioner)

def test_gemini_captioner_metadata():
    """Test the GeminiVideoCaptioner node's metadata."""
    assert GeminiVideoCaptioner.RETURN_TYPES == ("STRING",)
    assert GeminiVideoCaptioner.RETURN_NAMES == ("caption",)
    assert GeminiVideoCaptioner.FUNCTION == "caption_video"
    assert GeminiVideoCaptioner.CATEGORY == "KNF_Utils"

def test_gemini_captioner_input_types():
    """Test that the GeminiVideoCaptioner has correct input types."""
    input_types = GeminiVideoCaptioner.INPUT_TYPES()
    required = input_types["required"]
    
    assert "video_file" in required
    assert "prompt_text" in required
    assert "api_key" in required
    assert "model" in required
    
    # Check model choices
    model_choices = required["model"][0]
    assert "gemini-1.5-flash" in model_choices
    assert "gemini-1.5-pro" in model_choices
    assert "gemini-2.5-pro" in model_choices

def test_gemini_captioner_error_handling(gemini_captioner_node):
    """Test error handling for missing inputs."""
    # Test missing video file
    result = gemini_captioner_node.caption_video("", "test prompt", "test_key", "gemini-2.5-pro")
    assert "Error: Video file path and API key are required" in result[0]
    
    # Test missing API key
    result = gemini_captioner_node.caption_video("test.mp4", "test prompt", "", "gemini-2.5-pro")
    assert "Error: Video file path and API key are required" in result[0]

def test_custom_video_saver_initialization(custom_video_saver_node):
    """Test that the CustomVideoSaver node can be instantiated."""
    assert isinstance(custom_video_saver_node, CustomVideoSaver)

def test_custom_video_saver_metadata():
    """Test the CustomVideoSaver node's metadata."""
    assert CustomVideoSaver.RETURN_TYPES == ("STRING", "STRING", "STRING")
    assert CustomVideoSaver.RETURN_NAMES == ("video_path", "filename", "info")
    assert CustomVideoSaver.FUNCTION == "save_video"
    assert CustomVideoSaver.CATEGORY == "KNF_Utils/Video"

def test_custom_video_saver_input_types():
    """Test that the CustomVideoSaver has correct input types."""
    input_types = CustomVideoSaver.INPUT_TYPES()
    required = input_types["required"]
    
    assert "video_tensor" in required
    assert "filename_prefix" in required
    assert "custom_directory" in required
    assert "video_format" in required
    assert "fps" in required
    assert "quality" in required
    
    # Check video format choices
    format_choices = required["video_format"][0]
    assert "mp4" in format_choices
    assert "avi" in format_choices
    assert "mov" in format_choices

def test_custom_video_saver_save_to_temp_directory(custom_video_saver_node, sample_video_tensor):
    """Test saving video to a temporary directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        result = custom_video_saver_node.save_video(
            video_tensor=sample_video_tensor,
            filename_prefix="test_video",
            custom_directory=temp_dir,
            video_format="mp4",
            fps=30.0,
            quality=18,
            preserve_colors=True
        )
        
        video_path, filename, info = result
        
        # Check that the video was saved
        assert video_path != ""
        assert filename != ""
        assert "Video saved:" in info
        assert "Color Preserved" in info
        assert os.path.exists(video_path)
        assert filename.endswith(".mp4")

def test_custom_video_saver_fourcc_codes(custom_video_saver_node):
    """Test that the fourcc codec mapping works correctly."""
    # Test different video formats
    formats = ["mp4", "avi", "mov", "mkv", "webm", "wmv"]
    for fmt in formats:
        fourcc = custom_video_saver_node._get_fourcc(fmt)
        assert fourcc is not None
        assert isinstance(fourcc, int)

def test_custom_video_saver_tensor_conversion(custom_video_saver_node, sample_video_tensor):
    """Test tensor to video file conversion."""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "test_output.mp4")
        
        success = custom_video_saver_node._tensor_to_video_file(
            sample_video_tensor, output_path, 30.0, 18, "mp4", preserve_colors=True
        )
        
        assert success
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0

def test_custom_video_saver_color_preservation(custom_video_saver_node):
    """Test color preservation functionality."""
    # Create a test video with specific color values
    test_colors = np.array([
        [[255, 0, 0], [0, 255, 0], [0, 0, 255]],  # Red, Green, Blue
        [[128, 128, 128], [64, 64, 64], [192, 192, 192]]  # Grays
    ], dtype=np.uint8)
    
    # Convert to float32 normalized format
    test_video_float = test_colors.astype(np.float32) / 255.0
    test_video_tensor = torch.from_numpy(test_video_float)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "color_test.mp4")
        
        # Test with color preservation enabled
        success = custom_video_saver_node._tensor_to_video_file(
            test_video_tensor, output_path, 30.0, 18, "mp4", preserve_colors=True
        )
        
        assert success
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0
