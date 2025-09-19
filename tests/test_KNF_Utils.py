#!/usr/bin/env python

"""Tests for `KNF_Utils` package."""

import pytest
from src.KNF_Utils.nodes import KNF_Organizer, GeminiVideoCaptioner

@pytest.fixture
def knf_organizer_node():
    """Fixture to create a KNF_Organizer node instance."""
    return KNF_Organizer()

@pytest.fixture
def gemini_captioner_node():
    """Fixture to create a GeminiVideoCaptioner node instance."""
    return GeminiVideoCaptioner()

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
