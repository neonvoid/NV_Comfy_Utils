"""Tests for NV_R2VPromptSplitter — extract prompt and negative prompt from R2V captioner output."""

import pytest
from src.KNF_Utils.r2v_prompt_splitter import NV_R2VPromptSplitter


@pytest.fixture
def splitter():
    return NV_R2VPromptSplitter()


SAMPLE_CAPTION = """\
@Video1 sits at a wooden desk in a warmly lit home office, gesturing with \
their right hand while speaking. Bookshelves line the wall behind them. Warm \
tungsten key light from the upper left, soft fill from a window on the right. \
Medium close-up, static camera, shallow depth of field with background \
softly blurred. Natural movement, cinematic quality.
NEGATIVE: low quality, blurry, distorted faces, unnatural movement, text, \
watermarks, shaky camera"""


# ---------------------------------------------------------------------------
# Basic splitting
# ---------------------------------------------------------------------------

def test_basic_split(splitter):
    """Caption with NEGATIVE: line splits into prompt and negative."""
    prompt, negative, info = splitter.split_caption(SAMPLE_CAPTION)
    assert "@Video1" in prompt
    assert "wooden desk" in prompt
    assert "low quality" in negative
    assert "distorted faces" in negative
    assert "NEGATIVE:" not in prompt


def test_prompt_does_not_contain_negative_marker(splitter):
    """The prompt output should not contain the NEGATIVE: prefix."""
    prompt, _, _ = splitter.split_caption(SAMPLE_CAPTION)
    assert "NEGATIVE" not in prompt


def test_negative_extracted_clean(splitter):
    """Negative prompt is clean text without the NEGATIVE: prefix."""
    _, negative, _ = splitter.split_caption(SAMPLE_CAPTION)
    assert not negative.startswith("NEGATIVE")
    assert negative.startswith("low quality")


# ---------------------------------------------------------------------------
# No NEGATIVE: line
# ---------------------------------------------------------------------------

def test_no_negative_line(splitter):
    """Caption without NEGATIVE: returns empty negative_prompt."""
    caption = "@Video1 walks through a park. Sunny day, wide shot."
    prompt, negative, info = splitter.split_caption(caption)
    assert prompt == caption
    assert negative == ""
    assert "No NEGATIVE: line found" in info


# ---------------------------------------------------------------------------
# Case insensitivity
# ---------------------------------------------------------------------------

def test_case_insensitive_lowercase(splitter):
    """'negative:' (lowercase) is recognized."""
    caption = "Scene description here.\nnegative: ugly, blurry"
    prompt, negative, _ = splitter.split_caption(caption)
    assert "Scene description" in prompt
    assert "ugly" in negative


def test_case_insensitive_mixed(splitter):
    """'Negative:' (mixed case) is recognized."""
    caption = "Scene description here.\nNegative: ugly, blurry"
    prompt, negative, _ = splitter.split_caption(caption)
    assert "ugly" in negative


def test_case_insensitive_upper(splitter):
    """'NEGATIVE:' (uppercase) is recognized."""
    caption = "Scene description here.\nNEGATIVE: ugly, blurry"
    prompt, negative, _ = splitter.split_caption(caption)
    assert "ugly" in negative


# ---------------------------------------------------------------------------
# Multiple NEGATIVE: lines — last one wins
# ---------------------------------------------------------------------------

def test_multiple_negative_lines(splitter):
    """When multiple NEGATIVE: lines exist, last one wins."""
    caption = (
        "Scene part 1.\n"
        "NEGATIVE: first negative\n"
        "Scene part 2.\n"
        "NEGATIVE: second negative, final"
    )
    prompt, negative, _ = splitter.split_caption(caption)
    assert "second negative" in negative
    assert "Scene part 1" in prompt
    assert "Scene part 2" in prompt


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------

def test_prompt_truncation(splitter):
    """Long prompt is truncated with warning."""
    long_scene = "A " * 500  # ~1000 chars
    caption = f"{long_scene}\nNEGATIVE: bad quality"
    prompt, negative, info = splitter.split_caption(caption, max_prompt_chars=200)
    assert len(prompt) <= 200
    assert "TRUNCATED" in info
    assert "bad quality" in negative


def test_negative_truncation(splitter):
    """Long negative prompt is truncated with warning."""
    long_neg = "bad, " * 200  # ~1000 chars
    caption = f"Short prompt.\nNEGATIVE: {long_neg}"
    _, negative, info = splitter.split_caption(caption, max_negative_chars=100)
    assert len(negative) <= 100
    assert "TRUNCATED" in info


def test_no_truncation_within_limits(splitter):
    """No truncation when within limits."""
    _, _, info = splitter.split_caption(SAMPLE_CAPTION)
    assert "TRUNCATED" not in info


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_caption(splitter):
    """Empty caption returns empty strings."""
    prompt, negative, info = splitter.split_caption("")
    assert prompt == ""
    assert negative == ""
    assert "Empty" in info


def test_whitespace_only_caption(splitter):
    """Whitespace-only caption treated as empty."""
    prompt, negative, info = splitter.split_caption("   \n  \t  ")
    assert prompt == ""
    assert negative == ""
    assert "Empty" in info


def test_negative_at_very_start(splitter):
    """NEGATIVE: at the very beginning results in empty prompt."""
    caption = "NEGATIVE: bad quality, blurry"
    prompt, negative, _ = splitter.split_caption(caption)
    assert prompt == ""
    assert "bad quality" in negative


def test_negative_with_leading_whitespace(splitter):
    """NEGATIVE: with leading whitespace on the line is still matched."""
    caption = "Scene description.\n   NEGATIVE: bad quality"
    prompt, negative, _ = splitter.split_caption(caption)
    assert "Scene description" in prompt
    assert "bad quality" in negative


# ---------------------------------------------------------------------------
# Node metadata
# ---------------------------------------------------------------------------

def test_node_metadata():
    """Verify node class attributes."""
    assert NV_R2VPromptSplitter.RETURN_TYPES == ("STRING", "STRING", "STRING")
    assert NV_R2VPromptSplitter.RETURN_NAMES == ("prompt", "negative_prompt", "info")
    assert NV_R2VPromptSplitter.FUNCTION == "split_caption"
    assert NV_R2VPromptSplitter.CATEGORY == "NV_Utils/Prompt"


def test_info_char_counts(splitter):
    """Info output contains character counts."""
    _, _, info = splitter.split_caption(SAMPLE_CAPTION)
    assert "Prompt:" in info
    assert "chars" in info
    assert "Negative:" in info
