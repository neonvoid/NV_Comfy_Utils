"""Tests for NV_PromptRefiner - iterative LLM-powered captioner/refiner."""

import importlib.util
import json
import sys
import types
from unittest.mock import patch, MagicMock

import pytest

# Pre-register folder_paths mock before loading prompt_refiner
_mock_folder_paths = types.ModuleType("folder_paths")
_mock_folder_paths.get_temp_directory = MagicMock(return_value="/tmp")
sys.modules.setdefault("folder_paths", _mock_folder_paths)

# Load the module directly to bypass __init__.py import chain
_spec = importlib.util.spec_from_file_location(
    "prompt_refiner",
    str(__import__("pathlib").Path(__file__).resolve().parent.parent
        / "src" / "KNF_Utils" / "prompt_refiner.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

NV_PromptRefiner = _mod.NV_PromptRefiner
_hash_prompt = _mod._hash_prompt
_hash_media = _mod._hash_media
_format_conversation_log = _mod._format_conversation_log
_CONVERSATION_HISTORY = _mod._CONVERSATION_HISTORY
_LAST_REFINED = _mod._LAST_REFINED

# Import torch for tensor creation in media tests
import torch


def _mr(text):
    """Mock API return: matches the (text, token_info) tuple that API functions return."""
    return (text, {"prompt_tokens": 100, "completion_tokens": 50, "cost": 0.001})


@pytest.fixture(autouse=True)
def clear_global_state():
    """Reset module-level conversation state before each test."""
    _mod._CONVERSATION_HISTORY.clear()
    _mod._LAST_REFINED.clear()
    _mod._LAST_PROMPT_HASH = ""
    _mod._SESSION_COST = 0.0
    yield
    _mod._CONVERSATION_HISTORY.clear()
    _mod._LAST_REFINED.clear()
    _mod._LAST_PROMPT_HASH = ""
    _mod._SESSION_COST = 0.0


@pytest.fixture
def refiner():
    return NV_PromptRefiner()


@pytest.fixture
def default_args():
    return {
        "system_instruction": "You are a video analysis assistant. 80-120 words.",
        "initial_prompt": "Analyze this video and describe the scene.",
        "user_instruction": "Add more detail about the character's armor",
        "api_key": "test-key-123",
        "provider": "Gemini",
        "model": "gemini-2.5-flash",
        "temperature": 0.7,
        "max_tokens": 1500,
        "thinking_level": "auto",
        "clear_history": False,
        "use_cached": False,
    }


# ---------------------------------------------------------------------------
# Node metadata
# ---------------------------------------------------------------------------

def test_node_metadata():
    """Verify node class attributes."""
    assert NV_PromptRefiner.RETURN_TYPES == ("STRING", "STRING", "STRING", "STRING")
    assert NV_PromptRefiner.RETURN_NAMES == ("refined_prompt", "system_instruction", "conversation_log", "debug_info")
    assert NV_PromptRefiner.FUNCTION == "refine"
    assert NV_PromptRefiner.CATEGORY == "NV_Utils/Prompt"


def test_input_types_structure():
    """Verify INPUT_TYPES returns correct structure."""
    input_types = NV_PromptRefiner.INPUT_TYPES()
    required = input_types["required"]
    assert "system_instruction" in required
    assert "initial_prompt" in required
    assert "user_instruction" in required
    assert "api_key" in required
    assert "provider" in required
    assert "model" in required
    assert "temperature" in required
    assert "max_tokens" in required
    assert "thinking_level" in required
    assert "clear_history" in required
    assert "use_cached" in required


def test_is_changed_always_nan():
    """IS_CHANGED returns NaN to force re-execution."""
    import math
    result = NV_PromptRefiner.IS_CHANGED()
    assert math.isnan(result)


# ---------------------------------------------------------------------------
# Hash consistency
# ---------------------------------------------------------------------------

def test_hash_prompt_deterministic():
    """Same input always produces same hash."""
    h1 = _hash_prompt("test prompt")
    h2 = _hash_prompt("test prompt")
    assert h1 == h2


def test_hash_prompt_different_inputs():
    """Different inputs produce different hashes."""
    h1 = _hash_prompt("prompt A")
    h2 = _hash_prompt("prompt B")
    assert h1 != h2


# ---------------------------------------------------------------------------
# Edge cases: empty inputs
# ---------------------------------------------------------------------------

def test_empty_initial_prompt(refiner, default_args):
    """Empty initial_prompt returns error message."""
    default_args["initial_prompt"] = ""
    result = refiner.refine(**default_args)
    assert "no initial prompt" in result[0].lower()


def test_empty_user_instruction(refiner, default_args):
    """Empty user_instruction passes through initial_prompt unchanged."""
    default_args["user_instruction"] = ""
    result = refiner.refine(**default_args)
    assert result[0] == default_args["initial_prompt"]
    assert "pass-through" in result[3].lower()


def test_empty_api_key(refiner, default_args):
    """Empty api_key returns error."""
    default_args["api_key"] = ""
    result = refiner.refine(**default_args)
    assert "no API key" in result[0].lower() or "api_key" in result[3].lower()


# ---------------------------------------------------------------------------
# System instruction passthrough
# ---------------------------------------------------------------------------

def test_system_instruction_passthrough(refiner, default_args):
    """system_instruction is passed through as second output."""
    default_args["user_instruction"] = ""  # passthrough mode
    result = refiner.refine(**default_args)
    assert result[1] == default_args["system_instruction"]


# ---------------------------------------------------------------------------
# Conversation state management
# ---------------------------------------------------------------------------

def test_first_turn_includes_initial_prompt(refiner, default_args):
    """First turn message includes the initial prompt text."""
    with patch.object(_mod, "_call_gemini_text", return_value=_mr("refined text")) as mock:
        refiner.refine(**default_args)
        call_args = mock.call_args
        turns = call_args[0][0]  # first positional arg = conversation_turns
        # First turn should contain the initial prompt
        assert default_args["initial_prompt"] in turns[0]["content"]
        assert default_args["user_instruction"] in turns[0]["content"]


def test_conversation_history_persists(refiner, default_args):
    """Conversation history persists across calls with same initial_prompt."""
    with patch.object(_mod, "_call_gemini_text", return_value=_mr("refined v1")):
        refiner.refine(**default_args)

    # Second call with different instruction
    default_args["user_instruction"] = "Also add cape details"
    with patch.object(_mod, "_call_gemini_text", return_value=_mr("refined v2")) as mock:
        result = refiner.refine(**default_args)
        turns = mock.call_args[0][0]
        # Should have 3 turns: user(initial+inst1), assistant(v1), user(inst2)
        assert len(turns) == 3
        assert turns[0]["role"] == "user"
        assert turns[1]["role"] == "assistant"
        assert turns[1]["content"] == "refined v1"
        assert turns[2]["role"] == "user"
        assert turns[2]["content"] == "Also add cape details"


def test_auto_reset_on_prompt_change(refiner, default_args):
    """Changing initial_prompt clears conversation history."""
    with patch.object(_mod, "_call_gemini_text", return_value=_mr("refined v1")):
        refiner.refine(**default_args)

    # Change initial prompt
    default_args["initial_prompt"] = "Completely different prompt"
    default_args["user_instruction"] = "Refine this"
    with patch.object(_mod, "_call_gemini_text", return_value=_mr("refined v2")) as mock:
        refiner.refine(**default_args)
        turns = mock.call_args[0][0]
        # Should be fresh — only 1 turn (the new first message)
        assert len(turns) == 1
        assert "Completely different prompt" in turns[0]["content"]


def test_clear_history_resets(refiner, default_args):
    """clear_history=True clears conversation."""
    with patch.object(_mod, "_call_gemini_text", return_value=_mr("refined v1")):
        refiner.refine(**default_args)

    # Clear history
    default_args["clear_history"] = True
    with patch.object(_mod, "_call_gemini_text", return_value=_mr("refined v2")) as mock:
        refiner.refine(**default_args)
        turns = mock.call_args[0][0]
        # Should be fresh — only 1 turn
        assert len(turns) == 1


# ---------------------------------------------------------------------------
# Meta-system prompt
# ---------------------------------------------------------------------------

def test_meta_system_includes_system_instruction(refiner, default_args):
    """Meta-system prompt embeds the full pipeline system_instruction."""
    with patch.object(_mod, "_call_gemini_text", return_value=_mr("refined")) as mock:
        refiner.refine(**default_args)
        meta_system = mock.call_args[0][1]  # second positional arg
        assert default_args["system_instruction"] in meta_system
        assert "Pipeline Constraints" in meta_system


# ---------------------------------------------------------------------------
# Provider routing
# ---------------------------------------------------------------------------

def test_gemini_provider_routes_correctly(refiner, default_args):
    """Gemini provider calls _call_gemini_text."""
    default_args["provider"] = "Gemini"
    with patch.object(_mod, "_call_gemini_text", return_value=_mr("refined")) as mock_g, \
         patch.object(_mod, "_call_openrouter_text", return_value=_mr("refined")) as mock_o:
        refiner.refine(**default_args)
        assert mock_g.called
        assert not mock_o.called


def test_openrouter_provider_routes_correctly(refiner, default_args):
    """OpenRouter provider calls _call_openrouter_text."""
    default_args["provider"] = "OpenRouter"
    with patch.object(_mod, "_call_gemini_text", return_value=_mr("refined")) as mock_g, \
         patch.object(_mod, "_call_openrouter_text", return_value=_mr("refined")) as mock_o:
        refiner.refine(**default_args)
        assert not mock_g.called
        assert mock_o.called


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_api_error_returns_fallback(refiner, default_args):
    """API error returns last refined or initial prompt."""
    with patch.object(_mod, "_call_gemini_text", side_effect=RuntimeError("API error")):
        result = refiner.refine(**default_args)
        # Should return initial_prompt as fallback
        assert result[0] == default_args["initial_prompt"]
        assert "Error" in result[3]


def test_api_error_preserves_history(refiner, default_args):
    """API error on turn 2 doesn't corrupt history from turn 1."""
    with patch.object(_mod, "_call_gemini_text", return_value=_mr("refined v1")):
        refiner.refine(**default_args)

    default_args["user_instruction"] = "This will fail"
    with patch.object(_mod, "_call_gemini_text", side_effect=RuntimeError("timeout")):
        result = refiner.refine(**default_args)
        # Should return v1 as fallback
        assert result[0] == "refined v1"
        # History should still have 2 turns from successful call, not 3
        prompt_hash = _hash_prompt(default_args["initial_prompt"])
        history = _mod._CONVERSATION_HISTORY.get(prompt_hash, [])
        assert len(history) == 2  # user + assistant from turn 1


# ---------------------------------------------------------------------------
# Conversation log formatting
# ---------------------------------------------------------------------------

def test_format_conversation_log_empty():
    """Empty history returns placeholder."""
    assert "no conversation" in _format_conversation_log([]).lower()


def test_format_conversation_log_multi_turn():
    """Multi-turn history formats with turn numbers."""
    turns = [
        {"role": "user", "content": "first instruction"},
        {"role": "assistant", "content": "first response"},
        {"role": "user", "content": "second instruction"},
        {"role": "assistant", "content": "second response"},
    ]
    log = _format_conversation_log(turns)
    assert "Turn 1" in log
    assert "Turn 2" in log
    assert "first instruction" in log
    assert "second response" in log


# ---------------------------------------------------------------------------
# Debug info
# ---------------------------------------------------------------------------

def test_debug_info_content(refiner, default_args):
    """Debug info contains key diagnostics."""
    with patch.object(_mod, "_call_gemini_text", return_value=_mr("refined text")):
        result = refiner.refine(**default_args)
        debug = result[3]
        assert "Turn:" in debug
        assert "Provider:" in debug
        assert "Model:" in debug
        assert "Prompt Hash:" in debug
        assert "Refined Length:" in debug


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------

def test_output_tuple_length(refiner, default_args):
    """refine() returns 4-tuple."""
    with patch.object(_mod, "_call_gemini_text", return_value=_mr("refined")):
        result = refiner.refine(**default_args)
        assert len(result) == 4
        assert all(isinstance(s, str) for s in result)


# ---------------------------------------------------------------------------
# use_cached feature
# ---------------------------------------------------------------------------

def test_use_cached_skips_api_call(refiner, default_args):
    """use_cached=True with existing cache returns cached result without API call."""
    # First call: populate cache
    with patch.object(_mod, "_call_gemini_text", return_value=_mr("cached caption")):
        refiner.refine(**default_args)

    # Second call with use_cached=True
    default_args["use_cached"] = True
    with patch.object(_mod, "_call_gemini_text", return_value=_mr("should not be called")) as mock:
        result = refiner.refine(**default_args)
        assert not mock.called
        assert result[0] == "cached caption"
        assert "cached" in result[3].lower()


def test_use_cached_no_cache_still_calls_api(refiner, default_args):
    """use_cached=True with empty cache calls API normally."""
    default_args["use_cached"] = True
    # _LAST_REFINED is empty → should call API
    with patch.object(_mod, "_call_gemini_text", return_value=_mr("fresh result")) as mock:
        result = refiner.refine(**default_args)
        assert mock.called
        assert result[0] == "fresh result"


def test_use_cached_false_always_calls_api(refiner, default_args):
    """use_cached=False always calls API even with cached results."""
    with patch.object(_mod, "_call_gemini_text", return_value=_mr("v1")):
        refiner.refine(**default_args)

    default_args["use_cached"] = False
    default_args["user_instruction"] = "Refine more"
    with patch.object(_mod, "_call_gemini_text", return_value=_mr("v2")) as mock:
        result = refiner.refine(**default_args)
        assert mock.called
        assert result[0] == "v2"


# ---------------------------------------------------------------------------
# Cost tracking
# ---------------------------------------------------------------------------

def test_cost_in_debug_info(refiner, default_args):
    """Debug info shows cost information."""
    with patch.object(_mod, "_call_gemini_text", return_value=_mr("refined")):
        result = refiner.refine(**default_args)
        debug = result[3]
        assert "Cost:" in debug
        assert "this turn" in debug
        assert "session total" in debug


def test_session_cost_accumulates(refiner, default_args):
    """Session cost accumulates across multiple turns."""
    assert _mod._SESSION_COST == 0.0

    with patch.object(_mod, "_call_gemini_text", return_value=_mr("v1")):
        refiner.refine(**default_args)
    cost_after_1 = _mod._SESSION_COST

    default_args["user_instruction"] = "Refine again"
    with patch.object(_mod, "_call_gemini_text", return_value=_mr("v2")):
        refiner.refine(**default_args)
    cost_after_2 = _mod._SESSION_COST

    # Each call adds 0.001 (from _mr helper)
    assert cost_after_1 > 0.0
    assert cost_after_2 > cost_after_1


def test_tokens_in_debug_info(refiner, default_args):
    """Debug info shows token counts."""
    with patch.object(_mod, "_call_gemini_text", return_value=_mr("refined")):
        result = refiner.refine(**default_args)
        debug = result[3]
        assert "Tokens:" in debug
        assert "in /" in debug
        assert "out" in debug


# ===========================================================================
# MEDIA TESTS (Phase 3)
# ===========================================================================

# ---------------------------------------------------------------------------
# Media hash helpers
# ---------------------------------------------------------------------------

def test_hash_media_none_returns_empty():
    """_hash_media(None) returns empty string."""
    assert _hash_media(None) == ""


def test_hash_media_different_tensors():
    """Different tensors produce different hashes."""
    t1 = torch.rand(4, 64, 64, 3)
    t2 = torch.rand(4, 64, 64, 3)
    assert _hash_media(t1) != _hash_media(t2)


def test_hash_media_same_tensor():
    """Same tensor produces same hash."""
    t1 = torch.ones(4, 64, 64, 3)
    assert _hash_media(t1) == _hash_media(t1)


# ---------------------------------------------------------------------------
# Optional media inputs
# ---------------------------------------------------------------------------

def test_input_types_has_optional_media():
    """INPUT_TYPES includes optional media inputs."""
    it = NV_PromptRefiner.INPUT_TYPES()
    assert "optional" in it, "Missing 'optional' key"
    opt = it["optional"]
    assert "video_tensor" in opt
    assert "image_tensor" in opt
    assert "fps" in opt
    assert "media_resolution" in opt


# ---------------------------------------------------------------------------
# Mode selection (captioner vs refiner)
# ---------------------------------------------------------------------------

def test_text_only_mode_unchanged(refiner, default_args):
    """No media = text-only refiner mode, uses refinement system prompt."""
    with patch.object(_mod, "_call_gemini_text", return_value=_mr("refined text")) as mock:
        refiner.refine(**default_args)
        meta_system = mock.call_args[0][1]
        assert "refinement assistant" in meta_system.lower()
        assert "captioning assistant" not in meta_system.lower()
        assert mock.call_args.kwargs.get("media") is None


def test_media_uses_captioner_system_prompt(refiner, default_args):
    """With media, meta-system prompt says 'captioning assistant'."""
    default_args["image_tensor"] = torch.rand(1, 64, 64, 3)
    with patch.object(_mod, "_prepare_media", return_value=("base64data", "image/jpeg", "image 64x64")), \
         patch.object(_mod, "_call_gemini_text", return_value=_mr("caption text")) as mock:
        refiner.refine(**default_args)
        meta_system = mock.call_args[0][1]
        assert "captioning assistant" in meta_system.lower()
        assert "refinement assistant" not in meta_system.lower()


def test_no_media_uses_refiner_system_prompt(refiner, default_args):
    """Without media, meta-system says 'refinement assistant'."""
    with patch.object(_mod, "_call_gemini_text", return_value=_mr("refined")) as mock:
        refiner.refine(**default_args)
        meta_system = mock.call_args[0][1]
        assert "refinement assistant" in meta_system.lower()


def test_first_turn_captioner_framing(refiner, default_args):
    """With media, first turn says 'Captioning guidelines' not 'prompt to refine'."""
    default_args["image_tensor"] = torch.rand(1, 64, 64, 3)
    with patch.object(_mod, "_prepare_media", return_value=("base64data", "image/jpeg", "image 64x64")), \
         patch.object(_mod, "_call_gemini_text", return_value=_mr("caption text")) as mock:
        refiner.refine(**default_args)
        turns = mock.call_args[0][0]
        assert "Captioning guidelines" in turns[0]["content"]
        assert "prompt to refine" not in turns[0]["content"]


# ---------------------------------------------------------------------------
# Media routing to API calls
# ---------------------------------------------------------------------------

def test_media_passed_to_gemini_api(refiner, default_args):
    """Media tuple is passed to _call_gemini_text."""
    default_args["image_tensor"] = torch.rand(1, 64, 64, 3)
    with patch.object(_mod, "_prepare_media", return_value=("b64data", "image/jpeg", "info")), \
         patch.object(_mod, "_call_gemini_text", return_value=_mr("caption")) as mock:
        refiner.refine(**default_args)
        assert mock.call_args.kwargs["media"] == ("b64data", "image/jpeg")


def test_media_passed_to_openrouter_api(refiner, default_args):
    """Media tuple is passed to _call_openrouter_text."""
    default_args["provider"] = "OpenRouter"
    default_args["image_tensor"] = torch.rand(1, 64, 64, 3)
    with patch.object(_mod, "_prepare_media", return_value=("b64data", "image/jpeg", "info")), \
         patch.object(_mod, "_call_openrouter_text", return_value=_mr("caption")) as mock:
        refiner.refine(**default_args)
        assert mock.call_args.kwargs["media"] == ("b64data", "image/jpeg")


def test_image_priority_over_video(refiner, default_args):
    """When both image and video provided, _prepare_media receives image first."""
    img = torch.rand(1, 64, 64, 3)
    vid = torch.rand(4, 64, 64, 3)
    default_args["image_tensor"] = img
    default_args["video_tensor"] = vid
    with patch.object(_mod, "_prepare_media", return_value=("b64", "image/jpeg", "img")) as mock_prep, \
         patch.object(_mod, "_call_gemini_text", return_value=_mr("caption")):
        refiner.refine(**default_args)
        call_args = mock_prep.call_args[0]
        assert call_args[0] is img  # image_tensor
        assert call_args[1] is vid  # video_tensor


# ---------------------------------------------------------------------------
# Conversation state with media
# ---------------------------------------------------------------------------

def test_media_change_resets_conversation(refiner, default_args):
    """Different media tensor triggers conversation reset."""
    # Turn 1 with tensor A
    tensor_a = torch.zeros(1, 32, 32, 3)
    default_args["image_tensor"] = tensor_a
    with patch.object(_mod, "_prepare_media", return_value=("b64a", "image/jpeg", "a")), \
         patch.object(_mod, "_call_gemini_text", return_value=_mr("caption v1")):
        refiner.refine(**default_args)

    # Turn 2 with different tensor B (different shape = different hash)
    tensor_b = torch.ones(1, 64, 64, 3)
    default_args["image_tensor"] = tensor_b
    default_args["user_instruction"] = "Different instruction"
    with patch.object(_mod, "_prepare_media", return_value=("b64b", "image/jpeg", "b")), \
         patch.object(_mod, "_call_gemini_text", return_value=_mr("caption v2")) as mock:
        refiner.refine(**default_args)
        turns = mock.call_args[0][0]
        # Should be fresh — only 1 turn (auto-reset due to media change)
        assert len(turns) == 1


def test_media_preparation_failure_graceful(refiner, default_args):
    """Media preparation failure falls back to text-only mode."""
    default_args["image_tensor"] = torch.rand(1, 64, 64, 3)
    with patch.object(_mod, "_prepare_media", side_effect=RuntimeError("cv2 failed")), \
         patch.object(_mod, "_call_gemini_text", return_value=_mr("refined text")) as mock:
        result = refiner.refine(**default_args)
        # Should still succeed (text-only fallback)
        assert result[0] == "refined text"
        # Media kwarg should be None (preparation failed)
        assert mock.call_args.kwargs.get("media") is None
        # Should use text-mode system prompt
        meta_system = mock.call_args[0][1]
        assert "refinement assistant" in meta_system.lower()


# ---------------------------------------------------------------------------
# Debug info with media
# ---------------------------------------------------------------------------

def test_media_in_debug_info(refiner, default_args):
    """Debug info shows Media line when media connected."""
    default_args["image_tensor"] = torch.rand(1, 64, 64, 3)
    with patch.object(_mod, "_prepare_media", return_value=("b64", "image/jpeg", "image 64x64 (jpeg), 0.1MB base64")), \
         patch.object(_mod, "_call_gemini_text", return_value=_mr("caption")):
        result = refiner.refine(**default_args)
        assert "Media:" in result[3]
        assert "image 64x64" in result[3]


def test_mode_in_debug_info(refiner, default_args):
    """Debug info shows Mode: captioner or refiner."""
    # Text-only = refiner
    with patch.object(_mod, "_call_gemini_text", return_value=_mr("refined")):
        result = refiner.refine(**default_args)
        assert "Mode: refiner" in result[3]
