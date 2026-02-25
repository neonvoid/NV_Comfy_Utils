"""Tests for NV_V2VPromptBuilder - parameterized V2V LoRA captioning prompt assembly."""

import pytest
from src.KNF_Utils.v2v_prompt_builder import (
    NV_V2VPromptBuilder,
    _build_character_table,
    _select_motion_clause,
    _select_denoise_notes,
    _select_budget_row_label,
    _build_chunked_section,
)


@pytest.fixture
def builder():
    return NV_V2VPromptBuilder()


@pytest.fixture
def default_required():
    return {
        "task_mode": "full_restyle",
        "trigger_word": "ohwx",
        "style_description": "cinematic film grain with warm tones",
        "subject_count": 1,
        "motion_intensity": "medium",
        "denoise_strength": 0.65,
        "word_count_min": 80,
        "word_count_max": 120,
    }


@pytest.fixture
def cs_required():
    """Default required args for character_swap mode."""
    return {
        "task_mode": "character_swap",
        "trigger_word": "GO1DNITE",
        "style_description": "",
        "subject_count": 1,
        "motion_intensity": "high",
        "denoise_strength": 1.0,
        "word_count_min": 200,
        "word_count_max": 300,
    }


# ---------------------------------------------------------------------------
# Basic output structure
# ---------------------------------------------------------------------------

def test_basic_output_structure(builder, default_required):
    """build_prompt returns 3 non-empty strings."""
    result = builder.build_prompt(**default_required)
    assert len(result) == 3
    system_instruction, prompt_text, debug_info = result
    assert isinstance(system_instruction, str) and len(system_instruction) > 0
    assert isinstance(prompt_text, str) and len(prompt_text) > 0
    assert isinstance(debug_info, str) and len(debug_info) > 0


def test_node_metadata():
    """Verify node class attributes match expected values."""
    assert NV_V2VPromptBuilder.RETURN_TYPES == ("STRING", "STRING", "STRING")
    assert NV_V2VPromptBuilder.RETURN_NAMES == ("system_instruction", "prompt_text", "debug_info")
    assert NV_V2VPromptBuilder.FUNCTION == "build_prompt"
    assert NV_V2VPromptBuilder.CATEGORY == "NV_Utils/Prompt"


def test_input_types_structure():
    """Verify INPUT_TYPES returns correct structure."""
    input_types = NV_V2VPromptBuilder.INPUT_TYPES()
    required = input_types["required"]
    optional = input_types["optional"]

    assert "task_mode" in required
    assert "trigger_word" in required
    assert "style_description" in required
    assert "subject_count" in required
    assert "motion_intensity" in required
    assert "denoise_strength" in required
    assert "word_count_min" in required
    assert "word_count_max" in required

    assert "scene_subjects" in optional
    assert "scene_setting" in optional
    assert "scene_props" in optional
    assert "character_tokens" in optional
    assert "camera_behavior" in optional
    assert "video_duration" in optional
    assert "processing_mode" in optional
    assert "previous_chunk_prompt" in optional
    assert "chunk_index" in optional
    assert "custom_system_override" in optional


# ---------------------------------------------------------------------------
# Trigger word
# ---------------------------------------------------------------------------

def test_trigger_word_in_system_instruction(builder, default_required):
    """Trigger word appears in system_instruction output format."""
    system_instruction, _, _ = builder.build_prompt(**default_required)
    assert "ohwx" in system_instruction


def test_trigger_word_empty(builder, default_required):
    """Empty trigger word produces 'No trigger word' language."""
    default_required["trigger_word"] = ""
    system_instruction, prompt_text, _ = builder.build_prompt(**default_required)
    assert "No trigger word" in system_instruction or "no trigger word" in system_instruction.lower()
    assert "(none)" in prompt_text


# ---------------------------------------------------------------------------
# Denoise modes
# ---------------------------------------------------------------------------

def test_denoise_preservation_mode(builder, default_required):
    """Low denoise (<0.5) triggers structural fidelity emphasis."""
    default_required["denoise_strength"] = 0.3
    system_instruction, _, debug_info = builder.build_prompt(**default_required)
    assert "structural fidelity" in system_instruction.lower() or "preservation" in debug_info.lower()
    assert "preservation" in debug_info


def test_denoise_balanced_mode(builder, default_required):
    """Mid denoise (0.5-0.7) has no extra clause."""
    default_required["denoise_strength"] = 0.6
    _, _, debug_info = builder.build_prompt(**default_required)
    assert "balanced" in debug_info


def test_denoise_creative_mode(builder, default_required):
    """High denoise (>0.7) triggers style transformation emphasis."""
    default_required["denoise_strength"] = 0.85
    system_instruction, _, debug_info = builder.build_prompt(**default_required)
    assert "style" in system_instruction.lower() or "creative" in debug_info.lower()
    assert "creative" in debug_info


# ---------------------------------------------------------------------------
# Character tokens
# ---------------------------------------------------------------------------

def test_character_table_parsing(builder, default_required):
    """Character tokens parsed into markdown table in system_instruction."""
    system_instruction, _, _ = builder.build_prompt(
        **default_required,
        character_tokens="ohwx_man: tall man with beard\nsks_woman: short woman with red hair",
    )
    assert "ohwx_man" in system_instruction
    assert "sks_woman" in system_instruction
    assert "tall man with beard" in system_instruction
    assert "Character Recognition" in system_instruction


def test_character_table_pipe_separator():
    """Pipe separator also works for character token parsing."""
    result = _build_character_table("S1LV3N1TE | Knight with fur mantle\nG0lDNITE | Knight in red armor")
    assert "S1LV3N1TE" in result
    assert "G0lDNITE" in result
    assert "Knight with fur mantle" in result


def test_character_table_empty(builder, default_required):
    """No character tokens = no Character Recognition section."""
    system_instruction, _, _ = builder.build_prompt(**default_required)
    assert "Character Recognition" not in system_instruction


def test_character_table_malformed_lines():
    """Malformed lines (no separator) are skipped."""
    result = _build_character_table("good_token: good description\nbad line without separator\nalso_good: works")
    assert "good_token" in result
    assert "also_good" in result
    assert "bad line" not in result


# ---------------------------------------------------------------------------
# Word budget
# ---------------------------------------------------------------------------

def test_budget_row_single_low():
    """Single subject + low motion selects correct row."""
    label = _select_budget_row_label(1, "low")
    assert "1 subject" in label and "low" in label


def test_budget_row_single_high():
    """Single subject + high motion selects correct row."""
    label = _select_budget_row_label(1, "high")
    assert "1 subject" in label and "high" in label


def test_budget_row_multi_low():
    """Multiple subjects + low motion selects correct row."""
    label = _select_budget_row_label(3, "low")
    assert "2+" in label and "low" in label


def test_budget_row_multi_high():
    """Multiple subjects + high motion selects correct row."""
    label = _select_budget_row_label(2, "high")
    assert "2+" in label and "high" in label


def test_budget_row_environment_only():
    """Zero subjects = environment only."""
    label = _select_budget_row_label(0, "medium")
    assert "environment" in label.lower()


def test_budget_row_in_system_instruction(builder, default_required):
    """Active budget row label appears in system_instruction."""
    default_required["subject_count"] = 3
    default_required["motion_intensity"] = "high"
    system_instruction, _, _ = builder.build_prompt(**default_required)
    assert "2+ subjects, high motion" in system_instruction


# ---------------------------------------------------------------------------
# Motion clauses
# ---------------------------------------------------------------------------

def test_motion_clause_low():
    """Low motion clause mentions stillness/subtle."""
    clause = _select_motion_clause("low")
    assert "subtle" in clause.lower() or "stillness" in clause.lower()


def test_motion_clause_high():
    """High motion clause mentions trajectories/dynamic."""
    clause = _select_motion_clause("high")
    assert "trajectories" in clause.lower() or "dynamic" in clause.lower()


# ---------------------------------------------------------------------------
# Chunked processing
# ---------------------------------------------------------------------------

def test_chunked_first_chunk(builder, default_required):
    """Chunk 0 in chunked mode mentions 'establishing baseline'."""
    _, prompt_text, _ = builder.build_prompt(
        **default_required,
        processing_mode="chunked",
        chunk_index=0,
    )
    assert "establishing baseline" in prompt_text.lower() or "first chunk" in prompt_text.lower()


def test_chunked_continuation(builder, default_required):
    """Continuation chunk includes previous chunk caption."""
    prev = "A woman walks through an autumn forest clearing."
    _, prompt_text, _ = builder.build_prompt(
        **default_required,
        processing_mode="chunked",
        chunk_index=2,
        previous_chunk_prompt=prev,
    )
    assert "Previous Chunk Caption" in prompt_text
    assert prev in prompt_text
    assert "continuation" in prompt_text.lower()


def test_chunked_no_context(builder, default_required):
    """Chunk > 0 with empty previous prompt mentions no previous context."""
    _, prompt_text, _ = builder.build_prompt(
        **default_required,
        processing_mode="chunked",
        chunk_index=3,
        previous_chunk_prompt="",
    )
    assert "no previous context" in prompt_text.lower()


def test_single_mode_no_chunked_section(builder, default_required):
    """Single mode produces no chunked processing text."""
    _, prompt_text, _ = builder.build_prompt(
        **default_required,
        processing_mode="single",
    )
    assert "Processing Mode" not in prompt_text


# ---------------------------------------------------------------------------
# Custom system override
# ---------------------------------------------------------------------------

def test_custom_system_override(builder, default_required):
    """Custom override replaces entire system instruction."""
    custom = "You are a custom video analyst. Describe everything."
    system_instruction, _, debug_info = builder.build_prompt(
        **default_required,
        custom_system_override=custom,
    )
    assert system_instruction == custom
    assert "yes" in debug_info.lower().split("system override:")[1].split("\n")[0]


def test_custom_system_override_with_placeholders(builder, default_required):
    """Custom override supports {trigger_word} placeholder substitution."""
    custom = "Use the style token {trigger_word} in every caption. Target {word_count_min}-{word_count_max} words."
    system_instruction, _, _ = builder.build_prompt(
        **default_required,
        custom_system_override=custom,
    )
    assert "Use the style token ohwx" in system_instruction
    assert "Target 80-120 words" in system_instruction


# ---------------------------------------------------------------------------
# Word count range
# ---------------------------------------------------------------------------

def test_word_count_range_in_outputs(builder, default_required):
    """Word count range appears in both system_instruction and prompt_text."""
    default_required["word_count_min"] = 100
    default_required["word_count_max"] = 150
    system_instruction, prompt_text, _ = builder.build_prompt(**default_required)
    assert "100-150" in system_instruction
    assert "100-150" in prompt_text


def test_word_count_swap_when_inverted(builder, default_required):
    """Inverted min/max gets auto-swapped."""
    default_required["word_count_min"] = 200
    default_required["word_count_max"] = 100
    system_instruction, prompt_text, _ = builder.build_prompt(**default_required)
    # After swap, should be 100-200
    assert "100-200" in system_instruction
    assert "100-200" in prompt_text


# ---------------------------------------------------------------------------
# Optional input fallbacks
# ---------------------------------------------------------------------------

def test_empty_optional_fallbacks(builder, default_required):
    """Empty optional inputs use fallback text in prompt_text."""
    _, prompt_text, _ = builder.build_prompt(**default_required)
    assert "observe and describe" in prompt_text.lower()


def test_scene_context_populated(builder, default_required):
    """Populated scene context appears in prompt_text."""
    _, prompt_text, _ = builder.build_prompt(
        **default_required,
        scene_subjects="a dog playing fetch",
        scene_setting="suburban park with oak trees",
        scene_props="tennis ball, red collar",
    )
    assert "a dog playing fetch" in prompt_text
    assert "suburban park with oak trees" in prompt_text
    assert "tennis ball, red collar" in prompt_text


# ---------------------------------------------------------------------------
# Debug info
# ---------------------------------------------------------------------------

def test_debug_info_contains_all_params(builder, default_required):
    """Debug info contains key parameter summaries."""
    _, _, debug_info = builder.build_prompt(**default_required)
    assert "Trigger Word:" in debug_info
    assert "ohwx" in debug_info
    assert "Style:" in debug_info
    assert "Subjects:" in debug_info
    assert "Motion:" in debug_info
    assert "Denoise:" in debug_info
    assert "Word Count:" in debug_info
    assert "Budget Row:" in debug_info
    assert "Denoise Mode:" in debug_info
    assert "Processing:" in debug_info
    assert "System Instruction Length:" in debug_info
    assert "Prompt Text Length:" in debug_info


# ---------------------------------------------------------------------------
# Helper function unit tests
# ---------------------------------------------------------------------------

def test_build_character_table_empty():
    assert _build_character_table("") == ""
    assert _build_character_table(None) == ""
    assert _build_character_table("   \n  ") == ""


def test_select_denoise_notes_boundaries():
    """Test boundary values for denoise mode selection."""
    assert _select_denoise_notes(0.0)["mode_label"] == "preservation"
    assert _select_denoise_notes(0.49)["mode_label"] == "preservation"
    assert _select_denoise_notes(0.5)["mode_label"] == "balanced"
    assert _select_denoise_notes(0.7)["mode_label"] == "balanced"
    assert _select_denoise_notes(0.71)["mode_label"] == "creative"
    assert _select_denoise_notes(1.0)["mode_label"] == "creative"


def test_build_chunked_section_single():
    """Single mode returns empty string."""
    assert _build_chunked_section("single", 0, "") == ""


# ---------------------------------------------------------------------------
# Task mode: character_swap
# ---------------------------------------------------------------------------

def test_character_swap_system_role(builder, cs_required):
    """character_swap system role mentions character replacement / inpainting."""
    system_instruction, _, _ = builder.build_prompt(**cs_required)
    assert "character replacement" in system_instruction.lower() or "inpainting" in system_instruction.lower()


def test_character_swap_word_budget_columns(builder, cs_required):
    """character_swap word budget uses Character/Pose columns, not Environment/Materials."""
    system_instruction, _, _ = builder.build_prompt(**cs_required)
    assert "Character" in system_instruction
    assert "Pose/Action" in system_instruction


def test_character_swap_output_format_ordering(builder, cs_required):
    """character_swap output format leads with character identity."""
    system_instruction, _, _ = builder.build_prompt(**cs_required)
    # Character identity should appear before scene context in the output format ordering
    char_pos = system_instruction.find("Character identity")
    scene_pos = system_instruction.find("Scene context for spatial")
    assert char_pos != -1 and scene_pos != -1
    assert char_pos < scene_pos


def test_character_swap_trigger_word_in_system(builder, cs_required):
    """Trigger word appears in character_swap system instruction."""
    system_instruction, _, _ = builder.build_prompt(**cs_required)
    assert "GO1DNITE" in system_instruction


def test_character_swap_prompt_has_replace_target(builder, cs_required):
    """character_swap prompt template uses 'Replace Target' label."""
    _, prompt_text, _ = builder.build_prompt(
        **cs_required,
        scene_subjects="blonde man in blank tank top",
    )
    assert "Replace Target" in prompt_text
    assert "blonde man in blank tank top" in prompt_text


def test_character_swap_prompt_says_character_replacement(builder, cs_required):
    """character_swap prompt template says 'character replacement caption'."""
    _, prompt_text, _ = builder.build_prompt(**cs_required)
    assert "character replacement" in prompt_text.lower()


def test_character_swap_character_tokens(builder, cs_required):
    """character_swap uses 'Character Replacement Tokens' section header."""
    system_instruction, _, _ = builder.build_prompt(
        **cs_required,
        character_tokens="GO1DNITE: Knight in golden armor with red cape",
    )
    assert "Character Replacement Tokens" in system_instruction
    assert "GO1DNITE" in system_instruction
    assert "golden armor" in system_instruction


def test_character_swap_no_tokens_no_section(builder, cs_required):
    """No character tokens = no Character Replacement Tokens section."""
    system_instruction, _, _ = builder.build_prompt(**cs_required)
    assert "Character Replacement Tokens" not in system_instruction


def test_character_swap_debug_info_shows_mode(builder, cs_required):
    """Debug info shows task mode for character_swap."""
    _, _, debug_info = builder.build_prompt(**cs_required)
    assert "Task Mode: character_swap" in debug_info


def test_full_restyle_debug_info_shows_mode(builder, default_required):
    """Debug info shows task mode for full_restyle."""
    _, _, debug_info = builder.build_prompt(**default_required)
    assert "Task Mode: full_restyle" in debug_info


def test_full_restyle_unchanged_regression(builder, default_required):
    """full_restyle mode still produces V2V style transfer language."""
    system_instruction, prompt_text, _ = builder.build_prompt(**default_required)
    assert "style LoRA" in system_instruction.lower() or "style transfer" in system_instruction.lower()
    assert "V2V caption" in prompt_text


def test_character_swap_chunked_works(builder, cs_required):
    """Chunked processing works with character_swap mode."""
    _, prompt_text, _ = builder.build_prompt(
        **cs_required,
        processing_mode="chunked",
        chunk_index=0,
    )
    assert "establishing baseline" in prompt_text.lower() or "first chunk" in prompt_text.lower()
