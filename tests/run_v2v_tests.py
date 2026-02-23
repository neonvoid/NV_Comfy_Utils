"""Standalone test runner for v2v_prompt_builder (no torch/pytest dependency)."""
import importlib.util
import sys
import os

# Load module directly, bypassing package __init__.py (avoids torch dependency)
spec = importlib.util.spec_from_file_location(
    "v2v_prompt_builder",
    os.path.join(os.path.dirname(__file__), "..", "src", "KNF_Utils", "v2v_prompt_builder.py"),
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

NV_V2VPromptBuilder = mod.NV_V2VPromptBuilder
_build_character_table = mod._build_character_table
_select_motion_clause = mod._select_motion_clause
_select_denoise_notes = mod._select_denoise_notes
_select_budget_row_label = mod._select_budget_row_label
_build_chunked_section = mod._build_chunked_section

# --- Test infrastructure ---
passed = 0
failed = 0
errors = []


def run_test(name, fn):
    global passed, failed
    try:
        fn()
        passed += 1
        print(f"  PASS: {name}")
    except Exception as e:
        failed += 1
        errors.append((name, e))
        print(f"  FAIL: {name} - {e}")


builder = NV_V2VPromptBuilder()
default_req = {
    "trigger_word": "ohwx",
    "style_description": "cinematic film grain with warm tones",
    "subject_count": 1,
    "motion_intensity": "medium",
    "denoise_strength": 0.65,
    "word_count_min": 80,
    "word_count_max": 120,
}

print("Running NV_V2VPromptBuilder tests...")
print()

# === Basic structure ===


def test_basic_output_structure():
    result = builder.build_prompt(**default_req)
    assert len(result) == 3
    assert all(isinstance(r, str) and len(r) > 0 for r in result)


run_test("basic_output_structure", test_basic_output_structure)


def test_node_metadata():
    assert NV_V2VPromptBuilder.RETURN_TYPES == ("STRING", "STRING", "STRING")
    assert NV_V2VPromptBuilder.FUNCTION == "build_prompt"
    assert NV_V2VPromptBuilder.CATEGORY == "NV_Utils/Prompt"


run_test("node_metadata", test_node_metadata)


def test_input_types():
    it = NV_V2VPromptBuilder.INPUT_TYPES()
    for k in ["trigger_word", "style_description", "subject_count",
              "motion_intensity", "denoise_strength", "word_count_min", "word_count_max"]:
        assert k in it["required"], f"missing required: {k}"
    for k in ["scene_subjects", "scene_setting", "scene_props", "character_tokens",
              "camera_behavior", "video_duration", "processing_mode",
              "previous_chunk_prompt", "chunk_index", "custom_system_override"]:
        assert k in it["optional"], f"missing optional: {k}"


run_test("input_types", test_input_types)

# === Trigger word ===


def test_trigger_word_present():
    si, _, _ = builder.build_prompt(**default_req)
    assert "ohwx" in si


run_test("trigger_word_present", test_trigger_word_present)


def test_trigger_word_empty():
    r = {**default_req, "trigger_word": ""}
    si, pt, _ = builder.build_prompt(**r)
    assert "no trigger word" in si.lower()
    assert "(none)" in pt


run_test("trigger_word_empty", test_trigger_word_empty)

# === Denoise ===


def test_denoise_preservation():
    r = {**default_req, "denoise_strength": 0.3}
    _, _, di = builder.build_prompt(**r)
    assert "preservation" in di


run_test("denoise_preservation", test_denoise_preservation)


def test_denoise_balanced():
    r = {**default_req, "denoise_strength": 0.6}
    _, _, di = builder.build_prompt(**r)
    assert "balanced" in di


run_test("denoise_balanced", test_denoise_balanced)


def test_denoise_creative():
    r = {**default_req, "denoise_strength": 0.85}
    _, _, di = builder.build_prompt(**r)
    assert "creative" in di


run_test("denoise_creative", test_denoise_creative)

# === Character tokens ===


def test_char_table_parsing():
    si, _, _ = builder.build_prompt(
        **default_req,
        character_tokens="ohwx_man: tall man\nsks_woman: short woman",
    )
    assert "ohwx_man" in si and "sks_woman" in si
    assert "Character Recognition" in si


run_test("char_table_parsing", test_char_table_parsing)


def test_char_table_pipe():
    r = _build_character_table("S1LV3N1TE | Knight with fur\nG0lDNITE | Knight in red")
    assert "S1LV3N1TE" in r and "G0lDNITE" in r


run_test("char_table_pipe", test_char_table_pipe)


def test_char_table_empty():
    si, _, _ = builder.build_prompt(**default_req)
    assert "Character Recognition" not in si


run_test("char_table_empty", test_char_table_empty)


def test_char_table_malformed():
    r = _build_character_table("good: desc\nbad line\nalso_good: works")
    assert "good" in r and "also_good" in r and "bad line" not in r


run_test("char_table_malformed", test_char_table_malformed)

# === Budget rows ===


def test_budget_single_low():
    assert "low" in _select_budget_row_label(1, "low")


run_test("budget_single_low", test_budget_single_low)


def test_budget_multi_high():
    assert "2+" in _select_budget_row_label(2, "high")


run_test("budget_multi_high", test_budget_multi_high)


def test_budget_env_only():
    assert "environment" in _select_budget_row_label(0, "medium").lower()


run_test("budget_env_only", test_budget_env_only)


def test_budget_in_system():
    r = {**default_req, "subject_count": 3, "motion_intensity": "high"}
    si, _, _ = builder.build_prompt(**r)
    assert "2+ subjects, high motion" in si


run_test("budget_in_system", test_budget_in_system)

# === Motion ===


def test_motion_low():
    c = _select_motion_clause("low").lower()
    assert "subtle" in c or "stillness" in c


run_test("motion_low", test_motion_low)


def test_motion_high():
    c = _select_motion_clause("high").lower()
    assert "trajectories" in c or "dynamic" in c


run_test("motion_high", test_motion_high)

# === Chunked ===


def test_chunked_first():
    _, pt, _ = builder.build_prompt(**default_req, processing_mode="chunked", chunk_index=0)
    assert "first chunk" in pt.lower() or "establishing baseline" in pt.lower()


run_test("chunked_first", test_chunked_first)


def test_chunked_continuation():
    prev = "A woman walks through forest."
    _, pt, _ = builder.build_prompt(
        **default_req,
        processing_mode="chunked",
        chunk_index=2,
        previous_chunk_prompt=prev,
    )
    assert prev in pt and "continuation" in pt.lower()


run_test("chunked_continuation", test_chunked_continuation)


def test_chunked_no_context():
    _, pt, _ = builder.build_prompt(
        **default_req,
        processing_mode="chunked",
        chunk_index=3,
        previous_chunk_prompt="",
    )
    assert "no previous context" in pt.lower()


run_test("chunked_no_context", test_chunked_no_context)


def test_single_no_chunk():
    _, pt, _ = builder.build_prompt(**default_req, processing_mode="single")
    assert "Processing Mode" not in pt


run_test("single_no_chunk", test_single_no_chunk)

# === Custom override ===


def test_custom_override():
    custom = "Custom instructions here."
    si, _, _ = builder.build_prompt(**default_req, custom_system_override=custom)
    assert si == custom


run_test("custom_override", test_custom_override)


def test_custom_override_placeholders():
    custom = "Use {trigger_word}. Target {word_count_min}-{word_count_max} words."
    si, _, _ = builder.build_prompt(**default_req, custom_system_override=custom)
    assert "Use ohwx" in si and "80-120" in si


run_test("custom_override_placeholders", test_custom_override_placeholders)

# === Word count ===


def test_word_count_in_outputs():
    r = {**default_req, "word_count_min": 100, "word_count_max": 150}
    si, pt, _ = builder.build_prompt(**r)
    assert "100-150" in si and "100-150" in pt


run_test("word_count_in_outputs", test_word_count_in_outputs)


def test_word_count_swap():
    r = {**default_req, "word_count_min": 200, "word_count_max": 100}
    si, pt, _ = builder.build_prompt(**r)
    assert "100-200" in si and "100-200" in pt


run_test("word_count_swap", test_word_count_swap)

# === Fallbacks ===


def test_empty_fallbacks():
    _, pt, _ = builder.build_prompt(**default_req)
    assert "observe and describe" in pt.lower()


run_test("empty_fallbacks", test_empty_fallbacks)


def test_scene_populated():
    _, pt, _ = builder.build_prompt(
        **default_req,
        scene_subjects="a dog",
        scene_setting="park",
        scene_props="ball",
    )
    assert "a dog" in pt and "park" in pt and "ball" in pt


run_test("scene_populated", test_scene_populated)

# === Denoise boundary values ===


def test_denoise_boundaries():
    assert _select_denoise_notes(0.0)["mode_label"] == "preservation"
    assert _select_denoise_notes(0.49)["mode_label"] == "preservation"
    assert _select_denoise_notes(0.5)["mode_label"] == "balanced"
    assert _select_denoise_notes(0.7)["mode_label"] == "balanced"
    assert _select_denoise_notes(0.71)["mode_label"] == "creative"
    assert _select_denoise_notes(1.0)["mode_label"] == "creative"


run_test("denoise_boundaries", test_denoise_boundaries)

# === Debug info ===


def test_debug_info():
    _, _, di = builder.build_prompt(**default_req)
    for key in ["Trigger Word:", "Style:", "Subjects:", "Motion:", "Denoise:",
                "Word Count:", "Budget Row:", "Denoise Mode:", "Processing:",
                "System Instruction Length:", "Prompt Text Length:"]:
        assert key in di, f"missing {key} in debug_info"


run_test("debug_info", test_debug_info)

# === Character table edge cases ===


def test_char_table_none():
    assert _build_character_table(None) == ""
    assert _build_character_table("") == ""
    assert _build_character_table("   \n  ") == ""


run_test("char_table_none", test_char_table_none)


def test_chunked_section_single():
    assert _build_chunked_section("single", 0, "") == ""


run_test("chunked_section_single", test_chunked_section_single)

# === Summary ===
print()
print(f"Results: {passed} passed, {failed} failed out of {passed + failed} tests")
if errors:
    print()
    for name, e in errors:
        print(f"  FAILED {name}: {e}")
    sys.exit(1)
else:
    print("All tests passed!")
