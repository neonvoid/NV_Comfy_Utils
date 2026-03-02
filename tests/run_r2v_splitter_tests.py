"""Standalone test runner for r2v_prompt_splitter (no torch/pytest dependency)."""
import importlib.util
import sys
import os

# Load module directly, bypassing package __init__.py (avoids torch dependency)
spec = importlib.util.spec_from_file_location(
    "r2v_prompt_splitter",
    os.path.join(os.path.dirname(__file__), "..", "src", "KNF_Utils", "r2v_prompt_splitter.py"),
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

NV_R2VPromptSplitter = mod.NV_R2VPromptSplitter

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


splitter = NV_R2VPromptSplitter()

SAMPLE = (
    "@Video1 sits at a wooden desk in a warmly lit home office, gesturing with "
    "their right hand while speaking. Bookshelves line the wall behind them. Warm "
    "tungsten key light from the upper left, soft fill from a window on the right. "
    "Medium close-up, static camera, shallow depth of field with background "
    "softly blurred. Natural movement, cinematic quality.\n"
    "NEGATIVE: low quality, blurry, distorted faces, unnatural movement, text, "
    "watermarks, shaky camera"
)

print("Running NV_R2VPromptSplitter tests...")
print()

# === Basic splitting ===


def test_basic_split():
    prompt, negative, info = splitter.split_caption(SAMPLE)
    assert "@Video1" in prompt
    assert "wooden desk" in prompt
    assert "low quality" in negative
    assert "distorted faces" in negative
    assert "NEGATIVE:" not in prompt


run_test("basic_split", test_basic_split)


def test_negative_clean():
    _, negative, _ = splitter.split_caption(SAMPLE)
    assert not negative.startswith("NEGATIVE")
    assert negative.startswith("low quality")


run_test("negative_clean", test_negative_clean)

# === No NEGATIVE: line ===


def test_no_negative():
    caption = "@Video1 walks through a park. Sunny day, wide shot."
    prompt, negative, info = splitter.split_caption(caption)
    assert prompt == caption
    assert negative == ""
    assert "No NEGATIVE: line found" in info


run_test("no_negative", test_no_negative)

# === Case insensitivity ===


def test_case_lower():
    caption = "Scene here.\nnegative: ugly, blurry"
    _, negative, _ = splitter.split_caption(caption)
    assert "ugly" in negative


run_test("case_lower", test_case_lower)


def test_case_mixed():
    caption = "Scene here.\nNegative: ugly, blurry"
    _, negative, _ = splitter.split_caption(caption)
    assert "ugly" in negative


run_test("case_mixed", test_case_mixed)

# === Multiple NEGATIVE: lines ===


def test_multiple_negative():
    caption = "Part 1.\nNEGATIVE: first\nPart 2.\nNEGATIVE: second, final"
    prompt, negative, _ = splitter.split_caption(caption)
    assert "second" in negative
    assert "Part 1" in prompt
    assert "Part 2" in prompt


run_test("multiple_negative", test_multiple_negative)

# === Truncation ===


def test_prompt_truncation():
    long_scene = "A " * 500
    caption = f"{long_scene}\nNEGATIVE: bad quality"
    prompt, negative, info = splitter.split_caption(caption, max_prompt_chars=200)
    assert len(prompt) <= 200
    assert "TRUNCATED" in info
    assert "bad quality" in negative


run_test("prompt_truncation", test_prompt_truncation)


def test_negative_truncation():
    long_neg = "bad, " * 200
    caption = f"Short prompt.\nNEGATIVE: {long_neg}"
    _, negative, info = splitter.split_caption(caption, max_negative_chars=100)
    assert len(negative) <= 100
    assert "TRUNCATED" in info


run_test("negative_truncation", test_negative_truncation)


def test_no_truncation():
    _, _, info = splitter.split_caption(SAMPLE)
    assert "TRUNCATED" not in info


run_test("no_truncation", test_no_truncation)

# === Edge cases ===


def test_empty():
    prompt, negative, info = splitter.split_caption("")
    assert prompt == ""
    assert negative == ""
    assert "Empty" in info


run_test("empty", test_empty)


def test_whitespace_only():
    prompt, negative, info = splitter.split_caption("   \n  \t  ")
    assert prompt == ""
    assert negative == ""
    assert "Empty" in info


run_test("whitespace_only", test_whitespace_only)


def test_negative_at_start():
    caption = "NEGATIVE: bad quality, blurry"
    prompt, negative, _ = splitter.split_caption(caption)
    assert prompt == ""
    assert "bad quality" in negative


run_test("negative_at_start", test_negative_at_start)


def test_leading_whitespace():
    caption = "Scene description.\n   NEGATIVE: bad quality"
    prompt, negative, _ = splitter.split_caption(caption)
    assert "Scene description" in prompt
    assert "bad quality" in negative


run_test("leading_whitespace", test_leading_whitespace)

# === Node metadata ===


def test_metadata():
    assert NV_R2VPromptSplitter.RETURN_TYPES == ("STRING", "STRING", "STRING")
    assert NV_R2VPromptSplitter.RETURN_NAMES == ("prompt", "negative_prompt", "info")
    assert NV_R2VPromptSplitter.FUNCTION == "split_caption"
    assert NV_R2VPromptSplitter.CATEGORY == "NV_Utils/Prompt"


run_test("metadata", test_metadata)


def test_info_counts():
    _, _, info = splitter.split_caption(SAMPLE)
    assert "Prompt:" in info
    assert "chars" in info
    assert "Negative:" in info


run_test("info_counts", test_info_counts)

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
