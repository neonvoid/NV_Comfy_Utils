"""
NV V2V Prompt Builder - Assemble parameterized prompts for V2V LoRA video captioning.

Builds structured system instructions and per-video prompt text from individual
parameters. Designed to feed directly into GeminiVideoCaptioner (or any LLM
captioner that accepts system_instruction + prompt_text).

Supports:
- Automatic word budget selection based on subject count and motion intensity
- Denoise-strength-aware analysis priority weighting
- Chunked processing mode with temporal continuity from previous chunk captions
- Character token recognition tables
- Custom system instruction override for advanced users
"""


# ---------------------------------------------------------------------------
# Template constants
# ---------------------------------------------------------------------------

_SYSTEM_ROLE = """\
You are a video analysis assistant guiding style LoRA application in a \
video-to-video transformation pipeline. Analyze the input video to produce \
a scene description that enables consistent, temporally-coherent stylization \
while preserving the source footage's motion, structure, and composition."""

_SYSTEM_ANALYSIS_PRIORITIES = """\
## Analysis Priorities

- **Scene content**: Treat the entire frame as a single unified environment. \
Describe subjects, setting, and background as one cohesive scene regardless of \
apparent depth separation or lighting variation.
- **Structural preservation**: Motion trajectories, pose sequences, spatial \
relationships between subjects, and compositional framing that must survive the \
style transfer.{denoise_structural_note}
- **Surface and material cues**: Describe materials, textures, and surface \
properties as style anchor points — these guide the LoRA toward appropriate \
texture replacement. Use language compatible with the target style.{denoise_surface_note}
- **Temporal considerations**: Describe motion arcs, camera evolution, and \
temporal risk areas.{temporal_note}
{motion_priority_clause}"""

_SYSTEM_OUTPUT_CONSTRAINTS = """\
## Output Constraints

- Describe the scene as a single continuous environment — do not distinguish \
between foreground elements and background elements as separate sources or layers.
- Do not reference screens, displays, projections, compositing, lighting rigs, \
or any production infrastructure.
- Do not flag lighting direction mismatches, color temperature differences, or \
edge boundaries between subjects and environment — treat all lighting as naturally \
occurring within the scene.
- Absorb the scene context naturally into your description. Do not echo it back \
as a list or preamble — weave it into the scene prose as observed detail.
- Use language and terminology that aligns with the provided style description. \
If the style is painterly, lean toward surface and material terms. If the style \
is graphic or cel-shaded, lean toward edge, silhouette, and flat region terms.
- Use plain, technical language — no poetic flourishes, metaphors, or dramatic framing.
- Do not infer narrative, emotion, or intent beyond observable action.
- Do not add details that are not clearly visible.
- Be precise and factual."""

_SYSTEM_WORD_BUDGET = """\
## Word Budget Allocation

Distribute the target word count based on these priorities:

| Condition | Environment | Subject(s) | Temporal | Materials |
|-----------|-------------|------------|----------|-----------|
| 1 subject, low motion | 25% | 30% | 15% | 30% |
| 1 subject, high motion | 20% | 25% | 30% | 25% |
| 2+ subjects, low motion | 20% | 35% | 15% | 30% |
| 2+ subjects, high motion | 15% | 30% | 30% | 25% |
| No subjects (environment only) | 40% | 0% | 25% | 35% |

**Active row for this video**: {budget_row_label}

These are guidelines, not rigid constraints. Prioritize whatever aspect of the \
scene is most critical for a successful style transfer."""

_SYSTEM_CHARACTER_RECOGNITION = """\
## Character Recognition

When you recognize these characters in the video, use their exact token names:

| Token | Description |
|-------|-------------|
{character_table_rows}

Only apply a character token when the subject is clearly recognizable and \
matches the token description. If recognition is ambiguous, describe by \
visible attributes instead."""

_SYSTEM_OUTPUT_FORMAT = """\
## Output Format

{word_count_min}-{word_count_max} words of flowing prose. \
{trigger_word_clause}

Cover in this order:
1. Scene environment and subject positioning
2. Subject actions and motion arc
3. Surface and material cues for style anchoring
4. Temporal risk areas and stable regions"""

_PROMPT_TEMPLATE = """\
Analyze this video and generate a V2V caption using these parameters:

**Trigger Word**: {trigger_word_display}
**Style**: {style_display}
**Scene Context**:
- Subjects: {subjects_display}
- Setting: {setting_display}
- Props: {props_display}
**Video Duration**: {duration_display}
**Camera**: {camera_display}
**Subject Count**: {subject_count}
**Motion Intensity**: {motion_intensity}
**Denoise Strength**: {denoise_strength}
**Target Word Count**: {word_count_min}-{word_count_max} words
{chunked_section}"""

_CHUNKED_FIRST = """
**Processing Mode**: Chunked (Chunk 0 — establishing baseline)
This is the first chunk. Establish the scene, characters, and style baseline. \
Subsequent chunks will reference your description for temporal continuity."""

_CHUNKED_CONTINUATION = """
**Processing Mode**: Chunked (Chunk {chunk_index} — continuation)
**Previous Chunk Caption**:
> {previous_chunk_prompt}

Maintain temporal continuity with the previous chunk. Continue the described \
action and motion naturally. Preserve consistent subject descriptions and style \
terminology. Note any new elements or changes from the previous chunk."""

_CHUNKED_NO_CONTEXT = """
**Processing Mode**: Chunked (Chunk {chunk_index} — no previous context)
No previous chunk caption was provided. Describe this chunk independently \
but be aware it is part of a longer video sequence."""


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _build_character_table(character_tokens):
    """Parse multiline 'token: description' into markdown table rows.

    Returns empty string if no valid entries found.
    """
    if not character_tokens or not character_tokens.strip():
        return ""

    rows = []
    for line in character_tokens.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # Support both "token: description" and "token | description"
        if ":" in line:
            parts = line.split(":", 1)
        elif "|" in line:
            parts = line.split("|", 1)
        else:
            continue
        token = parts[0].strip()
        desc = parts[1].strip() if len(parts) > 1 else ""
        if token:
            rows.append(f"| {token} | {desc} |")

    return "\n".join(rows)


def _select_motion_clause(motion_intensity):
    """Return motion-specific analysis guidance."""
    clauses = {
        "low": (
            "- **Motion emphasis**: Motion is minimal. Focus on subtle weight "
            "shifts, micro-expressions, and stillness as compositional elements."
        ),
        "medium": (
            "- **Motion emphasis**: Balance motion description with spatial "
            "detail. Note the rhythm and pacing of movements."
        ),
        "high": (
            "- **Motion emphasis**: Prioritize motion trajectories, speed "
            "changes, and dynamic action. Flag fast-moving regions as temporal "
            "risk areas for the style transfer."
        ),
    }
    return clauses.get(motion_intensity, clauses["medium"])


def _select_denoise_notes(denoise_strength):
    """Return denoise-aware weighting notes for analysis priorities.

    Returns a dict with keys: structural_note, surface_note, mode_label.
    """
    if denoise_strength < 0.5:
        return {
            "structural_note": " (Weight this heavily — low denoise strength means structural fidelity is critical.)",
            "surface_note": "",
            "mode_label": "preservation",
        }
    elif denoise_strength > 0.7:
        return {
            "structural_note": "",
            "surface_note": " (Weight this heavily — high denoise strength allows more stylistic transformation.)",
            "mode_label": "creative",
        }
    else:
        return {
            "structural_note": "",
            "surface_note": "",
            "mode_label": "balanced",
        }


def _select_budget_row_label(subject_count, motion_intensity):
    """Return the human-readable label for the active word budget row."""
    if subject_count == 0:
        return "No subjects (environment only)"
    elif subject_count == 1:
        if motion_intensity == "high":
            return "1 subject, high motion"
        else:
            return "1 subject, low motion"
    else:
        if motion_intensity == "high":
            return "2+ subjects, high motion"
        else:
            return "2+ subjects, low motion"


def _build_temporal_note(video_duration):
    """Return a temporal note scaled to video duration."""
    if not video_duration or not video_duration.strip():
        return ""
    return f" (Scale depth of temporal description to the video duration: {video_duration.strip()}.)"


def _build_chunked_section(processing_mode, chunk_index, previous_chunk_prompt):
    """Build the chunked processing section for prompt_text."""
    if processing_mode != "chunked":
        return ""

    if chunk_index == 0:
        return _CHUNKED_FIRST

    if previous_chunk_prompt and previous_chunk_prompt.strip():
        return _CHUNKED_CONTINUATION.format(
            chunk_index=chunk_index,
            previous_chunk_prompt=previous_chunk_prompt.strip(),
        )

    return _CHUNKED_NO_CONTEXT.format(chunk_index=chunk_index)


# ---------------------------------------------------------------------------
# Node class
# ---------------------------------------------------------------------------

class NV_V2VPromptBuilder:
    """Assemble parameterized V2V LoRA captioning prompts from individual inputs."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trigger_word": ("STRING", {
                    "default": "",
                    "tooltip": "LoRA trigger word that begins every output caption (e.g., 'ohwx', 'N1TEF1TESTLY3'). Leave empty if no trigger word needed."
                }),
                "style_description": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Brief description of the target style the LoRA produces (e.g., 'Dark fantasy illustration with metallic textures and painterly lighting')."
                }),
                "subject_count": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 10,
                    "step": 1,
                    "tooltip": "Number of primary subjects in the video. 0 = environment-only shot. Affects word budget allocation."
                }),
                "motion_intensity": (["low", "medium", "high"], {
                    "default": "medium",
                    "tooltip": "Overall motion intensity. 'low' = mostly static, 'medium' = walking/gesturing, 'high' = fast action. Affects word budget and motion emphasis."
                }),
                "denoise_strength": ("FLOAT", {
                    "default": 0.65,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Denoise strength for the V2V pipeline. <0.5 = preservation mode (emphasize structural fidelity), >0.7 = creative mode (emphasize style transformation)."
                }),
                "word_count_min": ("INT", {
                    "default": 80,
                    "min": 20,
                    "max": 500,
                    "step": 10,
                    "tooltip": "Minimum word count for the output caption."
                }),
                "word_count_max": ("INT", {
                    "default": 120,
                    "min": 30,
                    "max": 600,
                    "step": 10,
                    "tooltip": "Maximum word count for the output caption."
                }),
            },
            "optional": {
                "scene_subjects": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Ground-truth descriptions of subjects. One per line (e.g., 'Man with long hair in a bun, wearing dark layered clothing')."
                }),
                "scene_setting": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Intended environment/setting (e.g., 'Autumn forest clearing with a large central oak tree')."
                }),
                "scene_props": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Notable objects that need to survive the style transfer (e.g., 'Two-handed longsword, round wooden shield')."
                }),
                "character_tokens": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "LoRA-trained character tokens, one per line: 'token: description' or 'token | description'. Only applied when the VLM can confirm the character."
                }),
                "camera_behavior": ("STRING", {
                    "default": "",
                    "tooltip": "Camera movement description (e.g., 'Static wide shot', 'Slow pan left to right', 'Handheld tracking, moderate shake')."
                }),
                "video_duration": ("STRING", {
                    "default": "",
                    "tooltip": "Approximate video duration as free text (e.g., '3 seconds', '12 seconds'). Affects temporal description depth."
                }),
                "processing_mode": (["single", "chunked"], {
                    "default": "single",
                    "tooltip": "'single' = standalone video. 'chunked' = part of a multi-chunk pipeline (enables temporal continuity instructions)."
                }),
                "previous_chunk_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Caption from the previous chunk. Only used when processing_mode='chunked'. Enables temporal continuity between chunks."
                }),
                "chunk_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 999,
                    "step": 1,
                    "tooltip": "Current chunk index (0-based). Chunk 0 = establishing baseline. Only used when processing_mode='chunked'."
                }),
                "custom_system_override": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "If non-empty, replaces the entire built-in system instruction. Supports {trigger_word}, {word_count_min}, {word_count_max} placeholders."
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("system_instruction", "prompt_text", "debug_info")
    FUNCTION = "build_prompt"
    CATEGORY = "NV_Utils/Prompt"
    DESCRIPTION = (
        "Assemble a structured V2V LoRA captioning prompt from individual parameters. "
        "Outputs system_instruction and prompt_text that connect directly to "
        "Multi-API Video Captioner. Supports denoise-aware analysis priorities, "
        "word budget allocation, character token tables, and chunked processing "
        "mode for temporal continuity."
    )

    def build_prompt(self, trigger_word, style_description, subject_count,
                     motion_intensity, denoise_strength, word_count_min,
                     word_count_max, **kwargs):
        # Extract optional inputs
        scene_subjects = kwargs.get("scene_subjects", "").strip()
        scene_setting = kwargs.get("scene_setting", "").strip()
        scene_props = kwargs.get("scene_props", "").strip()
        character_tokens = kwargs.get("character_tokens", "").strip()
        camera_behavior = kwargs.get("camera_behavior", "").strip()
        video_duration = kwargs.get("video_duration", "").strip()
        processing_mode = kwargs.get("processing_mode", "single")
        previous_chunk_prompt = kwargs.get("previous_chunk_prompt", "")
        chunk_index = kwargs.get("chunk_index", 0)
        custom_system_override = kwargs.get("custom_system_override", "").strip()

        trigger_word = trigger_word.strip()
        style_description = style_description.strip()

        # Validate word count range
        if word_count_min > word_count_max:
            word_count_min, word_count_max = word_count_max, word_count_min

        # ----- Build system_instruction -----
        if custom_system_override:
            # Substitute known placeholders in override text
            system_instruction = custom_system_override.replace(
                "{trigger_word}", trigger_word
            ).replace(
                "{word_count_min}", str(word_count_min)
            ).replace(
                "{word_count_max}", str(word_count_max)
            )
        else:
            system_instruction = self._assemble_system_instruction(
                trigger_word=trigger_word,
                subject_count=subject_count,
                motion_intensity=motion_intensity,
                denoise_strength=denoise_strength,
                word_count_min=word_count_min,
                word_count_max=word_count_max,
                character_tokens=character_tokens,
                video_duration=video_duration,
            )

        # ----- Build prompt_text -----
        prompt_text = self._assemble_prompt_text(
            trigger_word=trigger_word,
            style_description=style_description,
            scene_subjects=scene_subjects,
            scene_setting=scene_setting,
            scene_props=scene_props,
            camera_behavior=camera_behavior,
            video_duration=video_duration,
            subject_count=subject_count,
            motion_intensity=motion_intensity,
            denoise_strength=denoise_strength,
            word_count_min=word_count_min,
            word_count_max=word_count_max,
            processing_mode=processing_mode,
            chunk_index=chunk_index,
            previous_chunk_prompt=previous_chunk_prompt,
        )

        # ----- Build debug_info -----
        denoise_notes = _select_denoise_notes(denoise_strength)
        budget_label = _select_budget_row_label(subject_count, motion_intensity)
        char_table = _build_character_table(character_tokens)
        char_count = len(char_table.strip().splitlines()) if char_table else 0

        debug_info = (
            f"=== NV_V2VPromptBuilder Debug ===\n"
            f"Trigger Word: {trigger_word or '(none)'}\n"
            f"Style: {style_description[:60] or '(none)'}\n"
            f"Subjects: {subject_count} | Motion: {motion_intensity} | Denoise: {denoise_strength}\n"
            f"Word Count: {word_count_min}-{word_count_max}\n"
            f"Budget Row: {budget_label}\n"
            f"Denoise Mode: {denoise_notes['mode_label']}\n"
            f"Processing: {processing_mode}"
            + (f" (chunk {chunk_index})" if processing_mode == "chunked" else "") + "\n"
            f"Character Tokens: {char_count or 'none'}\n"
            f"System Override: {'yes' if custom_system_override else 'no'}\n"
            f"System Instruction Length: {len(system_instruction)} chars\n"
            f"Prompt Text Length: {len(prompt_text)} chars"
        )

        print(f"[NV_V2VPromptBuilder] Built prompt ({len(system_instruction)} chars system, {len(prompt_text)} chars prompt)")

        return (system_instruction, prompt_text, debug_info)

    def _assemble_system_instruction(self, trigger_word, subject_count,
                                     motion_intensity, denoise_strength,
                                     word_count_min, word_count_max,
                                     character_tokens, video_duration):
        """Assemble the full system instruction from template sections."""
        denoise_notes = _select_denoise_notes(denoise_strength)
        motion_clause = _select_motion_clause(motion_intensity)
        temporal_note = _build_temporal_note(video_duration)
        budget_label = _select_budget_row_label(subject_count, motion_intensity)

        # Section 2: Analysis Priorities
        analysis_priorities = _SYSTEM_ANALYSIS_PRIORITIES.format(
            denoise_structural_note=denoise_notes["structural_note"],
            denoise_surface_note=denoise_notes["surface_note"],
            temporal_note=temporal_note,
            motion_priority_clause=motion_clause,
        )

        # Section 4: Word Budget
        word_budget = _SYSTEM_WORD_BUDGET.format(budget_row_label=budget_label)

        # Section 5: Character Recognition (conditional)
        char_table_rows = _build_character_table(character_tokens)
        character_section = ""
        if char_table_rows:
            character_section = "\n\n" + _SYSTEM_CHARACTER_RECOGNITION.format(
                character_table_rows=char_table_rows
            )

        # Section 6: Output Format
        if trigger_word:
            trigger_clause = f"Always begin with the trigger word: **{trigger_word}**"
        else:
            trigger_clause = "Begin directly with the scene description. No trigger word is needed."

        output_format = _SYSTEM_OUTPUT_FORMAT.format(
            word_count_min=word_count_min,
            word_count_max=word_count_max,
            trigger_word_clause=trigger_clause,
        )

        # Assemble all sections
        sections = [
            _SYSTEM_ROLE,
            analysis_priorities,
            _SYSTEM_OUTPUT_CONSTRAINTS,
            word_budget,
            output_format,
        ]

        result = "\n\n".join(sections)

        # Insert character section before output format
        if character_section:
            # Insert before the last section (output format)
            parts = result.rsplit("\n\n## Output Format", 1)
            if len(parts) == 2:
                result = parts[0] + character_section + "\n\n## Output Format" + parts[1]

        return result

    def _assemble_prompt_text(self, trigger_word, style_description,
                              scene_subjects, scene_setting, scene_props,
                              camera_behavior, video_duration, subject_count,
                              motion_intensity, denoise_strength,
                              word_count_min, word_count_max,
                              processing_mode, chunk_index,
                              previous_chunk_prompt):
        """Assemble the per-video prompt text."""
        chunked_section = _build_chunked_section(
            processing_mode, chunk_index, previous_chunk_prompt
        )

        prompt_text = _PROMPT_TEMPLATE.format(
            trigger_word_display=trigger_word or "(none)",
            style_display=style_description or "(not specified — observe and describe the visual style)",
            subjects_display=scene_subjects or "(observe and describe)",
            setting_display=scene_setting or "(observe and describe)",
            props_display=scene_props or "(observe and describe)",
            duration_display=video_duration or "(not specified)",
            camera_display=camera_behavior or "(observe and describe)",
            subject_count=subject_count,
            motion_intensity=motion_intensity,
            denoise_strength=denoise_strength,
            word_count_min=word_count_min,
            word_count_max=word_count_max,
            chunked_section=chunked_section,
        )

        return prompt_text.strip()


NODE_CLASS_MAPPINGS = {
    "NV_V2VPromptBuilder": NV_V2VPromptBuilder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_V2VPromptBuilder": "NV V2V Prompt Builder",
}
