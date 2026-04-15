"""
NV V2V Prompt Builder - Assemble parameterized prompts for V2V LoRA video captioning.

Builds structured system instructions and per-video prompt text from individual
parameters. Designed to feed directly into GeminiVideoCaptioner (or any LLM
captioner that accepts system_instruction + prompt_text).

Supports:
- Task mode selection: 'full_restyle' (scene-wide style transfer),
  'character_swap' (targeted character replacement via inpainting),
  'r2v_bootstrap' (scene prompts for WAN 2.6 R2V API),
  'kling_edit' (Kling API edit mode), 'kling_reference' (Kling API reference mode),
  'seedance_ref' (ByteDance Seedance 2.0 multimodal ref-to-video),
  or 'wan27_edit' (Alibaba Tongyi Wan 2.7 Video Edit)
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
| 1 subject, low/medium motion | 25% | 30% | 20% | 25% |
| 1 subject, high motion | 20% | 25% | 30% | 25% |
| 2+ subjects, low motion | 20% | 35% | 15% | 30% |
| 2+ subjects, low/medium motion | 20% | 35% | 20% | 25% |
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
# Character Swap mode template constants
# ---------------------------------------------------------------------------

_CS_SYSTEM_ROLE = """\
You are a video analysis assistant guiding character replacement in a \
video inpainting pipeline. Analyze the input video to produce a \
character-focused description that enables the inpainting model to \
replace the target subject with a LoRA-trained character while \
preserving the scene environment, motion, and composition."""

_CS_ANALYSIS_PRIORITIES = """\
## Analysis Priorities

- **Character identity**: Describe the replacement character using the exact \
LoRA trigger word and character token. Detail their physical attributes, \
clothing, accessories, and distinguishing features so the model renders the \
correct identity.{denoise_structural_note}
- **Pose and motion continuity**: The replacement character must match the \
original subject's body position, gestures, gait, and motion arc. Describe \
the pose precisely — limb positions, weight distribution, facing direction.{denoise_surface_note}
- **Scene integration**: Provide enough environment context that the replacement \
looks naturally placed — lighting direction on the character, depth and scale \
relative to surroundings, ground contact, and any occluding objects.
- **Temporal consistency**: The character's appearance must remain stable across \
frames. Flag moments where occlusion, fast motion, or angle changes could \
break identity continuity.{temporal_note}
{motion_priority_clause}"""

_CS_OUTPUT_CONSTRAINTS = """\
## Output Constraints

- Lead with the replacement character — trigger word first, then appearance details.
- Include scene context only as needed for spatial grounding, not as primary content.
- Do not over-describe the unchanged environment — focus the word budget on the \
character's appearance, pose, and actions.
- Describe the character's pose and action within the scene, not the scene itself.
- Use the character token name consistently throughout — never fall back to generic \
pronouns after the initial introduction.
- Absorb the replacement target description naturally — describe what the \
replacement character is doing in that position, not what they are replacing.
- Use plain, technical language — no poetic flourishes, metaphors, or dramatic framing.
- Do not infer narrative, emotion, or intent beyond observable action.
- Do not add details that are not clearly visible.
- Be precise and factual."""

_CS_WORD_BUDGET = """\
## Word Budget Allocation

Distribute the target word count based on these priorities:

| Condition | Character | Pose/Action | Scene Context | Temporal |
|-----------|-----------|-------------|---------------|----------|
| 1 subject, low motion | 50% | 15% | 20% | 15% |
| 1 subject, low/medium motion | 45% | 20% | 15% | 20% |
| 1 subject, high motion | 40% | 25% | 10% | 25% |
| 2+ subjects, low motion | 45% | 15% | 20% | 20% |
| 2+ subjects, low/medium motion | 40% | 20% | 15% | 25% |
| 2+ subjects, high motion | 35% | 25% | 15% | 25% |
| No subjects (environment only) | 40% | 0% | 25% | 35% |

**Active row for this video**: {budget_row_label}

These are guidelines, not rigid constraints. Prioritize whatever aspect of the \
character is most critical for a convincing replacement."""

_CS_OUTPUT_FORMAT = """\
## Output Format

{word_count_min}-{word_count_max} words of flowing prose. \
{trigger_word_clause}

Cover in this order:
1. Character identity (trigger word + appearance details)
2. Pose, action, and motion arc
3. Scene context for spatial grounding
4. Temporal risk areas for identity consistency"""

_CS_PROMPT_TEMPLATE = """\
Analyze this video and generate a character replacement caption using these parameters:

**Trigger Word**: {trigger_word_display}
**Style**: {style_display}
**Replace Target**: {subjects_display}
**Scene Context**:
- Setting: {setting_display}
- Props: {props_display}
**Video Duration**: {duration_display}
**Camera**: {camera_display}
**Subject Count**: {subject_count}
**Motion Intensity**: {motion_intensity}
**Denoise Strength**: {denoise_strength}
**Target Word Count**: {word_count_min}-{word_count_max} words
{chunked_section}"""

_CS_CHARACTER_RECOGNITION = """\
## Character Replacement Tokens

Use these exact token names when describing the replacement character:

| Token | Description |
|-------|-------------|
{character_table_rows}

Always use the character token for the replacement subject. The token must \
appear as the first identifier when introducing the character."""


# ---------------------------------------------------------------------------
# R2V Bootstrap mode template constants
# ---------------------------------------------------------------------------

_R2V_SYSTEM_ROLE = """\
You are a video analysis assistant preparing scene descriptions for a \
reference-to-video (R2V) generation pipeline. Analyze the input video to \
produce a scene and action description that will guide the WAN 2.6 R2V model \
to generate a replacement character performing the same actions in a matching \
environment. The R2V model already knows what the character looks like from \
reference video — your job is to describe everything ELSE: the scene, the \
action, the camera, and the lighting."""

_R2V_ANALYSIS_PRIORITIES = """\
## Analysis Priorities

- **Action and pose**: Describe precisely what the target subject is doing — \
body position, gestures, gait, facing direction, movement trajectory. The R2V \
model must generate the replacement character performing these same actions. \
This is the highest priority.{denoise_structural_note}
- **Scene environment**: Describe the setting in enough detail that R2V \
generates a visually compatible background — indoor/outdoor, architecture, \
surfaces, depth, spatial layout. Match the level of detail to what the \
downstream VACE pipeline needs for seamless compositing.{denoise_surface_note}
- **Lighting and color**: Describe lighting direction, color temperature, \
contrast, and any dominant color palette. The R2V output must match the \
target scene's lighting for VACE to blend without seams.
- **Camera behavior**: Shot type (close-up, medium, wide), camera motion \
(static, pan, track, handheld), and any depth-of-field characteristics. \
R2V output framing should approximate the target.{temporal_note}
{motion_priority_clause}"""

_R2V_OUTPUT_CONSTRAINTS = """\
## Output Constraints

- Do NOT describe the original subject's identity, face, or appearance — \
the R2V model gets identity from reference video, not from your caption. \
Refer to the subject only by their actions and spatial position.
- Use the provided R2V reference tags (e.g. @Video1) as the subject \
identifier throughout. If multiple subjects are being replaced, use the \
appropriate tag for each.
- Describe the scene as the character would experience it — lighting hitting \
them, ground beneath them, objects around them. This grounds the R2V \
generation in the correct spatial context.
- Include explicit camera framing — the R2V output must match shot type and \
angle for VACE compatibility.
- Use cinematic vocabulary: prefer precise terms like "tracking shot", \
"dolly zoom", "shallow depth of field", "tungsten key light" over vague \
descriptors like "smooth camera" or "warm lighting".
- This prompt may be auto-expanded by the R2V API — be precise and specific \
rather than exhaustive. Concise prompts with strong vocabulary outperform \
verbose ones when prompt expansion is enabled.
- Do not reference production infrastructure, compositing, VFX, or the \
pipeline itself.
- Use plain, technical language — no poetic flourishes, metaphors, or \
dramatic framing.
- Do not infer narrative, emotion, or intent beyond observable action.
- Do not add details that are not clearly visible.
- Be precise and factual.
- End with quality-negative guidance on a separate line prefixed with \
NEGATIVE: to populate the R2V negative prompt field."""

_R2V_WORD_BUDGET = """\
## Word Budget Allocation

Distribute the target word count based on these priorities:

| Condition | Action/Pose | Environment | Lighting/Color | Camera |
|-----------|-------------|-------------|---------------|--------|
| 1 subject, low motion | 25% | 30% | 25% | 20% |
| 1 subject, low/medium motion | 30% | 25% | 25% | 20% |
| 1 subject, high motion | 40% | 20% | 20% | 20% |
| 2+ subjects, low motion | 30% | 25% | 25% | 20% |
| 2+ subjects, low/medium motion | 35% | 20% | 25% | 20% |
| 2+ subjects, high motion | 40% | 20% | 20% | 20% |
| No subjects (environment only) | 10% | 40% | 30% | 20% |

**Active row for this video**: {budget_row_label}

These are guidelines, not rigid constraints. Prioritize whatever aspect of \
the scene is most critical for generating a visually compatible R2V output. \
Note: WAN 2.6 R2V prompt limit is 800 characters — stay concise."""

_R2V_OUTPUT_FORMAT = """\
## Output Format

{word_count_min}-{word_count_max} words of flowing prose. \
{trigger_word_clause}

Cover in this order:
1. Subject action and spatial position (using R2V reference tags)
2. Scene environment and spatial context
3. Lighting direction, color temperature, and mood
4. Camera framing and movement
5. NEGATIVE: quality-negative terms (on its own line, always include)

Example structure:
"@Video1 sits at a wooden desk in a warmly lit home office, gesturing with \
their right hand while speaking. Bookshelves line the wall behind them. Warm \
tungsten key light from the upper left, soft fill from a window on the right. \
Medium close-up, static camera, shallow depth of field with background \
softly blurred. Natural movement, cinematic quality.
NEGATIVE: low quality, blurry, distorted faces, unnatural movement, text, \
watermarks, shaky camera" """

_R2V_PROMPT_TEMPLATE = """\
Analyze this video and generate an R2V (reference-to-video) scene prompt \
using these parameters:

**R2V Subject References**: {trigger_word_display}
**Target Pipeline**: R2V bootstrap → VACE inpainting (scene must match for compositing)
**Scene Context**:
- Original Subject(s): {subjects_display}
- Setting: {setting_display}
- Props: {props_display}
**Video Duration**: {duration_display}
**Camera**: {camera_display}
**Subject Count**: {subject_count}
**Motion Intensity**: {motion_intensity}
**Scene Match Strictness**: {denoise_strength}
**Target Word Count**: {word_count_min}-{word_count_max} words
{chunked_section}

Remember: Do NOT describe the subject's identity or appearance — only their \
actions, pose, and the scene around them. Identity comes from reference video."""

_R2V_CHARACTER_RECOGNITION = """\
## Subject Reference Mapping

Map the subjects you observe to these R2V reference tags:

| Reference Tag | Replaces |
|---------------|----------|
{character_table_rows}

Use the reference tag (e.g., @Video1) consistently throughout. Never describe \
the replacement character's appearance — only their actions and position."""


# ---------------------------------------------------------------------------
# Kling Edit mode template constants
# ---------------------------------------------------------------------------

_KE_SYSTEM_ROLE = """\
You are a prompt engineer for the Kling AI video editing API. The API takes \
an input video and modifies it according to a text prompt. Your job is to \
write a clear, concise edit instruction that tells Kling what to change.

The downstream pipeline handles reference image tags (@image1, @image2, etc.), \
alias legends, and negative prompt formatting automatically — you only need \
to write the main prompt body. If reference images are described below, you \
may mention them naturally in your prompt using @image1, @image2, etc., but \
only when it adds clarity. Do not add a [References: ...] legend or \
"Avoid: ..." line — those are appended by the pipeline."""

_KE_ANALYSIS_PRIORITIES = """\
## Analysis Priorities

- **Transformation target**: What specifically should change in the video? \
Focus on the delta between input and desired output — not on describing \
what already exists. Be direct: "change X to Y", "add Z", "remove W".{denoise_structural_note}
- **Identity preservation**: When reference images provide character or style \
identity, describe the target appearance concisely. Kling handles identity \
transfer from reference images — your prompt should guide the transformation, \
not exhaustively describe the reference.{denoise_surface_note}
- **Scene continuity**: Note which elements should remain unchanged. Kling \
edits the entire frame — explicitly protect important elements by mentioning \
them (e.g., "keep the background cityscape unchanged").{temporal_note}
{motion_priority_clause}"""

_KE_OUTPUT_CONSTRAINTS = """\
## Output Constraints

- Write edit instructions, not scene descriptions. "Transform the character \
into a samurai wearing red armor" beats "A samurai in red armor stands in \
a field" for edit mode.
- Be specific about the transformation: what changes, what stays the same.
- The Kling API has a 2500 character total limit. Your prompt body should \
stay under ~1800 characters to leave room for reference legends (~300 chars) \
and negative prompt (~400 chars) that are appended automatically.
- When reference images are available, mention @image1, @image2 etc. naturally \
in context. Example: "Transform the character to match @image1's appearance \
while keeping the original pose and setting."
- Do NOT add [References: ...] legends — handled downstream.
- Do NOT add "Avoid: ..." lines — handled downstream.
- Do NOT add <<<image_N>>> API wire format — handled downstream.
- Use plain, direct language. Kling responds best to clear, unambiguous \
instructions rather than poetic or abstract descriptions.
- For subtle edits (low intensity): focus on specific details to change.
- For dramatic edits (high intensity): describe the complete target state."""

_KE_WORD_BUDGET = """\
## Word Budget Allocation

Distribute the target word count based on edit scope:

| Condition | Transformation | Identity/Style | Preservation | Spatial |
|-----------|---------------|----------------|--------------|---------|
| 1 subject, low intensity | 40% | 25% | 25% | 10% |
| 1 subject, medium intensity | 35% | 30% | 20% | 15% |
| 1 subject, high intensity | 30% | 35% | 15% | 20% |
| 2+ subjects, low intensity | 35% | 30% | 20% | 15% |
| 2+ subjects, medium intensity | 30% | 35% | 15% | 20% |
| 2+ subjects, high intensity | 25% | 35% | 15% | 25% |
| Scene-only (no subjects) | 50% | 20% | 15% | 15% |

**Active row for this video**: {budget_row_label}

Kling edit prompts benefit from brevity — a focused 80-word prompt often \
outperforms a detailed 200-word one. Stay concise."""

_KE_OUTPUT_FORMAT = """\
## Output Format

{word_count_min}-{word_count_max} words of direct edit instructions. \
{trigger_word_clause}

Structure your prompt in this priority order:
1. Primary transformation (what changes)
2. Reference image usage (if applicable — @image1, @image2, etc.)
3. Style/aesthetic target
4. Preservation notes (what must stay unchanged)"""

_KE_PROMPT_TEMPLATE = """\
Write a Kling video edit prompt using these parameters:

**Edit Goal**: {style_display}
**Subjects in Scene**: {subjects_display}
**Target Setting**: {setting_display}
**Props/Details**: {props_display}
**Edit Intensity**: {denoise_strength} (0.0 = subtle tweak, 1.0 = dramatic transformation)
**Video Duration**: {duration_display}
**Camera**: {camera_display}
**Subject Count**: {subject_count}
**Motion Level**: {motion_intensity}
**Target Word Count**: {word_count_min}-{word_count_max} words
{chunked_section}

Write the prompt body only — reference legends and negative prompt are added \
by the downstream pipeline. Stay under ~1800 characters."""

_KE_CHARACTER_RECOGNITION = """\
## Reference Image Mapping

These reference images will be connected to the Kling node. Use the @image \
tags naturally in your prompt when referencing them:

| Tag | Role |
|-----|------|
{character_table_rows}

Example usage: "Transform the character to match @image1's face and hairstyle \
while wearing the outfit from @image2." Only reference images that are relevant \
to the edit — don't force-include every reference."""


# ---------------------------------------------------------------------------
# Kling Reference mode template constants
# ---------------------------------------------------------------------------

_KR_SYSTEM_ROLE = """\
You are a prompt engineer for the Kling AI reference-to-video API. The API \
uses an input video as a style and motion template to generate entirely new \
content. Your job is to describe the desired output scene — the input video \
provides motion/style guidance, not content to preserve.

The downstream pipeline handles reference image tags (@image1, @image2, etc.), \
alias legends, @video tags, and negative prompt formatting automatically — \
you only need to write the main prompt body. If reference images are described \
below, you may mention them using @image1, @image2, etc. The input video is \
automatically tagged as @video by the pipeline."""

_KR_ANALYSIS_PRIORITIES = """\
## Analysis Priorities

- **Scene description**: Describe the scene you want generated, not the input \
video. The input provides motion rhythm and visual style — your prompt defines \
the new content.{denoise_structural_note}
- **Character identity**: When reference images provide character identity, \
mention them naturally with @image tags. Kling will extract the character's \
appearance from the image.{denoise_surface_note}
- **Visual style and mood**: Describe the target aesthetic — color palette, \
lighting, atmosphere. The input video's style is a starting point; your prompt \
can steer it.
- **Camera and framing**: Describe the desired shot type and camera behavior. \
Kling will use the input video's camera motion as a baseline but your prompt \
can influence it.{temporal_note}
{motion_priority_clause}"""

_KR_OUTPUT_CONSTRAINTS = """\
## Output Constraints

- Describe the OUTPUT scene, not the input video. The input is a template — \
you're telling Kling what new content to generate using that template's \
motion and style.
- The Kling API has a 2500 character total limit. Your prompt body should \
stay under ~1800 characters to leave room for reference legends (~300 chars) \
and negative prompt (~400 chars) that are appended automatically.
- When reference images are available, mention @image1, @image2 etc. naturally \
in context. The input video is @video (auto-injected by pipeline).
- Do NOT add [References: ...] legends — handled downstream.
- Do NOT add "Avoid: ..." lines — handled downstream.
- Do NOT add @video tags — auto-injected by the pipeline.
- Do NOT add <<<image_N>>> API wire format — handled downstream.
- Use cinematic vocabulary: shot types, lighting terms, movement descriptors.
- Kling reference mode generates 3-15 seconds of video. Describe a scene \
achievable in that timeframe — no complex narratives or scene transitions.
- Be specific about visual elements. "A warrior in red samurai armor walking \
through cherry blossoms at sunset" beats "an epic battle scene"."""

_KR_WORD_BUDGET = """\
## Word Budget Allocation

Distribute the target word count based on scene complexity:

| Condition | Scene/Action | Character | Style/Mood | Camera |
|-----------|-------------|-----------|------------|--------|
| 1 subject, low motion | 25% | 30% | 25% | 20% |
| 1 subject, medium motion | 30% | 25% | 25% | 20% |
| 1 subject, high motion | 35% | 20% | 20% | 25% |
| 2+ subjects, low motion | 30% | 30% | 20% | 20% |
| 2+ subjects, medium motion | 30% | 25% | 20% | 25% |
| 2+ subjects, high motion | 35% | 20% | 20% | 25% |
| Scene-only (no subjects) | 40% | 0% | 35% | 25% |

**Active row for this video**: {budget_row_label}

Reference mode benefits from rich visual description — be specific about \
the scene you want generated."""

_KR_OUTPUT_FORMAT = """\
## Output Format

{word_count_min}-{word_count_max} words of flowing scene description. \
{trigger_word_clause}

Structure your prompt in this priority order:
1. Scene and subject action (what's happening)
2. Character identity via @image references (if applicable)
3. Visual style, lighting, and atmosphere
4. Camera framing and movement"""

_KR_PROMPT_TEMPLATE = """\
Write a Kling reference-to-video scene prompt using these parameters:

**Scene Goal**: {style_display}
**Subjects**: {subjects_display}
**Setting**: {setting_display}
**Props/Details**: {props_display}
**Creative Freedom**: {denoise_strength} (0.0 = closely match input, 1.0 = free interpretation)
**Video Duration**: {duration_display}
**Camera**: {camera_display}
**Subject Count**: {subject_count}
**Motion Level**: {motion_intensity}
**Target Word Count**: {word_count_min}-{word_count_max} words
{chunked_section}

Describe the desired OUTPUT scene — the input video is just a motion/style \
template. Write the prompt body only — reference legends, @video tag, and \
negative prompt are added by the downstream pipeline. Stay under ~1800 characters."""

_KR_CHARACTER_RECOGNITION = """\
## Reference Image Mapping

These reference images will be connected to the Kling node. Use the @image \
tags naturally when describing characters or style elements:

| Tag | Role |
|-----|------|
{character_table_rows}

The input video is automatically referenced as @video by the pipeline. \
Example: "A warrior with @image1's face walks through a bamboo forest at \
golden hour, matching the cinematic style of the input." Only reference \
images that are relevant to the scene."""


# ---------------------------------------------------------------------------
# Seedance 2.0 Reference-to-Video (ByteDance API) templates
#
# Seedance 2.0 uses role-tagged multimodal content (images + videos + audio)
# rather than positional tokens. The official prompt convention documented by
# ByteDance Seed + BytePlus ModelArk is `@Image1 / @Image 1` for image refs;
# `@Video1` and `@Audio1` follow the same pattern by inference.
#
# Mode selection is prompt-directed — the API has no explicit "edit vs extend
# vs compose" flag. The phrasing of the prompt tells the model what to do:
#   - "Extend @Video1, showing..."           → continuation
#   - "Edit @Video1: replace the car with..."→ editing
#   - "Generate a scene featuring @Image1..."→ composition (default)
#   - "...with the camera movement of @Video1, and the look of @Image1"
#                                            → motion/style reference
#   - "...synchronized to @Audio1"           → MV / audio-driven
#
# No negative prompt field exists on the API — avoidance must be expressed as
# positive language inside the main prompt.
# ---------------------------------------------------------------------------

_SD_SYSTEM_ROLE = """\
You are a prompt engineer for the ByteDance Seedance 2.0 reference-to-video \
API. Seedance 2.0 accepts a multimodal reference bundle — images, video clips, \
and audio — and synthesizes a new video that fuses those references according \
to your prompt.

Reference handles follow ByteDance's documented convention:
- `@Image1`, `@Image2`, … for reference images (characters, objects, style anchors)
- `@Video1` for a reference video (camera language, motion, pacing, or source to edit/extend)
- `@Audio1` for reference audio (music rhythm, lip sync, sound characteristics)

Tags must be woven into the prose inline — not listed as a prefix. "A woman \
resembling @Image1 wearing @Image2's outfit, with the camera movement of \
@Video1" is the correct shape. "@Image1 @Image2 @Video1: a woman…" is wrong.

The operating mode (compose / motion-transfer / extend / edit / style-ref / \
mv-with-audio) is selected by how you phrase the prompt, not by any API flag. \
Choose phrasing that matches the user's intent:
- Compose: "Generate a scene featuring @Image1 and @Image2 in …"
- Motion transfer: "…with the camera movement and pacing of @Video1"
- Extend: "Extend @Video1 forward, continuing the action as …"
- Edit: "Edit @Video1: replace X with …"
- Style ref: "…in the visual style of @Video1"
- MV / audio-driven: "…synchronized to the rhythm of @Audio1" or "lip-syncing to @Audio1"

Seedance 2.0 has no negative prompt field — express avoidances as positive \
language ("clear sky" instead of "no clouds")."""

_SD_ANALYSIS_PRIORITIES = """\
## Analysis Priorities

- **Subject identity from @Image refs**: When reference images provide \
character or object identity, mention them inline with @Image tags. Describe \
the attribute you want carried over ("the facial features of @Image1", "the \
outfit in @Image2") rather than just naming the tag.{denoise_structural_note}
- **Motion and camera from @Video1**: If a reference video is present, \
explicitly describe what to borrow from it — camera trajectory, pacing, \
choreography, or visual style. Seedance treats @Video1 as a general \
conditioning signal; the prompt disambiguates which axis to transfer.{denoise_surface_note}
- **Rhythm and audio from @Audio1**: If a reference audio is present, describe \
how the visual should respond to it (beat-cut rhythm, drum hits, lip sync, \
ambient layering). Seedance can align visual timing to audio features.
- **Scene composition**: Describe the environment, lighting, and atmosphere of \
the output. Seedance generates 4–15 seconds at 480p or 720p — scope the scene \
to that timeframe.{temporal_note}
{motion_priority_clause}"""

_SD_OUTPUT_CONSTRAINTS = """\
## Output Constraints

- Weave @Image / @Video / @Audio tags INLINE into the scene prose. Never list \
them as a prefix or in a bracketed legend.
- The output operating mode (compose / motion-transfer / extend / edit / \
style-ref / mv-with-audio) must be clear from your phrasing. Use verbs like \
"extend", "edit", "featuring", "with the camera of", "in the style of", \
"synchronized to" to signal intent.
- Do NOT add "Avoid:" or "Negative:" lines — Seedance has no negative prompt \
field. Convert avoidances to positive language ("empty street" not "no people").
- Do NOT add bracketed [References: …] legends — Seedance parses tags inline.
- Do NOT add <<<image_N>>>, <image>, or other API-wire syntax — use `@Image1` \
style only (the downstream node handles API serialization).
- Keep the prompt focused on ONE coherent shot (4–15s). No scene transitions, \
no multi-shot sequences, no narrative arcs.
- Be specific about visual elements: shot type, lighting direction, colour \
palette, material cues. "A woman resembling @Image1 walks through a rain-slick \
alley at dusk, warm sodium street lamps" beats "a cool scene with @Image1".
- Assume 24fps output and 720p maximum resolution — do not describe 4K-level \
micro-detail that the model cannot produce.
- If the user lists multiple ref images, use ALL of them purposefully. A ref \
that is uploaded but never mentioned is wasted API cost."""

_SD_WORD_BUDGET = """\
## Word Budget Allocation

Distribute the target word count based on reference density and motion:

| Condition | Scene/Action | Subject (refs) | Style/Lighting | Camera/Motion |
|-----------|-------------|----------------|----------------|---------------|
| 1 subject, low motion | 25% | 35% | 25% | 15% |
| 1 subject, medium motion | 30% | 30% | 20% | 20% |
| 1 subject, high motion | 30% | 25% | 20% | 25% |
| 2+ subjects, low motion | 25% | 40% | 20% | 15% |
| 2+ subjects, medium motion | 30% | 35% | 15% | 20% |
| 2+ subjects, high motion | 30% | 30% | 15% | 25% |
| Scene-only (no subjects) | 40% | 0% | 35% | 25% |

**Active row for this video**: {budget_row_label}

When @Video1 is present and motion-transfer intent applies, shift ~10% into \
Camera/Motion. When @Audio1 is present, reserve ~10% for rhythm/sync language \
pulled from Scene/Action."""

_SD_OUTPUT_FORMAT = """\
## Output Format

{word_count_min}-{word_count_max} words of flowing scene description with \
reference tags woven inline. {trigger_word_clause}

Structure your prompt in this priority order:
1. Primary subject and action (introduce @Image / @Video refs here)
2. Scene, environment, and lighting
3. Camera framing and motion (explicit @Video1 borrow, if applicable)
4. Audio sync language (if @Audio1 is present)

Write as a single paragraph, no bullet points, no headings. The Seedance API \
consumes the prompt as free text."""

_SD_PROMPT_TEMPLATE = """\
Write a Seedance 2.0 reference-to-video scene prompt using these parameters:

**Scene Goal**: {style_display}
**Subjects**: {subjects_display}
**Setting**: {setting_display}
**Props/Details**: {props_display}
**Reference Tags Available**: {trigger_word_display}
**Creative Freedom**: {denoise_strength} (0.0 = closely follow refs, 1.0 = free interpretation)
**Video Duration**: {duration_display}
**Camera**: {camera_display}
**Subject Count**: {subject_count}
**Motion Level**: {motion_intensity}
**Target Word Count**: {word_count_min}-{word_count_max} words
{chunked_section}

Write the prompt body only — weave @Image / @Video / @Audio tags inline, no \
bracketed legends, no "Avoid:" lines, no API-wire syntax. Signal operating \
mode (compose / extend / edit / motion-transfer / style-ref / mv-audio) \
through your verb choice."""

_SD_CHARACTER_RECOGNITION = """\
## Reference Tag Mapping

These reference assets will be supplied to the Seedance node. Mention each \
relevant tag INLINE when describing its role in the scene:

| Tag | Role |
|-----|------|
{character_table_rows}

Example phrasings:
- `@Image1`: "a woman with the features of @Image1"
- `@Image2`: "wearing the outfit seen in @Image2"
- `@Video1` (motion): "with the slow dolly-in camera movement of @Video1"
- `@Video1` (extend): "extend @Video1 forward as she turns toward…"
- `@Audio1` (rhythm): "cuts synchronized to the drum beat of @Audio1"

Only reference tags that materially contribute to the scene. Unused refs waste \
API tokens."""


# ---------------------------------------------------------------------------
# Wan 2.7 Video Edit (Alibaba Tongyi) templates
#
# Wan 2.7 Video Edit takes an input video + optional reference images and
# applies edit instructions per prompt. Structurally similar to Kling Omni
# Edit but with these key differences:
#   - NO `negative_prompt` field on Video Edit schema (present on T2V/I2V/R2V)
#   - NO `prompt_extend` on Video Edit (present elsewhere) → prompts go verbatim
#   - Reference images are CONTEXTUAL — no explicit tag convention like Kling's
#     @image1 or Seedance's @Image1. Describe their role naturally.
#   - Bilingual (English + Chinese both supported)
#   - 2-10s duration envelope, 720P/1080P only
# ---------------------------------------------------------------------------

_W27E_SYSTEM_ROLE = """\
You are a prompt engineer for the Alibaba Tongyi Wan 2.7 Video Edit API. The \
API takes an input video and modifies it per text instructions. Your job is \
to write a clear, concise edit instruction that tells the model what to \
change about the input video.

Wan 2.7 Video Edit specifics:
- Supports English and Chinese prompts (do not machine-translate between them).
- Up to 4 optional reference images may be attached — they are contextual, \
with NO explicit tag convention. Describe their role in natural language \
(e.g., "wearing the outfit shown in the reference image").
- Output is 2-10 seconds at 720P or 1080P.
- There is NO negative prompt field on Video Edit — express avoidances as \
positive language ("clear sky" instead of "no clouds").
- Prompts go through verbatim; the server does not auto-expand your prompt."""

_W27E_ANALYSIS_PRIORITIES = """\
## Analysis Priorities

- **Edit deltas**: Describe what to CHANGE about the input video — not the \
full scene. If the input shows a person walking, and you want them running, \
say "make the subject run" not "a person running down a street."{denoise_structural_note}
- **Preservation cues**: Note what should be preserved from the input — camera \
angle, framing, subject identity, background structure — to help the model \
lock those while changing what's requested.{denoise_surface_note}
- **Reference image roles**: If reference images are attached, describe what \
each contributes (identity, outfit, style, object appearance). Without a tag \
convention, natural-language grounding is the only binding mechanism.
- **Style + lighting terminology**: Use concrete visual terms. Wan 2.7 \
responds to cinematography vocabulary.{temporal_note}
{motion_priority_clause}"""

_W27E_OUTPUT_CONSTRAINTS = """\
## Output Constraints

- Write EDIT INSTRUCTIONS, not scene descriptions. Focus on the delta between \
the input video and the desired output. Use directive language: "replace X \
with Y", "change the color to Z", "add W to the scene".
- Do NOT add "Avoid:" or "Negative:" lines — Wan 2.7 Video Edit has no \
negative prompt field. Convert avoidances into positive language.
- Do NOT use @image, @video, @subject, <<<image_N>>>, or any tag convention. \
Reference images are contextual — describe their role in prose.
- Stay concise — the edit instruction should be direct and unambiguous. Long \
scene descriptions dilute the edit signal.
- Scope to a single shot. Wan 2.7 Video Edit produces 2-10 seconds — no \
multi-shot sequences, no scene transitions, no narrative arcs.
- English or Chinese are both acceptable. Do not mix unless the user's draft \
already mixes.
- Output ONLY the prompt text — no explanations, preamble, or \
meta-commentary.
- When iterating, build on the previous version — do not restart from \
scratch unless explicitly asked."""

_W27E_WORD_BUDGET = """\
## Word Budget Allocation

Distribute the target word count based on edit complexity:

| Condition | Edit Delta | Preservation Cues | Style/Lighting | References |
|-----------|-----------|-------------------|----------------|-----------|
| 1 subject, low motion | 40% | 25% | 25% | 10% |
| 1 subject, medium motion | 40% | 25% | 20% | 15% |
| 1 subject, high motion | 45% | 20% | 15% | 20% |
| 2+ subjects, low motion | 40% | 25% | 20% | 15% |
| 2+ subjects, medium motion | 40% | 25% | 15% | 20% |
| 2+ subjects, high motion | 45% | 20% | 15% | 20% |
| Scene-only (no subjects) | 45% | 30% | 25% | 0% |

**Active row for this video**: {budget_row_label}

Edit instructions benefit from high Delta weight. Lean into what to change, \
not how to describe the whole scene."""

_W27E_OUTPUT_FORMAT = """\
## Output Format

{word_count_min}-{word_count_max} words of direct edit instruction. \
{trigger_word_clause}

Structure your prompt in this priority order:
1. What to change (the primary edit delta)
2. What to preserve (structural/identity cues from input)
3. Style/lighting guidance
4. Reference image roles (if applicable)

Write as a single flowing paragraph. No bullet points, no headings."""

_W27E_PROMPT_TEMPLATE = """\
Write a Wan 2.7 Video Edit instruction using these parameters:

**Edit Goal**: {style_display}
**Subjects in input**: {subjects_display}
**Setting in input**: {setting_display}
**Notable props/details**: {props_display}
**Creative Freedom**: {denoise_strength} (0.0 = minimal edit, 1.0 = aggressive transformation)
**Video Duration**: {duration_display}
**Camera**: {camera_display}
**Subject Count**: {subject_count}
**Motion Level**: {motion_intensity}
**Target Word Count**: {word_count_min}-{word_count_max} words
{chunked_section}

Write the edit instruction only — no negative prompt, no tag conventions, no \
API-wire syntax. Keep the prompt direct and action-oriented."""

_W27E_CHARACTER_RECOGNITION = """\
## Reference Image Context

Reference images are attached to this edit. Wan 2.7 Video Edit has no tag \
convention — reference them by role in natural language:

| Description |
|------------|
{character_table_rows}

Example phrasings:
- "replace the subject's outfit with the one shown in the reference image"
- "match the color grading of the first reference"
- "restyle the background to resemble the reference photograph"

Only mention references that materially contribute to the edit. Unused \
references are ignored by the model."""


# ---------------------------------------------------------------------------
# Mode template registry
# ---------------------------------------------------------------------------

_MODE_TEMPLATES = {
    "full_restyle": {
        "system_role": _SYSTEM_ROLE,
        "analysis_priorities": _SYSTEM_ANALYSIS_PRIORITIES,
        "output_constraints": _SYSTEM_OUTPUT_CONSTRAINTS,
        "word_budget": _SYSTEM_WORD_BUDGET,
        "output_format": _SYSTEM_OUTPUT_FORMAT,
        "prompt_template": _PROMPT_TEMPLATE,
        "character_recognition": _SYSTEM_CHARACTER_RECOGNITION,
    },
    "character_swap": {
        "system_role": _CS_SYSTEM_ROLE,
        "analysis_priorities": _CS_ANALYSIS_PRIORITIES,
        "output_constraints": _CS_OUTPUT_CONSTRAINTS,
        "word_budget": _CS_WORD_BUDGET,
        "output_format": _CS_OUTPUT_FORMAT,
        "prompt_template": _CS_PROMPT_TEMPLATE,
        "character_recognition": _CS_CHARACTER_RECOGNITION,
    },
    "r2v_bootstrap": {
        "system_role": _R2V_SYSTEM_ROLE,
        "analysis_priorities": _R2V_ANALYSIS_PRIORITIES,
        "output_constraints": _R2V_OUTPUT_CONSTRAINTS,
        "word_budget": _R2V_WORD_BUDGET,
        "output_format": _R2V_OUTPUT_FORMAT,
        "prompt_template": _R2V_PROMPT_TEMPLATE,
        "character_recognition": _R2V_CHARACTER_RECOGNITION,
    },
    "kling_edit": {
        "system_role": _KE_SYSTEM_ROLE,
        "analysis_priorities": _KE_ANALYSIS_PRIORITIES,
        "output_constraints": _KE_OUTPUT_CONSTRAINTS,
        "word_budget": _KE_WORD_BUDGET,
        "output_format": _KE_OUTPUT_FORMAT,
        "prompt_template": _KE_PROMPT_TEMPLATE,
        "character_recognition": _KE_CHARACTER_RECOGNITION,
    },
    "kling_reference": {
        "system_role": _KR_SYSTEM_ROLE,
        "analysis_priorities": _KR_ANALYSIS_PRIORITIES,
        "output_constraints": _KR_OUTPUT_CONSTRAINTS,
        "word_budget": _KR_WORD_BUDGET,
        "output_format": _KR_OUTPUT_FORMAT,
        "prompt_template": _KR_PROMPT_TEMPLATE,
        "character_recognition": _KR_CHARACTER_RECOGNITION,
    },
    "seedance_ref": {
        "system_role": _SD_SYSTEM_ROLE,
        "analysis_priorities": _SD_ANALYSIS_PRIORITIES,
        "output_constraints": _SD_OUTPUT_CONSTRAINTS,
        "word_budget": _SD_WORD_BUDGET,
        "output_format": _SD_OUTPUT_FORMAT,
        "prompt_template": _SD_PROMPT_TEMPLATE,
        "character_recognition": _SD_CHARACTER_RECOGNITION,
    },
    "wan27_edit": {
        "system_role": _W27E_SYSTEM_ROLE,
        "analysis_priorities": _W27E_ANALYSIS_PRIORITIES,
        "output_constraints": _W27E_OUTPUT_CONSTRAINTS,
        "word_budget": _W27E_WORD_BUDGET,
        "output_format": _W27E_OUTPUT_FORMAT,
        "prompt_template": _W27E_PROMPT_TEMPLATE,
        "character_recognition": _W27E_CHARACTER_RECOGNITION,
    },
}


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
    motion_label = {"low": "low", "medium": "low/medium", "high": "high"}.get(motion_intensity, motion_intensity)
    if subject_count == 0:
        return "No subjects (environment only)"
    elif subject_count == 1:
        if motion_intensity == "high":
            return "1 subject, high motion"
        else:
            return f"1 subject, {motion_label} motion"
    else:
        if motion_intensity == "high":
            return "2+ subjects, high motion"
        else:
            return f"2+ subjects, {motion_label} motion"


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
                "task_mode": (list(_MODE_TEMPLATES.keys()), {
                    "default": "full_restyle",
                    "tooltip": (
                        "Prompt generation mode.\n"
                        "'full_restyle' = full scene V2V style transfer.\n"
                        "'character_swap' = targeted character replacement via inpainting.\n"
                        "'r2v_bootstrap' = scene prompts for WAN 2.6 R2V API.\n"
                        "'kling_edit' = Kling API edit mode — describe what to change.\n"
                        "'kling_reference' = Kling API reference mode — describe new scene to generate.\n"
                        "'seedance_ref' = ByteDance Seedance 2.0 multimodal ref-to-video (@Image1/@Video1/@Audio1).\n"
                        "'wan27_edit' = Alibaba Tongyi Wan 2.7 Video Edit — edit instructions on input video."
                    ),
                }),
                "trigger_word": ("STRING", {
                    "default": "",
                    "tooltip": "LoRA trigger word (full_restyle/character_swap) or R2V reference tags like '@Video1' (r2v_bootstrap). Leave empty for defaults."
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
        "Assemble a structured captioning/prompting prompt from individual parameters. "
        "Supports task modes: 'full_restyle' (V2V style transfer), "
        "'character_swap' (inpainting), 'r2v_bootstrap' (WAN R2V API), "
        "'kling_edit' (Kling edit API), 'kling_reference' (Kling reference API), "
        "'seedance_ref' (ByteDance Seedance 2.0 ref-to-video), "
        "'wan27_edit' (Alibaba Tongyi Wan 2.7 Video Edit). "
        "Outputs system_instruction and prompt_text for the LLM prompt refiner."
    )

    def build_prompt(self, task_mode, trigger_word, style_description,
                     subject_count, motion_intensity, denoise_strength,
                     word_count_min, word_count_max, **kwargs):
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

        # Resolve mode templates
        templates = _MODE_TEMPLATES.get(task_mode, _MODE_TEMPLATES["full_restyle"])

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
                templates=templates,
                task_mode=task_mode,
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
            templates=templates,
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
            f"Task Mode: {task_mode}\n"
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

        print(f"[NV_V2VPromptBuilder] Built {task_mode} prompt ({len(system_instruction)} chars system, {len(prompt_text)} chars prompt)")

        return (system_instruction, prompt_text, debug_info)

    def _assemble_system_instruction(self, templates, task_mode, trigger_word,
                                     subject_count, motion_intensity,
                                     denoise_strength, word_count_min,
                                     word_count_max, character_tokens,
                                     video_duration):
        """Assemble the full system instruction from mode-specific template sections."""
        denoise_notes = _select_denoise_notes(denoise_strength)
        motion_clause = _select_motion_clause(motion_intensity)
        temporal_note = _build_temporal_note(video_duration)
        budget_label = _select_budget_row_label(subject_count, motion_intensity)

        # Section 2: Analysis Priorities
        analysis_priorities = templates["analysis_priorities"].format(
            denoise_structural_note=denoise_notes["structural_note"],
            denoise_surface_note=denoise_notes["surface_note"],
            temporal_note=temporal_note,
            motion_priority_clause=motion_clause,
        )

        # Section 4: Word Budget
        word_budget = templates["word_budget"].format(budget_row_label=budget_label)

        # Section 5: Character Recognition (conditional)
        char_table_rows = _build_character_table(character_tokens)
        character_section = ""
        if char_table_rows:
            character_section = "\n\n" + templates["character_recognition"].format(
                character_table_rows=char_table_rows
            )

        # Section 6: Output Format — trigger clause is mode-aware
        if task_mode == "r2v_bootstrap":
            if trigger_word:
                trigger_clause = (
                    f"Use these R2V reference tags for subjects: **{trigger_word}**. "
                    f"Integrate them naturally at the start of the description."
                )
            else:
                trigger_clause = "Use @Video1 for the primary subject."
        elif task_mode in ("kling_edit", "kling_reference"):
            trigger_clause = (
                "Begin directly with the edit/scene instruction. "
                "Reference images (@image1, @image2, etc.) can be mentioned "
                "naturally where relevant — do not force them in."
            )
        elif task_mode == "seedance_ref":
            if trigger_word:
                trigger_clause = (
                    f"Available reference tags: **{trigger_word}**. "
                    "Weave each relevant tag inline where it contributes to the scene "
                    "(e.g., 'a woman with the features of @Image1'). Do not prefix "
                    "the prompt with a list of tags."
                )
            else:
                trigger_clause = (
                    "Weave @Image1..N and @Video1 tags inline where they contribute "
                    "to the scene (always include the numeric index — `@Image1`, not `@Image`). "
                    "Do not prefix the prompt with a list of tags. Only mention @Audio1 "
                    "if a reference audio is actually being supplied to the Seedance node."
                )
        elif task_mode == "wan27_edit":
            trigger_clause = (
                "Begin directly with the edit instruction. Do NOT use any tag "
                "convention (@image/@video/@subject) — Wan 2.7 Video Edit has no "
                "tag syntax. Reference images (if any) should be described by their "
                "role naturally (e.g., 'the outfit shown in the reference image'). "
                "Do NOT add any negative prompt line — Wan 2.7 Video Edit has no "
                "negative prompt field; express avoidances as positive language."
            )
        else:
            if trigger_word:
                trigger_clause = f"Always begin with the trigger word: **{trigger_word}**"
            else:
                trigger_clause = "Begin directly with the scene description. No trigger word is needed."

        output_format = templates["output_format"].format(
            word_count_min=word_count_min,
            word_count_max=word_count_max,
            trigger_word_clause=trigger_clause,
        )

        # Assemble all sections
        sections = [
            templates["system_role"],
            analysis_priorities,
            templates["output_constraints"],
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

    def _assemble_prompt_text(self, templates, trigger_word,
                              style_description, scene_subjects,
                              scene_setting, scene_props, camera_behavior,
                              video_duration, subject_count,
                              motion_intensity, denoise_strength,
                              word_count_min, word_count_max,
                              processing_mode, chunk_index,
                              previous_chunk_prompt):
        """Assemble the per-video prompt text using mode-specific template."""
        chunked_section = _build_chunked_section(
            processing_mode, chunk_index, previous_chunk_prompt
        )

        prompt_text = templates["prompt_template"].format(
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
