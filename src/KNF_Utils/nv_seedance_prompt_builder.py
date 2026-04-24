"""NV Seedance Prompt Builder — parametrized prompt construction for Seedance 2.0.

Declarative prompt assembly: you fill in semantic slots (source_subject,
target_subject, action_context, etc.), the node picks the right template
based on your ref configuration, and outputs a ready-to-translate prompt.

Pair with NV_SeedancePromptOptimizer for CN+bracket translation:

    [Builder] → prompt → [Optimizer translate_to_chinese] → optimized_prompt → [Prep V2]

Template selection:
  - `auto` (default): reads upload_config (if wired) and picks:
      mode=multimodal + n_images=1 → single_ref_swap
      mode=multimodal + n_images>1 + is_kling_hybrid=false → multi_ref_swap
      mode=multimodal + n_images>1 + is_kling_hybrid=true  → kling_hybrid
      mode=first_frame → i2v_first_frame
      mode=bridge      → bridge
      mode=text_only   → t2v
  - Manual override: pick any template by name; upload_config ignored for selection.

Templates are English strings with {slot} placeholders. Users can edit via
`custom_template` input (set template=custom). Each slot is optional — missing
slots render empty and surrounding punctuation is normalized.
"""

from __future__ import annotations

import json
import re

from .nv_seedance_upload_utils import (
    MODE_BRIDGE,
    MODE_FIRST_FRAME,
    MODE_MULTIMODAL,
    MODE_TEXT_ONLY,
    SEEDANCE_UPLOAD_CONFIG_V2,
)


# ---------------------------------------------------------------------------
# Templates — English, editable, with {slot} placeholders
# ---------------------------------------------------------------------------

_T_SINGLE_REF_SWAP = (
    "Replace {source_subject} in @Video1 with {target_subject} from @Image1"
    "{target_details_clause}. @Video1 IS the complete scene, motion, camera, "
    "lighting, props, and timing — preserve every aspect of @Video1 exactly. "
    "ONLY the central subject changes to match {target_subject}. "
    "{target_motion_clause}{output_style_clause}."
)

_T_MULTI_REF_SWAP = (
    "Replace {source_subject} in @Video1 with the person shown across the "
    "reference images. All reference images depict the SAME SUBJECT"
    "{target_details_clause}. @Video1 IS the complete scene, motion, camera, "
    "lighting, props, and timing — preserve every aspect of @Video1 exactly. "
    "ONLY the central subject changes to match the reference subject. "
    "{target_motion_clause}{output_style_clause}. "
    "Maintain consistent subject appearance — no identity drift across frames."
)

_T_KLING_HYBRID = (
    "Replace {source_subject} in @Video1 with the person shown in the "
    "reference images. All reference images depict the SAME SUBJECT.\n\n"
    "@Image1 is the canonical identity portrait{target_details_clause} — "
    "face area blurred for compliance; extract identity from hair, body "
    "proportions, and clothing.\n\n"
    "@Image2 onward are EXAMPLE SWAPS demonstrating how this subject should "
    "be rendered in a transformation similar to this one. Extract ONLY the "
    "subject's appearance (body, clothing, hair) from these examples. "
    "DO NOT carry over background elements, props, scenery, or compositional "
    "content from @Image2+ — scene and composition come ENTIRELY from "
    "@Video1.\n\n"
    "@Video1 IS the complete scene, motion, camera, lighting, props, and "
    "timing — preserve every aspect of @Video1 exactly. ONLY the central "
    "subject's identity changes. {target_motion_clause}{output_style_clause}. "
    "Maintain consistent subject appearance — no identity drift."
)

_T_T2V = (
    "{target_subject}{target_details_clause}{action_clause}"
    "{output_style_clause}."
)

_T_I2V_FIRST_FRAME = (
    "Animate @Image1 into a video. {action_clause_plain}"
    "{output_style_clause}."
)

_T_BRIDGE = (
    "Generate a video that smoothly transitions from @Image1 (first frame) "
    "to @Image2 (last frame). {action_clause_plain}{output_style_clause}."
)

_TEMPLATES = {
    "single_ref_swap": _T_SINGLE_REF_SWAP,
    "multi_ref_swap": _T_MULTI_REF_SWAP,
    "kling_hybrid": _T_KLING_HYBRID,
    "t2v": _T_T2V,
    "i2v_first_frame": _T_I2V_FIRST_FRAME,
    "bridge": _T_BRIDGE,
}

_TEMPLATE_CHOICES = ["auto"] + list(_TEMPLATES.keys()) + ["custom"]


# ---------------------------------------------------------------------------
# Template selection from upload_config mode
# ---------------------------------------------------------------------------

def _auto_select_template(config: dict | None, is_kling_hybrid: bool) -> str:
    """Pick a template name from the upload_config mode + hybrid flag."""
    if not isinstance(config, dict):
        return "multi_ref_swap"  # neutral default when no config wired
    mode = config.get("mode", MODE_TEXT_ONLY)
    n_images = (config.get("counts") or {}).get("images", 0)

    if mode == MODE_TEXT_ONLY:
        return "t2v"
    if mode == MODE_FIRST_FRAME:
        return "i2v_first_frame"
    if mode == MODE_BRIDGE:
        return "bridge"
    if mode == MODE_MULTIMODAL:
        if is_kling_hybrid:
            return "kling_hybrid"
        if n_images <= 1:
            return "single_ref_swap"
        return "multi_ref_swap"
    return "multi_ref_swap"


# ---------------------------------------------------------------------------
# Slot-fill helpers — graceful empty handling
# ---------------------------------------------------------------------------

def _clause(prefix: str, content: str, suffix: str = "") -> str:
    """Wrap content with prefix/suffix only if content is non-empty."""
    c = (content or "").strip()
    if not c:
        return ""
    return f"{prefix}{c}{suffix}"


def _normalize_punctuation(text: str) -> str:
    """Clean up artifacts from empty slot interpolation."""
    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)
    # Remove spaces before commas/periods/semicolons
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    # Collapse ", ." → "."
    text = re.sub(r",\s*\.", ".", text)
    # Collapse ". ." → "."
    text = re.sub(r"\.\s*\.", ".", text)
    # Collapse multiple consecutive periods
    text = re.sub(r"\.{2,}", ".", text)
    # Clean up leading/trailing whitespace on each line, preserve line breaks
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    # Collapse 3+ newlines into 2 (paragraph break)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class NV_SeedancePromptBuilder:
    """Parametrized prompt construction for Seedance 2.0 workflows.

    Fill in semantic slots, node picks a template based on wired config or
    your manual override, emits a ready-to-translate prompt.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "template": (_TEMPLATE_CHOICES, {
                    "default": "auto",
                    "tooltip": (
                        "auto (recommended): infers from upload_config — "
                        "multimodal+1img=single_ref_swap, multimodal+N=multi_ref_swap "
                        "(or kling_hybrid if flag set), first_frame=i2v, bridge, text_only. "
                        "Manual: force a specific template. "
                        "custom: use the custom_template input."
                    ),
                }),
                "source_subject": ("STRING", {
                    "multiline": True,
                    "default": "the man in the black tank top and cargo pants holding the green training staff",
                    "tooltip": "Who to REPLACE in the source video. Used by swap templates.",
                }),
                "target_subject": ("STRING", {
                    "multiline": True,
                    "default": "the person from @Image1",
                    "tooltip": "Who to REPLACE WITH. For multimodal, use generic like 'the subject from the reference images'.",
                }),
                "target_details": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": (
                        "Optional visual details of the target subject — clothing, hair, "
                        "distinguishing features. E.g., 'full chainmail armor, orange surcoat, "
                        "metal crusader helm'. Leave empty for generic refs."
                    ),
                }),
                "action_context": ("STRING", {
                    "multiline": True,
                    "default": "the staff choreography",
                    "tooltip": (
                        "What action the target subject does. Used by swap templates "
                        "(as 'react naturally to {action}') and t2v/i2v templates."
                    ),
                }),
                "output_style": ("STRING", {
                    "multiline": True,
                    "default": "Realistic cinematic look, neutral soundstage color grade",
                    "tooltip": "Cinematic / style description. Appended to the end of the prompt.",
                }),
                "is_kling_hybrid": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Set TRUE if reference_images contain Kling-generated frames acting as "
                        "in-context example swaps. auto template becomes kling_hybrid, which "
                        "includes anti-bleed language ('DO NOT carry over background elements')."
                    ),
                }),
                "include_identity_drift_clause": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Append 'no identity drift across frames' consistency language. Recommended for face-ID swaps.",
                }),
                "extra_constraints": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Free-form additional prompt text appended after the template body. Use for shot-specific instructions.",
                }),
                "custom_template": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": (
                        "Used ONLY when template=custom. Supports placeholders: "
                        "{source_subject}, {target_subject}, {target_details}, {action_context}, "
                        "{output_style}, {target_details_clause}, {target_motion_clause}, "
                        "{output_style_clause}, {action_clause}, {action_clause_plain}."
                    ),
                }),
            },
            "optional": {
                "upload_config": (SEEDANCE_UPLOAD_CONFIG_V2.io_type, {
                    "tooltip": "Optional — wire NV_SeedancePrep_V2's upload_config here. When template=auto, selection infers from config's mode + n_images.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "template_used", "info")
    FUNCTION = "build"
    CATEGORY = "NV_Utils/api"
    DESCRIPTION = (
        "Parametrized Seedance 2.0 prompt builder. Pick template (or auto-infer from "
        "upload_config), fill in subject/action/style slots, get a ready-to-translate "
        "English prompt. Chain into NV_SeedancePromptOptimizer for CN+bracket form."
    )

    def build(
        self,
        template: str,
        source_subject: str,
        target_subject: str,
        target_details: str,
        action_context: str,
        output_style: str,
        is_kling_hybrid: bool,
        include_identity_drift_clause: bool,
        extra_constraints: str,
        custom_template: str,
        upload_config: dict | None = None,
    ):
        # --- resolve template ---
        if template == "auto":
            resolved = _auto_select_template(upload_config, is_kling_hybrid)
        elif template == "custom":
            resolved = "custom"
        else:
            resolved = template

        if resolved == "custom":
            template_str = custom_template or ""
            if not template_str.strip():
                raise ValueError(
                    "[NV_SeedancePromptBuilder] template=custom but custom_template is empty."
                )
        else:
            template_str = _TEMPLATES[resolved]

        # --- build composed clauses ---
        # target_details_clause: ", wearing X" or " (X)" style — prepend comma if present
        target_details_clause = _clause(", wearing ", target_details)
        # If target_details starts with non-clothing word, use " — " instead
        if target_details and not re.match(r"^(wear|in|with|dressed)", target_details.lower()):
            target_details_clause = _clause(", ", target_details)

        # target_motion_clause: describes how the target should move
        if action_context.strip():
            target_motion_clause = (
                f"The subject's clothing and hair react naturally to {action_context}. "
            )
        else:
            target_motion_clause = ""

        # output_style_clause: just the style text with a leading space if present
        output_style_clause = _clause(" ", output_style)

        # action_clause / action_clause_plain for t2v/i2v/bridge templates
        action_clause = _clause(" ", action_context)
        action_clause_plain = _clause("", action_context, ". ") if action_context.strip() else ""

        # --- interpolate ---
        slots = {
            "source_subject": source_subject.strip() or "the subject",
            "target_subject": target_subject.strip() or "the target subject",
            "target_details": target_details.strip(),
            "action_context": action_context.strip(),
            "output_style": output_style.strip(),
            "target_details_clause": target_details_clause,
            "target_motion_clause": target_motion_clause,
            "output_style_clause": output_style_clause,
            "action_clause": action_clause,
            "action_clause_plain": action_clause_plain,
        }
        try:
            prompt = template_str.format(**slots)
        except KeyError as e:
            raise ValueError(
                f"[NV_SeedancePromptBuilder] Template references unknown slot {e}. "
                f"Valid slots: {sorted(slots.keys())}"
            )

        # --- clean up punctuation artifacts from empty-slot interpolation ---
        prompt = _normalize_punctuation(prompt)

        # --- optional identity-drift clause for swap templates ---
        if include_identity_drift_clause and resolved in ("single_ref_swap",):
            # multi_ref_swap and kling_hybrid already include this
            if "identity drift" not in prompt.lower():
                prompt = prompt.rstrip(".") + ". Maintain consistent subject appearance — no identity drift across frames."

        # --- append extra constraints ---
        if extra_constraints.strip():
            prompt = prompt.rstrip() + " " + extra_constraints.strip()
            prompt = _normalize_punctuation(prompt)

        info = {
            "template_requested": template,
            "template_resolved": resolved,
            "is_kling_hybrid": is_kling_hybrid,
            "config_mode": (upload_config or {}).get("mode") if upload_config else None,
            "config_n_images": ((upload_config or {}).get("counts") or {}).get("images") if upload_config else None,
            "config_n_videos": ((upload_config or {}).get("counts") or {}).get("videos") if upload_config else None,
            "prompt_length": len(prompt),
            "slots_used": {k: bool(v) for k, v in slots.items() if not k.endswith("_clause")},
        }

        print(f"[NV_SeedancePromptBuilder] template={resolved}, length={len(prompt)} chars")
        return (prompt, resolved, json.dumps(info, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "NV_SeedancePromptBuilder": NV_SeedancePromptBuilder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_SeedancePromptBuilder": "NV Seedance Prompt Builder",
}
