"""Helpers for assembling music pipeline LLM prompts."""
from __future__ import annotations

import json
from typing import Any, Dict, Optional, Iterable, Tuple

from ..templates import apply_template


_TEMPLATE_FIRST_SCORE = "prompts/music_first_score_prompt.md"
_TEMPLATE_METADATA_TRACKS = "prompts/music_metadata_tracks_prompt.md"
_TEMPLATE_CHECK = "prompts/music_check_prompt.md"
_TEMPLATE_SUBTLE_EDIT = "prompts/music_subtle_edit_prompt.md"
_TEMPLATE_MELODY_EMOTION = "prompts/music_melody_emotion_prompt.md"
_TEMPLATE_EMOTION_CHORD = "prompts/music_emotion_chord_prompt.md"
_TEMPLATE_IMPORT_SANITIZE = "prompts/music_import_sanitize_prompt.md"
_TEMPLATE_CSV_FORMAT = "prompts/musiccsv_format_prompt.txt"
_INSTRUCTIONS_MELODY_EMOTION = "prompts/melody_emotion_instructions.txt"
_INSTRUCTIONS_EMOTION_CHORD = "prompts/emotion_chord_instructions.txt"


# NOTE: The detailed MusicCSV format reference is no longer injected into the
# first-score prompt. It is still used for subtle-edit and check prompts.
_FORMAT_REFERENCE = apply_template(_TEMPLATE_CSV_FORMAT, {})


def _json_block(data: Any) -> str:
    try:
        return json.dumps(data, ensure_ascii=False, indent=2)
    except Exception:
        return str(data)


def _coerce_unique_strings(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in items:
        if raw is None:
            continue
        text = str(raw).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(text)
    return ordered


def build_first_score_prompt(
    *,
    prompt_payload: Dict[str, Any],
    existing_voice_notes: str,
    tp_index: int,
    tp_type: str,
    tp_text: str,
    voice_token: str,
    voice_chord: str,
    voice_register: str,
    voice_instrument: str,
    voice_idea: str,
    voice_role: str,
    metadata_json: str,
    melody_csv: str,
    melody_reduced_csv: str,
    other_voices_reduced_csv: str,
    measure_start: Optional[int] = None,
    measure_end: Optional[int] = None,
    block_index: Optional[int] = None,
    block_count: Optional[int] = None,
    measure_window_description: str = "",
    melody_block_csv: str = "",
    melody_previous_block_csv: str = "",
    voice_previous_block_csv: str = "",
    other_voices_block_csv: str = "",
    other_voices_previous_block_csv: str = "",
) -> str:
    payload_for_json = dict(prompt_payload)
    character_context = payload_for_json.pop("character_outlines", None)

    if measure_start is not None and measure_end is not None and not measure_window_description:
        if measure_start == measure_end:
            measure_window_description = f"Measure {measure_start}"
        else:
            measure_window_description = f"Measures {measure_start}-{measure_end}"
        if block_index is not None:
            if block_count is not None and block_count > 0:
                measure_window_description += f" (block {block_index} of {block_count})"
            else:
                measure_window_description += f" (block {block_index})"
        elif block_count is not None and block_count > 0:
            measure_window_description += f" (total blocks: {block_count})"

    block_index_str = str(block_index) if block_index is not None else ""
    block_count_str = str(block_count) if block_count is not None else ""

    if melody_block_csv.strip():
        melody_reduced_for_prompt = ""
    else:
        melody_reduced_for_prompt = melody_reduced_csv

    replacements = {
        "[VOICE_CONTEXT_JSON]": _json_block(payload_for_json),
        "[CHARACTER_CONTEXT_JSON]": _json_block(character_context or []),
        "[EXISTING_VOICE_SCORE_JSON]": existing_voice_notes.strip(),
        "[TOUCH_POINT_INDEX]": str(tp_index),
        "[TOUCH_POINT_TYPE]": tp_type,
        "[TOUCH_POINT_TEXT]": tp_text.strip(),
        "[VOICE_TOKEN]": voice_token,
        "[VOICE_CHORD]": voice_chord,
        "[VOICE_REGISTER]": voice_register,
        "[VOICE_INSTRUMENT]": voice_instrument,
        "[VOICE_IDEA]": voice_idea,
        "[VOICE_ROLE]": voice_role,
        "[METADATA_JSON]": metadata_json,
        "[MELODY_CSV]": melody_csv,
        "[MELODY_REDUCED_CSV]": melody_reduced_for_prompt,
        "[OTHER_VOICES_REDUCED_CSV]": other_voices_reduced_csv,
        "[MEASURE_WINDOW_DESCRIPTION]": measure_window_description,
        "[BLOCK_INDEX]": block_index_str,
        "[BLOCK_COUNT]": block_count_str,
        "[MELODY_BLOCK_CSV]": melody_block_csv,
        "[MELODY_PREVIOUS_BLOCK_CSV]": melody_previous_block_csv,
        "[VOICE_PREVIOUS_BLOCK_CSV]": voice_previous_block_csv,
        "[OTHER_VOICES_BLOCK_CSV]": other_voices_block_csv,
        "[OTHER_VOICES_PREVIOUS_BLOCK_CSV]": other_voices_previous_block_csv,
    }
    return apply_template(_TEMPLATE_FIRST_SCORE, replacements)


def build_melody_emotion_prompt(
    *,
    prompt_payload: Dict[str, Any],
    tp_index: int,
    tp_type: str,
    tp_title: str,
    tp_description: str,
    tp_prior_paragraph: str,
) -> str:
    payload_for_json = dict(prompt_payload or {})
    character_context = payload_for_json.pop("character_outlines", None)

    instructions = apply_template(_INSTRUCTIONS_MELODY_EMOTION, {})

    replacements = {
        "[VOICE_CONTEXT_JSON]": _json_block(payload_for_json),
        "[CHARACTER_CONTEXT_JSON]": _json_block(character_context or []),
        "[TOUCH_POINT_INDEX]": str(tp_index),
        "[TOUCH_POINT_TYPE]": tp_type,
        "[TOUCH_POINT_TITLE]": tp_title,
        "[TOUCH_POINT_DESCRIPTION]": tp_description,
        "[TOUCH_POINT_PRIOR_PARAGRAPH]": tp_prior_paragraph,
        "[MUSIC_MELODY_EMOTION_INSTRUCTIONS]": instructions,
    }
    return apply_template(_TEMPLATE_MELODY_EMOTION, replacements)


def build_import_sanitize_prompt(
    *,
    prompt_payload: Dict[str, Any],
    tp_index: int,
    tp_type: str,
    tp_title: str,
    tp_description: str,
    tp_prior_paragraph: str,
    melody_voice_token: str,
    import_musiccsv: str,
) -> str:
    payload_for_json = dict(prompt_payload or {})
    character_context = payload_for_json.pop("character_outlines", None)

    replacements = {
        "[VOICE_CONTEXT_JSON]": _json_block(payload_for_json),
        "[CHARACTER_CONTEXT_JSON]": _json_block(character_context or []),
        "[TOUCH_POINT_INDEX]": str(tp_index),
        "[TOUCH_POINT_TYPE]": tp_type,
        "[TOUCH_POINT_TITLE]": tp_title,
        "[TOUCH_POINT_DESCRIPTION]": tp_description,
        "[TOUCH_POINT_PRIOR_PARAGRAPH]": tp_prior_paragraph,
        "[MELODY_VOICE_TOKEN]": melody_voice_token,
        "[SANITIZED_IMPORT]": import_musiccsv.strip(),
    }
    return apply_template(_TEMPLATE_IMPORT_SANITIZE, replacements)


def build_emotion_chord_prompt(
    *,
    prompt_payload: Dict[str, Any],
    tp_index: int,
    tp_type: str,
    tp_title: str,
    tp_description: str,
    tp_prior_paragraph: str,
    feelings: Iterable[str],
    transitions: Iterable[Tuple[str, str]],
) -> str:
    payload_for_json = dict(prompt_payload or {})
    character_context = payload_for_json.pop("character_outlines", None)

    unique_feelings = _coerce_unique_strings(feelings)
    feelings_block = ", ".join(unique_feelings) if unique_feelings else "(none)"

    unique_pairs: list[Tuple[str, str]] = []
    seen_pairs: set[Tuple[str, str]] = set()
    for a, b in transitions:
        a_text = (a or "").strip()
        b_text = (b or "").strip()
        if not a_text or not b_text:
            continue
        key = (a_text.lower(), b_text.lower())
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        unique_pairs.append((a_text, b_text))

    transitions_block = (
        "\n".join(f"{a} -> {b}" for a, b in unique_pairs) if unique_pairs else "(none)"
    )

    instructions = apply_template(
        _INSTRUCTIONS_EMOTION_CHORD,
        {
            "[FEELINGS]": feelings_block,
            "[TRANSITIONS]": transitions_block,
        },
    )

    replacements = {
        "[VOICE_CONTEXT_JSON]": _json_block(payload_for_json),
        "[CHARACTER_CONTEXT_JSON]": _json_block(character_context or []),
        "[TOUCH_POINT_INDEX]": str(tp_index),
        "[TOUCH_POINT_TYPE]": tp_type,
        "[TOUCH_POINT_TITLE]": tp_title,
        "[TOUCH_POINT_DESCRIPTION]": tp_description,
        "[TOUCH_POINT_PRIOR_PARAGRAPH]": tp_prior_paragraph,
        "[EMOTION_CHORD_INSTRUCTIONS]": instructions,
    }
    return apply_template(_TEMPLATE_EMOTION_CHORD, replacements)


def build_music_check_prompt(
    *,
    prompt_payload: Dict[str, Any],
    tp_index: int,
    tp_type: str,
    tp_text: str,
    musiccsv_snippet: str,
) -> str:
    replacements = {
        "[VOICE_CONTEXT_JSON]": _json_block(prompt_payload),
        "[TOUCH_POINT_INDEX]": str(tp_index),
        "[TOUCH_POINT_TYPE]": tp_type,
        "[TOUCH_POINT_TEXT]": tp_text.strip(),
        "[MUSICCSV_SNIPPET]": musiccsv_snippet.strip(),
    }
    return apply_template(_TEMPLATE_CHECK, replacements)


def build_subtle_edit_prompt(
    *,
    prompt_payload: Dict[str, Any],
    tp_index: int,
    tp_type: str,
    tp_text: str,
    previous_score: str,
    author_feedback: str,
) -> str:
    replacements = {
        "[VOICE_CONTEXT_JSON]": _json_block(prompt_payload),
        "[TOUCH_POINT_INDEX]": str(tp_index),
        "[TOUCH_POINT_TYPE]": tp_type,
        "[TOUCH_POINT_TEXT]": tp_text.strip(),
        "[PREVIOUS_SCORE]": previous_score.strip(),
        "[AUTHOR_FEEDBACK]": author_feedback.strip() or "(No feedback supplied.)",
        "[MUSICCSV_FORMAT_PROMPT]": _FORMAT_REFERENCE,
    }
    return apply_template(_TEMPLATE_SUBTLE_EDIT, replacements)


def build_metadata_tracks_prompt(
    *,
    prompt_payload: Dict[str, Any],
    tp_index: int,
    tp_type: str,
    tp_title: str,
    tp_description: str,
    tp_prior_paragraph: str,
) -> str:
    payload_for_json = dict(prompt_payload or {})
    replacements = {
        "[VOICE_CONTEXT_JSON]": _json_block(payload_for_json),
        "[TOUCH_POINT_INDEX]": str(tp_index),
        "[TOUCH_POINT_TYPE]": tp_type,
        "[TOUCH_POINT_TITLE]": tp_title.strip(),
        "[TOUCH_POINT_DESCRIPTION]": tp_description.strip(),
        "[TOUCH_POINT_PRIOR_PARAGRAPH]": tp_prior_paragraph.strip(),
    }
    return apply_template(_TEMPLATE_METADATA_TRACKS, replacements)


__all__ = [
    "build_first_score_prompt",
    "build_melody_emotion_prompt",
    "build_import_sanitize_prompt",
    "build_emotion_chord_prompt",
    "build_music_check_prompt",
    "build_subtle_edit_prompt",
    "build_metadata_tracks_prompt",
]
