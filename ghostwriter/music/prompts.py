"""Helpers for assembling music pipeline LLM prompts."""
from __future__ import annotations

import json
from typing import Any, Dict

from ..templates import apply_template


_TEMPLATE_FIRST_SCORE = "prompts/music_first_score_prompt.md"
_TEMPLATE_METADATA_TRACKS = "prompts/music_metadata_tracks_prompt.md"
_TEMPLATE_CHECK = "prompts/music_check_prompt.md"
_TEMPLATE_SUBTLE_EDIT = "prompts/music_subtle_edit_prompt.md"
_TEMPLATE_CSV_FORMAT = "prompts/musiccsv_format_prompt.txt"


_FORMAT_REFERENCE = apply_template(_TEMPLATE_CSV_FORMAT, {})


def _json_block(data: Any) -> str:
    try:
        return json.dumps(data, ensure_ascii=False, indent=2)
    except Exception:
        return str(data)


def build_first_score_prompt(
    *,
    prompt_payload: Dict[str, Any],
    existing_scores: Dict[str, Any],
    tp_index: int,
    tp_type: str,
    tp_text: str,
) -> str:
    payload_for_json = dict(prompt_payload)
    character_context = payload_for_json.pop("character_outlines", None)
    replacements = {
        "[VOICE_CONTEXT_JSON]": _json_block(payload_for_json),
        "[CHARACTER_CONTEXT_JSON]": _json_block(character_context or []),
        "[EXISTING_SCORES_JSON]": _json_block(existing_scores) or "{}",
        "[TOUCH_POINT_INDEX]": str(tp_index),
        "[TOUCH_POINT_TYPE]": tp_type,
        "[TOUCH_POINT_TEXT]": tp_text.strip(),
        "[MUSICCSV_FORMAT_PROMPT]": _FORMAT_REFERENCE,
    }
    return apply_template(_TEMPLATE_FIRST_SCORE, replacements)


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
    "build_music_check_prompt",
    "build_subtle_edit_prompt",
    "build_metadata_tracks_prompt",
]
