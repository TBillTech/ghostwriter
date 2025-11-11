"""Helpers for assembling music pipeline LLM prompts."""
from __future__ import annotations

import json
from typing import Any, Dict

from ..templates import apply_template


_TEMPLATE_FIRST_SCORE = "prompts/music_first_score_prompt.md"
_TEMPLATE_CHECK = "prompts/music_check_prompt.md"
_TEMPLATE_SUBTLE_EDIT = "prompts/music_subtle_edit_prompt.md"


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
    replacements = {
        "[VOICE_CONTEXT_JSON]": _json_block(prompt_payload),
        "[EXISTING_SCORES_JSON]": _json_block(existing_scores) or "{}",
        "[TOUCH_POINT_INDEX]": str(tp_index),
        "[TOUCH_POINT_TYPE]": tp_type,
        "[TOUCH_POINT_TEXT]": tp_text.strip(),
    }
    return apply_template(_TEMPLATE_FIRST_SCORE, replacements)


def build_music_check_prompt(
    *,
    prompt_payload: Dict[str, Any],
    tp_index: int,
    tp_type: str,
    tp_text: str,
    musicxml_snippet: str,
) -> str:
    replacements = {
        "[VOICE_CONTEXT_JSON]": _json_block(prompt_payload),
        "[TOUCH_POINT_INDEX]": str(tp_index),
        "[TOUCH_POINT_TYPE]": tp_type,
        "[TOUCH_POINT_TEXT]": tp_text.strip(),
        "[MUSICXML_SNIPPET]": musicxml_snippet.strip(),
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
    }
    return apply_template(_TEMPLATE_SUBTLE_EDIT, replacements)


__all__ = [
    "build_first_score_prompt",
    "build_music_check_prompt",
    "build_subtle_edit_prompt",
]
