"""Music-specific pipeline helpers for touch-point gates."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional
import logging

from ..env import env_for_prompt
from ..llm import complete as llm_complete
from ..logging import log_warning as _log_warning, log_info as _log_info
from ..utils import save_text

from .context import VoiceContext
from .prompts import (
    build_first_score_prompt,
    build_music_check_prompt,
    build_subtle_edit_prompt,
)

logger = logging.getLogger(__name__)

_MAX_XML_SNIPPET = 4000


def ensure_first_score_gate(
    *,
    tp_dir: Path,
    tp_index: int,
    tp_type: str,
    tp_text: str,
    voice_context: Optional[VoiceContext],
    prompt_payload: Optional[Dict[str, Any]],
) -> bool:
    """Run the first-score gate if required, returning True when artifacts were generated."""

    if voice_context is None or not voice_context.voices or not prompt_payload:
        return False

    tp_dir = Path(tp_dir)
    tp_dir.mkdir(parents=True, exist_ok=True)

    first_score_path = tp_dir / "touch_point_first_score.musicxml"
    first_suggestions_path = tp_dir / "first_score_suggestions.txt"

    if first_score_path.exists() and first_suggestions_path.exists():
        return False

    existing_scores: Dict[str, Dict[str, Any]] = {}
    for spec in voice_context.voices:
        summary = voice_context.score_summaries.get(spec.token)
        if not summary:
            continue
        info: Dict[str, Any] = {
            "token": spec.token,
            "score_path": str(summary.score_path),
            "tempos": summary.tempos,
            "time_signatures": summary.time_signatures,
            "key_signatures": summary.key_signatures,
        }
        try:
            if summary.score_path.exists():
                xml = summary.score_path.read_text(encoding="utf-8")
                if len(xml) > _MAX_XML_SNIPPET:
                    xml = xml[:_MAX_XML_SNIPPET] + "\n<!-- truncated -->"
                info["musicxml"] = xml
        except Exception:
            pass
        existing_scores[spec.token] = info

    prompt = build_first_score_prompt(
        prompt_payload=prompt_payload,
        existing_scores=existing_scores,
        tp_index=tp_index,
        tp_type=tp_type,
        tp_text=tp_text,
    )
    model, temp, max_tokens = env_for_prompt(
        "music_first_score_prompt.md",
        "MUSIC_FIRST_SCORE",
        default_temp=0.5,
        default_max_tokens=2000,
    )
    try:
        response = llm_complete(
            prompt,
            system="Compose a valid MusicXML score that fits the touch-point context.",
            temperature=temp,
            max_tokens=max_tokens,
            model=model,
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        _log_warning(f"MUSIC first-score generation failed: {exc}", tp_dir)
        raise

    save_text(first_score_path, response)

    check_prompt = build_music_check_prompt(
        prompt_payload=prompt_payload,
        tp_index=tp_index,
        tp_type=tp_type,
        tp_text=tp_text,
        musicxml_snippet=response,
    )
    check_model, check_temp, check_max = env_for_prompt(
        "music_check_prompt.md",
        "MUSIC_SCORE_CHECK",
        default_temp=0.0,
        default_max_tokens=800,
    )
    suggestions = llm_complete(
        check_prompt,
        system="Provide concise, actionable feedback on the score.",
        temperature=check_temp,
        max_tokens=check_max,
        model=check_model,
    )
    save_text(first_suggestions_path, suggestions)

    try:
        _log_info(
            f"MUSIC: generated first score gate artifacts at {tp_dir}",
            tp_dir,
        )
    except Exception:
        pass

    return True


def run_subtle_score_pass(
    *,
    tp_dir: Path,
    tp_index: int,
    tp_type: str,
    tp_text: str,
    voice_context: Optional[VoiceContext],
    prompt_payload: Optional[Dict[str, Any]],
) -> bool:
    """Run the subtle score pass if edited first-score artifacts are present."""

    if voice_context is None or not voice_context.voices or not prompt_payload:
        return False

    tp_dir = Path(tp_dir)

    first_score_path = tp_dir / "touch_point_first_score.musicxml"
    first_feedback_path = tp_dir / "first_score_suggestions.txt"
    final_score_path = tp_dir / "touch_point_score.musicxml"
    final_feedback_path = tp_dir / "score_suggestions.txt"

    if not first_score_path.exists():
        return False

    if final_score_path.exists() and final_feedback_path.exists():
        return False

    try:
        previous_score = first_score_path.read_text(encoding="utf-8")
    except Exception as exc:
        _log_warning(f"MUSIC subtle pass skipped; unable to read first score: {exc}", tp_dir)
        return False

    try:
        author_feedback = first_feedback_path.read_text(encoding="utf-8") if first_feedback_path.exists() else ""
    except Exception:
        author_feedback = ""

    prompt = build_subtle_edit_prompt(
        prompt_payload=prompt_payload,
        tp_index=tp_index,
        tp_type=tp_type,
        tp_text=tp_text,
        previous_score=previous_score,
        author_feedback=author_feedback,
    )
    model, temp, max_tokens = env_for_prompt(
        "music_subtle_edit_prompt.md",
        "MUSIC_SUBTLE_EDIT",
        default_temp=0.4,
        default_max_tokens=2200,
    )
    response = llm_complete(
        prompt,
        system="Refine the MusicXML score according to feedback while keeping it valid.",
        temperature=temp,
        max_tokens=max_tokens,
        model=model,
    )
    save_text(final_score_path, response)

    check_prompt = build_music_check_prompt(
        prompt_payload=prompt_payload,
        tp_index=tp_index,
        tp_type=tp_type,
        tp_text=tp_text,
        musicxml_snippet=response,
    )
    check_model, check_temp, check_max = env_for_prompt(
        "music_check_prompt.md",
        "MUSIC_SCORE_CHECK",
        default_temp=0.0,
        default_max_tokens=800,
    )
    suggestions = llm_complete(
        check_prompt,
        system="Provide concise, actionable feedback on the score.",
        temperature=check_temp,
        max_tokens=check_max,
        model=check_model,
    )
    save_text(final_feedback_path, suggestions)

    try:
        _log_info(
            f"MUSIC: completed subtle score pass at {tp_dir}",
            tp_dir,
        )
    except Exception:
        pass

    return True


__all__ = [
    "ensure_first_score_gate",
    "run_subtle_score_pass",
]
