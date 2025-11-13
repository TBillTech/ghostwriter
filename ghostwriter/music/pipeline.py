"""Music-specific pipeline helpers for touch-point gates."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging

from ..env import env_for_prompt
from ..llm import complete as llm_complete
from ..logging import log_warning as _log_warning, log_info as _log_info
from ..utils import save_text
from ..context import UserActionRequired
from ..pipelines.common import llm_call_with_validation, reasoning_for_prompt
from ..musiccsv import (
    MusicCSV,
    MusicCSVValidationError,
    musiccsv_from_text,
    musiccsv_to_text,
    read_musiccsv,
    validate_musiccsv,
    write_musiccsv,
)

from .context import VoiceContext
from .prompts import (
    build_first_score_prompt,
    build_music_check_prompt,
    build_subtle_edit_prompt,
)

logger = logging.getLogger(__name__)

_MAX_MUSICCSV_SNIPPET = 4000


def _try_parse_musiccsv(text: str) -> Tuple[bool, str, Optional[MusicCSV]]:
    stripped = (text or "").strip()
    if not stripped:
        return False, "empty response", None
    try:
        music = musiccsv_from_text(stripped)
        validate_musiccsv(music)
        return True, "", music
    except MusicCSVValidationError as exc:
        return False, str(exc), None
    except Exception as exc:  # pragma: no cover - defensive
        return False, str(exc), None


def _validate_musiccsv(text: str) -> Tuple[bool, str]:
    ok, message, _ = _try_parse_musiccsv(text)
    return ok, message


def _render_monitor_midi(music: MusicCSV, midi_path: Path, log_dir: Optional[Path]) -> bool:
    try:
        midi_path.parent.mkdir(parents=True, exist_ok=True)
        music.to_midi(midi_path)
        return True
    except Exception as exc:  # pragma: no cover - defensive logging
        _log_warning(f"MUSIC: failed to render monitor MIDI: {exc}", log_dir)
        raise UserActionRequired(
            "Unable to render monitor MIDI from the generated MusicCSV. Inspect the score and resolve the rendering error."
        ) from exc


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

    first_score_path = tp_dir / "touch_point_first_score.musiccsv"
    first_suggestions_path = tp_dir / "first_score_suggestions.txt"
    first_score_trace = tp_dir / "first_score.txt"
    score_check_trace = tp_dir / "score_check.txt"
    first_monitor_path = tp_dir / "first_monitor.mid"

    if first_score_path.exists() and first_suggestions_path.exists():
        if first_monitor_path.exists():
            return False
        try:
            music = read_musiccsv(first_score_path)
        except Exception as exc:
            _log_warning(f"MUSIC: unable to read first score for monitor MIDI: {exc}", tp_dir)
            raise UserActionRequired(
                "Unable to read the stored MusicCSV first score to render the monitor MIDI. Inspect the score and try again."
            ) from exc
        _render_monitor_midi(music, first_monitor_path, tp_dir)
        return True

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
                music = read_musiccsv(summary.score_path)
                text = musiccsv_to_text(music)
                if len(text) > _MAX_MUSICCSV_SNIPPET:
                    text = text[:_MAX_MUSICCSV_SNIPPET] + "\n# truncated"
                info["musiccsv"] = text
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
    reasoning = reasoning_for_prompt("music_first_score_prompt.md", "MUSIC_FIRST_SCORE")
    try:
        response = llm_call_with_validation(
            "Compose a valid MusicCSV score that fits the touch-point context.",
            prompt,
            model=model,
            temperature=temp,
            max_tokens=max_tokens,
            validator=_validate_musiccsv,
            reasoning_effort=reasoning,
            context_tag=f"music_first_score tp={tp_index:02d}",
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        _log_warning(f"MUSIC first-score generation failed: {exc}", tp_dir)
        raise

    ok, message, music = _try_parse_musiccsv(response)
    if not ok or music is None:  # pragma: no cover - defensive
        raise UserActionRequired(
            "Generated score failed validation after LLM call. Inspect the response and try again."
        )
    write_musiccsv(first_score_path, music)
    try:
        if not first_score_trace.exists():
            trace_content = [
                "=== SYSTEM ===",
                "Compose a valid MusicCSV score that fits the touch-point context.",
                "",
                "=== USER ===",
                prompt,
                "",
                "=== RESPONSE ===",
                response,
                "",
            ]
            save_text(first_score_trace, "\n".join(trace_content))
    except Exception:
        pass

    _render_monitor_midi(music, first_monitor_path, tp_dir)

    snippet_text = musiccsv_to_text(music)
    if len(snippet_text) > _MAX_MUSICCSV_SNIPPET:
        snippet_text = snippet_text[:_MAX_MUSICCSV_SNIPPET] + "\n# truncated"

    check_prompt = build_music_check_prompt(
        prompt_payload=prompt_payload,
        tp_index=tp_index,
        tp_type=tp_type,
        tp_text=tp_text,
        musiccsv_snippet=snippet_text,
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
        if not score_check_trace.exists():
            trace_content = [
                "=== SYSTEM ===",
                "Provide concise, actionable feedback on the score.",
                "",
                "=== USER ===",
                check_prompt,
                "",
                "=== RESPONSE ===",
                suggestions,
                "",
            ]
            save_text(score_check_trace, "\n".join(trace_content))
    except Exception:
        pass

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

    first_score_path = tp_dir / "touch_point_first_score.musiccsv"
    first_feedback_path = tp_dir / "first_score_suggestions.txt"
    final_score_path = tp_dir / "touch_point_score.musiccsv"
    final_feedback_path = tp_dir / "score_suggestions.txt"
    subtle_score_trace = tp_dir / "subtle_score.txt"
    subtle_check_trace = tp_dir / "score_check.txt"  # reused name; overwritten after subtle pass

    if not first_score_path.exists():
        return False

    if final_score_path.exists() and final_feedback_path.exists():
        return False

    try:
        first_music = read_musiccsv(first_score_path)
        previous_score = musiccsv_to_text(first_music)
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
        system="Refine the MusicCSV score according to feedback while keeping it valid.",
        temperature=temp,
        max_tokens=max_tokens,
        model=model,
    )
    ok, message, music = _try_parse_musiccsv(response)
    if not ok or music is None:
        raise UserActionRequired(
            "Generated subtle score failed validation. Inspect the response and try again."
        )
    write_musiccsv(final_score_path, music)
    try:
        if not subtle_score_trace.exists():
            trace_content = [
                "=== SYSTEM ===",
                "Refine the MusicCSV score according to feedback while keeping it valid.",
                "",
                "=== USER ===",
                prompt,
                "",
                "=== RESPONSE ===",
                response,
                "",
            ]
            save_text(subtle_score_trace, "\n".join(trace_content))
    except Exception:
        pass

    subtle_snippet = musiccsv_to_text(music)
    if len(subtle_snippet) > _MAX_MUSICCSV_SNIPPET:
        subtle_snippet = subtle_snippet[:_MAX_MUSICCSV_SNIPPET] + "\n# truncated"

    check_prompt = build_music_check_prompt(
        prompt_payload=prompt_payload,
        tp_index=tp_index,
        tp_type=tp_type,
        tp_text=tp_text,
        musiccsv_snippet=subtle_snippet,
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
        trace_content = [
            "=== SYSTEM ===",
            "Provide concise, actionable feedback on the score.",
            "",
            "=== USER ===",
            check_prompt,
            "",
            "=== RESPONSE ===",
            suggestions,
            "",
        ]
        save_text(subtle_check_trace, "\n".join(trace_content))
    except Exception:
        pass

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
