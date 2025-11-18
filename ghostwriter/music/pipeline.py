"""Music-specific pipeline helpers for touch-point gates."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import logging

from ..env import env_for_prompt
from ..llm import complete as llm_complete
from ..logging import log_warning as _log_warning, log_info as _log_info
from ..utils import save_text, read_file
from ..context import UserActionRequired
from ..pipelines.common import llm_call_with_validation, reasoning_for_prompt
from ..templates import apply_template as _apply_template
from ..musiccsv import (
    MusicCSV,
    MusicCSVValidationError,
    musiccsv_from_text,
    musiccsv_to_text,
    read_musiccsv,
    validate_musiccsv,
    write_musiccsv,
)

from .context import VoiceContext, reduced_notes_csv
from .prompts import (
    build_first_score_prompt,
    build_music_check_prompt,
    build_subtle_edit_prompt,
    build_metadata_tracks_prompt,
)

logger = logging.getLogger(__name__)

_MAX_MUSICCSV_SNIPPET = 4000


def _truncate_musiccsv_text(text: str, limit: Optional[int]) -> str:
    """Return ``text`` clipped to ``limit`` characters without cutting mid-line."""

    if limit is None or limit <= 0 or len(text) <= limit:
        return text

    cutoff = text.rfind("\n", 0, limit)
    if cutoff == -1:
        cutoff = limit

    truncated = text[:cutoff].rstrip("\n")
    return truncated + "\n# truncated"


def run_metadata_tracks_step(
    *,
    tp_dir: Path,
    tp_index: int,
    tp_type: str,
    prompt_payload: Optional[Dict[str, Any]],
) -> bool:
    """Run the first metadata/tracks-only step for a music touch-point.

    This writes three artifacts when successful:
    - ``metadatatracks.txt`` — prompt + response log for this step.
    - ``metadata.json`` — pretty-printed JSON block for MusicCSV metadata.
    - ``tracks.csv`` — CSV describing track layout.
    """

    if not prompt_payload:
        return False

    tp_dir = Path(tp_dir)
    tp_dir.mkdir(parents=True, exist_ok=True)

    log_path = tp_dir / "metadatatracks.txt"
    metadata_path = tp_dir / "metadata.json"
    tracks_path = tp_dir / "tracks.csv"

    if metadata_path.exists() and tracks_path.exists():
        return False

    title = str(prompt_payload.get("touch_point_title", "") or "")
    description = str(prompt_payload.get("touch_point_description", "") or "")
    prior_paragraph = str(prompt_payload.get("touch_point_prior_paragraph", "") or "")

    prompt = build_metadata_tracks_prompt(
        prompt_payload=prompt_payload,
        tp_index=tp_index,
        tp_type=tp_type,
        tp_title=title,
        tp_description=description,
        tp_prior_paragraph=prior_paragraph,
    )
    model, temp, max_tokens = env_for_prompt(
        "music_metadata_tracks_prompt.md",
        "MUSIC_METADATA_TRACKS",
        default_temp=0.4,
        default_max_tokens=1800,
    )

    response = llm_complete(
        prompt,
        system=(
            "Design score metadata.json and tracks.csv only; "
            "do not generate measures or notes."
        ),
        temperature=temp,
        max_tokens=max_tokens,
        model=model,
    )

    try:
        log_content = [
            "=== SYSTEM ===",
            "Design score metadata.json and tracks.csv only; do not generate measures or notes.",
            "",
            "=== USER ===",
            prompt,
            "",
            "=== RESPONSE ===",
            response,
            "",
        ]
        save_text(log_path, "\n".join(log_content))
    except Exception:
        pass

    text = response or ""
    lower = text.lower()
    meta_idx = lower.find("metadata.json")
    tracks_idx = lower.find("tracks.csv")
    if meta_idx == -1 or tracks_idx == -1 or tracks_idx <= meta_idx:
        raise UserActionRequired(
            "Metadata/tracks step did not return both metadata.json and tracks.csv blocks. "
            "Edit metadatatracks.txt or retry."
        )

    meta_block = text[meta_idx:tracks_idx]
    tracks_block = text[tracks_idx:]

    meta_lines = meta_block.splitlines()[1:]
    tracks_lines = tracks_block.splitlines()[1:]

    meta_text = "\n".join(meta_lines).strip()
    tracks_text = "\n".join(tracks_lines).strip()

    if not meta_text or not tracks_text:
        raise UserActionRequired(
            "Unable to parse metadata.json or tracks.csv from the response. "
            "Edit metadatatracks.txt and retry."
        )

    save_text(metadata_path, meta_text + "\n")
    save_text(tracks_path, tracks_text + "\n")

    try:
        _log_info(
            f"MUSIC: wrote metadata.json and tracks.csv for tp={tp_index:02d} at {tp_dir}",
            tp_dir,
        )
    except Exception:
        pass

    return True


def run_melody_edges_step(
    *,
    tp_dir: Path,
    tp_index: int,
    tp_type: str,
    prompt_payload: Optional[Dict[str, Any]],
) -> bool:
    """Run the melody-elements / edges step for a music touch-point.

    This expects that ``metadata.json`` already exists in ``tp_dir`` and will
    construct a prompt that combines the existing music prompt payload,
    the metadata JSON, and the ``melody_elements_instructions.txt`` template.

    Artifacts written on success:
    - ``melodyelements.txt``  — prompt + response log for this step.
    - ``melody_edges.txt``    — raw dwell/edge description block from the LLM.

    The caller is responsible for raising ``UserActionRequired`` to pause and
    allow human review of the edges after this step completes.
    """

    if not prompt_payload:
        return False

    tp_dir = Path(tp_dir)
    tp_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = tp_dir / "metadata.json"
    if not metadata_path.exists():
        raise UserActionRequired(
            "Melody edges step requires metadata.json to exist. Run metadata/tracks first."
        )

    log_path = tp_dir / "melodyelements.txt"
    edges_path = tp_dir / "melody_edges.txt"

    # Idempotency: if edges already exist, nothing to do.
    if edges_path.exists():
        return False

    title = str(prompt_payload.get("touch_point_title", "") or "")
    description = str(prompt_payload.get("touch_point_description", "") or "")
    prior_paragraph = str(prompt_payload.get("touch_point_prior_paragraph", "") or "")

    # Load story-relative/factoid style context if present in payload
    story_relative = str(prompt_payload.get("story_relative_to_block", "") or "")
    factoids_block = str(prompt_payload.get("factoids_block", "") or "")

    try:
        metadata_text = metadata_path.read_text(encoding="utf-8")
    except Exception as exc:  # pragma: no cover - defensive
        raise UserActionRequired(
            f"Unable to read metadata.json for melody edges step: {exc}"
        ) from exc

    # Load the static melody-elements instructions from prompts
    from pathlib import Path as _P
    base_root = _P(__file__).resolve().parents[2]
    instr_path = base_root / "prompts" / "melody_elements_instructions.txt"
    try:
        instructions_text = read_file(str(instr_path))
    except Exception as exc:  # pragma: no cover - defensive
        raise UserActionRequired(
            f"Unable to read melody_elements_instructions.txt: {exc}"
        ) from exc

    model, temp, max_tokens = env_for_prompt(
        "music_melody_edges_prompt.md",
        "MUSIC_MELODY_EDGES",
        default_temp=0.4,
        default_max_tokens=2200,
    )
    # Build the user prompt from a dedicated template, with the
    # instructions injected as a replacement block.
    replacements: Dict[str, Any] = {
        "[PREVIOUS_PARAGRAPH]": prior_paragraph or "",
        "[STORY_RELATIVE]": story_relative or "",
        "[FACTOIDS]": factoids_block or "",
        "[MUSIC_TOUCH_POINT]": (f"Title: {title}\nDescription: {description}".strip()),
        "[METADATA_JSON]": metadata_text.strip(),
        "[MELODY_ELEMENTS_INSTRUCTIONS]": instructions_text.strip(),
    }
    try:
        user_prompt = _apply_template("prompts/music_melody_edges_prompt.md", {k: str(v) for k, v in replacements.items()})
    except Exception:
        # Fallback: simple concatenation if template application fails.
        lines = []
        lines.append("You are a composer designing melodic dwell notes and edges.")
        lines.append("")
        if prior_paragraph:
            lines.append("[PREVIOUS_PARAGRAPH]")
            lines.append(prior_paragraph)
            lines.append("")
        if story_relative:
            lines.append("[STORY_RELATIVE]")
            lines.append(str(story_relative))
            lines.append("")
        if factoids_block:
            lines.append("[FACTOIDS]")
            lines.append(str(factoids_block))
            lines.append("")
        if title or description:
            lines.append("[MUSIC_TOUCH_POINT]")
            if title:
                lines.append(f"Title: {title}")
            if description:
                lines.append(f"Description: {description}")
            lines.append("")
        lines.append("[METADATA_JSON]")
        lines.append(metadata_text.strip())
        lines.append("")
        lines.append("[MELODY_ELEMENTS_INSTRUCTIONS]")
        lines.append(instructions_text.strip())
        lines.append("")
        user_prompt = "\n".join(lines)

    response = llm_complete(
        user_prompt,
        system=(
            "Use the provided context and metadata to choose four dwell notes "
            "(A, B1/B2, C1, C2) and construct the nine melodic edges as "
            "described. Output first the dwell notes block, then each edge "
            "CSV exactly as specified in the instructions."
        ),
        temperature=temp,
        max_tokens=max_tokens,
        model=model,
    )

    # Always log full context for debugging and human editing.
    try:
        log_lines = [
            "=== SYSTEM ===",
            "Use the provided context and metadata to choose four dwell notes (A, B1/B2, C1, C2) and construct the nine melodic edges as described.",
            "",
            "=== USER ===",
            user_prompt,
            "",
            "=== RESPONSE ===",
            response or "",
            "",
        ]
        save_text(log_path, "\n".join(log_lines))
    except Exception:
        pass

    text = (response or "").strip()
    if not text:
        raise UserActionRequired(
            "Melody edges step produced an empty response. Edit melodyelements.txt or retry."
        )

    # For now, store the raw dwell + edge description for human editing.
    save_text(edges_path, text + "\n")

    try:
        _log_info(
            f"MUSIC: wrote melody edges artifact for tp={tp_index:02d} at {tp_dir}",
            tp_dir,
        )
    except Exception:
        pass

    return True


def _next_attempt_path(tp_dir: Path, prefix: str) -> Path:
    attempt = 1
    while True:
        candidate = tp_dir / f"{prefix}_{attempt}.txt"
        if not candidate.exists():
            return candidate
        attempt += 1


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


def _measures_from_melody_csv(melody_csv_text: str) -> List[Dict[str, Any]]:
    """Return the sorted list of measure numbers present in a melody CSV.

    The melody CSV is expected to have columns (measure, element, duration).
    We ignore the element and duration here and simply discover which
    measures exist so that we can emit a measures.csv that matches the
    MusicCSV schema using metadata for tempo, time signature, and key.
    """

    lines = [ln.strip() for ln in (melody_csv_text or "").splitlines() if ln.strip()]
    if not lines:
        return []
    header = lines[0].lower()
    start_idx = 1 if "measure" in header else 0

    measure_set: set[int] = set()
    for line in lines[start_idx:]:
        parts = [p.strip() for p in line.split(",")]
        if not parts:
            continue
        try:
            measure_no = int(parts[0])
        except Exception:
            continue
        measure_set.add(measure_no)

    rows: List[Dict[str, Any]] = []
    for m in sorted(measure_set):
        rows.append({"measure": m})
    return rows


def _compose_first_pass_for_voice(
    *,
    tp_dir: Path,
    tp_index: int,
    tp_type: str,
    tp_text: str,
    voice_context: VoiceContext,
    prompt_payload: Dict[str, Any],
    target_voice_index: int,
    variant: str = "standard",
    suggestions_text: str = "",
) -> MusicCSV:
    """Compose a first-pass MusicCSV score for a single voice.

    This helper is the internal building block for the multi-voice,
    multi-variant composer. It prepares reduced-note context and
    shared metadata/measures, builds the first-score prompt for the
    requested voice, and returns the validated MusicCSV result.
    """

    tp_dir = Path(tp_dir)
    tp_dir.mkdir(parents=True, exist_ok=True)

    if not voice_context.voices:
        raise UserActionRequired("No voices available in VoiceContext for music composition.")

    if target_voice_index < 0 or target_voice_index >= len(voice_context.voices):
        raise UserActionRequired("Requested target_voice_index is out of range for VoiceContext.voices.")

    # Load shared metadata and variant-specific measures.
    metadata_text = ""
    measures_text = ""
    try:
        meta_path = tp_dir / "metadata.json"
        if meta_path.exists():
            metadata_text = meta_path.read_text(encoding="utf-8").strip()
    except Exception:
        metadata_text = ""
    try:
        safe_variant = (variant or "standard").strip().lower()
        measures_path = tp_dir / f"measures_{safe_variant}.csv"
        if measures_path.exists():
            measures_text = measures_path.read_text(encoding="utf-8").strip()
    except Exception:
        measures_text = ""

    # Build reduced-note grids from any existing scores referenced in the voice context.
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
                music_obj = read_musiccsv(summary.score_path)
                text = musiccsv_to_text(music_obj)
                text = _truncate_musiccsv_text(text, _MAX_MUSICCSV_SNIPPET)
                info["musiccsv"] = text
        except Exception:
            pass
        existing_scores[spec.token] = info

    target_voice = voice_context.voices[target_voice_index]

    melody_reduced = ""
    other_reduced_blocks: List[str] = []
    for token, summary in existing_scores.items():
        score_path_str = summary.get("score_path")
        if not score_path_str:
            continue
        try:
            score_path = Path(score_path_str)
            if not score_path.exists():
                continue
            music_obj = read_musiccsv(score_path)
            reduced = reduced_notes_csv(music_obj)
        except Exception:
            continue
        label = f"# voice: {token}\n{reduced.strip()}" if reduced.strip() else f"# voice: {token} (no notes)"
        if token == target_voice.token:
            melody_reduced = reduced.strip()
        else:
            other_reduced_blocks.append(label)

    other_reduced = "\n\n".join(other_reduced_blocks).strip()

    # Include suggestion text for this pass if provided; callers are
    # responsible for scoping it by variant.
    payload_with_suggestions = dict(prompt_payload or {})
    if suggestions_text.strip():
        key = f"music_suggestions_{(variant or 'standard').strip().lower()}"
        payload_with_suggestions[key] = suggestions_text.strip()

    prompt = build_first_score_prompt(
        prompt_payload=payload_with_suggestions,
        existing_scores=existing_scores,
        tp_index=tp_index,
        tp_type=tp_type,
        tp_text=tp_text,
        voice_token=target_voice.token,
        voice_chord=target_voice.chord,
        voice_register=target_voice.register,
        voice_instrument=target_voice.instrument,
        voice_idea=target_voice.idea,
        voice_role=target_voice.role or "",
        metadata_json=metadata_text,
        measures_csv=measures_text,
        melody_reduced_csv=melody_reduced,
        other_voices_reduced_csv=other_reduced,
    )
    model, temp, max_tokens = env_for_prompt(
        "music_first_score_prompt.md",
        "MUSIC_FIRST_SCORE",
        default_temp=0.5,
        default_max_tokens=2000,
    )
    reasoning = reasoning_for_prompt("music_first_score_prompt.md", "MUSIC_FIRST_SCORE")

    def _log_path_for_attempt(attempt: int) -> Optional[Path]:
        # Persist prompt/response pairs even when validation fails so authors can debug.
        return tp_dir / f"first_score_attempt_{attempt}.txt"

    try:
        response = llm_call_with_validation(
            "Compose a valid MusicCSV score that fits the touch-point context.",
            prompt,
            model=model,
            temperature=temp,
            max_tokens=max_tokens,
            validator=_validate_musiccsv,
            reasoning_effort=reasoning,
            log_maker=_log_path_for_attempt,
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

    return music


def ensure_first_score_gate(
    *,
    tp_dir: Path,
    tp_index: int,
    tp_type: str,
    tp_text: str,
    voice_context: Optional[VoiceContext],
    prompt_payload: Optional[Dict[str, Any]],
) -> bool:
    """Legacy single-score first gate using the internal per-voice composer.

    This wrapper preserves existing behavior for callers that still expect a
    monolithic ``touch_point_first_score.musiccsv`` while internally
    delegating to ``_compose_first_pass_for_voice`` targeting the first
    declared voice and the "standard" variant.
    """

    if voice_context is None or not voice_context.voices or not prompt_payload:
        return False

    tp_dir = Path(tp_dir)
    tp_dir.mkdir(parents=True, exist_ok=True)

    first_score_path = tp_dir / "touch_point_first_score.musiccsv"
    first_suggestions_path = tp_dir / "first_score_suggestions.txt"
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

    music = _compose_first_pass_for_voice(
        tp_dir=tp_dir,
        tp_index=tp_index,
        tp_type=tp_type,
        tp_text=tp_text,
        voice_context=voice_context,
        prompt_payload=prompt_payload,
        target_voice_index=0,
        variant="standard",
    )

    write_musiccsv(first_score_path, music)
    _render_monitor_midi(music, first_monitor_path, tp_dir)

    snippet_text = _truncate_musiccsv_text(musiccsv_to_text(music), None)

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


def run_multi_voice_first_pass(
    *,
    tp_dir: Path,
    tp_index: int,
    tp_type: str,
    tp_text: str,
    voice_context: VoiceContext,
    prompt_payload: Dict[str, Any],
    variant: str = "standard",
    suggestions_text: str = "",
) -> List[Path]:
    """Compose first-pass per-voice scores for a single variant.

    Iterates ``voice_context.voices`` in order, calling the internal
    per-voice composer for each, and writes ``notes_<voice_token>_<variant>.csv``
    files beneath ``tp_dir``. Returns the list of note CSV paths.
    """

    tp_dir = Path(tp_dir)
    tp_dir.mkdir(parents=True, exist_ok=True)

    written: List[Path] = []
    safe_variant = (variant or "standard").strip().lower()

    for idx, voice in enumerate(voice_context.voices):
        notes_path = tp_dir / f"notes_{voice.token}_{safe_variant}.csv"
        if notes_path.exists():
            written.append(notes_path)
            continue

        music = _compose_first_pass_for_voice(
            tp_dir=tp_dir,
            tp_index=tp_index,
            tp_type=tp_type,
            tp_text=tp_text,
            voice_context=voice_context,
            prompt_payload=prompt_payload,
            target_voice_index=idx,
            variant=safe_variant,
            suggestions_text=suggestions_text,
        )

        # Extract notes for the target voice's track index if present; for
        # now, write the entire notes table as a CSV view for that voice.
        try:
            from io import StringIO
            import csv

            output = StringIO()
            writer = csv.writer(output)
            writer.writerow(["measure", "beat", "pitch", "duration"])
            for note in music.notes:
                measure = note.get("measure")
                beat = note.get("beat")
                pitch = note.get("pitch")
                duration = note.get("duration")
                if measure is None or beat is None or pitch is None or duration is None:
                    continue
                writer.writerow([measure, beat, pitch, duration])
            notes_path.write_text(output.getvalue(), encoding="utf-8")
        except Exception as exc:
            _log_warning(f"MUSIC: failed to write notes CSV for {voice.token}: {exc}", tp_dir)
            raise

        written.append(notes_path)

    return written


def run_melody_construction_step(
    *,
    tp_dir: Path,
    tp_index: int,
    tp_type: str,
    prompt_payload: Optional[Dict[str, Any]],
    variant: str = "standard",
) -> bool:
    """Construct a full melody using the music_melody_prompt.md template.

    This step consumes prior artifacts and templates but does *not* yet
    integrate with downstream track-building. It is focused on generating
    a structured melody artifact for later use.

    Expected inputs in ``tp_dir``:
    - ``metadata.json``     — score metadata from the metadata/tracks step.
    - ``melody_edges.txt``  — dwell notes and melodic edges (LLM + human edited).

    Artifacts written on success (for the given variant):
    - ``melody_{variant}.txt``  — prompt + response log for this step.
    - ``melody_{variant}.csv``  — raw melody CSV emitted by the LLM.

    The ``variant`` parameter is meant to support "standard", "complimentary",
    and "reprise" melodies, but this helper does not yet hard-code any
    additional_rules; those should be expressed in the substituted
    MELODY_INSTRUCTIONS template text.
    """

    if not prompt_payload:
        return False

    tp_dir = Path(tp_dir)
    tp_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = tp_dir / "metadata.json"
    edges_path = tp_dir / "melody_edges.txt"
    if not metadata_path.exists() or not edges_path.exists():
        raise UserActionRequired(
            "Melody construction step requires metadata.json and melody_edges.txt. "
            "Run the earlier music steps first."
        )

    # Variant-normalized names
    variant_safe = (variant or "standard").strip().lower()
    log_path = tp_dir / f"melody_{variant_safe}.txt"
    melody_csv_path = tp_dir / f"melody_{variant_safe}.csv"

    # Idempotency: if melody CSV already exists, do nothing.
    if melody_csv_path.exists():
        return False

    title = str(prompt_payload.get("touch_point_title", "") or "")
    description = str(prompt_payload.get("touch_point_description", "") or "")
    prior_paragraph = str(prompt_payload.get("touch_point_prior_paragraph", "") or "")

    story_relative = str(prompt_payload.get("story_relative_to_block", "") or "")
    factoids_block = str(prompt_payload.get("factoids_block", "") or "")

    try:
        metadata_text = metadata_path.read_text(encoding="utf-8")
    except Exception as exc:  # pragma: no cover - defensive
        raise UserActionRequired(
            f"Unable to read metadata.json for melody construction: {exc}"
        ) from exc

    try:
        edges_text = edges_path.read_text(encoding="utf-8")
    except Exception as exc:  # pragma: no cover - defensive
        raise UserActionRequired(
            f"Unable to read melody_edges.txt for melody construction: {exc}"
        ) from exc

    # Split dwell notes vs melodic edges. By convention, the line that
    # begins with "A-A" marks the first edge; everything above it is the
    # dwell-notes description block.
    dwell_block = edges_text.strip()
    edges_block = edges_text.strip()
    try:
        lines = edges_text.splitlines()
        split_index = None
        for idx, line in enumerate(lines):
            if line.strip().startswith("A-A"):
                split_index = idx
                break
        if split_index is not None:
            dwell_block = "\n".join(lines[:split_index]).strip()
            edges_block = "\n".join(lines[split_index:]).strip()
    except Exception:
        # On any parsing failure, fall back to treating the whole text
        # as both dwell and edge context so the prompt still has data.
        dwell_block = edges_text.strip()
        edges_block = edges_text.strip()

    # Load melody_instructions template text for inclusion.
    from pathlib import Path as _P
    base_root = _P(__file__).resolve().parents[2]
    instr_path = base_root / "prompts" / "melody_instructions.txt"
    try:
        melody_instructions = read_file(str(instr_path))
    except Exception as exc:  # pragma: no cover - defensive
        raise UserActionRequired(
            f"Unable to read melody_instructions.txt: {exc}"
        ) from exc

    # First, apply dwell notes, melodic edges, and additional_rules into
    # the melody_instructions template itself.
    variant_safe = (variant or "standard").strip().lower()
    additional_rules = ""
    if variant_safe == "complimentary":
        additional_rules = (
            "* Create a complimentary melody line by inverting the dwell-weight "
            "emphasis across the four dwell notes. For example, if the "
            "standard melody uses dwell weights (1.0, 0.5, 0.25, 0.125), then "
            "a complimentary line might approximate (0.5, 0.75, 0.875, 0.875). "
            "Keep the rhythm structure compatible with the standard melody, "
            "but let the complimentary line weave around it rather than sit "
            "directly on top of the same pitches."
        )
    elif variant_safe == "reprise":
        additional_rules = (
            "* Treat this as a reprise of the standard melody. Before "
            "constructing the final line, conceptually stretch each edge of "
            "the melodic graph by roughly one additional measure, adding "
            "connecting notes that make musical sense so that the total "
            "duration expands while preserving the recognizable contour of "
            "the original melody."
        )

    instr_replacements: Dict[str, Any] = {
        "[dwell_notes]": dwell_block,
        "[melodic_edges]": edges_block,
        "[additional_rules]": additional_rules.strip(),
    }
    rendered_instructions = _apply_template(
        "prompts/melody_instructions.txt",
        {k: str(v) for k, v in instr_replacements.items()},
    )

    # Now build the outer melody prompt using the fully rendered
    # instructions block.
    replacements: Dict[str, Any] = {
        "[PREVIOUS_PARAGRAPH]": prior_paragraph or "",
        "[STORY_RELATIVE]": story_relative or "",
        "[FACTOIDS]": factoids_block or "",
        "[MUSIC_TOUCH_POINT]": (f"Title: {title}\nDescription: {description}".strip()),
        "[METADATA_JSON]": metadata_text.strip(),
        "[MELODY_INSTRUCTIONS]": rendered_instructions.strip(),
    }

    try:
        user_prompt = _apply_template(
            "prompts/music_melody_prompt.md",
            {k: str(v) for k, v in replacements.items()},
        )
    except Exception:
        # Fallback: basic concatenation if templating fails.
        lines = []
        lines.append("You are a composer constructing a full melodic line for this piece.")
        lines.append("")
        if prior_paragraph:
            lines.append("[PREVIOUS_PARAGRAPH]")
            lines.append(prior_paragraph)
            lines.append("")
        if story_relative:
            lines.append("[STORY_RELATIVE]")
            lines.append(str(story_relative))
            lines.append("")
        if factoids_block:
            lines.append("[FACTOIDS]")
            lines.append(str(factoids_block))
            lines.append("")
        if title or description:
            lines.append("[MUSIC_TOUCH_POINT]")
            if title:
                lines.append(f"Title: {title}")
            if description:
                lines.append(f"Description: {description}")
            lines.append("")
        lines.append("[METADATA_JSON]")
        lines.append(metadata_text.strip())
        lines.append("")
        lines.append("[DWELL_NOTES]")
        lines.append(edges_text.strip())
        lines.append("")
        lines.append("[MELODIC_EDGES]")
        lines.append(edges_text.strip())
        lines.append("")
        lines.append("[MELODY_INSTRUCTIONS]")
        lines.append(melody_instructions.strip())
        lines.append("")
        user_prompt = "\n".join(lines)

    model, temp, max_tokens = env_for_prompt(
        "music_melody_prompt.md",
        "MUSIC_MELODY",
        default_temp=0.4,
        default_max_tokens=2200,
    )

    response = llm_complete(
        user_prompt,
        system=(
            "Using the dwell notes, melodic edges, and instructions, "
            "construct a single coherent melody as a CSV table."
        ),
        temperature=temp,
        max_tokens=max_tokens,
        model=model,
    )

    # Log the full context and response for human inspection.
    try:
        log_lines = [
            "=== SYSTEM ===",
            "Using the dwell notes, melodic edges, and instructions, construct a single coherent melody as a CSV table.",
            "",
            "=== USER ===",
            user_prompt,
            "",
            "=== RESPONSE ===",
            response or "",
            "",
        ]
        save_text(log_path, "\n".join(log_lines))
    except Exception:
        pass

    text = (response or "").strip()
    if not text:
        raise UserActionRequired(
            "Melody construction step produced an empty response. Edit the melody log and retry."
        )

    # Save the raw response as the melody CSV artifact.
    save_text(melody_csv_path, text + "\n")

    # Best-effort: derive measures_{variant}.csv files from each melody
    # variant using the MusicCSV measures schema and metadata-derived
    # tempo/time_signature/key_signature. This avoids additional LLM calls.
    try:
        measures_rows = _measures_from_melody_csv(text)
        if measures_rows:
            import csv
            import json

            # Pull defaults from metadata.json when available
            time_sig = ""
            key_sig = ""
            tempo_val: Optional[float] = None
            try:
                meta_obj = json.loads(metadata_text)
                if isinstance(meta_obj, dict):
                    ts = meta_obj.get("time_signature")
                    ks = meta_obj.get("key_signature")
                    tp = meta_obj.get("tempo")
                    if isinstance(ts, str):
                        time_sig = ts
                    if isinstance(ks, str):
                        key_sig = ks
                    try:
                        if tp is not None:
                            tempo_val = float(tp)
                    except Exception:
                        tempo_val = None
            except Exception:
                # If metadata is not valid JSON, fall back to empty/defaults
                pass

            if not time_sig:
                time_sig = "4/4"
            if not key_sig:
                key_sig = "C"
            if tempo_val is None:
                tempo_val = 120.0

            measures_path = tp_dir / f"measures_{variant_safe}.csv"
            with measures_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "measure",
                    "time_signature",
                    "key_signature",
                    "tempo",
                    "start_beat",
                    "pickup",
                ])
                for row in measures_rows:
                    mnum = int(row.get("measure", 0) or 0)
                    writer.writerow([mnum, time_sig, key_sig, tempo_val, 1, "false"])
    except Exception as exc:
        # Log to run.log via warning hook but do not fail the melody step;
        # measures can be regenerated or edited later.
        _log_warning(
            f"MUSIC: failed to synthesize measures_{variant_safe}.csv from melody CSV: {exc}",
            tp_dir,
        )

    try:
        _log_info(
            f"MUSIC: wrote melody CSV for variant '{variant_safe}' tp={tp_index:02d} at {tp_dir}",
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
    subtle_check_trace = tp_dir / "score_check.txt"  # reused name; overwritten after subtle pass
    monitor_mid_path = tp_dir / "monitor.mid"

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
    attempt_log_path = _next_attempt_path(tp_dir, "subtle_score_attempt")
    response = llm_complete(
        prompt,
        system="Refine the MusicCSV score according to feedback while keeping it valid.",
        temperature=temp,
        max_tokens=max_tokens,
        model=model,
    )
    try:
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
        save_text(attempt_log_path, "\n".join(trace_content))
    except Exception:
        pass
    ok, message, music = _try_parse_musiccsv(response)
    if not ok or music is None:
        raise UserActionRequired(
            "Generated subtle score failed validation. Inspect the response and try again."
        )
    write_musiccsv(final_score_path, music)
    _render_monitor_midi(music, monitor_mid_path, tp_dir)

    subtle_snippet = _truncate_musiccsv_text(musiccsv_to_text(music), None)

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
    "run_metadata_tracks_step",
    "run_melody_edges_step",
    "run_melody_construction_step",
    "ensure_first_score_gate",
    "run_subtle_score_pass",
    "run_multi_voice_first_pass",
    "assemble_first_pass_variant",
]


def assemble_first_pass_variant(
    *,
    tp_dir: Path,
    title: str,
    variant: str,
) -> Optional[Path]:
    """Assemble a first-pass MusicCSV and monitor MIDI for a single variant.

    This helper loads ``metadata.json``, ``tracks.csv``, and
    ``measures_<variant>.csv`` from ``tp_dir``, merges all
    ``notes_<voice_token>_<variant>.csv`` files that follow the
    canonical naming convention into a single ``MusicCSV`` object, and
    writes ``first_<title>_<variant>.musiccsv`` plus
    ``first_monitor_<title>_<variant>.mid``.

    Returns the path to the assembled ``first_*.musiccsv`` or ``None``
    if required inputs are missing. No LLM calls are involved.
    """

    tp_dir = Path(tp_dir)
    safe_variant = (variant or "standard").strip().lower()

    metadata_path = tp_dir / "metadata.json"
    tracks_path = tp_dir / "tracks.csv"
    measures_path = tp_dir / f"measures_{safe_variant}.csv"
    if not (metadata_path.exists() and tracks_path.exists() and measures_path.exists()):
        return None

    # Start from an empty MusicCSV and populate core tables.
    music = MusicCSV(metadata={}, measures=[], tracks=[], notes=[])

    import csv

    # Load metadata.json (as JSON dict) if possible; otherwise, treat as opaque text.
    try:
        import json as _json

        meta_text = metadata_path.read_text(encoding="utf-8")
        meta_obj = _json.loads(meta_text)
        if isinstance(meta_obj, dict):
            music.metadata = meta_obj
        else:
            music.metadata = {"raw": meta_text}
    except Exception:
        try:
            music.metadata = {"raw": metadata_path.read_text(encoding="utf-8")}
        except Exception:
            music.metadata = {}

    # Load tracks.csv verbatim.
    try:
        with tracks_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            music.tracks = [dict(row) for row in reader]
    except Exception as exc:
        _log_warning(f"MUSIC: failed to read tracks.csv for first-pass assembly: {exc}", tp_dir)
        return None

    # Load measures_<variant>.csv verbatim.
    try:
        with measures_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            music.measures = [dict(row) for row in reader]
    except Exception as exc:
        _log_warning(f"MUSIC: failed to read measures_{safe_variant}.csv for first-pass assembly: {exc}", tp_dir)
        return None

    # Collect all notes_<voice_token>_<variant>.csv files under tp_dir.
    notes_files: List[Path] = []
    suffix = f"_{safe_variant}.csv"
    for path in tp_dir.glob("notes_*_*.csv"):
        name = path.name
        if not name.endswith(suffix):
            continue
        notes_files.append(path)

    if not notes_files:
        # Nothing to assemble for this variant.
        return None

    # Append note rows from each notes CSV. At this stage we treat all
    # notes as belonging to a single logical score; track assignment is
    # left to the LLM and metadata/tracks.csv.
    for path in sorted(notes_files):
        try:
            with path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Ensure minimal required columns exist.
                    if not row.get("measure") or not row.get("beat") or not row.get("pitch"):
                        continue
                    music.notes.append(dict(row))
        except Exception as exc:
            _log_warning(f"MUSIC: failed to merge notes from {path}: {exc}", tp_dir)

    # Validate and write out the assembled first-pass score.
    try:
        validate_musiccsv(music)
    except Exception as exc:
        _log_warning(f"MUSIC: assembled first-pass variant failed validation: {exc}", tp_dir)
        # Continue anyway; caller can inspect the artifact.

    safe_title = str(title or "untitled").strip().replace(" ", "_")
    first_score_path = tp_dir / f"first_{safe_title}_{safe_variant}.musiccsv"
    write_musiccsv(first_score_path, music)

    # Render monitor MIDI for this variant.
    first_monitor_path = tp_dir / f"first_monitor_{safe_title}_{safe_variant}.mid"
    _render_monitor_midi(music, first_monitor_path, tp_dir)

    try:
        _log_info(
            f"MUSIC: assembled first-pass variant '{safe_variant}' at {first_score_path}",
            tp_dir,
        )
    except Exception:
        pass

    return first_score_path
