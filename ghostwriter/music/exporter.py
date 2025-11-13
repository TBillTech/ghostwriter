"""Music score aggregation and export helpers (Slice 5).

This module assembles per-touch-point MusicCSV artifacts into version-level
bundles and renders MIDI exports for downstream consumption. It mirrors the
prose export flow by producing deterministic outputs inside the chapter's
iteration directory.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import re
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from ..templates import iter_dir_for
from ..musiccsv import (
    MusicCSV,
    read_musiccsv,
    resolve_derived_fields,
    validate_musiccsv,
    write_musiccsv,
)
from .context import VoiceContext

logger = logging.getLogger(__name__)


@dataclass
class _TouchPointScore:
    index: int
    tp_type: str
    source_path: Path
    finalized: bool


def _parse_tp_dir_name(name: str) -> Tuple[Optional[int], str]:
    try:
        prefix, tp_type = name.split("_", 1)
    except ValueError:
        return None, ""
    try:
        return int(prefix), tp_type
    except ValueError:
        return None, tp_type


def _collect_touchpoint_scores(pipeline_dir: Path) -> List[_TouchPointScore]:
    scores: List[_TouchPointScore] = []
    for child in sorted(pipeline_dir.iterdir()):
        if not child.is_dir():
            continue
        idx, tp_type = _parse_tp_dir_name(child.name)
        if idx is None:
            continue
        final_path = child / "touch_point_score.musiccsv"
        first_path = child / "touch_point_first_score.musiccsv"
        if final_path.exists():
            scores.append(_TouchPointScore(idx, tp_type, final_path, True))
        elif first_path.exists():
            scores.append(_TouchPointScore(idx, tp_type, first_path, False))
    return sorted(scores, key=lambda item: item.index)


def _safe_filename(token: str) -> str:
    text = token.strip().lower().replace(" ", "_")
    text = re.sub(r"[^a-z0-9._-]", "_", text)
    return text or "voice"


def _relative_to(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _piece_duration_beats(derived: MusicCSV) -> float:
    max_end = 0.0
    for note in derived.notes:
        absolute = float(note.get("absolute_beat", 0.0) or 0.0)
        duration = float(note.get("duration", 0.0) or 0.0)
        end = absolute + duration
        if end > max_end:
            max_end = end
    for measure in derived.measures:
        start = float(measure.get("start_beat", 0.0) or 0.0)
        beats = float(measure.get("beats_per_measure", 0.0) or 0.0)
        end = start + beats
        if end > max_end:
            max_end = end
    return max_end


def _merge_scores(
    scores: Iterable[Tuple[_TouchPointScore, MusicCSV]],
    *,
    chapter_id: str,
    version: int,
) -> Tuple[MusicCSV, List[Dict[str, object]]]:
    scores_list = list(scores)
    if not scores_list:
        raise ValueError("No scores provided for aggregation")

    manifest_entries: List[Dict[str, object]] = []
    aggregate_tracks: List[Dict[str, object]] = []
    aggregate_measures: List[Dict[str, object]] = []
    aggregate_notes: List[Dict[str, object]] = []

    base_metadata = dict(scores_list[0][1].metadata or {})
    base_metadata.setdefault("title", f"{chapter_id} score v{version}")
    base_metadata["version"] = str(version)

    next_track_id = 1
    next_measure_number = 1
    beat_offset = 0.0

    for item, music in scores_list:
        manifest_entries.append(
            {
                "index": item.index,
                "type": item.tp_type,
                "path": str(item.source_path),
                "finalized": item.finalized,
            }
        )

        track_map: Dict[int, int] = {}
        for track in sorted(music.tracks, key=lambda row: int(row.get("track", 0) or 0)):
            new_track_id = next_track_id
            next_track_id += 1
            track_map[int(track.get("track", 0) or 0)] = new_track_id
            aggregate_tracks.append(
                {
                    "track": new_track_id,
                    "label": track.get("label"),
                    "part": track.get("part"),
                    "instrument": track.get("instrument"),
                    "channel": track.get("channel"),
                    "program": track.get("program"),
                    "volume": track.get("volume"),
                }
            )

        derived = resolve_derived_fields(music)
        derived_map = {
            int(row.get("measure", 0) or 0): row for row in derived.measures
        }
        measure_map: Dict[int, int] = {}
        measures_sorted = sorted(music.measures, key=lambda row: int(row.get("measure", 0) or 0))
        for measure in measures_sorted:
            original_number = int(measure.get("measure", 0) or 0)
            if original_number <= 0:
                continue
            derived_measure = derived_map.get(original_number, {})
            start_beat = derived_measure.get("start_beat")
            if start_beat is None:
                start_beat = measure.get("start_beat")
            beats_per_measure = derived_measure.get("beats_per_measure")
            if start_beat is None and beats_per_measure is not None:
                start_beat = (original_number - 1) * float(beats_per_measure)

            new_measure_number = next_measure_number
            next_measure_number += 1
            measure_map[original_number] = new_measure_number

            aggregate_measures.append(
                {
                    "measure": new_measure_number,
                    "time_signature": measure.get("time_signature"),
                    "key_signature": measure.get("key_signature"),
                    "tempo": measure.get("tempo"),
                    "start_beat": (float(start_beat) + beat_offset) if start_beat is not None else None,
                    "pickup": measure.get("pickup"),
                }
            )

        for note in music.notes:
            original_track = int(note.get("track", 0) or 0)
            new_track = track_map.get(original_track)
            if new_track is None:
                continue
            original_measure = int(note.get("measure", 0) or 0)
            new_measure = measure_map.get(original_measure)
            if new_measure is None:
                continue
            aggregate_notes.append(
                {
                    "track": new_track,
                    "measure": new_measure,
                    "beat": note.get("beat"),
                    "pitch": note.get("pitch"),
                    "duration": note.get("duration"),
                    "velocity": note.get("velocity"),
                    "tie": note.get("tie"),
                    "articulation": note.get("articulation"),
                    "pedal": note.get("pedal"),
                    "lyric": note.get("lyric"),
                    "ornament": note.get("ornament"),
                    "comment": note.get("comment"),
                    "grace": note.get("grace"),
                    "repeat": note.get("repeat"),
                    "tuplet": note.get("tuplet"),
                }
            )

    beat_offset += _piece_duration_beats(derived)

    aggregate_music = MusicCSV(
        metadata=base_metadata,
        tracks=aggregate_tracks,
        measures=aggregate_measures,
        notes=aggregate_notes,
    )
    validate_musiccsv(aggregate_music)
    return aggregate_music, manifest_entries


def finalize_music_exports(
    chapter_id: str,
    version: int,
    *,
    voice_context: Optional[VoiceContext] = None,
) -> Dict[str, object]:
    """Aggregate per-touch-point scores into version-level exports.

    Parameters
    ----------
    chapter_id:
        Identifier for the chapter (e.g., ``CHAPTER_001``).
    version:
        Current pipeline version number (``vN``).
    voice_context:
        Optional :class:`VoiceContext` used to derive per-voice metadata and
        render per-voice MIDI exports. When omitted, aggregation still occurs
        but individual voice renders are skipped.

    Returns
    -------
    dict
        Metadata about generated artifacts. ``{"written": False}`` is
        returned when no aggregated exports were produced.
    """

    chapter_dir = iter_dir_for(chapter_id)
    pipeline_dir = chapter_dir / f"pipeline_v{version}"
    if not pipeline_dir.exists():
        return {"written": False, "reason": "pipeline_dir_missing"}

    tp_scores = _collect_touchpoint_scores(pipeline_dir)
    if not tp_scores:
        return {"written": False, "reason": "no_touchpoint_scores"}

    loaded_scores: List[Tuple[_TouchPointScore, MusicCSV]] = []
    for item in tp_scores:
        try:
            music = read_musiccsv(item.source_path)
            validate_musiccsv(music)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("MUSIC export: skipping %s (%s)", item.source_path, exc)
            continue
        loaded_scores.append((item, music))

    if not loaded_scores:
        return {"written": False, "reason": "no_valid_scores"}

    aggregate_music, tp_manifest = _merge_scores(loaded_scores, chapter_id=chapter_id, version=version)

    chapter_dir.mkdir(parents=True, exist_ok=True)
    target_score = chapter_dir / f"score_v{version}.musiccsv"
    write_musiccsv(target_score, aggregate_music)

    score_dir = chapter_dir / "score"
    score_dir.mkdir(parents=True, exist_ok=True)
    final_musiccsv = score_dir / "final.musiccsv"
    write_musiccsv(final_musiccsv, aggregate_music)

    per_voice_midis: List[Path] = []
    voice_manifest: List[Dict[str, object]] = []

    final_midi_path = score_dir / "final.mid"
    try:
        aggregate_music.to_midi(final_midi_path)
        final_midi: Optional[Path] = final_midi_path
    except Exception as exc:  # pragma: no cover - optional dependencies
        logger.warning("MUSIC export: failed to render final.mid (%s)", exc)
        final_midi = None

    if voice_context is not None:
        for spec in voice_context.voices:
            entry: Dict[str, object] = {
                "token": spec.token,
                "instrument": spec.instrument,
                "role": spec.role,
                "midi": None,
                "score_source": None,
            }
            summary = voice_context.score_summaries.get(spec.token)
            if summary and summary.score_path.exists():
                entry["score_source"] = _relative_to(summary.score_path, chapter_dir)
                midi_path = score_dir / f"{_safe_filename(spec.token)}.mid"
                try:
                    music = read_musiccsv(summary.score_path)
                    music.to_midi(midi_path)
                    entry["midi"] = _relative_to(midi_path, chapter_dir)
                    per_voice_midis.append(midi_path)
                except Exception as exc:  # pragma: no cover - defensive logging
                    logger.warning("MUSIC export: failed to render MIDI for %s (%s)", spec.token, exc)
            voice_manifest.append(entry)
    else:
        voice_manifest = []

    manifest = {
        "chapter_id": chapter_id,
        "version": version,
        "directive": getattr(voice_context, "directive", ""),
        "touch_points": tp_manifest,
        "voices": voice_manifest,
        "missing_voice_assets": getattr(voice_context, "missing_assets", []),
        "outputs": {
            "score_musiccsv": _relative_to(target_score, chapter_dir),
            "final_musiccsv": _relative_to(final_musiccsv, chapter_dir),
            "final_midi": _relative_to(final_midi, chapter_dir) if final_midi else None,
            "per_voice_midis": [
                _relative_to(path, chapter_dir)
                for path in per_voice_midis
            ],
        },
    }

    manifest_path = score_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    bundle_path = score_dir / f"score_bundle_v{version}.zip"
    bundle_files: List[Tuple[Path, str]] = [
        (target_score, target_score.name),
        (final_musiccsv, f"score/{final_musiccsv.name}"),
    ]
    if final_midi:
        bundle_files.append((final_midi, f"score/{final_midi.name}"))
    for midi_path in per_voice_midis:
        bundle_files.append((midi_path, f"score/{midi_path.name}"))
    bundle_files.append((manifest_path, f"score/{manifest_path.name}"))

    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path, arcname in bundle_files:
            if file_path and file_path.exists():
                zf.write(str(file_path), arcname)

    return {
        "written": True,
        "score_path": target_score,
        "final_musiccsv": final_musiccsv,
        "final_midi": final_midi,
        "per_voice_midis": per_voice_midis,
        "manifest_path": manifest_path,
        "bundle_path": bundle_path,
        "touch_points": tp_manifest,
    }


__all__ = ["finalize_music_exports"]
