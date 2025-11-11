"""Music score aggregation and export helpers (Slice 5).

This module assembles per-touch-point MusicXML artifacts into version-level
bundles and renders MIDI exports for downstream consumption. It mirrors the
prose export flow by producing deterministic outputs inside the chapter's
iteration directory.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:  # pragma: no cover - gated in tests via pytest.importorskip
    from music21 import converter, metadata, stream
except ImportError as exc:  # pragma: no cover - surfaced to caller
    raise ImportError(
        "music21 is required for the music export helpers. Install optional music"
        " dependencies via `pip install -r requirements.txt`."
    ) from exc

from ..templates import iter_dir_for
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
        final_path = child / "touch_point_score.musicxml"
        first_path = child / "touch_point_first_score.musicxml"
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


def _build_opus(scores: Iterable[_TouchPointScore]) -> Tuple[Any, List[Dict[str, object]]]:
    opus: Any = stream.Opus()  # type: ignore[attr-defined]
    manifest_entries: List[Dict[str, object]] = []
    for item in scores:
        try:
            parsed = converter.parse(str(item.source_path))
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("MUSIC export: skipping %s (%s)", item.source_path, exc)
            continue
        md = parsed.metadata or metadata.Metadata()
        md.movementNumber = str(item.index)
        md.movementName = md.movementName or f"{item.index:02d}_{item.tp_type}"
        parsed.metadata = md
        opus.append(parsed)
        manifest_entries.append(
            {
                "index": item.index,
                "type": item.tp_type,
                "path": str(item.source_path),
                "finalized": item.finalized,
            }
        )
    return opus, manifest_entries


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

    opus, tp_manifest = _build_opus(tp_scores)
    if not tp_manifest:
        return {"written": False, "reason": "no_valid_scores"}

    chapter_dir.mkdir(parents=True, exist_ok=True)
    target_score = chapter_dir / f"score_v{version}.musicxml"
    written_path = Path(opus.write("musicxml", fp=str(target_score)))
    score_path = written_path if written_path.exists() else target_score
    if score_path != target_score:
        try:
            if score_path.exists():
                score_path.replace(target_score)
            score_path = target_score
        except Exception:
            score_path = score_path if score_path.exists() else target_score

    score_dir = chapter_dir / "score"
    score_dir.mkdir(parents=True, exist_ok=True)
    final_musicxml = score_dir / "final.musicxml"
    shutil.copy2(score_path, final_musicxml)

    per_voice_midis: List[Path] = []
    voice_manifest: List[Dict[str, object]] = []

    try:
        aggregate_stream = converter.parse(str(score_path))
        final_midi = score_dir / "final.mid"
        aggregate_stream.write("midi", fp=str(final_midi))
    except Exception as exc:  # pragma: no cover - defensive logging
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
                    parsed = converter.parse(str(summary.score_path))
                    parsed.write("midi", fp=str(midi_path))
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
            "score_musicxml": _relative_to(score_path, chapter_dir),
            "final_musicxml": _relative_to(final_musicxml, chapter_dir),
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
        (score_path, score_path.name),
        (final_musicxml, f"score/{final_musicxml.name}"),
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
        "score_path": score_path,
        "final_musicxml": final_musicxml,
        "final_midi": final_midi,
        "per_voice_midis": per_voice_midis,
        "manifest_path": manifest_path,
        "bundle_path": bundle_path,
        "touch_points": tp_manifest,
    }


__all__ = ["finalize_music_exports"]
