"""Sanitize user-edited MusicCSV scores and generate monitor MIDIs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
from typing import Optional

from ..musiccsv import (
    MusicCSV,
    read_musiccsv,
    validate_musiccsv,
    write_musiccsv,
)

logger = logging.getLogger(__name__)


@dataclass
class SanitizedArtifacts:
    score_path: Path
    monitor_mid_path: Path


def sanitize_import(import_path: Path, score_path: Path, monitor_mid_path: Path) -> SanitizedArtifacts:
    """Parse ``import_path`` and generate sanitized score + monitor MIDI."""

    import_path = Path(import_path)
    score_path = Path(score_path)
    monitor_mid_path = Path(monitor_mid_path)

    if not import_path.exists():
        raise FileNotFoundError(f"Expected import.musiccsv at {import_path}")

    logger.info("Sanitizing import MusicCSV: %s", import_path)
    music = read_musiccsv(import_path)
    validate_musiccsv(music)

    sanitized = _normalize_musiccsv(music)
    validate_musiccsv(sanitized)

    score_path.parent.mkdir(parents=True, exist_ok=True)
    write_musiccsv(score_path, sanitized)

    monitor_mid_path.parent.mkdir(parents=True, exist_ok=True)
    sanitized.to_midi(monitor_mid_path)

    return SanitizedArtifacts(score_path=score_path, monitor_mid_path=monitor_mid_path)


def ensure_sanitized(import_dir: Path) -> Optional[SanitizedArtifacts]:
    """Helper that mirrors TODO Slice 1 behaviour for a voice import folder."""

    import_dir = Path(import_dir)
    import_path = import_dir / "import.musiccsv"
    score_path = import_dir / "score.musiccsv"
    monitor_mid = import_dir / "monitor.mid"

    if not import_path.exists():
        logger.debug("Skipping sanitizer; missing %s", import_path)
        return None

    # Always sanitize if the score is missing; otherwise only ensure monitor exists.
    if not score_path.exists():
        return sanitize_import(import_path, score_path, monitor_mid)

    if not monitor_mid.exists():
        music = read_musiccsv(score_path)
        monitor_mid.parent.mkdir(parents=True, exist_ok=True)
        music.to_midi(monitor_mid)
        return SanitizedArtifacts(score_path=score_path, monitor_mid_path=monitor_mid)

    return SanitizedArtifacts(score_path=score_path, monitor_mid_path=monitor_mid)


def _normalize_musiccsv(music: MusicCSV) -> MusicCSV:
    metadata = dict(music.metadata or {})
    tracks = sorted(music.tracks, key=lambda row: int(row.get("track", 0) or 0))
    measures = sorted(music.measures, key=lambda row: int(row.get("measure", 0) or 0))
    notes = sorted(
        music.notes,
        key=lambda row: (
            int(row.get("measure", 0) or 0),
            float(row.get("beat", 0.0) or 0.0),
            int(row.get("track", 0) or 0),
            str(row.get("pitch", "")),
        ),
    )
    return MusicCSV(
        metadata=metadata,
        tracks=[dict(row) for row in tracks],
        measures=[dict(row) for row in measures],
        notes=[dict(row) for row in notes],
    )
