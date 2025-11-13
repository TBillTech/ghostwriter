"""Utilities for normalizing imported musical assets.

Slice 1 of the music feature focuses on converting raw MIDI files into a
normalized MusicCSV representation that GhostWriter (and users) can reason
about. The importer exposes helpers that process a per-voice import folder,
generate ``import.musiccsv`` when possible, and ensure the downstream sanitizer
can build a stable score file.

The functions in this module intentionally avoid coupling to the broader
pipeline orchestration so that future slices can plug them into narration,
dialog, and mixed pipelines incrementally.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import logging

from ..musiccsv import MusicCSV, read_musiccsv, write_musiccsv, validate_musiccsv

SUPPORTED_IMPORT_EXTENSIONS = {".mid", ".midi", ".mid2"}
_IMPORT_FILENAME = "import.musiccsv"

logger = logging.getLogger(__name__)



@dataclass
class ImportMetadata:
    """Summary information extracted from an imported score."""

    tempos: List[str]
    time_signatures: List[str]
    key_signatures: List[str]
    instruments: List[str]


@dataclass
class ImportArtifacts:
    """Paths created during a normalization pass."""

    import_path: Path
    metadata: ImportMetadata
    source_files: List[Path]


def process_import_directory(import_dir: Path) -> Optional[ImportArtifacts]:
    """Normalize raw musical assets inside *import_dir*.

    Workflow (aligned with MUSIC_REQUIREMENTS.md):
     1. If ``import.musiccsv`` is missing and raw MIDI files exist, convert the
         first available raw file into normalized MusicCSV.
     2. Leave creation of ``score.musiccsv`` and ``monitor.mid`` to the
         sanitizer (invoked in a later stage). The importer simply guarantees the
         presence of ``import.musiccsv`` when raw assets are available.

    Returns
    -------
    ImportArtifacts | None
        Metadata about the generated file, or ``None`` if nothing was created.
    """

    import_dir = Path(import_dir)
    import_dir.mkdir(parents=True, exist_ok=True)

    import_path = import_dir / _IMPORT_FILENAME
    if import_path.exists():
        logger.debug("Import file already present: %s", import_path)
        music = read_musiccsv(import_path)
        validate_musiccsv(music)
        metadata = _extract_metadata(music)
        _maybe_sanitize(import_dir)
        return ImportArtifacts(import_path=import_path, metadata=metadata, source_files=[])

    raw_files = _find_importable_files(import_dir)
    if not raw_files:
        logger.debug("No importable audio files found in %s", import_dir)
        return None

    music = MusicCSV.from_midi(raw_files[0])
    validate_musiccsv(music)
    metadata = _extract_metadata(music)
    write_musiccsv(import_path, music)
    _maybe_sanitize(import_dir)
    return ImportArtifacts(import_path=import_path, metadata=metadata, source_files=raw_files)
def generate_import_musiccsv(import_dir: Path) -> Optional[Path]:
    """Generate ``import.musiccsv`` for *import_dir* when possible."""

    artifacts = process_import_directory(import_dir)
    return artifacts.import_path if artifacts else None

# ---------------------------------------------------------------------------
# Internal helpers


def _find_importable_files(directory: Path) -> List[Path]:
    files = [p for p in directory.iterdir() if p.is_file()]
    # Filter only supported extensions (case-insensitive), excluding monitor.mid
    results: List[Path] = []
    for candidate in sorted(files):
        ext = candidate.suffix.lower()
        if ext in SUPPORTED_IMPORT_EXTENSIONS:
            results.append(candidate)
    return results
def _extract_metadata(music: MusicCSV) -> ImportMetadata:
    def _fmt(value: Optional[float | int | str]) -> Optional[str]:
        if value is None:
            return None
        try:
            number = float(value)
            text = f"{number:.2f}"
            return text.rstrip("0").rstrip(".") if "." in text else text
        except Exception:
            return str(value)

    tempos: set[str] = set()
    meta_tempo = (music.metadata or {}).get("tempo")
    if meta_tempo is not None:
        formatted = _fmt(meta_tempo)
        if formatted:
            tempos.add(formatted)
    for measure in music.measures:
        tempo_value = measure.get("tempo")
        formatted = _fmt(tempo_value)
        if formatted:
            tempos.add(formatted)

    time_signatures: set[str] = set()
    meta_sig = (music.metadata or {}).get("time_signature")
    if isinstance(meta_sig, str) and meta_sig.strip():
        time_signatures.add(meta_sig.strip())
    for measure in music.measures:
        value = measure.get("time_signature")
        if isinstance(value, str) and value.strip():
            time_signatures.add(value.strip())

    key_signatures: set[str] = set()
    meta_key = (music.metadata or {}).get("key_signature")
    if isinstance(meta_key, str) and meta_key.strip():
        key_signatures.add(meta_key.strip())
    for measure in music.measures:
        value = measure.get("key_signature")
        if isinstance(value, str) and value.strip():
            key_signatures.add(value.strip())

    instruments_found: set[str] = set()
    for track in music.tracks:
        instruments_found.add(_instrument_name(track))

    return ImportMetadata(
        tempos=sorted(tempos) or ["Unknown"],
        time_signatures=sorted(time_signatures) or ["Unknown"],
        key_signatures=sorted(key_signatures) or ["Unknown"],
        instruments=sorted(instruments_found) or ["Unknown"],
    )


def _instrument_name(track: dict) -> str:
    instrument_name = track.get("instrument")
    if isinstance(instrument_name, str) and instrument_name.strip():
        return instrument_name.strip()
    label = track.get("label") or track.get("part")
    if isinstance(label, str) and label.strip():
        return label.strip()
    track_id = track.get("track")
    if track_id:
        return f"Track {track_id}"
    return "Unknown"


def _maybe_sanitize(import_dir: Path) -> None:
    try:
        from .sanitizer import ensure_sanitized
    except ImportError:  # pragma: no cover - defensive
        return

    ensure_sanitized(import_dir)