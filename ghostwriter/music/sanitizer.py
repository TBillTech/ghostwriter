"""Sanitize user-edited MusicXML scores and generate monitor MIDIs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
from typing import Optional
import warnings

try:  # pragma: no cover
    from music21 import converter
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "music21 is required for the music sanitizer. Install optional music"
        " dependencies via `pip install -r requirements.txt`."
    ) from exc

warnings.filterwarnings("ignore", module="music21")

logger = logging.getLogger(__name__)


@dataclass
class SanitizedArtifacts:
    score_path: Path
    monitor_mid_path: Path


def sanitize_import(import_path: Path, score_path: Path, monitor_mid_path: Path) -> SanitizedArtifacts:
    """Parse ``import_path`` and generate sanitized score + monitor MIDI.

    The sanitizer is intentionally conservative: it relies on ``music21`` to
    reparse the human-edited ``import.musicxml`` and emits a fresh
    ``score.musicxml`` with normalized measure numbering. A matching
    ``monitor.mid`` is produced so authors can audition the result quickly.
    """

    import_path = Path(import_path)
    score_path = Path(score_path)
    monitor_mid_path = Path(monitor_mid_path)

    if not import_path.exists():
        raise FileNotFoundError(f"Expected import.musicxml at {import_path}")

    logger.info("Sanitizing import MusicXML: %s", import_path)
    stream = converter.parse(str(import_path))
    stream.makeNotation(inPlace=True)

    score_path.parent.mkdir(parents=True, exist_ok=True)
    stream.write("musicxml", fp=str(score_path))
    _pretty_format_musicxml(score_path)

    monitor_mid_path.parent.mkdir(parents=True, exist_ok=True)
    stream.write("midi", fp=str(monitor_mid_path))

    return SanitizedArtifacts(score_path=score_path, monitor_mid_path=monitor_mid_path)


def ensure_sanitized(import_dir: Path) -> Optional[SanitizedArtifacts]:
    """Helper that mirrors TODO Slice 1 behaviour for a voice import folder."""

    import_dir = Path(import_dir)
    import_path = import_dir / "import.musicxml"
    score_path = import_dir / "score.musicxml"
    monitor_mid = import_dir / "monitor.mid"

    if not import_path.exists():
        logger.debug("Skipping sanitizer; missing %s", import_path)
        return None

    # Always sanitize if the score is missing; otherwise only ensure monitor exists.
    if not score_path.exists():
        return sanitize_import(import_path, score_path, monitor_mid)

    if not monitor_mid.exists():
        stream = converter.parse(str(score_path))
        stream.write("midi", fp=str(monitor_mid))
        return SanitizedArtifacts(score_path=score_path, monitor_mid_path=monitor_mid)

    return SanitizedArtifacts(score_path=score_path, monitor_mid_path=monitor_mid)


def _pretty_format_musicxml(path: Path) -> None:
    """Reformat MusicXML for readability by adding indentation."""

    text = Path(path).read_text(encoding="utf-8")

    from xml.dom import minidom

    dom = minidom.parseString(text)
    pretty = dom.toprettyxml(indent="  ")
    Path(path).write_text(pretty, encoding="utf-8")
