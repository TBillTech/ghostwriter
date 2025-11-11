"""Utilities for normalizing imported musical assets.

Slice 1 of the music feature focuses on converting raw MIDI / MusicXML files
into a normalized MusicXML representation that GhostWriter (and users) can
reason about. The importer exposes helpers that process a per-voice import
folder, generate `import.musicxml` when possible, and ensure the downstream
sanitizer can build a stable score file.

The functions in this module intentionally avoid coupling to the broader
pipeline orchestration so that future slices can plug them into narration,
dialog, and mixed pipelines incrementally.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional
import logging
import warnings

try:  # pragma: no cover - guard import for environments lacking music libs
    from music21 import converter, instrument, key, meter, tempo
except ImportError as exc:  # pragma: no cover - surfaced to caller
    raise ImportError(
        "music21 is required for the music importer. Install optional music"
        " dependencies via `pip install -r requirements.txt`."
    ) from exc

SUPPORTED_IMPORT_EXTENSIONS = {".mid", ".midi", ".mid2"}
_IMPORT_FILENAME = "import.musicxml"

logger = logging.getLogger(__name__)

# Silence verbose music21 warnings that would otherwise pollute CLI output.
warnings.filterwarnings("ignore", module="music21")


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
    1. If ``import.musicxml`` is missing and raw MIDI files exist, convert the
       first available raw file into normalized MusicXML, inserting helpful
       metadata as XML comments.
    2. Leave creation of ``score.musicxml`` and ``monitor.mid`` to the
       sanitizer (invoked in a later stage). The importer simply guarantees the
       presence of ``import.musicxml`` when raw assets are available.

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
        stream = converter.parse(str(import_path))
        metadata = _extract_metadata(stream)
        _maybe_sanitize(import_dir)
        return ImportArtifacts(import_path=import_path, metadata=metadata, source_files=[])

    raw_files = _find_importable_files(import_dir)
    if not raw_files:
        logger.debug("No importable audio files found in %s", import_dir)
        return None

    stream = _load_stream(raw_files[0])
    if hasattr(stream, "flatten"):
        try:
            stream = stream.flatten()  # type: ignore[assignment]
        except Exception:
            pass
    elif hasattr(stream, "flat"):
        try:
            stream = stream.flat  # type: ignore[attr-defined,assignment]
        except Exception:
            pass
    metadata = _extract_metadata(stream)
    _write_musicxml(stream, import_path, metadata)
    _maybe_sanitize(import_dir)
    return ImportArtifacts(import_path=import_path, metadata=metadata, source_files=raw_files)


def generate_import_musicxml(import_dir: Path) -> Optional[Path]:
    """Legacy-friendly alias returning the generated ``import.musicxml`` path."""

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


def _load_stream(source: Path):
    logger.info("Loading music asset: %s", source)
    return converter.parse(str(source))


def _extract_metadata(stream) -> ImportMetadata:
    tempos = {
        f"{mark.number:.2f}" if getattr(mark, "number", None) else str(mark)
        for _offset, _end, mark in stream.metronomeMarkBoundaries()
    }
    time_signatures = {
        ts.ratioString
        for ts in stream.recurse().getElementsByClass("TimeSignature")
    }
    key_sigs = set()
    detected_key = stream.analyze("key")
    if detected_key is not None:
        key_sigs.add(str(detected_key))
    key_sigs.update(str(k) for k in stream.recurse().getElementsByClass(key.Key))
    key_sigs.update(str(k) for k in stream.recurse().getElementsByClass(key.KeySignature))

    instruments_found = {
        _instrument_name(instr)
        for instr in stream.recurse().getElementsByClass(instrument.Instrument)
    }
    if not instruments_found and hasattr(stream, "parts"):
        for part in stream.parts:
            maybe_instr = part.getInstrument(returnDefault=True)
            instruments_found.add(_instrument_name(maybe_instr))

    return ImportMetadata(
        tempos=sorted(filter(None, tempos)) or ["Unknown"],
        time_signatures=sorted(filter(None, time_signatures)) or ["Unknown"],
        key_signatures=sorted(filter(None, key_sigs)) or ["Unknown"],
        instruments=sorted(filter(None, instruments_found)) or ["Unknown"],
    )


def _instrument_name(instr) -> str:
    if instr is None:
        return "Unknown"
    name = getattr(instr, "instrumentName", None) or getattr(instr, "bestName", None)
    if name:
        return str(name)
    if getattr(instr, "midiProgram", None) is not None:
        return f"Program {instr.midiProgram}"
    return instr.__class__.__name__


def _metadata_comment(metadata: ImportMetadata) -> str:
    parts = [
        f"Tempo(s): {', '.join(metadata.tempos)}",
        f"Time Signature(s): {', '.join(metadata.time_signatures)}",
        f"Key Signature(s): {', '.join(metadata.key_signatures)}",
        f"Instrument(s): {', '.join(metadata.instruments)}",
    ]
    return " | ".join(parts)


def _write_musicxml(stream, destination: Path, metadata: ImportMetadata) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    xml_path = stream.write("musicxml")
    # music21 returns a filename when fp is omitted; read and pretty-print before writing.
    xml_text = Path(xml_path).read_text(encoding="utf-8")

    from xml.dom import minidom  # Local import to avoid module load during tests if not needed.

    dom = minidom.parseString(xml_text)
    comment_text = _metadata_comment(metadata)
    comment_node = dom.createComment(comment_text)
    dom.insertBefore(comment_node, dom.documentElement)
    pretty_xml = dom.toprettyxml(indent="  ")
    destination.write_text(pretty_xml, encoding="utf-8")

    # Clean up temporary file generated by music21 when no explicit fp is provided.
    Path(xml_path).unlink(missing_ok=True)


def _maybe_sanitize(import_dir: Path) -> None:
    try:
        from .sanitizer import ensure_sanitized
    except ImportError:  # pragma: no cover - sanitizer depends on music21 too
        return

    ensure_sanitized(import_dir)