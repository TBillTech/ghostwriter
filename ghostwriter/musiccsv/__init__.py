"""MusicCSV format utilities.

This module implements the core file I/O helpers and the
``MusicCSV`` object used to load, manipulate, and serialize
music material according to ``MusicCSVSpecification.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import csv
import io
import json
import zipfile
import math
import re

__all__ = [
    "MusicCSV",
    "read_musiccsv",
    "write_musiccsv",
    "musiccsv_to_text",
    "musiccsv_from_text",
    "from_midi",
    "to_midi",
    "validate_musiccsv",
    "resolve_derived_fields",
    "MusicCSVValidationError",
    "METADATA_JSON_SCHEMA",
]


_METADATA_FILENAME = "metadata.json"
_TRACKS_FILENAME = "tracks.csv"
_MEASURES_FILENAME = "measures.csv"
_NOTES_FILENAME = "notes.csv"
_MUSICCSV_SUFFIX = ".musiccsv"
_MUSICCSV_EMBED_PREFIX = "musiccsv:data:"
_DEFAULT_VERSION = "0.1"
_DEFAULT_KEY_SIGNATURE = "C"
_DEFAULT_TEMPO_BPM = 120.0

_NOTE_BASE_OFFSETS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_NOTE_NAMES_SHARP = [
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "A#",
    "B",
]
_PITCH_RE = re.compile(r"^\s*([A-Ga-g])([#b]?)(-?\d+)\s*$")
_TIME_SIGNATURE_RE = re.compile(r"^\s*(\d+)\s*/\s*(\d+)\s*$")
_KEY_SIGNATURE_NORMALIZE_RE = re.compile(
    r"^\s*([A-Ga-g])(?:\s*([#b♯♭]|sharp|flat))?(?:\s*([Mm](?:aj(?:or)?|in(?:or)?)?))?\s*$"
)
_VALID_TIES = {"start", "stop", "continue", "none"}
_VALID_REPEAT_VALUES = {"start", "end", "segno", "coda", "fine", "dal_capo", "dal_segno", "to_coda", "alternate"}
_TUPLET_RE = re.compile(r"^\s*(\d+)\s*:\s*(\d+)\s*$")

METADATA_JSON_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://ghostwriter.ai/musiccsv/metadata.schema.json",
    "title": "MusicCSV metadata",
    "type": "object",
    "properties": {
        "title": {"type": "string", "minLength": 1},
        "composer": {"type": "string"},
        "tempo": {"type": "number", "minimum": 0},
        "time_signature": {"type": "string", "pattern": r"^\d+/\d+$"},
        "key_signature": {"type": "string", "minLength": 1},
        "divisions_per_quarter": {"type": "integer", "minimum": 1},
        "version": {"type": "string", "minLength": 1},
        "description": {"type": "string"},
        "encoding_software": {"type": "string"},
        "created": {"type": "string", "format": "date-time"},
        "copyright": {"type": "string"},
    },
    "required": [
        "title",
        "tempo",
        "time_signature",
        "key_signature",
        "divisions_per_quarter",
        "version",
    ],
    "additionalProperties": True,
}


def _parse_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text == "":
        return None
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse boolean from {value!r}")


def _format_bool(value: Optional[bool]) -> str:
    if value is None:
        return ""
    return "true" if value else "false"


def _normalize_key_signature(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return ""

    normalized = text.replace("♯", "#").replace("♭", "b")
    match = _KEY_SIGNATURE_NORMALIZE_RE.match(normalized)
    if not match:
        return text

    note = match.group(1).upper()
    accidental_token = (match.group(2) or "").lower()
    mode_token = (match.group(3) or "").lower()

    accidental_map = {
        "": "",
        "#": "#",
        "sharp": "#",
        "♯": "#",
        "b": "b",
        "flat": "b",
        "♭": "b",
    }
    accidental = accidental_map.get(accidental_token, "")

    if mode_token in {"", "m", "maj", "major"}:
        suffix = "" if mode_token in {"", "maj", "major"} else "m"
    elif mode_token in {"min", "minor"}:
        suffix = "m"
    else:
        suffix = ""

    if mode_token in {"m", "min", "minor"}:
        suffix = "m"

    return f"{note}{accidental}{suffix}"


@dataclass
class MusicCSV:
    """In-memory representation of a MusicCSV score."""

    metadata: Dict[str, Any]
    tracks: List[Dict[str, Any]]
    measures: List[Dict[str, Any]]
    notes: List[Dict[str, Any]]

    @classmethod
    def read(cls, path: str | Path) -> "MusicCSV":
        return read_musiccsv(path)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MusicCSV":
        return cls(
            metadata=json.loads(json.dumps(data.get("metadata", {}))),
            tracks=[dict(row) for row in data.get("tracks", [])],
            measures=[dict(row) for row in data.get("measures", [])],
            notes=[dict(row) for row in data.get("notes", [])],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": json.loads(json.dumps(self.metadata)),
            "tracks": [dict(row) for row in self.tracks],
            "measures": [dict(row) for row in self.measures],
            "notes": [dict(row) for row in self.notes],
        }

    def write(self, path: str | Path) -> None:
        write_musiccsv(path, self)

    def summary(self, *, max_tracks: int = 5, max_measures: int = 5, max_notes: int = 10) -> str:
        return _summarize_musiccsv(self, max_tracks=max_tracks, max_measures=max_measures, max_notes=max_notes)

    def to_text(self) -> str:
        return musiccsv_to_text(self)

    @classmethod
    def from_text(cls, text: str) -> "MusicCSV":
        return musiccsv_from_text(text)

    @classmethod
    def from_midi(cls, path: str | Path) -> "MusicCSV":
        return from_midi(path)

    def to_midi(self, path: str | Path) -> None:
        to_midi(self, path)


class MusicCSVValidationError(ValueError):
    """Raised when ``validate_musiccsv`` encounters schema violations."""

    def __init__(self, errors: Sequence[str]):
        self.errors = list(errors)
        message = "MusicCSV validation failed:\n" + "\n".join(f"- {err}" for err in self.errors)
        super().__init__(message)


_TRACK_FIELD_TYPES: Sequence[Tuple[str, Any, bool]] = (
    ("track", int, True),
    ("label", str, False),
    ("part", str, False),
    ("instrument", str, False),
    ("channel", int, False),
    ("program", int, False),
    ("volume", int, False),
)

_MEASURE_FIELD_TYPES: Sequence[Tuple[str, Any, bool]] = (
    ("measure", int, True),
    ("time_signature", str, False),
    ("key_signature", str, False),
    ("tempo", float, False),
    ("start_beat", float, False),
    ("pickup", bool, False),
)

_NOTE_FIELD_TYPES: Sequence[Tuple[str, Any, bool]] = (
    ("track", int, True),
    ("measure", int, True),
    ("beat", float, True),
    ("pitch", str, True),
    ("duration", float, True),
    ("velocity", int, False),
    ("tie", str, False),
    ("articulation", str, False),
    ("pedal", bool, False),
    ("lyric", str, False),
    ("ornament", str, False),
    ("comment", str, False),
    ("grace", bool, False),
    ("repeat", str, False),
    ("tuplet", str, False),
)


def _convert_value(value: str, field_type: Any, required: bool) -> Any:
    if field_type is bool:
        parsed = _parse_bool(value)
        if parsed is None and required:
            raise ValueError("Boolean field is required but missing")
        return parsed
    if field_type is int:
        if value is None or value == "":
            if required:
                raise ValueError("Integer field is required but missing")
            return None
        return int(value)
    if field_type is float:
        if value is None or value == "":
            if required:
                raise ValueError("Float field is required but missing")
            return None
        return float(value)
    if value is None:
        return "" if required else None
    return value


def _stringify_value(value: Any, field_type: Any) -> str:
    if value is None:
        return ""
    if field_type is bool:
        return _format_bool(value)
    return str(value)


class _Loader:
    def __init__(self, base: Path):
        self.base = base

    def read_bytes(self, name: str) -> bytes:
        raise NotImplementedError

    def exists(self, name: str) -> bool:
        raise NotImplementedError

    def close(self) -> None:
        return None


class _DirLoader(_Loader):
    def read_bytes(self, name: str) -> bytes:
        path = self.base / name
        return path.read_bytes()

    def exists(self, name: str) -> bool:
        return (self.base / name).exists()


class _ZipLoader(_Loader):
    def __init__(self, archive: Path):
        super().__init__(archive)
        self._zip = zipfile.ZipFile(archive, "r")

    def read_bytes(self, name: str) -> bytes:
        with self._zip.open(name) as fh:
            return fh.read()

    def exists(self, name: str) -> bool:
        return name in self._zip.namelist()

    def close(self) -> None:
        self._zip.close()


def _sanitize_midi_text(value: Any) -> str:
    text = "" if value is None else str(value)
    try:
        text.encode("latin-1")
        return text
    except UnicodeEncodeError:
        return text.encode("latin-1", errors="replace").decode("latin-1")


def read_musiccsv(path: str | Path) -> MusicCSV:
    path = Path(path)
    if path.is_dir():
        loader: _Loader = _DirLoader(path)
    elif path.suffix == _MUSICCSV_SUFFIX and path.exists():
        if zipfile.is_zipfile(path):
            loader = _ZipLoader(path)
        else:
            text = path.read_text(encoding="utf-8")
            return musiccsv_from_text(text)
    else:
        raise FileNotFoundError(f"Unsupported MusicCSV path: {path}")

    for required in (_METADATA_FILENAME, _TRACKS_FILENAME, _MEASURES_FILENAME, _NOTES_FILENAME):
        if not loader.exists(required):
            raise FileNotFoundError(f"Missing required MusicCSV component: {required}")

    metadata = json.loads(loader.read_bytes(_METADATA_FILENAME).decode("utf-8"))
    tracks = _load_csv(loader.read_bytes(_TRACKS_FILENAME), _TRACK_FIELD_TYPES)
    measures = _load_csv(loader.read_bytes(_MEASURES_FILENAME), _MEASURE_FIELD_TYPES)
    notes = _load_csv(loader.read_bytes(_NOTES_FILENAME), _NOTE_FIELD_TYPES)
    if isinstance(loader, _ZipLoader):
        loader.close()
    return MusicCSV(metadata=metadata, tracks=tracks, measures=measures, notes=notes)


def write_musiccsv(path: str | Path, data: MusicCSV | Dict[str, Any]) -> None:
    music = data if isinstance(data, MusicCSV) else MusicCSV.from_dict(data)
    path = Path(path)

    def _write_to_fs(base: Path) -> None:
        base.mkdir(parents=True, exist_ok=True)
        (base / _METADATA_FILENAME).write_text(
            json.dumps(music.metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (base / _TRACKS_FILENAME).write_text(_dump_csv(music.tracks, _TRACK_FIELD_TYPES), encoding="utf-8")
        (base / _MEASURES_FILENAME).write_text(_dump_csv(music.measures, _MEASURE_FIELD_TYPES), encoding="utf-8")
        (base / _NOTES_FILENAME).write_text(_dump_csv(music.notes, _NOTE_FIELD_TYPES), encoding="utf-8")

    if path.suffix == _MUSICCSV_SUFFIX:
        text = musiccsv_to_text(music)
        if not text.endswith("\n"):
            text += "\n"
        path.write_text(text, encoding="utf-8")
    else:
        _write_to_fs(path)


def musiccsv_to_text(data: MusicCSV | Dict[str, Any]) -> str:
    music = data if isinstance(data, MusicCSV) else MusicCSV.from_dict(data)
    parts = [
        "### metadata.json",
        json.dumps(music.metadata, indent=2, sort_keys=True),
        "",
        "### tracks.csv",
        _dump_csv(music.tracks, _TRACK_FIELD_TYPES, newline="\n").strip(),
        "",
        "### measures.csv",
        _dump_csv(music.measures, _MEASURE_FIELD_TYPES, newline="\n").strip(),
        "",
        "### notes.csv",
        _dump_csv(music.notes, _NOTE_FIELD_TYPES, newline="\n").strip(),
        "",
    ]
    return "\n".join(parts)


def musiccsv_from_text(text: str) -> MusicCSV:
    sections = _split_sections(text)
    metadata_text = sections.get("metadata.json")
    if metadata_text is None:
        raise ValueError("metadata.json section missing")
    metadata = json.loads(metadata_text)
    tracks_text = sections.get("tracks.csv")
    measures_text = sections.get("measures.csv")
    notes_text = sections.get("notes.csv")
    if tracks_text is None or measures_text is None or notes_text is None:
        raise ValueError("MusicCSV text missing required CSV sections")
    tracks = _load_csv(tracks_text.encode("utf-8"), _TRACK_FIELD_TYPES)
    measures = _load_csv(measures_text.encode("utf-8"), _MEASURE_FIELD_TYPES)
    notes = _load_csv(notes_text.encode("utf-8"), _NOTE_FIELD_TYPES)
    return MusicCSV(metadata=metadata, tracks=tracks, measures=measures, notes=notes)


def from_midi(path: str | Path) -> MusicCSV:
    """Create a ``MusicCSV`` object from a MIDI file."""

    path = Path(path)
    try:
        import mido  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency may be optional
        raise ImportError(
            "mido is required for MusicCSV MIDI conversion. Install optional music"
            " dependencies via `pip install -r requirements.txt`."
        ) from exc

    midi = mido.MidiFile(str(path))
    embedded = _extract_embedded_musiccsv(midi)
    if embedded is not None:
        music = MusicCSV.from_dict(embedded)
        metadata = dict(music.metadata)
        if not metadata.get("title"):
            metadata["title"] = path.stem
        if not metadata.get("divisions_per_quarter"):
            metadata["divisions_per_quarter"] = midi.ticks_per_beat or 480
        if not metadata.get("version"):
            metadata["version"] = _DEFAULT_VERSION
        music.metadata = metadata
        return music

    return _from_midi_events(path, midi, mido)


def to_midi(data: MusicCSV | Dict[str, Any], path: str | Path) -> None:
    """Write ``MusicCSV`` content to a MIDI file."""

    music = data if isinstance(data, MusicCSV) else MusicCSV.from_dict(data)
    try:
        import mido  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency may be optional
        raise ImportError(
            "mido is required for MusicCSV MIDI conversion. Install optional music"
            " dependencies via `pip install -r requirements.txt`."
        ) from exc

    path = Path(path)
    ticks_per_beat = int(music.metadata.get("divisions_per_quarter") or 480)
    midi = mido.MidiFile(ticks_per_beat=ticks_per_beat)

    meta_track = mido.MidiTrack()
    midi.tracks.append(meta_track)
    _write_meta_track(meta_track, music, ticks_per_beat, mido)

    tracks_sorted = sorted(music.tracks, key=lambda row: row.get("track", 0))
    channel_map = _assign_track_channels(tracks_sorted)
    measure_context = _build_measure_context(music)

    for track_row in tracks_sorted:
        track_id = int(track_row.get("track", 0))
        midi_track = mido.MidiTrack()
        midi.tracks.append(midi_track)
        channel = channel_map.get(track_id, 0)
        _write_instrument_track(
            midi_track,
            track_row,
            channel,
            music.notes,
            measure_context,
            ticks_per_beat,
            mido,
        )

    midi.save(str(path))


def _extract_embedded_musiccsv(midi) -> Optional[Dict[str, Any]]:
    for track in getattr(midi, "tracks", []):
        for message in track:
            if getattr(message, "is_meta", False) and message.type == "text":
                text = getattr(message, "text", "")
                if text.startswith(_MUSICCSV_EMBED_PREFIX):
                    payload = text[len(_MUSICCSV_EMBED_PREFIX) :]
                    try:
                        return json.loads(payload)
                    except json.JSONDecodeError:
                        continue
    return None


def _from_midi_events(path: Path, midi, mido_module) -> MusicCSV:
    ticks_per_beat = midi.ticks_per_beat or 480
    tempo_us_per_beat: Optional[int] = None
    time_signature: Optional[Tuple[int, int]] = None
    key_signature: Optional[str] = None
    title = path.stem

    tracks: List[Dict[str, Any]] = []
    note_entries: List[Dict[str, Any]] = []
    next_track_id = 1

    for index, midi_track in enumerate(midi.tracks):
        current_tick = 0
        track_name: Optional[str] = None
        instrument_name: Optional[str] = None
        program: Optional[int] = None
        channel_hint: Optional[int] = None
        active_notes: Dict[Tuple[int, int], List[Tuple[int, int]]] = defaultdict(list)
        track_notes: List[Dict[str, Any]] = []

        for message in midi_track:
            current_tick += message.time
            if message.is_meta:
                if message.type == "track_name" and track_name is None:
                    track_name = message.name
                    if index == 0 and message.name:
                        title = message.name
                elif message.type == "set_tempo" and tempo_us_per_beat is None:
                    tempo_us_per_beat = message.tempo
                elif message.type == "time_signature" and time_signature is None:
                    time_signature = (message.numerator, message.denominator)
                elif message.type == "key_signature" and key_signature is None:
                    key_signature = message.key
                elif message.type == "instrument_name" and instrument_name is None:
                    instrument_name = message.name
                continue

            if message.type == "program_change":
                program = message.program
                channel_hint = message.channel
                continue

            if message.type == "note_on":
                if message.velocity > 0:
                    if channel_hint is None:
                        channel_hint = message.channel
                    active_notes[(message.channel, message.note)].append((current_tick, message.velocity))
                else:
                    stack = active_notes.get((message.channel, message.note))
                    if stack:
                        start_tick, velocity = stack.pop()
                        duration_ticks = max(1, current_tick - start_tick)
                        track_notes.append(
                            {
                                "start_tick": start_tick,
                                "duration_ticks": duration_ticks,
                                "note": message.note,
                                "velocity": velocity,
                            }
                        )
                continue

            if message.type == "note_off":
                stack = active_notes.get((message.channel, message.note))
                if stack:
                    start_tick, velocity = stack.pop()
                    duration_ticks = max(1, current_tick - start_tick)
                    track_notes.append(
                        {
                            "start_tick": start_tick,
                            "duration_ticks": duration_ticks,
                            "note": message.note,
                            "velocity": velocity,
                        }
                    )
                continue

        if not track_notes:
            continue

        track_id = next_track_id
        next_track_id += 1
        label = track_name or f"Track {track_id}"
        instrument_value = instrument_name or (f"Program {program}" if program is not None else None)
        track_dict = {
            "track": track_id,
            "label": label,
            "part": label,
            "instrument": instrument_value,
            "channel": (channel_hint + 1) if channel_hint is not None else None,
            "program": program,
            "volume": None,
        }
        tracks.append(track_dict)

        for entry in track_notes:
            entry_with_track = dict(entry)
            entry_with_track["track"] = track_id
            note_entries.append(entry_with_track)

    if tempo_us_per_beat is None:
        tempo_us_per_beat = mido_module.bpm2tempo(_DEFAULT_TEMPO_BPM)
    tempo_bpm = float(round(mido_module.tempo2bpm(tempo_us_per_beat), 6))
    if time_signature is None:
        time_signature = (4, 4)
    time_signature_str = f"{time_signature[0]}/{time_signature[1]}"
    key_signature_str = key_signature or _DEFAULT_KEY_SIGNATURE
    divisions = ticks_per_beat or 480

    beats_per_measure = _beats_per_measure_from_signature(time_signature_str)
    max_end_tick = 0
    for entry in note_entries:
        end_tick = entry["start_tick"] + entry["duration_ticks"]
        if end_tick > max_end_tick:
            max_end_tick = end_tick
    max_end_beats = max_end_tick / divisions if divisions else 0.0
    measure_count = max(1, int(math.ceil(max_end_beats / beats_per_measure))) if beats_per_measure else 1

    measures: List[Dict[str, Any]] = []
    for idx in range(measure_count):
        start_beat = round(idx * beats_per_measure, 6) if beats_per_measure else 0.0
        measures.append(
            {
                "measure": idx + 1,
                "time_signature": time_signature_str,
                "key_signature": key_signature_str,
                "tempo": tempo_bpm,
                "start_beat": start_beat,
                "pickup": False,
            }
        )

    notes: List[Dict[str, Any]] = []
    for entry in sorted(note_entries, key=lambda item: (item["track"], item["start_tick"], item["note"])):
        absolute_beats = entry["start_tick"] / divisions if divisions else 0.0
        if beats_per_measure:
            measure_index = int(math.floor(absolute_beats / beats_per_measure))
        else:
            measure_index = 0
        measure_index = max(0, min(measure_count - 1, measure_index))
        measure_number = measure_index + 1
        measure_start = measures[measure_index]["start_beat"]
        beat_in_measure = absolute_beats - measure_start + 1.0
        duration_beats = entry["duration_ticks"] / divisions if divisions else 0.0
        notes.append(
            {
                "track": entry["track"],
                "measure": measure_number,
                "beat": round(beat_in_measure, 6),
                "pitch": _midi_to_pitch(entry["note"]),
                "duration": round(duration_beats, 6),
                "velocity": entry["velocity"],
                "tie": "none",
                "articulation": None,
                "pedal": False,
                "lyric": None,
                "ornament": None,
                "comment": None,
                "grace": False,
                "repeat": None,
                "tuplet": None,
            }
        )

    metadata = {
        "title": title,
        "tempo": tempo_bpm,
        "time_signature": time_signature_str,
        "key_signature": key_signature_str,
        "divisions_per_quarter": divisions,
        "version": _DEFAULT_VERSION,
    }

    return MusicCSV(metadata=metadata, tracks=tracks, measures=measures, notes=notes)


def _write_meta_track(meta_track, music: MusicCSV, ticks_per_beat: int, mido_module) -> None:
    metadata = music.metadata or {}
    title = _sanitize_midi_text(metadata.get("title") or "MusicCSV Export")
    meta_track.append(mido_module.MetaMessage("track_name", name=title, time=0))

    payload = json.dumps(music.to_dict(), separators=(",", ":"), ensure_ascii=True)
    meta_track.append(
        mido_module.MetaMessage("text", text=f"{_MUSICCSV_EMBED_PREFIX}{payload}", time=0)
    )

    events: List[Tuple[int, Any]] = []

    last_tempo = metadata.get("tempo")
    last_time_signature = metadata.get("time_signature")
    last_key_signature = _normalize_key_signature(metadata.get("key_signature"))

    if last_tempo is not None:
        events.append(
            (
                0,
                mido_module.MetaMessage(
                    "set_tempo", tempo=mido_module.bpm2tempo(float(last_tempo)), time=0
                ),
            )
        )
    if last_time_signature:
        numerator, denominator = _parse_time_signature_components(str(last_time_signature))
        events.append(
            (
                0,
                mido_module.MetaMessage(
                    "time_signature",
                    numerator=numerator,
                    denominator=denominator,
                    clocks_per_click=24,
                    notated_32nd_notes_per_beat=8,
                    time=0,
                ),
            )
        )
    if last_key_signature:
        events.append(
            (
                0,
                mido_module.MetaMessage(
                    "key_signature", key=str(last_key_signature), time=0
                ),
            )
        )

    measures_sorted = sorted(music.measures, key=lambda row: float(row.get("start_beat", 0.0)))
    for measure in measures_sorted:
        start_beat = float(measure.get("start_beat", 0.0))
        tick = int(round(start_beat * ticks_per_beat))
        tempo = measure.get("tempo")
        if tempo is not None and tempo != last_tempo:
            events.append(
                (
                    tick,
                    mido_module.MetaMessage(
                        "set_tempo", tempo=mido_module.bpm2tempo(float(tempo)), time=0
                    ),
                )
            )
            last_tempo = tempo
        time_signature_value = measure.get("time_signature")
        if time_signature_value and time_signature_value != last_time_signature:
            numerator, denominator = _parse_time_signature_components(str(time_signature_value))
            events.append(
                (
                    tick,
                    mido_module.MetaMessage(
                        "time_signature",
                        numerator=numerator,
                        denominator=denominator,
                        clocks_per_click=24,
                        notated_32nd_notes_per_beat=8,
                        time=0,
                    ),
                )
            )
            last_time_signature = time_signature_value
        key_signature_value = _normalize_key_signature(measure.get("key_signature"))
        if key_signature_value and key_signature_value != last_key_signature:
            events.append(
                (
                    tick,
                    mido_module.MetaMessage(
                        "key_signature", key=str(key_signature_value), time=0
                    ),
                )
            )
            last_key_signature = key_signature_value

    events.sort(key=lambda item: item[0])
    current_tick = 0
    for tick, message in events:
        delta = max(0, tick - current_tick)
        current_tick = tick
        message.time = delta
        meta_track.append(message)

    meta_track.append(mido_module.MetaMessage("end_of_track", time=0))


def _assign_track_channels(tracks: Sequence[Dict[str, Any]]) -> Dict[int, int]:
    assigned = set()
    channel_map: Dict[int, int] = {}
    next_channel = 0
    for track in tracks:
        track_id = int(track.get("track", 0))
        channel_value = track.get("channel")
        channel = None
        if channel_value is None or (isinstance(channel_value, str) and not channel_value.strip()):
            channel = None
        else:
            try:
                raw = int(channel_value)
            except (TypeError, ValueError):
                raw = None
            if raw is not None:
                if raw <= 0:
                    channel = 0
                else:
                    channel = (raw - 1) % 16
        if channel is None:
            while next_channel in assigned:
                next_channel = (next_channel + 1) % 16
            channel = next_channel
            assigned.add(channel)
            next_channel = (next_channel + 1) % 16
        else:
            assigned.add(channel)
        channel_map[track_id] = channel
    return channel_map


def _build_measure_context(music: MusicCSV) -> Dict[int, Dict[str, Any]]:
    context: Dict[int, Dict[str, Any]] = {}
    default_signature = str(music.metadata.get("time_signature") or "4/4")
    measures_sorted = sorted(music.measures, key=lambda row: row.get("measure", 0))
    previous_entry: Optional[Dict[str, Any]] = None
    tolerance = 1e-6

    def _coerce_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    fallback_signature = default_signature
    for measure in measures_sorted:
        number = int(measure.get("measure", 0))
        if number <= 0:
            continue
        signature = str(measure.get("time_signature") or fallback_signature)
        beats_in_measure = _beats_per_measure_from_signature(signature)
        raw_start = _coerce_float(measure.get("start_beat"))

        if previous_entry is None:
            if raw_start is None:
                start_beat = 0.0
            elif raw_start >= 1.0 - tolerance:
                start_beat = 0.0
            else:
                start_beat = raw_start
        else:
            prev_start = float(previous_entry["start_beat"])
            prev_length = float(previous_entry.get("length_beats", _beats_per_measure_from_signature(previous_entry["time_signature"])))
            expected_start = prev_start + prev_length
            if raw_start is None or raw_start <= prev_start + tolerance or raw_start <= 1.0 + tolerance:
                start_beat = expected_start
            else:
                start_beat = raw_start

        entry = {
            "start_beat": float(start_beat),
            "time_signature": signature,
            "length_beats": beats_in_measure,
        }
        context[number] = entry
        previous_entry = entry
        fallback_signature = signature

    if not context:
        context[1] = {
            "start_beat": 0.0,
            "time_signature": default_signature,
            "length_beats": _beats_per_measure_from_signature(default_signature),
        }
    return context


def _ensure_measure_entry(
    context: Dict[int, Dict[str, Any]], measure_number: int, default_signature: str
) -> Dict[str, Any]:
    if measure_number not in context:
        prior_numbers = [num for num in context if num < measure_number]
        if prior_numbers:
            last_number = max(prior_numbers)
            last_entry = context[last_number]
            last_length = float(
                last_entry.get("length_beats", _beats_per_measure_from_signature(last_entry["time_signature"]))
            )
            delta = measure_number - last_number
            start_beat = float(last_entry["start_beat"]) + last_length * delta
        else:
            start_beat = 0.0
        context[measure_number] = {
            "start_beat": float(start_beat),
            "time_signature": default_signature,
            "length_beats": _beats_per_measure_from_signature(default_signature),
        }
    return context[measure_number]


def _write_instrument_track(
    midi_track,
    track_row: Dict[str, Any],
    channel: int,
    notes: Sequence[Dict[str, Any]],
    measure_context: Dict[int, Dict[str, Any]],
    ticks_per_beat: int,
    mido_module,
) -> None:
    track_id = int(track_row.get("track", 0))
    label = _sanitize_midi_text(track_row.get("label") or f"Track {track_id}")
    midi_track.append(mido_module.MetaMessage("track_name", name=label, time=0))

    instrument_name = track_row.get("instrument")
    if instrument_name:
        midi_track.append(
            mido_module.MetaMessage("instrument_name", name=_sanitize_midi_text(instrument_name), time=0)
        )

    program = track_row.get("program")
    if program is not None:
        midi_track.append(
            mido_module.Message("program_change", program=int(program), channel=channel % 16, time=0)
        )

    default_signature = str(measure_context.get(1, {"time_signature": "4/4"})["time_signature"])

    events: List[Tuple[int, str, int, int]] = []
    for note in notes:
        if int(note.get("track", 0)) != track_id:
            continue
        measure_number = int(note.get("measure", 1))
        measure_info = _ensure_measure_entry(measure_context, measure_number, default_signature)
        start_beat = float(measure_info["start_beat"])
        beat_value = float(note.get("beat", 1.0))
        absolute_beat = start_beat + max(0.0, beat_value - 1.0)
        duration_beats = float(note.get("duration", 1.0))
        start_tick = int(round(absolute_beat * ticks_per_beat))
        duration_ticks = max(1, int(round(duration_beats * ticks_per_beat)))
        pitch_name = note.get("pitch")
        if not pitch_name:
            continue
        if isinstance(pitch_name, str) and pitch_name.strip().lower() == "rest":
            continue
        midi_note = _pitch_to_midi(str(pitch_name))
        velocity = int(note.get("velocity") or 64)
        events.append((start_tick, "on", midi_note, velocity))
        events.append((start_tick + duration_ticks, "off", midi_note, velocity))

    events.sort(key=lambda item: (item[0], 0 if item[1] == "off" else 1, item[2]))

    current_tick = 0
    for tick, kind, midi_note, velocity in events:
        delta = max(0, tick - current_tick)
        current_tick = tick
        if kind == "on":
            midi_track.append(
                mido_module.Message(
                    "note_on", note=midi_note, velocity=velocity, channel=channel % 16, time=delta
                )
            )
        else:
            midi_track.append(
                mido_module.Message(
                    "note_off", note=midi_note, velocity=0, channel=channel % 16, time=delta
                )
            )

    midi_track.append(mido_module.MetaMessage("end_of_track", time=0))


def _parse_time_signature_components(value: str) -> Tuple[int, int]:
    match = _TIME_SIGNATURE_RE.match(value)
    if not match:
        return (4, 4)
    numerator = int(match.group(1)) if match.group(1) else 4
    denominator = int(match.group(2)) if match.group(2) else 4
    if denominator <= 0:
        denominator = 4
    return (numerator, denominator)


def _beats_per_measure_from_signature(signature: str) -> float:
    numerator, denominator = _parse_time_signature_components(signature)
    if denominator == 0:
        return float(numerator)
    return numerator * (4.0 / denominator)


def _midi_to_pitch(note: int) -> str:
    if note < 0 or note > 127:
        raise ValueError(f"Invalid MIDI note value: {note}")
    octave = note // 12 - 1
    name = _NOTE_NAMES_SHARP[note % 12]
    return f"{name}{octave}"


def _pitch_to_midi(pitch: str) -> int:
    match = _PITCH_RE.match(pitch)
    if not match:
        raise ValueError(f"Invalid pitch value: {pitch!r}")
    base = match.group(1).upper()
    accidental = match.group(2) or ""
    octave = int(match.group(3))
    semitone = _NOTE_BASE_OFFSETS.get(base)
    if semitone is None:
        raise ValueError(f"Unsupported pitch base: {pitch!r}")
    if accidental == "#":
        semitone += 1
    elif accidental == "b":
        semitone -= 1
    semitone %= 12
    midi_value = (octave + 1) * 12 + semitone
    if midi_value < 0 or midi_value > 127:
        raise ValueError(f"Pitch {pitch!r} is outside the MIDI range")
    return midi_value


def validate_musiccsv(data: MusicCSV | Dict[str, Any]) -> None:
    """Validate structural invariants for a MusicCSV payload.

    Raises ``MusicCSVValidationError`` when violations are detected."""

    music = data if isinstance(data, MusicCSV) else MusicCSV.from_dict(data)

    errors: List[str] = []

    metadata = music.metadata or {}
    if not isinstance(metadata, dict):
        errors.append("metadata must be an object")
    else:
        title = metadata.get("title")
        if not isinstance(title, str) or not title.strip():
            errors.append("metadata.title must be a non-empty string")

        tempo = metadata.get("tempo")
        if tempo is None or not isinstance(tempo, (int, float)) or tempo <= 0:
            errors.append("metadata.tempo must be a positive number")

        time_signature = metadata.get("time_signature")
        if not isinstance(time_signature, str) or not _TIME_SIGNATURE_RE.match(time_signature.strip()):
            errors.append("metadata.time_signature must be a string like '4/4'")

        key_signature = metadata.get("key_signature")
        if not isinstance(key_signature, str) or not key_signature.strip():
            errors.append("metadata.key_signature must be a non-empty string")

        divisions = metadata.get("divisions_per_quarter")
        if divisions is None or not isinstance(divisions, int) or divisions <= 0:
            errors.append("metadata.divisions_per_quarter must be a positive integer")

        version = metadata.get("version")
        if not isinstance(version, str) or not version.strip():
            errors.append("metadata.version must be a non-empty string")

    track_ids: set[int] = set()
    for index, track in enumerate(music.tracks, start=1):
        if not isinstance(track, dict):
            errors.append(f"tracks[{index}] must be an object")
            continue
        track_prefix = f"tracks[{index}]"
        track_id = track.get("track")
        if not isinstance(track_id, int) or track_id <= 0:
            errors.append(f"{track_prefix}.track must be a positive integer")
        else:
            if track_id in track_ids:
                errors.append(f"{track_prefix}.track duplicates track id {track_id}")
            track_ids.add(track_id)

        channel = track.get("channel")
        if channel is not None and (not isinstance(channel, int) or not 0 <= channel <= 15):
            errors.append(f"{track_prefix}.channel must be an integer between 0 and 15 or null")

        program = track.get("program")
        if program is not None and (not isinstance(program, int) or not 0 <= program <= 127):
            errors.append(f"{track_prefix}.program must be an integer between 0 and 127 or null")

        volume = track.get("volume")
        if volume is not None and (not isinstance(volume, int) or not 0 <= volume <= 127):
            errors.append(f"{track_prefix}.volume must be an integer between 0 and 127 or null")

        for field in ("label", "part", "instrument"):
            value = track.get(field)
            if value is not None and not isinstance(value, str):
                errors.append(f"{track_prefix}.{field} must be a string or null")

    measure_numbers: Dict[int, Dict[str, Any]] = {}
    for index, measure in enumerate(music.measures, start=1):
        if not isinstance(measure, dict):
            errors.append(f"measures[{index}] must be an object")
            continue
        measure_prefix = f"measures[{index}]"
        number = measure.get("measure")
        if not isinstance(number, int) or number <= 0:
            errors.append(f"{measure_prefix}.measure must be a positive integer")
        else:
            if number in measure_numbers:
                errors.append(f"{measure_prefix}.measure duplicates measure {number}")
            measure_numbers[number] = measure

        start_beat = measure.get("start_beat")
        if start_beat is not None and (not isinstance(start_beat, (int, float)) or start_beat < 0):
            errors.append(f"{measure_prefix}.start_beat must be a non-negative number or null")

        tempo_value = measure.get("tempo")
        if tempo_value is not None and (not isinstance(tempo_value, (int, float)) or tempo_value <= 0):
            errors.append(f"{measure_prefix}.tempo must be a positive number or null")

        time_signature_value = measure.get("time_signature")
        if time_signature_value is not None:
            if not isinstance(time_signature_value, str) or not _TIME_SIGNATURE_RE.match(time_signature_value.strip()):
                errors.append(f"{measure_prefix}.time_signature must match 'N/D' format")

        pickup_value = measure.get("pickup")
        if pickup_value is not None and not isinstance(pickup_value, bool):
            errors.append(f"{measure_prefix}.pickup must be a boolean or null")

    if not measure_numbers and music.notes:
        errors.append("notes present but no measures defined")

    valid_track_ids = track_ids
    valid_measure_numbers = set(measure_numbers.keys()) if measure_numbers else {1}
    for index, note in enumerate(music.notes, start=1):
        if not isinstance(note, dict):
            errors.append(f"notes[{index}] must be an object")
            continue
        note_prefix = f"notes[{index}]"
        track_ref = note.get("track")
        if not isinstance(track_ref, int) or track_ref not in valid_track_ids:
            errors.append(f"{note_prefix}.track references unknown track {track_ref}")

        measure_ref = note.get("measure")
        if not isinstance(measure_ref, int) or measure_ref <= 0:
            errors.append(f"{note_prefix}.measure must be a positive integer")
        elif measure_numbers and measure_ref not in valid_measure_numbers:
            errors.append(f"{note_prefix}.measure references unknown measure {measure_ref}")

        beat = note.get("beat")
        if beat is None or not isinstance(beat, (int, float)) or beat <= 0:
            errors.append(f"{note_prefix}.beat must be a positive number")

        duration = note.get("duration")
        if duration is None or not isinstance(duration, (int, float)) or duration <= 0:
            errors.append(f"{note_prefix}.duration must be a positive number")

        pitch = note.get("pitch")
        if not isinstance(pitch, str) or not pitch.strip():
            errors.append(f"{note_prefix}.pitch must be a non-empty string")
        else:
            if pitch.strip().lower() != "rest":
                try:
                    _pitch_to_midi(pitch)
                except ValueError:
                    errors.append(f"{note_prefix}.pitch value {pitch!r} is invalid")

        velocity = note.get("velocity")
        if velocity is not None and (not isinstance(velocity, int) or not 0 <= velocity <= 127):
            errors.append(f"{note_prefix}.velocity must be an integer between 0 and 127 or null")

        tie = note.get("tie")
        if tie is not None:
            if not isinstance(tie, str):
                errors.append(f"{note_prefix}.tie must be a string or null")
            else:
                tie_value = tie.strip()
                if tie_value == "":
                    note["tie"] = ""
                elif tie_value in _VALID_TIES:
                    note["tie"] = tie_value
                else:
                    errors.append(f"{note_prefix}.tie must be blank or one of {_VALID_TIES} or null")

        pedal = note.get("pedal")
        if pedal is not None and not isinstance(pedal, bool):
            errors.append(f"{note_prefix}.pedal must be a boolean or null")

        for field in ("articulation", "lyric", "ornament", "comment"):
            value = note.get(field)
            if value is not None and not isinstance(value, str):
                errors.append(f"{note_prefix}.{field} must be a string or null")

        grace = note.get("grace")
        if grace is not None and not isinstance(grace, bool):
            errors.append(f"{note_prefix}.grace must be a boolean or null")

        repeat = note.get("repeat")
        if repeat is not None:
            if not isinstance(repeat, str) or repeat.strip() == "":
                errors.append(f"{note_prefix}.repeat must be a non-empty string or null")
            elif repeat not in _VALID_REPEAT_VALUES:
                errors.append(f"{note_prefix}.repeat must be one of {_VALID_REPEAT_VALUES} or null")

        tuplet = note.get("tuplet")
        if tuplet is not None:
            if not isinstance(tuplet, str):
                errors.append(f"{note_prefix}.tuplet must be a string or null")
            else:
                match = _TUPLET_RE.match(tuplet)
                if not match:
                    errors.append(f"{note_prefix}.tuplet must be in the form 'N:M'")
                else:
                    if int(match.group(2)) == 0:
                        errors.append(f"{note_prefix}.tuplet denominator cannot be zero")

    if errors:
        raise MusicCSVValidationError(errors)


def resolve_derived_fields(data: MusicCSV | Dict[str, Any]) -> MusicCSV:
    """Return a copy of ``data`` with common derived fields populated.

    The returned ``MusicCSV`` adds ``start_tick`` and ``beats_per_measure`` to
    measures, and ``absolute_beat``, ``absolute_tick``, ``duration_ticks``, and
    ``midi_note`` to notes. The input is left untouched."""

    music = data if isinstance(data, MusicCSV) else MusicCSV.from_dict(data)
    validate_musiccsv(music)

    derived = MusicCSV.from_dict(music.to_dict())

    divisions = int((derived.metadata or {}).get("divisions_per_quarter") or 480)
    measure_context = _build_measure_context(derived)
    default_signature = str((derived.metadata or {}).get("time_signature") or "4/4")

    for measure in derived.measures:
        number = int(measure.get("measure", 0) or 0)
        if number <= 0:
            continue
        ctx = _ensure_measure_entry(measure_context, number, default_signature)
        if measure.get("start_beat") is None:
            measure["start_beat"] = ctx["start_beat"]
        measure["beats_per_measure"] = _beats_per_measure_from_signature(ctx["time_signature"])
        measure["start_tick"] = int(round(ctx["start_beat"] * divisions))

    for note in derived.notes:
        measure_number = int(note.get("measure", 1) or 1)
        ctx = _ensure_measure_entry(measure_context, measure_number, default_signature)
        beat_value = float(note.get("beat", 1.0))
        absolute_beat = ctx["start_beat"] + max(0.0, beat_value - 1.0)
        note["absolute_beat"] = round(absolute_beat, 6)
        note["absolute_tick"] = int(round(note["absolute_beat"] * divisions))
        duration_beats = float(note.get("duration", 0.0))
        note["duration_ticks"] = max(1, int(round(duration_beats * divisions)))
        pitch = note.get("pitch")
        if isinstance(pitch, str) and pitch.strip() and pitch.strip().lower() != "rest":
            try:
                note["midi_note"] = _pitch_to_midi(pitch)
            except ValueError:
                # Invalid pitches are caught during validation; keep best effort.
                pass

        tuplet = note.get("tuplet")
        if isinstance(tuplet, str):
            match = _TUPLET_RE.match(tuplet)
            if match:
                numerator = int(match.group(1))
                denominator = int(match.group(2))
                if denominator != 0:
                    note["tuplet_ratio"] = numerator / denominator

    return derived


def _split_sections(text: str) -> Dict[str, str]:
    sections: Dict[str, List[str]] = {}
    current: Optional[str] = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("### "):
            current = stripped[4:]
            sections[current] = []
        else:
            if current is None:
                continue
            sections[current].append(line)
    return {name: "\n".join(lines).strip() for name, lines in sections.items()}


def _load_csv(content: bytes, schema: Sequence[Tuple[str, Any, bool]]) -> List[Dict[str, Any]]:
    text = content.decode("utf-8")
    reader = csv.DictReader(io.StringIO(text))
    rows: List[Dict[str, Any]] = []
    expected_fields = [field for field, _type, _req in schema]
    if reader.fieldnames is None:
        raise ValueError("CSV file missing header row")
    header = reader.fieldnames
    missing_required = [field for field, _type, required in schema if required and field not in header]
    if missing_required:
        raise ValueError(f"CSV missing required columns: {missing_required}")
    for raw_row in reader:
        row: Dict[str, Any] = {}
        for field, field_type, required in schema:
            value = raw_row.get(field)
            if field_type is bool:
                row[field] = _parse_bool(value)
            elif field_type in {int, float}:
                if value is None or value == "":
                    if required:
                        raise ValueError(f"Column '{field}' is required")
                    row[field] = None
                else:
                    row[field] = field_type(value)
            else:
                if value is None or (not required and value == ""):
                    row[field] = "" if required else None
                else:
                    row[field] = value
        rows.append(row)
    return rows


def _dump_csv(rows: Iterable[Dict[str, Any]], schema: Sequence[Tuple[str, Any, bool]], *, newline: str = "\n") -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=[field for field, _type, _req in schema])
    writer.writeheader()
    for row in rows:
        out_row: Dict[str, str] = {}
        for field, field_type, _required in schema:
            value = row.get(field)
            if field_type is bool:
                out_row[field] = _format_bool(value)
            elif value is None:
                out_row[field] = ""
            else:
                out_row[field] = str(value)
        writer.writerow(out_row)
    return output.getvalue().replace("\r\n", newline)


def _summarize_musiccsv(music: MusicCSV, *, max_tracks: int, max_measures: int, max_notes: int) -> str:
    meta = music.metadata or {}
    lines = ["MusicCSV Summary:"]
    title = meta.get("title") or "<untitled>"
    composer = meta.get("composer")
    tempo = meta.get("tempo")
    lines.append(f"  Title: {title}")
    if composer:
        lines.append(f"  Composer: {composer}")
    if tempo is not None:
        lines.append(f"  Tempo: {tempo}")
    lines.append(f"  Tracks: {len(music.tracks)}")
    lines.extend(_summary_table("Tracks", music.tracks, ("track", "label", "instrument"), max_tracks))
    lines.append(f"  Measures: {len(music.measures)}")
    lines.extend(_summary_table("Measures", music.measures, ("measure", "time_signature", "tempo"), max_measures))
    lines.append(f"  Notes: {len(music.notes)}")
    lines.extend(_summary_table("Notes", music.notes, ("track", "measure", "pitch", "duration"), max_notes))
    return "\n".join(lines)


def _summary_table(title: str, rows: Sequence[Dict[str, Any]], keys: Sequence[str], limit: int) -> List[str]:
    if not rows:
        return [f"    {title}: <none>"]
    display_rows = rows[:limit]
    rendered = [f"    {title} sample:"]
    for row in display_rows:
        parts = [f"{key}={row.get(key)!r}" for key in keys if key in row]
        rendered.append("      - " + ", ".join(parts))
    if len(rows) > limit:
        rendered.append(f"      ... ({len(rows) - limit} more)")
    return rendered

