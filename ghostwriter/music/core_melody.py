"""Parsing helpers and dataclasses for the Core Melody feature."""
from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path
import csv
import re
from typing import List, Sequence, Tuple

from ..context import UserActionRequired


@dataclass(frozen=True)
class EmotionWordRow:
    word: str
    emotion: str
    duration: float


@dataclass(frozen=True)
class FeelingRow:
    emotion: str
    semitones: Tuple[int, ...]


@dataclass(frozen=True)
class TransitionRow:
    emotion_a: str
    emotion_b: str
    semitone: int


@dataclass(frozen=True)
class CoreMelodyRow:
    measure: int
    beat: float
    emotion: str
    duration: float
    semitones: Tuple[int, ...]
    transition: Tuple[int, ...]


_BLOCK_HEADER_RE = re.compile(r"^\s*[A-Za-z0-9 _.-]+\.csv\s*$")
_CORE_MELODY_HEADER = ["measure", "beat", "duration", "semi_tones", "transition"]


def _normalize_row(row: dict) -> dict:
    normalized: dict[str, str] = {}
    for key, value in row.items():
        normalized_key = (key or "").strip().lower()
        if not normalized_key:
            continue
        if isinstance(value, list):
            pieces = [piece.strip() for piece in value if piece and piece.strip()]
            normalized_value = ", ".join(pieces)
        else:
            normalized_value = (value or "").strip()
        normalized[normalized_key] = normalized_value
    return normalized


def _format_number(value: float) -> str:
    if int(value) == value:
        return str(int(value))
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _format_tuple(values: Sequence[int]) -> str:
    if not values:
        return "()"
    return "(" + ", ".join(str(int(v)) for v in values) + ")"


def _extract_csv_block(text: str, block_name: str) -> str:
    lines = text.splitlines()
    header = block_name.strip().lower()
    start_idx = None
    for idx, line in enumerate(lines):
        if line.strip().lower() == header:
            start_idx = idx + 1
            break
    if start_idx is None:
        raise UserActionRequired(f"Unable to find '{block_name}' in the response.")

    buffer: List[str] = []
    header_canonical: str | None = None
    data_started = False
    for line in lines[start_idx:]:
        stripped = line.strip()
        if not stripped:
            if header_canonical and data_started:
                break
            continue
        if stripped.startswith("```"):
            break
        if _BLOCK_HEADER_RE.match(stripped):
            break
        if header_canonical and "," not in stripped:
            break
        if header_canonical is None:
            columns = [col.strip() for col in stripped.split(",")]
            if len(columns) < 2:
                raise UserActionRequired(
                    f"Header row for '{block_name}' appears malformed: '{stripped}'."
                )
            header_canonical = ",".join(columns)
            buffer.append(header_canonical)
            continue
        buffer.append(stripped)
        data_started = True
    if not buffer:
        raise UserActionRequired(f"CSV block '{block_name}' did not contain any rows.")
    if len(buffer) == 1:
        raise UserActionRequired(f"CSV block '{block_name}' did not contain any data rows.")
    return "\n".join(buffer)


def _merge_extra_field(base: str, extra: str) -> str:
    base_clean = (base or "").strip()
    extra_clean = (extra or "").strip()
    if not base_clean:
        return extra_clean
    if not extra_clean:
        return base_clean
    if base_clean.endswith(","):
        return f"{base_clean} {extra_clean}"
    if extra_clean.startswith(","):
        return f"{base_clean}{extra_clean}"
    return f"{base_clean}, {extra_clean}"


def _parse_emotion_csv_text(csv_text: str, block_name: str) -> List[EmotionWordRow]:
    reader = csv.DictReader(StringIO(csv_text), restkey="__extra__")
    required = {"word", "emotion", "duration"}
    normalized_fieldnames = [ (name or "").strip().lower() for name in (reader.fieldnames or []) ]
    if not required.issubset(normalized_fieldnames):
        missing = sorted(required - set(normalized_fieldnames))
        raise UserActionRequired(
            f"CSV block '{block_name}' is missing required columns: {missing}"
        )

    last_field = normalized_fieldnames[-1] if normalized_fieldnames else ""

    rows: List[EmotionWordRow] = []
    for idx, raw in enumerate(reader, start=1):
        normalized = _normalize_row(raw)
        extra = normalized.pop("__extra__", "")
        if extra and last_field:
            normalized[last_field] = _merge_extra_field(normalized.get(last_field, ""), extra)
        if not any(normalized.values()):
            continue
        word = normalized.get("word") or ""
        emotion = normalized.get("emotion") or ""
        duration_val = normalized.get("duration") or ""
        if not word or not emotion or not duration_val:
            raise UserActionRequired(f"Row {idx} in '{block_name}' is missing required values.")
        duration = _parse_float(duration_val, block_name, "duration", idx)
        rows.append(EmotionWordRow(word=word, emotion=emotion, duration=duration))
    if not rows:
        raise UserActionRequired(f"CSV block '{block_name}' did not contain any data rows.")
    return rows


def _parse_feelings_csv_text(csv_text: str, block_name: str) -> List[FeelingRow]:
    reader = csv.DictReader(StringIO(csv_text), restkey="__extra__")
    required = {"emotion", "semi-tones"}
    normalized_fieldnames = [ (name or "").strip().lower() for name in (reader.fieldnames or []) ]
    if not required.issubset(normalized_fieldnames):
        raise UserActionRequired(
            f"{block_name} is missing required columns 'emotion' and 'semi-tones'."
        )
    last_field = normalized_fieldnames[-1] if normalized_fieldnames else ""

    rows: List[FeelingRow] = []
    for idx, raw in enumerate(reader, start=1):
        normalized = _normalize_row(raw)
        extra = normalized.pop("__extra__", "")
        if extra and last_field:
            normalized[last_field] = _merge_extra_field(normalized.get(last_field, ""), extra)
        if not any(normalized.values()):
            continue
        emotion = normalized.get("emotion") or ""
        semi_val = normalized.get("semi-tones") or ""
        if not emotion or not semi_val:
            raise UserActionRequired(f"Row {idx} in '{block_name}' is missing required values.")
        semitones = _parse_semitone_tuple(semi_val, block_name, idx)
        rows.append(FeelingRow(emotion=emotion, semitones=semitones))
    if not rows:
        raise UserActionRequired(f"{block_name} did not contain any data rows.")
    return rows


def _parse_transitions_csv_text(csv_text: str, block_name: str) -> List[TransitionRow]:
    reader = csv.DictReader(StringIO(csv_text), restkey="__extra__")
    required = {"emotion a", "emotion b", "semi-tone"}
    normalized_fieldnames = [ (name or "").strip().lower() for name in (reader.fieldnames or []) ]
    if not required.issubset(normalized_fieldnames):
        raise UserActionRequired(
            f"{block_name} must contain 'emotion A', 'emotion B', and 'semi-tone' columns."
        )
    last_field = normalized_fieldnames[-1] if normalized_fieldnames else ""

    rows: List[TransitionRow] = []
    for idx, raw in enumerate(reader, start=1):
        normalized = _normalize_row(raw)
        extra = normalized.pop("__extra__", "")
        if extra and last_field:
            normalized[last_field] = _merge_extra_field(normalized.get(last_field, ""), extra)
        if not any(normalized.values()):
            continue
        emotion_a = normalized.get("emotion a") or ""
        emotion_b = normalized.get("emotion b") or ""
        semi_val = normalized.get("semi-tone") or ""
        if not emotion_a or not emotion_b or not semi_val:
            raise UserActionRequired(f"Row {idx} in '{block_name}' is missing required values.")
        semitone = _parse_int(semi_val, block_name, "semi-tone", idx)
        rows.append(TransitionRow(emotion_a=emotion_a, emotion_b=emotion_b, semitone=semitone))
    if not rows:
        raise UserActionRequired(f"{block_name} did not contain any data rows.")
    return rows


def _parse_float(value: str, block_name: str, field: str, row_index: int) -> float:
    try:
        return float(value)
    except Exception as exc:  # pragma: no cover - defensive
        raise UserActionRequired(
            f"Unable to parse '{field}' as a number in {block_name} (row {row_index})."
        ) from exc


def _parse_int(value: str, block_name: str, field: str, row_index: int) -> int:
    try:
        return int(float(value))
    except Exception as exc:  # pragma: no cover - defensive
        raise UserActionRequired(
            f"Unable to parse '{field}' as an integer in {block_name} (row {row_index})."
        ) from exc


def _parse_semitone_tuple(value: str, block_name: str, row_index: int) -> Tuple[int, ...]:
    stripped = value.strip()
    if stripped.startswith("(") and stripped.endswith(")"):
        stripped = stripped[1:-1]
    stripped = stripped.strip()
    if not stripped:
        return ()
    parts = [part.strip() for part in stripped.split(",") if part.strip()]
    if not parts:
        return ()
    result: List[int] = []
    for part in parts:
        result.append(_parse_int(part, block_name, "semi-tones", row_index))
    return tuple(result)


def parse_emotion_prompt_response(text: str) -> Tuple[List[EmotionWordRow], List[EmotionWordRow]]:
    """Return PRO/ANTI tuples parsed from the melody emotion prompt response."""

    pro_rows = _parse_emotion_word_block(text, "PRO.csv")
    anti_rows = _parse_emotion_word_block(text, "ANTI.csv")
    return pro_rows, anti_rows


def _parse_emotion_word_block(text: str, block_name: str) -> List[EmotionWordRow]:
    csv_text = _extract_csv_block(text, block_name)
    return _parse_emotion_csv_text(csv_text, block_name)


def parse_emotion_chord_response(text: str) -> Tuple[List[FeelingRow], List[TransitionRow]]:
    """Return (Feelings, Transitions) parsed from the chord prompt response."""

    feelings = _parse_feelings_block(text)
    transitions = _parse_transitions_block(text)
    return feelings, transitions


def load_emotion_rows(path: str | Path, *, block_name: str | None = None) -> List[EmotionWordRow]:
    name = block_name or Path(path).name or "PRO.csv"
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise UserActionRequired(f"Missing required file '{path}'.") from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise UserActionRequired(f"Unable to read '{path}': {exc}") from exc
    return _parse_emotion_csv_text(text, name)


def load_feelings_rows(path: str | Path, *, block_name: str | None = None) -> List[FeelingRow]:
    name = block_name or Path(path).name or "Emotions.csv"
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise UserActionRequired(f"Missing required file '{path}'.") from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise UserActionRequired(f"Unable to read '{path}': {exc}") from exc
    return _parse_feelings_csv_text(text, name)


def load_transitions_rows(path: str | Path, *, block_name: str | None = None) -> List[TransitionRow]:
    name = block_name or Path(path).name or "Transitions.csv"
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise UserActionRequired(f"Missing required file '{path}'.") from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise UserActionRequired(f"Unable to read '{path}': {exc}") from exc
    return _parse_transitions_csv_text(text, name)


def build_core_melody_rows(
    pro_rows: Sequence[EmotionWordRow],
    anti_rows: Sequence[EmotionWordRow],
    feelings: Sequence[FeelingRow],
    transitions: Sequence[TransitionRow],
    *,
    wrap_last_to_first: bool = False,
    beats_per_measure: float = 4.0,
) -> List[CoreMelodyRow]:
    sequence: List[EmotionWordRow] = list(pro_rows or []) + list(anti_rows or [])
    if not sequence:
        raise UserActionRequired("Unable to build CORE_MELODY.csv without emotion rows.")

    feelings_map: dict[str, Tuple[int, ...]] = {}
    for row in feelings or []:
        key = row.emotion.strip().lower()
        if not key or key in feelings_map:
            continue
        feelings_map[key] = tuple(row.semitones)

    transitions_map: dict[Tuple[str, str], int] = {}
    for row in transitions or []:
        a = row.emotion_a.strip().lower()
        b = row.emotion_b.strip().lower()
        if not a or not b:
            continue
        key = (a, b)
        if key in transitions_map:
            continue
        transitions_map[key] = row.semitone

    rows: List[CoreMelodyRow] = []
    total = len(sequence)
    if beats_per_measure <= 0:
        beats_per_measure = 4.0
    beat_cursor = 0.0
    for idx, current in enumerate(sequence):
        emotion_key = current.emotion.strip().lower()
        feeling = feelings_map.get(emotion_key)
        if feeling is None:
            raise UserActionRequired(
                f"CORE MELODY requires a Feelings entry for emotion '{current.emotion}'."
            )

        next_row: EmotionWordRow | None = None
        if idx + 1 < total:
            next_row = sequence[idx + 1]
        elif wrap_last_to_first and total > 1:
            next_row = sequence[0]

        transition_tuple: Tuple[int, ...]
        if next_row is None:
            transition_tuple = (0,)
        else:
            next_key = next_row.emotion.strip().lower()
            value = transitions_map.get((emotion_key, next_key))
            if value is None:
                transition_tuple = (0,)
            else:
                transition_tuple = (value,)

        measure_num = int(beat_cursor // beats_per_measure) + 1
        beat_value = (beat_cursor % beats_per_measure) + 1
        rows.append(
            CoreMelodyRow(
                measure=measure_num,
                beat=beat_value,
                emotion=current.emotion,
                duration=current.duration,
                semitones=feeling,
                transition=transition_tuple,
            )
        )
        beat_cursor += float(current.duration)

    return rows


def core_melody_rows_to_csv(rows: Sequence[CoreMelodyRow]) -> str:
    if not rows:
        raise UserActionRequired("CORE MELODY requires at least one row before writing CSV.")
    lines = [",".join(_CORE_MELODY_HEADER)]
    transition_text = _format_tuple((0,))
    for row in rows:
        line = ",".join(
            [
                str(int(row.measure)),
                _format_number(row.beat),
                _format_number(row.duration),
                _format_tuple(row.semitones),
                transition_text,
            ]
        )
        lines.append(line)
    return "\n".join(lines) + "\n"


def write_core_melody_csv(path: str | Path, rows: Sequence[CoreMelodyRow]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = core_melody_rows_to_csv(rows)
    path.write_text(text, encoding="utf-8")


def build_core_melody_csv(
    *,
    pro_csv_path: str | Path,
    anti_csv_path: str | Path,
    feelings_csv_path: str | Path,
    transitions_csv_path: str | Path,
    output_csv_path: str | Path,
    overwrite: bool = False,
    wrap_last_to_first: bool = False,
    beats_per_measure: float = 4.0,
) -> List[CoreMelodyRow]:
    """Load all prerequisite CSVs and write CORE_MELODY.csv.

    Returns the list of rows regardless of whether the output file was rewritten.
    """

    pro_rows = load_emotion_rows(pro_csv_path, block_name="PRO.csv")
    anti_rows = load_emotion_rows(anti_csv_path, block_name="ANTI.csv")
    feelings = load_feelings_rows(feelings_csv_path, block_name="Emotions.csv")
    transitions = load_transitions_rows(
        transitions_csv_path, block_name="Transitions.csv"
    )

    rows = build_core_melody_rows(
        pro_rows,
        anti_rows,
        feelings,
        transitions,
        wrap_last_to_first=wrap_last_to_first,
        beats_per_measure=beats_per_measure,
    )

    output_path = Path(output_csv_path)
    if not output_path.exists() or overwrite:
        write_core_melody_csv(output_path, rows)
    return rows


def _parse_feelings_block(text: str) -> List[FeelingRow]:
    csv_text = _extract_csv_block(text, "Emotions.csv")
    return _parse_feelings_csv_text(csv_text, "Emotions.csv")


def _parse_transitions_block(text: str) -> List[TransitionRow]:
    csv_text = _extract_csv_block(text, "Transitions.csv")
    return _parse_transitions_csv_text(csv_text, "Transitions.csv")


__all__ = [
    "EmotionWordRow",
    "FeelingRow",
    "TransitionRow",
    "CoreMelodyRow",
    "parse_emotion_prompt_response",
    "parse_emotion_chord_response",
    "load_emotion_rows",
    "load_feelings_rows",
    "load_transitions_rows",
    "build_core_melody_rows",
    "core_melody_rows_to_csv",
    "write_core_melody_csv",
    "build_core_melody_csv",
]
