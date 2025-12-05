"""Music-specific pipeline helpers for touch-point gates."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Iterable
from io import StringIO
import csv
import hashlib
import json
import math
import shutil
import logging
import re

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

from .context import (
    VoiceContext,
    reduced_notes_csv,
    block_measure_range,
    describe_measure_window,
    slice_csv_by_measure,
)
from .prompts import (
    build_first_score_prompt,
    build_music_check_prompt,
    build_subtle_edit_prompt,
    build_metadata_tracks_prompt,
    build_melody_emotion_prompt,
    build_emotion_chord_prompt,
)
from .core_melody import (
    EmotionWordRow,
    FeelingRow,
    TransitionRow,
    parse_emotion_prompt_response,
    parse_emotion_chord_response,
    load_emotion_rows,
    build_core_melody_csv,
)

logger = logging.getLogger(__name__)

_MAX_MUSICCSV_SNIPPET = 4000
_NOTE_HEADER = [
    "measure",
    "beat",
    "pitch",
    "duration",
    "velocity",
    "tie",
    "articulation",
]
_BLOCK_SIZE = 10
_SCIENTIFIC_PITCH_RE = re.compile(r"^[A-Ga-g](?:[#b])?\d+$")
_ALLOWED_TIES = {"start", "continue", "stop"}

_STATE_FILE_NAME = "core_melody_state.json"
_EMOTION_HEADER = ["word", "emotion", "duration"]
_FEELINGS_HEADER = ["emotion", "semi-tones"]
_TRANSITIONS_HEADER = ["emotion A", "emotion B", "semi-tone"]


def _core_melody_state_path(tp_dir: Path) -> Path:
    return Path(tp_dir) / _STATE_FILE_NAME


def _load_core_melody_state(tp_dir: Path) -> Dict[str, Any]:
    path = _core_melody_state_path(tp_dir)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_core_melody_state(tp_dir: Path, state: Dict[str, Any]) -> None:
    path = _core_melody_state_path(tp_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    except Exception:
        pass


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hash_file(path: Path) -> str:
    try:
        return _hash_text(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return ""


def _write_emotion_rows_csv(path: Path, rows: Iterable[EmotionWordRow]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(_EMOTION_HEADER)
        for row in rows:
            writer.writerow([row.word, row.emotion, _format_number(row.duration)])


def _write_feelings_rows_csv(path: Path, rows: Iterable[FeelingRow]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(_FEELINGS_HEADER)
        for row in rows:
            writer.writerow([row.emotion, "(" + ", ".join(str(int(val)) for val in row.semitones) + ")"])


def _write_transitions_rows_csv(path: Path, rows: Iterable[TransitionRow]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(_TRANSITIONS_HEADER)
        for row in rows:
            writer.writerow([row.emotion_a, row.emotion_b, str(row.semitone)])


def _coerce_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        if "." in text:
            return int(float(text))
        return int(text)
    except Exception:
        return None


def _coerce_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        return float(text)
    except Exception:
        return None


def _coerce_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"1", "true", "yes", "y", "t"}:
        return True
    if text in {"0", "false", "no", "n", "f"}:
        return False
    return None


def _format_number(value: float) -> str:
    if int(value) == value:
        return str(int(value))
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _parse_time_signature(signature: str) -> Tuple[int, int]:
    try:
        parts = signature.split("/")
        numerator = int(parts[0]) if parts and parts[0] else 4
        denominator = int(parts[1]) if len(parts) > 1 and parts[1] else 4
        if denominator <= 0:
            denominator = 4
        return numerator, denominator
    except Exception:
        return (4, 4)


def _beats_per_measure_from_signature(signature: str) -> float:
    numerator, denominator = _parse_time_signature(signature)
    if denominator == 0:
        return float(numerator)
    return numerator * (4.0 / denominator)


def _beats_per_measure_from_metadata(metadata_text: str) -> float:
    try:
        metadata = json.loads(metadata_text)
    except Exception:
        metadata = {}
    signature = str(metadata.get("time_signature") or "4/4")
    signature = signature.strip() or "4/4"
    return _beats_per_measure_from_signature(signature)


def _voice_is_melody(spec: Any) -> bool:
    token = str(getattr(spec, "token", "") or "").lower()
    idea = str(getattr(spec, "idea", "") or "").lower()
    role = str(getattr(spec, "role", "") or "").lower()
    joined = " ".join(part for part in (token, idea, role) if part)
    return any(keyword in joined for keyword in ("melody", "lead"))


def _notes_csv_path(tp_dir: Path, voice_token: str, variant: str) -> Path:
    safe_variant = (variant or "standard").strip().lower()
    return Path(tp_dir) / f"notes_{voice_token}_{safe_variant}.csv"


def _block_csv_path(tp_dir: Path, voice_token: str, variant: str, block_index: int) -> Path:
    safe_variant = (variant or "standard").strip().lower()
    return Path(tp_dir) / f"notes_{voice_token}_{safe_variant}_block{block_index:02d}.csv"


def _measure_count_for_variant(tp_dir: Path, variant: str) -> int:
    tp_dir = Path(tp_dir)
    safe_variant = (variant or "standard").strip().lower()
    measures_path = tp_dir / f"measures_{safe_variant}.csv"
    max_measure = 0
    if measures_path.exists():
        try:
            import csv

            with measures_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    value = row.get("measure")
                    if value is None:
                        continue
                    try:
                        mnum = int(float(value))
                    except Exception:
                        continue
                    max_measure = max(max_measure, mnum)
        except Exception:
            max_measure = 0
    if max_measure:
        return max_measure

    melody_path = tp_dir / f"melody_{safe_variant}.csv"
    if melody_path.exists():
        try:
            text = melody_path.read_text(encoding="utf-8")
            rows = _measures_from_melody_csv(text)
            if rows:
                return max(int(row.get("measure", 0) or 0) for row in rows)
        except Exception:
            return 0
    return 0


def _progress_path(tp_dir: Path, variant: str) -> Path:
    safe_variant = (variant or "standard").strip().lower()
    return Path(tp_dir) / f"music_progress_{safe_variant}.json"


def _load_music_progress(tp_dir: Path, variant: str) -> Dict[str, Any]:
    path = _progress_path(tp_dir, variant)
    if not path.exists():
        return {"voices": {}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"voices": {}}


def _save_music_progress(tp_dir: Path, variant: str, data: Dict[str, Any]) -> None:
    path = _progress_path(tp_dir, variant)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass


def _read_note_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        import csv

        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows: List[Dict[str, Any]] = []
            for row in reader:
                entry = {key: row.get(key, "") for key in _NOTE_HEADER}
                rows.append(entry)
            return rows
    except Exception:
        return []


def _write_note_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_NOTE_HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in _NOTE_HEADER})


def _replace_rows_in_range(
    path: Path,
    new_rows: List[Dict[str, Any]],
    *,
    measure_start: int,
    measure_end: int,
) -> None:
    existing = _read_note_rows(path)

    def _measure_value(entry: Dict[str, Any]) -> int:
        try:
            return int(float(entry.get("measure", 0) or 0))
        except Exception:
            return 0

    kept = [row for row in existing if not (measure_start <= _measure_value(row) <= measure_end)]
    merged = kept + new_rows

    def _sort_key(entry: Dict[str, Any]) -> Tuple[int, float, str]:
        try:
            measure = int(float(entry.get("measure", 0) or 0))
        except Exception:
            measure = 0
        try:
            beat = float(entry.get("beat", 0) or 0)
        except Exception:
            beat = 0.0
        pitch = str(entry.get("pitch", ""))
        return (measure, beat, pitch)

    merged.sort(key=_sort_key)
    _write_note_rows(path, merged)


def _note_dicts_from_music(music: MusicCSV) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for note in music.notes:
        try:
            row = {
                "measure": str(int(note.get("measure", 0) or 0)),
                "beat": str(note.get("beat", "")),
                "pitch": str(note.get("pitch", "")),
                "duration": str(note.get("duration", "")),
                "velocity": str(note.get("velocity", "")),
                "tie": str(note.get("tie", "")),
                "articulation": str(note.get("articulation", "")),
            }
        except Exception:
            continue
        rows.append(row)
    return rows


def _filter_rows_by_measure(
    rows: List[Dict[str, Any]],
    *,
    measure_start: int,
    measure_end: int,
) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    for row in rows:
        try:
            measure = int(float(row.get("measure", 0) or 0))
        except Exception:
            continue
        if measure_start <= measure <= measure_end:
            filtered.append(row)
    return filtered


def _slice_notes_file(path: Path, *, measure_start: int, measure_end: int) -> str:
    if not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return ""
    return slice_csv_by_measure(text, measure_start, measure_end)


def _collect_other_voice_blocks(
    *,
    tp_dir: Path,
    variant: str,
    voice_context: VoiceContext,
    exclude_token: str,
    measure_start: int,
    measure_end: int,
) -> str:
    blocks: List[str] = []
    for spec in voice_context.voices:
        token = spec.token
        if token == exclude_token:
            continue
        notes_path = _notes_csv_path(tp_dir, token, variant)
        block_text = _slice_notes_file(notes_path, measure_start=measure_start, measure_end=measure_end)
        if block_text.strip():
            blocks.append(f"# voice: {token}\n{block_text.strip()}")
    return "\n\n".join(blocks).strip()


def _order_tracks_csv(csv_text: str, voice_tokens: Iterable[str]) -> str:
    tokens = [str(tok).strip() for tok in voice_tokens if isinstance(tok, str) and tok.strip()]
    if not tokens:
        return csv_text
    text = csv_text.strip()
    if not text:
        return csv_text
    import csv
    from io import StringIO

    reader = csv.DictReader(StringIO(text))
    header = reader.fieldnames
    if not header or "voice_token" not in header:
        return csv_text
    rows = list(reader)
    if not rows:
        return csv_text

    order = {token: idx for idx, token in enumerate(tokens)}
    indexed_rows = list(enumerate(rows))

    def _sort_key(item: Tuple[int, Dict[str, Any]]) -> Tuple[int, int]:
        original_idx, row = item
        token = str(row.get("voice_token", "") or "").strip()
        if token in order:
            return (0, order[token])
        return (1, original_idx)

    sorted_rows = [row for _, row in sorted(indexed_rows, key=_sort_key)]
    for idx, row in enumerate(sorted_rows, start=1):
        row["track"] = str(idx)

    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=header)
    writer.writeheader()
    for row in sorted_rows:
        writer.writerow(row)
    return output.getvalue().strip() + "\n"


def _sanitize_block_note_rows(
    rows: List[Dict[str, Any]],
    *,
    measure_start: int,
    measure_end: int,
    beats_per_measure: float,
    preserve_input_order: bool = False,
) -> List[Dict[str, Any]]:
    if beats_per_measure <= 0:
        beats_per_measure = 4.0

    sanitized: List[Dict[str, Any]] = []
    valid_velocities: List[int] = []

    for order_idx, row in enumerate(rows):
        measure = _coerce_int(row.get("measure"))
        if measure is None:
            measure = measure_start
        measure = max(measure_start, min(measure, measure_end))

        beat_value = _coerce_float(row.get("beat"))
        if beat_value is None:
            beat_value = 1.0
        beat = max(1.0, beat_value)
        start_offset = beat - 1.0
        while start_offset >= beats_per_measure - 1e-6:
            start_offset -= beats_per_measure
            measure += 1
        beat = start_offset + 1.0

        duration = _coerce_float(row.get("duration")) or 1.0
        if duration <= 0:
            duration = 0.25

        raw_pitch = str(row.get("pitch", "") or "").strip()
        if not raw_pitch:
            pitch = "rest"
        elif raw_pitch.lower() == "rest":
            pitch = "rest"
        elif _SCIENTIFIC_PITCH_RE.match(raw_pitch):
            pitch = raw_pitch[0].upper() + raw_pitch[1:]
        else:
            pitch = "rest"

        velocity = _coerce_int(row.get("velocity"))
        if velocity is not None and 0 <= velocity <= 127:
            valid_velocities.append(velocity)
        else:
            velocity = None

        articulation = str(row.get("articulation", "") or "").strip()

        sanitized.append(
            {
                "_order": order_idx,
                "measure": measure,
                "beat": beat,
                "duration": duration,
                "pitch": pitch,
                "velocity": velocity,
                "tie": "",
                "articulation": articulation,
            }
        )

    avg_velocity = valid_velocities and sum(valid_velocities) // len(valid_velocities) or 90
    for item in sanitized:
        if item["velocity"] is None or not (0 <= int(item["velocity"]) <= 127):
            item["velocity"] = avg_velocity

    def _abs_position(entry: Dict[str, Any]) -> float:
        return (entry["measure"] - 1) * beats_per_measure + (entry["beat"] - 1)

    if preserve_input_order:
        sanitized.sort(key=lambda item: item["_order"])  # keep LLM ordering for melody lines
    else:
        sanitized.sort(key=lambda item: (_abs_position(item), item["_order"]))
    ordered_rows = [item for item in sanitized if item["duration"] > 0.0]

    split_rows: List[Dict[str, Any]] = []
    for item in ordered_rows:
        remaining = item["duration"]
        current_measure = item["measure"]
        current_beat = item["beat"]
        segments: List[Dict[str, Any]] = []
        seg_idx = 0
        while remaining > 1e-6 and current_measure <= measure_end:
            room = beats_per_measure - (current_beat - 1)
            if room <= 1e-6:
                current_measure += 1
                current_beat = 1.0
                continue
            take = min(remaining, room)
            segments.append(
                {
                    "measure": current_measure,
                    "beat": current_beat,
                    "duration": take,
                    "pitch": item["pitch"],
                    "velocity": item["velocity"],
                    "tie": item["tie"],
                    "articulation": item["articulation"],
                    "_order": item["_order"] + seg_idx * 0.001,
                }
            )
            remaining -= take
            current_measure += 1
            current_beat = 1.0
            seg_idx += 1

        if not segments:
            continue
        if len(segments) > 1:
            for seg_idx, segment in enumerate(segments):
                if seg_idx == 0:
                    segment["tie"] = "start"
                elif seg_idx == len(segments) - 1:
                    segment["tie"] = "stop"
                else:
                    segment["tie"] = "continue"
        split_rows.extend(segments)

    if not split_rows:
        split_rows = [
            {
                "measure": measure_start,
                "beat": 1.0,
                "duration": beats_per_measure,
                "pitch": "rest",
                "velocity": avg_velocity,
                "tie": "",
                "articulation": "",
                "_order": 1e6,
            }
        ]

    present_measures = {item["measure"] for item in split_rows}
    max_order = max((item.get("_order", 0.0) for item in split_rows), default=0.0)
    for offset, measure in enumerate(range(measure_start, measure_end + 1), start=1):
        if measure in present_measures:
            continue
        split_rows.append(
            {
                "measure": measure,
                "beat": 1.0,
                "duration": beats_per_measure,
                "pitch": "rest",
                "velocity": avg_velocity,
                "tie": "",
                "articulation": "",
                "_order": max_order + offset,
            }
        )

    if preserve_input_order:
        split_rows.sort(key=lambda item: item.get("_order", 0.0))
    else:
        split_rows.sort(key=lambda item: (_abs_position(item), item["pitch"]))

    formatted: List[Dict[str, Any]] = []
    for item in split_rows:
        formatted.append(
            {
                "measure": str(int(item["measure"])),
                "beat": _format_number(float(item["beat"])),
                "pitch": item["pitch"],
                "duration": _format_number(float(item["duration"])),
                "velocity": str(int(item["velocity"])),
                "tie": item["tie"],
                "articulation": item["articulation"],
            }
        )

    return formatted


def _latest_attempt_log(tp_dir: Path, prefix: str) -> Optional[Path]:
    try:
        candidates = list(Path(tp_dir).glob(f"{prefix}_attempt_*.txt"))
    except Exception:
        return None
    best: Optional[Path] = None
    best_attempt = -1
    for path in candidates:
        match = re.search(r"_attempt_(\d+)\.txt$", path.name)
        if not match:
            continue
        try:
            attempt_no = int(match.group(1))
        except Exception:
            continue
        if attempt_no >= best_attempt:
            best_attempt = attempt_no
            best = path
    return best


def _mirror_block_attempt_logs(
    *,
    tp_dir: Path,
    voice_token: str,
    variant: str,
    block_index: int,
    block_count: int,
    source_prefix: str,
) -> None:
    if block_count <= 0 or block_index not in {1, block_count}:
        return
    latest = _latest_attempt_log(tp_dir, source_prefix)
    if latest is None or not latest.exists():
        return
    safe_variant = (variant or "standard").strip().lower()
    token_safe = (voice_token or "voice").replace(".", "_")
    dest_attempt = 1 if block_index == 1 else max(block_count, 1)
    dest_path = Path(tp_dir) / f"notes_{token_safe}_{safe_variant}.first_score_attempt_{dest_attempt}.txt"
    try:
        shutil.copy2(latest, dest_path)
    except Exception:
        pass


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
        log_file=str(log_path),
    )

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

    voice_tokens = prompt_payload.get("touch_point_voices") or []
    if not voice_tokens:
        meta_voices = (prompt_payload.get("touch_point_metadata") or {}).get("voices")
        if isinstance(meta_voices, list):
            voice_tokens = meta_voices
    ordered_tracks = _order_tracks_csv(tracks_text, voice_tokens)
    save_text(tracks_path, ordered_tracks)

    try:
        _log_info(
            f"MUSIC: wrote metadata.json and tracks.csv for tp={tp_index:02d} at {tp_dir}",
            tp_dir,
        )
    except Exception:
        pass

    return True


def run_melody_emotion_step(
    *,
    tp_dir: Path,
    tp_index: int,
    tp_type: str,
    prompt_payload: Optional[Dict[str, Any]],
    force: bool = False,
) -> bool:
    """Run the melody emotion prompt and persist PRO/ANTI CSVs."""

    if not prompt_payload:
        return False

    tp_dir = Path(tp_dir)
    tp_dir.mkdir(parents=True, exist_ok=True)

    pro_path = tp_dir / "PRO.csv"
    anti_path = tp_dir / "ANTI.csv"
    log_path = tp_dir / "music_melody_emotion.txt"

    state = _load_core_melody_state(tp_dir)
    state.setdefault("version", 1)

    existing_pro_hash = _hash_file(pro_path)
    existing_anti_hash = _hash_file(anti_path)

    if not force and pro_path.exists() and anti_path.exists():
        state["pro_hash"] = existing_pro_hash
        state["anti_hash"] = existing_anti_hash
        _save_core_melody_state(tp_dir, state)
        return False

    title = str(prompt_payload.get("touch_point_title", "") or "")
    description = str(prompt_payload.get("touch_point_description", "") or "")
    prior_paragraph = str(prompt_payload.get("touch_point_prior_paragraph", "") or "")

    prompt = build_melody_emotion_prompt(
        prompt_payload=prompt_payload,
        tp_index=tp_index,
        tp_type=tp_type,
        tp_title=title,
        tp_description=description,
        tp_prior_paragraph=prior_paragraph,
    )

    model, temp, max_tokens = env_for_prompt(
        "music_melody_emotion_prompt.md",
        "MUSIC_MELODY_EMOTION",
        default_temp=0.4,
        default_max_tokens=2200,
    )

    response = llm_complete(
        prompt,
        system=(
            "Generate the requested PRO and ANTI statements, then emit the PRO.csv and ANTI.csv blocks "
            "exactly as specified."
        ),
        temperature=temp,
        max_tokens=max_tokens,
        model=model,
        log_file=str(log_path),
    )

    text = (response or "").strip()
    if not text:
        raise UserActionRequired(
            "Melody emotion step produced an empty response. Inspect music_melody_emotion.txt and retry."
        )

    pro_rows, anti_rows = parse_emotion_prompt_response(text)
    _write_emotion_rows_csv(pro_path, pro_rows)
    _write_emotion_rows_csv(anti_path, anti_rows)

    state["pro_hash"] = _hash_file(pro_path)
    state["anti_hash"] = _hash_file(anti_path)
    _save_core_melody_state(tp_dir, state)

    try:
        _log_info(
            f"MUSIC: wrote melody emotion artifacts for tp={tp_index:02d} at {tp_dir}",
            tp_dir,
        )
    except Exception:
        pass

    return True


def run_emotion_chord_step(
    *,
    tp_dir: Path,
    tp_index: int,
    tp_type: str,
    prompt_payload: Optional[Dict[str, Any]],
    force: bool = False,
) -> bool:
    """Run the emotion-chord prompt and persist Feelings/Transitions CSVs."""

    if not prompt_payload:
        return False

    tp_dir = Path(tp_dir)
    tp_dir.mkdir(parents=True, exist_ok=True)

    pro_path = tp_dir / "PRO.csv"
    anti_path = tp_dir / "ANTI.csv"
    if not pro_path.exists() or not anti_path.exists():
        raise UserActionRequired(
            "Emotion chord step requires PRO.csv and ANTI.csv. Run melody emotion step first."
        )

    pro_rows = load_emotion_rows(pro_path, block_name="PRO.csv")
    anti_rows = load_emotion_rows(anti_path, block_name="ANTI.csv")
    ordered_emotions = [row.emotion for row in (pro_rows + anti_rows) if row.emotion]
    if not ordered_emotions:
        raise UserActionRequired("Emotion chord step requires at least one emotion entry.")

    unique_emotions: List[str] = []
    seen: set[str] = set()
    for emotion in ordered_emotions:
        key = emotion.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        unique_emotions.append(emotion)

    transition_pairs: List[Tuple[str, str]] = []
    for idx in range(len(ordered_emotions) - 1):
        transition_pairs.append((ordered_emotions[idx], ordered_emotions[idx + 1]))

    sequence_hash = _hash_text("|".join(val.strip().lower() for val in ordered_emotions))

    feelings_path = tp_dir / "Feelings.csv"
    transitions_path = tp_dir / "Transitions.csv"
    log_path = tp_dir / "music_emotion_chord.txt"

    state = _load_core_melody_state(tp_dir)
    state.setdefault("version", 1)

    existing_feelings_hash = _hash_file(feelings_path)
    existing_transitions_hash = _hash_file(transitions_path)

    if not force and feelings_path.exists() and transitions_path.exists():
        state["feelings_hash"] = existing_feelings_hash
        state["transitions_hash"] = existing_transitions_hash
        state["emotion_sequence_hash"] = sequence_hash
        _save_core_melody_state(tp_dir, state)
        return False

    title = str(prompt_payload.get("touch_point_title", "") or "")
    description = str(prompt_payload.get("touch_point_description", "") or "")
    prior_paragraph = str(prompt_payload.get("touch_point_prior_paragraph", "") or "")

    prompt = build_emotion_chord_prompt(
        prompt_payload=prompt_payload,
        tp_index=tp_index,
        tp_type=tp_type,
        tp_title=title,
        tp_description=description,
        tp_prior_paragraph=prior_paragraph,
        feelings=unique_emotions,
        transitions=transition_pairs,
    )

    model, temp, max_tokens = env_for_prompt(
        "music_emotion_chord_prompt.md",
        "MUSIC_EMOTION_CHORD",
        default_temp=0.35,
        default_max_tokens=2200,
    )

    response = llm_complete(
        prompt,
        system="Produce Emotions.csv and Transitions.csv blocks using the provided feelings and adjacency pairs.",
        temperature=temp,
        max_tokens=max_tokens,
        model=model,
        log_file=str(log_path),
    )

    text = (response or "").strip()
    if not text:
        raise UserActionRequired(
            "Emotion chord step produced an empty response. Inspect music_emotion_chord.txt and retry."
        )

    feelings_rows, transition_rows = parse_emotion_chord_response(text)
    _write_feelings_rows_csv(feelings_path, feelings_rows)
    _write_transitions_rows_csv(transitions_path, transition_rows)

    state["feelings_hash"] = _hash_file(feelings_path)
    state["transitions_hash"] = _hash_file(transitions_path)
    state["emotion_sequence_hash"] = sequence_hash
    _save_core_melody_state(tp_dir, state)

    try:
        _log_info(
            f"MUSIC: wrote emotion chord artifacts for tp={tp_index:02d} at {tp_dir}",
            tp_dir,
        )
    except Exception:
        pass

    return True


def ensure_core_melody_csv(
    *,
    tp_dir: Path,
    wrap_last_to_first: bool = False,
    force: bool = False,
) -> bool:
    """Build CORE_MELODY.csv if needed, respecting user edits unless forced."""

    tp_dir = Path(tp_dir)
    tp_dir.mkdir(parents=True, exist_ok=True)

    pro_path = tp_dir / "PRO.csv"
    anti_path = tp_dir / "ANTI.csv"
    feelings_path = tp_dir / "Feelings.csv"
    transitions_path = tp_dir / "Transitions.csv"
    core_path = tp_dir / "CORE_MELODY.csv"

    for path, label in (
        (pro_path, "PRO.csv"),
        (anti_path, "ANTI.csv"),
        (feelings_path, "Feelings.csv"),
        (transitions_path, "Transitions.csv"),
    ):
        if not path.exists():
            raise UserActionRequired(
                f"CORE MELODY build requires {label}. Run earlier melody steps first."
            )

    state = _load_core_melody_state(tp_dir)
    state.setdefault("version", 1)

    pro_hash = _hash_file(pro_path)
    anti_hash = _hash_file(anti_path)
    feelings_hash = _hash_file(feelings_path)
    transitions_hash = _hash_file(transitions_path)
    source_hash = _hash_text("|".join([pro_hash, anti_hash, feelings_hash, transitions_hash]))

    core_exists = core_path.exists()
    current_core_hash = _hash_file(core_path) if core_exists else ""
    stored_core_hash = state.get("core_melody_hash", "")
    stored_source_hash = state.get("core_source_hash", "")
    user_modified = bool(core_exists and stored_core_hash and current_core_hash and stored_core_hash != current_core_hash)

    needs_build = False
    if force or not core_exists:
        needs_build = True
    elif not user_modified and stored_source_hash and stored_source_hash != source_hash:
        needs_build = True
    elif not user_modified and not stored_source_hash:
        needs_build = True

    if needs_build:
        if user_modified and not force:
            raise UserActionRequired(
                "CORE_MELODY.csv was edited manually. Rerun with --force to overwrite or delete the file."
            )
        build_core_melody_csv(
            pro_csv_path=pro_path,
            anti_csv_path=anti_path,
            feelings_csv_path=feelings_path,
            transitions_csv_path=transitions_path,
            output_csv_path=core_path,
            overwrite=True,
            wrap_last_to_first=wrap_last_to_first,
        )
        current_core_hash = _hash_file(core_path)
        state["core_source_hash"] = source_hash
        state["core_melody_hash"] = current_core_hash
        state["pro_hash"] = pro_hash
        state["anti_hash"] = anti_hash
        state["feelings_hash"] = feelings_hash
        state["transitions_hash"] = transitions_hash
        _save_core_melody_state(tp_dir, state)
        try:
            _log_info(
                f"MUSIC: built CORE_MELODY.csv at {core_path}",
                tp_dir,
            )
        except Exception:
            pass
        return True

    if not user_modified:
        state["core_source_hash"] = source_hash
        if current_core_hash:
            state["core_melody_hash"] = current_core_hash
    state["pro_hash"] = pro_hash
    state["anti_hash"] = anti_hash
    state["feelings_hash"] = feelings_hash
    state["transitions_hash"] = transitions_hash
    _save_core_melody_state(tp_dir, state)
    return False


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
    beats_per_measure = _beats_per_measure_from_metadata(metadata_text)

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

    core_melody_path = tp_dir / "CORE_MELODY.csv"
    if not core_melody_path.exists():
        raise UserActionRequired(
            "Melody edges step requires CORE_MELODY.csv. Run the core melody builder first."
        )
    try:
        core_melody_text = core_melody_path.read_text(encoding="utf-8").strip()
    except Exception as exc:
        raise UserActionRequired(
            f"Unable to read CORE_MELODY.csv: {exc}"
        ) from exc
    if not core_melody_text:
        raise UserActionRequired(
            "CORE_MELODY.csv is empty. Rebuild the core melody before running edges."
        )
    instructions_text = instructions_text.replace("[CORE_MELODY]", core_melody_text)

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
        log_file=str(log_path),
    )

    text = (response or "").strip()
    if not text:
        raise UserActionRequired(
            "Melody edges step produced an empty response. Edit melodyelements.txt or retry."
        )

    processed_text = _reformat_melody_edges_response(text, beats_per_measure)

    # Store the dwell + edge description with validated measure numbers for human editing.
    save_text(edges_path, processed_text + ("" if processed_text.endswith("\n") else "\n"))

    try:
        _log_info(
            f"MUSIC: wrote melody edges artifact for tp={tp_index:02d} at {tp_dir}",
            tp_dir,
        )
    except Exception:
        pass

    return True


def _reformat_melody_edges_response(text: str, beats_per_measure: float) -> str:
    """Parse the LLM response and validate the provided measure numbers."""

    lines = [ln.rstrip() for ln in (text or "").splitlines()]
    if not lines:
        raise UserActionRequired("Melody edges response was empty after stripping whitespace.")

    idx = 0
    total = len(lines)

    # Gather dwell-note lines until a blank separator is encountered.
    dwell_lines: List[str] = []
    while idx < total:
        current = lines[idx]
        if not current.strip():
            idx += 1
            break
        dwell_lines.append(current)
        idx += 1

    # Skip any additional blank lines before the edge sections.
    while idx < total and not lines[idx].strip():
        idx += 1

    if not dwell_lines:
        raise UserActionRequired(
            "Melody edges response did not include dwell notes before the edge sections."
        )

    beats_per_measure = beats_per_measure or 4.0
    edge_blocks: List[str] = []

    while idx < total:
        # Skip stray blank lines between edge sections.
        while idx < total and not lines[idx].strip():
            idx += 1
        if idx >= total:
            break

        edge_name = lines[idx].strip()
        idx += 1

        while idx < total and not lines[idx].strip():
            idx += 1
        if idx >= total:
            raise UserActionRequired(
                f"Melody edge '{edge_name}' is missing its CSV header."
            )

        header_line = lines[idx].strip()
        idx += 1

        data_lines: List[str] = []
        while idx < total and lines[idx].strip():
            data_lines.append(lines[idx].strip())
            idx += 1

        # Prepare the normalized CSV rows and verify measure alignment.
        csv_rows = _edge_rows_with_measures(edge_name, header_line, data_lines, beats_per_measure)

        if edge_blocks:
            edge_blocks.append("")
        edge_blocks.append(edge_name)
        edge_blocks.append("measure,element,duration")
        edge_blocks.extend(csv_rows)

    if not edge_blocks:
        raise UserActionRequired(
            "Melody edges response did not contain any edge CSV sections."
        )

    output_lines: List[str] = []
    output_lines.extend(dwell_lines)
    output_lines.append("")
    output_lines.extend(edge_blocks)

    return "\n".join(line for line in output_lines).strip()


def _edge_rows_with_measures(
    edge_name: str,
    header_line: str,
    data_lines: List[str],
    beats_per_measure: float,
) -> List[str]:
    if not header_line.strip():
        raise UserActionRequired(f"Melody edge '{edge_name}' is missing a CSV header.")
    if not data_lines:
        raise UserActionRequired(f"Melody edge '{edge_name}' did not include any rows to parse.")

    csv_text = "\n".join([header_line] + data_lines)
    try:
        reader = csv.DictReader(StringIO(csv_text))
    except Exception as exc:  # pragma: no cover - defensive
        raise UserActionRequired(
            f"Unable to parse CSV for melody edge '{edge_name}': {exc}"
        ) from exc

    fieldnames = [fn.strip().lower() for fn in (reader.fieldnames or []) if isinstance(fn, str)]
    if "measure" not in fieldnames:
        raise UserActionRequired(
            f"Melody edge '{edge_name}' must include a 'measure' column copied from the CORE_MELODY rows."
        )
    if "root" not in fieldnames:
        raise UserActionRequired(
            f"Melody edge '{edge_name}' must include a 'root' column so the dwell reference can be validated."
        )
    if "duration" not in fieldnames or ("element" not in fieldnames and "note" not in fieldnames):
        raise UserActionRequired(
            f"Melody edge '{edge_name}' must include 'element' (or 'note') and 'duration' columns."
        )

    beats_per_measure = beats_per_measure or 4.0
    if beats_per_measure <= 0:
        beats_per_measure = 4.0

    rows: List[str] = []
    beat_cursor = 0.0
    row_index = 0
    for raw_row in reader:
        row_index += 1
        normalized = {
            (key or "").strip().lower(): (value or "").strip()
            for key, value in (raw_row or {}).items()
            if key is not None
        }
        measure_text = normalized.get("measure")
        if not measure_text:
            raise UserActionRequired(
                f"Melody edge '{edge_name}' row {row_index} is missing a measure value."
            )
        try:
            measure = int(float(measure_text))
        except Exception as exc:  # pragma: no cover - defensive
            raise UserActionRequired(
                f"Melody edge '{edge_name}' row {row_index} has an invalid measure '{measure_text}'."
            ) from exc

        root = normalized.get("root")
        if not root:
            raise UserActionRequired(
                f"Melody edge '{edge_name}' row {row_index} is missing a root value."
            )

        element = normalized.get("element") or normalized.get("note")
        if not element:
            raise UserActionRequired(
                f"Melody edge '{edge_name}' row {row_index} is missing an element value."
            )
        duration_text = normalized.get("duration")
        if not duration_text:
            raise UserActionRequired(
                f"Melody edge '{edge_name}' row {row_index} is missing a duration value."
            )
        try:
            duration = float(duration_text)
        except Exception as exc:  # pragma: no cover - defensive
            raise UserActionRequired(
                f"Melody edge '{edge_name}' row {row_index} has an invalid duration '{duration_text}'."
            ) from exc

        expected_measure = int(math.floor(beat_cursor / beats_per_measure)) + 1
        if measure != expected_measure:
            raise UserActionRequired(
                "Melody edge '{edge}' row {row} has measure {reported} but the durations so far indicate measure {expected}. "
                "Fix the CSV so measure numbers match the CORE_MELODY."
                .format(edge=edge_name, row=row_index, reported=measure, expected=expected_measure)
            )

        rows.append(f"{measure},{element},{_format_number(duration)}")
        beat_cursor += duration

    if not rows:
        raise UserActionRequired(
            f"Melody edge '{edge_name}' CSV did not contain any parseable rows."
        )

    return rows


def _next_attempt_path(tp_dir: Path, prefix: str) -> Path:
    attempt = 1
    while True:
        candidate = tp_dir / f"{prefix}_{attempt}.txt"
        if not candidate.exists():
            return candidate
        attempt += 1


def _write_llm_trace(log_path: Path, *, system: str = "", user: str = "", response: str = "") -> None:
    try:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as handle:
            handle.write("=== SYSTEM ===\n" + (system or "") + "\n\n")
            handle.write("=== USER ===\n" + (user or "") + "\n\n")
            handle.write("=== RESPONSE ===\n" + (response or "") + "\n")
    except Exception:
        pass


def _mirror_first_score_attempt_logs(tp_dir: Path, *, voice_token: str, variant: str) -> None:
    tp_dir = Path(tp_dir)
    token_safe = (voice_token or "voice").replace(".", "_")
    variant_safe = (variant or "standard").strip().lower()
    pattern = f"notes_{token_safe}_{variant_safe}.first_score_attempt_*.txt"
    for src in sorted(tp_dir.glob(pattern)):
        stem = src.stem
        marker = ".first_score_attempt_"
        if marker not in stem:
            continue
        attempt = stem.split(marker)[-1]
        if not attempt.isdigit():
            continue
        dest = tp_dir / f"first_score_attempt_{attempt}.txt"
        try:
            dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        except Exception:
            pass


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


def _validate_first_pass_response(text: str) -> Tuple[bool, str]:
    stripped = (text or "").strip()
    if not stripped:
        return False, "empty response"

    import csv
    from io import StringIO

    try:
        reader = csv.DictReader(StringIO(stripped))
        fieldnames_raw = reader.fieldnames or []
        fieldnames = [fn.strip().lower() for fn in fieldnames_raw if isinstance(fn, str)]
    except Exception:
        fieldnames = []
        reader = None

    required = {"measure", "beat", "pitch", "duration", "velocity"}
    if fieldnames and required.issubset(set(fieldnames)):
        has_row = False
        if reader is not None:
            for row in reader:
                if any((row or {}).values()):
                    has_row = True
                    break
        if has_row:
            return True, ""
        return False, "compact CSV contained no note rows"

    ok, message = _validate_musiccsv(stripped)
    if ok:
        return True, ""
    return False, message or "invalid first-pass response"


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
    measure_start: Optional[int] = None,
    measure_end: Optional[int] = None,
    block_index: Optional[int] = None,
    block_count: Optional[int] = None,
    melody_block_csv: str = "",
    melody_previous_block_csv: str = "",
    voice_previous_block_csv: str = "",
    other_voices_block_csv: str = "",
    other_voices_previous_block_csv: str = "",
    attempt_log_prefix: Optional[str] = None,
) -> Tuple[MusicCSV, List[Dict[str, Any]]]:
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

    # Load shared metadata and variant-specific melody CSV.
    metadata_text = ""
    melody_text = ""
    try:
        meta_path = tp_dir / "metadata.json"
        if meta_path.exists():
            metadata_text = meta_path.read_text(encoding="utf-8").strip()
    except Exception:
        metadata_text = ""
    try:
        safe_variant = (variant or "standard").strip().lower()
        melody_path = tp_dir / f"melody_{safe_variant}.csv"
        if melody_path.exists():
            melody_text = melody_path.read_text(encoding="utf-8").strip()
    except Exception:
        melody_text = ""

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

    # If there is no reduced melody grid yet for the canonical
    # melodic line, fall back to using the raw melody CSV text for
    # alignment so the model still sees a complete spine.
    if not melody_reduced and melody_text.strip():
        melody_reduced = melody_text.strip()

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
        melody_csv=melody_text,
        melody_reduced_csv=melody_reduced,
        other_voices_reduced_csv=other_reduced,
        measure_start=measure_start,
        measure_end=measure_end,
        block_index=block_index,
        block_count=block_count,
        measure_window_description=describe_measure_window(
            measure_start or 1,
            measure_end or (measure_start or 1),
            block_index=block_index,
            block_count=block_count,
        )
        if measure_start is not None
        else "",
        melody_block_csv=melody_block_csv,
        melody_previous_block_csv=melody_previous_block_csv,
        voice_previous_block_csv=voice_previous_block_csv,
        other_voices_block_csv=other_voices_block_csv,
        other_voices_previous_block_csv=other_voices_previous_block_csv,
    )
    model, temp, max_tokens = env_for_prompt(
        "music_first_score_prompt.md",
        "MUSIC_FIRST_SCORE",
        default_temp=0.5,
        default_max_tokens=2000,
    )
    reasoning = reasoning_for_prompt("music_first_score_prompt.md", "MUSIC_FIRST_SCORE")

    def _log_path_for_attempt(attempt: int) -> Optional[Path]:
        # Persist prompt/response pairs per voice/variant/attempt for debugging.
        safe_variant = (variant or "standard").strip().lower()
        token_safe = (target_voice.token or "voice").replace(".", "_")
        prefix = attempt_log_prefix or f"notes_{token_safe}_{safe_variant}.first_score_attempt"
        return tp_dir / f"{prefix}_{attempt}.txt"

    # Let the validator drive retries so empty/invalid tables are retried automatically.
    response = llm_call_with_validation(
        (
            "Compose a valid notes CSV for this single voice using "
            "the specified header: measure,beat,pitch,duration,velocity,tie,articulation."
        ),
        prompt,
        model=model,
        temperature=temp,
        max_tokens=max_tokens,
        validator=_validate_first_pass_response,
        reasoning_effort=reasoning,
        log_maker=_log_path_for_attempt,
        context_tag=f"music-first-score-{target_voice.token}-{variant}",
    )

    text = (response or "").strip()
    if not text:  # pragma: no cover - defensive safeguard
        raise UserActionRequired(
            "First-pass music step produced an empty response. Edit the first_score_attempt log and retry."
        )

    import csv
    from io import StringIO

    compact_rows: List[Dict[str, Any]] = []
    header_lookup: Dict[str, str] = {}
    try:
        reader = csv.DictReader(StringIO(text))
        raw_fields = reader.fieldnames or []
        header_lookup = {fn.strip().lower(): fn for fn in raw_fields if isinstance(fn, str) and fn.strip()}
        required_cols = {"measure", "beat", "pitch", "duration", "velocity"}
        has_compact_header = required_cols.issubset(header_lookup.keys())
        if has_compact_header:
            compact_rows = list(reader)
    except Exception:
        compact_rows = []
        header_lookup = {}

    # If the response is not in compact CSV form, fall back to parsing a full MusicCSV blob.
    if not compact_rows:
        ok, message, music = _try_parse_musiccsv(text)
        if ok and music is not None:
            return music, _note_dicts_from_music(music)
        raise UserActionRequired(
            "Unable to parse the first-pass response as compact CSV or MusicCSV. "
            + (message or "Edit the attempt log and retry.")
        )

    # Build a MusicCSV using existing metadata/measures and a single notes track.
    music = MusicCSV(metadata={}, measures=[], tracks=[], notes=[])

    # Reuse metadata/measures already loaded above where possible.
    try:
        import json as _json
        if metadata_text.strip():
            meta_obj = _json.loads(metadata_text)
            if isinstance(meta_obj, dict):
                music.metadata = meta_obj
    except Exception:
        music.metadata = {}

    # Measures are derived separately from the melody CSV; they should
    # already exist as measures_<variant>.csv for downstream assembly.

    # Single track entry for this voice; further refinement can add more metadata later.
    music.tracks = [
        {
            "track": 1,
            "label": target_voice.token,
            "part": target_voice.role or target_voice.idea or "voice",
            "instrument": target_voice.instrument,
            "channel": 1,
            "program": 1,
            "volume": 100,
        }
    ]

    # Map compact CSV rows into MusicCSV note dicts.
    tie_key = header_lookup.get("tie")
    art_key = header_lookup.get("articulation")
    for row in compact_rows:
        try:
            measure = int(row.get(header_lookup.get("measure", ""), 0) or 0)
            beat = float(row.get(header_lookup.get("beat", ""), 0) or 0)
            pitch = str(row.get(header_lookup.get("pitch", ""), "") or "").strip()
            duration = float(row.get(header_lookup.get("duration", ""), 0) or 0)
            velocity = int(row.get(header_lookup.get("velocity", ""), 0) or 0)
            tie = str(row.get(tie_key, "") or "").strip() if tie_key else ""
            articulation = str(row.get(art_key, "") or "").strip() if art_key else ""
        except Exception:
            continue
        if not (measure and beat and pitch and duration):
            continue
        note = {
            "track": 1,
            "measure": measure,
            "beat": beat,
            "pitch": pitch,
            "duration": duration,
            "velocity": velocity,
            "tie": tie,
            "articulation": articulation,
            "pedal": "",
            "lyric": "",
            "ornament": "",
            "comment": "",
        }
        music.notes.append(note)

    if not music.notes:
        raise UserActionRequired(
            "First-pass music CSV contained no valid note rows. Edit the first_score_attempt log and retry."
        )

    rows = _note_dicts_from_music(music)
    if measure_start is not None and measure_end is not None:
        rows = _filter_rows_by_measure(rows, measure_start=measure_start, measure_end=measure_end)
    return music, rows


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

    music, _ = _compose_first_pass_for_voice(
        tp_dir=tp_dir,
        tp_index=tp_index,
        tp_type=tp_type,
        tp_text=tp_text,
        voice_context=voice_context,
        prompt_payload=prompt_payload,
        target_voice_index=0,
        variant="standard",
    )

    try:
        _mirror_first_score_attempt_logs(
            tp_dir,
            voice_token=voice_context.voices[0].token,
            variant="standard",
        )
    except Exception:
        pass

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
        log_file=str(score_check_trace),
    )
    _write_llm_trace(
        score_check_trace,
        system="Provide concise, actionable feedback on the score.",
        user=check_prompt,
        response=suggestions or "",
    )
    save_text(first_suggestions_path, suggestions)

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
    """Compose first-pass per-voice scores for a single variant using block windows.

    Each voice is composed in 10-measure blocks, persisting block CSVs,
    per-voice progress JSON, and merged ``notes_<voice>_<variant>.csv`` files.
    """

    tp_dir = Path(tp_dir)
    tp_dir.mkdir(parents=True, exist_ok=True)

    written: List[Path] = []
    safe_variant = (variant or "standard").strip().lower()

    total_measures = _measure_count_for_variant(tp_dir, safe_variant)
    if total_measures <= 0:
        total_measures = _BLOCK_SIZE
    block_count = max(1, math.ceil(total_measures / _BLOCK_SIZE))

    melody_text = ""
    melody_path = tp_dir / f"melody_{safe_variant}.csv"
    if melody_path.exists():
        try:
            melody_text = melody_path.read_text(encoding="utf-8")
        except Exception:
            melody_text = ""

    metadata_dict: Dict[str, Any] = {}
    metadata_path = tp_dir / "metadata.json"
    if metadata_path.exists():
        try:
            metadata_dict = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            metadata_dict = {}
    time_signature = str(metadata_dict.get("time_signature") or "4/4")
    beats_per_measure = _beats_per_measure_from_signature(time_signature)

    progress = _load_music_progress(tp_dir, safe_variant)
    voices_state: Dict[str, Dict[str, Any]] = progress.setdefault("voices", {})

    for idx, voice in enumerate(voice_context.voices):
        token_safe = (voice.token or "voice").replace(".", "_")
        notes_path = _notes_csv_path(tp_dir, voice.token, safe_variant)

        state = voices_state.get(voice.token)
        if state is None:
            if notes_path.exists():
                # Assume legacy runs completed; mark as done so we do not clobber edited files.
                voices_state[voice.token] = {"next_block": block_count + 1}
                written.append(notes_path)
                continue
            state = {"next_block": 1}
            voices_state[voice.token] = state

        next_block = int(state.get("next_block", 1) or 1)
        if next_block > block_count:
            if notes_path.exists():
                written.append(notes_path)
            continue

        is_melody_voice = _voice_is_melody(voice)

        while next_block <= block_count:
            measure_start, measure_end = block_measure_range(
                next_block,
                block_size=_BLOCK_SIZE,
                total_measures=total_measures,
            )
            prev_start = prev_end = None
            if next_block > 1:
                prev_start, prev_end = block_measure_range(
                    next_block - 1,
                    block_size=_BLOCK_SIZE,
                    total_measures=total_measures,
                )

            melody_block_csv = slice_csv_by_measure(melody_text, measure_start, measure_end)
            melody_prev_csv = (
                slice_csv_by_measure(melody_text, prev_start, prev_end)
                if prev_start is not None and prev_end is not None
                else ""
            )
            voice_prev_block_csv = (
                _slice_notes_file(notes_path, measure_start=prev_start, measure_end=prev_end)
                if prev_start is not None and prev_end is not None
                else ""
            )
            other_block_csv = _collect_other_voice_blocks(
                tp_dir=tp_dir,
                variant=safe_variant,
                voice_context=voice_context,
                exclude_token=voice.token,
                measure_start=measure_start,
                measure_end=measure_end,
            )
            other_prev_csv = (
                _collect_other_voice_blocks(
                    tp_dir=tp_dir,
                    variant=safe_variant,
                    voice_context=voice_context,
                    exclude_token=voice.token,
                    measure_start=prev_start,
                    measure_end=prev_end,
                )
                if prev_start is not None and prev_end is not None
                else ""
            )

            block_log_prefix = f"notes_{token_safe}_{safe_variant}.block{next_block:02d}"

            music, block_rows = _compose_first_pass_for_voice(
                tp_dir=tp_dir,
                tp_index=tp_index,
                tp_type=tp_type,
                tp_text=tp_text,
                voice_context=voice_context,
                prompt_payload=prompt_payload,
                target_voice_index=idx,
                variant=safe_variant,
                suggestions_text=suggestions_text,
                measure_start=measure_start,
                measure_end=measure_end,
                block_index=next_block,
                block_count=block_count,
                melody_block_csv=melody_block_csv,
                melody_previous_block_csv=melody_prev_csv,
                voice_previous_block_csv=voice_prev_block_csv,
                other_voices_block_csv=other_block_csv,
                other_voices_previous_block_csv=other_prev_csv,
                attempt_log_prefix=block_log_prefix,
            )

            block_rows = _sanitize_block_note_rows(
                block_rows,
                measure_start=measure_start,
                measure_end=measure_end,
                beats_per_measure=beats_per_measure,
                preserve_input_order=is_melody_voice,
            )

            block_csv_path = _block_csv_path(tp_dir, voice.token, safe_variant, next_block)
            _write_note_rows(block_csv_path, block_rows)
            _replace_rows_in_range(
                notes_path,
                block_rows,
                measure_start=measure_start,
                measure_end=measure_end,
            )

            _mirror_block_attempt_logs(
                tp_dir=tp_dir,
                voice_token=voice.token,
                variant=safe_variant,
                block_index=next_block,
                block_count=block_count,
                source_prefix=block_log_prefix,
            )

            state["next_block"] = next_block + 1
            progress["voices"] = voices_state
            _save_music_progress(tp_dir, safe_variant, progress)
            next_block = state["next_block"]

        written.append(notes_path)

    _save_music_progress(tp_dir, safe_variant, progress)
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
    standard_sequence = (
        "* Start with this sequence of edges: A-A, A-A, A-B1, B1-B2, B2-A, A-A. "
        "Follow these edges in order before selecting additional paths to reach the target length."
    )
    complimentary_sequence = (
        "* Start with this sequence of edges: A-B1, B1-B2, B2-B1, B1-B2, B2-A. "
        "Use it as the opening gesture before weaving new material."
    )
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
        additional_rules += "\n" + complimentary_sequence
    elif variant_safe == "reprise":
        additional_rules = (
            "* Treat this as a reprise of the standard melody. Before "
            "constructing the final line, conceptually stretch each edge of "
            "the melodic graph by roughly one additional measure, adding "
            "connecting notes that make musical sense so that the total "
            "duration expands while preserving the recognizable contour of "
            "the original melody."
        )
        additional_rules += "\n" + standard_sequence
    else:
        additional_rules = standard_sequence

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
        log_file=str(log_path),
    )

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
        log_file=str(attempt_log_path),
    )
    _write_llm_trace(
        attempt_log_path,
        system="Refine the MusicCSV score according to feedback while keeping it valid.",
        user=prompt,
        response=response or "",
    )
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
        log_file=str(subtle_check_trace),
    )
    _write_llm_trace(
        subtle_check_trace,
        system="Provide concise, actionable feedback on the score.",
        user=check_prompt,
        response=suggestions or "",
    )
    save_text(final_feedback_path, suggestions)

    try:
        _log_info(
            f"MUSIC: completed subtle score pass at {tp_dir}",
            tp_dir,
        )
    except Exception:
        pass

    return True


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
    touch_point_path = tp_dir / "music_touch_point.json"
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

    if isinstance(music.metadata, dict) and not str(music.metadata.get("version", "")).strip():
        music.metadata["version"] = "1.0"

    # Load tracks.csv with basic type coercion.
    try:
        with tracks_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            cleaned: List[Dict[str, Any]] = []
            for row in reader:
                entry = dict(row)
                track_id = _coerce_int(entry.get("track"))
                if track_id is not None:
                    entry["track"] = track_id
                channel = _coerce_int(entry.get("channel"))
                if channel is not None:
                    entry["channel"] = channel
                program = _coerce_int(entry.get("program"))
                if program is not None:
                    entry["program"] = program
                volume = _coerce_int(entry.get("volume"))
                if volume is not None:
                    entry["volume"] = volume
                cleaned.append(entry)
            music.tracks = cleaned
    except Exception as exc:
        _log_warning(f"MUSIC: failed to read tracks.csv for first-pass assembly: {exc}", tp_dir)
        return None

    # Load touch-point manifest for voice ordering if present.
    voice_tokens: List[str] = []
    if touch_point_path.exists():
        try:
            import json as _json

            tp_data = _json.loads(touch_point_path.read_text(encoding="utf-8"))
            raw_tokens = tp_data.get("voice_tokens") or tp_data.get("raw_payload", {}).get("voices")
            if isinstance(raw_tokens, list):
                voice_tokens = [str(tok) for tok in raw_tokens if isinstance(tok, str) and tok.strip()]
        except Exception:
            voice_tokens = []

    # Load measures_<variant>.csv with type coercion.
    try:
        with measures_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            cleaned_measures: List[Dict[str, Any]] = []
            for row in reader:
                entry = dict(row)
                measure_no = _coerce_int(entry.get("measure"))
                if measure_no is not None:
                    entry["measure"] = measure_no
                start_beat = _coerce_float(entry.get("start_beat"))
                if start_beat is not None:
                    entry["start_beat"] = start_beat
                tempo_value = _coerce_float(entry.get("tempo"))
                if tempo_value is not None:
                    entry["tempo"] = tempo_value
                pickup_value = _coerce_bool(entry.get("pickup"))
                if pickup_value is not None:
                    entry["pickup"] = pickup_value
                cleaned_measures.append(entry)
            music.measures = cleaned_measures
    except Exception as exc:
        _log_warning(f"MUSIC: failed to read measures_{safe_variant}.csv for first-pass assembly: {exc}", tp_dir)
        return None

    # Build a simple lookup from track label to numeric track index so we
    # can assign the correct track value to each note row. We treat the
    # ``label`` column in tracks.csv as the canonical voice token.
    label_to_track: Dict[str, int] = {}
    track_order: List[Tuple[int, Dict[str, Any]]] = []
    for tr in music.tracks:
        try:
            label = str(tr.get("label", "") or "").strip()
            tval = tr.get("track")
            track_num = int(tval) if tval is not None and str(tval) != "" else None
        except Exception:
            label = ""
            track_num = None
        if label and track_num is not None:
            label_to_track[label] = track_num
        token_label = str(tr.get("voice_token", "") or "").strip()
        if token_label and track_num is not None:
            label_to_track[token_label] = track_num
        if track_num is not None:
            track_order.append((track_num, tr))

    track_order.sort(key=lambda item: item[0])
    voice_token_to_track: Dict[str, int] = {}
    if voice_tokens and len(voice_tokens) == len(track_order):
        for idx, token in enumerate(voice_tokens):
            voice_token_to_track[token] = track_order[idx][0]

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

    # Append note rows from each notes CSV. Track assignment is derived
    # from the voice token encoded in the file name and mapped through
    # tracks.csv so that the assembled MusicCSV satisfies the schema.
    for path in sorted(notes_files):
        try:
            with path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Ensure minimal required columns exist.
                    measure_val = _coerce_int(row.get("measure"))
                    beat_val = _coerce_float(row.get("beat"))
                    duration_val = _coerce_float(row.get("duration"))
                    pitch_val = str(row.get("pitch", "") or "").strip()
                    velocity_val = _coerce_int(row.get("velocity"))
                    if measure_val is None or measure_val <= 0:
                        continue
                    if beat_val is None or beat_val <= 0:
                        continue
                    if duration_val is None or duration_val <= 0:
                        continue
                    if not pitch_val:
                        pitch_val = "rest"
                    tie_val = str(row.get("tie", "") or "").strip()
                    if tie_val.lower() not in _ALLOWED_TIES:
                        tie_val = ""
                    articulation_val = str(row.get("articulation", "") or "").strip()
                    if velocity_val is None or not (0 <= velocity_val <= 127):
                        velocity_val = None

                    note = {
                        "measure": measure_val,
                        "beat": beat_val,
                        "pitch": pitch_val,
                        "duration": duration_val,
                        "velocity": velocity_val,
                        "tie": tie_val,
                        "articulation": articulation_val,
                    }
                    # Derive track from file name: notes_<voice_token>_<variant>.csv
                    # We strip the leading 'notes_' prefix and trailing '_<variant>.csv'.
                    name = path.name
                    try:
                        core = name[len("notes_") : -len(suffix)] if name.startswith("notes_") and name.endswith(suffix) else ""
                    except Exception:
                        core = ""
                    track_num: Optional[int] = None
                    if core and core in label_to_track:
                        track_num = label_to_track[core]
                    # Fallback: if no direct label match, attempt a loose match
                    # by splitting on '.' and matching the longest prefix.
                    if track_num is None and core:
                        parts = [p for p in core.split(".") if p]
                        for k, v in label_to_track.items():
                            if not k:
                                continue
                            if k == core:
                                track_num = v
                                break
                            if parts and k.lower().endswith(parts[-1].lower()):
                                track_num = v
                                break
                    if track_num is None and core and voice_token_to_track:
                        track_num = voice_token_to_track.get(core)
                    if track_num is None:
                        continue
                    note["track"] = track_num
                    music.notes.append(note)
        except Exception as exc:
            _log_warning(f"MUSIC: failed to merge notes from {path}: {exc}", tp_dir)

    # Validate and write out the assembled first-pass score.
    try:
        validate_musiccsv(music)
    except Exception as exc:
        _log_warning(
            f"MUSIC: assembled first-pass variant '{safe_variant}' failed validation at {tp_dir}: {exc}",
            tp_dir,
        )
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


def run_first_pass_checks(
    *,
    tp_dir: Path,
    tp_index: int,
    tp_type: str,
    tp_text: str,
    prompt_payload: Optional[Dict[str, Any]],
    title: str,
    variants: Optional[Iterable[str]] = None,
) -> bool:
    """Run the music_check prompt over each assembled first-pass variant.

    Writes ``first_score_check_<variant>.txt`` (prompt + response) and
    ``first_score_suggestions_<variant>.txt`` beside the assembled
    ``first_<title>_<variant>.musiccsv`` artifacts. Returns True if at least
    one variant produced suggestions.
    """

    if not prompt_payload:
        return False

    tp_dir = Path(tp_dir)
    tp_dir.mkdir(parents=True, exist_ok=True)

    title_safe = str(title or prompt_payload.get("touch_point_title") or "untitled").strip()
    if not title_safe:
        title_safe = "untitled"
    score_title = title_safe.replace(" ", "_")

    variant_list = list(variants or ("standard",))
    wrote_any = False

    for variant in variant_list:
        variant_safe = (variant or "standard").strip().lower()
        score_path = tp_dir / f"first_{score_title}_{variant_safe}.musiccsv"
        if not score_path.exists():
            continue
        try:
            music_obj = read_musiccsv(score_path)
            snippet_text = _truncate_musiccsv_text(musiccsv_to_text(music_obj), None)
        except Exception as exc:
            _log_warning(
                f"MUSIC: unable to prepare first-pass variant '{variant_safe}' for checks: {exc}",
                tp_dir,
            )
            continue

        check_prompt = build_music_check_prompt(
            prompt_payload=prompt_payload,
            tp_index=tp_index,
            tp_type=tp_type,
            tp_text=tp_text,
            musiccsv_snippet=snippet_text,
        )
        model, temp, max_tokens = env_for_prompt(
            "music_check_prompt.md",
            "MUSIC_SCORE_CHECK",
            default_temp=0.0,
            default_max_tokens=800,
        )
        log_path = tp_dir / f"first_score_check_{variant_safe}.txt"
        try:
            suggestions = llm_complete(
                check_prompt,
                system="Provide concise, actionable feedback on the score.",
                temperature=temp,
                max_tokens=max_tokens,
                model=model,
                log_file=str(log_path),
            )
        except Exception as exc:
            raise UserActionRequired(
                f"First-pass music check failed for variant '{variant_safe}'. Inspect {log_path.name} and retry."
            ) from exc

        sugg_path = tp_dir / f"first_score_suggestions_{variant_safe}.txt"
        save_text(sugg_path, (suggestions or "").strip() + ("\n" if suggestions and not suggestions.endswith("\n") else ""))
        wrote_any = True

    return wrote_any


__all__ = [
    "run_metadata_tracks_step",
    "run_melody_emotion_step",
    "run_emotion_chord_step",
    "ensure_core_melody_csv",
    "run_melody_edges_step",
    "run_melody_construction_step",
    "ensure_first_score_gate",
    "run_subtle_score_pass",
    "run_multi_voice_first_pass",
    "assemble_first_pass_variant",
    "run_first_pass_checks",
]
