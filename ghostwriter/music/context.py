"""Voice metadata parsing and context assembly for music pipelines.

Slice 2 of the music feature introduces utilities that extract voice tokens
from chapter YAML, map those voices onto known story entities, and collect
basic tempo/key summaries from any sanitized scores already present in the
iterations directory. The resulting structures feed later prompt builders
and gate orchestration.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import logging

from ..context import RunContext, UserActionRequired
from ..templates import iter_dir_for
from ..utils import _norm_token
from ..musiccsv import read_musiccsv, validate_musiccsv, resolve_derived_fields, MusicCSV


logger = logging.getLogger(__name__)

_ALLOWED_CHORD_QUALITIES = {
    "major",
    "minor",
    "diminished",
    "augmented",
    "chromatic",
    "modal",
    "neutral",
    "suspended",
}

_ALLOWED_REGISTERS = {
    "bass",
    "baritone",
    "tenor",
    "alto",
    "mezzo",
    "soprano",
    "treble",
    "beat",
    "percussion",
    "drone",
}


@dataclass
class VoiceSpec:
    """Structured representation of a chapter voice token."""

    token: str
    chord: str
    register: str
    instrument: str
    idea: str
    role: Optional[str] = None
    character_refs: List[Dict[str, Any]] = field(default_factory=list)
    factoid_refs: List[Dict[str, Any]] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def character_ids(self) -> List[str]:
        ids: List[str] = []
        for ref in self.character_refs:
            cid = ref.get("id") or ref.get("name")
            if not cid:
                continue
            text = str(cid)
            if text not in ids:
                ids.append(text)
        return ids

    @property
    def character_names(self) -> List[str]:
        names: List[str] = []
        for ref in self.character_refs:
            name = ref.get("name") or ref.get("id")
            if not name:
                continue
            text = str(name)
            if text not in names:
                names.append(text)
        return names

    @property
    def factoid_names(self) -> List[str]:
        names: List[str] = []
        for ref in self.factoid_refs:
            name = ref.get("name")
            if not name:
                continue
            text = str(name)
            if text not in names:
                names.append(text)
        return names


@dataclass
class ScoreSummary:
    """Lightweight metadata derived from a sanitized MusicCSV score."""

    voice_token: str
    score_path: Path
    tempos: List[str]
    time_signatures: List[str]
    key_signatures: List[str]
    measure_count: int
    duration_quarter_length: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "voice_token": self.voice_token,
            "score_path": str(self.score_path),
            "tempos": list(self.tempos),
            "time_signatures": list(self.time_signatures),
            "key_signatures": list(self.key_signatures),
            "measure_count": self.measure_count,
            "duration_quarter_length": self.duration_quarter_length,
        }


@dataclass
class VoiceContext:
    """Aggregated music context for a chapter iteration."""

    directive: str
    voices: List[VoiceSpec]
    score_summaries: Dict[str, ScoreSummary]
    aggregate_tempos: List[str]
    aggregate_time_signatures: List[str]
    aggregate_key_signatures: List[str]
    missing_assets: List[str]

    def as_prompt_payload(self) -> Dict[str, Any]:
        tempo_summary = ", ".join(self.aggregate_tempos)
        time_sig_summary = ", ".join(self.aggregate_time_signatures)
        key_summary = ", ".join(self.aggregate_key_signatures)

        summary_entries: List[Dict[str, Any]] = []
        for spec in self.voices:
            summary = self.score_summaries.get(spec.token)
            if summary:
                summary_entries.append(summary.as_dict())
        for token, summary in self.score_summaries.items():
            if all(entry.get("voice_token") != token for entry in summary_entries):
                summary_entries.append(summary.as_dict())

        # Optionally include character outlines for any referenced characters across voices
        include_char_ctx = os.getenv("GW_INCLUDE_CHARACTER_OUTLINES_MUSIC", "1") == "1"
        character_outlines: List[Dict[str, Any]] = []
        if include_char_ctx:
            seen_keys: set[str] = set()
            wanted_fields = {
                "id",
                "name",
                "background",
                "traits",
                "cadence",
                "lexicon",
                "prefer",
                "avoid",
                "mannerisms",
                "sample_lines",
                "common_lines",
                "rare_lines",
                "forbidden",
            }
            for spec in self.voices:
                for ref in getattr(spec, "character_refs", []) or []:
                    if not isinstance(ref, dict):
                        continue
                    key = str(ref.get("id") or ref.get("name") or "").strip().lower()
                    if not key or key in seen_keys:
                        continue
                    seen_keys.add(key)
                    outline: Dict[str, Any] = {k: ref[k] for k in wanted_fields if k in ref}
                    # Ensure deterministic key ordering by adding a token list reference if applicable
                    if outline:
                        character_outlines.append(outline)

        return {
            "directive": self.directive,
            "tempo_summary": tempo_summary,
            "time_signature_summary": time_sig_summary,
            "key_summary": key_summary,
            "voices": [
                {
                    "token": spec.token,
                    "chord": spec.chord,
                    "register": spec.register,
                    "instrument": spec.instrument,
                    "idea": spec.idea,
                    "role": spec.role or "",
                    "characters": spec.character_ids,
                    "character_names": spec.character_names,
                    "factoids": spec.factoid_names,
                    "issues": list(spec.issues),
                    "metadata": dict(spec.metadata),
                }
                for spec in self.voices
            ],
            "score_summaries": summary_entries,
            "character_outlines": character_outlines,
        }


def parse_voice_token(token: str, *, metadata: Optional[Dict[str, Any]] = None) -> VoiceSpec:
    """Parse a dot-separated voice token into structured components.

    Parameters
    ----------
    token:
        Voice descriptor in the format ``chord.register.instrument.idea[.role]``.
    metadata:
        Optional extra information carried alongside the voice entry in YAML.
    """

    metadata = dict(metadata or {})
    raw = (token or "").strip()
    issues: List[str] = []
    if not raw:
        issues.append("Voice token is empty.")
        return VoiceSpec(
            token="",
            chord="",
            register="",
            instrument="",
            idea="",
            role=None,
            issues=issues,
            metadata=metadata,
        )

    parts = [segment.strip() for segment in raw.split(".") if segment.strip()]
    chord = parts[0] if len(parts) >= 1 else ""
    register = parts[1] if len(parts) >= 2 else ""
    instrument = parts[2] if len(parts) >= 3 else ""
    idea = parts[3] if len(parts) >= 4 else ""

    if len(parts) < 4:
        issues.append("Voice token must contain chord.register.instrument.idea segments.")
    if chord and chord.lower() not in _ALLOWED_CHORD_QUALITIES:
        issues.append("Chord quality '%s' not recognized (expected e.g. major/minor/augmented)." % chord)
    if register and register.lower() not in _ALLOWED_REGISTERS:
        issues.append("Register '%s' not recognized (expected e.g. bass/tenor/alto)." % register)
    if not instrument:
        issues.append("Instrument segment is empty.")
    if not idea:
        issues.append("Idea segment is empty.")

    if len(parts) >= 5:
        role = ".".join(parts[4:])
    else:
        meta_role = metadata.get("role")
        role = str(meta_role).strip() if isinstance(meta_role, str) and meta_role.strip() else None

    return VoiceSpec(
        token=raw,
        chord=chord,
        register=register,
        instrument=instrument,
        idea=idea,
        role=role,
        issues=issues,
        metadata=metadata,
    )


def build_voice_context(
    ctx: RunContext,
    *,
    pipeline_version: Optional[int] = None,
    voice_tokens: Optional[Any] = None,
    directive: Optional[str] = None,
) -> VoiceContext:
    """Construct a :class:`VoiceContext` for the provided run context."""

    if voice_tokens is not None:
        voice_entries: List[tuple[str, Dict[str, Any]]] = []
        sources = voice_tokens if isinstance(voice_tokens, (list, tuple, set)) else [voice_tokens]
        for entry in sources:
            voice_entries.extend(_normalize_voice_entry(entry))
    else:
        voice_entries = _extract_voice_entries(ctx.chapter)

    if not voice_entries:
        raise UserActionRequired(
            "No valid music voices could be extracted from the chapter. "
            "Ensure your CHAPTER_XXX.yaml provides a 'voices' entry either at "
            "the top level or inside the music touch-point."
        )

    seen_tokens: set[str] = set()
    raw_specs: List[VoiceSpec] = []
    for token, meta in voice_entries:
        norm = _normalize_lookup(token)
        if norm in seen_tokens:
            continue
        seen_tokens.add(norm)
        raw_specs.append(parse_voice_token(token, metadata=meta))

    # Reorder voices so that melodic lines are composed first.
    # Heuristic: prefer ideas containing "melody" or roles containing
    # "melody"/"lead", then fall back to the original order.
    def _is_melody(spec: VoiceSpec) -> bool:
        idea = (spec.idea or "").lower()
        role = (spec.role or "").lower()
        return ("melody" in idea) or ("melody" in role) or ("lead" in role)

    melody_specs = [s for s in raw_specs if _is_melody(s)]
    non_melody_specs = [s for s in raw_specs if not _is_melody(s)]
    voice_specs: List[VoiceSpec] = melody_specs + non_melody_specs
    _attach_entity_links(voice_specs, ctx)

    version = pipeline_version if pipeline_version is not None else ctx.version
    score_summaries = _collect_score_summaries(ctx, voice_specs, version)

    aggregates = _aggregate_summaries(score_summaries)
    missing_assets = [spec.token for spec in voice_specs if spec.token not in score_summaries]

    if directive is None:
        directive_text = _extract_music_directive(ctx.chapter)
    else:
        directive_text = str(directive)

    return VoiceContext(
        directive=directive_text,
        voices=voice_specs,
        score_summaries=score_summaries,
        aggregate_tempos=aggregates["tempos"],
        aggregate_time_signatures=aggregates["time_signatures"],
        aggregate_key_signatures=aggregates["key_signatures"],
        missing_assets=missing_assets,
    )


def build_music_prompt_context(
    ctx: RunContext,
    *,
    pipeline_version: Optional[int] = None,
    voice_tokens: Optional[Any] = None,
    directive: Optional[str] = None,
) -> Dict[str, Any]:
    """Return a serialisable payload describing chapter music context."""

    voice_context = build_voice_context(
        ctx,
        pipeline_version=pipeline_version,
        voice_tokens=voice_tokens,
        directive=directive,
    )
    return voice_context.as_prompt_payload()


def write_voice_token(import_dir: Path, token: str) -> Path:
    """Persist the canonical voice token inside an import directory."""

    import_dir = Path(import_dir)
    import_dir.mkdir(parents=True, exist_ok=True)
    token_path = import_dir / "voice.token"
    token_path.write_text(f"{token.strip()}\n", encoding="utf-8")
    return token_path


# ---------------------------------------------------------------------------
# Internal helpers


def _music_value_to_text(value: Any) -> str:
    if isinstance(value, dict):
        title = str(value.get("title") or "").strip()
        description = str(value.get("description") or "").strip()
        directive = str(value.get("directive") or "").strip()
        pieces: List[str] = []
        if title and description:
            pieces.append(f"{title}: {description}")
        elif title:
            pieces.append(title)
        elif description:
            pieces.append(description)
        if directive:
            pieces.append(directive)
        summary = "\n".join([p for p in pieces if p.strip()])
        if summary.strip():
            return summary.strip()
        return str(value)
    if isinstance(value, str) and value.strip():
        return value.strip()
    if value is not None:
        text = str(value).strip()
        if text:
            return text
    return ""


def _extract_music_directive(chapter: Dict[str, Any]) -> str:
    raw = _get_case_insensitive(chapter, "music")
    text = _music_value_to_text(raw)
    if text:
        return text
    for entry in _gather_touchpoint_values(chapter, "music"):
        text = _music_value_to_text(entry)
        if text:
            return text
    return ""


def _extract_voice_entries(chapter: Dict[str, Any]) -> List[tuple[str, Dict[str, Any]]]:
    entries: List[tuple[str, Dict[str, Any]]] = []
    raw = _get_case_insensitive(chapter, "voices")
    if raw is not None:
        entries.extend(_normalize_voice_entry(raw))
    for entry in _gather_touchpoint_values(chapter, "voices"):
        entries.extend(_normalize_voice_entry(entry))
    return entries


def _normalize_voice_entry(entry: Any) -> List[tuple[str, Dict[str, Any]]]:
    if isinstance(entry, str):
        return [
            (segment.strip(), {})
            for segment in entry.split(",")
            if segment.strip()
        ]
    if isinstance(entry, dict):
        meta = {k: v for k, v in entry.items() if k not in {"token", "voice", "tokens", "voices"}}
        tokens: List[tuple[str, Dict[str, Any]]] = []
        token_value = entry.get("token") or entry.get("voice")
        if isinstance(token_value, str) and token_value.strip():
            tokens.append((token_value.strip(), meta))
        nested = entry.get("tokens") or entry.get("voices")
        if isinstance(nested, list):
            for sub in nested:
                tokens.extend(_normalize_voice_entry(sub))
        return tokens
    if isinstance(entry, (list, tuple)):
        tokens: List[tuple[str, Dict[str, Any]]] = []
        for sub in entry:
            tokens.extend(_normalize_voice_entry(sub))
        return tokens
    text = str(entry).strip()
    return [(text, {})] if text else []


def _attach_entity_links(voice_specs: Iterable[VoiceSpec], ctx: RunContext) -> None:
    characters = ctx.characters if isinstance(ctx.characters, list) else []
    factoids_source = []
    if isinstance(ctx.setting, dict):
        maybe_facts = ctx.setting.get("Factoids")
        if isinstance(maybe_facts, list):
            factoids_source = [f for f in maybe_facts if isinstance(f, dict)]

    character_index = _build_index(characters, keys=("id", "name", "aliases"))
    factoid_index = _build_index(factoids_source, keys=("name",))

    for spec in voice_specs:
        requested_chars = _coerce_to_iterable(spec.metadata.get("character") or spec.metadata.get("characters"))
        requested_facts = _coerce_to_iterable(spec.metadata.get("factoid") or spec.metadata.get("factoids"))

        char_refs: List[Dict[str, Any]] = []
        for token in requested_chars:
            char_refs.extend(character_index.get(token, []))
        idea_norm = _normalize_lookup(spec.idea)
        char_refs.extend(character_index.get(idea_norm, []))
        spec.character_refs = _dedupe_refs(char_refs)

        fact_refs: List[Dict[str, Any]] = []
        for token in requested_facts:
            fact_refs.extend(factoid_index.get(token, []))
        fact_refs.extend(factoid_index.get(idea_norm, []))
        spec.factoid_refs = _dedupe_refs(fact_refs)


def _collect_score_summaries(ctx: RunContext, voice_specs: List[VoiceSpec], version: int) -> Dict[str, ScoreSummary]:
    if not voice_specs:
        return {}
    pipeline_dir = iter_dir_for(ctx.chapter_id) / f"pipeline_v{version}"
    if not pipeline_dir.exists():
        return {}

    index = {_normalize_lookup(spec.token): spec for spec in voice_specs}
    summaries: Dict[str, ScoreSummary] = {}

    for import_dir in pipeline_dir.rglob("*_import"):
        if not import_dir.is_dir():
            continue
        token = _discover_voice_token(import_dir)
        if not token:
            continue
        spec = index.get(_normalize_lookup(token))
        if spec is None:
            continue
        score_path = import_dir / "score.musiccsv"
        if not score_path.exists():
            score_path = import_dir / "import.musiccsv"
        if not score_path.exists():
            continue
        try:
            summary = _summarize_musiccsv(score_path, spec.token)
        except Exception as exc:  # pragma: no cover - defensive branch
            logger.debug("Failed to summarize %s: %s", score_path, exc)
            continue
        summaries[spec.token] = summary
    return summaries


def _summarize_musiccsv(path: Path, voice_token: str) -> ScoreSummary:
    music = read_musiccsv(path)
    validate_musiccsv(music)
    derived = resolve_derived_fields(music)

    def _format_number(value: Any) -> str:
        try:
            number = float(value)
        except Exception:
            return str(value)
        text = f"{number:.2f}"
        return text.rstrip("0").rstrip(".") if "." in text else text

    tempos: set[str] = set()
    meta_tempo = (music.metadata or {}).get("tempo")
    if meta_tempo is not None:
        tempos.add(_format_number(meta_tempo))
    for measure in music.measures:
        tempo_value = measure.get("tempo")
        if tempo_value is not None:
            tempos.add(_format_number(tempo_value))
    if not tempos:
        tempos.add("Unknown")

    time_signatures: set[str] = set()
    meta_signature = (music.metadata or {}).get("time_signature")
    if isinstance(meta_signature, str) and meta_signature.strip():
        time_signatures.add(meta_signature.strip())
    for measure in music.measures:
        value = measure.get("time_signature")
        if isinstance(value, str) and value.strip():
            time_signatures.add(value.strip())
    if not time_signatures:
        time_signatures.add("Unknown")

    key_signatures: set[str] = set()
    meta_key = (music.metadata or {}).get("key_signature")
    if isinstance(meta_key, str) and meta_key.strip():
        key_signatures.add(meta_key.strip())
    for measure in music.measures:
        value = measure.get("key_signature")
        if isinstance(value, str) and value.strip():
            key_signatures.add(value.strip())
    if not key_signatures:
        key_signatures.add("Unknown")

    measure_numbers = [int(row.get("measure", 0) or 0) for row in music.measures if row.get("measure")]
    measure_count = max(measure_numbers) if measure_numbers else 0
    if measure_count == 0:
        note_measures = [int(note.get("measure", 0) or 0) for note in music.notes if note.get("measure")]
        if note_measures:
            measure_count = max(note_measures)

    divisions = int((music.metadata or {}).get("divisions_per_quarter") or 480)
    total_ticks = 0
    for note in derived.notes:
        absolute_tick = note.get("absolute_tick")
        duration_ticks = note.get("duration_ticks")
        if absolute_tick is None or duration_ticks is None:
            continue
        total_ticks = max(total_ticks, int(absolute_tick) + int(duration_ticks))
    duration_quarter_length = float(total_ticks / divisions) if divisions else 0.0

    return ScoreSummary(
        voice_token=voice_token,
        score_path=Path(path),
        tempos=sorted(tempos),
        time_signatures=sorted(time_signatures),
        key_signatures=sorted(key_signatures),
        measure_count=measure_count,
        duration_quarter_length=duration_quarter_length,
    )


def _discover_voice_token(import_dir: Path) -> Optional[str]:
    token_file = import_dir / "voice.token"
    if token_file.exists():
        text = token_file.read_text(encoding="utf-8").strip()
        if text:
            return text

    name = import_dir.name
    stem = name[:-7] if name.endswith("_import") else name
    if "_track_" in stem:
        stem = stem.split("_track_", 1)[1]
    candidate = stem.replace("__", ".").replace("_", ".").strip("._ ")
    return candidate or None


def _aggregate_summaries(summaries: Dict[str, ScoreSummary]) -> Dict[str, List[str]]:
    tempos = sorted({tempo for summary in summaries.values() for tempo in summary.tempos if tempo})
    time_signatures = sorted({ts for summary in summaries.values() for ts in summary.time_signatures if ts})
    key_signatures = sorted({ks for summary in summaries.values() for ks in summary.key_signatures if ks})
    return {
        "tempos": tempos,
        "time_signatures": time_signatures,
        "key_signatures": key_signatures,
    }


def _build_index(rows: Iterable[Dict[str, Any]], keys: Iterable[str]) -> Dict[str, List[Dict[str, Any]]]:
    index: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        tokens: List[str] = []
        for key in keys:
            value = row.get(key)
            if isinstance(value, list):
                tokens.extend(_normalize_lookup(v) for v in value)
            else:
                tokens.append(_normalize_lookup(value))
        for token in tokens:
            if not token:
                continue
            index.setdefault(token, []).append(row)
    return index


def _dedupe_refs(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[int] = set()
    unique: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        marker = id(item)
        if marker in seen:
            continue
        seen.add(marker)
        unique.append(item)
    return unique


def _normalize_lookup(value: Any) -> str:
    text = _norm_token(value)
    return text.replace(" ", "").replace("_", "").replace("-", "")


def _coerce_to_iterable(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [_normalize_lookup(v) for v in value if str(v).strip()]
    if isinstance(value, str):
        return [_normalize_lookup(value)] if value.strip() else []
    return []


def _get_case_insensitive(data: Dict[str, Any], key: str) -> Any:
    if not isinstance(data, dict):
        return None
    target = key.lower()
    for k, value in data.items():
        if isinstance(k, str) and k.lower() == target:
            return value
    return None


def _gather_touchpoint_values(chapter: Dict[str, Any], key: str) -> List[Any]:
    tps = _get_case_insensitive(chapter, "Touch-Points")
    if not isinstance(tps, list):
        return []
    values: List[Any] = []
    target = key.lower()
    for item in tps:
        if not isinstance(item, dict):
            continue
        for k, value in item.items():
            if isinstance(k, str) and k.strip().lower() == target:
                values.append(value)
    return values


__all__ = [
    "VoiceSpec",
    "ScoreSummary",
    "VoiceContext",
    "parse_voice_token",
    "build_voice_context",
    "build_music_prompt_context",
    "write_voice_token",
]


def reduced_notes_csv(music: MusicCSV, *, track_filter: Optional[int] = None) -> str:
    """Return a compact CSV view of notes as measure,beat,pitch,duration.

    This helper is intended for prompt context, not for round-tripping. It
    omits track, velocity, ties, and other columns so that large scores can
    be presented to the model in a token-efficient form while still
    preserving harmonic and rhythmic information.
    """

    import csv
    from io import StringIO

    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["measure", "beat", "pitch", "duration"])

    for note in music.notes:
        try:
            track = note.get("track")
            if track_filter is not None and track != track_filter:
                continue
            measure = note.get("measure")
            beat = note.get("beat")
            pitch = note.get("pitch")
            duration = note.get("duration")
            if measure is None or beat is None or pitch is None or duration is None:
                continue
            writer.writerow([measure, beat, pitch, duration])
        except Exception:
            continue

    return output.getvalue().strip()

