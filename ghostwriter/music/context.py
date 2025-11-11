"""Voice metadata parsing and context assembly for music pipelines.

Slice 2 of the music feature introduces utilities that extract voice tokens
from chapter YAML, map those voices onto known story entities, and collect
basic tempo/key summaries from any sanitized scores already present in the
iterations directory. The resulting structures feed later prompt builders
and gate orchestration.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import logging
import warnings

try:  # pragma: no cover - exercised via pytest.importorskip in tests
    from music21 import converter, key as m21_key
except ImportError as exc:  # pragma: no cover - surfaced to caller
    raise ImportError(
        "music21 is required for the music context helpers. Install optional music"
        " dependencies via `pip install -r requirements.txt`."
    ) from exc

from ..context import RunContext
from ..templates import iter_dir_for
from ..utils import _norm_token

warnings.filterwarnings("ignore", module="music21")

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
    """Lightweight metadata derived from a sanitized MusicXML score."""

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
            "missing_assets": list(self.missing_assets),
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


def build_voice_context(ctx: RunContext, *, pipeline_version: Optional[int] = None) -> VoiceContext:
    """Construct a :class:`VoiceContext` for the provided run context."""

    voice_entries = _extract_voice_entries(ctx.chapter)
    voice_specs: List[VoiceSpec] = [parse_voice_token(token, metadata=meta) for token, meta in voice_entries]
    _attach_entity_links(voice_specs, ctx)

    version = pipeline_version if pipeline_version is not None else ctx.version
    score_summaries = _collect_score_summaries(ctx, voice_specs, version)

    aggregates = _aggregate_summaries(score_summaries)
    missing_assets = [spec.token for spec in voice_specs if spec.token not in score_summaries]

    directive = _extract_music_directive(ctx.chapter)

    return VoiceContext(
        directive=directive,
        voices=voice_specs,
        score_summaries=score_summaries,
        aggregate_tempos=aggregates["tempos"],
        aggregate_time_signatures=aggregates["time_signatures"],
        aggregate_key_signatures=aggregates["key_signatures"],
        missing_assets=missing_assets,
    )


def build_music_prompt_context(ctx: RunContext, *, pipeline_version: Optional[int] = None) -> Dict[str, Any]:
    """Return a serialisable payload describing chapter music context."""

    voice_context = build_voice_context(ctx, pipeline_version=pipeline_version)
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


def _extract_music_directive(chapter: Dict[str, Any]) -> str:
    raw = _get_case_insensitive(chapter, "music")
    if isinstance(raw, str):
        return raw.strip()
    return str(raw).strip() if raw is not None else ""


def _extract_voice_entries(chapter: Dict[str, Any]) -> List[tuple[str, Dict[str, Any]]]:
    raw = _get_case_insensitive(chapter, "voices")
    if raw is None:
        return []
    entries: List[tuple[str, Dict[str, Any]]] = []
    for token, meta in _normalize_voice_entry(raw):
        entries.append((token, meta))
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
        score_path = import_dir / "score.musicxml"
        if not score_path.exists():
            score_path = import_dir / "import.musicxml"
        if not score_path.exists():
            continue
        try:
            summary = _summarize_musicxml(score_path, spec.token)
        except Exception as exc:  # pragma: no cover - defensive branch
            logger.debug("Failed to summarize %s: %s", score_path, exc)
            continue
        summaries[spec.token] = summary
    return summaries


def _summarize_musicxml(path: Path, voice_token: str) -> ScoreSummary:
    stream = converter.parse(str(path))

    tempos = set()
    try:
        for _offset, _end, mark in stream.metronomeMarkBoundaries():
            number = getattr(mark, "number", None)
            if number is not None:
                tempos.add(f"{float(number):.2f}")
            else:
                tempos.add(str(mark))
    except Exception:
        pass
    if not tempos:
        tempos.add("Unknown")

    time_signatures = {
        ts.ratioString for ts in stream.recurse().getElementsByClass("TimeSignature")
    }
    if not time_signatures:
        time_signatures.add("Unknown")

    key_signatures = set()
    try:
        detected = stream.analyze("key")
        if detected:
            key_signatures.add(str(detected))
    except Exception:
        pass
    for key_obj in stream.recurse().getElementsByClass(m21_key.Key):
        key_signatures.add(str(key_obj))
    for key_sig in stream.recurse().getElementsByClass(m21_key.KeySignature):
        try:
            as_key = key_sig.asKey()
            key_signatures.add(str(as_key))
        except Exception:
            key_signatures.add(str(key_sig))
    if not key_signatures:
        key_signatures.add("Unknown")

    measure_numbers: List[int] = []
    for measure in stream.recurse().getElementsByClass("Measure"):
        number = getattr(measure, "measureNumber", None) or getattr(measure, "number", None)
        if number is None:
            continue
        try:
            measure_numbers.append(int(number))
        except Exception:
            continue
    measure_count = max(measure_numbers) if measure_numbers else 0

    try:
        duration = float(getattr(stream.duration, "quarterLength", 0.0) or 0.0)
    except Exception:
        duration = 0.0

    return ScoreSummary(
        voice_token=voice_token,
        score_path=Path(path),
        tempos=sorted(tempos),
        time_signatures=sorted(time_signatures),
        key_signatures=sorted(key_signatures),
        measure_count=measure_count,
        duration_quarter_length=duration,
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
    return text.replace(" ", "")


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


__all__ = [
    "VoiceSpec",
    "ScoreSummary",
    "VoiceContext",
    "parse_voice_token",
    "build_voice_context",
    "build_music_prompt_context",
    "write_voice_token",
]
