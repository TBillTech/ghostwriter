from __future__ import annotations

from pathlib import Path

import pytest
import yaml

music21 = pytest.importorskip("music21")
mido = pytest.importorskip("mido")

from ghostwriter.context import RunContext
from ghostwriter.templates import iter_dir_for
from ghostwriter.music.context import (
    build_music_prompt_context,
    build_voice_context,
    parse_voice_token,
    write_voice_token,
)


def _write_score_assets(
    directory: Path,
    *,
    tempo_bpm: int,
    numerator: int,
    denominator: int,
    midi_pitch: int,
) -> None:
    import music21

    score = music21.stream.Score()  # type: ignore[attr-defined]
    part = music21.stream.Part()  # type: ignore[attr-defined]
    part.append(music21.meter.TimeSignature(f"{numerator}/{denominator}"))  # type: ignore[attr-defined]
    part.append(music21.tempo.MetronomeMark(number=tempo_bpm))  # type: ignore[attr-defined]
    part.append(music21.instrument.Instrument())  # type: ignore[attr-defined]
    tone = music21.note.Note(midi_pitch)
    tone.quarterLength = 1
    part.append(tone)
    score.append(part)

    score.write("musicxml", fp=str(directory / "score.musicxml"))
    score.write("musicxml", fp=str(directory / "import.musicxml"))
    score.write("midi", fp=str(directory / "monitor.mid"))


def test_parse_voice_token_handles_roles() -> None:
    spec = parse_voice_token("major.alto.flute.red.melody")
    assert spec.chord == "major"
    assert spec.role == "melody"
    assert not spec.issues

    spec_meta = parse_voice_token("minor.tenor.viola.wolf", metadata={"role": "harmony"})
    assert spec_meta.role == "harmony"

    bad = parse_voice_token("invalid-token")
    assert any("must contain" in issue for issue in bad.issues)


def test_build_voice_context_collects_metadata(use_lr_book_env, lr_book_dir: Path) -> None:
    chapter_path = lr_book_dir / "chapters/CHAPTER_001.yaml"
    data = yaml.safe_load(chapter_path.read_text(encoding="utf-8"))
    data["voices"] = [
        "major.alto.flute.red.melody",
        {
            "token": "minor.bass.cello.wolf",
            "role": "support",
        },
    ]
    data["music"] = "Slow burn in D minor; start around 90 BPM."
    chapter_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    pipeline_dir = iter_dir_for("CHAPTER_001") / "pipeline_v1" / "01_narration"
    pipeline_dir.mkdir(parents=True, exist_ok=True)

    voice_defs = [
        ("major.alto.flute.red.melody", {"tempo": 90, "numerator": 3, "denominator": 4, "note": 60}),
        ("minor.bass.cello.wolf", {"tempo": 110, "numerator": 4, "denominator": 4, "note": 50}),
    ]

    for idx, (token, params) in enumerate(voice_defs):
        voice_dir = pipeline_dir / f"{idx:02d}_track_{token.replace('.', '_')}_import"
        voice_dir.mkdir(parents=True, exist_ok=True)
        _write_score_assets(
            voice_dir,
            tempo_bpm=params["tempo"],
            numerator=params["numerator"],
            denominator=params["denominator"],
            midi_pitch=params["note"],
        )
        write_voice_token(voice_dir, token)

    ctx = RunContext.from_paths(chapter_path=str(chapter_path), version=1)
    voice_context = build_voice_context(ctx, pipeline_version=1)

    assert voice_context.directive.startswith("Slow burn")
    assert len(voice_context.voices) == 2

    red_spec = next(spec for spec in voice_context.voices if spec.idea.lower() == "red")
    wolf_spec = next(spec for spec in voice_context.voices if spec.idea.lower() == "wolf")

    assert any(cid.lower() == "red" for cid in red_spec.character_ids)
    assert any(cid.lower() == "wolf" for cid in wolf_spec.character_ids)

    assert not voice_context.missing_assets

    red_summary = voice_context.score_summaries[red_spec.token]
    assert "3/4" in red_summary.time_signatures
    assert red_summary.tempos

    payload = build_music_prompt_context(ctx, pipeline_version=1)
    assert payload["directive"] == voice_context.directive
    assert "90" in payload["tempo_summary"]
    assert "3/4" in payload["time_signature_summary"]

    payload_tokens = [item["token"] for item in payload["voices"]]
    assert red_spec.token in payload_tokens

    red_payload = next(item for item in payload["voices"] if item["token"] == red_spec.token)
    assert any(cid.lower() == "red" for cid in red_payload["characters"])

    assert payload["missing_assets"] == []
    assert any(entry["voice_token"] == red_spec.token for entry in payload["score_summaries"])
