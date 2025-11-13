from __future__ import annotations

from pathlib import Path

import pytest
import yaml

mido = pytest.importorskip("mido")

from ghostwriter.context import RunContext
from ghostwriter.templates import iter_dir_for
from ghostwriter.music.context import (
    build_music_prompt_context,
    build_voice_context,
    parse_voice_token,
    write_voice_token,
)
from ghostwriter.musiccsv import MusicCSV, write_musiccsv


def _write_score_assets(
    directory: Path,
    *,
    tempo_bpm: int,
    numerator: int,
    denominator: int,
    midi_pitch: int,
) -> None:
    pitch_name = _midi_to_pitch(midi_pitch)
    music = MusicCSV(
        metadata={
            "title": "Context Score",
            "composer": "Tester",
            "tempo": tempo_bpm,
            "time_signature": f"{numerator}/{denominator}",
            "key_signature": "C",
            "divisions_per_quarter": 480,
            "version": "0.1",
        },
        tracks=[
            {
                "track": 1,
                "label": "Track",
                "part": "Part",
                "instrument": "Piano",
                "channel": 1,
                "program": 0,
                "volume": 100,
            }
        ],
        measures=[
            {
                "measure": 1,
                "time_signature": f"{numerator}/{denominator}",
                "key_signature": "C",
                "tempo": tempo_bpm,
                "start_beat": 0.0,
                "pickup": False,
            }
        ],
        notes=[
            {
                "track": 1,
                "measure": 1,
                "beat": 1.0,
                "pitch": pitch_name,
                "duration": 1.0,
                "velocity": 80,
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
        ],
    )

    directory.mkdir(parents=True, exist_ok=True)
    score_path = directory / "score.musiccsv"
    import_path = directory / "import.musiccsv"
    write_musiccsv(score_path, music)
    write_musiccsv(import_path, music)
    midi_path = directory / "monitor.mid"
    music.to_midi(midi_path)


def _midi_to_pitch(midi_value: int) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    octave = midi_value // 12 - 1
    name = names[midi_value % 12]
    return f"{name}{octave}"


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
        "major.bass.soundscape.forest_path",
    ]
    data["music"] = "Slow burn in D minor; start around 90 BPM."
    chapter_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    pipeline_dir = iter_dir_for("CHAPTER_001") / "pipeline_v1" / "01_narration"
    pipeline_dir.mkdir(parents=True, exist_ok=True)

    voice_defs = [
        ("major.alto.flute.red.melody", {"tempo": 90, "numerator": 3, "denominator": 4, "note": 60}),
        ("minor.bass.cello.wolf", {"tempo": 110, "numerator": 4, "denominator": 4, "note": 50}),
        ("major.bass.soundscape.forest_path", {"tempo": 72, "numerator": 4, "denominator": 4, "note": 48}),
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
    assert len(voice_context.voices) == 3

    red_spec = next(spec for spec in voice_context.voices if spec.idea.lower() == "red")
    wolf_spec = next(spec for spec in voice_context.voices if spec.idea.lower() == "wolf")
    forest_spec = next(spec for spec in voice_context.voices if spec.idea.lower() == "forest_path")

    assert any(cid.lower() == "red" for cid in red_spec.character_ids)
    assert any(cid.lower() == "wolf" for cid in wolf_spec.character_ids)
    assert any(name == "Forest Path" for name in forest_spec.factoid_names)

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

    forest_payload = next(item for item in payload["voices"] if item["token"] == forest_spec.token)
    assert "Forest Path" in forest_payload["factoids"]

    assert "missing_assets" not in payload
    assert any(entry["voice_token"] == red_spec.token for entry in payload["score_summaries"])
