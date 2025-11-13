from __future__ import annotations

import json
from pathlib import Path
import zipfile

import pytest
import yaml

mido = pytest.importorskip("mido")  # noqa: F401

from ghostwriter.context import RunContext
from ghostwriter.templates import iter_dir_for
from ghostwriter.music.context import build_voice_context, write_voice_token
from ghostwriter.music.exporter import finalize_music_exports
from ghostwriter.musiccsv import MusicCSV, write_musiccsv


_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _midi_to_pitch(midi: int) -> str:
    octave = midi // 12 - 1
    name = _NOTE_NAMES[midi % 12]
    return f"{name}{octave}"


def _make_musiccsv(pitch: int) -> MusicCSV:
    pitch_name = _midi_to_pitch(pitch)
    return MusicCSV(
        metadata={
            "title": "Test Score",
            "composer": "Tester",
            "tempo": 90,
            "time_signature": "4/4",
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
                "time_signature": "4/4",
                "key_signature": "C",
                "tempo": 90,
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


def _write_simple_score(path: Path, *, pitch: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_musiccsv(path, _make_musiccsv(pitch))


def _write_voice_assets(base_dir: Path, token: str, *, pitch: int) -> Path:
    import_dir = base_dir / f"00_track_{token.replace('.', '_')}_import"
    import_dir.mkdir(parents=True, exist_ok=True)
    score_path = import_dir / "score.musiccsv"
    _write_simple_score(score_path, pitch=pitch)
    import_path = import_dir / "import.musiccsv"
    write_musiccsv(import_path, _make_musiccsv(pitch))
    (import_dir / "monitor.mid").write_bytes(b"")
    write_voice_token(import_dir, token)
    return score_path


def test_finalize_music_exports_creates_bundle(use_lr_book_env, lr_book_dir: Path) -> None:
    chapter_path = lr_book_dir / "chapters/CHAPTER_001.yaml"
    data = yaml.safe_load(chapter_path.read_text(encoding="utf-8"))
    data["voices"] = [
        "major.alto.flute.red.melody",
        "minor.tenor.violin.wolf",
    ]
    data["music"] = "Slow burn underscore in D minor"
    chapter_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    pipeline_dir = iter_dir_for("CHAPTER_001") / "pipeline_v1"
    tp1_dir = pipeline_dir / "01_narration"
    tp2_dir = pipeline_dir / "02_dialog"
    tp1_dir.mkdir(parents=True, exist_ok=True)
    tp2_dir.mkdir(parents=True, exist_ok=True)

    _write_simple_score(tp1_dir / "touch_point_score.musiccsv", pitch=60)
    _write_simple_score(tp2_dir / "touch_point_first_score.musiccsv", pitch=65)

    _write_voice_assets(tp1_dir, "major.alto.flute.red.melody", pitch=60)
    _write_voice_assets(tp1_dir, "minor.tenor.violin.wolf", pitch=67)

    ctx = RunContext.from_paths(chapter_path=str(chapter_path), version=1)
    voice_context = build_voice_context(ctx, pipeline_version=1)

    result = finalize_music_exports("CHAPTER_001", 1, voice_context=voice_context)
    assert result["written"] is True

    chapter_dir = iter_dir_for("CHAPTER_001")
    score_path = chapter_dir / "score_v1.musiccsv"
    assert score_path.exists()

    score_dir = chapter_dir / "score"
    final_musiccsv = score_dir / "final.musiccsv"
    final_midi = score_dir / "final.mid"
    manifest_path = score_dir / "manifest.json"
    bundle_path = score_dir / "score_bundle_v1.zip"

    assert final_musiccsv.exists()
    assert final_midi.exists()
    assert manifest_path.exists()
    assert bundle_path.exists()

    assert result["final_musiccsv"] == final_musiccsv

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["chapter_id"] == "CHAPTER_001"
    assert manifest["version"] == 1
    assert len(manifest["touch_points"]) == 2
    dialog_entry = next(entry for entry in manifest["touch_points"] if entry["type"] == "dialog")
    assert dialog_entry["finalized"] is False

    outputs = manifest["outputs"]
    assert outputs["score_musiccsv"] == "score_v1.musiccsv"
    assert outputs["final_musiccsv"] == "score/final.musiccsv"
    assert outputs["final_midi"] == "score/final.mid"

    midi_assets = [score_dir / path.split("/")[-1] for path in outputs["per_voice_midis"]]
    for midi_path in midi_assets:
        assert midi_path.exists()

    with zipfile.ZipFile(bundle_path) as zf:
        names = set(zf.namelist())
        assert "score_v1.musiccsv" in names
        assert "score/final.musiccsv" in names
    assert "score/manifest.json" in names