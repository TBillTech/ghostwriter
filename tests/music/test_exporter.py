from __future__ import annotations

import json
from pathlib import Path
import zipfile

import pytest
import yaml

music21 = pytest.importorskip("music21")  # noqa: F401
mido = pytest.importorskip("mido")  # noqa: F401

from ghostwriter.context import RunContext
from ghostwriter.templates import iter_dir_for
from ghostwriter.music.context import build_voice_context, write_voice_token
from ghostwriter.music.exporter import finalize_music_exports


def _write_simple_score(path: Path, *, pitch: int) -> None:
    score = music21.stream.Score()  # type: ignore[attr-defined]
    part = music21.stream.Part()  # type: ignore[attr-defined]
    part.append(music21.meter.TimeSignature("4/4"))  # type: ignore[attr-defined]
    part.append(music21.tempo.MetronomeMark(number=90))  # type: ignore[attr-defined]
    part.append(music21.instrument.Instrument())  # type: ignore[attr-defined]
    note = music21.note.Note(pitch)
    note.quarterLength = 1
    part.append(note)
    score.append(part)
    path.parent.mkdir(parents=True, exist_ok=True)
    score.write("musicxml", fp=str(path))


def _write_voice_assets(base_dir: Path, token: str, *, pitch: int) -> Path:
    import_dir = base_dir / f"00_track_{token.replace('.', '_')}_import"
    import_dir.mkdir(parents=True, exist_ok=True)
    score_path = import_dir / "score.musicxml"
    _write_simple_score(score_path, pitch=pitch)
    (import_dir / "import.musicxml").write_text(score_path.read_text(encoding="utf-8"), encoding="utf-8")
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

    _write_simple_score(tp1_dir / "touch_point_score.musicxml", pitch=60)
    _write_simple_score(tp2_dir / "touch_point_first_score.musicxml", pitch=65)

    _write_voice_assets(tp1_dir, "major.alto.flute.red.melody", pitch=60)
    _write_voice_assets(tp1_dir, "minor.tenor.violin.wolf", pitch=67)

    ctx = RunContext.from_paths(chapter_path=str(chapter_path), version=1)
    voice_context = build_voice_context(ctx, pipeline_version=1)

    result = finalize_music_exports("CHAPTER_001", 1, voice_context=voice_context)
    assert result["written"] is True

    chapter_dir = iter_dir_for("CHAPTER_001")
    score_path = chapter_dir / "score_v1.musicxml"
    assert score_path.exists()

    score_dir = chapter_dir / "score"
    final_musicxml = score_dir / "final.musicxml"
    final_midi = score_dir / "final.mid"
    manifest_path = score_dir / "manifest.json"
    bundle_path = score_dir / "score_bundle_v1.zip"

    assert final_musicxml.exists()
    assert final_midi.exists()
    assert manifest_path.exists()
    assert bundle_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["chapter_id"] == "CHAPTER_001"
    assert manifest["version"] == 1
    assert len(manifest["touch_points"]) == 2
    dialog_entry = next(entry for entry in manifest["touch_points"] if entry["type"] == "dialog")
    assert dialog_entry["finalized"] is False

    midi_assets = [score_dir / path.split("/")[-1] for path in manifest["outputs"]["per_voice_midis"]]
    for midi_path in midi_assets:
        assert midi_path.exists()

    with zipfile.ZipFile(bundle_path) as zf:
        names = set(zf.namelist())
        assert "score_v1.musicxml" in names
        assert "score/final.musicxml" in names
    assert "score/manifest.json" in names