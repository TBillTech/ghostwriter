from __future__ import annotations

from pathlib import Path

import pytest

music21 = pytest.importorskip("music21")
mido = pytest.importorskip("mido")

from ghostwriter.music import generate_import_musicxml, process_import_directory
from ghostwriter.music.sanitizer import ensure_sanitized


def _make_test_midi(path: Path) -> None:
    from mido import MetaMessage, Message, MidiFile, MidiTrack, bpm2tempo

    midi = MidiFile()
    track = MidiTrack()
    midi.tracks.append(track)

    track.append(MetaMessage("time_signature", numerator=3, denominator=4, time=0))
    track.append(MetaMessage("set_tempo", tempo=bpm2tempo(90), time=0))
    track.append(Message("program_change", program=12, time=0))
    track.append(Message("note_on", note=60, velocity=64, time=0))
    track.append(Message("note_off", note=60, velocity=32, time=480))

    midi.save(path)


def test_process_import_directory_creates_artifacts(tmp_path: Path) -> None:
    import_dir = tmp_path / "00_track_voice_import"
    import_dir.mkdir()

    raw_midi = import_dir / "input.mid"
    _make_test_midi(raw_midi)

    artifacts = process_import_directory(import_dir)
    assert artifacts is not None

    import_path = artifacts.import_path
    assert import_path.exists()

    content = import_path.read_text(encoding="utf-8")
    assert "Tempo(s):" in content
    assert "Time Signature(s):" in content
    assert "Instrument(s):" in content

    score_path = import_dir / "score.musicxml"
    monitor_path = import_dir / "monitor.mid"
    assert score_path.exists(), "Sanitizer should emit score.musicxml"
    assert monitor_path.exists(), "Sanitizer should emit monitor.mid"

    # File should not be empty and parseable by music21.
    stream = music21.converter.parse(str(score_path))
    assert len(stream.parts) > 0


def test_generate_import_musicxml_is_idempotent(tmp_path: Path) -> None:
    import_dir = tmp_path / "00_track_voice_import"
    import_dir.mkdir()

    raw_midi = import_dir / "input.mid"
    _make_test_midi(raw_midi)

    first_path = generate_import_musicxml(import_dir)
    second_path = generate_import_musicxml(import_dir)

    assert first_path == second_path
    assert first_path is not None and Path(first_path).exists()

    # Remove monitor and ensure sanitizer regenerates it.
    monitor = import_dir / "monitor.mid"
    monitor.unlink()
    ensure_sanitized(import_dir)
    assert monitor.exists()
