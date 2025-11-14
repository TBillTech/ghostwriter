from __future__ import annotations

from pathlib import Path
import zipfile

from ghostwriter.musiccsv import (
    MusicCSV,
    musiccsv_from_text,
    musiccsv_to_text,
    read_musiccsv,
    write_musiccsv,
)


def _sample_musiccsv() -> MusicCSV:
    return MusicCSV(
        metadata={
            "title": "Sample Piece",
            "composer": "Test Composer",
            "tempo": 100,
            "time_signature": "4/4",
            "key_signature": "C",
            "divisions_per_quarter": 480,
            "version": "0.1",
        },
        tracks=[
            {
                "track": 1,
                "label": "Piano RH",
                "part": "Right Hand",
                "instrument": "Acoustic Grand Piano",
                "channel": 1,
                "program": 0,
                "volume": 100,
            },
            {
                "track": 2,
                "label": "Piano LH",
                "part": "Left Hand",
                "instrument": "Acoustic Grand Piano",
                "channel": 2,
                "program": 0,
                "volume": 100,
            },
        ],
        measures=[
            {
                "measure": 1,
                "time_signature": "4/4",
                "key_signature": "C",
                "tempo": 100.0,
                "start_beat": 0.0,
                "pickup": False,
            },
            {
                "measure": 2,
                "time_signature": "4/4",
                "key_signature": "C",
                "tempo": 100.0,
                "start_beat": 4.0,
                "pickup": False,
            },
        ],
        notes=[
            {
                "track": 1,
                "measure": 1,
                "beat": 1.0,
                "pitch": "C4",
                "duration": 1.0,
                "velocity": 90,
                "tie": "none",
                "articulation": "legato",
                "pedal": False,
                "lyric": None,
                "ornament": None,
                "comment": None,
                "grace": True,
                "repeat": "start",
                "tuplet": "3:2",
            },
            {
                "track": 2,
                "measure": 1,
                "beat": 1.0,
                "pitch": "C3",
                "duration": 4.0,
                "velocity": 80,
                "tie": "none",
                "articulation": None,
                "pedal": False,
                "lyric": None,
                "ornament": None,
                "comment": "bass",
                "grace": False,
                "repeat": "end",
                "tuplet": None,
            },
        ],
    )


def test_directory_roundtrip(tmp_path: Path) -> None:
    music = _sample_musiccsv()
    write_musiccsv(tmp_path / "score", music)
    loaded = read_musiccsv(tmp_path / "score")
    assert loaded == music


def test_file_roundtrip(tmp_path: Path) -> None:
    music = _sample_musiccsv()
    file_path = tmp_path / "score.musiccsv"
    write_musiccsv(file_path, music)
    text = file_path.read_text(encoding="utf-8")
    assert "### metadata.json" in text
    assert "### notes.csv" in text
    loaded = read_musiccsv(file_path)
    assert loaded == music


def test_read_legacy_archive(tmp_path: Path) -> None:
    music = _sample_musiccsv()
    legacy_dir = tmp_path / "legacy"
    write_musiccsv(legacy_dir, music)
    archive = tmp_path / "legacy.musiccsv"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in ("metadata.json", "tracks.csv", "measures.csv", "notes.csv"):
            zf.write(legacy_dir / name, arcname=name)
    loaded = read_musiccsv(archive)
    assert loaded == music


def test_musiccsv_text_roundtrip() -> None:
    music = _sample_musiccsv()
    text = musiccsv_to_text(music)
    restored = musiccsv_from_text(text)
    assert restored == music


def test_summary_contains_expected_fields() -> None:
    music = _sample_musiccsv()
    summary = music.summary()
    assert "MusicCSV Summary" in summary
    assert "Title" in summary
    assert "Tracks" in summary
    assert "Notes" in summary


def test_to_dict_isolated_copy() -> None:
    music = _sample_musiccsv()
    data = music.to_dict()
    data["metadata"]["title"] = "Mutated"
    assert music.metadata["title"] == "Sample Piece"


def test_text_requires_sections() -> None:
    try:
        musiccsv_from_text("### tracks.csv\ntrack,label\n")
    except ValueError as exc:
        assert "metadata" in str(exc)
    else:
        raise AssertionError("Expected ValueError when metadata section missing")


def test_text_roundtrip_preserves_additional_columns() -> None:
    music = _sample_musiccsv()
    text = musiccsv_to_text(music)
    assert "grace" in text
    restored = musiccsv_from_text(text)
    assert restored.notes[0]["grace"] is True
    assert restored.notes[0]["tuplet"] == "3:2"

