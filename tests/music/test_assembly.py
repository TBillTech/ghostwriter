from __future__ import annotations

import json
from pathlib import Path

from ghostwriter.music.pipeline import assemble_first_pass_variant
from ghostwriter.musiccsv import read_musiccsv


def _write_csv(path: Path, header: list[str], rows: list[list]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(",".join(header) + "\n")
        for row in rows:
            handle.write(",".join(str(value) for value in row) + "\n")


def test_assemble_first_pass_variant_assigns_tracks(tmp_path: Path):
    tp_dir = tmp_path / "music"
    tp_dir.mkdir(parents=True, exist_ok=True)

    (tp_dir / "metadata.json").write_text(
        json.dumps({
            "title": "Test Song",
            "tempo": 120,
            "time_signature": "4/4",
            "key_signature": "C",
            "divisions_per_quarter": 480,
        }),
        encoding="utf-8",
    )

    tracks_header = ["track", "label", "part", "instrument", "channel", "program", "volume"]
    tracks_rows = [
        [1, "Wolf Harmony", "Harmony", "Horn", 1, 61, 100],
        [2, "Red Melody", "Melody", "Flute", 1, 74, 100],
    ]
    _write_csv(tp_dir / "tracks.csv", tracks_header, tracks_rows)

    measures_header = ["measure", "time_signature", "key_signature", "tempo", "start_beat", "pickup"]
    measures_rows = [[1, "4/4", "C", 120, 1, False]]
    _write_csv(tp_dir / "measures_standard.csv", measures_header, measures_rows)

    (tp_dir / "music_touch_point.json").write_text(
        json.dumps({
            "voice_tokens": [
                "major.tenor.base.wolf.harmony",
                "major.alto.flute.red.melody",
            ]
        }),
        encoding="utf-8",
    )

    note_header = ["measure", "beat", "pitch", "duration", "velocity", "tie", "articulation"]
    wolf_rows = [[1, 1.0, "C4", 1.0, 80, "", "legato"]]
    red_rows = [[1, 1.0, "E4", 1.0, 70, "", "legato"]]
    _write_csv(tp_dir / "notes_major.tenor.base.wolf.harmony_standard.csv", note_header, wolf_rows)
    _write_csv(tp_dir / "notes_major.alto.flute.red.melody_standard.csv", note_header, red_rows)

    score_path = assemble_first_pass_variant(
        tp_dir=tp_dir,
        title="Test Song",
        variant="standard",
    )

    assert score_path is not None
    music = read_musiccsv(score_path)
    assert len(music.notes) == 2
    tracks = {note.get("pitch"): note.get("track") for note in music.notes}
    assert tracks["C4"] == 1
    assert tracks["E4"] == 2
