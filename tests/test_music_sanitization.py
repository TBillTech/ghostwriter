from pathlib import Path
from io import StringIO
import csv

import pytest

from ghostwriter.music.pipeline import _sanitize_block_note_rows


def _load_llm_response_rows(attempt_path: Path):
    text = attempt_path.read_text(encoding="utf-8")
    marker = "measure,beat,pitch,duration,velocity,tie,articulation"
    idx = text.rfind(marker)
    assert idx != -1, "LLM response block not found in attempt file"
    csv_text = text[idx:]
    return list(csv.DictReader(StringIO(csv_text)))


def test_melody_measure_three_respects_core_sequence():
    attempt_path = Path(
        "testdata/SongTest/iterations/CHAPTER_001/pipeline_v1/06_music/notes_major_alto_flute_red_melody_standard.block01_1.txt"
    )
    if not attempt_path.exists():
        pytest.skip("LLM attempt fixture was removed; restore SongTest assets to re-enable this check.")
    rows = _load_llm_response_rows(attempt_path)

    sanitized = _sanitize_block_note_rows(
        rows,
        measure_start=1,
        measure_end=10,
        beats_per_measure=4,
        preserve_input_order=True,
    )

    measure_three_pitches = [row["pitch"] for row in sanitized if row["measure"] == "3" and row["pitch"] != "rest"]

    assert measure_three_pitches[:6] == ["A3", "C4", "E4", "A4", "G4", "E4"], (
        "Melody block ordering drifted from the CORE_MELODY sequence; "
        "see notes_major_alto_flute_red_melody_standard.block01_1.txt"
    )
