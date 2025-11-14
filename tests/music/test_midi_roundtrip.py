from __future__ import annotations

from pathlib import Path

import pytest

from ghostwriter.musiccsv import from_midi, read_musiccsv


def test_musiccsv_midi_roundtrip(tmp_path: Path) -> None:
	"""Round-trip a MusicCSV score through MIDI and back."""

	pytest.importorskip("mido")

	source_path = Path(__file__).resolve().parent.parent / "data" / "touch_point_score_roundtrip.musiccsv"
	original = read_musiccsv(source_path)

	midi_path = tmp_path / "roundtrip.mid"
	original.to_midi(midi_path)

	rebuilt = from_midi(midi_path)

	assert rebuilt.to_dict() == original.to_dict()
