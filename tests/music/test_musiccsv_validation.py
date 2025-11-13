from __future__ import annotations

import pytest

from ghostwriter.musiccsv import (
    MusicCSVValidationError,
    resolve_derived_fields,
    validate_musiccsv,
)
from tests.music.test_musiccsv_io import _sample_musiccsv


def test_validate_musiccsv_accepts_well_formed_payload() -> None:
    music = _sample_musiccsv()
    validate_musiccsv(music)


def test_validate_musiccsv_reports_schema_issues() -> None:
    music = _sample_musiccsv()
    music.metadata.pop("tempo")

    with pytest.raises(MusicCSVValidationError) as exc:
        validate_musiccsv(music)

    assert any("metadata.tempo" in error for error in exc.value.errors)


def test_validate_musiccsv_rejects_bad_pitch() -> None:
    music = _sample_musiccsv()
    music.notes[0]["pitch"] = "H#9"

    with pytest.raises(MusicCSVValidationError) as exc:
        validate_musiccsv(music)

    assert any("pitch" in error for error in exc.value.errors)


def test_validate_musiccsv_rejects_invalid_tuplet() -> None:
    music = _sample_musiccsv()
    music.notes[0]["tuplet"] = "3"  # Missing ratio

    with pytest.raises(MusicCSVValidationError) as exc:
        validate_musiccsv(music)

    assert any("tuplet" in error for error in exc.value.errors)


def test_resolve_derived_fields_adds_ticks_without_mutation() -> None:
    music = _sample_musiccsv()
    derived = resolve_derived_fields(music)

    for key in ("absolute_beat", "absolute_tick", "duration_ticks", "midi_note"):
        assert key in derived.notes[0]
        assert key not in music.notes[0]

    assert derived.notes[0]["absolute_tick"] == 0
    assert derived.notes[0]["duration_ticks"] == 480
    assert derived.notes[0]["midi_note"] == 60
    assert derived.notes[0]["tuplet_ratio"] == pytest.approx(1.5)

    assert derived.measures[0]["start_tick"] == 0
    assert "start_tick" not in music.measures[0]
