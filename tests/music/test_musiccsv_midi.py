from __future__ import annotations

from pathlib import Path

import mido
import pytest

from ghostwriter.musiccsv import MusicCSV, from_midi, to_midi


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


def test_to_midi_roundtrip_preserves_musiccsv(tmp_path: Path) -> None:
    music = _sample_musiccsv()
    midi_path = tmp_path / "roundtrip.mid"
    to_midi(music, midi_path)
    restored = from_midi(midi_path)
    assert restored == music
    assert restored.notes[0]["grace"] is True
    assert restored.notes[0]["tuplet"] == "3:2"


def test_from_midi_without_embedded_metadata(tmp_path: Path) -> None:
    midi_path = tmp_path / "simple.mid"
    midi = mido.MidiFile(ticks_per_beat=480)

    meta_track = mido.MidiTrack()
    meta_track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120), time=0))
    meta_track.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    meta_track.append(mido.MetaMessage("end_of_track", time=0))
    midi.tracks.append(meta_track)

    note_track = mido.MidiTrack()
    note_track.append(mido.MetaMessage("track_name", name="Piano", time=0))
    note_track.append(mido.Message("program_change", program=0, channel=0, time=0))
    note_track.append(mido.Message("note_on", note=60, velocity=90, channel=0, time=0))
    note_track.append(mido.Message("note_off", note=60, velocity=0, channel=0, time=480))
    note_track.append(mido.MetaMessage("end_of_track", time=0))
    midi.tracks.append(note_track)

    midi.save(midi_path)

    music = MusicCSV.from_midi(midi_path)

    assert music.metadata["tempo"] == pytest.approx(120)
    assert music.metadata["time_signature"] == "4/4"
    assert music.metadata["divisions_per_quarter"] == 480
    assert len(music.tracks) == 1
    assert len(music.notes) == 1
    note = music.notes[0]
    assert note["pitch"] == "C4"
    assert note["duration"] == pytest.approx(1.0)
    assert note["beat"] == pytest.approx(1.0)
    assert note["grace"] is False
    assert note["tuplet"] is None


def test_to_midi_respects_track_channels(tmp_path: Path) -> None:
    music = _sample_musiccsv()
    music.tracks[0]["channel"] = 5
    midi_path = tmp_path / "channels.mid"
    music.to_midi(midi_path)

    midi = mido.MidiFile(midi_path)
    channels = {
        msg.channel
        for track in midi.tracks
        for msg in track
        if not getattr(msg, "is_meta", False) and hasattr(msg, "channel")
    }
    assert 4 in channels


def test_inverse_roundtrip_from_existing_midi(tmp_path: Path) -> None:
    midi_path = tmp_path / "baseline.mid"
    midi = mido.MidiFile(ticks_per_beat=480)
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(90), time=0))
    meta.append(mido.MetaMessage("time_signature", numerator=3, denominator=4, time=0))
    meta.append(mido.MetaMessage("end_of_track", time=0))
    midi.tracks.append(meta)

    track = mido.MidiTrack()
    track.append(mido.MetaMessage("track_name", name="Melody", time=0))
    track.append(mido.Message("program_change", program=40, channel=3, time=0))
    track.append(mido.Message("note_on", note=64, velocity=70, channel=3, time=0))
    track.append(mido.Message("note_off", note=64, velocity=0, channel=3, time=360))
    track.append(mido.MetaMessage("end_of_track", time=0))
    midi.tracks.append(track)
    midi.save(midi_path)

    music = MusicCSV.from_midi(midi_path)
    reexport_path = tmp_path / "baseline_reexport.mid"
    music.to_midi(reexport_path)
    round_trip = MusicCSV.from_midi(reexport_path)

    assert round_trip == music


def test_to_midi_sanitizes_unicode_metadata(tmp_path: Path) -> None:
    music = _sample_musiccsv()
    music.metadata["title"] = "Wolf — Red Encounter"
    music.tracks[0]["label"] = "Lead — Flute"

    midi_path = tmp_path / "unicode.mid"
    music.to_midi(midi_path)

    midi = mido.MidiFile(midi_path)
    for track in midi.tracks:
        for msg in track:
            msg_type = getattr(msg, "type", "")
            if msg_type in {"track_name", "instrument_name", "text"}:
                data = getattr(msg, "name", None)
                if data is None and hasattr(msg, "text"):
                    data = msg.text
                if data is not None:
                    data.encode("latin-1")


def test_to_midi_handles_uniform_start_beats(tmp_path: Path) -> None:
    music = MusicCSV(
        metadata={
            "title": "Uniform Starts",
            "composer": "Tester",
            "tempo": 120,
            "time_signature": "4/4",
            "key_signature": "C",
            "divisions_per_quarter": 480,
            "version": "0.1",
        },
        tracks=[
            {
                "track": 1,
                "label": "Lead",
                "instrument": "Acoustic Grand Piano",
                "channel": 1,
                "program": 0,
                "volume": 100,
            }
        ],
        measures=[
            {"measure": 1, "time_signature": "4/4", "key_signature": "C", "tempo": 120.0, "start_beat": 1.0},
            {"measure": 2, "time_signature": "4/4", "key_signature": "C", "tempo": 120.0, "start_beat": 1.0},
            {"measure": 3, "time_signature": "4/4", "key_signature": "C", "tempo": 120.0, "start_beat": 1.0},
        ],
        notes=[
            {"track": 1, "measure": 1, "beat": 1.0, "pitch": "C4", "duration": 1.0, "velocity": 90},
            {"track": 1, "measure": 3, "beat": 1.0, "pitch": "E4", "duration": 1.0, "velocity": 90},
        ],
    )

    midi_path = tmp_path / "uniform.mid"
    music.to_midi(midi_path)

    midi = mido.MidiFile(midi_path)
    ticks_per_measure = 4 * midi.ticks_per_beat
    on_times = []
    for track in midi.tracks:
        elapsed = 0
        for msg in track:
            elapsed += msg.time
            if not getattr(msg, "is_meta", False) and msg.type == "note_on" and msg.velocity > 0:
                on_times.append(elapsed)

    assert on_times, "Expected at least one note_on event"
    assert max(on_times) >= ticks_per_measure * 2
