from __future__ import annotations

from pathlib import Path

import mido

from ghostwriter.musiccsv import MusicCSV


def _build_simple_midi(path: Path) -> None:
    midi = mido.MidiFile(ticks_per_beat=480)
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(100), time=0))
    meta.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    meta.append(mido.MetaMessage("end_of_track", time=0))
    midi.tracks.append(meta)

    track = mido.MidiTrack()
    track.append(mido.MetaMessage("track_name", name="Lead", time=0))
    track.append(mido.Message("program_change", program=1, channel=0, time=0))
    track.append(mido.Message("note_on", note=60, velocity=70, channel=0, time=0))
    track.append(mido.Message("note_off", note=60, velocity=0, channel=0, time=480))
    track.append(mido.MetaMessage("end_of_track", time=0))
    midi.tracks.append(track)

    midi.save(path)


def test_musiccsv_api_example(tmp_path: Path) -> None:
    midi_path = tmp_path / "input.mid"
    _build_simple_midi(midi_path)

    score = MusicCSV.from_midi(midi_path)
    out_dir = tmp_path / "score.musiccsv"
    score.write(out_dir)

    reopened = MusicCSV.read(out_dir)
    reexport_mid = tmp_path / "reexport.mid"
    reopened.to_midi(reexport_mid)

    assert reexport_mid.exists()
    re_round = MusicCSV.from_midi(reexport_mid)
    assert re_round.metadata["title"] == score.metadata["title"]
