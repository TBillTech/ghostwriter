from __future__ import annotations

import json
import csv
from io import StringIO
from pathlib import Path
from typing import Dict, Any, List

import pytest

from ghostwriter.music.pipeline import (
    run_melody_emotion_step,
    run_emotion_chord_step,
    ensure_core_melody_csv,
)
from ghostwriter.musiccsv import MusicCSV, write_musiccsv


@pytest.fixture()
def prompt_payload() -> Dict[str, Any]:
    return {
        "touch_point_title": "Into the Clearing",
        "touch_point_description": "Lead flute theme",
        "touch_point_prior_paragraph": "The wolves advance.",
        "voices": [
            {
                "token": "major.alto.flute.red.melody",
                "idea": "heroic",
                "role": "melody",
            },
            {
                "token": "major.tenor.bass.grey.harmony",
                "idea": "counterline",
                "role": "support",
            },
        ],
    }


def _stub_env_for_prompt(prompt_name: str, env_key: str, **_: Any) -> tuple[str, float, int]:
    return ("test-model", 0.1, 256)


@pytest.fixture()
def fake_llm(monkeypatch) -> List[str]:
    calls: List[str] = []

    def _fake_llm(prompt: str, **_: Any) -> str:
        calls.append(prompt)
        return (
            "CORE_MELODY.csv\n"
            "measure,beat,pitch,duration\n"
            "1,1,A3,0.25\n"
            "1,1.25,C4,0.125\n"
            "1,1.375,rest,0.25\n"
        )

    monkeypatch.setattr("ghostwriter.music.pipeline.llm_complete", _fake_llm)
    monkeypatch.setattr("ghostwriter.music.pipeline.env_for_prompt", _stub_env_for_prompt)
    monkeypatch.setattr("ghostwriter.music.pipeline.process_import_directory", lambda *_: None)
    return calls


def _setup_import_dir(tp_dir: Path) -> Path:
    import_dir = tp_dir / "01_track_major_alto_flute_red_melody_import"
    import_dir.mkdir()
    (import_dir / "voice.token").write_text("major.alto.flute.red.melody", encoding="utf-8")

    music = MusicCSV(
        metadata={
            "title": "Test",
            "tempo": 90,
            "time_signature": "4/4",
            "key_signature": "C",
            "divisions_per_quarter": 480,
            "version": "0.1",
        },
        tracks=[{"track": 1, "label": "melody", "instrument": "flute"}],
        measures=[],
        notes=[
            {"track": 1, "measure": 1, "beat": 1, "pitch": "A3", "duration": 0.25, "velocity": 80},
            {"track": 1, "measure": 1, "beat": 1.25, "pitch": "C4", "duration": 0.25, "velocity": 120},
        ],
    )
    write_musiccsv(import_dir / "score.musiccsv", music)
    return import_dir


def test_melody_import_flow_short_circuits(tmp_path: Path, prompt_payload: Dict[str, Any], fake_llm: List[str]) -> None:
    tp_dir = tmp_path / "tp01"
    tp_dir.mkdir()
    _setup_import_dir(tp_dir)
    (tp_dir / "metadata.json").write_text(json.dumps({"time_signature": "4/4"}), encoding="utf-8")

    ran = run_melody_emotion_step(
        tp_dir=tp_dir,
        tp_index=1,
        tp_type="music",
        prompt_payload=prompt_payload,
    )
    assert ran is True
    assert len(fake_llm) == 1

    core_path = tp_dir / "CORE_MELODY.csv"
    assert core_path.exists()
    core_text = core_path.read_text(encoding="utf-8")
    lines = [line for line in core_text.strip().splitlines() if line]
    assert lines[0] == "measure,beat,duration,semi_tones,transition,relative_velocity"
    assert lines[1].endswith(",0.8")
    assert lines[2].endswith(",1.2")
    assert lines[3].endswith(",1.2")

    reader = csv.DictReader(StringIO(core_text))
    rows = list(reader)
    assert rows[0]["beat"] == "1"
    assert rows[1]["beat"] == "1.25"
    assert rows[0]["duration"] == "0.25"
    assert rows[1]["duration"] == "0.125"
    assert rows[0]["transition"] == "(3)"
    assert rows[1]["transition"] == "(0)"
    assert rows[2]["transition"] == "(0)"
    assert not (tp_dir / "PRO.csv").exists()

    state_path = tp_dir / "core_melody_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state.get("import_core_melody") is True
    assert state.get("import_voice_token") == "major.alto.flute.red.melody"
    assert state.get("core_start_pitch") == "A3"

    # Second run reuses cached CORE and skips LLM work.
    ran_again = run_melody_emotion_step(
        tp_dir=tp_dir,
        tp_index=1,
        tp_type="music",
        prompt_payload=prompt_payload,
    )
    assert ran_again is False
    assert len(fake_llm) == 1

    # Emotion chord and core melody builder no-op under import mode.
    assert run_emotion_chord_step(tp_dir=tp_dir, tp_index=1, tp_type="music", prompt_payload=prompt_payload) is False
    assert ensure_core_melody_csv(tp_dir=tp_dir) is False

    # Mutating the score hash should trigger a rebuild.
    score_path = tp_dir / "01_track_major_alto_flute_red_melody_import" / "score.musiccsv"
    updated_music = MusicCSV(
        metadata={
            "title": "Test",
            "tempo": 92,
            "time_signature": "4/4",
            "key_signature": "C",
            "divisions_per_quarter": 480,
            "version": "0.1",
        },
        tracks=[{"track": 1, "label": "melody", "instrument": "flute"}],
        measures=[],
        notes=[
            {"track": 1, "measure": 1, "beat": 1, "pitch": "A3", "duration": 0.25, "velocity": 70},
            {"track": 1, "measure": 1, "beat": 1.25, "pitch": "C4", "duration": 0.25, "velocity": 130},
        ],
    )
    write_musiccsv(score_path, updated_music)

    reran = run_melody_emotion_step(
        tp_dir=tp_dir,
        tp_index=1,
        tp_type="music",
        prompt_payload=prompt_payload,
    )
    assert reran is True
    assert len(fake_llm) == 2
