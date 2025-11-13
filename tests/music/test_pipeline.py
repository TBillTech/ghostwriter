from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest

from ghostwriter.context import UserActionRequired
from ghostwriter.music.context import ScoreSummary, VoiceContext, VoiceSpec
from ghostwriter.music.pipeline import ensure_first_score_gate, run_subtle_score_pass
from ghostwriter.musiccsv import MusicCSV, musiccsv_to_text, read_musiccsv, write_musiccsv


def _build_musiccsv(note_pitch: str = "C4") -> MusicCSV:
    return MusicCSV(
        metadata={
            "title": "Test Score",
            "composer": "Test",
            "tempo": 90,
            "time_signature": "4/4",
            "key_signature": "C",
            "divisions_per_quarter": 480,
            "version": "0.1",
        },
        tracks=[
            {
                "track": 1,
                "label": "Flute",
                "part": "Flute",
                "instrument": "Flute",
                "channel": 1,
                "program": 73,
                "volume": 100,
            }
        ],
        measures=[
            {
                "measure": 1,
                "time_signature": "4/4",
                "key_signature": "C",
                "tempo": 90,
                "start_beat": 0.0,
                "pickup": False,
            }
        ],
        notes=[
            {
                "track": 1,
                "measure": 1,
                "beat": 1.0,
                "pitch": note_pitch,
                "duration": 1.0,
                "velocity": 64,
                "tie": "none",
                "articulation": None,
                "pedal": False,
                "lyric": None,
                "ornament": None,
                "comment": None,
                "grace": False,
                "repeat": None,
                "tuplet": None,
            }
        ],
    )


def _musiccsv_text(note_pitch: str = "C4") -> str:
    return musiccsv_to_text(_build_musiccsv(note_pitch))


@pytest.fixture()
def voice_context_payload(tmp_path: Path) -> tuple[VoiceContext, dict]:
    score_dir = tmp_path / "score_assets"
    score_dir.mkdir(parents=True, exist_ok=True)
    score_path = score_dir / "score.musiccsv"
    write_musiccsv(score_path, _build_musiccsv("E4"))

    spec = VoiceSpec(
        token="major.alto.flute.red",
        chord="major",
        register="alto",
        instrument="flute",
        idea="red",
    )
    summary = ScoreSummary(
        voice_token=spec.token,
        score_path=score_path,
        tempos=["90"],
        time_signatures=["4/4"],
        key_signatures=["C"],
        measure_count=1,
        duration_quarter_length=1.0,
    )
    context = VoiceContext(
        directive="Slow burn in D minor",
        voices=[spec],
        score_summaries={spec.token: summary},
        aggregate_tempos=["90"],
        aggregate_time_signatures=["4/4"],
        aggregate_key_signatures=["C"],
        missing_assets=[],
    )
    return context, context.as_prompt_payload()


def _patch_llm(monkeypatch: pytest.MonkeyPatch, outputs: Iterator[str]) -> None:
    stub = lambda *args, **kwargs: next(outputs)
    monkeypatch.setattr("ghostwriter.music.pipeline.llm_complete", stub)
    monkeypatch.setattr("ghostwriter.pipelines.common.llm_complete", stub)


def _patch_monitor_renderer(monkeypatch: pytest.MonkeyPatch) -> None:
    def _stub(_music: MusicCSV, midi_path: Path, log_dir):
        midi_path.write_bytes(b"MThd")
        return True

    monkeypatch.setattr("ghostwriter.music.pipeline._render_monitor_midi", _stub)


def test_ensure_first_score_gate_creates_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, voice_context_payload):
    voice_context, payload = voice_context_payload
    tp_dir = tmp_path / "01_narration"

    _patch_monitor_renderer(monkeypatch)
    responses = iter(
        [
            _musiccsv_text("C4"),
            "- Balance flute dynamics with cello support.",
        ]
    )
    _patch_llm(monkeypatch, responses)

    created = ensure_first_score_gate(
        tp_dir=tp_dir,
        tp_index=1,
        tp_type="narration",
        tp_text="Red walks the forest path.",
        voice_context=voice_context,
        prompt_payload=payload,
    )
    assert created

    first_score = tp_dir / "touch_point_first_score.musiccsv"
    suggestions = tp_dir / "first_score_suggestions.txt"
    first_score_trace = tp_dir / "first_score.txt"
    score_check_trace = tp_dir / "score_check.txt"
    first_monitor = tp_dir / "first_monitor.mid"
    assert first_score.exists()
    assert suggestions.exists()
    assert first_score_trace.exists()
    assert score_check_trace.exists()
    assert first_monitor.exists()

    # Second invocation should no-op once artifacts exist
    again = ensure_first_score_gate(
        tp_dir=tp_dir,
        tp_index=1,
        tp_type="narration",
        tp_text="Red walks the forest path.",
        voice_context=voice_context,
        prompt_payload=payload,
    )
    assert not again


def test_ensure_first_score_gate_regenerates_monitor_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, voice_context_payload):
    voice_context, payload = voice_context_payload
    tp_dir = tmp_path / "01_narration"

    _patch_monitor_renderer(monkeypatch)
    responses = iter(
        [
            _musiccsv_text("C4"),
            "- Balance flute dynamics with cello support.",
        ]
    )
    _patch_llm(monkeypatch, responses)

    ensure_first_score_gate(
        tp_dir=tp_dir,
        tp_index=1,
        tp_type="narration",
        tp_text="Red walks the forest path.",
        voice_context=voice_context,
        prompt_payload=payload,
    )

    midi_path = tp_dir / "first_monitor.mid"
    assert midi_path.exists()
    midi_path.unlink()

    # Prevent LLM from being called again; monitor renderer still patched to recreate file
    monkeypatch.setattr(
        "ghostwriter.music.pipeline.llm_complete",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("llm should not be called")),
    )

    regenerated = ensure_first_score_gate(
        tp_dir=tp_dir,
        tp_index=1,
        tp_type="narration",
        tp_text="Red walks the forest path.",
        voice_context=voice_context,
        prompt_payload=payload,
    )
    assert regenerated
    assert midi_path.exists()


def test_ensure_first_score_gate_raises_when_monitor_fails_on_regen(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, voice_context_payload):
    voice_context, payload = voice_context_payload
    tp_dir = tmp_path / "01_narration"

    _patch_monitor_renderer(monkeypatch)
    responses = iter(
        [
            _musiccsv_text("C4"),
            "- Balance flute dynamics with cello support.",
        ]
    )
    _patch_llm(monkeypatch, responses)

    ensure_first_score_gate(
        tp_dir=tp_dir,
        tp_index=1,
        tp_type="narration",
        tp_text="Red walks the forest path.",
        voice_context=voice_context,
        prompt_payload=payload,
    )

    midi_path = tp_dir / "first_monitor.mid"
    assert midi_path.exists()
    midi_path.unlink()

    monkeypatch.setattr(
        "ghostwriter.music.pipeline._render_monitor_midi",
        lambda *args, **kwargs: (_ for _ in ()).throw(UserActionRequired("boom")),
    )

    with pytest.raises(UserActionRequired):
        ensure_first_score_gate(
            tp_dir=tp_dir,
            tp_index=1,
            tp_type="narration",
            tp_text="Red walks the forest path.",
            voice_context=voice_context,
            prompt_payload=payload,
        )
    assert (tp_dir / "first_score.txt").exists()


def test_first_score_trace_written_before_monitor_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, voice_context_payload):
    voice_context, payload = voice_context_payload
    tp_dir = tmp_path / "01_narration"

    responses = iter(
        [
            _musiccsv_text("C4"),
            "- Balance flute dynamics with cello support.",
        ]
    )
    _patch_llm(monkeypatch, responses)

    monkeypatch.setattr(
        "ghostwriter.music.pipeline._render_monitor_midi",
        lambda *args, **kwargs: (_ for _ in ()).throw(UserActionRequired("boom")),
    )

    with pytest.raises(UserActionRequired):
        ensure_first_score_gate(
            tp_dir=tp_dir,
            tp_index=1,
            tp_type="narration",
            tp_text="Red walks the forest path.",
            voice_context=voice_context,
            prompt_payload=payload,
        )

    assert (tp_dir / "touch_point_first_score.musiccsv").exists()
    assert (tp_dir / "first_score.txt").exists()


def test_first_score_gate_retries_on_empty_response(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, voice_context_payload):
    voice_context, payload = voice_context_payload
    tp_dir = tmp_path / "01_narration"

    _patch_monitor_renderer(monkeypatch)
    responses = iter(
            [
                "",  # first attempt invalid
                _musiccsv_text("C4"),
                "- Balance flute dynamics with cello support.",
            ]
        )
    _patch_llm(monkeypatch, responses)

    created = ensure_first_score_gate(
        tp_dir=tp_dir,
        tp_index=1,
        tp_type="narration",
        tp_text="Red walks the forest path.",
        voice_context=voice_context,
        prompt_payload=payload,
    )

    assert created
    assert (tp_dir / "touch_point_first_score.musiccsv").exists()
    assert (tp_dir / "first_monitor.mid").exists()


def test_first_score_gate_raises_after_failed_retries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, voice_context_payload):
    voice_context, payload = voice_context_payload
    tp_dir = tmp_path / "01_narration"

    _patch_monitor_renderer(monkeypatch)
    responses = iter(["", "", ""])
    _patch_llm(monkeypatch, responses)

    with pytest.raises(ValueError):
        ensure_first_score_gate(
            tp_dir=tp_dir,
            tp_index=1,
            tp_type="narration",
            tp_text="Red walks the forest path.",
            voice_context=voice_context,
            prompt_payload=payload,
        )

def test_run_subtle_score_pass_uses_feedback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, voice_context_payload):
    voice_context, payload = voice_context_payload
    tp_dir = tmp_path / "01_narration"

    _patch_monitor_renderer(monkeypatch)
    first_gate_responses = iter(
        [
            _musiccsv_text("C4"),
            "- Add more motion to measure 2.",
        ]
    )
    _patch_llm(monkeypatch, first_gate_responses)
    ensure_first_score_gate(
        tp_dir=tp_dir,
        tp_index=1,
        tp_type="narration",
        tp_text="Red walks the forest path.",
        voice_context=voice_context,
        prompt_payload=payload,
    )

    # Simulate user edits by tweaking the first score
    first_score = tp_dir / "touch_point_first_score.musiccsv"
    write_musiccsv(first_score, _build_musiccsv("D4"))
    (tp_dir / "first_score_suggestions.txt").write_text("- Emphasize the downbeat.", encoding="utf-8")

    subtle_responses = iter(
        [
            _musiccsv_text("E4"),
            "- Dynamics look balanced now.",
        ]
    )
    _patch_llm(monkeypatch, subtle_responses)

    ran = run_subtle_score_pass(
        tp_dir=tp_dir,
        tp_index=1,
        tp_type="narration",
        tp_text="Red walks the forest path.",
        voice_context=voice_context,
        prompt_payload=payload,
    )
    assert ran

    final_score = tp_dir / "touch_point_score.musiccsv"
    final_feedback = tp_dir / "score_suggestions.txt"
    subtle_score_trace = tp_dir / "subtle_score.txt"
    subtle_check_trace = tp_dir / "score_check.txt"
    assert final_score.exists()
    assert final_feedback.exists()
    updated_music = read_musiccsv(final_score)
    assert any(note.get("pitch") == "E4" for note in updated_music.notes)
    assert final_feedback.read_text(encoding="utf-8").startswith("-")
    assert subtle_score_trace.exists()
    assert subtle_check_trace.exists()