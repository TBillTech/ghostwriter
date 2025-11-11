from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest

from ghostwriter.music.context import ScoreSummary, VoiceContext, VoiceSpec
from ghostwriter.music.pipeline import ensure_first_score_gate, run_subtle_score_pass


@pytest.fixture()
def voice_context_payload(tmp_path: Path) -> tuple[VoiceContext, dict]:
    score_dir = tmp_path / "score_assets"
    score_dir.mkdir(parents=True, exist_ok=True)
    score_path = score_dir / "score.musicxml"
    score_path.write_text("<score-partwise version='3.1'><part id='P1'></part></score-partwise>", encoding="utf-8")

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
        time_signatures=["3/4"],
        key_signatures=["D minor"],
        measure_count=4,
        duration_quarter_length=16.0,
    )
    context = VoiceContext(
        directive="Slow burn in D minor",
        voices=[spec],
        score_summaries={spec.token: summary},
        aggregate_tempos=["90"],
        aggregate_time_signatures=["3/4"],
        aggregate_key_signatures=["D minor"],
        missing_assets=[],
    )
    return context, context.as_prompt_payload()


def _patch_llm(monkeypatch: pytest.MonkeyPatch, outputs: Iterator[str]) -> None:
    monkeypatch.setattr(
        "ghostwriter.music.pipeline.llm_complete",
        lambda *args, **kwargs: next(outputs),
    )


def test_ensure_first_score_gate_creates_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, voice_context_payload):
    voice_context, payload = voice_context_payload
    tp_dir = tmp_path / "01_narration"

    responses = iter(
        [
            "<score-partwise version='3.1'><part id='P1'></part></score-partwise>",
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

    first_score = tp_dir / "touch_point_first_score.musicxml"
    suggestions = tp_dir / "first_score_suggestions.txt"
    assert first_score.exists()
    assert suggestions.exists()

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


def test_run_subtle_score_pass_uses_feedback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, voice_context_payload):
    voice_context, payload = voice_context_payload
    tp_dir = tmp_path / "01_narration"

    first_gate_responses = iter(
        [
            "<score-partwise version='3.1'><part id='P1'></part></score-partwise>",
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
    first_score = tp_dir / "touch_point_first_score.musicxml"
    first_score.write_text(
        "<score-partwise version='3.1'><part id='P1'><measure number='1'></measure></part></score-partwise>",
        encoding="utf-8",
    )
    (tp_dir / "first_score_suggestions.txt").write_text("- Emphasize the downbeat.", encoding="utf-8")

    subtle_responses = iter(
        [
            "<score-partwise version='3.1'><part id='P1'><measure number='1'><note/></measure></part></score-partwise>",
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

    final_score = tp_dir / "touch_point_score.musicxml"
    final_feedback = tp_dir / "score_suggestions.txt"
    assert final_score.exists()
    assert final_feedback.exists()
    assert "note" in final_score.read_text(encoding="utf-8")
    assert final_feedback.read_text(encoding="utf-8").startswith("-")