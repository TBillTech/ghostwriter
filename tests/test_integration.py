import os
from pathlib import Path
import pytest

from ghostwriter.chapter import run_pipelines_for_chapter


def test_resume_run_completes_and_keeps_artifacts(monkeypatch: pytest.MonkeyPatch, use_lr_book_env, lr_book_dir: Path):
    # Ensure MockLLM is used and strict
    monkeypatch.setenv("GW_USE_MOCK_LLM", "1")
    monkeypatch.setenv("GW_MOCKLLM_FALLBACK", "0")

    chapter_path = lr_book_dir / "chapters/CHAPTER_001.yaml"
    assert chapter_path.exists()

    # Run pipelines for v1 on an already-completed golden; expect a RESUME flow without exceptions
    run_pipelines_for_chapter(str(chapter_path), 1, log_llm=True)

    # Verify artifacts for the first contentful step exist (05_narration)
    base = lr_book_dir / "iterations/CHAPTER_001/pipeline_v1/05_narration"
    assert base.exists()
    # Presence of brainstorm DONE file in LRRH:
    assert (base / "brainstorm.txt").exists()
    # Gate outputs exist from golden data
    assert (base / "touch_point_first_draft.txt").exists()
    assert (base / "first_suggestions.txt").exists()
    # A debug check.txt (prompt+response) should be present
    assert (base / "check.txt").exists()
    # Final outputs should be regenerated/written
    iterdir = lr_book_dir / "iterations/CHAPTER_001"
    assert (iterdir / "draft_v1.txt").exists()
    assert (iterdir / "final.txt").exists()


def test_resume_after_user_gate_runs_subtle_edit_and_writes_draft(monkeypatch: pytest.MonkeyPatch, use_lr_book_env, lr_book_dir: Path):
    # Force MockLLM strict mode
    monkeypatch.setenv("GW_USE_MOCK_LLM", "1")
    monkeypatch.setenv("GW_MOCKLLM_FALLBACK", "0")

    chapter_path = lr_book_dir / "chapters/CHAPTER_001.yaml"
    base = lr_book_dir / "iterations/CHAPTER_001/pipeline_v1/05_narration"
    assert chapter_path.exists() and base.exists()

    # Simulate a user-gate resume: keep first_draft and first_suggestions, remove final draft and suggestions
    first_draft = base / "touch_point_first_draft.txt"
    first_suggestions = base / "first_suggestions.txt"
    final_draft = base / "touch_point_draft.txt"
    suggestions = base / "suggestions.txt"
    # Preconditions: first gate files exist in golden
    assert first_draft.exists()
    assert first_suggestions.exists()
    # Remove resume targets if present
    if final_draft.exists():
        final_draft.unlink()
    if suggestions.exists():
        suggestions.unlink()

    # Run pipelines; subtle_edit should execute for the narration step and write the final draft + suggestions
    run_pipelines_for_chapter(str(chapter_path), 1, log_llm=True)

    assert final_draft.exists()
    assert final_draft.read_text(encoding="utf-8").strip()
    assert suggestions.exists()
    # Ensure draft_v1 and final.txt exist at chapter level, too
    iterdir = lr_book_dir / "iterations/CHAPTER_001"
    assert (iterdir / "draft_v1.txt").exists()
    assert (iterdir / "final.txt").exists()
