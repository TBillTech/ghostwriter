from __future__ import annotations

from pathlib import Path
import os

import pytest

from ghostwriter.mock_support import regenerate_prompt_for_log, parse_prompt_response_file
from ghostwriter.llm import complete as llm_complete


@pytest.fixture()
def use_mock_llm(monkeypatch: pytest.MonkeyPatch, lr_source_dir: Path):
    monkeypatch.setenv("GW_USE_MOCK_LLM", "1")
    monkeypatch.setenv("GW_BOOK_BASE_DIR", str(lr_source_dir))
    # Ensure any optional fallback is disabled to enforce strict matching in tests
    monkeypatch.setenv("GW_MOCKLLM_FALLBACK", "0")
    yield


def test_mock_llm_returns_sequenced_golden_responses(use_mock_llm, lr_source_dir: Path):
    # Pick a known golden log and regenerate its prompt using current templates
    base_log = lr_source_dir / "iterations/CHAPTER_001/pipeline_v1/06_dialog/06_actor_assignment.txt"
    assert base_log.exists()
    user_prompt = regenerate_prompt_for_log(base_log)
    assert user_prompt and isinstance(user_prompt, str)
    expected_first = parse_prompt_response_file(base_log).get("RESPONSE", "")

    # Optional subsequent retries: pick known variants if present
    candidates = []
    for name in ("06_actor_assignment_42.txt", "06_actor_assignment_r2.txt"):
        p = base_log.parent / name
        if p.exists():
            candidates.append(parse_prompt_response_file(p).get("RESPONSE", ""))

    # First call: baseline response (can be empty)
    got1 = llm_complete(user_prompt, system="test")
    assert isinstance(got1, str)
    assert got1.strip() == (expected_first or "").strip()

    # Second call: next available retry (should be non-empty for this dataset if any candidate exists)
    got2 = llm_complete(user_prompt, system="test")
    assert isinstance(got2, str)
    if candidates:
        assert got2.strip() in {c.strip() for c in candidates if c is not None}
        assert got2.strip() != (expected_first or "").strip()
    else:
        # If no retry files are present, at least assert idempotence (saturates at last)
        assert got2.strip() == (expected_first or "").strip()


def test_mock_llm_raises_on_unknown_prompt(use_mock_llm):
    with pytest.raises(ValueError):
        llm_complete("this prompt will not match any golden logs", system="x")
