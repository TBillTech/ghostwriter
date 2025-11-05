from __future__ import annotations

from pathlib import Path
import os

import pytest

from ghostwriter.mock_support import (
    compute_prompts_hash,
    write_prompt_hash,
    read_prompt_hash,
    prompts_hash_matches,
    parse_prompt_response_file,
    write_prompt_response_file,
    update_golden_prompts,
    regenerate_prompt_for_log,
    copy_partial_book,
    prepare_user_gate,
    get_current_step_prompt,
    get_current_step_response,
)


def test_compute_write_read_hash_roundtrip(tmp_path: Path):
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir(parents=True)
    (prompts_dir / "a.md").write_text("hello", encoding="utf-8")
    (prompts_dir / "sub").mkdir()
    (prompts_dir / "sub" / "b.md").write_text("world", encoding="utf-8")

    h1 = compute_prompts_hash(prompts_dir)
    assert isinstance(h1, str) and len(h1) == 64

    # Write and read back
    hv = write_prompt_hash(tmp_path, h1)
    assert hv == h1
    r = read_prompt_hash(tmp_path)
    assert r == h1
    assert prompts_hash_matches(prompts_dir, tmp_path) is True

    # Mutate a file -> hash should change and mismatch
    (prompts_dir / "a.md").write_text("HELLO", encoding="utf-8")
    h2 = compute_prompts_hash(prompts_dir)
    assert h2 != h1
    assert prompts_hash_matches(prompts_dir, tmp_path) is False


def test_parse_write_prompt_response_roundtrip(tmp_path: Path):
    p = tmp_path / "log.txt"
    sys = "SYSTEM TEXT"
    usr = "USER TEXT\nline2"
    resp = "RESPONSE TEXT"
    write_prompt_response_file(p, sys, usr, resp)
    parts = parse_prompt_response_file(p)
    assert parts.get("SYSTEM") == sys
    assert parts.get("USER") == usr
    assert parts.get("RESPONSE") == resp


def test_update_golden_prompts_rewrites_user_preserves_response(lr_book_dir: Path, use_lr_book_env):
    # Pick a known log and intentionally damage the USER section
    log = lr_book_dir / "iterations/CHAPTER_001/pipeline_v1/06_dialog/06_actor_assignment.txt"
    assert log.exists()
    parts = parse_prompt_response_file(log)
    original_system = parts.get("SYSTEM", "")
    original_response = parts.get("RESPONSE", "")

    # Overwrite with bogus USER and same RESPONSE
    write_prompt_response_file(log, original_system, "BOGUS USER CONTENT", original_response)

    # Run updater
    results = update_golden_prompts(lr_book_dir, dry_run=False)
    # Ensure our file was updated
    entry = next((r for r in results if r.get("file") == str(log)), None)
    assert entry is not None and entry.get("updated") == "yes"

    # USER should match regenerated prompt; RESPONSE preserved
    regen = regenerate_prompt_for_log(log) or ""
    updated = parse_prompt_response_file(log)
    import re
    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "")).strip()
    assert _norm(updated.get("USER", "")) == _norm(regen)
    assert updated.get("RESPONSE", "") == original_response

    # prompt_hash should be written to book base dir
    assert (lr_book_dir / "prompt_hash").exists()


def test_copy_partial_book_and_prepare_user_gate(lr_source_dir: Path, tmp_path: Path):
    dest = tmp_path / "partial"
    copy_partial_book(
        lr_source_dir,
        dest,
        chapter_id="CHAPTER_001",
        version=1,
        upto_step_dir="06_dialog",
        upto_filename="06_dialog_batch.txt",
        clear_dest=True,
    )
    # Root files copied
    assert (dest / "SETTING.yaml").exists()
    assert (dest / "CHARACTERS.yaml").exists()
    # Earlier steps copied entirely
    assert (dest / "iterations/CHAPTER_001/pipeline_v1/03_actors").exists()
    # Step 06_dialog exists with subset of files
    d6 = dest / "iterations/CHAPTER_001/pipeline_v1/06_dialog"
    assert d6.exists()
    # Should include actor_assignment, agenda, body_language, brainstorm_resume (<= dialog_batch) and dialog_batch itself
    expected_subset = {
        "06_actor_assignment.txt",
        "06_agenda.txt",
        "06_body_language.txt",
        "06_brainstorm_resume.txt",
        "06_dialog_batch.txt",
    }
    present = {p.name for p in d6.iterdir() if p.is_file()}
    assert expected_subset.issubset(present)
    # Should exclude reactions and later
    assert "06_reactions.txt" not in present

    # Prepare user gate: removes suggestions.txt and touch_point_draft.txt; strips DONE from brainstorm.txt
    prepare_user_gate(dest, chapter_id="CHAPTER_001", version=1, tp_dir="06_dialog")
    for name in ("touch_point_draft.txt", "suggestions.txt"):
        p = d6 / name
        assert not p.exists()
    # Brainstorm file should not end with DONE
    bs = d6 / "brainstorm.txt"
    if bs.exists():
        last = bs.read_text(encoding="utf-8").strip().splitlines()[-1].strip()
        assert last.lower() != "done"


def test_current_step_getters(lr_book_dir: Path):
    # Should return the latest step's prompt/response
    prompt = get_current_step_prompt(lr_book_dir) or ""
    response = get_current_step_response(lr_book_dir) or ""
    assert isinstance(prompt, str) and isinstance(response, str)
    assert prompt.strip() != "" and response.strip() != ""
    # Cross-check against filesystem-derived latest log
    base = Path(lr_book_dir)
    it = base / "iterations"
    chapters = sorted([d for d in it.iterdir() if d.is_dir() and d.name.startswith("CHAPTER_")])
    last_ch = chapters[-1]
    pipelines = sorted([d for d in last_ch.iterdir() if d.is_dir() and d.name.startswith("pipeline_v")], key=lambda d: int(d.name.split("_v")[-1]))
    last_pipe = pipelines[-1]
    steps = sorted([d for d in last_pipe.iterdir() if d.is_dir() and d.name.split("_",1)[0].isdigit()], key=lambda d: int(d.name.split("_",1)[0]))
    last_step = steps[-1]
    logs = sorted([f for f in last_step.glob("*.txt") if f.name[:2].isdigit()])
    last_log = logs[-1]
    parts = parse_prompt_response_file(last_log)
    assert prompt.strip() == (parts.get("USER", "").strip())
    assert response.strip() == (parts.get("RESPONSE", "").strip())


def test_regenerate_prompt_for_log_matches_user(lr_source_dir: Path, use_lr_book_env):
    # Use a known log and confirm normalized USER matches regenerated prompt
    log = lr_source_dir / "iterations/CHAPTER_001/pipeline_v1/06_dialog/06_actor_assignment.txt"
    regen = regenerate_prompt_for_log(log) or ""
    parts = parse_prompt_response_file(log)
    got_user = parts.get("USER", "")
    import re
    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "")).strip()
    # Allow golden USER to include extra seed bullets or appended lines; ensure regenerated is contained
    assert _norm(regen) in _norm(got_user)

