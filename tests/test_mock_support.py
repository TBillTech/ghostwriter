from __future__ import annotations

import os
from pathlib import Path

import pytest

from ghostwriter.mock_support import (
    compute_prompts_hash,
    write_prompt_hash,
    read_prompt_hash,
    prompts_hash_matches,
    parse_prompt_response_file,
    write_prompt_response_file,
    regenerate_prompt_for_log,
    update_golden_prompts,
    copy_partial_book,
    prepare_user_gate,
    get_current_step_prompt,
    get_current_step_response,
)


def test_prompt_hash_write_and_match(lr_book_dir: Path):
    h = compute_prompts_hash("prompts")
    assert isinstance(h, str) and len(h) >= 32
    # Read any existing prompt_hash (may already exist if source testdata has it)
    existing = read_prompt_hash(lr_book_dir)
    # Write and verify
    write_prompt_hash(lr_book_dir, h)
    assert read_prompt_hash(lr_book_dir) == h
    assert prompts_hash_matches("prompts", lr_book_dir) is True


def test_parse_and_write_roundtrip(tmp_path: Path):
    p = tmp_path / "log.txt"
    sys = "system text\nwith lines"
    usr = "user text"
    resp = "response text"
    write_prompt_response_file(p, sys, usr, resp)
    parts = parse_prompt_response_file(p)
    assert parts["SYSTEM"].startswith("system text")
    assert parts["USER"] == usr
    assert parts["RESPONSE"] == resp
    # Modify and write again
    write_prompt_response_file(p, parts["SYSTEM"], parts["USER"] + "!", parts["RESPONSE"]) 
    parts2 = parse_prompt_response_file(p)
    assert parts2["USER"].endswith("!")


def test_regenerate_prompt_for_log_returns_text(use_lr_book_env, lr_book_dir: Path):
    log = lr_book_dir / "iterations/CHAPTER_001/pipeline_v1/06_dialog/06_actor_assignment.txt"
    assert log.exists(), f"Missing expected log file: {log}"
    regenerated = regenerate_prompt_for_log(log)
    assert regenerated is not None and isinstance(regenerated, str) and len(regenerated) > 0


def test_update_golden_prompts_dry_run_does_not_modify(use_lr_book_env, lr_book_dir: Path):
    target = lr_book_dir / "iterations/CHAPTER_001/pipeline_v1/06_dialog/06_agenda.txt"
    before = parse_prompt_response_file(target)["USER"]
    hash_before = read_prompt_hash(lr_book_dir)
    results = update_golden_prompts(lr_book_dir, dry_run=True)
    after = parse_prompt_response_file(target)["USER"]
    assert after == before
    # dry_run should not create or change prompt_hash
    assert read_prompt_hash(lr_book_dir) == hash_before
    assert isinstance(results, list)


def test_update_golden_prompts_updates_modified_file_and_writes_hash(use_lr_book_env, lr_book_dir: Path):
    target = lr_book_dir / "iterations/CHAPTER_001/pipeline_v1/06_dialog/06_actor_assignment.txt"
    parts = parse_prompt_response_file(target)
    # Overwrite USER with a sentinel
    write_prompt_response_file(target, parts.get("SYSTEM", ""), "THIS IS A TEST USER PROMPT", parts.get("RESPONSE", ""))
    # Run updater (non-dry)
    results = update_golden_prompts(lr_book_dir, dry_run=False)
    parts_after = parse_prompt_response_file(target)
    assert parts_after["USER"] != "THIS IS A TEST USER PROMPT"
    # prompt_hash should be written and match current prompts dir
    h_now = compute_prompts_hash("prompts")
    assert read_prompt_hash(lr_book_dir) == h_now
    assert prompts_hash_matches("prompts", lr_book_dir) is True
    assert isinstance(results, list)


def test_golden_update_summary_and_idempotency(use_lr_book_env, lr_book_dir: Path):
    # First run should update some files and write prompt_hash
    results1 = update_golden_prompts(lr_book_dir, dry_run=False)
    assert isinstance(results1, list) and len(results1) > 0
    assert read_prompt_hash(lr_book_dir) is not None
    # Expect at least one item without a template mapping (e.g., check.txt files)
    assert any(r.get("reason") == "no-template" for r in results1)

    # Second run should be idempotent: anything with a template mapping should now be unchanged
    results2 = update_golden_prompts(lr_book_dir, dry_run=False)
    updated_again = [r for r in results2 if r.get("updated") == "yes"]
    assert not updated_again, f"Expected idempotent update, but found updates: {updated_again[:3]}"
    # And the hash should still match
    assert prompts_hash_matches("prompts", lr_book_dir) is True


def test_copy_partial_book_and_prepare_user_gate(tmp_path: Path, lr_book_dir: Path):
    # 1) Copy with filename cutoff and verify inclusion/exclusion
    dest1 = tmp_path / "partial1"
    copy_partial_book(
        lr_book_dir,
        dest1,
        chapter_id="CHAPTER_001",
        version=1,
        upto_step_dir="06_dialog",
        upto_filename="06_body_language.txt",
        clear_dest=True,
    )
    assert (dest1 / "SETTING.yaml").exists()
    assert (dest1 / "chapters/CHAPTER_001.yaml").exists()
    # Later step should not be present
    assert not (dest1 / "iterations/CHAPTER_001/pipeline_v1/07_mixed").exists()
    # File up to cutoff should exist
    assert (dest1 / "iterations/CHAPTER_001/pipeline_v1/06_dialog/06_body_language.txt").exists()
    # File beyond cutoff should not be copied
    assert not (dest1 / "iterations/CHAPTER_001/pipeline_v1/06_dialog/06_reactions.txt").exists()

    # 2) Copy entire step (no filename cutoff) and prepare a user gate
    dest2 = tmp_path / "partial2"
    copy_partial_book(
        lr_book_dir,
        dest2,
        chapter_id="CHAPTER_001",
        version=1,
        upto_step_dir="06_dialog",
        upto_filename=None,
        clear_dest=True,
    )
    step = dest2 / "iterations/CHAPTER_001/pipeline_v1/06_dialog"
    # Ensure required files exist first
    assert (step / "brainstorm.txt").exists()
    assert (step / "touch_point_draft.txt").exists()
    assert (step / "suggestions.txt").exists()
    # Prepare user gate should strip DONE and remove two files
    prepare_user_gate(dest2, chapter_id="CHAPTER_001", version=1, tp_dir="06_dialog")
    # Verify files removed
    assert not (step / "touch_point_draft.txt").exists()
    assert not (step / "suggestions.txt").exists()
    # Verify last line DONE removed
    content = (step / "brainstorm.txt").read_text(encoding="utf-8").strip().splitlines()
    assert content and content[-1].strip().upper() != "DONE"


def test_current_step_getters_on_partial(tmp_path: Path, lr_book_dir: Path):
    dest = tmp_path / "partial3"
    copy_partial_book(
        lr_book_dir,
        dest,
        chapter_id="CHAPTER_001",
        version=1,
        upto_step_dir="06_dialog",
        upto_filename=None,
        clear_dest=True,
    )
    # Should return prompt/response from the last log in 06_dialog
    prompt = get_current_step_prompt(dest)
    response = get_current_step_response(dest)
    assert isinstance(prompt, (str, type(None)))
    assert isinstance(response, (str, type(None)))
    assert prompt is not None and len(prompt) > 0
    assert response is not None and len(response) > 0
