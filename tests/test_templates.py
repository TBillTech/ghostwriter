import os
import shutil
from pathlib import Path

import pytest

from ghostwriter.templates import (
    prompt_key_from_filename,
    iter_dir_for,
    list_versions,
    get_latest_version,
    read_latest,
    build_common_replacements,
    build_master_initial_prompt,
    build_master_prompt,
    build_polish_prompt,
    build_check_prompt,
    build_story_so_far_prompt,
    build_story_relative_to_prompt,
    apply_template,
    _yaml_section_fallback,
)
from ghostwriter.env import get_iterations_dir


def test_prompt_key_from_filename():
    assert prompt_key_from_filename("narration_brain_storm_prompt.md") == "NARRATION_BRAIN_STORM"
    assert prompt_key_from_filename("check.md") == "CHECK"
    assert prompt_key_from_filename("my-file_name.Prompt.MD") == "MY_FILE_NAME_PROMPT"


def test_yaml_section_fallback_basic():
    ch = {"Story-Relative-To": {"a": 1}}
    out = _yaml_section_fallback(ch, "Story-Relative-To")
    # Should be JSON text containing the key and value
    assert "Story-Relative-To" in out
    assert "1" in out
    # Missing returns empty string
    assert _yaml_section_fallback(ch, "Not-There") == ""


def test_iter_dir_and_versions_and_read_latest(tmp_path: Path):
    chapter_id = "UNIT_TEST_CH"
    # Use configured iterations dir so tests are environment-agnostic
    base = get_iterations_dir() / chapter_id
    try:
        # Ensure clean slate
        if base.exists():
            shutil.rmtree(base)
        base.mkdir(parents=True, exist_ok=True)

        # Create some versioned files
        (base / "draft_v1.txt").write_text("d1", encoding="utf-8")
        (base / "draft_v2.txt").write_text("d2", encoding="utf-8")
        (base / "check_v1.txt").write_text("c1", encoding="utf-8")

        assert iter_dir_for(chapter_id) == base
        assert list_versions(chapter_id, "draft") == [1, 2]
        assert get_latest_version(chapter_id) == 2
        assert read_latest(chapter_id, "draft") == "d2"
        assert read_latest(chapter_id, "check") == "c1"
    finally:
        if base.exists():
            shutil.rmtree(base)


def test_build_common_replacements_fallback_and_files():
    chapter_id = "UNIT_TEST_CH2"
    base = get_iterations_dir() / chapter_id
    try:
        if base.exists():
            shutil.rmtree(base)
        # Case 1: No files present, use chapter fallbacks
        setting = {"title": "S"}
        chapter = {"Story-So-Far": "ssf_chapter", "Story-Relative-To": {"note": "rel"}}
        reps = build_common_replacements(setting, chapter, chapter_id, 1)
        assert reps["[story_so_far.txt]"] == "ssf_chapter"
        # Story-Relative-To serialized JSON should include key
        assert "Story-Relative-To" in reps["[story_relative_to.txt]"]

        # Case 2: Files present override chapter
        base.mkdir(parents=True, exist_ok=True)
        (base / "story_so_far.txt").write_text("ssf_file", encoding="utf-8")
        (base / "story_relative_to.txt").write_text("srt_file", encoding="utf-8")
        reps2 = build_common_replacements(setting, chapter, chapter_id, 1)
        assert reps2["[story_so_far.txt]"] == "ssf_file"
        assert reps2["[story_relative_to.txt]"] == "srt_file"
    finally:
        if base.exists():
            shutil.rmtree(base)


def test_apply_template_basic():
    prompts_dir = Path("prompts")
    test_file = prompts_dir / "unit_test_template.md"
    try:
        prompts_dir.mkdir(parents=True, exist_ok=True)
        test_file.write_text("Hello [NAME]!", encoding="utf-8")
        out = apply_template(str(test_file), {"[NAME]": "World"})
        assert out == "Hello World!"
    finally:
        if test_file.exists():
            test_file.unlink()
        # Don't remove prompts_dir if it already existed for project prompts