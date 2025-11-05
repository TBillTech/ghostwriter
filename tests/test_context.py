import os
from pathlib import Path
import pytest

from ghostwriter.context import load_yaml, RunContext, MissingFileError, InvalidYAMLError


def test_load_yaml_missing_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Ensure logs go somewhere writable
    monkeypatch.setenv("GW_BOOK_BASE_DIR", str(tmp_path))
    with pytest.raises(MissingFileError):
        load_yaml(str(tmp_path / "does_not_exist.yaml"))


def test_load_yaml_invalid_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GW_BOOK_BASE_DIR", str(tmp_path))
    bad = tmp_path / "bad.yaml"
    bad.write_text("key: [ unclosed", encoding="utf-8")
    with pytest.raises(InvalidYAMLError):
        load_yaml(str(bad))


def test_runcontext_from_paths_with_lr_book(use_lr_book_env, lr_book_dir: Path):
    ch = lr_book_dir / "chapters/CHAPTER_001.yaml"
    ctx = RunContext.from_paths(chapter_path=str(ch), version=1)
    # Sanity checks on loaded context
    assert isinstance(ctx.setting, dict)
    assert isinstance(ctx.chapter, dict)
    assert isinstance(ctx.characters, list) and ctx.characters
    assert ctx.chapter_id == "CHAPTER_001"
    assert ctx.chapters_dir.name == "chapters"
    # Optional content table present in LRRH
    assert ctx.content_table is None or isinstance(ctx.content_table, dict)
