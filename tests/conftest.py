import os
import shutil
from pathlib import Path

import pytest

# Load a test-specific environment file so pytest runs are consistent locally and in VS Code
try:
    from dotenv import load_dotenv
    _root = Path(__file__).resolve().parents[1]
    _env_test = _root / ".env.test"
    if _env_test.exists():
        load_dotenv(dotenv_path=_env_test, override=True)
except Exception:
    # Non-fatal if dotenv is unavailable; tests also work without it
    pass


@pytest.fixture(scope="session")
def lr_source_dir() -> Path:
    # Source of the Little Red Riding Hood testbed within the repo
    return Path(__file__).resolve().parents[1] / "testdata" / "LittleRedRidingHood"


@pytest.fixture()
def lr_book_dir(tmp_path: Path, lr_source_dir: Path) -> Path:
    # Copy the LRRH testbed into a sandbox per-test directory
    dest = tmp_path / "book"
    shutil.copytree(lr_source_dir, dest)
    return dest


@pytest.fixture()
def use_lr_book_env(monkeypatch: pytest.MonkeyPatch, lr_book_dir: Path):
    # Point the program to the sandboxed book directory
    monkeypatch.setenv("GW_BOOK_BASE_DIR", str(lr_book_dir))
    # Unset overrides to exercise default resolution under base dir
    for k in ("GW_SETTING_PATH", "GW_CHARACTERS_PATH", "GW_CHAPTERS_DIR", "GW_ITERATIONS_DIR"):
        try:
            monkeypatch.delenv(k, raising=False)
        except Exception:
            pass
    yield
