"""High-level command wrappers that delegate to the current driver implementation.

This module intentionally avoids importing scripts.driver at module import time
to prevent circular imports when scripts/driver.py invokes ghostwriter.cli.
"""
from __future__ import annotations

import sys
from types import ModuleType
from typing import Optional, Tuple
from pathlib import Path
from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader


def _get_driver_module() -> ModuleType:
    """Load the legacy driver implementation from source.

    We deliberately avoid __main__ heuristics to prevent accidental recursion
    when invoked via the CLI module.
    """
    # Load from source path to avoid package import/cycles
    root = Path(__file__).resolve().parents[1]
    driver_path = root / "scripts" / "driver.py"
    if driver_path.exists():
        loader = SourceFileLoader("_gw_driver", str(driver_path))
        spec = spec_from_loader(loader.name, loader)
        if spec is None:
            raise ImportError(f"Unable to load driver spec from {driver_path}")
        mod = module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)  # type: ignore[arg-type]
        return mod

    # Last resort: try package import if scripts is a package
    import importlib
    return importlib.import_module("scripts.driver")


def run_pipelines_for_chapter(chapter_path: str, version_num: int, *, log_llm: bool = False) -> None:
    drv = _get_driver_module()
    return drv.run_pipelines_for_chapter(chapter_path, version_num, log_llm=log_llm)


def generate_pre_draft(chapter_path: str, version_num: int) -> Tuple[str, object, str]:
    drv = _get_driver_module()
    return drv.generate_pre_draft(chapter_path, version_num)


def verify_predraft(pre_draft_text: str, chapter_path: str, version_num: int) -> Tuple[str, object]:
    drv = _get_driver_module()
    return drv.verify_predraft(pre_draft_text, chapter_path, version_num)


def polish_prose(text_to_polish: str, chapter_path: str, version_num: int) -> Tuple[str, object]:
    drv = _get_driver_module()
    return drv.polish_prose(text_to_polish, chapter_path, version_num)


def validate_and_prepare(chapter_path: str) -> None:
    drv = _get_driver_module()
    return drv._validate_inputs_and_prepare(chapter_path)


def get_latest_version(chapter_id: str) -> int:
    drv = _get_driver_module()
    return drv.get_latest_version(chapter_id)


def chapter_id_from_path(chapter_path: str) -> str:
    drv = _get_driver_module()
    return drv.chapter_id_from_path(chapter_path)
