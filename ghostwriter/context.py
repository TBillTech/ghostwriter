"""RunContext and YAML loading utilities.

Contract:
- RunContext.from_paths(chapter_path: str, version: int) -> RunContext
- load_yaml(path: str) -> dict | list | scalar

This is the only module that performs YAML reads during a run.
Other modules accept a RunContext instance and avoid YAML I/O.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Any

import yaml
from yaml.loader import SafeLoader as _PySafeLoader
from .env import get_setting_path, get_characters_path, resolve_chapter_path

class GWError(Exception):
    pass

class MissingFileError(GWError):
    pass

class InvalidYAMLError(GWError):
    pass


def _yaml_load_py(content: str):
    return yaml.load(content, Loader=_PySafeLoader)


def load_yaml(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        raise MissingFileError(f"Required file not found: {path}")
    except Exception as e:
        raise GWError(f"Unable to read file {path}: {e}")
    try:
        return _yaml_load_py(content)
    except yaml.YAMLError as e:  # type: ignore[attr-defined]
        raise InvalidYAMLError(f"Invalid YAML in {path}: {e}")
    except Exception as e:
        raise InvalidYAMLError(f"Invalid YAML in {path}: {e}")


def chapter_id_from_path(chapter_path: str) -> str:
    return Path(chapter_path).stem


@dataclass(frozen=True)
class RunContext:
    setting: dict
    chapter: dict
    characters: List[dict]
    chapter_id: str
    version: int

    @classmethod
    def from_paths(cls, *, chapter_path: str, version: int) -> "RunContext":
        # Resolve configurable paths (ENV can change locations)
        setting_path = get_setting_path()
        characters_path = get_characters_path()
        resolved_chapter = resolve_chapter_path(chapter_path)
        if not setting_path.exists():
            raise MissingFileError(f"Missing required SETTING.yaml: {setting_path}")
        setting = load_yaml(str(setting_path))
        if not resolved_chapter.exists():
            raise MissingFileError(f"Missing chapter file: {chapter_path} (resolved: {resolved_chapter})")
        chapter = load_yaml(str(resolved_chapter))
        if not characters_path.exists():
            raise MissingFileError(f"Missing required CHARACTERS.yaml: {characters_path}")
        chars_yaml = load_yaml(str(characters_path))
        characters_list: List[dict] = []
        if isinstance(chars_yaml, list):
            characters_list = [c for c in chars_yaml if isinstance(c, dict)]
        elif isinstance(chars_yaml, dict) and isinstance(chars_yaml.get("Characters"), list):
            characters_list = [c for c in chars_yaml.get("Characters") if isinstance(c, dict)]
        elif isinstance(setting, dict) and isinstance(setting.get("Characters"), list):
            characters_list = [c for c in setting.get("Characters") if isinstance(c, dict)]
        cid = chapter_id_from_path(str(resolved_chapter))
        return cls(setting=setting, chapter=chapter, characters=characters_list, chapter_id=cid, version=version)
