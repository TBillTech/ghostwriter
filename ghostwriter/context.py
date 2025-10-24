"""RunContext and YAML loading utilities.

Contract:
- RunContext.from_paths(chapter_path: str, version: int) -> RunContext
- load_yaml(path: str) -> dict | list | scalar

This is the only module that performs YAML reads during a run.
Other modules accept a RunContext instance and avoid YAML I/O.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Any, Optional, Dict

import yaml
from yaml.loader import SafeLoader as _PySafeLoader
from .env import get_setting_path, get_characters_path, resolve_chapter_path, get_book_base_dir, get_chapters_dir
from .logging import breadcrumb as _breadcrumb

class GWError(Exception):
    pass

class MissingFileError(GWError):
    pass

class InvalidYAMLError(GWError):
    pass


class UserActionRequired(GWError):
    """Raised to gracefully stop execution when human input is required.

    Handlers should catch this exception at the top level and exit cleanly
    without treating it as an error condition.
    """
    pass


def _yaml_load_py(content: str):
    return yaml.load(content, Loader=_PySafeLoader)


def load_yaml(path: str):
    # Log YAML parsing attempts to base directory
    def _log_parse(msg: str) -> None:
        try:
            base = get_book_base_dir()
            base.mkdir(parents=True, exist_ok=True)
            log_path = base / "yaml_parse.log"
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write(f"[{ts}] {msg}\n")
        except Exception:
            pass
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        try:
            _breadcrumb(f"yaml:read:{path}")
        except Exception:
            pass
    except FileNotFoundError:
        _log_parse(f"ERROR FileNotFound: {path}")
        try:
            _breadcrumb(f"yaml:error:not_found:{path}")
        except Exception:
            pass
        raise MissingFileError(f"Required file not found: {path}")
    except Exception as e:
        _log_parse(f"ERROR ReadFailure: {path} :: {e}")
        try:
            _breadcrumb(f"yaml:error:read_failure:{path}")
        except Exception:
            pass
        raise GWError(f"Unable to read file {path}: {e}")
    try:
        data = _yaml_load_py(content)
        _log_parse(f"OK Parsed: {path}")
        try:
            _breadcrumb(f"yaml:parsed_ok:{path}")
        except Exception:
            pass
        return data
    except yaml.YAMLError as e:  # type: ignore[attr-defined]
        _log_parse(f"ERROR InvalidYAML: {path} :: {e}")
        try:
            _breadcrumb(f"yaml:error:invalid_yaml:{path}")
        except Exception:
            pass
        raise InvalidYAMLError(f"Invalid YAML in {path}: {e}")
    except Exception as e:
        _log_parse(f"ERROR InvalidYAML: {path} :: {e}")
        try:
            _breadcrumb(f"yaml:error:invalid_yaml:{path}")
        except Exception:
            pass
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
    content_table: Optional[dict]
    content_table_text: str
    chapters_dir: Path
    chapter_settings_index: List[Dict[str, Any]]

    @classmethod
    def from_paths(cls, *, chapter_path: str, version: int, allow_missing_chapter: bool = False) -> "RunContext":
        # Resolve configurable paths (ENV can change locations)
        setting_path = get_setting_path()
        characters_path = get_characters_path()
        resolved_chapter = resolve_chapter_path(chapter_path)
        if not setting_path.exists():
            raise MissingFileError(f"Missing required SETTING.yaml: {setting_path}")
        setting = load_yaml(str(setting_path))
        if not resolved_chapter.exists():
            if allow_missing_chapter:
                chapter = {}
            else:
                raise MissingFileError(f"Missing chapter file: {chapter_path} (resolved: {resolved_chapter})")
        else:
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
        # Optional content table and chapter settings index
        ch_dir = get_chapters_dir()
        ct_path = ch_dir / "CONTENT_TABLE.yaml"
        ct_text = ""
        ct_obj: Optional[dict] = None
        try:
            _breadcrumb(f"content_table:path={ct_path} exists={ct_path.exists()}")
        except Exception:
            pass
        if ct_path.exists():
            try:
                ct_text = Path(ct_path).read_text(encoding="utf-8")
                try:
                    _breadcrumb("content_table:read_text_ok")
                except Exception:
                    pass
            except Exception:
                ct_text = ""
                try:
                    _breadcrumb("content_table:read_text_fail")
                except Exception:
                    pass
            try:
                ct_yaml = load_yaml(str(ct_path))
                ct_obj = ct_yaml if isinstance(ct_yaml, dict) else None
            except Exception:
                ct_obj = None
        # Build index of chapter settings (non-dereferenced)
        index: List[Dict[str, Any]] = []
        try:
            for p in sorted(ch_dir.glob("CHAPTER_*.yaml")):
                try:
                    ch = load_yaml(str(p))
                    if isinstance(ch, dict) and "setting" in ch:
                        index.append({"chapter": p.stem, "setting": ch.get("setting")})
                except Exception:
                    continue
        except Exception:
            index = []
        return cls(
            setting=setting,
            chapter=chapter,
            characters=characters_list,
            chapter_id=cid,
            version=version,
            content_table=ct_obj,
            content_table_text=ct_text,
            chapters_dir=ch_dir,
            chapter_settings_index=index,
        )
