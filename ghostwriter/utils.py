"""Utility helpers: file I/O, env loading, and safe JSON text rendering.

Public helpers:
- save_text(path, content)
- read_file(path)
- read_text(path)  # alias-like lower-level without GWError wrapping
- to_text(obj)
- load_env()  # loads .env with override
"""
from __future__ import annotations
from pathlib import Path
from typing import Any
import json
from dotenv import load_dotenv

# Prefer unified error types from context
try:  # avoid hard dependency during tooling
    from .context import GWError, MissingFileError  # type: ignore
except Exception:  # fallback local definitions for isolated use
    class GWError(Exception):
        pass
    class MissingFileError(GWError):
        pass


def save_text(path: str | Path, content: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def read_text(path: str | Path) -> str:
    p = Path(path)
    if not p.exists():
        raise MissingFileError(f"Required file not found: {path}")
    return p.read_text(encoding="utf-8")


def read_file(path: str | Path) -> str:
    """Driver-friendly wrapper that matches previous behavior: raises GWError on read failure."""
    p = Path(path)
    if not p.exists():
        raise MissingFileError(f"Required file not found: {path}")
    try:
        return p.read_text(encoding="utf-8")
    except Exception as e:
        raise GWError(f"Unable to read file {path}: {e}")


def to_text(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2, default=str)
    except Exception:
        return str(obj)


def load_env() -> None:
    """Load .env if present; prefer .env values to override shell/session leftovers."""
    try:
        load_dotenv(override=True)
    except Exception:
        # Best-effort; ignore if dotenv is unavailable
        pass


def _norm_token(s: Any) -> str:
    """Normalize a token for case-insensitive, quote-insensitive comparisons.
    - Cast to str, strip whitespace
    - Remove symmetrical leading/trailing single or double quotes
    - Lowercase
    """
    try:
        x = str(s).strip()
        if (x.startswith('"') and x.endswith('"')) or (x.startswith("'") and x.endswith("'")):
            x = x[1:-1].strip()
        return x.lower()
    except Exception:
        return str(s).lower() if s is not None else ""


def _extract_numeric_hint(text: str, key: str, default: float) -> float:
    """Extract a numeric hint from free text.
    Looks for lines like 'temperature_hint: 0.25' or 'max_tokens_line = 90'.
    Returns the parsed float if found, otherwise the provided default.
    """
    import re
    try:
        m = re.search(rf"{re.escape(key)}\s*[:=]\s*([0-9]*\.?[0-9]+)", str(text))
        if not m:
            return float(default)
        return float(m.group(1))
    except Exception:
        return float(default)
