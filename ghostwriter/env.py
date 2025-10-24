"""Environment helpers for GhostWriter.

Centralizes reading environment variables, resolving model/token settings,
masking secrets for logging, normalizing base URLs, and capturing a
program environment snapshot for diagnostics.
"""
from __future__ import annotations

import os
import re
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

from dotenv import load_dotenv


def load_env() -> None:
    """Load environment variables from a local .env file if present.

    We set override=True so the local .env takes precedence over shell state
    during development, which avoids confusion from lingering env values.
    """
    load_dotenv(override=True)


def env_str(name: str) -> Optional[str]:
    val = os.getenv(name)
    if val is None or str(val).strip() == "":
        return None
    return val


def env_int(name: str, default: int) -> int:
    try:
        v = int(os.getenv(name, str(default)))
        if v <= 0:
            return default
        return v
    except Exception:
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def resolve_temp(step_key: str, default_temp: float) -> float:
    """Resolve temperature with precedence: GW_TEMP_{STEP} -> GW_TEMP_DEFAULT -> default_temp."""
    val = os.getenv(f"GW_TEMP_{step_key}")
    if val is not None and str(val).strip() != "":
        try:
            return float(val)
        except Exception:
            pass
    g = os.getenv("GW_TEMP_DEFAULT")
    if g is not None and str(g).strip() != "":
        try:
            return float(g)
        except Exception:
            pass
    return float(default_temp)


def resolve_max_tokens(step_key: str, default_max_tokens: int) -> int:
    """Resolve max tokens with precedence: GW_MAX_TOKENS_{STEP} -> GW_MAX_TOKENS_DEFAULT -> default."""
    val = os.getenv(f"GW_MAX_TOKENS_{step_key}")
    if val is not None and str(val).strip() != "":
        try:
            v = int(val)
            return v if v > 0 else default_max_tokens
        except Exception:
            pass
    g = os.getenv("GW_MAX_TOKENS_DEFAULT")
    if g is not None and str(g).strip() != "":
        try:
            v = int(g)
            return v if v > 0 else default_max_tokens
        except Exception:
            pass
    return int(default_max_tokens)


# ---------------------------
# Path resolution for configurable book directories (Task 11)
# ---------------------------

def _as_path(val: Optional[str]) -> Optional[Path]:
    if val is None or str(val).strip() == "":
        return None
    p = Path(val)
    return p


def get_book_base_dir() -> Path:
    """Resolve the base working directory for a given book.

    Env: GW_BOOK_BASE_DIR
    Default: current working directory
    """
    base = env_str("GW_BOOK_BASE_DIR")
    if base:
        p = Path(base)
        return p if p.is_absolute() else (Path.cwd() / p)
    # Default to relative path to preserve existing tests that compare Path("iterations")
    return Path(".")


def get_setting_path() -> Path:
    """Resolve SETTING.yaml path.

    Env: GW_SETTING_PATH (relative to base if not absolute)
    Default: <base>/SETTING.yaml
    """
    base = get_book_base_dir()
    p = _as_path(env_str("GW_SETTING_PATH"))
    if p is None:
        return base / "SETTING.yaml"
    return p if p.is_absolute() else (base / p)


def get_characters_path() -> Path:
    """Resolve CHARACTERS.yaml path.

    Env: GW_CHARACTERS_PATH (relative to base if not absolute)
    Default: <base>/CHARACTERS.yaml
    """
    base = get_book_base_dir()
    p = _as_path(env_str("GW_CHARACTERS_PATH"))
    if p is None:
        return base / "CHARACTERS.yaml"
    return p if p.is_absolute() else (base / p)


def get_chapters_dir() -> Path:
    """Resolve chapters directory.

    Env: GW_CHAPTERS_DIR (relative to base if not absolute)
    Default: <base>/chapters
    """
    base = get_book_base_dir()
    p = _as_path(env_str("GW_CHAPTERS_DIR"))
    if p is None:
        return base / "chapters"
    return p if p.is_absolute() else (base / p)


def get_iterations_dir() -> Path:
    """Resolve iterations directory.

    Env: GW_ITERATIONS_DIR (relative to base if not absolute)
    Default: <base>/iterations
    """
    base = get_book_base_dir()
    p = _as_path(env_str("GW_ITERATIONS_DIR"))
    if p is None:
        return base / "iterations"
    return p if p.is_absolute() else (base / p)


def resolve_chapter_path(chapter_path: str) -> Path:
    """Resolve a chapter path, allowing bare filenames or relative names.

    Precedence:
    - If chapter_path exists as given (absolute or relative to cwd), use it
    - Else try <chapters_dir>/<chapter_path>
    - Else try <chapters_dir>/<basename(chapter_path)>
    """
    p = Path(chapter_path)
    if p.exists():
        return p
    ch_dir = get_chapters_dir()
    cand = ch_dir / p
    if cand.exists():
        return cand
    cand2 = ch_dir / p.name
    return cand2


# ---------------------------
# Composite resolvers for steps and prompt templates
# ---------------------------

def env_for(step_key: str, *, default_temp: float = 0.2, default_max_tokens: int = 800) -> Tuple[Optional[str], float, int]:
    """Resolve (model, temperature, max_tokens) for a logical step.

    Precedence:
    - GW_MODEL_{STEP}, GW_TEMP_{STEP}, GW_MAX_TOKENS_{STEP}
    - Dialog-specific fallbacks for CHARACTER_DIALOG: GW_MODEL_DIALOG, GW_TEMP_DIALOG, GW_MAX_TOKENS_DIALOG
    - GW_TEMP_DEFAULT, GW_MAX_TOKENS_DEFAULT
    - Provided defaults
    """
    model = env_str(f"GW_MODEL_{step_key}")
    temp = resolve_temp(step_key, default_temp)
    max_tokens = resolve_max_tokens(step_key, default_max_tokens)
    if step_key == "CHARACTER_DIALOG":
        # No additional aliases; use the CHARACTER_DIALOG keys only
        pass
    return model, float(temp), int(max_tokens)


def env_for_prompt(template_filename: str, fallback_step_key: str, *, default_temp: float, default_max_tokens: int) -> Tuple[Optional[str], float, int]:
    """Resolve env for a specific prompt template with per-prompt overrides.

    Precedence:
    - GW_MODEL_PROMPT_{KEY}, GW_TEMP_PROMPT_{KEY}, GW_MAX_TOKENS_PROMPT_{KEY}
    - Fallback to env_for(fallback_step_key)
    - Provided defaults
    """
    # Lazy import to avoid any chance of circular import
    from .templates import prompt_key_from_filename

    key = prompt_key_from_filename(template_filename)
    model = None
    temp: Optional[float] = None
    max_tokens: Optional[int] = None
    if key:
        model = env_str(f"GW_MODEL_PROMPT_{key}") or None
        t = os.getenv(f"GW_TEMP_PROMPT_{key}")
        if t is not None and str(t).strip() != "":
            try:
                temp = float(t)
            except Exception:
                temp = None
        mt = os.getenv(f"GW_MAX_TOKENS_PROMPT_{key}")
        if mt is not None and str(mt).strip() != "":
            try:
                v = int(mt)
                max_tokens = v if v > 0 else None
            except Exception:
                max_tokens = None
    step_model, step_temp, step_max = env_for(fallback_step_key, default_temp=default_temp, default_max_tokens=default_max_tokens)
    model = model or step_model
    if temp is None:
        temp = step_temp
    if max_tokens is None:
        max_tokens = step_max
    return model, float(temp), int(max_tokens)


def _normalize_reasoning(val: Optional[str]) -> Optional[str]:
    if val is None:
        return None
    v = str(val).strip().lower()
    return v if v in ("low", "medium", "high") else None


def reasoning_for(step_key: str) -> Optional[str]:
    """Resolve reasoning effort with precedence:
    GW_REASONING_EFFORT_{STEP} -> GW_REASONING_EFFORT_DEFAULT -> GW_REASONING_EFFORT
    Returns one of 'low'|'medium'|'high' or None if unset/invalid.
    """
    v = os.getenv(f"GW_REASONING_EFFORT_{step_key}")
    r = _normalize_reasoning(v)
    if r:
        return r
    r = _normalize_reasoning(os.getenv("GW_REASONING_EFFORT_DEFAULT"))
    if r:
        return r
    return _normalize_reasoning(os.getenv("GW_REASONING_EFFORT"))


def reasoning_for_prompt(template_filename: str, fallback_step_key: str) -> Optional[str]:
    """Resolve per-prompt reasoning effort with precedence:
    GW_REASONING_EFFORT_PROMPT_{KEY} -> reasoning_for(fallback_step_key)
    """
    from .templates import prompt_key_from_filename  # lazy import
    key = prompt_key_from_filename(template_filename)
    if key:
        r = _normalize_reasoning(os.getenv(f"GW_REASONING_EFFORT_PROMPT_{key}"))
        if r:
            return r
    return reasoning_for(fallback_step_key)


def mask_env_value(k: str, v: Optional[str]) -> str:
    """Mask secrets in environment values while retaining minimal suffix for debugging.
    Masks keys containing key/secret/token/password regardless of prefix.
    """
    try:
        if v is None:
            return ""
        kl = (k or "").lower()
        if any(s in kl for s in ("key", "secret", "token", "password")):
            s = str(v)
            if len(s) <= 8:
                return "***"
            return ("*" * (len(s) - 4)) + s[-4:]
        return str(v)
    except Exception:
        return ""


def normalize_base_url_from_env() -> Dict[str, str]:
    """Infer the effective base URL or Azure endpoint from environment without creating a client."""
    info: Dict[str, str] = {}
    try:
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("AZURE_OPENAI_API_BASE")
        if azure_endpoint:
            info["azure_endpoint"] = azure_endpoint.strip()
            info["azure_api_version"] = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
            return info
        base_url = (
            os.getenv("OPENAI_BASE_URL")
            or os.getenv("OPENAI_API_BASE")
        )
        if base_url:
            bu = base_url.strip()
            lower = bu.lower()
            is_azure = ("azure.com" in lower) or ("openai.azure" in lower)
            if (not is_azure) and not re.search(r"/v\d+/?$", bu):
                bu = bu.rstrip("/") + "/v1"
            info["base_url"] = bu
        return info
    except Exception:
        return info


def collect_program_env_snapshot(get_model_cb=None, crash_trace_cb=None) -> Dict[str, Any]:
    """Collect program-relevant environment settings for diagnostics.
    Includes GW_*, OPENAI_*, AZURE_OPENAI_* variables with secret masking,
    plus derived fields such as base URL and default model.
    """
    prefixes = ("GW_", "OPENAI_", "AZURE_OPENAI_")
    env_items: List[Tuple[str, str]] = []
    try:
        for k, v in os.environ.items():
            if any(k.startswith(p) for p in prefixes):
                env_items.append((k, mask_env_value(k, v)))
    except Exception:
        pass
    env_items.sort(key=lambda kv: kv[0])
    derived: Dict[str, Any] = {}
    try:
        derived.update(normalize_base_url_from_env())
        if get_model_cb is not None:
            derived["model_default"] = get_model_cb()
        derived["tokens_param_selected"] = (os.getenv("GW_TOKENS_PARAM", "max_tokens").strip() or "max_tokens")
        derived["include_prompt_tokens"] = (os.getenv("GW_INCLUDE_PROMPT_TOKENS", "0") == "1")
        try:
            derived["chars_per_token"] = float(os.getenv("GW_CHARS_PER_TOKEN", "4"))
        except Exception:
            derived["chars_per_token"] = 4.0
        if crash_trace_cb is not None:
            derived["crash_trace_file"] = crash_trace_cb() or ""
        derived["dotenv_override_enabled"] = True
    except Exception:
        pass
    return {
        "env": {k: v for k, v in env_items},
        "derived": derived,
    }
