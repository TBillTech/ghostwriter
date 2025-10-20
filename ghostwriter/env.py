"""Environment helpers for GhostWriter.

Centralizes reading environment variables, resolving model/token settings,
masking secrets for logging, normalizing base URLs, and capturing a
program environment snapshot for diagnostics.
"""
from __future__ import annotations

import os
import re
from typing import Optional, Dict, Any, List, Tuple

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
        if model is None:
            model = env_str("GW_MODEL_DIALOG")
        if os.getenv("GW_TEMP_CHARACTER_DIALOG") is None and os.getenv("GW_TEMP_DIALOG") is not None:
            try:
                temp = float(os.getenv("GW_TEMP_DIALOG", str(temp)))
            except Exception:
                pass
        if os.getenv("GW_MAX_TOKENS_CHARACTER_DIALOG") is None and os.getenv("GW_MAX_TOKENS_DIALOG") is not None:
            try:
                v = int(os.getenv("GW_MAX_TOKENS_DIALOG", str(max_tokens)))
                max_tokens = v if v > 0 else max_tokens
            except Exception:
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
