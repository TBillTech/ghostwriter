"""Shared helpers for pipelines.

Contains environment resolution, validation retry, brainstorm utilities, setting
factoid merge helpers, and body-language templating. This allows pipeline logic
to live in this package without depending on scripts/driver.py internals.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from ..templates import apply_template, prompt_key_from_filename
from ..env import env_for as _env_for, env_for_prompt as _env_for_prompt, reasoning_for_prompt as _reasoning_for_prompt
from ..validation import validate_text, validate_bullet_list, validate_actor_list
from ..llm import complete as llm_complete
from ..logging import log_run as _log_run
from ..utils import to_text
from ..characters import load_characters_list

__all__ = [
    "apply_template",
    "prompt_key_from_filename",
    "validate_text",
    "validate_bullet_list",
    "validate_actor_list",
    "llm_complete",
    # Helpers
    "env_for",  # provided as thin wrappers below
    "env_for_prompt",
    "reasoning_for_prompt",
    "llm_call_with_validation",
    "extract_bullet_contents",
    "rebuild_bullets",
    "filter_narration_brainstorm",
    "truncate_brainstorm",
    "brainstorm_has_done",
    "strip_trailing_done",
    "factoids_block_from_setting",
    "selected_factoid_names_from_sources",
    "merge_setting_with_factoids",
    "build_pipeline_replacements",
    "apply_body_template",
]


# ---------------------------
# Environment helpers (thin wrappers around central env module)
# ---------------------------

def env_for(step_key: str, *, default_temp: float = 0.2, default_max_tokens: int = 800) -> Tuple[Optional[str], float, int]:
    return _env_for(step_key, default_temp=default_temp, default_max_tokens=default_max_tokens)


def env_for_prompt(template_filename: str, fallback_step_key: str, *, default_temp: float, default_max_tokens: int) -> Tuple[Optional[str], float, int]:
    return _env_for_prompt(template_filename, fallback_step_key, default_temp=default_temp, default_max_tokens=default_max_tokens)


def reasoning_for_prompt(template_filename: str, fallback_step_key: str) -> Optional[str]:
    return _reasoning_for_prompt(template_filename, fallback_step_key)


# ---------------------------
# LLM call with validation and logging
# ---------------------------

def llm_call_with_validation(
    system: str,
    user: str,
    *,
    model: Optional[str],
    temperature: float,
    max_tokens: int,
    validator,
    reasoning_effort: Optional[str] = None,
    log_maker: Optional[Callable[[int], Optional[Path]]] = None,
    context_tag: Optional[str] = None,
) -> str:
    last_reason = ""
    for attempt in range(1, 4):
        # Increase token budget on retries to mitigate truncation/empty responses
        factor = 1.0 if attempt == 1 else (1.5 if attempt == 2 else 2.0)
        max_toks_try = int(max_tokens * factor)
        try:
            if context_tag:
                _log_run(f"LLM ctx | {context_tag}")
        except Exception:
            pass
        out = llm_complete(user, system=system, temperature=temperature, max_tokens=max_toks_try, model=model, reasoning_effort=reasoning_effort)
        log_path = log_maker(attempt) if log_maker else None
        if log_path is not None:
            try:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("w", encoding="utf-8") as f:
                    f.write("=== SYSTEM ===\n" + (system or "") + "\n\n")
                    f.write("=== USER ===\n" + user + "\n\n")
                    f.write("=== RESPONSE ===\n" + out + "\n")
            except Exception:
                pass
        ok, reason = validator(out)
        if ok:
            return out
        last_reason = reason
    raise ValueError(f"Output validation failed after 3 attempts: {last_reason}")


# ---------------------------
# Brainstorm helpers
# ---------------------------

def extract_bullet_contents(text: str) -> List[str]:
    contents: List[str] = []
    for ln in text.splitlines():
        s = ln.lstrip()
        if not s or s.startswith('#'):
            continue
        if s.startswith(('*', '-')):
            after = s[1:].lstrip()
            if after:
                contents.append(after)
    return contents


def rebuild_bullets(contents: List[str]) -> str:
    return "\n".join([f"* {c}" for c in contents])


def filter_narration_brainstorm(text: str, min_chars: int = 24) -> str:
    orig = extract_bullet_contents(text)
    if not orig:
        return text
    filtered = [c for c in orig if len(c.strip()) >= min_chars]
    if len(filtered) < 2:
        sorted_by_len = sorted(orig, key=lambda c: len(c.strip()), reverse=True)
        filtered = sorted_by_len[: max(2, len(sorted_by_len))]
        filtered = filtered[:2]
    return rebuild_bullets(filtered)


def truncate_brainstorm(text: str, limit: int = 10) -> str:
    contents = extract_bullet_contents(text)
    if not contents:
        return text
    return rebuild_bullets(contents[:limit])


def brainstorm_has_done(text: str) -> bool:
    if text is None:
        return False
    try:
        lines = [ln.strip() for ln in str(text).splitlines() if ln.strip()]
        if not lines:
            return False
        last = lines[-1].strip().upper()
        # Accept DONE optionally followed by punctuation/spaces
        import re as _re
        return bool(_re.match(r"^DONE(?:\s*[\.!\-—…]*)?$", last))
    except Exception:
        return False


def strip_trailing_done(text: str) -> str:
    if text is None:
        return ""
    try:
        lines = text.splitlines()
        while lines and not lines[-1].strip():
            lines.pop()
        if lines:
            last = lines[-1].strip()
            import re as _re
            if _re.match(r"^(?i:done)(?:\s*[\.!\-—…]*)?$", last):
                lines.pop()
        return "\n".join(lines).rstrip() + ("\n" if lines else "")
    except Exception:
        return str(text)


# ---------------------------
# Setting Factoids merge
# ---------------------------

def factoids_block_from_setting(setting: dict, selected_names: Optional[Iterable[str]] = None) -> str:
    try:
        if not isinstance(setting, dict):
            return ""
        facts = setting.get("Factoids")
        if not isinstance(facts, list):
            return ""
        selected: Optional[set] = None
        if selected_names is not None:
            try:
                selected = {str(n).strip().lower() for n in selected_names if str(n).strip()}
            except Exception:
                selected = None
        items = []
        for f in facts:
            if not isinstance(f, dict):
                continue
            name = f.get("name")
            desc = f.get("description")
            if not (isinstance(name, str) and name.strip()):
                continue
            if selected and str(name).strip().lower() not in selected:
                continue
            items.append({
                "name": str(name).strip(),
                "description": str(desc).strip() if isinstance(desc, str) else "",
            })
        if not items:
            return ""
        return to_text({"Factoids": items})
    except Exception:
        return ""


def selected_factoid_names_from_sources(chapter: dict, setting_block_text: str) -> Optional[Iterable[str]]:
    try:
        ch_setting = chapter.get("setting") if isinstance(chapter, dict) else None
        if isinstance(ch_setting, dict) and isinstance(ch_setting.get("factoids"), list):
            names = [str(x) for x in ch_setting.get("factoids") if isinstance(x, (str, int, float))]
            if names:
                return names
    except Exception:
        pass
    try:
        text = (setting_block_text or "").strip()
        if text:
            try:
                obj = json.loads(text)
            except Exception:
                import ast
                obj = ast.literal_eval(text)
            if isinstance(obj, dict):
                f = obj.get("factoids")
                if isinstance(f, list):
                    names = [str(x) for x in f if isinstance(x, (str, int, float))]
                    if names:
                        return names
    except Exception:
        pass
    return None


def merge_setting_with_factoids(setting_block_text: str, setting: dict, chapter: Optional[dict] = None) -> str:
    try:
        selected = None
        if chapter is not None:
            selected = selected_factoid_names_from_sources(chapter, setting_block_text)
        facts_text = factoids_block_from_setting(setting, selected_names=selected)
        base = (setting_block_text or "").strip()
        if base and facts_text:
            return base + "\n\n" + facts_text
        return base or facts_text or ""
    except Exception:
        return setting_block_text or ""


# ---------------------------
# Replacement builder
# ---------------------------

def build_pipeline_replacements(setting: dict, chapter: dict, chapter_id: str, version: int, tp: Dict[str, str], state: Any, *, prior_paragraph: str = "", ctx: Any = None) -> Dict[str, str]:
    # Build the common replacements via legacy driver to preserve behavior
    from ..commands import _get_driver_module  # lazy load to avoid cycles
    drv = _get_driver_module()
    reps = drv.build_common_replacements(setting, chapter, chapter_id, version)
    touch_text = tp.get("content", "")
    tp_type = tp.get("type", "")
    extra = {
        "[TOUCH_POINT]": touch_text,
        "[touch_point]": touch_text,
        "[TOUCH-POINT]": touch_text,
        "[touch-point]": touch_text,
        "[TOUCHPOINT]": touch_text,
        "[touchpoint]": touch_text,
        "[TOUCH_POINT_TYPE]": tp_type,
        "[ACTIVE_ACTORS]": ", ".join(getattr(state, "active_actors", []) or []),
        "[actors]": ", ".join(getattr(state, "active_actors", []) or []),
        "[SCENE]": getattr(state, "current_scene", None) or "",
        "[scene]": getattr(state, "current_scene", None) or "",
        "[FORESHADOWING]": ", ".join(getattr(state, "foreshadowing", []) or []),
        "[PRIOR_PARAGRAPH]": prior_paragraph or "",
        "[prior_paragraph]": prior_paragraph or "",
    }
    # Setting block + Factoids merge (filtered by chapter selection)
    state_setting_block = getattr(state, "setting_block", "")
    if state_setting_block:
        extra["[SETTING]"] = merge_setting_with_factoids(state_setting_block, setting, chapter=chapter)
    else:
        extra["[SETTING]"] = merge_setting_with_factoids("", setting, chapter=chapter)

    # Characters block: selected block or subset of CHARACTERS limited to active actors
    chars_block = getattr(state, "characters_block", "")
    if not chars_block and getattr(state, "active_actors", None):
        try:
            sel_chars = []
            all_chars = load_characters_list(ctx)
            if isinstance(all_chars, list) and all_chars:
                wanted = {a.strip().lower() for a in state.active_actors if a.strip()}
                for ch in all_chars:
                    cid = str(ch.get("id", "")).strip().lower()
                    cname = str(ch.get("name", "")).strip().lower()
                    if cid in wanted or cname in wanted:
                        sel_chars.append(ch)
            if sel_chars:
                chars_block = to_text({"Selected-Characters": sel_chars})
        except Exception:
            chars_block = chars_block or ""
    extra["[CHARACTERS]"] = chars_block or ""

    # STATE_CHARACTERS — strictly active actors subset
    state_chars_block = ""
    try:
        sel_state_chars = []
        candidates = load_characters_list(ctx)
        if isinstance(candidates, list) and getattr(state, "active_actors", None):
            wanted2 = {a.strip().lower() for a in state.active_actors if a.strip()}
            for ch in candidates:
                cid = str(ch.get("id", "")).strip().lower()
                cname = str(ch.get("name", "")).strip().lower()
                if cid in wanted2 or cname in wanted2:
                    sel_state_chars.append(ch)
        if sel_state_chars:
            state_chars_block = to_text({"Selected-Characters": sel_state_chars})
    except Exception:
        state_chars_block = ""
    extra["[STATE_CHARACTERS]"] = state_chars_block
    extra["[state_characters]"] = state_chars_block

    # Dialog history
    try:
        dialog_map = {a: state.recent_dialog(a) for a in (getattr(state, "active_actors", []) or [])}
        extra["[DIALOG_HISTORY]"] = to_text(dialog_map)
    except Exception:
        extra["[DIALOG_HISTORY]"] = ""
    reps.update(extra)
    return reps


# ---------------------------
# Body-language templating
# ---------------------------

def _inline_body_with_dialog(body_bullet: str, dialog: str) -> str:
    if not body_bullet:
        return dialog.strip()
    body = body_bullet.strip()
    if body.startswith(('*', '-')):
        body = body[1:].lstrip()
    import re as _re
    if not _re.search(r"[\.,?!:;—-]\s*$", body):
        body = body + ","
    return f"{body} {dialog.strip()}".strip()


def apply_body_template(body_bullet: str, dialog: str) -> str:
    if not body_bullet:
        return (dialog or "").strip()
    body = body_bullet.strip()
    if body.startswith(('*', '-')):
        body = body[1:].lstrip()

    d = (dialog or "").strip()
    if len(d) >= 2 and ((d[0] == d[-1]) and d[0] in ('"', "'", '“', '”', '‘', '’')):
        d = d[1:-1].strip()

    patterns = [
        r'"([^"\\]*(?:\\.[^"\\]*)*)',
        r"'([^'\\]*(?:\\.[^'\\]*)*)",
        r'“([^”]*)”',
        r'‘([^’]*)’',
    ]

    earliest = None
    import re as _re
    for pat in patterns:
        for m in _re.finditer(pat, body):
            if earliest is None or m.start() < earliest.start():
                earliest = m
            break

    if earliest is None:
        return _inline_body_with_dialog(body_bullet, dialog)

    start, end = earliest.start(0), earliest.end(0)
    quote_open = body[start]
    quote_close = body[end-1]
    before = body[:start]
    after = body[end:]
    replaced = f"{before}{quote_open}{d}{quote_close}{after}"
    return replaced.strip()

