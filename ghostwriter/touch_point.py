"""Touch-point level helpers and utilities.

This module contains functions that operate on a single touch-point or
are used during touch-point pipeline steps.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable, Any, Callable
import os
import re
import json

from .context import GWError
from .utils import to_text as _gw_to_text, read_file, save_text
from .templates import apply_template, build_common_replacements, prompt_key_from_filename
from .env import env_for as _env_for_env, env_for_prompt as _env_for_prompt_env
from .openai import llm_complete
from .characters import load_characters_list
from .factoids import merge_setting_with_factoids as _merge_setting_with_factoids


class ValidationError(GWError):
    pass


# ---- Output validators

def validate_text(output: str) -> Tuple[bool, str]:
    return (True, "ok")


_ACTOR_LINE_RE = re.compile(r"^\s*([A-Za-z0-9_.\-]+)\s*:\s+.+")


def validate_bullet_list(output: str) -> Tuple[bool, str]:
    lines = [ln.rstrip() for ln in output.splitlines() if ln.strip() and not ln.lstrip().startswith('#')]
    bullets = [ln for ln in lines if ln.lstrip().startswith("*")]
    if len(bullets) >= 2:
        return True, "ok"
    return False, "Expected a bullet list with at least 2 lines starting with '*'"


def validate_actor_list(output: str) -> Tuple[bool, str]:
    lines = [ln for ln in output.splitlines() if ln.strip() and not ln.lstrip().startswith('#')]
    count = 0
    for ln in lines:
        if _ACTOR_LINE_RE.match(ln):
            count += 1
    if count >= 2:
        return True, "ok"
    return False, "Expected an actor list with at least 2 actor-attributed lines like 'id: ...'"


def validate_agenda_list(output: str) -> Tuple[bool, str]:
    lines = [ln for ln in output.splitlines() if ln.strip() and not ln.lstrip().startswith('#')]
    bullets = [ln for ln in lines if ln.lstrip().startswith(('*', '-'))]
    if len(bullets) >= 2:
        return True, "ok"
    colon = [ln for ln in lines if re.search(r"\w\s*:\s+.+", ln)]
    if len(colon) >= 2:
        return True, "ok"
    return False, "Expected an agenda list (>=2 items as bullets or 'key: value' lines)"


def _retry_with_validation(callable_fn, validator, *, max_attempts: int = 3) -> str:
    last_err = ""
    for attempt in range(1, max_attempts + 1):
        out = callable_fn()
        ok, reason = validator(out)
        if ok:
            return out
        last_err = reason
    raise ValidationError(f"Output validation failed after {max_attempts} attempts: {last_err}")


def _extract_bullet_contents(text: str) -> List[str]:
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


def _rebuild_bullets(contents: List[str]) -> str:
    return "\n".join([f"* {c}" for c in contents])


def _filter_narration_brainstorm(text: str, min_chars: int = 24) -> str:
    orig = _extract_bullet_contents(text)
    if not orig:
        return text
    filtered = [c for c in orig if len(c.strip()) >= min_chars]
    if len(filtered) < 2:
        sorted_by_len = sorted(orig, key=lambda c: len(c.strip()), reverse=True)
        filtered = sorted_by_len[: max(2, len(sorted_by_len))]
        filtered = filtered[:2]
    return _rebuild_bullets(filtered)


def _truncate_brainstorm(text: str, limit: int = 10) -> str:
    contents = _extract_bullet_contents(text)
    if not contents:
        return text
    return _rebuild_bullets(contents[:limit])


def _brainstorm_has_done(text: str) -> bool:
    if text is None:
        return False
    try:
        lines = [ln.strip() for ln in str(text).splitlines() if ln.strip()]
        if not lines:
            return False
        return lines[-1].strip().upper() == "DONE"
    except Exception:
        return False


def _strip_trailing_done(text: str) -> str:
    if text is None:
        return ""
    try:
        lines = text.splitlines()
        while lines and not lines[-1].strip():
            lines.pop()
        if lines and lines[-1].strip().upper() == "DONE":
            lines.pop()
        return "\n".join(lines).rstrip() + ("\n" if lines else "")
    except Exception:
        return str(text)


def _ensure_brainstorm_done_on_resume(tp_type: str, reps: Dict[str, str], *, log_dir: Optional[Path], tp_index: int) -> None:
    if tp_type not in ("narration", "explicit", "implicit"):
        return
    if log_dir is None:
        return
    bs_path = log_dir / "brainstorm.txt"
    try:
        existing_bs = bs_path.read_text(encoding="utf-8") if bs_path.exists() else ""
    except Exception:
        existing_bs = ""
    if existing_bs and _brainstorm_has_done(existing_bs):
        return
    if tp_type == "narration":
        sys1 = "You are brainstorming narrative beats as concise bullet points."
        tpl1 = "narration_brain_storm_prompt.md"
    elif tp_type == "explicit":
        sys1 = "Brainstorm explicit dialog beats as bullet points."
        tpl1 = "explicit_brain_storm_prompt.md"
    else:
        sys1 = "Brainstorm implicit dialog beats (indirect, subtext) as bullet points."
        tpl1 = "implicit_brain_storm_prompt.md"
    user1_base = _apply_step(tpl1, reps)
    model, temp, max_toks = _env_for_prompt(tpl1, "BRAIN_STORM", default_temp=0.45 if tp_type != "narration" else 0.4, default_max_tokens=600)
    seed_bullets = _strip_trailing_done(existing_bs)
    user1 = user1_base
    if seed_bullets.strip():
        user1 = user1 + "\n\n" + seed_bullets.strip() + "\n"
    brainstorm_raw = _llm_call_with_validation(
        sys1,
        user1,
        model=model,
        temperature=temp,
        max_tokens=max_toks,
        validator=validate_bullet_list,
        log_maker=(lambda attempt: (log_dir / f"{tp_index:02d}_brainstorm_resume{'_r'+str(attempt) if attempt>1 else ''}.txt")),
    )
    try:
        combined = (seed_bullets.strip() + ("\n" if seed_bullets.strip() and not seed_bullets.strip().endswith("\n") else "")) + brainstorm_raw
        bs_path.write_text(combined, encoding="utf-8")
    except Exception:
        pass
    if not _brainstorm_has_done(brainstorm_raw):
        print("Brainstorming still in progress.")
        import sys as _sys
        _sys.exit(0)


def _inline_body_with_dialog(body_bullet: str, dialog: str) -> str:
    if not body_bullet:
        return dialog.strip()
    body = body_bullet.strip()
    if body.startswith(('*', '-')):
        body = body[1:].lstrip()
    if not re.search(r"[\.,?!:;—-]\s*$", body):
        body = body + ","
    return f"{body} {dialog.strip()}".strip()


def _apply_body_template(body_bullet: str, dialog: str) -> str:
    if not body_bullet:
        return (dialog or "").strip()
    body = body_bullet.strip()
    if body.startswith(('*', '-')):
        body = body[1:].lstrip()
    d = (dialog or "").strip()
    if len(d) >= 2 and ((d[0] == d[-1]) and d[0] in ('"', "'", '“', '”', '‘', '’')):
        d = d[1:-1].strip()
    patterns = [
        r'"([^"\\]*(?:\\.[^"\\]*)*)"',
        r"'([^'\\]*(?:\\.[^'\\]*)*)'",
        r'“([^”]*)”',
        r'‘([^’]*)’',
    ]
    earliest = None
    for pat in patterns:
        for m in re.finditer(pat, body):
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


def _parse_agenda_by_actor(text: str) -> Dict[str, str]:
    by_actor: Dict[str, List[str]] = {}
    current: Optional[str] = None
    for ln in text.splitlines():
        if not ln.strip():
            continue
        if not ln.lstrip().startswith(('*', '-')) and re.match(r"^\s*([A-Za-z0-9_.\-]+)\s*:\s*$", ln):
            m = re.match(r"^\s*([A-Za-z0-9_.\-]+)\s*:\s*$", ln)
            if m:
                current = m.group(1).strip()
                by_actor.setdefault(current, [])
            continue
        if current is not None and ln.lstrip().startswith(('*', '-')):
            content = ln.lstrip()[1:].lstrip()
            if content:
                by_actor[current].append(content)
    return {aid: _rebuild_bullets(items) for aid, items in by_actor.items() if items}


def _pick_bullet_by_index(text: str, index1: int) -> str:
    if index1 <= 0:
        return ""
    items = _extract_bullet_contents(text)
    if 1 <= index1 <= len(items):
        return f"* {items[index1-1]}"
    return ""


def _build_pipeline_replacements(setting: dict, chapter: dict, chapter_id: str, version: int, tp: Dict[str, str], state, *, prior_paragraph: str = "", ctx=None) -> Dict[str, str]:
    reps = build_common_replacements(setting, chapter, chapter_id, version)
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
        "[ACTIVE_ACTORS]": ", ".join(state.active_actors),
        "[actors]": ", ".join(state.active_actors),
        "[SCENE]": state.current_scene or "",
        "[scene]": state.current_scene or "",
        "[FORESHADOWING]": ", ".join(state.foreshadowing),
        "[PRIOR_PARAGRAPH]": prior_paragraph or "",
        "[prior_paragraph]": prior_paragraph or "",
    }
    if getattr(state, "setting_block", ""):
        extra["[SETTING]"] = _merge_setting_with_factoids(state.setting_block, setting, chapter=ctx.chapter if ctx else None)
    else:
        extra["[SETTING]"] = _merge_setting_with_factoids("", setting, chapter=ctx.chapter if ctx else None)
    chars_block = getattr(state, "characters_block", "")
    if not chars_block and state.active_actors:
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
                chars_block = _gw_to_text({"Selected-Characters": sel_chars})
        except Exception:
            chars_block = chars_block or ""
    extra["[CHARACTERS]"] = chars_block or ""
    state_chars_block = ""
    try:
        sel_state_chars = []
        parsed_chars_block = None
        if getattr(state, "characters_block", ""):
            try:
                parsed = json.loads(state.characters_block)
                if isinstance(parsed, dict):
                    parsed_chars_block = parsed.get("Selected-Characters")
            except Exception:
                parsed_chars_block = None
        candidates = parsed_chars_block if isinstance(parsed_chars_block, list) else load_characters_list(ctx)
        if isinstance(candidates, list) and state.active_actors:
            wanted2 = {a.strip().lower() for a in state.active_actors if a.strip()}
            for ch in candidates:
                cid = str(ch.get("id", "")).strip().lower()
                cname = str(ch.get("name", "")).strip().lower()
                if cid in wanted2 or cname in wanted2:
                    sel_state_chars.append(ch)
        if sel_state_chars:
            state_chars_block = _gw_to_text({"Selected-Characters": sel_state_chars})
    except Exception:
        state_chars_block = ""
    extra["[STATE_CHARACTERS]"] = state_chars_block
    extra["[state_characters]"] = state_chars_block
    try:
        dialog_map = {a: state.recent_dialog(a) for a in (state.active_actors or [])}
        extra["[DIALOG_HISTORY]"] = _gw_to_text(dialog_map)
    except Exception:
        extra["[DIALOG_HISTORY]"] = ""
    reps.update(extra)
    return reps


def _apply_step(template_filename: str, reps: Dict[str, str]) -> str:
    path = Path("prompts") / template_filename
    if path.exists():
        return apply_template(str(path), reps)
    return (f"Context follows.\n\n{reps.get('[SETTING.yaml]', '')}\n\n{reps.get('[CHAPTER_xx.yaml]', '')}\n\n"
            f"Touch-Point ({reps.get('[TOUCH_POINT_TYPE]', '')}): {reps.get('[TOUCH_POINT]', '')}\nState: actors={reps.get('[ACTIVE_ACTORS]', '')}, scene={reps.get('[SCENE]', '')}\n")


def _env_for(step_key: str, *, default_temp: float = 0.2, default_max_tokens: int = 800) -> Tuple[Optional[str], float, int]:
    return _env_for_env(step_key, default_temp=default_temp, default_max_tokens=default_max_tokens)


def _env_for_prompt(template_filename: str, fallback_step_key: str, *, default_temp: float, default_max_tokens: int) -> Tuple[Optional[str], float, int]:
    return _env_for_prompt_env(template_filename, fallback_step_key, default_temp=default_temp, default_max_tokens=default_max_tokens)


def _llm_call_with_validation(
    system: str,
    user: str,
    *,
    model: Optional[str],
    temperature: float,
    max_tokens: int,
    validator,
    log_maker: Optional[Callable[[int], Optional[Path]]] = None,
) -> str:
    last_reason = ""
    for attempt in range(1, 4):
        out = llm_complete(user, system=system, temperature=temperature, max_tokens=max_tokens, model=model)
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
    raise ValidationError(f"Output validation failed after 3 attempts: {last_reason}")


def _polish_snippet(text: str, setting: dict, chapter: dict, chapter_id: str, version: int) -> str:
    from .templates import build_polish_prompt
    prompt = build_polish_prompt(setting, chapter, chapter_id, version, text)
    model, temp, max_tokens = _env_for_prompt("polish_prose_prompt.md", "POLISH_PROSE", default_temp=0.2, default_max_tokens=2000)
    return llm_complete(prompt, system="You are a ghostwriter polishing and cleaning prose.", temperature=temp, max_tokens=max_tokens, model=model)


def _parse_actor_lines(actor_list_text: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for ln in actor_list_text.splitlines():
        m = _ACTOR_LINE_RE.match(ln)
        if not m:
            continue
        actor_id = m.group(1)
        _, _, rest = ln.partition(":")
        pairs.append((actor_id.strip(), rest.strip()))
    return pairs
