"""Character helpers: access list from RunContext and rendering utilities."""
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Optional, Tuple
import os
import re

from .context import RunContext
from .utils import to_text, read_text, _extract_numeric_hint
from .llm import complete
from .validation import validate_text
from .env import env_for as _env_for


def load_characters_list(ctx: Optional[RunContext]) -> List[dict]:
    if ctx is None:
        return []
    chars = ctx.characters
    if isinstance(chars, list):
        return [c for c in chars if isinstance(c, dict)]
    return []


def render_character_dialog(
    *,
    character_id: str,
    template_path: str | Path | None,
    character_yaml_text: str,
    agenda_text: str,
    dialog_context: List[str],
    temperature: float = 0.3,
    max_tokens: int = 120,
    log_file: Optional[Path] = None,
) -> str:
    """Render a single character dialog line using the character dialog template.

    This is a minimal adapter; scripts/driver.py has a more advanced implementation.
    """
    # Load template or fallback
    default_template = (
        "<id/>\n"
        "You are role playing/acting out the following character:\n"
        "<character_yaml/>\n"
        "You are aware of or deeply care about the following details:\n"
        "<agenda/>\n"
        "The last N lines of dialog are:\n"
        "<dialog>N</dialog>\n"
        "The director now expects you to say something that matches your character, and he gives you this prompt:\n"
        "<prompt/>\n"
    )
    try:
        user = read_text(template_path) if template_path else default_template
    except Exception:
        user = default_template

    # Simple substitutions
    user = user.replace("<id/>", f"<id>{character_id}</id>")
    user = user.replace("<character_yaml/>", character_yaml_text or "")
    user = user.replace("<agenda/>", agenda_text or "")
    # Replace a single <dialog>N</dialog>
    if "<dialog>" in user:
        tail = "\n".join([ln for ln in dialog_context if ln.strip()][-8:])
        user = user.replace("<dialog>N</dialog>", tail)
    # Replace <prompt/> with agenda as a fallback prompt (call sites will override in pipelines)
    if "<prompt/>" in user:
        user = user.replace("<prompt/>", agenda_text or "Respond in character.")

    system = (
        f"You are the character with id '{character_id}'. Return only dialog or inner monologue."
    )
    resp = complete(user, system=system, temperature=temperature, max_tokens=max_tokens)
    # Optional logging
    if log_file is not None:
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with log_file.open("w", encoding="utf-8") as f:
                f.write("=== SYSTEM ===\n" + system + "\n\n")
                f.write("=== USER ===\n" + user + "\n\n")
                f.write("=== RESPONSE ===\n" + (resp or "") + "\n")
        except Exception:
            pass
    return resp


# Advanced CHARACTER template parsing and rendering (migrated from driver)

CHAR_TEMPLATE_RE = re.compile(
    r"<CHARACTER TEMPLATE>\s*\n\s*<id>(?P<id>[^<]+)</id>\s*\n(?P<body>.*?)\n\s*</CHARACTER TEMPLATE>",
    re.DOTALL | re.IGNORECASE,
)

CHAR_CALL_RE = re.compile(
    r"<CHARACTER>\s*<id>(?P<id>[^<]+)</id>\s*(?:<agenda>(?P<agenda>.*?)</agenda>\s*)?(?:<dialog>(?P<dialogn>\d+)</dialog>\s*)?<prompt>(?P<prompt>.*?)</prompt>\s*</CHARACTER>",
    re.DOTALL | re.IGNORECASE,
)


def parse_character_blocks(pre_draft_text: str) -> Tuple[Dict[str, str], List[Dict[str, str]]]:
    templates: Dict[str, str] = {}
    for m in CHAR_TEMPLATE_RE.finditer(pre_draft_text):
        templates[m.group("id").strip()] = m.group("body").strip()
    calls: List[Dict[str, str]] = []
    for m in CHAR_CALL_RE.finditer(pre_draft_text):
        calls.append({
            "id": m.group("id").strip(),
            "prompt": m.group("prompt").strip(),
            "agenda": (m.group("agenda") or "").strip() if "agenda" in m.groupdict() else "",
            "dialogn": int(m.group("dialogn")) if m.groupdict().get("dialogn") and m.group("dialogn") else None,
            "full_match": m.group(0),
        })
    return templates, calls


def _log_warning(msg: str, log_dir: Optional[Path]) -> None:
    try:
        print(f"WARNING: {msg}")
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            with (log_dir / "warnings.txt").open("a", encoding="utf-8") as f:
                f.write(msg + "\n")
    except Exception:
        pass


# numeric hint extractor moved to utils._extract_numeric_hint


def render_character_call(
    character_id: str,
    call_prompt: str,
    dialog_lines: List[str],
    *,
    temperature: float = 0.3,
    max_tokens_line: int = 100,
    log_file: Optional[Path] = None,
    agenda: str = "",
    character_yaml: Optional[str] = None,
    dialog_n_override: Optional[int] = None,
) -> str:
    # Load external character dialog template if available
    template_path = Path("prompts") / "character_dialog_prompt.md"
    default_template = (
        "<id/>\n"
        "You are role playing/acting out the following character:\n"
        "<character_yaml/>\n"
        "You are aware of or deeply care about the following details:\n"
        "<agenda/>\n"
        "The last N lines of dialog are:\n"
        "<dialog>N</dialog>\n"
        "The director now expects you to say something that matches your character, and he gives you this prompt:\n"
        "<prompt/>\n"
    )
    try:
        user = read_text(template_path) if template_path.exists() else default_template
    except Exception:
        user = default_template

    # Warn if missing YAML
    try:
        warn_dir = log_file.parent if log_file is not None else None
        if not (character_yaml or "").strip():
            _log_warning(f"CHARACTER: Missing character_yaml for id '{character_id}'.", warn_dir)
    except Exception:
        pass

    # Substitute <character_yaml/>, <id/>, <agenda/>
    if "<character_yaml/>" in user:
        user = user.replace("<character_yaml/>", character_yaml or "")
    if "<id/>" in user:
        user = user.replace("<id/>", f"<id>{character_id}</id>")
    user = user.replace("<agenda/>", agenda or "")

    # Determine N for dialog context
    try:
        env_default_n = int(os.getenv("GW_DIALOG_CONTEXT_LINES", "8"))
    except Exception:
        env_default_n = 8
    chosen_n = dialog_n_override if (isinstance(dialog_n_override, int) and dialog_n_override > 0) else None
    if chosen_n is None:
        m_first = re.search(r"<dialog>\s*(\d+)\s*</dialog>", user)
        if m_first:
            try:
                chosen_n = int(m_first.group(1))
            except Exception:
                chosen_n = None
    if chosen_n is None:
        chosen_n = env_default_n

    def _last_n_lines(lines: List[str], n: int) -> str:
        if n <= 0:
            return ""
        tail = [ln for ln in lines if ln.strip()]
        chosen = tail[-n:] if tail else lines[-n:]
        return "\n".join(chosen)

    # Replace dialog placeholders
    user = re.sub(r"The last\s+N\s+lines of dialog", f"The last {chosen_n} lines of dialog", user)
    def _repl_dialog(m: re.Match) -> str:
        token = (m.group(1) or "").strip()
        if token.lower() == "n":
            n = chosen_n
        else:
            try:
                n = int(token)
            except Exception:
                n = 0
        return _last_n_lines(dialog_lines, n)  # type: ignore[arg-type]
    user = re.sub(r"<dialog>\s*(\d+|[Nn])\s*</dialog>", _repl_dialog, user)

    # Substitute <prompt/>
    user = user.replace("<prompt/>", call_prompt)

    system = (
        f"You are the character with id '{character_id}'. "
        f"Return only the character's dialog or inner monologue; no notes or brackets."
    )
    dialog_model, dialog_temp_env, dialog_max_tokens = _env_for("CHARACTER_DIALOG", default_temp=temperature if temperature is not None else 0.3, default_max_tokens=max_tokens_line)
    effective_temp = temperature if temperature is not None else dialog_temp_env

    last_reason = ""
    for attempt in range(1, 4):
        factor = 1.0 if attempt == 1 else (1.5 if attempt == 2 else 2.0)
        try_max = int(dialog_max_tokens * factor)
        response = complete(
            user,
            system=system,
            temperature=effective_temp,
            max_tokens=try_max,
            model=dialog_model,
        )
        # Attempt-specific logging
        if log_file is not None:
            try:
                log_file.parent.mkdir(parents=True, exist_ok=True)
                lf = log_file
                if attempt > 1:
                    # Insert _rN before suffix
                    try:
                        suffix = log_file.suffix
                        stem = log_file.stem
                        lf = log_file.with_name(f"{stem}_r{attempt}{suffix}")
                    except Exception:
                        lf = log_file
                with lf.open("w", encoding="utf-8") as f:
                    f.write("=== SYSTEM ===\n")
                    f.write(system)
                    f.write("\n\n=== USER ===\n")
                    f.write(user)
                    f.write("\n\n=== RESPONSE ===\n")
                    f.write(response or "")
                    f.write("\n")
            except Exception:
                pass
        ok, reason = validate_text(response)
        if ok:
            return response
        last_reason = reason
    raise ValueError(f"Character dialog generation failed after 3 attempts: {last_reason}")


def substitute_character_calls(
    pre_draft_text: str,
    *,
    log_dir: Optional[Path] = None,
    context_before_chars: int = 800,
    context_after_chars: int = 400,
    ctx: Optional[RunContext] = None,
) -> Tuple[str, Dict[str, int]]:
    result = pre_draft_text
    templates, _ = parse_character_blocks(result)
    stats = {"templates": len(templates), "calls": 0, "missing_templates": 0}
    call_index = 0

    # Preload character YAML and hints once
    char_yaml_by_id: Dict[str, str] = {}
    char_hints_by_id: Dict[str, Tuple[float, int]] = {}
    try:
        chars_list = load_characters_list(ctx)
        if isinstance(chars_list, list) and chars_list:
            for ch in chars_list:
                cid = str(ch.get("id", "")).strip()
                if not cid:
                    continue
                try:
                    char_yaml_by_id[cid.lower()] = to_text(ch)
                    temp = float(ch.get("temperature_hint", 0.3))
                    max_toks = int(ch.get("max_tokens_line", 100))
                    char_hints_by_id[cid.lower()] = (temp, max_toks)
                except Exception:
                    continue
    except Exception:
        pass

    while True:
        m = CHAR_CALL_RE.search(result)
        if not m:
            break
        call_index += 1
        stats["calls"] += 1
        cid = m.group("id").strip()
        call_prompt = m.group("prompt").strip()
        full_match = m.group(0)
        start, end = m.start(), m.end()

        tpl = templates.get(cid)
        if tpl:
            temp = _extract_numeric_hint(tpl, "temperature_hint", default=0.3)
            max_tokens_line = int(_extract_numeric_hint(tpl, "max_tokens_line", default=100))
        else:
            stats["missing_templates"] += 1
            temp, max_tokens_line = char_hints_by_id.get(cid.lower(), (0.3, 100))

        ctx_before_start = max(0, start - context_before_chars)
        ctx_after_end = min(len(result), end + context_after_chars)
        context_before = result[ctx_before_start:start]
        dialog_lines = context_before.splitlines()

        log_file = None
        if log_dir is not None:
            safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", cid) or "character"
            log_file = log_dir / f"{call_index:02d}_{safe_id}.txt"

        replacement = render_character_call(
            cid,
            call_prompt,
            dialog_lines,  # type: ignore[arg-type]
            temperature=temp,
            max_tokens_line=max_tokens_line,
            log_file=log_file,
            agenda=(m.group("agenda") or "") if "agenda" in m.re.groupindex else "",
            character_yaml=char_yaml_by_id.get(cid.lower()),
            dialog_n_override=(int(m.group("dialogn")) if ("dialogn" in m.re.groupindex and m.group("dialogn")) else None),
        )

        result = result[:start] + replacement + result[end:]

    return result, stats
