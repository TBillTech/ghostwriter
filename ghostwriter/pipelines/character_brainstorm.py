"""Character brainstorming pipeline.

Triggers when a chapter's setting references an actor not present in
CHARACTERS.yaml, or when an existing character has `brainstorming: True`.

Artifacts written:
- <base>/character_brainstorm.txt (prompt + response; overwritten each run)
- Appends the LLM result to CHARACTERS.yaml
Stops after updating exactly one character outline.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

from ..context import RunContext
from ..env import (
    get_book_base_dir,
    get_characters_path,
    env_for_prompt,
)
from ..templates import apply_template
from ..utils import to_text as _to_text, read_text as _read_text, save_text as _save_text, _norm_token
from ..llm import complete as llm_complete
from ..factoids import factoids_block_from_setting
from ..context import load_yaml
import yaml


def _selected_characters_block(characters: List[dict], names: List[str], exclude: List[str]) -> str:
    if not characters or not names:
        return ""
    wanted = { _norm_token(n) for n in names if str(n).strip() }
    exclude_set = { _norm_token(e) for e in (exclude or []) }
    sel: List[dict] = []
    for ch in characters:
        cid = _norm_token(ch.get("id", ""))
        cname = _norm_token(ch.get("name", ""))
        if (cid in wanted or cname in wanted) and (cid not in exclude_set and cname not in exclude_set):
            sel.append(ch)
    if not sel:
        return ""
    return _to_text({"Selected-Characters": sel})


def _chapter_yaml_text(ctx: RunContext) -> str:
    # Prefer original YAML text for fidelity
    try:
        from ..env import resolve_chapter_path
        p = resolve_chapter_path(f"{ctx.chapter_id}.yaml")
        if not p.exists():
            # fallback to provided path if any
            p = resolve_chapter_path(f"chapters/{ctx.chapter_id}.yaml")
        return _read_text(p) if p.exists() else _to_text(ctx.chapter)
    except Exception:
        return _to_text(ctx.chapter)


def _read_multiline_input(prompt: str) -> str:
    print(prompt)
    print("End your input with a blank line.")
    lines: List[str] = []
    try:
        while True:
            line = input()
            if not line.strip():
                break
            lines.append(line)
    except EOFError:
        pass
    return "\n".join(lines).strip()


def run_character_brainstorm(*, ctx: RunContext, target_name: str, version_num: int, user_description: Optional[str] = None) -> None:
    chapter = ctx.chapter
    setting = ctx.setting
    characters = ctx.characters

    # Build dereferenced blocks from chapter setting
    chs = chapter.get("setting") if isinstance(chapter, dict) else None
    fact_block = ""
    chars_block = ""
    if isinstance(chs, dict):
        fact_list = chs.get("factoids") if isinstance(chs.get("factoids"), list) else []
        actor_list = chs.get("actors") if isinstance(chs.get("actors"), list) else []
        fact_block = factoids_block_from_setting(setting, selected_names=fact_list)
        chars_block = _selected_characters_block(characters, actor_list, exclude=[target_name])

    # Chapter YAML full text
    ch_yaml_text = _chapter_yaml_text(ctx)

    # If the character exists and has prior data, optionally use that as description
    if user_description is None:
        # Try to find existing character and use its YAML
        existing = None
        for ch in characters or []:
            if _norm_token(ch.get("id")) == _norm_token(target_name) or _norm_token(ch.get("name")) == _norm_token(target_name):
                existing = ch
                break
        if existing is not None and existing.get("brainstorming") is True:
            user_description = _to_text(existing)
        else:
            user_description = _read_multiline_input(
                f"Enter a brief description for character '{target_name}' (traits, cadence, lexicon, mannerisms, etc.):"
            )

    # Prepare example character (Red) if no characters were dereferenced
    example_block = ""
    if not chars_block.strip():
        try:
            repo_root = Path(__file__).resolve().parents[2]
            lrrh_chars = repo_root / "testdata" / "LittleRedRidingHood" / "CHARACTERS.yaml"
            if lrrh_chars.exists():
                y = load_yaml(str(lrrh_chars))
                items = []
                if isinstance(y, list):
                    items = y
                elif isinstance(y, dict) and isinstance(y.get("Characters"), list):
                    items = y.get("Characters")
                red = None
                for it in items:
                    try:
                        if _norm_token(it.get("id")) == "red" or _norm_token(it.get("name")) == "red":
                            red = it
                            break
                    except Exception:
                        continue
                if red is not None:
                    # Dump as a single-item YAML list snippet, then strip leading '- ' later in the template guidance
                    example_block = yaml.safe_dump([red], sort_keys=False, allow_unicode=True)
        except Exception:
            example_block = ""

    # Prepare replacements
    reps: Dict[str, str] = {
        "[TARGET_NAME]": str(target_name),
        "[CHAPTER_YAML]": ch_yaml_text,
        "[SETTING_DEREF]": (fact_block + ("\n\n" if fact_block and chars_block else "") + chars_block).strip(),
        "[USER_DESCRIPTION]": user_description or "",
        "[EXAMPLE_CHARACTER]": example_block or "",
    }

    # Compose prompt and call LLM
    tpl_name = "brainstorm_character_outline.md"
    user_prompt = apply_template(Path("prompts") / tpl_name, reps)
    model, temp, max_tokens = env_for_prompt(tpl_name, "BRAIN_STORM", default_temp=0.45, default_max_tokens=900)
    system = "You brainstorm and draft a single character YAML entry starting with '- id:'."
    out = llm_complete(user_prompt, system=system, temperature=temp, max_tokens=max_tokens, model=model)

    # Save prompt + result to base/character_brainstorm.txt
    try:
        base = get_book_base_dir()
        log_path = base / "character_brainstorm.txt"
        txt_body = (
            "=== SYSTEM ===\n" + (system or "") +
            "\n\n=== USER ===\n" + (user_prompt or "") +
            "\n\n=== RESPONSE ===\n" + (out or "") + "\n"
        )
        _save_text(log_path, txt_body)
    except Exception:
        pass

    # Append result to CHARACTERS.yaml
    try:
        char_path = get_characters_path()
        # Ensure file exists
        char_path.parent.mkdir(parents=True, exist_ok=True)
        if not char_path.exists():
            char_path.write_text("[]\n", encoding="utf-8")
        with char_path.open("a", encoding="utf-8") as f:
            # Ensure a separating newline
            f.write("\n" + (out or "").strip() + "\n")
    except Exception:
        # Best-effort append; if this fails, the run still ends after logging
        pass

    print(f"Brainstormed character outline for '{target_name}' appended to {get_characters_path()}")
