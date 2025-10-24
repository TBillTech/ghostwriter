"""CONTENT_TABLE.yaml brainstorming pipeline.

When invoked (typically by running the CLI with chapters/CONTENT_TABLE.yaml),
this pipeline will:
  - Parse the table of contents and detect the next chapter id (e.g., 007)
  - Collect non-dereferenced `setting` blocks from existing chapter YAML files
    - Prompt the LLM for the most creative yet natural next chapter synopsis
    - Strictly APPEND the new chapter entry (do not alter any '???' brainstorm placeholder)
  - Save the prompt + result to chapters/CONTENT_TABLE_brainstorm.txt
Stops after updating once.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import yaml

from ..env import get_chapters_dir
from ..utils import read_text as _read_text, save_text as _save_text, to_text as _to_text
from ..llm import complete as llm_complete
from ..templates import apply_template
from ..env import env_for_prompt, reasoning_for_prompt
from ..context import RunContext


def _content_table_path() -> Path:
    return get_chapters_dir() / "CONTENT_TABLE.yaml"


def _next_chapter_id(toc: List[Any]) -> Tuple[str, int]:
    max_num = 0
    for item in toc or []:
        if isinstance(item, dict) and item:
            key = next(iter(item.keys()))
            try:
                n = int(str(key))
                if n > max_num:
                    max_num = n
            except Exception:
                continue
    return f"{max_num+1:03d}", max_num + 1


def _find_numeric_placeholder_id(toc: List[Any]) -> Optional[int]:
    """Return the first numeric chapter id whose value looks like a brainstorm placeholder.

    Placeholder detection rules for values:
    - exactly '???' (any surrounding whitespace)
    - starts with '???' (e.g., '??? fill later')
    - starts with 'brainstorm' (case-insensitive)
    Chooses the smallest numeric id that matches to fill earlier gaps first.
    """
    candidates: List[int] = []
    for item in toc or []:
        if isinstance(item, dict) and item:
            key = next(iter(item.keys()))
            try:
                n = int(str(key))
            except Exception:
                continue
            val = item[key]
            if isinstance(val, str):
                s = val.strip()
                if s == "???" or s.startswith("???") or s.lower().startswith("brainstorm"):
                    candidates.append(n)
    if not candidates:
        return None
    return min(candidates)


def _collect_chapter_settings_from_ctx(ctx: RunContext) -> str:
    items: List[Dict[str, Any]] = []
    try:
        for it in ctx.chapter_settings_index or []:
            if isinstance(it, dict) and it.get("chapter") and it.get("setting") is not None:
                items.append({"chapter": it.get("chapter"), "setting": it.get("setting")})
    except Exception:
        pass
    return _to_text({"Chapter-Settings": items})



def _set_or_append_entry(ct_yaml: Dict[str, Any], key: str | int, synopsis: str) -> Dict[str, Any]:
    # Set or append into TABLE_OF_CONTENTS list. If the numeric key exists, replace its value; else append.
    toc = ct_yaml.get("TABLE_OF_CONTENTS") or ct_yaml.get("table_of_contents")
    if not isinstance(toc, list):
        return ct_yaml
    # Use numeric chapter keys in YAML (e.g., 7 instead of '007') to match existing entries
    try:
        numeric_key = int(str(key))
    except Exception:
        numeric_key = key
    replaced = False
    for i, item in enumerate(toc):
        if isinstance(item, dict) and item:
            k = next(iter(item.keys()))
            try:
                kn = int(str(k))
            except Exception:
                continue
            if kn == numeric_key:
                toc[i] = {numeric_key: synopsis}
                replaced = True
                break
    if not replaced:
        toc.append({numeric_key: synopsis})
    ct_yaml["TABLE_OF_CONTENTS"] = toc
    return ct_yaml


def run_content_table_brainstorm(*, ctx: RunContext) -> None:
    ct_path = _content_table_path()
    ct_text = ctx.content_table_text or _read_text(ct_path)
    ct_yaml = ctx.content_table or {}
    toc = ct_yaml.get("TABLE_OF_CONTENTS") if isinstance(ct_yaml, dict) else None
    if not isinstance(toc, list):
        print("CONTENT_TABLE.yaml: TABLE_OF_CONTENTS not parsed as a list; aborting.")
        return

    # Prefer filling a numeric placeholder like 6: '??? ...' if present; otherwise pick next id
    placeholder_id = _find_numeric_placeholder_id(toc)
    if placeholder_id is not None:
        target_id_num = placeholder_id
        next_id_str = f"{target_id_num:03d}"
    else:
        next_id_str, target_id_num = _next_chapter_id(toc)
    ch_settings_text = _collect_chapter_settings_from_ctx(ctx)

    reps: Dict[str, str] = {
        "[CONTENT_TABLE.yaml]": ct_text,
        "[CHAPTER_SETTINGS]": ch_settings_text,
        "[NEXT_CHAPTER_ID]": next_id_str,
    }
    tpl = "brainstorm_content_table_next.md"
    user = apply_template(Path("prompts") / tpl, reps)
    model, temp, max_tokens = env_for_prompt(tpl, "BRAIN_STORM", default_temp=0.45, default_max_tokens=500)
    reason = reasoning_for_prompt(tpl, "BRAIN_STORM")
    system = "You propose the next chapter synopsis based on the table of contents and chapter settings."
    out = llm_complete(user, system=system, temperature=temp, max_tokens=max_tokens, model=model, reasoning_effort=reason)

    # Save prompt + result
    try:
        log_file = ct_path.parent / "CONTENT_TABLE_brainstorm.txt"
        log_body = (
            "=== SYSTEM ===\n" + system +
            "\n\n=== USER ===\n" + user +
            "\n\n=== RESPONSE ===\n" + (out or "") + "\n"
        )
        _save_text(log_file, log_body)
    except Exception:
        pass

    # Write updated CONTENT_TABLE.yaml (replace placeholder if present; else append)
    try:
        # Ensure multi-line strings are preserved nicely as YAML literal blocks (|)
        updated = _set_or_append_entry(ct_yaml, target_id_num, out.strip())

        class _LiteralDumper(yaml.SafeDumper):
            pass

        def _str_presenter(dumper, data):
            # Prefer literal block style for multi-line strings
            if isinstance(data, str) and ("\n" in data):
                return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
            return dumper.represent_scalar('tag:yaml.org,2002:str', data)

        yaml.add_representer(str, _str_presenter, Dumper=_LiteralDumper)

        dumped = yaml.dump(
            updated,
            Dumper=_LiteralDumper,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )
        ct_path.write_text(dumped, encoding="utf-8")
        action = "Replaced placeholder for" if placeholder_id is not None else "Appended"
        print(f"{action} chapter {next_id_str} synopsis to {ct_path}")
    except Exception as e:
        print(f"Failed to update CONTENT_TABLE.yaml: {e}")
