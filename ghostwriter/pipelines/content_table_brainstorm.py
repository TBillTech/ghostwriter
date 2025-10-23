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
from ..env import env_for_prompt
from ..context import load_yaml


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


def _collect_chapter_settings() -> str:
    chapters_dir = get_chapters_dir()
    items: List[Dict[str, Any]] = []
    for p in sorted(chapters_dir.glob("CHAPTER_*.yaml")):
        try:
            ch = load_yaml(str(p))
            if isinstance(ch, dict) and "setting" in ch:
                items.append({"chapter": p.stem, "setting": ch.get("setting")})
        except Exception:
            continue
    return _to_text({"Chapter-Settings": items})



def _append_next_entry(ct_yaml: Dict[str, Any], key: str, synopsis: str) -> Dict[str, Any]:
    # Append into TABLE_OF_CONTENTS list; preserve any existing '???' brainstorm item
    toc = ct_yaml.get("TABLE_OF_CONTENTS") or ct_yaml.get("table_of_contents")
    if not isinstance(toc, list):
        return ct_yaml
    toc.append({key: synopsis})
    ct_yaml["TABLE_OF_CONTENTS"] = toc
    return ct_yaml


def run_content_table_brainstorm() -> None:
    ct_path = _content_table_path()
    ct_text = _read_text(ct_path)
    ct_yaml = load_yaml(str(ct_path))
    toc = ct_yaml.get("TABLE_OF_CONTENTS") if isinstance(ct_yaml, dict) else None
    if not isinstance(toc, list):
        print("CONTENT_TABLE.yaml: TABLE_OF_CONTENTS not parsed as a list; aborting.")
        return

    next_id_str, _ = _next_chapter_id(toc)
    ch_settings_text = _collect_chapter_settings()

    reps: Dict[str, str] = {
        "[CONTENT_TABLE.yaml]": ct_text,
        "[CHAPTER_SETTINGS]": ch_settings_text,
        "[NEXT_CHAPTER_ID]": next_id_str,
    }
    tpl = "brainstorm_content_table_next.md"
    user = apply_template(Path("prompts") / tpl, reps)
    model, temp, max_tokens = env_for_prompt(tpl, "BRAIN_STORM", default_temp=0.45, default_max_tokens=500)
    system = "You propose the next chapter synopsis based on the table of contents and chapter settings."
    out = llm_complete(user, system=system, temperature=temp, max_tokens=max_tokens, model=model)

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

    # Write updated CONTENT_TABLE.yaml (no backup; strictly append)
    try:
        # Ensure multi-line strings are preserved nicely
        updated = _append_next_entry(ct_yaml, next_id_str, out.strip())
        ct_path.write_text(yaml.safe_dump(updated, sort_keys=False, allow_unicode=True), encoding="utf-8")
        print(f"Appended chapter {next_id_str} synopsis to {ct_path}")
    except Exception as e:
        print(f"Failed to update CONTENT_TABLE.yaml: {e}")
