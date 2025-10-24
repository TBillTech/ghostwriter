"""Chapter brainstorming pipeline.

Triggers when a chapter file is missing or when a chapter contains a
touch-point `{ brainstorming: True }`. Produces a new CHAPTER_XXX.yaml outline
based on CONTENT_TABLE synopsis, story-so-far, and setting/characters context.

Artifacts written:
- Backup old chapter as CHAPTER_XXX.N.yaml (N = next cycle)
- Overwrite or create chapters/CHAPTER_XXX.yaml with LLM result (no Story sections)
- Save prompt + result to chapters/CHAPTER_XXX.txt
- Save combined result + prior chapter summaries to iterations/CHAPTER_(XXX-1)/pipeline_v2/

Exits the program after completing one outline.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

from ..env import (
    get_chapters_dir,
    env_for_prompt,
    reasoning_for_prompt
)
from ..templates import apply_template, iter_dir_for
from ..utils import to_text as _to_text, read_text as _read_text, save_text as _save_text
from ..context import RunContext
from ..validation import validate_text
from ..llm import complete as llm_complete
from ..characters import load_characters_list
from ..factoids import factoids_block_from_setting


def _chapter_id_from_path(chapter_path: str) -> str:
    return Path(chapter_path).stem


def _parse_chapter_number(chapter_id: str) -> Optional[int]:
    # Expect formats like CHAPTER_001
    import re
    m = re.search(r"(\d+)$", chapter_id)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _strip_story_sections(ch_yaml: Dict[str, Any]) -> str:
    if not isinstance(ch_yaml, dict):
        return _to_text(ch_yaml)
    obj = dict(ch_yaml)
    # Remove Story sections if present
    for k in ["Story-So-Far", "Story-Relative-To", "Story Relative To", "Story So Far"]:
        if k in obj:
            obj.pop(k, None)
    return _to_text(obj)


def _selected_characters_block(ctx_chars: List[dict], names: List[str]) -> str:
    if not ctx_chars or not names:
        return ""
    wanted = {str(n).strip().lower() for n in names if str(n).strip()}
    sel: List[dict] = []
    for ch in ctx_chars:
        cid = str(ch.get("id", "")).strip().lower()
        cname = str(ch.get("name", "")).strip().lower()
        if cid in wanted or cname in wanted:
            sel.append(ch)
    if not sel:
        return ""
    return _to_text({"Selected-Characters": sel})


def _names_only_block(setting: dict, characters: List[dict]) -> str:
    # Factoid names
    fact_names: List[str] = []
    try:
        facts = setting.get("Factoids") if isinstance(setting, dict) else None
        if isinstance(facts, list):
            for f in facts:
                if isinstance(f, dict) and isinstance(f.get("name"), str):
                    fact_names.append(f.get("name").strip())
    except Exception:
        pass
    # Character names
    char_names: List[str] = []
    try:
        for ch in characters:
            name = ch.get("name") or ch.get("id")
            if isinstance(name, str) and name.strip():
                char_names.append(name.strip())
    except Exception:
        pass
    return _to_text({
        "Factoid-Names": fact_names,
        "Character-Names": char_names,
    })


def _load_prior_summaries(prev_chapter_id: Optional[str]) -> Tuple[str, str]:
    if not prev_chapter_id:
        return "", ""
    d = iter_dir_for(prev_chapter_id)
    ssf = (d / "story_so_far.txt").read_text(encoding="utf-8") if (d / "story_so_far.txt").exists() else ""
    srt = (d / "story_relative_to.txt").read_text(encoding="utf-8") if (d / "story_relative_to.txt").exists() else ""
    return ssf, srt


def _backup_existing_chapter(path: Path) -> Optional[Path]:
    if not path.exists():
        return None
    # Determine next backup index: CHAPTER_XXX.N.yaml
    backups = sorted(path.parent.glob(f"{path.stem}.*{path.suffix}"))
    import re
    max_n = 0
    for b in backups:
        m = re.search(r"\.([0-9]+)\%s$" % (path.suffix.replace(".", r"\.")), b.name)
        if m:
            try:
                n = int(m.group(1))
                max_n = max(max_n, n)
            except Exception:
                pass
    next_n = max_n + 1
    backup_path = path.with_name(f"{path.stem}.{next_n}{path.suffix}")
    try:
        backup_path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        return backup_path
    except Exception:
        return None


def run_chapter_brainstorm(*, ctx: RunContext, log_llm: bool = False) -> None:
    chapter_id = ctx.chapter_id
    version_num = ctx.version
    chapters_dir = ctx.chapters_dir or get_chapters_dir()
    chapter_file = chapters_dir / f"{chapter_id}.yaml"

    # Use SETTING + CHARACTERS from context
    setting = ctx.setting
    characters = ctx.characters

    # CONTENT_TABLE.yaml text from context (may be empty)
    content_tbl_text = ctx.content_table_text or ""

    # Derive current chapter YAML from context if present (without Story sections)
    current_chapter_no_story = ""
    chapter_has_setting = False
    sel_factoids_block = ""
    sel_characters_block = ""
    names_only_block = ""
    try:
        ch_yaml = ctx.chapter
        if isinstance(ch_yaml, dict) and ch_yaml:
            current_chapter_no_story = _strip_story_sections(ch_yaml)
            if isinstance(ch_yaml.get("setting"), dict):
                chs = ch_yaml.get("setting")
                chapter_has_setting = True
                fact_list = chs.get("factoids") if isinstance(chs.get("factoids"), list) else []
                actor_list = chs.get("actors") if isinstance(chs.get("actors"), list) else []
                sel_factoids_block = factoids_block_from_setting(setting, selected_names=fact_list)
                sel_characters_block = _selected_characters_block(characters, actor_list)
    except Exception:
        current_chapter_no_story = ""
    # If no structured chapter setting found, provide names-only list
    if not chapter_has_setting:
        names_only_block = _names_only_block(setting, characters)

    # Story so far (from prior completed chapter)
    ch_num = _parse_chapter_number(chapter_id) or 0
    prev_id = f"CHAPTER_{ch_num-1:03d}" if ch_num > 0 else None
    ssf, srt = _load_prior_summaries(prev_id)

    # Build replacements for prompt
    reps: Dict[str, str] = {
        "[CHAPTER_ID]": chapter_id,
        "[CONTENT_TABLE.yaml]": content_tbl_text,
        "[STORY_SO_FAR]": ssf,
        "[CURRENT_CHAPTER_NO_STORY]": current_chapter_no_story,
        "[SETTING_DEREF]": (sel_factoids_block + ("\n\n" if sel_factoids_block and sel_characters_block else "") + sel_characters_block).strip(),
        "[NAMES_ONLY]": names_only_block,
        "[CHAPTER_NUM]": str(ch_num),
        "[VERSION_NUM]": str(version_num),
        "[FORMAT_HINT]": ("Output must be valid YAML for the chapter file, containing Touch-Points and an optional 'setting' with 'factoids' (list of names), 'actors' (list of character ids or names), and 'scene' (string). Do NOT include Story-So-Far or Story-Relative-To sections." if not chapter_file.exists() else ""),
    }

    # Apply template and call LLM
    tpl_name = "brainstorm_chapter_outline.md"
    user_prompt = apply_template(Path("prompts") / tpl_name, reps)
    model, temp, max_tokens = env_for_prompt(tpl_name, "BRAIN_STORM", default_temp=0.5, default_max_tokens=800)
    reason = reasoning_for_prompt(tpl_name, "BRAIN_STORM")
    system = "You brainstorm and draft a chapter outline as YAML."
    out = llm_complete(user_prompt, system=system, temperature=temp, max_tokens=max_tokens, model=model, reasoning_effort=reason)

    # Log prompt + result alongside the chapter file as CHAPTER_XXX.txt
    try:
        chapters_dir.mkdir(parents=True, exist_ok=True)
        txt_path = chapters_dir / f"{chapter_id}.txt"
        txt_body = (
            "=== SYSTEM ===\n" + (system or "") +
            "\n\n=== USER ===\n" + (user_prompt or "") +
            "\n\n=== RESPONSE ===\n" + (out or "") + "\n"
        )
        _save_text(txt_path, txt_body)
    except Exception:
        pass

    # Backup existing and write new chapter YAML
    _backup_existing_chapter(chapter_file)
    try:
        chapter_file.parent.mkdir(parents=True, exist_ok=True)
        chapter_file.write_text(out, encoding="utf-8")
    except Exception:
        # If write fails, still continue to save auxiliary artifact
        pass

    # Save combined result into previous chapter's pipeline_v2 folder
    if prev_id:
        try:
            out_dir = iter_dir_for(prev_id) / "pipeline_v2"
            out_dir.mkdir(parents=True, exist_ok=True)
            combo = (
                "# Chapter Brainstorm Result\n\n" + out.strip() + "\n\n" +
                "# Story-So-Far (previous chapter)\n\n" + ssf.strip() + "\n\n" +
                "# Story-Relative-To (previous chapter)\n\n" + srt.strip() + "\n"
            )
            _save_text(out_dir / "chapter_brainstorm_result.txt", combo)
        except Exception:
            pass

    # Exit after one outline update (the caller should end the run)
    print(f"Brainstormed chapter outline written to {chapter_file}")
