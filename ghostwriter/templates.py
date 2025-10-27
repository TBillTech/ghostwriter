"""Template application, prompt-key helpers, and prompt builders.

This module centralizes:
- apply_template and prompt_key_from_filename
- Chapter iteration I/O helpers (iter_dir_for, list_versions, get_latest_version, read_latest)
- Prompt builders (build_common_replacements, build_master_* and related)
"""
from __future__ import annotations

import re
import os
from pathlib import Path
from typing import Dict, List, Optional

from .utils import read_text, read_file, to_text
from .env import get_iterations_dir


def apply_template(template_path: str | Path, replacements: Dict[str, str]) -> str:
    template = read_text(template_path)
    for k, v in replacements.items():
        template = template.replace(k, v)
    return template


def prompt_key_from_filename(filename: str) -> str:
    base = Path(filename).name
    if base.lower().endswith(".md"):
        base = base[:-3]
    if base.lower().endswith("_prompt"):
        base = base[:-7]
    key = re.sub(r"[^A-Za-z0-9]+", "_", base).strip("_").upper()
    return key


# ---------------------------
# I/O helpers for iterations and artifacts
# ---------------------------

def iter_dir_for(chapter_id: str) -> Path:
    """Return the iterations directory for a chapter id.

    Always uses the configured location resolved by get_iterations_dir(),
    typically <GW_BOOK_BASE_DIR or .>/iterations/<chapter_id>.
    """
    base = get_iterations_dir()
    return base / chapter_id


def list_versions(chapter_id: str, prefix: str) -> List[int]:
    d = iter_dir_for(chapter_id)
    if not d.exists():
        return []
    versions: List[int] = []
    for p in d.glob(f"{prefix}_v*.txt"):
        m = re.search(r"_v(\d+)\.txt$", p.name)
        if m:
            versions.append(int(m.group(1)))
    return sorted(set(versions))


def get_latest_version(chapter_id: str) -> int:
    """Find the highest version number across draft/check/suggestions/predraft files."""
    prefixes = ["draft", "check", "suggestions", "pre_draft"]
    latest = 0
    for pref in prefixes:
        vs = list_versions(chapter_id, pref)
        if vs:
            latest = max(latest, vs[-1])
    return latest


def read_latest(chapter_id: str, prefix: str) -> Optional[str]:
    vs = list_versions(chapter_id, prefix)
    if not vs:
        return None
    path = iter_dir_for(chapter_id) / f"{prefix}_v{vs[-1]}.txt"
    return read_file(str(path))


# ---------------------------
# Prompt builders
# ---------------------------

def _yaml_section_fallback(chapter: dict, key: str) -> str:
    val = chapter.get(key)
    if val is None:
        return ""
    try:
        return to_text({key: val})
    except Exception:
        return str(val)


def build_common_replacements(setting: dict, chapter: dict, chapter_id: str, current_version: int) -> Dict[str, str]:
    d = iter_dir_for(chapter_id)
    # Prefer story summaries from the previous chapter if current chapter doesn't have them yet
    def _prev_chapter_id(cid: str) -> Optional[str]:
        import re as _re
        m = _re.match(r"^(.*?)(\d+)$", cid.replace("CHAPTER_", ""))
        # Common chapter id format: CHAPTER_001
        m2 = _re.match(r"^CHAPTER_(\d+)$", cid)
        if m2:
            n = int(m2.group(1))
            if n > 1:
                return f"CHAPTER_{n-1:03d}"
            return None
        if m:
            prefix = "CHAPTER_"
            try:
                n = int(m.group(2))
                if n > 1:
                    return f"{prefix}{n-1:03d}"
            except Exception:
                return None
        return None
    prev_id = _prev_chapter_id(chapter_id)
    prev_dir = iter_dir_for(prev_id) if prev_id else None
    story_so_far = (d / "story_so_far.txt").read_text(encoding="utf-8") if (d / "story_so_far.txt").exists() else (
        (prev_dir / "story_so_far.txt").read_text(encoding="utf-8") if (prev_dir and (prev_dir / "story_so_far.txt").exists()) else chapter.get("Story-So-Far", "")
    )
    story_relative = (d / "story_relative_to.txt").read_text(encoding="utf-8") if (d / "story_relative_to.txt").exists() else (
        (prev_dir / "story_relative_to.txt").read_text(encoding="utf-8") if (prev_dir and (prev_dir / "story_relative_to.txt").exists()) else _yaml_section_fallback(chapter, "Story-Relative-To")
    )
    include_raw = os.getenv("GW_INCLUDE_RAW_YAML", "0") == "1"
    rep: Dict[str, str] = {
        "[SETTING.yaml]": to_text(setting) if include_raw else "",
        "[CHAPTER_xx.yaml]": to_text(chapter) if include_raw else "",
        "[story_so_far.txt]": story_so_far,
        "[story_relative_to.txt]": story_relative,
    }
    # Provide uppercase synonym keys to improve template compatibility
    rep["[STORY_SO_FAR]"] = story_so_far
    rep["[STORY_RELATIVE_TO]"] = story_relative
    for key, pref in (
        ("[draft_v?.txt]", "draft"),
        ("[suggestions_v?.txt]", "suggestions"),
        ("[check_v?.txt]", "check"),
        ("[predraft_v?.txt]", "pre_draft"),
    ):
        rep[key] = read_latest(chapter_id, pref) or ""
    return rep


def build_master_initial_prompt(setting: dict, chapter: dict, chapter_id: str, version: int) -> str:
    reps = build_common_replacements(setting, chapter, chapter_id, version)
    return apply_template("prompts/master_initial_prompt.md", reps)


def build_master_prompt(setting: dict, chapter: dict, chapter_id: str, version: int) -> str:
    reps = build_common_replacements(setting, chapter, chapter_id, version)
    return apply_template("prompts/master_prompt.md", reps)


def build_polish_prompt(setting: dict, chapter: dict, chapter_id: str, version: int, rough_text: str) -> str:
    reps = build_common_replacements(setting, chapter, chapter_id, version)
    reps["[rough_draft]"] = rough_text
    return apply_template("prompts/polish_prose_prompt.md", reps)


def build_check_prompt(setting: dict, chapter: dict, chapter_id: str, version: int, predraft_text: str) -> str:
    reps = build_common_replacements(setting, chapter, chapter_id, version)
    reps["[predraft_v?.txt]"] = predraft_text
    return apply_template("prompts/check_prompt.md", reps)


def build_story_so_far_prompt(setting: dict, chapter: dict, chapter_id: str, version: int, predraft_text: str) -> str:
    reps = build_common_replacements(setting, chapter, chapter_id, version)
    reps["[predraft_v?.txt]"] = predraft_text
    return apply_template("prompts/story_so_far_prompt.md", reps)


def build_story_relative_to_prompt(setting: dict, chapter: dict, chapter_id: str, version: int, predraft_text: str) -> str:
    reps = build_common_replacements(setting, chapter, chapter_id, version)
    reps["[predraft_v?.txt]"] = predraft_text
    return apply_template("prompts/story_relative_to_prompt.md", reps)
