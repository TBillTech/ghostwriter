"""Subtle edit pipeline (migrated)."""
from __future__ import annotations

from typing import Optional
from pathlib import Path

from ..templates import apply_template
from ..validation import validate_text
from ..llm import complete as llm_complete
from .common import env_for_prompt, build_pipeline_replacements


def run_subtle_edit_pipeline(tp, state, *, setting: dict, chapter: dict, chapter_id: str, version: int, tp_index: int, prior_polished: str, prior_suggestions: str, prior_paragraph: str = "", log_dir: Optional[Path] = None) -> str:
    reps = build_pipeline_replacements(setting, chapter, chapter_id, version, tp, state, prior_paragraph=prior_paragraph)
    reps["[PRIOR_POLISHED]"] = prior_polished
    reps["[SUGGESTIONS]"] = prior_suggestions
    sys1 = "Apply subtle edits to the provided prose respecting suggestions and style."
    tpl_se = "subtle_edit_prompt.md"
    from pathlib import Path as _P
    user1 = apply_template(str(_P("prompts") / tpl_se), reps) if (_P("prompts") / tpl_se).exists() else (
        f"Apply subtle edits to this text:\n\n{prior_polished}\n\nSuggestions:\n{prior_suggestions}\n"
    )
    m1, t1, k1 = env_for_prompt(tpl_se, "SUBTLE_EDIT", default_temp=0.2, default_max_tokens=1000)
    edited = llm_complete(
        user1,
        system=sys1,
        temperature=t1,
        max_tokens=k1,
        model=m1,
    )
    # Final polish
    from ..templates import build_polish_prompt
    polish_prompt = build_polish_prompt(setting, chapter, chapter_id, version, edited)
    mp, tpv, kp = env_for_prompt("polish_prose_prompt.md", "POLISH_PROSE", default_temp=0.2, default_max_tokens=2000)
    out = llm_complete(
        polish_prompt,
        system="You are a ghostwriter polishing and cleaning prose.",
        temperature=tpv,
        max_tokens=kp,
        model=mp,
    )
    state.last_appended_dialog = {}
    return out
