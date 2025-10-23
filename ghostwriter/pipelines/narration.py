"""Narration pipeline (migrated).

Implements the narration pipeline using shared helpers from pipelines.common.
Keeps behavior consistent with the legacy driver: brainstorm gating with manual
DONE, ordering, generate narration, then polish.
"""
from __future__ import annotations

from typing import Optional, Dict, List
from pathlib import Path

from ..context import RunContext
from ..templates import apply_template
from ..llm import complete as llm_complete
from ..validation import validate_text, validate_bullet_list
from .common import (
    env_for_prompt,
    llm_call_with_validation,
    build_pipeline_replacements,
    strip_trailing_done,
    brainstorm_has_done,
)


def run_narration_pipeline(tp, state, *, ctx: RunContext, tp_index: int, prior_paragraph: str = "", log_dir: Optional[Path] = None) -> str:
    # Clear last appended dialog; narration adds none
    state.last_appended_dialog = {}
    reps = build_pipeline_replacements(ctx.setting, ctx.chapter, ctx.chapter_id, ctx.version, tp, state, prior_paragraph=prior_paragraph, ctx=ctx)
    # 1) Brainstorm → bullet list (persistent with DONE gating)
    sys1 = "You are brainstorming narrative beats as concise bullet points."
    tpl1 = "narration_brain_storm_prompt.md"
    user1_base = apply_template(str(Path("prompts") / tpl1), reps) if (Path("prompts") / tpl1).exists() else (
        f"Context follows.\n\n{reps.get('[SETTING.yaml]', '')}\n\n{reps.get('[CHAPTER_xx.yaml]', '')}\n\n"
        f"Touch-Point ({reps.get('[TOUCH_POINT_TYPE]', '')}): {reps.get('[TOUCH_POINT]', '')}\n"
        f"State: actors={reps.get('[ACTIVE_ACTORS]', '')}, scene={reps.get('[SCENE]', '')}\n"
    )
    model, temp, max_toks = env_for_prompt(tpl1, "BRAIN_STORM", default_temp=0.4, default_max_tokens=600)
    # Brainstorm persistence path
    bs_path = (log_dir / "brainstorm.txt") if log_dir else None
    existing_bs = ""
    if bs_path is not None and bs_path.exists():
        try:
            existing_bs = bs_path.read_text(encoding="utf-8")
        except Exception:
            existing_bs = ""

    brainstorm_for_use = None  # cleaned without DONE for downstream
    if existing_bs and brainstorm_has_done(existing_bs):
        brainstorm_for_use = strip_trailing_done(existing_bs)
    else:
        # Seed with previous bullets by appending them to the end of the prompt without headings
        seed_bullets = strip_trailing_done(existing_bs) if existing_bs else ""
        user1 = user1_base
        if seed_bullets.strip():
            user1 = user1 + "\n\n" + seed_bullets.strip() + "\n"
        # Generate additional bullets
            brainstorm_new = llm_call_with_validation(
            sys1, user1, model=model, temperature=temp, max_tokens=max_toks, validator=validate_bullet_list,
            log_maker=(lambda attempt: (log_dir / f"{tp_index:02d}_brainstorm{'_r'+str(attempt) if attempt>1 else ''}.txt")) if log_dir else None,
        )
        # Persist brainstorm as previous bullets followed by new bullets (cumulative). No DONE written by LLM.
        if bs_path is not None:
            try:
                bs_path.parent.mkdir(parents=True, exist_ok=True)
                combined = (seed_bullets.strip() + ("\n" if seed_bullets.strip() and not seed_bullets.strip().endswith("\n") else "")) + brainstorm_new
                bs_path.write_text(combined, encoding="utf-8")
            except Exception:
                pass
        # Require human to add DONE manually to proceed
        print("Brainstorming still in progress.")
        raise SystemExit(0)

    # 2) Ordering → bullet list
    reps2 = dict(reps)
    reps2["[BULLET_IDEAS]"] = brainstorm_for_use or ""
    reps2["[BULLETS]"] = brainstorm_for_use or ""
    reps2["[bullets]"] = brainstorm_for_use or ""
    sys2 = "You will order and refine brainstormed bullet points for coherent flow."
    tpl2 = "ordering_prompt.md"
    if (Path("prompts") / tpl2).exists():
        user2 = apply_template(str(Path("prompts") / tpl2), reps2)
    else:
        user2 = f"Order the following bullets for coherent flow:\n\n{reps2['[bullets]']}\n"
    model2, temp2, max2 = env_for_prompt(tpl2, "ORDERING", default_temp=0.2, default_max_tokens=600)
    ordered = llm_call_with_validation(
        sys2, user2, model=model2, temperature=temp2, max_tokens=max2, validator=validate_bullet_list,
        log_maker=(lambda attempt: (log_dir / f"{tp_index:02d}_ordering{'_r'+str(attempt) if attempt>1 else ''}.txt")) if log_dir else None,
    )

    # 3) Generate narration → text
    reps3 = dict(reps2)
    reps3["[ORDERED_BULLETS]"] = ordered
    reps3["[ordered_bullets]"] = ordered
    reps3["[bullets]"] = ordered
    sys3 = "You write narrative prose from ordered bullets, keeping voice consistent."
    tpl3 = "generate_narration_prompt.md"
    if (Path("prompts") / tpl3).exists():
        user3 = apply_template(str(Path("prompts") / tpl3), reps3)
    else:
        user3 = f"Write prose from these bullets:\n\n{ordered}\n"
    model3, temp3, max3 = env_for_prompt(tpl3, "GENERATE_NARRATION", default_temp=0.35, default_max_tokens=1000)
    narration = llm_call_with_validation(
        sys3, user3, model=model3, temperature=temp3, max_tokens=max3, validator=validate_text,
        log_maker=(lambda attempt: (log_dir / f"{tp_index:02d}_generate_narration{'_r'+str(attempt) if attempt>1 else ''}.txt")) if log_dir else None,
    )

    # 4) Polish
    from ..templates import build_polish_prompt
    polish_prompt = build_polish_prompt(ctx.setting, ctx.chapter, ctx.chapter_id, ctx.version, narration)
    modelp, tempp, maxp = env_for_prompt("polish_prose_prompt.md", "POLISH_PROSE", default_temp=0.2, default_max_tokens=2000)
    polished = llm_call_with_validation(
        system="You are a ghostwriter polishing and cleaning prose.",
        user=polish_prompt,
        model=modelp,
        temperature=tempp,
        max_tokens=maxp,
        validator=validate_text,
        log_maker=(lambda attempt: (log_dir / f"{tp_index:02d}_polish{'_r'+str(attempt) if attempt>1 else ''}.txt")) if log_dir else None,
    )
    return polished
