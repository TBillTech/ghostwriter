"""Dialog pipeline.

"""
from __future__ import annotations

from typing import Optional, List, Dict
import os
from pathlib import Path

from ..context import RunContext
from ..templates import apply_template
from ..validation import validate_bullet_list, validate_actor_list, validate_text
from ..llm import complete as llm_complete
from .common import (
    env_for_prompt,
    llm_call_with_validation,
    build_pipeline_replacements,
    strip_trailing_done,
    brainstorm_has_done,
    filter_narration_brainstorm,
    truncate_brainstorm,
    apply_body_template,
)
from ..characters import load_characters_list, render_character_call
from ..utils import to_text


def run_dialog_pipeline(tp, state, *, ctx: RunContext, tp_index: int, prior_paragraph: str = "", log_dir: Optional[Path] = None) -> str:
    appended: Dict[str, List[str]] = {}
    reps = build_pipeline_replacements(ctx.setting, ctx.chapter, ctx.chapter_id, ctx.version, tp, state, prior_paragraph=prior_paragraph, ctx=ctx)
    # Brainstorm
    sys1 = "Brainstorm dialog beats as bullet points."
    tpl1 = "dialog_brain_storm_prompt.md"
    user1_base = apply_template(str(Path("prompts") / tpl1), reps) if (Path("prompts") / tpl1).exists() else ""
    m1, t1, k1 = env_for_prompt(tpl1, "BRAIN_STORM", default_temp=0.45, default_max_tokens=600)
    bs_path = (log_dir / "brainstorm.txt") if log_dir else None
    existing_bs = ""
    if bs_path is not None and bs_path.exists():
        try:
            existing_bs = bs_path.read_text(encoding="utf-8")
        except Exception:
            existing_bs = ""
    if existing_bs and brainstorm_has_done(existing_bs):
        brainstorm_work = strip_trailing_done(existing_bs)
    else:
        seed_bullets = strip_trailing_done(existing_bs) if existing_bs else ""
        user1 = user1_base
        if seed_bullets.strip():
            user1 = user1 + "\n\n" + seed_bullets.strip() + "\n"
        brainstorm_new = llm_call_with_validation(
            sys1, user1, model=m1, temperature=t1, max_tokens=k1, validator=validate_bullet_list,
            log_maker=(lambda attempt: (log_dir / f"{tp_index:02d}_brainstorm{'_r'+str(attempt) if attempt>1 else ''}.txt")) if log_dir else None,
        )
        if bs_path is not None:
            try:
                bs_path.parent.mkdir(parents=True, exist_ok=True)
                combined = (seed_bullets.strip() + ("\n" if seed_bullets.strip() and not seed_bullets.strip().endswith("\n") else "")) + brainstorm_new
                bs_path.write_text(combined, encoding="utf-8")
            except Exception:
                pass
        print("Brainstorming still in progress.")
        raise SystemExit(0)

    brainstorm = filter_narration_brainstorm(brainstorm_work, min_chars=24)
    brainstorm = truncate_brainstorm(brainstorm, limit=10)

    # Ordering (respect env skip via dialog key)
    reps2 = dict(reps)
    reps2["[BULLET_IDEAS]"] = brainstorm
    reps2["[BULLETS]"] = brainstorm
    reps2["[bullets]"] = brainstorm
    disable_order_dialog = os.getenv("GW_DISABLE_ORDERING_DIALOG", "0") == "1"
    if disable_order_dialog:
        ordered = brainstorm
    else:
        sys2 = "Order dialog beats for flow."
        tpl2 = "ordering_prompt.md"
        user2 = apply_template(str(Path("prompts") / tpl2), reps2) if (Path("prompts") / tpl2).exists() else f"Order bullets:\n\n{brainstorm}\n"
        m2, t2, k2 = env_for_prompt(tpl2, "ORDERING", default_temp=0.25, default_max_tokens=600)
        ordered = llm_call_with_validation(
            sys2, user2, model=m2, temperature=t2, max_tokens=k2, validator=validate_bullet_list,
            log_maker=(lambda attempt: (log_dir / f"{tp_index:02d}_ordering{'_r'+str(attempt) if attempt>1 else ''}.txt")) if log_dir else None,
        )

    # Actor assignment
    reps3 = dict(reps2)
    reps3["[ORDERED_BULLETS]"] = ordered
    reps3["[ordered_bullets]"] = ordered
    reps3["[bullets]"] = ordered
    sys3 = "Assign dialog lines to actors as 'id: line' entries."
    tpl3 = "actor_assignment_prompt.md"
    user3 = apply_template(str(Path("prompts") / tpl3), reps3) if (Path("prompts") / tpl3).exists() else f"Assign actors for bullets:\n\n{ordered}\n"
    m3, t3, k3 = env_for_prompt(tpl3, "ACTOR_ASSIGNMENT", default_temp=0.25, default_max_tokens=600)
    actor_lines = llm_call_with_validation(
        sys3, user3, model=m3, temperature=t3, max_tokens=k3, validator=validate_actor_list,
        log_maker=(lambda attempt: (log_dir / f"{tp_index:02d}_actor_assignment{'_r'+str(attempt) if attempt>1 else ''}.txt")) if log_dir else None,
    )

    # Build ACTOR_LIST (ids only)
    import re as _re
    _ACTOR_LINE_RE = _re.compile(r"^\s*([A-Za-z0-9_.\-]+)\s*:\s+.+")
    pairs_for_ids: List[str] = []
    for ln in actor_lines.splitlines():
        m = _ACTOR_LINE_RE.match(ln)
        if not m:
            continue
        aid = m.group(1).strip()
        if aid and aid not in pairs_for_ids:
            pairs_for_ids.append(aid)
    actor_list = ", ".join(pairs_for_ids)

    # Body language, Agenda, Reactions
    reps4 = dict(reps3); reps4["[ACTOR_LIST]"] = actor_list; reps4["[ACTOR_LINES]"] = actor_lines
    # Body language
    sys4a = "List body language cues as bullet points."
    tpl4a = "body_language_prompt.md"
    user4a = apply_template(str(Path("prompts") / tpl4a), reps4) if (Path("prompts") / tpl4a).exists() else ""
    m4a, t4a, k4a = env_for_prompt(tpl4a, "BODY_LANGUAGE", default_temp=0.3, default_max_tokens=300)
    body_lang = llm_call_with_validation(
        sys4a, user4a, model=m4a, temperature=t4a, max_tokens=k4a, validator=validate_bullet_list,
        log_maker=(lambda attempt: (log_dir / f"{tp_index:02d}_body_language{'_r'+str(attempt) if attempt>1 else ''}.txt")) if log_dir else None,
    )
    # Agenda (grouped by actor)
    sys4b = "Produce an agenda list grouped by actor id."
    tpl4b = "agenda_prompt.md"
    user4b = apply_template(str(Path("prompts") / tpl4b), reps4) if (Path("prompts") / tpl4b).exists() else ""
    m4b, t4b, k4b = env_for_prompt(tpl4b, "AGENDA", default_temp=0.3, default_max_tokens=300)
    agenda_text = llm_call_with_validation(
        sys4b, user4b, model=m4b, temperature=t4b, max_tokens=k4b, validator=validate_text,
        log_maker=(lambda attempt: (log_dir / f"{tp_index:02d}_agenda{'_r'+str(attempt) if attempt>1 else ''}.txt")) if log_dir else None,
    )
    # Parse agenda into per-actor bullets
    def _parse_agenda_by_actor(text: str) -> Dict[str, str]:
        by_actor: Dict[str, List[str]] = {}
        current: Optional[str] = None
        import re as _re2
        for ln in text.splitlines():
            if not ln.strip():
                continue
            if not ln.lstrip().startswith(('*', '-')) and _re2.match(r"^\s*([A-Za-z0-9_.\-]+)\s*:\s*$", ln):
                m = _re2.match(r"^\s*([A-Za-z0-9_.\-]+)\s*:\s*$", ln)
                if m:
                    current = m.group(1).strip()
                    by_actor.setdefault(current, [])
                continue
            if current is not None and ln.lstrip().startswith(('*', '-')):
                content = ln.lstrip()[1:].lstrip()
                if content:
                    by_actor[current].append(content)
        return {aid: "\n".join([f"* {i}" for i in items]) for aid, items in by_actor.items() if items}

    agenda_by_actor = _parse_agenda_by_actor(agenda_text)

    # Reactions
    sys4c = "Produce a reaction for each line, referencing the previous line."
    tpl4c = "reaction_prompt.md"
    user4c = apply_template(str(Path("prompts") / tpl4c), reps4) if (Path("prompts") / tpl4c).exists() else ""
    m4c, t4c, k4c = env_for_prompt(tpl4c, "REACTIONS", default_temp=0.3, default_max_tokens=400)
    reactions_text = llm_call_with_validation(
        sys4c, user4c, model=m4c, temperature=t4c, max_tokens=k4c, validator=validate_actor_list,
        log_maker=(lambda attempt: (log_dir / f"{tp_index:02d}_reactions{'_r'+str(attempt) if attempt>1 else ''}.txt")) if log_dir else None,
    )

    # Preload character YAML for ids
    char_yaml_by_id: Dict[str, str] = {}
    try:
        all_chars = load_characters_list(ctx)
        if isinstance(all_chars, list) and all_chars:
            for ch in all_chars:
                cid = str(ch.get("id", "")).strip()
                if cid:
                    char_yaml_by_id[cid.lower()] = to_text(ch)
    except Exception:
        pass

    # Align reactions as list of (actor_id, reaction)
    import re as _re3
    _A_RE = _re3.compile(r"^\s*([A-Za-z0-9_.\-]+)\s*:\s+(.+)")
    reactions_pairs: List[str] = []
    reactions_vals: List[str] = []
    for ln in reactions_text.splitlines():
        m = _A_RE.match(ln)
        if m:
            reactions_pairs.append(m.group(1))
            reactions_vals.append(m.group(2))

    # Iterate assigned actor lines
    outputs: List[str] = []
    # reuse parser for actor_lines
    lines_pairs: List[tuple[str, str]] = []
    for ln in actor_lines.splitlines():
        m = _A_RE.match(ln)
        if m:
            lines_pairs.append((m.group(1), m.group(2)))
    for b_index, (actor_id, line_hint) in enumerate(lines_pairs, start=1):
        agenda_block = agenda_by_actor.get(actor_id, "")
        reaction_line = reactions_vals[b_index - 1] if 1 <= b_index <= len(reactions_vals) else ""
        # Pick body-language bullet for this line
        # simple nth bullet picker
        def _pick(text: str, n: int) -> str:
            cnt = [ln.lstrip()[1:].lstrip() for ln in text.splitlines() if ln.lstrip().startswith(('*', '-'))]
            if 1 <= n <= len(cnt):
                return f"* {cnt[n-1]}"
            return ""

        body_for_line = _pick(body_lang, b_index)
        agenda_combined = (agenda_block + ("\n" if agenda_block else "") + body_for_line).strip() if body_for_line.strip() else agenda_block

        combined_prompt = (
            f"Reaction: {reaction_line}\n"
            f"Line intent: {line_hint}\n"
        )
        dialog_lines_ctx = state.recent_dialog(actor_id)
        resp = render_character_call(
            actor_id,
            combined_prompt,
            dialog_lines_ctx,
            temperature=0.35,
            max_tokens_line=120,
            log_file=(log_dir / f"{tp_index:02d}_b{b_index:02d}_{actor_id}.txt") if log_dir else None,
            character_yaml=char_yaml_by_id.get(actor_id.strip().lower()),
            agenda=agenda_combined,
        )
        outputs.append(apply_body_template(body_for_line, resp))
        if resp.strip():
            state.add_dialog_line(actor_id, resp.strip())
            appended.setdefault(actor_id, []).append(resp.strip())
    text = "\n".join(outputs)
    # Polish
    from ..templates import build_polish_prompt
    polish_prompt = build_polish_prompt(ctx.setting, ctx.chapter, ctx.chapter_id, ctx.version, text)
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
    state.last_appended_dialog = appended
    return polished
