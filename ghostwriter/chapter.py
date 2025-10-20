"""Chapter-level orchestration helpers.

Includes parseable artifact helpers, story summary generation, and
the main per-chapter pipeline runner wrapper (importing pipelines).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable, Any
import os
import json
import re
import time

from .context import RunContext, GWError
from .templates import (
    iter_dir_for,
    read_latest,
    get_latest_version,
    build_check_prompt,
    build_story_so_far_prompt,
    build_story_relative_to_prompt,
)
from .utils import to_text as _gw_to_text, save_text, read_file, _norm_token
from .logging import breadcrumb as _breadcrumb, log_warning as _log_warning, crash_trace_file as _crash_trace_file
from .openai import llm_complete, get_model
from .characters import load_characters_list
from .env import collect_program_env_snapshot as _collect_program_env_snapshot
from .factoids import merge_setting_with_factoids as _merge_setting_with_factoids

# Imports from touch_point kept here to avoid circular dependencies elsewhere
from .env import env_for_prompt as _env_for_prompt
from .touch_point import (
    _apply_step,
    _build_pipeline_replacements,
    _ensure_brainstorm_done_on_resume,
)

try:
    from .pipelines import (
        run_narration_pipeline as gw_run_narration_pipeline,
        run_explicit_pipeline as gw_run_explicit_pipeline,
        run_implicit_pipeline as gw_run_implicit_pipeline,
        run_subtle_edit_pipeline as gw_run_subtle_edit_pipeline,
    )
except Exception:
    gw_run_narration_pipeline = None  # type: ignore
    gw_run_explicit_pipeline = None  # type: ignore
    gw_run_implicit_pipeline = None  # type: ignore
    gw_run_subtle_edit_pipeline = None  # type: ignore

try:
    from .artifacts import (
        format_draft_record as _gw_format_draft_record,
        write_records as _gw_write_records,
        read_records as _gw_read_records,
    )
    from .resume import scan_completed as _gw_scan_completed
except Exception:
    _gw_format_draft_record = None  # type: ignore
    _gw_write_records = None  # type: ignore
    _gw_read_records = None  # type: ignore
    _gw_scan_completed = None  # type: ignore


BEGIN_TP = "BEGIN_TOUCHPOINT"
END_TP = "END_TOUCHPOINT"
BEGIN_RES = "BEGIN_RESULT"
END_RES = "END_RESULT"


def format_draft_record(tp_id: str, tp_type: str, touchpoint_text: str, polished_text: str) -> str:
    return (
        f"{BEGIN_TP} id={tp_id} type={tp_type}\n"
        f"{touchpoint_text}\n"
        f"{END_TP}\n"
        f"{BEGIN_RES}\n"
        f"{polished_text}\n"
        f"{END_RES}\n\n"
    )


def write_draft_records(chapter_id: str, version: int, records: Iterable[Tuple[str, str, str, str]]) -> Path:
    out_path = iter_dir_for(chapter_id) / f"draft_v{version}.txt"
    if _gw_write_records is None:
        raise GWError("ghostwriter.artifacts.write_records not available")
    return _gw_write_records(out_path, records)


def read_draft_records(path: str) -> List[Dict[str, str]]:
    if _gw_read_records is None:
        raise GWError("ghostwriter.artifacts.read_records not available")
    return _gw_read_records(path)


def format_suggestions_record(tp_id: str, tp_type: str, touchpoint_text: str, suggestions_text: str) -> str:
    return format_draft_record(tp_id, tp_type, touchpoint_text, suggestions_text)


def write_suggestions_records(chapter_id: str, version: int, records: Iterable[Tuple[str, str, str, str]]) -> Path:
    out_path = iter_dir_for(chapter_id) / f"suggestions_v{version}.txt"
    chunks = [format_suggestions_record(*r) for r in records]
    save_text(out_path, "".join(chunks))
    return out_path


def read_suggestions_records(path: str) -> List[Dict[str, str]]:
    return read_draft_records(path)


def _generate_final_txt_from_records(chapter_id: str, records: List[Tuple[str, str, str, str]]) -> Path:
    final_text = []
    for (_tpid, _tptype, _touch, polished) in records:
        if polished is None:
            continue
        final_text.append(polished.strip())
    out_path = iter_dir_for(chapter_id) / "final.txt"
    save_text(out_path, "\n\n".join([t for t in final_text if t]))
    return out_path


def check_iteration_complete(check_text: str) -> bool:
    return "missing" not in check_text.lower()


def generate_story_so_far_and_relative(ctx: RunContext, pre_draft_text: str) -> None:
    setting = ctx.setting
    chapter = ctx.chapter
    chapter_id = ctx.chapter_id

    ssf_prompt = build_story_so_far_prompt(setting, chapter, chapter_id, ctx.version, pre_draft_text)
    ssf_model, ssf_temp, ssf_max_tokens = _env_for_prompt("story_so_far_prompt.md", "STORY_SO_FAR", default_temp=0.2, default_max_tokens=1200)
    ssf = llm_complete(
        ssf_prompt,
        system="You summarize story so far.",
        temperature=ssf_temp,
        max_tokens=ssf_max_tokens,
        model=ssf_model,
    )
    save_text(iter_dir_for(chapter_id) / "story_so_far.txt", ssf)

    srt_prompt = build_story_relative_to_prompt(setting, chapter, chapter_id, ctx.version, pre_draft_text)
    srt_model, srt_temp, srt_max_tokens = _env_for_prompt("story_relative_to_prompt.md", "STORY_RELATIVE", default_temp=0.2, default_max_tokens=1400)
    srt = llm_complete(
        srt_prompt,
        system="You summarize story relative to each character.",
        temperature=srt_temp,
        max_tokens=srt_max_tokens,
        model=srt_model,
    )
    save_text(iter_dir_for(chapter_id) / "story_relative_to.txt", srt)


TouchPoint = Dict[str, str]


def parse_touchpoints_from_chapter(chapter: dict) -> List[TouchPoint]:
    """Parse chapter['Touch-Points'] into a normalized list of touch-points with command and text.

    Accepts flexible YAML item shapes like:
      - explicit: "..."
      - implicit: "..."
      - narration: "..."
      - actors: ["henry", "jim"]   (stored as comma-separated string)
      - scene: "Battlefield"
      - foreshadowing: "Storm"
      - "explicit: Henry meets Jim" (string with 'key: value')

    Returns list of dicts with keys: id (1-based), type, content (string), raw (original-ish).
    """
    tps_raw = chapter.get("Touch-Points") or chapter.get("TouchPoints") or []
    result: List[TouchPoint] = []
    if not isinstance(tps_raw, list):
        return result
    allowed = {"actors", "scene", "foreshadowing", "narration", "explicit", "implicit", "setting"}

    def _normalize(item) -> List[TouchPoint]:
        # Dict form: {key: value}
        if isinstance(item, dict) and len(item) == 1:
            k = str(next(iter(item.keys()))).strip().lower()
            v = next(iter(item.values()))
            if k in allowed:
                if k == "actors" and isinstance(v, list):
                    content = ", ".join([str(x) for x in v])
                else:
                    content = str(v)
                return [{"type": k, "content": content, "raw": _gw_to_text(item).strip()}]
        # String form: "key: value"
        if isinstance(item, str):
            m = re.match(r"^\s*([A-Za-z_\-]+)\s*:\s*(.*)$", item.strip())
            if m:
                k = m.group(1).strip().lower()
                v = m.group(2).strip()
                if k in allowed:
                    return [{"type": k, "content": v, "raw": item}]
                # default to narration if unknown key
                return [{"type": "narration", "content": item.strip(), "raw": item}]
            # fallback: treat as narration
            return [{"type": "narration", "content": item.strip(), "raw": item}]
        # Other types: YAML scalars/seqs
        try:
            return [{"type": "narration", "content": str(item), "raw": _gw_to_text(item).strip()}]
        except Exception:
            return [{"type": "narration", "content": str(item), "raw": str(item)}]

    for idx, item in enumerate(tps_raw, start=1):
        for tp in _normalize(item):
            tp["id"] = str(idx)
            result.append(tp)
    return result


class ChapterState:
    """Processing state shared across touch-points."""
    def __init__(self, *, dialog_context_lines: int = 8) -> None:
        from collections import defaultdict, deque
        self.active_actors: List[str] = []
        self.current_scene: Optional[str] = None
        self.foreshadowing: List[str] = []
        self.dialog_history = defaultdict(lambda: deque(maxlen=dialog_context_lines))  # id -> deque[str]
        self.prior_context: Dict[str, Dict[str, str]] = {}  # touchpoint_id -> {polished_text, suggestions}
        self._dialog_maxlen = dialog_context_lines
        # New: selected context blocks populated by 'setting' touch-point
        self.setting_block: str = ""
        self.characters_block: str = ""
        # For checkpointing: lines appended by last explicit/implicit pipeline
        self.last_appended_dialog: Dict[str, List[str]] = {}

    def set_actors(self, actors_csv: str) -> None:
        self.active_actors = [a.strip() for a in re.split(r",|\s+", actors_csv) if a.strip()]

    def set_scene(self, scene: str) -> None:
        self.current_scene = scene.strip()

    def add_foreshadowing(self, item: str) -> None:
        val = item.strip()
        if val:
            self.foreshadowing.append(val)

    def add_dialog_line(self, actor_id: str, line: str) -> None:
        if not actor_id:
            return
        self.dialog_history[str(actor_id).strip().lower()].append(line)

    def recent_dialog(self, actor_id: str) -> List[str]:
        key = str(actor_id).strip().lower()
        return list(self.dialog_history.get(key, []))


def generate_pre_draft(chapter_path: str, version_num: int) -> Tuple[str, Path, str]:
    """Generate a pre_draft_vN using master_initial or master prompt."""
    from .templates import build_master_initial_prompt, build_master_prompt

    ctx = RunContext.from_paths(chapter_path=chapter_path, version=version_num)
    setting = ctx.setting
    chapter = ctx.chapter
    chapter_id = ctx.chapter_id

    # Decide prompt: initial if first version, else master
    latest = get_latest_version(chapter_id)
    use_initial = latest == 0 and version_num == 1
    if use_initial:
        prompt = build_master_initial_prompt(setting, chapter, chapter_id, version_num)
        tpl = "master_initial_prompt.md"
    else:
        prompt = build_master_prompt(setting, chapter, chapter_id, version_num)
        tpl = "master_prompt.md"

    pre_model, pre_temp, pre_max_tokens = _env_for_prompt(tpl, "PRE_DRAFT", default_temp=0.2, default_max_tokens=800)
    pre_draft = llm_complete(
        prompt,
        system="You are a Director creating pre-prose with character templates and call sites.",
        max_tokens=pre_max_tokens,
        temperature=pre_temp,
        model=pre_model,
    )

    out_path = iter_dir_for(chapter_id) / f"pre_draft_v{version_num}.txt"
    save_text(out_path, pre_draft)
    print(f"pre_draft saved to {out_path}")
    return pre_draft, out_path, chapter_id


def polish_prose(text_to_polish: str, chapter_path: str, version_num: int) -> Tuple[str, Path]:
    from .templates import build_polish_prompt

    ctx = RunContext.from_paths(chapter_path=chapter_path, version=version_num)
    setting = ctx.setting
    chapter = ctx.chapter
    chapter_id = ctx.chapter_id

    # Polish only; assumes dialog substitution already applied
    polish_prompt = build_polish_prompt(setting, chapter, chapter_id, version_num, text_to_polish)
    draft_model, draft_temp, draft_max_tokens = _env_for_prompt("polish_prose_prompt.md", "DRAFT", default_temp=0.2, default_max_tokens=2000)
    polished = llm_complete(
        polish_prompt,
        system="You are a ghostwriter polishing and cleaning prose.",
        temperature=draft_temp,
        max_tokens=draft_max_tokens,
        model=draft_model,
    )

    out_path = iter_dir_for(chapter_id) / f"draft_v{version_num}.txt"
    save_text(out_path, polished)
    print(f"draft saved to {out_path}")
    return polished, out_path


def verify_predraft(pre_draft_text: str, chapter_path: str, version_num: int) -> Tuple[str, Path]:
    ctx = RunContext.from_paths(chapter_path=chapter_path, version=version_num)
    setting = ctx.setting
    chapter = ctx.chapter
    chapter_id = ctx.chapter_id

    check_prompt = build_check_prompt(setting, chapter, chapter_id, version_num, pre_draft_text)
    check_model, check_temp, check_max_tokens = _env_for_prompt("check_prompt.md", "CHECK", default_temp=0.0, default_max_tokens=1200)
    check_results = llm_complete(
        check_prompt,
        system="You are an evaluator checking touch-point coverage.",
        temperature=check_temp,
        max_tokens=check_max_tokens,
        model=check_model,
    )

    # Save check results
    check_path = iter_dir_for(chapter_id) / f"check_v{version_num}.txt"
    save_text(check_path, check_results)
    print(f"check saved to {check_path}")

    # Save suggestions artifact as well; allow independent generation/length
    suggestions_path = iter_dir_for(chapter_id) / f"suggestions_v{version_num}.txt"
    try:
        sugg_model, sugg_temp, sugg_max_tokens = _env_for_prompt("check_prompt.md", "SUGGESTIONS", default_temp=0.0, default_max_tokens=800)
        # Minimal suggestions generation: reuse the check prompt to ask for actionable items only
        suggestions_prompt = (
            check_prompt
            + "\n\n---\nNow produce a concise, actionable list of suggested changes and fixes only. Do not restate the findings verbatim; output just the suggestions."
        )
        suggestions = llm_complete(
            suggestions_prompt,
            system="You are an evaluator extracting actionable suggestions only.",
            temperature=sugg_temp,
            max_tokens=sugg_max_tokens,
            model=sugg_model,
        )
    except Exception:
        # Fallback to mirroring the check output if generation fails
        suggestions = check_results
    save_text(suggestions_path, suggestions)
    # Keep log concise; suggestions often duplicate check content

    return check_results, check_path


def _tp_dir(base_log_dir: Optional[Path], index: int, tp_type: str) -> Optional[Path]:
    if base_log_dir is None:
        return None
    return base_log_dir / f"{index:02d}_{tp_type}"


def _write_tp_checkpoint(tp_dir: Optional[Path], tp_id: str, tp_type: str, tp_text: str, polished_text: str, state: ChapterState) -> None:
    if tp_dir is None:
        return
    try:
        tp_dir.mkdir(parents=True, exist_ok=True)
        save_text(tp_dir / "touch_point_draft.txt", polished_text)
        # Save minimal state deltas for reconstruction on resume
        cp = {
            "id": tp_id,
            "type": tp_type,
            "touchpoint": tp_text,
            "active_actors": state.active_actors,
            "scene": state.current_scene,
            "foreshadowing": state.foreshadowing,
            "setting_block": state.setting_block,
            "characters_block": state.characters_block,
            "appended_dialog": state.last_appended_dialog,
        }
        save_text(tp_dir / "touch_point_state.json", _gw_to_text(cp))
    except Exception:
        pass


def _resume_scan(base_log_dir: Optional[Path]) -> Dict[int, Dict[str, str]]:
    """Return map of completed tp_index -> info using ghostwriter.resume exclusively."""
    if base_log_dir is None or not base_log_dir.exists():
        return {}
    if _gw_scan_completed is None:
        raise GWError("ghostwriter.resume.scan_completed not available")
    try:
        return _gw_scan_completed(base_log_dir)
    except Exception as e:
        _log_warning(f"RESUME scan failed: {e}", base_log_dir)
        return {}


def _resume_apply_state(state: ChapterState, info: Dict[str, Any]) -> None:
    try:
        state.active_actors = list(info.get("active_actors", []) or [])
        sc = info.get("scene")
        if isinstance(sc, str):
            state.current_scene = sc
        fh = info.get("foreshadowing")
        if isinstance(fh, list):
            state.foreshadowing = list(fh)
        sb = info.get("setting_block")
        if isinstance(sb, str):
            state.setting_block = sb
        cb = info.get("characters_block")
        if isinstance(cb, str):
            state.characters_block = cb
        # Rebuild dialog history from appended_dialog
        app = info.get("appended_dialog") or {}
        if isinstance(app, dict):
            for aid, lines in app.items():
                if not isinstance(lines, list):
                    continue
                for ln in lines:
                    if isinstance(ln, str) and ln.strip():
                        state.add_dialog_line(aid, ln.strip())
    except Exception:
        pass


def run_pipelines_for_chapter(chapter_path: str, version_num: int, *, log_llm: bool = False) -> None:
    """Execute deterministic pipelines per touch-point and write artifacts for vN."""
    # Construct RunContext (this loads and validates YAML up front)
    ctx = RunContext.from_paths(chapter_path=chapter_path, version=version_num)
    setting = ctx.setting
    chapter = ctx.chapter
    chapter_id = ctx.chapter_id
    # Prepare log dir for pre-run warnings if enabled
    warn_log_dir = (iter_dir_for(chapter_id) / f"pipeline_v{version_num}") if log_llm else None

    # Validate Factoids parsing
    try:
        factoids = setting.get("Factoids") if isinstance(setting, dict) else None
        if not isinstance(factoids, list):
            _log_warning("SETTING.yaml: Factoids not parsed as a list.", warn_log_dir)
            factoids = []
        if len(factoids) == 0:
            _log_warning("SETTING.yaml: Factoids length is 0.", warn_log_dir)
    except Exception:
        _log_warning("SETTING.yaml: Error reading Factoids.", warn_log_dir)
        factoids = []

    # Validate Characters parsing (prefer CHARACTERS.yaml)
    try:
        characters = load_characters_list(ctx)
        if len(characters) == 0:
            _log_warning("CHARACTERS.yaml: Characters length is 0 or file missing.", warn_log_dir)
    except Exception:
        _log_warning("CHARACTERS.yaml: Error reading Characters.", warn_log_dir)
        characters = []

    # (Moved) Scene validation occurs after touch-point parsing

    # Decide branch based on prior drafts
    latest = get_latest_version(chapter_id)
    branch_b = latest > 0  # if any prior draft/suggestions exist, treat as edit branch
    prior_draft = read_latest(chapter_id, "draft") or ""
    prior_suggestions = read_latest(chapter_id, "suggestions") or ""

    # Parse touch-points and init state
    tps = parse_touchpoints_from_chapter(chapter)
    if not tps:
        _log_warning("CHAPTER: No Touch-Points parsed from chapter.", warn_log_dir)
        print("No Touch-Points found; nothing to do.")
        return

    # Scene presence and preview from touch-points
    try:
        scene_tps = [tp for tp in tps if tp.get("type") == "scene"]
        if not scene_tps:
            _log_warning("CHAPTER: No 'scene' touch-points parsed.", warn_log_dir)
        else:
            # Basic serialization/preview for first couple scenes
            if warn_log_dir is not None:
                outp = warn_log_dir / "scene_previews.txt"
                outp.parent.mkdir(parents=True, exist_ok=True)
                with outp.open("w", encoding="utf-8") as f:
                    for idx, stp in enumerate(scene_tps[:2], start=1):
                        content = stp.get("content", "")
                        raw = stp.get("raw", "")
                        f.write(f"-- Scene TP {idx} --\n")
                        if content:
                            f.write("[CONTENT]\n")
                            f.write(str(content)[:2000] + "\n\n")
                        if raw:
                            f.write("[RAW]\n")
                            f.write(str(raw)[:2000] + "\n\n")
                        if not content and not raw:
                            _log_warning("CHAPTER: A scene touch-point has no content or raw text.", warn_log_dir)
    except Exception:
        pass

    # Chapter-level setting match validations (if chapter provides a structured 'setting' entry)
    try:
        ch_setting = chapter.get("setting") if isinstance(chapter, dict) else None
        if isinstance(ch_setting, dict):
            factoid_names = ch_setting.get("factoids") if isinstance(ch_setting.get("factoids"), list) else []
            actor_names = ch_setting.get("actors") if isinstance(ch_setting.get("actors"), list) else []
            # Length warnings
            if not factoid_names:
                _log_warning("CHAPTER setting: Factoids present but length is 0 or not parsed as list.", warn_log_dir)
            if not actor_names:
                _log_warning("CHAPTER setting: Actors present but length is 0 or not parsed as list.", warn_log_dir)
            # Build lookup sets from SETTING/CHARACTERS
            fact_set = { _norm_token(f.get("name", "")) for f in factoids if isinstance(f, dict) }
            char_id_set = { _norm_token(c.get("id", "")) for c in characters if isinstance(c, dict) }
            # Compare, ignoring case and leading/trailing quotes
            for x in (factoid_names or []):
                if _norm_token(x) and _norm_token(x) not in fact_set:
                    _log_warning(f"CHAPTER setting: factoid '{x}' does not match any Factoids.name in SETTING.yaml.", warn_log_dir)
            for a in (actor_names or []):
                if _norm_token(a) and _norm_token(a) not in char_id_set:
                    _log_warning(f"CHAPTER setting: actor '{a}' does not match any Characters.id in CHARACTERS.yaml.", warn_log_dir)
    except Exception:
        # Non-fatal, continue
        pass
    try:
        dialog_ctx_n = int(os.getenv("GW_DIALOG_CONTEXT_LINES", "8"))
    except Exception:
        dialog_ctx_n = 8
    state = ChapterState(dialog_context_lines=dialog_ctx_n)

    # Logging
    base_log_dir = iter_dir_for(chapter_id) / f"pipeline_v{version_num}"
    base_log_dir.mkdir(parents=True, exist_ok=True)

    # Iterate touch-points and run pipelines
    total = len(tps)
    records: List[Tuple[str, str, str, str]] = []
    prior_paragraph = ""
    # Resume support: scan completed TPs and rebuild state
    completed = _resume_scan(base_log_dir)
    if completed:
        # Apply state in order and populate records/prior_paragraph
        for i in sorted(completed.keys()):
            info = completed[i]
            # Load state
            try:
                sp = info.get("state_path")
                if sp:
                    state_info = json.loads(read_file(sp))
                    _resume_apply_state(state, state_info)
            except Exception:
                pass
            # Load polished text for prior_paragraph and draft record
            try:
                polished = read_file(info.get("draft_path"))
            except Exception:
                polished = ""
            # Recreate basic record (tp_id unknown here; assign i)
            records.append((str(i), info.get("type", ""), "", polished))
            if polished.strip():
                prior_paragraph = polished.strip().splitlines()[-1]

    # Parsing complete marker: YAML files loaded and touch-points parsed
    try:
        msg = f"[v{version_num}] Parsing complete: SETTING.yaml + {chapter_path} loaded; {total} touch-points."
        print(msg)
        _breadcrumb("parse:complete")
        if base_log_dir is not None:
            base_log_dir.mkdir(parents=True, exist_ok=True)
            with (base_log_dir / "parse_complete.txt").open("w", encoding="utf-8") as f:
                f.write(msg + "\n")
                # Also record a small snapshot for debugging boundaries
                f.write(f"chapter_id={chapter_id}\n")
                f.write(f"timestamp={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
                f.write(f"cwd={os.getcwd()}\n")
                # Record full program environment snapshot for this run
                try:
                    snapshot = _collect_program_env_snapshot(get_model_cb=get_model, crash_trace_cb=_crash_trace_file)
                    f.write("\n[ENVIRONMENT]\n")
                    f.write(_gw_to_text(snapshot))
                    f.write("\n")
                except Exception:
                    pass
    except Exception:
        pass

    try:
        from .pipelines import (
            run_narration_pipeline as gw_run_narration_pipeline,
            run_explicit_pipeline as gw_run_explicit_pipeline,
            run_implicit_pipeline as gw_run_implicit_pipeline,
            run_subtle_edit_pipeline as gw_run_subtle_edit_pipeline,
        )
    except Exception:
        gw_run_narration_pipeline = None  # type: ignore
        gw_run_explicit_pipeline = None  # type: ignore
        gw_run_implicit_pipeline = None  # type: ignore
        gw_run_subtle_edit_pipeline = None  # type: ignore

    for i, tp in enumerate(tps, start=1):
        tp_type = tp.get("type", "")
        tp_id = tp.get("id", str(i))
        tp_text = tp.get("content", "")
        print(f"[v{version_num}] Touch-point {i}/{total} – {tp_type}: {tp_text[:60]}")
        tp_log_dir = (base_log_dir / f"{i:02d}_{tp_type}") if base_log_dir else None

        # Skip already completed touch-points by checkpoint presence
        if i in completed:
            _log_warning(f"RESUME: Skipping completed touch-point {i} ({tp_type}).", base_log_dir)
            # Even if draft exists, ensure brainstorm DONE gating is satisfied; if not, re-run brainstorm and exit
            try:
                reps_resume = _build_pipeline_replacements(setting, chapter, chapter_id, version_num, tp, state, prior_paragraph=prior_paragraph, ctx=ctx)
                _ensure_brainstorm_done_on_resume(tp_type, reps_resume, log_dir=tp_log_dir, tp_index=i)
            except Exception:
                pass
            continue

        polished_text = ""
        if tp_type == "actors":
            state.set_actors(tp.get("content", ""))
        elif tp_type == "scene":
            state.set_scene(tp.get("content", ""))
        elif tp_type == "foreshadowing":
            state.add_foreshadowing(tp.get("content", ""))
        elif tp_type == "setting":
            # Do not re-parse YAML here; store the content and compute character subset from active state
            try:
                state.setting_block = tp_text.strip()
            except Exception:
                state.setting_block = ""
            try:
                sel_chars: List[dict] = []
                all_chars = load_characters_list(ctx)
                if isinstance(all_chars, list) and all_chars and state.active_actors:
                    wanted = {a.strip().lower() for a in state.active_actors if a.strip()}
                    for ch in all_chars:
                        cid = str(ch.get("id", "")).strip().lower()
                        cname = str(ch.get("name", "")).strip().lower()
                        if cid in wanted or cname in wanted:
                            sel_chars.append(ch)
                state.characters_block = _gw_to_text({"Selected-Characters": sel_chars}) if sel_chars else ""
            except Exception:
                state.characters_block = ""
        elif tp_type in ("narration", "explicit", "implicit"):
            # Before running pipelines, if brainstorm exists but lacks DONE, enforce resume gating
            try:
                reps_pre = _build_pipeline_replacements(setting, chapter, chapter_id, version_num, tp, state, prior_paragraph=prior_paragraph, ctx=ctx)
                _ensure_brainstorm_done_on_resume(tp_type, reps_pre, log_dir=tp_log_dir, tp_index=i)
            except Exception:
                pass
            if tp_type == "narration":
                if branch_b:
                    # Use previous version's per-touchpoint artifacts if available
                    prev_polished_tp = ""
                    prev_suggestions_tp = ""
                    try:
                        prev_dir = iter_dir_for(chapter_id) / f"pipeline_v{version_num-1}"
                        prev_tp_dir = prev_dir / f"{i:02d}_{tp_type}"
                        if prev_tp_dir.exists():
                            try:
                                prev_polished_tp = read_file(str(prev_tp_dir / "touch_point_draft.txt"))
                            except Exception:
                                prev_polished_tp = ""
                            try:
                                prev_suggestions_tp = read_file(str(prev_tp_dir / "suggestions.txt"))
                            except Exception:
                                prev_suggestions_tp = ""
                    except Exception:
                        pass
                    use_pol = prev_polished_tp or prior_draft
                    use_sugg = prev_suggestions_tp or prior_suggestions
                    # Use package subtle edit pipeline exclusively
                    if gw_run_subtle_edit_pipeline is None:
                        raise GWError("ghostwriter.pipelines.run_subtle_edit_pipeline not available")
                    polished_text = gw_run_subtle_edit_pipeline(tp, state, setting=setting, chapter=chapter, chapter_id=chapter_id, version=version_num, tp_index=i, prior_polished=use_pol, prior_suggestions=use_sugg, prior_paragraph=prior_paragraph, log_dir=tp_log_dir)
                else:
                    if gw_run_narration_pipeline is None:
                        raise GWError("ghostwriter.pipelines.run_narration_pipeline not available")
                    polished_text = gw_run_narration_pipeline(tp, state, ctx=ctx, tp_index=i, prior_paragraph=prior_paragraph, log_dir=tp_log_dir)
            elif tp_type == "explicit":
                if branch_b:
                    prev_polished_tp = ""
                    prev_suggestions_tp = ""
                    try:
                        prev_dir = iter_dir_for(chapter_id) / f"pipeline_v{version_num-1}"
                        prev_tp_dir = prev_dir / f"{i:02d}_{tp_type}"
                        if prev_tp_dir.exists():
                            try:
                                prev_polished_tp = read_file(str(prev_tp_dir / "touch_point_draft.txt"))
                            except Exception:
                                prev_polished_tp = ""
                            try:
                                prev_suggestions_tp = read_file(str(prev_tp_dir / "suggestions.txt"))
                            except Exception:
                                prev_suggestions_tp = ""
                    except Exception:
                        pass
                    use_pol = prev_polished_tp or prior_draft
                    use_sugg = prev_suggestions_tp or prior_suggestions
                    if gw_run_subtle_edit_pipeline is None:
                        raise GWError("ghostwriter.pipelines.run_subtle_edit_pipeline not available")
                    polished_text = gw_run_subtle_edit_pipeline(tp, state, setting=setting, chapter=chapter, chapter_id=chapter_id, version=version_num, tp_index=i, prior_polished=use_pol, prior_suggestions=use_sugg, prior_paragraph=prior_paragraph, log_dir=tp_log_dir)
                else:
                    if gw_run_explicit_pipeline is None:
                        raise GWError("ghostwriter.pipelines.run_explicit_pipeline not available")
                    polished_text = gw_run_explicit_pipeline(tp, state, ctx=ctx, tp_index=i, prior_paragraph=prior_paragraph, log_dir=tp_log_dir)
            else:  # implicit
                if branch_b:
                    prev_polished_tp = ""
                    prev_suggestions_tp = ""
                    try:
                        prev_dir = iter_dir_for(chapter_id) / f"pipeline_v{version_num-1}"
                        prev_tp_dir = prev_dir / f"{i:02d}_{tp_type}"
                        if prev_tp_dir.exists():
                            try:
                                prev_polished_tp = read_file(str(prev_tp_dir / "touch_point_draft.txt"))
                            except Exception:
                                prev_polished_tp = ""
                            try:
                                prev_suggestions_tp = read_file(str(prev_tp_dir / "suggestions.txt"))
                            except Exception:
                                prev_suggestions_tp = ""
                    except Exception:
                        pass
                    use_pol = prev_polished_tp or prior_draft
                    use_sugg = prev_suggestions_tp or prior_suggestions
                    if gw_run_subtle_edit_pipeline is None:
                        raise GWError("ghostwriter.pipelines.run_subtle_edit_pipeline not available")
                    polished_text = gw_run_subtle_edit_pipeline(tp, state, setting=setting, chapter=chapter, chapter_id=chapter_id, version=version_num, tp_index=i, prior_polished=use_pol, prior_suggestions=use_sugg, prior_paragraph=prior_paragraph, log_dir=tp_log_dir)
                else:
                    if gw_run_implicit_pipeline is None:
                        raise GWError("ghostwriter.pipelines.run_implicit_pipeline not available")
                    polished_text = gw_run_implicit_pipeline(tp, state, ctx=ctx, tp_index=i, prior_paragraph=prior_paragraph, log_dir=tp_log_dir)
        else:
            # Unknown types treated as narration by default
            if branch_b:
                if gw_run_subtle_edit_pipeline is None:
                    raise GWError("ghostwriter.pipelines.run_subtle_edit_pipeline not available")
                polished_text = gw_run_subtle_edit_pipeline(tp, state, setting=setting, chapter=chapter, chapter_id=chapter_id, version=version_num, prior_polished=prior_draft, prior_suggestions=prior_suggestions, log_dir=tp_log_dir)
            else:
                if gw_run_narration_pipeline is None:
                    raise GWError("ghostwriter.pipelines.run_narration_pipeline not available")
                polished_text = gw_run_narration_pipeline(tp, state, ctx=ctx, tp_index=i, prior_paragraph=prior_paragraph, log_dir=tp_log_dir)

        # Record (for actors/scene/foreshadowing, polished_text may be empty)
        records.append((tp_id, tp_type, tp_text, polished_text))
        # After producing a polished draft for contentful types, run per-touch-point check and suggestions
        if tp_type in ("narration", "explicit", "implicit") and polished_text and tp_log_dir is not None:
            try:
                # Build replacements expected by check_* prompts
                # Setting block
                setting_block = getattr(state, "setting_block", "") or ""
                # Characters block: prefer state.characters_block, else select by active actors
                characters_block = getattr(state, "characters_block", "") or ""
                if not characters_block and state.active_actors:
                    try:
                        sel_chars = []
                        all_chars = setting.get("Characters") if isinstance(setting, dict) else None
                        if isinstance(all_chars, list):
                            wanted = {a.strip().lower() for a in state.active_actors if a.strip()}
                            for ch in all_chars:
                                cid = str(ch.get("id", "")).strip().lower()
                                cname = str(ch.get("name", "")).strip().lower()
                                if cid in wanted or cname in wanted:
                                    sel_chars.append(ch)
                        if sel_chars:
                            characters_block = _gw_to_text({"Selected-Characters": sel_chars})
                    except Exception:
                        characters_block = characters_block or ""
                # Recent context: last up to 3 polished segments
                try:
                    recent_polished = []
                    for (_tid, _ttype, _touch, _pol) in records[-4:-1]:
                        if _pol and str(_pol).strip():
                            recent_polished.append(str(_pol).strip())
                    context_block = "\n\n---\n\n".join(recent_polished)
                except Exception:
                    context_block = ""
                # Foreshadowing and actors
                foreshadowing = ", ".join(state.foreshadowing)
                actors_csv = ", ".join(state.active_actors)
                # Choose check template per tp type
                check_tpl = {
                    "narration": "check_narration_prompt.md",
                    "explicit": "check_explicit_prompt.md",
                    "implicit": "check_implicit_prompt.md",
                }.get(tp_type, "check_narration_prompt.md")
                # Prepare replacements
                from .templates import build_common_replacements
                check_reps = build_common_replacements(setting, chapter, chapter_id, version_num)
                check_reps.update({
                    "[SETTING]": _merge_setting_with_factoids(setting_block, setting, chapter=chapter),
                    "[CHARACTERS]": characters_block,
                    "[context]": context_block,
                    "[foreshadowing]": foreshadowing,
                    "[actors]": actors_csv,
                    "[TOUCH_POINT]": tp_text,
                    "[prose]": polished_text,
                })
                check_user = _apply_step(check_tpl, check_reps)
                check_model, check_temp, check_max_tokens = _env_for_prompt(check_tpl, "CHECK", default_temp=0.0, default_max_tokens=1200)
                check_sys = "You are an evaluator checking touch-point coverage."
                check_out = llm_complete(check_user, system=check_sys, temperature=check_temp, max_tokens=check_max_tokens, model=check_model)
                # Compose check.txt with full prompt and response for easier review
                check_file_text = (
                    "=== SYSTEM ===\n" + (check_sys or "") +
                    "\n\n=== USER ===\n" + (check_user or "") +
                    "\n\n=== RESPONSE ===\n" + (check_out or "") + "\n"
                )
                # Suggestions: reuse the same prompt with an appended instruction
                sugg_tpl = check_tpl
                sugg_reps = dict(check_reps)
                suggestions_user = check_user + "\n\n---\nNow produce a concise, actionable list of suggested changes and fixes only. Do not restate the findings verbatim; output just the suggestions."
                sugg_model, sugg_temp, sugg_max_tokens = _env_for_prompt(sugg_tpl, "SUGGESTIONS", default_temp=0.0, default_max_tokens=800)
                suggestions_out = llm_complete(suggestions_user, system="You are an evaluator extracting actionable suggestions only.", temperature=sugg_temp, max_tokens=sugg_max_tokens, model=sugg_model)
                # Write files next to touch_point_draft.txt
                try:
                    tp_log_dir.mkdir(parents=True, exist_ok=True)
                    save_text(tp_log_dir / "check.txt", check_file_text)
                    save_text(tp_log_dir / "suggestions.txt", suggestions_out)
                except Exception:
                    pass
            except Exception:
                pass
        # Write checkpoint for resume
        _write_tp_checkpoint(tp_log_dir, tp_id, tp_type, tp_text, polished_text, state)
        # Update prior_paragraph to the latest polished segment (for templates needing previous context)
        if polished_text:
            prior_paragraph = polished_text.strip().splitlines()[-1] if polished_text.strip() else ""

    # Write draft_vN.txt (parseable)
    out_path = write_draft_records(chapter_id, version_num, records)
    print(f"draft saved to {out_path}")

    # Write final.txt (polished only)
    final_path = _generate_final_txt_from_records(chapter_id, records)
    print(f"final saved to {final_path}")

    # Regenerate summaries using final text
    if os.getenv("GW_SKIP_SUMMARIES", "0") != "1":
        full_text = (final_path.read_text(encoding="utf-8") if Path(final_path).exists() else "")
        generate_story_so_far_and_relative(ctx, full_text)
        print("Regenerated story_so_far.txt and story_relative_to.txt")
