"""Chapter-level orchestration helpers."""
try:
    from .pipelines import (
        run_narration_pipeline as gw_run_narration_pipeline,
        run_dialog_pipeline as gw_run_dialog_pipeline,
        run_implicit_pipeline as gw_run_implicit_pipeline,
        run_mixed_pipeline as gw_run_mixed_pipeline,
        run_subtle_edit_pipeline as gw_run_subtle_edit_pipeline,
    )
except Exception:
    gw_run_narration_pipeline = None  # type: ignore
    gw_run_dialog_pipeline = None  # type: ignore
    gw_run_implicit_pipeline = None  # type: ignore
    gw_run_mixed_pipeline = None  # type: ignore
    gw_run_subtle_edit_pipeline = None  # type: ignore

from .music import (
    build_voice_context as gw_music_build_voice_context,
    ensure_first_score_gate as gw_music_first_gate,
    run_subtle_score_pass as gw_music_subtle_pass,
    finalize_music_exports as gw_music_finalize_exports,
    # New metadata/tracks helper is exported via music.__all__
    run_metadata_tracks_step as gw_music_metadata_tracks,
    run_melody_edges_step as gw_music_melody_edges,
    run_melody_construction_step as gw_music_melody_construct,
    run_multi_voice_first_pass as gw_music_multi_voice_first_pass,
    assemble_first_pass_variant as gw_music_assemble_first_variant,
)

# Standard library imports
import os
import re
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import copy

from .context import RunContext, GWError, UserActionRequired
from .templates import (
    iter_dir_for,
    read_latest,
    get_latest_version,
    build_check_prompt,
    build_story_so_far_prompt,
    build_story_relative_to_prompt,
)
from .utils import to_text as _gw_to_text, save_text, read_file, _norm_token
from .logging import breadcrumb as _breadcrumb, log_warning as _log_warning, log_info as _log_info, crash_trace_file as _crash_trace_file, log_error_base as _log_error_base, log_run as _log_run
# Route all LLM calls through the central llm module so GW_USE_MOCK_LLM is consistently honored
from .llm import complete as llm_complete, get_model
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

    # In strict MockLLM mode, avoid calling the LLM for summaries since goldens typically
    # don't index these prompts. If summaries already exist for this chapter, keep them.
    import os as _os
    from .templates import iter_dir_for as _iter_dir_for
    base_dir = _iter_dir_for(chapter_id)
    if (_os.getenv("GW_USE_MOCK_LLM", "0") == "1"):
        ssf_path = base_dir / "story_so_far.txt"
        srt_path = base_dir / "story_relative_to.txt"
        if ssf_path.exists() and srt_path.exists():
            return
        # Deterministic, non-LLM fallback: write minimal stubs based on the current draft
        try:
            head = (pre_draft_text or "").strip()
            if head:
                head = head[:2000]
            else:
                head = ""
            ssf_path.parent.mkdir(parents=True, exist_ok=True)
            ssf_path.write_text(head, encoding="utf-8")
            srt_path.write_text(head, encoding="utf-8")
            return
        except Exception:
            # If writing stubs fails, fall through to normal path (may be skipped later)
            pass

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


def _tp_dir_path(chapter_id: str, version_num: int, tp_index: int, tp_type: str) -> Path:
    return iter_dir_for(chapter_id) / f"pipeline_v{version_num}" / f"{tp_index:02d}_{tp_type}"


def _read_tp_draft_or_empty(tp_dir: Path) -> str:
    try:
        return read_file(str(tp_dir / "touch_point_draft.txt"))
    except Exception:
        return ""


def reconcile_chapter_global_edits(ctx: RunContext, version_num: int) -> bool:
    """Propagate edits from draft_vN.txt into per-touch-point drafts and refresh suggestions.

    Returns True if any per-touch-point draft was updated.
    """
    chapter_id = ctx.chapter_id
    # Always reconcile against the latest existing draft_vN.txt (not necessarily the run version)
    try:
        latest_existing = get_latest_version(chapter_id)
    except Exception:
        latest_existing = 0
    if latest_existing <= 0:
        return False
    target_version = latest_existing
    draft_path = iter_dir_for(chapter_id) / f"draft_v{target_version}.txt"
    if not draft_path.exists():
        return False
    try:
        records = read_draft_records(str(draft_path))
    except Exception:
        return False
    if not records:
        return False
    any_changes = False
    # Iterate records in order and compare with pipeline drafts
    # Maintain a rolling context of prior polished to build suggestions' [context]
    polished_so_far: List[str] = []
    for rec in records:
        tp_id = rec.get("id", "").strip()
        tp_type = rec.get("type", "").strip()
        tp_text = rec.get("touchpoint", "")
        polished = rec.get("result", "")
        if not tp_id or not tp_type:
            continue
        try:
            tp_index = int(tp_id)
        except Exception:
            # Fallback: try to parse numeric prefix
            try:
                tp_index = int(tp_id.split("_")[0])
            except Exception:
                continue
        tp_dir = _tp_dir_path(chapter_id, target_version, tp_index, tp_type)
        if not tp_dir.exists():
            # Nothing to sync
            if polished.strip():
                polished_so_far.append(polished.strip())
            continue
        current = _read_tp_draft_or_empty(tp_dir)
        if (current or "").strip() != (polished or "").strip():
            # Overwrite draft to reflect global edit
            try:
                tp_dir.mkdir(parents=True, exist_ok=True)
                save_text(tp_dir / "touch_point_draft.txt", polished)
                any_changes = True
            except Exception:
                pass
            # Rebuild suggestions for this touch-point using stored state and nearby context
            try:
                # Load minimal state snapshot captured at pipeline time, if available
                try:
                    import json as _json
                    st_path = tp_dir / "touch_point_state.json"
                    st_info = _json.loads(read_file(str(st_path))) if st_path.exists() else {}
                except Exception:
                    st_info = {}
                foreshadowing = ", ".join(st_info.get("foreshadowing", []) or [])
                actors_csv = ", ".join(st_info.get("active_actors", []) or [])
                setting_block = st_info.get("setting_block", "") or ""
                characters_block = st_info.get("characters_block", "") or ""
                # If characters block is missing, select from CHARACTERS by active actors
                if not characters_block:
                    try:
                        selected: List[dict] = []
                        all_chars = load_characters_list(ctx)
                        if isinstance(all_chars, list) and all_chars:
                            active = [a.strip() for a in (st_info.get("active_actors", []) or []) if a.strip()]
                            if active:
                                wanted = {a.lower() for a in active}
                                for ch in all_chars:
                                    cid = str(ch.get("id", "")).strip().lower()
                                    cname = str(ch.get("name", "")).strip().lower()
                                    if cid in wanted or cname in wanted:
                                        selected.append(ch)
                            else:
                                selected = list(all_chars)
                        if selected:
                            characters_block = _gw_to_text({"Selected-Characters": selected})
                    except Exception:
                        characters_block = characters_block or ""
                # Build short context from recent polished
                try:
                    recent_polished = []
                    for chunk in polished_so_far[-3:]:
                        if chunk and str(chunk).strip():
                            recent_polished.append(str(chunk).strip())
                    context_block = "\n\n---\n\n".join(recent_polished)
                except Exception:
                    context_block = ""
                # Template selection and replacements
                check_tpl = {
                    "narration": "check_narration_prompt.md",
                    "dialog": "check_dialog_prompt.md",
                    "implicit": "check_implicit_prompt.md",
                    "mixed": "check_mixed_prompt.md",
                }.get(tp_type, "check_narration_prompt.md")
                from .templates import build_common_replacements
                check_reps = build_common_replacements(ctx.setting, ctx.chapter, ctx.chapter_id, target_version)
                check_reps.update({
                    "[SETTING]": _merge_setting_with_factoids(setting_block, ctx.setting, chapter=ctx.chapter),
                    "[CHARACTERS]": characters_block,
                    "[context]": context_block,
                    "[foreshadowing]": foreshadowing,
                    "[actors]": actors_csv,
                    "[TOUCH_POINT]": tp_text,
                    "[prose]": polished,
                })
                check_user = _apply_step(check_tpl, check_reps)
                suggestions_user = check_user + "\n\n---\nNow produce a concise, actionable list of suggested changes and fixes only. Do not restate the findings verbatim; output just the suggestions."
                sugg_model, sugg_temp, sugg_max_tokens = _env_for_prompt(check_tpl, "SUGGESTIONS", default_temp=0.0, default_max_tokens=800)
                suggestions_out = llm_complete(
                    suggestions_user,
                    system="You are an evaluator extracting actionable suggestions only.",
                    temperature=sugg_temp,
                    max_tokens=sugg_max_tokens,
                    model=sugg_model,
                )
                # Save suggestions and a check.txt debug mirror of the single prompt+response
                try:
                    with (tp_dir / "check.txt").open("w", encoding="utf-8") as f:
                        f.write("=== SYSTEM ===\nYou are an evaluator extracting actionable suggestions only.\n\n")
                        f.write("=== USER ===\n" + suggestions_user + "\n\n")
                        f.write("=== RESPONSE ===\n" + (suggestions_out or "") + "\n")
                except Exception:
                    pass
                save_text(tp_dir / "suggestions.txt", suggestions_out or "")
            except Exception:
                # Do not fail reconciliation on suggestions errors
                pass
        # Update rolling context
        if polished.strip():
            polished_so_far.append(polished.strip())

    # Always regenerate final.txt from current per-TP drafts
    try:
        rec_tuples: List[Tuple[str, str, str, str]] = []
        for rec in records:
            tp_id = rec.get("id", "").strip()
            tp_type = rec.get("type", "").strip()
            tp_text = rec.get("touchpoint", "")
            try:
                tp_index = int(tp_id)
            except Exception:
                try:
                    tp_index = int(tp_id.split("_")[0])
                except Exception:
                    tp_index = 0
            tp_dir = _tp_dir_path(chapter_id, target_version, tp_index, tp_type) if (tp_id and tp_type) else None
            pol = _read_tp_draft_or_empty(tp_dir) if tp_dir else rec.get("result", "")
            rec_tuples.append((tp_id, tp_type, tp_text, pol))
        final_path = _generate_final_txt_from_records(chapter_id, rec_tuples)
        try:
            _log_info(f"Global edits: rebuilt final.txt from v{target_version} per-TP drafts at {final_path}")
        except Exception:
            pass
    except Exception as e:
        try:
            # Provide a log_dir for structured logging to satisfy type checkers
            _log_warning(f"Global edits: failed to rebuild final.txt: {e}", iter_dir_for(chapter_id))
        except Exception:
            pass

    # Regenerate summaries if missing
    try:
        base = iter_dir_for(chapter_id)
        ssf_missing = not (base / "story_so_far.txt").exists()
        srt_missing = not (base / "story_relative_to.txt").exists()
        if ssf_missing or srt_missing:
            full_text = ""
            try:
                full_text = (base / "final.txt").read_text(encoding="utf-8")
            except Exception:
                full_text = ""
            generate_story_so_far_and_relative(ctx, full_text)
    except Exception:
        pass

    return any_changes


TouchPoint = Dict[str, Any]


def parse_touchpoints_from_chapter(chapter: dict) -> List[TouchPoint]:
    """Parse chapter['Touch-Points'] into a normalized list of touch-points with command and text.

    Accepts flexible YAML item shapes like:
        - dialog: "..."
        - implicit: "..."
        - narration: "..."
        - actors: ["henry", "jim"]   (stored as comma-separated string)
        - scene: "Battlefield"
        - foreshadowing: "Storm"
        - "dialog: Henry meets Jim" (string with 'key: value')

    Returns list of dicts with keys: id (1-based), type, content (string), raw (original-ish).
    """
    tps_raw = chapter.get("Touch-Points") or chapter.get("TouchPoints") or []
    result: List[TouchPoint] = []
    if not isinstance(tps_raw, list):
        return result
    # Supported touch-point types
    allowed = {"actors", "scene", "foreshadowing", "narration", "implicit", "mixed", "setting", "dialog", "voices", "music"}

    def _normalize(item) -> List[TouchPoint]:
        # Dict form: {key: value}
        if isinstance(item, dict) and len(item) == 1:
            k = str(next(iter(item.keys()))).strip().lower()
            v = next(iter(item.values()))
            if k in allowed:
                if k == "actors" and isinstance(v, list):
                    content = ", ".join([str(x) for x in v])
                    return [{"type": k, "content": content, "raw": _gw_to_text(item).strip()}]
                if k == "music":
                    payload: Dict[str, Any]
                    if isinstance(v, dict):
                        payload = copy.deepcopy(v)
                    else:
                        payload = {"description": str(v) if v is not None else ""}
                    title = str(payload.get("title", "") or "").strip()
                    description = str(payload.get("description", "") or "").strip()
                    if not description and isinstance(v, dict):
                        description = _gw_to_text(v).strip()
                    content = description or title or _gw_to_text(item).strip()
                    tp_entry = {"type": k, "content": content, "raw": _gw_to_text(item).strip()}
                    if payload:
                        tp_entry["payload"] = payload  # type: ignore[assignment]
                    return [tp_entry]
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


def _flatten_voice_tokens(value: Any) -> List[str]:
    tokens: List[str] = []

    def _collect(node: Any) -> None:
        if node is None:
            return
        if isinstance(node, str):
            token = node.strip()
            if token:
                tokens.append(token)
            return
        if isinstance(node, dict):
            token_val = node.get("token") or node.get("voice")
            if isinstance(token_val, str) and token_val.strip():
                tokens.append(token_val.strip())
            nested = node.get("tokens") or node.get("voices")
            if isinstance(nested, (list, tuple, set)):
                for sub in nested:
                    _collect(sub)
            return
        if isinstance(node, (list, tuple, set)):
            for sub in node:
                _collect(sub)

    _collect(value)
    return tokens


def _build_music_prompt_payload_for_tp(
    ctx: RunContext,
    version_num: int,
    tp: TouchPoint,
    prior_paragraph: str,
) -> Tuple[Optional[Any], Dict[str, Any], Dict[str, Any]]:
    payload_data = tp.get("payload") if isinstance(tp.get("payload"), dict) else {}
    if not isinstance(payload_data, dict):
        payload_data = {}

    title = str(payload_data.get("title") or "").strip()
    description = str(payload_data.get("description") or "").strip()
    directive_override = str(payload_data.get("directive") or "").strip()
    if not description:
        description = str(tp.get("content", "") or "").strip()
    if not directive_override:
        directive_override = description or title

    voices_source = payload_data.get("voices")
    voice_input = copy.deepcopy(voices_source) if voices_source is not None else None
    voice_tokens_flat = _flatten_voice_tokens(voice_input if voice_input is not None else voices_source)
    if voice_input is None:
        voice_input = voice_tokens_flat

    # Basic validation: music touch-points must provide title, description, and voices.
    if not title or not description or not voices_source:
        raise UserActionRequired(
            "Music touch-point payload is missing required fields. "
            "Expected mapping form with 'title', 'description', and 'voices'. "
            "Example:\n  - music:\n      title: The Forest Path\n      description: ...\n      voices: [voice.one, voice.two]"
        )

    voice_context = None
    prompt_payload: Dict[str, Any] = {}
    if gw_music_build_voice_context is not None:
        try:
            voice_context = gw_music_build_voice_context(
                ctx,
                pipeline_version=version_num,
                voice_tokens=voice_input,
                directive=directive_override or title,
            )
        except TypeError as exc:
            raise UserActionRequired(
                f"Music voice-context builder received unexpected arguments: {exc}. "
                "Check your CHAPTER_XXX.yaml music 'voices' entry."
            ) from exc
        except UserActionRequired:
            # Bubble up structured validation errors from the music context layer unchanged
            raise
        except Exception as exc:
            # Any unexpected failure in voice context construction should halt the run
            # with a clear, actionable message instead of silently skipping music.
            raise UserActionRequired(
                f"Music voice-context construction failed: {exc}. "
                "Verify the music 'voices' section and any related character/factoid references."
            ) from exc

    if voice_context is not None:
        prompt_payload = copy.deepcopy(voice_context.as_prompt_payload())
    else:
        prompt_payload = {}

    prompt_payload.setdefault("directive", directive_override or title or "")
    prompt_payload["touch_point_title"] = title
    prompt_payload["touch_point_description"] = description
    prompt_payload["touch_point_prior_paragraph"] = prior_paragraph or ""
    prompt_payload["touch_point_voices"] = list(voice_tokens_flat)
    prompt_payload["touch_point_metadata"] = copy.deepcopy(payload_data)

    meta = {
        "title": title,
        "description": description,
        "directive": directive_override,
        "voice_tokens": list(voice_tokens_flat),
        "raw_payload": copy.deepcopy(payload_data),
    }

    return voice_context, prompt_payload, meta


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
    # For checkpointing: lines appended by last dialog/implicit pipeline
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

    music_voice_context = None
    if gw_music_build_voice_context is not None:
        try:
            mvc = gw_music_build_voice_context(ctx, pipeline_version=version_num)
            if mvc and getattr(mvc, "voices", None):
                music_voice_context = mvc
        except Exception as exc:
            _log_warning(f"MUSIC: failed to build voice context ({exc})", base_log_dir)
            music_voice_context = None

    # Chapter Global Editing: if user edited draft_vN.txt, propagate to per-touch-point drafts and refresh suggestions
    try:
        changed = reconcile_chapter_global_edits(ctx, version_num)
        if changed:
            # Stop gracefully to allow user to review/edit regenerated suggestions before next revision
            try:
                latest_existing = get_latest_version(chapter_id)
            except Exception:
                latest_existing = version_num
            raise UserActionRequired(f"Reconciled version v{latest_existing} and regenerated suggestions.")
    except UserActionRequired:
        # Bubble up to CLI for a clean exit and message
        raise
    except Exception as e:
        _log_warning(f"Global edit reconciliation skipped due to error: {e}", base_log_dir)

    # Iterate touch-points and run pipelines
    total = len(tps)
    records: List[Tuple[str, str, str, str]] = []
    prior_paragraph = ""
    # Resume support: scan completed TPs and rebuild state
    _breadcrumb("resume:scan:start")
    completed = _resume_scan(base_log_dir)
    if completed:
        try:
            _breadcrumb(f"resume:scan:found:{sorted(completed.keys())}")
        except Exception:
            pass
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
            tp_type_i = info.get("type", "")
            try:
                dp = info.get("draft_path")
                polished = read_file(str(dp)) if dp else ""
            except Exception:
                polished = ""
            # Recreate basic record (tp_id unknown here; assign i)
            records.append((str(i), tp_type_i, "", polished))
            # Only contentful types (narration/dialog/implicit/mixed) should influence prior_paragraph
            if tp_type_i in ("narration", "dialog", "implicit", "mixed") and polished.strip():
                prior_paragraph = polished.strip()

    # Parsing complete marker: YAML files loaded and touch-points parsed
    try:
        msg = f"[v{version_num}] Parsing complete: SETTING.yaml + {chapter_path} loaded; {total} touch-points."
        print(msg)
        _breadcrumb("parse:complete")
        # Write parse and environment snapshot under base dir
        from .env import get_book_base_dir as _get_book_base_dir
        try:
            base = _get_book_base_dir()
            base.mkdir(parents=True, exist_ok=True)
            with (base / "parse_and_env.log").open("w", encoding="utf-8") as f:
                f.write(msg + "\n")
                f.write(f"chapter_id={chapter_id}\n")
                f.write(f"timestamp={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
                f.write(f"cwd={os.getcwd()}\n")
                try:
                    snapshot = _collect_program_env_snapshot(get_model_cb=get_model, crash_trace_cb=_crash_trace_file)
                    f.write("\n[ENVIRONMENT]\n")
                    f.write(_gw_to_text(snapshot))
                    f.write("\n")
                except Exception:
                    pass
        except Exception:
            pass
    except Exception:
        pass

    try:
        from .pipelines import (
            run_narration_pipeline as gw_run_narration_pipeline,
            run_dialog_pipeline as gw_run_dialog_pipeline,
            run_implicit_pipeline as gw_run_implicit_pipeline,
            run_mixed_pipeline as gw_run_mixed_pipeline,
            run_subtle_edit_pipeline as gw_run_subtle_edit_pipeline,
        )
    except Exception:
        gw_run_narration_pipeline = None  # type: ignore
        gw_run_dialog_pipeline = None  # type: ignore
        gw_run_implicit_pipeline = None  # type: ignore
        gw_run_mixed_pipeline = None  # type: ignore
        gw_run_subtle_edit_pipeline = None  # type: ignore

    for i, tp in enumerate(tps, start=1):
        tp_type = tp.get("type", "")
        tp_id = tp.get("id", str(i))
        tp_text = tp.get("content", "")
        print(f"[v{version_num}] Touch-point {i}/{total} – {tp_type}: {tp_text[:60]}")
        tp_log_dir = (base_log_dir / f"{i:02d}_{tp_type}") if base_log_dir else None

        music_tp_voice_context = None
        music_tp_prompt_payload: Optional[Dict[str, Any]] = None
        music_tp_meta: Dict[str, Any] = {}
        if tp_type == "music":
            try:
                _breadcrumb(f"music:enter:i={i}")
            except Exception:
                pass
            try:
                music_tp_voice_context, music_tp_prompt_payload, music_tp_meta = _build_music_prompt_payload_for_tp(
                    ctx,
                    version_num,
                    tp,
                    prior_paragraph,
                )
            except UserActionRequired:
                # Never swallow explicit human-in-the-loop pauses; let them
                # propagate so the CLI can present the exact message.
                raise
            except Exception as exc:
                _log_warning(f"MUSIC: failed to build per-touch-point payload ({exc})", tp_log_dir)
                music_tp_voice_context = None
                music_tp_prompt_payload = None
                music_tp_meta = {}
            if music_tp_voice_context is None:
                music_tp_voice_context = music_voice_context
            if music_tp_prompt_payload is None and music_tp_voice_context is not None:
                try:
                    music_tp_prompt_payload = music_tp_voice_context.as_prompt_payload()
                except Exception:
                    music_tp_prompt_payload = {}

        # Skip already completed touch-points by checkpoint presence
        if i in completed:
            # For music, require new metadata/tracks artifacts before treating as completed
            if tp_type == "music":
                meta_ok = False
                try:
                    if tp_log_dir is not None:
                        meta_ok = (tp_log_dir / "metadata.json").exists() and (tp_log_dir / "tracks.csv").exists()
                except Exception:
                    meta_ok = False
                if not meta_ok:
                    try:
                        _breadcrumb(f"music:resume:not_completed:i={i}")
                    except Exception:
                        pass
                    # Treat as not completed: fall through into normal music handling below
                    completed.pop(i, None)
                else:
                    try:
                        _breadcrumb(f"music:resume:completed:i={i}")
                    except Exception:
                        pass
                    _log_info(f"RESUME: Skipping completed touch-point {i} ({tp_type}).", base_log_dir)
                    try:
                        if tp_log_dir is not None and music_tp_meta:
                            tp_log_dir.mkdir(parents=True, exist_ok=True)
                            save_text(tp_log_dir / "music_touch_point.json", json.dumps(music_tp_meta, indent=2))
                    except Exception:
                        pass
                    continue
            else:
                _log_info(f"RESUME: Skipping completed touch-point {i} ({tp_type}).", base_log_dir)
            # Even if draft exists, ensure brainstorm DONE gating is satisfied; if not, re-run brainstorm and exit
            try:
                if not branch_b:  # Only enforce brainstorm gating on initial authoring runs
                    reps_resume = _build_pipeline_replacements(setting, chapter, chapter_id, version_num, tp, state, prior_paragraph=prior_paragraph, ctx=ctx)
                    _ensure_brainstorm_done_on_resume(tp_type, reps_resume, log_dir=tp_log_dir, tp_index=i)
            except UserActionRequired:
                # Propagate HIL pause
                raise
            except Exception:
                pass
            # Regeneration path: if gate files were removed, recreate them from existing final artifacts (applies even if latest>0)
            try:
                # Gate-file regeneration should only apply to initial authoring (branch_b False). Edit branches (v2+) skip this to avoid
                # spurious human-in-loop pauses when users clean up artifacts.
                if (not branch_b) and tp_type in ("narration", "dialog", "implicit", "mixed") and tp_log_dir is not None:
                    fdraft_p = tp_log_dir / "touch_point_first_draft.txt"
                    fsugg_p = tp_log_dir / "first_suggestions.txt"
                    final_p = tp_log_dir / "touch_point_draft.txt"
                    sugg_p = tp_log_dir / "suggestions.txt"
                    # If either first gate file missing but we have a final draft (or suggestions), reconstruct gate and pause
                    if (not fdraft_p.exists() or not fsugg_p.exists()):
                        # Use existing final draft as first draft; if missing, attempt to read prior polished from state (no-op fallback)
                        if final_p.exists():
                            try:
                                save_text(fdraft_p, read_file(str(final_p)) or "")
                            except Exception:
                                save_text(fdraft_p, "")
                        else:
                            # No final draft: create empty stub to force user intervention
                            save_text(fdraft_p, "")
                        # Derive first suggestions from suggestions.txt if present else empty
                        if sugg_p.exists():
                            try:
                                save_text(fsugg_p, read_file(str(sugg_p)) or "")
                            except Exception:
                                save_text(fsugg_p, "")
                        else:
                            save_text(fsugg_p, "")
                        # Remove subtle edit outputs so subsequent resume takes proper gate path
                        try:
                            if final_p.exists():
                                final_p.unlink()
                        except Exception:
                            pass
                        try:
                            if sugg_p.exists():
                                sugg_p.unlink()
                        except Exception:
                            pass
                        raise UserActionRequired("Waiting for user suggestions on first draft.")
            except UserActionRequired:
                raise
            except Exception:
                pass
            # On edit branches (v2+), ensure suggestions.txt exists even for previously completed steps
            try:
                if branch_b and tp_type in ("narration", "dialog", "implicit", "mixed") and tp_log_dir is not None:
                    sug_p = tp_log_dir / "suggestions.txt"
                    if not sug_p.exists():
                        # Prefer mirroring first_suggestions.txt if present; else derive from check.txt; else create empty
                        fs_p = tp_log_dir / "first_suggestions.txt"
                        if fs_p.exists():
                            save_text(sug_p, read_file(str(fs_p)) or "")
                        else:
                            try:
                                chk_p = tp_log_dir / "check.txt"
                                if chk_p.exists():
                                    txt = read_file(str(chk_p))
                                    import re as _re
                                    m = _re.split(r"^===\s*RESPONSE\s*===\s*$", txt, maxsplit=1, flags=_re.M)
                                    save_text(sug_p, (m[1].strip() if len(m) == 2 else ""))
                                else:
                                    save_text(sug_p, "")
                            except Exception:
                                try:
                                    save_text(sug_p, "")
                                except Exception:
                                    pass
            except Exception:
                pass
            continue

        if tp_type == "music":
            polished_text = str(music_tp_meta.get("description", "") or "").strip()
            if tp_log_dir is not None:
                try:
                    tp_log_dir.mkdir(parents=True, exist_ok=True)
                    if music_tp_meta:
                        save_text(tp_log_dir / "music_touch_point.json", json.dumps(music_tp_meta, indent=2))
                except Exception:
                    pass

            if gw_music_metadata_tracks is None and gw_music_first_gate is None and gw_music_subtle_pass is None:
                records.append((tp_id, tp_type, tp_text, polished_text))
                _write_tp_checkpoint(tp_log_dir, tp_id, tp_type, tp_text, polished_text, state)
                continue

            if music_tp_voice_context is None or music_tp_prompt_payload is None:
                msg = "Music touch-point is missing required voice context. Update the YAML and retry."
                _log_warning("MUSIC: Missing voice context or payload; cannot proceed with music touch-point.", tp_log_dir)
                try:
                    _log_run(f"[MUSIC ERROR] tp={i:02d} {tp_type}: {msg}")
                except Exception:
                    pass
                raise UserActionRequired(msg)

            # Step 1: metadata + tracks layout
            try:
                if gw_music_metadata_tracks is not None:
                    try:
                        _breadcrumb(f"music:metadata:start:i={i}")
                    except Exception:
                        pass
                    gw_music_metadata_tracks(
                        tp_dir=tp_log_dir or base_log_dir,
                        tp_index=i,
                        tp_type=tp_type,
                        prompt_payload=music_tp_prompt_payload,
                    )
                    try:
                        _breadcrumb(f"music:metadata:done:i={i}")
                    except Exception:
                        pass
            except UserActionRequired:
                raise
            except Exception as exc:
                _log_warning(f"MUSIC: metadata/tracks step failed ({exc})", tp_log_dir)
                raise UserActionRequired("Music metadata/tracks step failed; inspect metadatatracks.txt and retry.") from exc

            # Step 2: melody dwell notes and edges (user-in-the-loop after success)
            try:
                if gw_music_melody_edges is not None:
                    try:
                        _breadcrumb(f"music:edges:start:i={i}")
                    except Exception:
                        pass
                    edges_created = gw_music_melody_edges(
                        tp_dir=tp_log_dir or base_log_dir,
                        tp_index=i,
                        tp_type=tp_type,
                        prompt_payload=music_tp_prompt_payload,
                    )
                    try:
                        _breadcrumb(f"music:edges:done:i={i}")
                    except Exception:
                        pass
                    if edges_created:
                        # Always pause after a new edges artifact so authors can review/edit
                        raise UserActionRequired(
                            "Melody dwell notes and edges generated; review and tweak melody_edges.txt before continuing."
                        )
            except UserActionRequired:
                raise
            except Exception as exc:
                _log_warning(f"MUSIC: melody edges step failed ({exc})", tp_log_dir)
                raise UserActionRequired("Music melody edges step failed; inspect melodyelements.txt and retry.") from exc

            # Step 3: construct the standard melody from dwell/edge elements.
            try:
                if gw_music_melody_construct is not None:
                    try:
                        _breadcrumb(f"music:melody:start:i={i}")
                    except Exception:
                        pass
                    gw_music_melody_construct(
                        tp_dir=tp_log_dir or base_log_dir,
                        tp_index=i,
                        tp_type=tp_type,
                        prompt_payload=music_tp_prompt_payload,
                        variant="standard",
                    )
                    try:
                        _breadcrumb(f"music:melody:done:i={i}")
                    except Exception:
                        pass
            except UserActionRequired:
                raise
            except Exception as exc:
                _log_warning(f"MUSIC: melody construction step failed ({exc})", tp_log_dir)
                raise UserActionRequired("Music melody construction step failed; inspect the melody log and retry.") from exc

            # Step 3b: construct complimentary and reprise melodies unconditionally.
            try:
                if gw_music_melody_construct is not None:
                    try:
                        _breadcrumb(f"music:melody_complimentary:start:i={i}")
                    except Exception:
                        pass
                    gw_music_melody_construct(
                        tp_dir=tp_log_dir or base_log_dir,
                        tp_index=i,
                        tp_type=tp_type,
                        prompt_payload=music_tp_prompt_payload,
                        variant="complimentary",
                    )
                    try:
                        _breadcrumb(f"music:melody_complimentary:done:i={i}")
                    except Exception:
                        pass
                    try:
                        _breadcrumb(f"music:melody_reprise:start:i={i}")
                    except Exception:
                        pass
                    gw_music_melody_construct(
                        tp_dir=tp_log_dir or base_log_dir,
                        tp_index=i,
                        tp_type=tp_type,
                        prompt_payload=music_tp_prompt_payload,
                        variant="reprise",
                    )
                    try:
                        _breadcrumb(f"music:melody_reprise:done:i={i}")
                    except Exception:
                        pass
            except UserActionRequired:
                raise
            except Exception as exc:
                _log_warning(f"MUSIC: complimentary/reprise melody construction failed ({exc})", tp_log_dir)
                raise UserActionRequired("Music complimentary/reprise melody construction failed; inspect the melody logs and retry.") from exc

            # Step 4: per-voice first-pass composition (no user feedback in v1).
            try:
                if gw_music_multi_voice_first_pass is not None:
                    for variant in ("standard", "complimentary", "reprise"):
                        suggestions_text = ""
                        # On edit branches (v2+), feed any existing first-pass suggestions
                        # into the second-pass composer as additional conditioning.
                        if branch_b and tp_log_dir is not None:
                            variant_safe = variant.strip().lower()
                            sugg_path = tp_log_dir / f"first_score_suggestions_{variant_safe}.txt"
                            try:
                                if sugg_path.exists():
                                    suggestions_text = read_file(str(sugg_path)) or ""
                            except Exception:
                                suggestions_text = ""
                        gw_music_multi_voice_first_pass(
                            tp_dir=tp_log_dir or base_log_dir,
                            tp_index=i,
                            tp_type=tp_type,
                            tp_text=tp_text,
                            voice_context=music_tp_voice_context,
                            prompt_payload=music_tp_prompt_payload,
                            variant=variant,
                            suggestions_text=suggestions_text,
                        )
            except UserActionRequired:
                raise
            except Exception as exc:
                _log_warning(f"MUSIC: multi-voice first-pass composition failed ({exc})", tp_log_dir)
                raise UserActionRequired("Music multi-voice first-pass composition failed; inspect logs before retrying.") from exc

            # Step 5: assemble per-variant first-pass MusicCSV + monitor MIDIs.
            try:
                if gw_music_assemble_first_variant is not None:
                    title = str(music_tp_meta.get("title") or "").strip() or "untitled"
                    for variant in ("standard", "complimentary", "reprise"):
                        gw_music_assemble_first_variant(
                            tp_dir=tp_log_dir or base_log_dir,
                            title=title,
                            variant=variant,
                        )
            except Exception as exc:
                _log_warning(f"MUSIC: first-pass aggregation failed ({exc})", tp_log_dir)
                raise UserActionRequired("Music first-pass aggregation failed; inspect logs before retrying.") from exc

            # Step 6: run per-variant first-pass checks and write suggestions files.
            try:
                from .music import assemble_first_pass_variant as _gw_music_assemble_first_variant  # re-import for type checkers
                from .music import MusicCSV as _MusicCSV
                from .music import musiccsv_to_text as _musiccsv_to_text
                if gw_music_assemble_first_variant is not None:
                    title = str(music_tp_meta.get("title") or "").strip() or "untitled"
                    for variant in ("standard", "complimentary", "reprise"):
                        variant_safe = variant.strip().lower()
                        first_score_path = (tp_log_dir or base_log_dir) / f"first_{title.replace(' ', '_')}_{variant_safe}.musiccsv"
                        if not first_score_path.exists():
                            continue
                        try:
                            from .musiccsv import read_musiccsv as _read_musiccsv
                        except Exception:
                            _read_musiccsv = read_musiccsv  # type: ignore[name-defined]
                        try:
                            music_obj = _read_musiccsv(first_score_path)
                            snippet_text = _truncate_musiccsv_text(_musiccsv_to_text(music_obj), None)
                        except Exception:
                            continue
                        try:
                            check_prompt = build_music_check_prompt(
                                prompt_payload=music_tp_prompt_payload,
                                tp_index=i,
                                tp_type=tp_type,
                                tp_text=tp_text,
                                musiccsv_snippet=snippet_text,
                            )
                            check_model, check_temp, check_max = _env_for_prompt(
                                "music_check_prompt.md",
                                "MUSIC_SCORE_CHECK",
                                default_temp=0.0,
                                default_max_tokens=800,
                            )
                            suggestions = llm_complete(
                                check_prompt,
                                system="Provide concise, actionable feedback on the score.",
                                temperature=check_temp,
                                max_tokens=check_max,
                                model=check_model,
                            )
                            sugg_name = f"first_score_suggestions_{variant_safe}.txt"
                            if tp_log_dir is not None:
                                save_text(tp_log_dir / sugg_name, suggestions)
                        except Exception:
                            # Best-effort diagnostics only; do not fail v1 on check errors.
                            continue
            except Exception as exc:
                _log_warning(f"MUSIC: first-pass per-variant checks encountered an error ({exc})", tp_log_dir)

            records.append((tp_id, tp_type, tp_text, polished_text))
            _write_tp_checkpoint(tp_log_dir, tp_id, tp_type, tp_text, polished_text, state)
            continue

        polished_text = ""
        if tp_type == "actors":
            state.set_actors(tp.get("content", ""))
        elif tp_type == "scene":
            state.set_scene(tp.get("content", ""))
        elif tp_type == "foreshadowing":
            state.add_foreshadowing(tp.get("content", ""))
        elif tp_type in ("voices", ):
            polished_text = ""
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
        elif tp_type in ("narration", "dialog", "implicit", "mixed"):
            # Before running pipelines, if brainstorm exists but lacks DONE, enforce resume gating
            try:
                if not branch_b:  # Skip brainstorm enforcement for edit branches (v2+)
                    reps_pre = _build_pipeline_replacements(setting, chapter, chapter_id, version_num, tp, state, prior_paragraph=prior_paragraph, ctx=ctx)
                    _ensure_brainstorm_done_on_resume(tp_type, reps_pre, log_dir=tp_log_dir, tp_index=i)
            except UserActionRequired:
                # Propagate HIL pause from resume gating
                raise
            except Exception:
                pass
            # If a first-draft gate exists, skip generation and proceed directly to subtle-edit phase
            first_draft_path = (tp_log_dir / "touch_point_first_draft.txt") if tp_log_dir else None
            final_draft_path = (tp_log_dir / "touch_point_draft.txt") if tp_log_dir else None
            first_suggestions_path = (tp_log_dir / "first_suggestions.txt") if tp_log_dir else None
            subtle_edit_post_gate = False
            if tp_log_dir is not None and first_draft_path and first_draft_path.exists() and not (final_draft_path and final_draft_path.exists()):
                # Resume after user-in-the-loop: run subtle edit using the latest on-disk first draft and suggestions
                try:
                    prior_pol = read_file(str(first_draft_path))
                except Exception:
                    prior_pol = ""
                try:
                    prior_sugg = read_file(str(first_suggestions_path)) if (first_suggestions_path and first_suggestions_path.exists()) else ""
                except Exception:
                    prior_sugg = ""
                if gw_run_subtle_edit_pipeline is None:
                    raise GWError("ghostwriter.pipelines.run_subtle_edit_pipeline not available")
                polished_text = gw_run_subtle_edit_pipeline(tp, state, setting=setting, chapter=chapter, chapter_id=chapter_id, version=version_num, tp_index=i, prior_polished=prior_pol, prior_suggestions=prior_sugg, prior_paragraph=prior_paragraph, log_dir=tp_log_dir, ctx=ctx)
                subtle_edit_post_gate = True
            else:
                # Execute the contentful pipeline with graceful error handling
                try:
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
                            polished_text = gw_run_subtle_edit_pipeline(tp, state, setting=setting, chapter=chapter, chapter_id=chapter_id, version=version_num, tp_index=i, prior_polished=use_pol, prior_suggestions=use_sugg, prior_paragraph=prior_paragraph, log_dir=tp_log_dir, ctx=ctx)
                        else:
                            if gw_run_narration_pipeline is None:
                                raise GWError("ghostwriter.pipelines.run_narration_pipeline not available")
                            polished_text = gw_run_narration_pipeline(tp, state, ctx=ctx, tp_index=i, prior_paragraph=prior_paragraph, log_dir=tp_log_dir)
                    elif tp_type == "dialog":
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
                            polished_text = gw_run_subtle_edit_pipeline(tp, state, setting=setting, chapter=chapter, chapter_id=chapter_id, version=version_num, tp_index=i, prior_polished=use_pol, prior_suggestions=use_sugg, prior_paragraph=prior_paragraph, log_dir=tp_log_dir, ctx=ctx)
                        else:
                            if gw_run_dialog_pipeline is None:
                                raise GWError("ghostwriter.pipelines.run_dialog_pipeline not available")
                            polished_text = gw_run_dialog_pipeline(tp, state, ctx=ctx, tp_index=i, prior_paragraph=prior_paragraph, log_dir=tp_log_dir)
                    else:  # implicit/mixed
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
                            polished_text = gw_run_subtle_edit_pipeline(tp, state, setting=setting, chapter=chapter, chapter_id=chapter_id, version=version_num, tp_index=i, prior_polished=use_pol, prior_suggestions=use_sugg, prior_paragraph=prior_paragraph, log_dir=tp_log_dir, ctx=ctx)
                        else:
                            # Prefer mixed pipeline; fall back to implicit if mixed unavailable
                            runner = gw_run_mixed_pipeline or gw_run_implicit_pipeline
                            if runner is None:
                                raise GWError("ghostwriter.pipelines.run_mixed_pipeline not available")
                            polished_text = runner(tp, state, ctx=ctx, tp_index=i, prior_paragraph=prior_paragraph, log_dir=tp_log_dir)
                except Exception as e:
                    # If this is a human-in-the-loop pause, log as INFO and stop gracefully
                    if isinstance(e, UserActionRequired):
                        pause_msg = str(e).strip() or "Human input required."
                        info = f"Touch-point {i} ({tp_type}) paused: {pause_msg}"
                        _log_info(info, base_log_dir)
                        return
                    # Otherwise, log as error and stop
                    msg = f"Touch-point {i} ({tp_type}) failed: {e}"
                    _log_error_base(msg)
                    try:
                        _log_run(msg)
                    except Exception:
                        pass
                    print("Stopping run due to error. See run_error.log in base directory.")
                    return

        else:
            # Unknown types treated as narration by default
            if branch_b:
                if gw_run_subtle_edit_pipeline is None:
                    raise GWError("ghostwriter.pipelines.run_subtle_edit_pipeline not available")
                polished_text = gw_run_subtle_edit_pipeline(tp, state, setting=setting, chapter=chapter, chapter_id=chapter_id, version=version_num, tp_index=i, prior_polished=prior_draft, prior_suggestions=prior_suggestions, log_dir=tp_log_dir, ctx=ctx)
            else:
                if gw_run_narration_pipeline is None:
                    raise GWError("ghostwriter.pipelines.run_narration_pipeline not available")
                polished_text = gw_run_narration_pipeline(tp, state, ctx=ctx, tp_index=i, prior_paragraph=prior_paragraph, log_dir=tp_log_dir)

        # Record (for actors/scene/foreshadowing, polished_text may be empty)
        records.append((tp_id, tp_type, tp_text, polished_text))
        # After producing content for narration/dialog/implicit, enforce the new first-draft user-in-the-loop gate
        if tp_type in ("narration", "dialog", "implicit", "mixed") and polished_text and tp_log_dir is not None:
            try:
                # Build replacements shared by suggestions prompt
                setting_block = getattr(state, "setting_block", "") or ""
                characters_block = getattr(state, "characters_block", "") or ""
                if not characters_block:
                    try:
                        # Prefer selecting by active actors; else include all characters
                        selected: List[dict] = []
                        all_chars = load_characters_list(ctx)
                        if isinstance(all_chars, list) and all_chars:
                            if state.active_actors:
                                wanted = {a.strip().lower() for a in state.active_actors if a.strip()}
                                for ch in all_chars:
                                    cid = str(ch.get("id", "")).strip().lower()
                                    cname = str(ch.get("name", "")).strip().lower()
                                    if cid in wanted or cname in wanted:
                                        selected.append(ch)
                            else:
                                selected = list(all_chars)
                        if selected:
                            characters_block = _gw_to_text({"Selected-Characters": selected})
                    except Exception:
                        characters_block = characters_block or ""
                try:
                    recent_polished = []
                    for (_tid, _ttype, _touch, _pol) in records[-4:-1]:
                        if _pol and str(_pol).strip():
                            recent_polished.append(str(_pol).strip())
                    context_block = "\n\n---\n\n".join(recent_polished)
                except Exception:
                    context_block = ""
                foreshadowing = ", ".join(state.foreshadowing)
                actors_csv = ", ".join(state.active_actors)
                check_tpl = {
                    "narration": "check_narration_prompt.md",
                    "dialog": "check_dialog_prompt.md",
                    "implicit": "check_implicit_prompt.md",
                    "mixed": "check_mixed_prompt.md",
                }.get(tp_type, "check_narration_prompt.md")
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
                suggestions_user = check_user + "\n\n---\nNow produce a concise, actionable list of suggested changes and fixes only. Do not restate the findings verbatim; output just the suggestions."
                sugg_model, sugg_temp, sugg_max_tokens = _env_for_prompt(check_tpl, "SUGGESTIONS", default_temp=0.0, default_max_tokens=800)
                # Log LLM context to run.log before the request for traceability
                try:
                    from .logging import log_run as _log_run
                    _log_run(f"LLM ctx | tp={i:02d} type={tp_type} template={check_tpl} step=SUGGESTIONS")
                except Exception:
                    pass
                suggestions_out = llm_complete(suggestions_user, system="You are an evaluator extracting actionable suggestions only.", temperature=sugg_temp, max_tokens=sugg_max_tokens, model=sugg_model)
                # Save the single prompt+response that produced suggestions as check.txt (debugging artifact)
                try:
                    with (tp_log_dir / "check.txt").open("w", encoding="utf-8") as f:
                        f.write("=== SYSTEM ===\nYou are an evaluator extracting actionable suggestions only.\n\n")
                        f.write("=== USER ===\n" + suggestions_user + "\n\n")
                        f.write("=== RESPONSE ===\n" + suggestions_out + "\n")
                except Exception:
                    pass
                try:
                    tp_log_dir.mkdir(parents=True, exist_ok=True)
                    if subtle_edit_post_gate:
                        # Post-gate resume run: update final suggestions only, regardless of branch
                        save_text(tp_log_dir / "suggestions.txt", suggestions_out)
                    elif not branch_b:
                        # Initial authoring run: create first-draft gate and pause
                        save_text(tp_log_dir / "touch_point_first_draft.txt", polished_text)
                        save_text(tp_log_dir / "first_suggestions.txt", suggestions_out)
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
                        save_text(tp_log_dir / "touch_point_state.json", _gw_to_text(cp))
                        raise UserActionRequired("Waiting for user suggestions on first draft.")
                except UserActionRequired:
                    # Re-raise to propagate graceful stop to CLI/driver
                    raise
                except Exception:
                    pass
                # Ensure suggestions.txt exists after a resume subtle-edit even if the LLM write path above failed
                try:
                    if subtle_edit_post_gate and tp_log_dir is not None:
                        sug_p = tp_log_dir / "suggestions.txt"
                        if not sug_p.exists():
                            fs_p = tp_log_dir / "first_suggestions.txt"
                            if fs_p.exists():
                                save_text(sug_p, read_file(str(fs_p)) or "")
                except Exception:
                    pass
            except UserActionRequired:
                # Ensure the first-draft gate pause is not swallowed by broad exception handlers
                raise
            except Exception:
                pass
        # Write checkpoint for resume (only after final draft is produced)
        _write_tp_checkpoint(tp_log_dir, tp_id, tp_type, tp_text, polished_text, state)
        # Update prior_paragraph to the entire previous touch-point draft (not just the last line)
        if polished_text:
            prior_paragraph = polished_text.strip()
        # Post-condition: ensure suggestions.txt exists for contentful types in resume/edit flows
        try:
            if tp_type in ("narration", "dialog", "implicit", "mixed") and tp_log_dir is not None:
                sug_p2 = tp_log_dir / "suggestions.txt"
                if not sug_p2.exists():
                    fs_p2 = tp_log_dir / "first_suggestions.txt"
                    if fs_p2.exists():
                        save_text(sug_p2, read_file(str(fs_p2)) or "")
                    else:
                        # Fallback: if a check.txt exists (single prompt+response), mirror the response
                        try:
                            chk_p = tp_log_dir / "check.txt"
                            if chk_p.exists():
                                txt = read_file(str(chk_p))
                                # Extract simple RESPONSE section
                                import re as _re
                                m = _re.split(r"^===\s*RESPONSE\s*===\s*$", txt, maxsplit=1, flags=_re.M)
                                if len(m) == 2:
                                    save_text(sug_p2, m[1].strip())
                                else:
                                    save_text(sug_p2, "")
                            else:
                                save_text(sug_p2, "")
                        except Exception:
                            # At minimum create an empty suggestions.txt to satisfy downstream expectations
                            try:
                                save_text(sug_p2, "")
                            except Exception:
                                pass
        except Exception:
            pass

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

    if gw_music_finalize_exports is not None:
        try:
            gw_music_finalize_exports(
                chapter_id=chapter_id,
                version=version_num,
                voice_context=music_voice_context,
            )
        except Exception as exc:
            try:
                _log_warning(f"MUSIC: export bundle failed ({exc})", base_log_dir)
            except Exception:
                pass
