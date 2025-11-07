"""MockLLM scaffolding utilities (Task 7).

This module provides helpers to:
- Compute and persist a hash of all prompt templates
- Detect prompt hash mismatches between the working prompts/ and a golden book base (e.g., LRRH)
- Update golden logs (prompt+response files) to match current prompt templates while preserving responses
- Copy a partial snapshot of a golden book up to a given pipeline step
- Modify a partial snapshot to simulate user-in-the-loop gates
- Extract the current step prompt/response from a partial snapshot

The functions are designed to be deterministic and side-effect free except where
explicit file writes are requested.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Local imports kept light to avoid cycles
from .templates import apply_template
from .context import RunContext
from .commands import run_pipelines_for_chapter as _run_pipelines_for_chapter
from .commands import chapter_id_from_path as _chapter_id_from_path
from .logging import log_run as _log_run

# ---------------------------
# Prompt hash utilities
# ---------------------------

def compute_prompts_hash(prompts_dir: str | Path = "prompts") -> str:
    """Compute a stable SHA-256 hash over all files in prompts_dir.

    - Sort files by relative path for stability
    - Include filename and content to detect renames
    """
    pdir = Path(prompts_dir)
    if not pdir.exists():
        return hashlib.sha256(b"<missing-prompts>").hexdigest()
    h = hashlib.sha256()
    files = sorted([p for p in pdir.rglob("*") if p.is_file()])
    for p in files:
        rel = str(p.relative_to(pdir)).replace("\\", "/")
        try:
            data = p.read_bytes()
        except Exception:
            data = b""
        h.update(rel.encode("utf-8") + b"\0" + data + b"\0")
    return h.hexdigest()


def write_prompt_hash(target_dir: str | Path, hash_value: Optional[str] = None) -> str:
    """Write the prompt hash to target_dir/prompt_hash and return the hash string."""
    tdir = Path(target_dir)
    tdir.mkdir(parents=True, exist_ok=True)
    hv = hash_value or compute_prompts_hash("prompts")
    (tdir / "prompt_hash").write_text(hv + "\n", encoding="utf-8")
    return hv


def read_prompt_hash(dir_path: str | Path) -> Optional[str]:
    p = Path(dir_path) / "prompt_hash"
    if not p.exists():
        return None
    try:
        return p.read_text(encoding="utf-8").strip()
    except Exception:
        return None


def prompts_hash_matches(prompts_dir: str | Path, book_base_dir: str | Path) -> bool:
    current = compute_prompts_hash(prompts_dir)
    golden = read_prompt_hash(book_base_dir)
    return golden == current and golden is not None


# ---------------------------
# Prompt+response log parsing
# ---------------------------

_SECTION_RE = re.compile(r"^===\s*(SYSTEM|USER|RESPONSE)\s*===\s*$")


def parse_prompt_response_file(path: str | Path) -> Dict[str, str]:
    """Parse a log file containing === SYSTEM ===, === USER ===, === RESPONSE ===.
    Returns a dict with keys 'SYSTEM', 'USER', 'RESPONSE' (missing keys -> empty string).
    """
    text = Path(path).read_text(encoding="utf-8")
    parts: Dict[str, List[str]] = {"SYSTEM": [], "USER": [], "RESPONSE": []}
    current: Optional[str] = None
    for ln in text.splitlines():
        m = _SECTION_RE.match(ln.strip())
        if m:
            current = m.group(1).upper()
            continue
        if current in parts:
            parts[current].append(ln)
    return {k: ("\n".join(v).rstrip()) for k, v in parts.items()}


def write_prompt_response_file(path: str | Path, system: str, user: str, response: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    content = (
        f"=== SYSTEM ===\n{system or ''}\n\n"  # ensure blank line separators
        f"=== USER ===\n{user or ''}\n\n"
        f"=== RESPONSE ===\n{response or ''}\n"
    )
    Path(path).write_text(content, encoding="utf-8")


# ---------------------------
# Template mapping and regeneration
# ---------------------------

_TEMPLATE_BY_NAME: List[Tuple[re.Pattern[str], str]] = [
    (re.compile(r"ordering", re.I), "ordering_prompt.md"),
    (re.compile(r"generate_narration", re.I), "generate_narration_prompt.md"),
    (re.compile(r"actor_assignment", re.I), "actor_assignment_prompt.md"),
    (re.compile(r"agenda", re.I), "agenda_prompt.md"),
    (re.compile(r"body_language", re.I), "body_language_prompt.md"),
    (re.compile(r"reactions?", re.I), "reaction_prompt.md"),
    (re.compile(r"subtle_edit", re.I), "subtle_edit_prompt.md"),
    (re.compile(r"dialog_batch", re.I), "character_dialog_prompt.md"),
    (re.compile(r"brainstorm" , re.I), ""),  # mapped by touch-point type
]


def _template_for_file(step_type: str, filename: str) -> Optional[str]:
    base = Path(filename).name
    for pat, tpl in _TEMPLATE_BY_NAME:
        if pat.search(base):
            if tpl:
                return tpl
            # Brainstorm depends on type
            if step_type.lower() == "dialog":
                return "dialog_brain_storm_prompt.md"
            if step_type.lower() == "mixed":
                return "mixed_brain_storm_prompt.md"
            return "narration_brain_storm_prompt.md"
    return None


@dataclass
class _TPState:
    active_actors: List[str]
    current_scene: Optional[str]
    foreshadowing: List[str]
    setting_block: str
    characters_block: str
    appended_dialog: Dict[str, List[str]]

    def recent_dialog(self, actor: str) -> List[str]:  # compatibility with build helpers
        try:
            return self.appended_dialog.get(actor, [])
        except Exception:
            return []


def _load_tp_state(step_dir: Path) -> Tuple[_TPState, str]:
    state_path = step_dir / "touch_point_state.json"
    if not state_path.exists():
        # minimal empty state
        return _TPState([], None, [], "", "", {}), ""
    raw = json.loads(state_path.read_text(encoding="utf-8"))
    st = _TPState(
        active_actors=list(raw.get("active_actors", []) or []),
        current_scene=raw.get("scene"),
        foreshadowing=list(raw.get("foreshadowing", []) or []),
        setting_block=str(raw.get("setting_block", "") or ""),
        characters_block=str(raw.get("characters_block", "") or ""),
        appended_dialog=dict(raw.get("appended_dialog", {}) or {}),
    )
    tp_type = str(raw.get("type", "")).strip().lower()
    return st, tp_type


def _build_replacements_for_step(step_dir: Path, chapter_id: str, version: int) -> Dict[str, str]:
    # Build replacements using the same common logic as runtime pipelines
    # Avoid relying on env by deriving base dir and chapter path from the step_dir
    from .pipelines.common import build_pipeline_replacements as _build
    from .touch_point import _strip_trailing_done as _tp_strip_done
    try:
        base_dir = step_dir.parents[3]  # <base>/iterations/CHAPTER_xxx/pipeline_vN/<NN_type>
        chapter_path = base_dir / "chapters" / f"{chapter_id}.yaml"
    except Exception:
        # Fallback to env-based resolution if structure is unexpected
        from .env import get_chapters_dir as _get_chapters_dir
        chapter_path = _get_chapters_dir() / f"{chapter_id}.yaml"
    # Ensure RunContext resolves SETTING/CHARACTERS/chapters relative to the derived base
    prev_base = os.getenv("GW_BOOK_BASE_DIR")
    try:
        os.environ["GW_BOOK_BASE_DIR"] = str(base_dir)
        ctx = RunContext.from_paths(chapter_path=str(chapter_path), version=version, allow_missing_chapter=False)
    finally:
        if prev_base is None:
            try:
                del os.environ["GW_BOOK_BASE_DIR"]
            except Exception:
                pass
        else:
            os.environ["GW_BOOK_BASE_DIR"] = prev_base
    state, _ = _load_tp_state(step_dir)
    # Determine touch-point dict for content/type labels
    tp_dict = {"type": step_dir.name.split("_", 1)[-1], "content": _read_touchpoint_text(step_dir)}
    # Reconstruct prior_paragraph by scanning previous contentful touch-points in this pipeline
    prior_paragraph = ""
    try:
        # Determine numeric step index from the current step directory name (e.g., '06_dialog')
        step_idx = None
        try:
            step_idx = int(step_dir.name.split("_", 1)[0])
        except Exception:
            step_idx = None
        if step_idx is not None and step_idx > 1:
            pipeline_dir = step_dir.parent  # .../pipeline_vN
            prev_steps = sorted(
                [p for p in pipeline_dir.iterdir() if p.is_dir() and re.match(r"^\d+_", p.name)],
                key=lambda d: int(d.name.split("_", 1)[0])
            )
            # Walk backwards to find the latest available draft text
            for ps in reversed([p for p in prev_steps if int(p.name.split("_", 1)[0]) < step_idx]):
                draft_p = ps / "touch_point_draft.txt"
                first_draft_p = ps / "touch_point_first_draft.txt"
                try:
                    if draft_p.exists():
                        prior_paragraph = draft_p.read_text(encoding="utf-8").strip()
                        if prior_paragraph:
                            break
                    if first_draft_p.exists():
                        prior_paragraph = first_draft_p.read_text(encoding="utf-8").strip()
                        if prior_paragraph:
                            break
                except Exception:
                    continue
    except Exception:
        prior_paragraph = prior_paragraph or ""

    reps = _build(ctx.setting, ctx.chapter, chapter_id, version, tp_dict, state, prior_paragraph=prior_paragraph or "", ctx=ctx)
    # Provide [bullets] from brainstorm.txt when present so regenerated prompts match goldens
    try:
        bs_path = step_dir / "brainstorm.txt"
        if bs_path.exists():
            raw = bs_path.read_text(encoding="utf-8")
            reps["[bullets]"] = _tp_strip_done(raw).strip()
    except Exception:
        # Leave placeholder if anything goes wrong
        pass

    # Provide subtle-edit placeholders when present so regenerated prompts match runtime
    # Use the user-gate first-draft and first suggestions, which subtle_edit consumes on resume
    try:
        # Prefer new gate filename; fall back to legacy per-TP draft name
        first_draft = step_dir / "touch_point_first_draft.txt"
        if first_draft.exists():
            reps["[draft_text]"] = first_draft.read_text(encoding="utf-8")
        else:
            legacy_draft = step_dir / "touch_point_draft.txt"
            if legacy_draft.exists():
                reps["[draft_text]"] = legacy_draft.read_text(encoding="utf-8")
    except Exception:
        pass
    try:
        # Prefer new gate filename; fall back to legacy suggestions or parse from check.txt
        first_sugg = step_dir / "first_suggestions.txt"
        if first_sugg.exists():
            reps["[suggestions]"] = first_sugg.read_text(encoding="utf-8")
        else:
            legacy_sugg = step_dir / "suggestions.txt"
            if legacy_sugg.exists():
                reps["[suggestions]"] = legacy_sugg.read_text(encoding="utf-8")
            else:
                # As a last resort, extract RESPONSE from check.txt (suggestions-only prompt)
                check_log = step_dir / "check.txt"
                if check_log.exists():
                    try:
                        parts = parse_prompt_response_file(check_log)
                        reps["[suggestions]"] = parts.get("RESPONSE", "")
                    except Exception:
                        pass
    except Exception:
        pass
    # Story summary placeholders should reference the PRIOR chapter (runtime behavior) not the current
    # chapter's own summaries (which are produced only after final draft). Using current chapter summaries
    # during early-step regeneration causes anachronistic substitution in brainstorm resumes.
    try:
        import re as _re
        m = _re.search(r"(\d+)$", chapter_id)
        prev_id = None
        if m:
            try:
                num = int(m.group(1))
                if num > 1:  # CHAPTER_001 has no previous
                    prev_id = f"CHAPTER_{num-1:03d}"
            except Exception:
                prev_id = None
        base_dir = step_dir.parents[3]  # <base>/iterations/CHAPTER_xxx/pipeline_vN/<NN_type>
        if prev_id:
            prev_iter_dir = base_dir / "iterations" / prev_id
            ssf_path = prev_iter_dir / "story_so_far.txt"
            srt_path = prev_iter_dir / "story_relative_to.txt"
            if ssf_path.exists():
                reps["[story_so_far.txt]"] = ssf_path.read_text(encoding="utf-8")
                reps["[STORY_SO_FAR]"] = reps["[story_so_far.txt]"]
            if srt_path.exists():
                reps["[story_relative_to.txt]"] = srt_path.read_text(encoding="utf-8")
                reps["[STORY_RELATIVE_TO]"] = reps["[story_relative_to.txt]"]
        # If no prior chapter, leave placeholders absent so templates fall back to defaults/blank.
    except Exception:
        pass
    return reps


def _read_touchpoint_text(step_dir: Path) -> str:
    # Heuristic: use the line from touch_point_state.json 'touchpoint' if present, else empty
    p = step_dir / "touch_point_state.json"
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        val = raw.get("touchpoint") or raw.get("touch_point") or raw.get("touch_point_text")
        return str(val or "")
    except Exception:
        return ""


def regenerate_prompt_for_log(log_path: str | Path) -> Optional[str]:
    """Regenerate the USER prompt for a given prompt+response log using current templates.

    Returns the regenerated USER prompt, or None if the file can't be mapped.
    """
    log_p = Path(log_path)
    step_dir = log_p.parent
    # Extract chapter/version from path: .../iterations/CHAPTER_xxx/pipeline_vN/<NN_type>/file.txt
    try:
        version = int(step_dir.parent.name.split("_v")[-1])  # pipeline_vN
        # Path layout: <base>/iterations/CHAPTER_xxx/pipeline_vN/<NN_type>/file.txt
        # step_dir = .../pipeline_vN/<NN_type>
        # chapter id is the parent of pipeline_vN
        chapter_id = step_dir.parent.parent.name  # CHAPTER_xxx
    except Exception:
        return None
    state, tp_type = _load_tp_state(step_dir)
    step_kind = tp_type or step_dir.name.split("_", 1)[-1]
    template = _template_for_file(step_kind, log_p.name)
    if not template:
        return None
    reps = _build_replacements_for_step(step_dir, chapter_id, version)
    # Inject derived placeholders by reading sibling logs so regenerated prompts match runtime
    try:
        # Determine numeric step index from filename prefix (e.g., '05_generate_narration.txt' -> 5)
        name = log_p.name
        step_idx = None
        m = re.match(r"^(\d+)\_", name)
        if m:
            try:
                step_idx = int(m.group(1))
            except Exception:
                step_idx = None

        def _read_latest_response(prefix: str) -> Optional[str]:
            """Read RESPONSE from the latest sibling log starting with given prefix (e.g., '05_ordering')."""
            try:
                if step_idx is None:
                    return None
                pat = f"{step_idx:02d}_{prefix}"
                candidates = sorted([p for p in step_dir.glob(f"{pat}*.txt") if p.is_file()])
                if not candidates:
                    return None
                last = candidates[-1]
                parts = parse_prompt_response_file(last)
                return parts.get("RESPONSE", "")
            except Exception:
                return None

        # For generation or actor-assignment steps, supply ORDERED_BULLETS from prior ordering
        if template in ("generate_narration_prompt.md", "actor_assignment_prompt.md"):
            ordered = _read_latest_response("ordering")
            ord_txt = (ordered or "")
            if ord_txt.strip():
                reps["[ORDERED_BULLETS]"] = ord_txt
                reps["[ordered_bullets]"] = ord_txt
                # Many templates alias [bullets] to the ordered list at this stage
                reps.setdefault("[bullets]", ord_txt)
                reps.setdefault("[BULLETS]", ord_txt)

    # For body_language / agenda / reactions, provide ACTOR_LINES and ACTOR_LIST from prior actor_assignment
        if template in ("body_language_prompt.md", "agenda_prompt.md", "reaction_prompt.md"):
            actor_lines = _read_latest_response("actor_assignment")
            act_txt = (actor_lines or "")
            if act_txt.strip():
                reps["[ACTOR_LINES]"] = act_txt
                # Build ACTOR_LIST by parsing ids before ':' per line
                try:
                    ids: List[str] = []
                    seen: set[str] = set()
                    for ln in act_txt.splitlines():
                        m2 = re.match(r"^\s*([A-Za-z0-9_.\-]+)\s*:\s+.+", ln)
                        if m2:
                            aid = m2.group(1)
                            low = aid.strip().lower()
                            if low not in seen:
                                seen.add(low)
                                ids.append(aid)
                    reps["[ACTOR_LIST]"] = ", ".join(ids)
                except Exception:
                    reps["[ACTOR_LIST]"] = ""
    except Exception:
        # Best-effort enhancements; ignore on error
        pass

    # Subtle edit in vN>1 uses prior version's per-TP draft and suggestions
    try:
        if template == "subtle_edit_prompt.md" and version and version > 1:
            prev_tp_dir = step_dir.parent.parent / f"pipeline_v{version-1}" / step_dir.name
            if prev_tp_dir.exists():
                try:
                    prev_draft = (prev_tp_dir / "touch_point_draft.txt").read_text(encoding="utf-8") if (prev_tp_dir / "touch_point_draft.txt").exists() else ""
                except Exception:
                    prev_draft = ""
                try:
                    prev_sugg = (prev_tp_dir / "suggestions.txt").read_text(encoding="utf-8") if (prev_tp_dir / "suggestions.txt").exists() else ""
                except Exception:
                    prev_sugg = ""
                if prev_draft.strip():
                    reps["[draft_text]"] = prev_draft
                if prev_sugg.strip():
                    reps["[suggestions]"] = prev_sugg
    except Exception:
        pass
    # Special handling: dialog/mixed batch prompts are not simple templates; they are assembled
    # programmatically during runtime. Reconstruct the exact batch USER content here to ensure
    # MockLLM index keys match runtime prompts.
    try:
        name_low = log_p.name.lower()
        if "dialog_batch" in name_low and template == "character_dialog_prompt.md":
            # Recreate the batch user content like in pipelines.dialog/mixed
            from .context import RunContext as _RunContext
            from .characters import load_characters_list as _load_chars
            from .utils import to_text as _to_text
            from .characters import build_character_call_prompt as _build_char_call
            ctx = _RunContext.from_paths(chapter_path=str(step_dir.parents[3] / "chapters" / f"{chapter_id}.yaml"), version=version, allow_missing_chapter=False)

            # Helper: read RESPONSE from sibling logs for the same step index
            def _read_latest_response(prefix: str) -> Optional[str]:
                try:
                    pat = f"{step_idx:02d}_{prefix}"
                    candidates = sorted([p for p in step_dir.glob(f"{pat}*.txt") if p.is_file()])
                    if not candidates:
                        return None
                    parts = parse_prompt_response_file(candidates[-1])
                    return parts.get("RESPONSE", "")
                except Exception:
                    return None

            # Determine step index (e.g., 06)
            step_idx = None
            msi = re.match(r"^(\d+)\_", log_p.name)
            if msi:
                try:
                    step_idx = int(msi.group(1))
                except Exception:
                    step_idx = None
            if step_idx is None:
                return apply_template(Path("prompts") / template, reps)

            # Gather prior artifacts needed for composition
            actor_lines = _read_latest_response("actor_assignment") or ""
            body_lang = _read_latest_response("body_language") or ""
            agenda_text = _read_latest_response("agenda") or ""
            reactions_text = _read_latest_response("reactions") or ""

            # Parse actor lines into ordered pairs
            _ACT_RE = re.compile(r"^\s*([A-Za-z0-9_.\-]+)\s*:\s+(.+)")
            lines_pairs: List[tuple[str, str]] = []
            actor_ids_in_order: List[str] = []
            for ln in (actor_lines or "").splitlines():
                m = _ACT_RE.match(ln)
                if m:
                    aid = m.group(1)
                    lines_pairs.append((aid, m.group(2)))
                    if aid not in actor_ids_in_order:
                        actor_ids_in_order.append(aid)

            # Agenda by actor: replicate parser from pipelines
            def _parse_agenda_by_actor(text: str) -> Dict[str, str]:
                by_actor: Dict[str, List[str]] = {}
                current: Optional[str] = None
                _ARE = re.compile(r"^\s*([A-Za-z0-9_.\-]+)\s*:\s*$")
                for ln in (text or "").splitlines():
                    if not ln.strip():
                        continue
                    if not ln.lstrip().startswith(('*', '-')) and _ARE.match(ln):
                        m2 = _ARE.match(ln)
                        if m2:
                            aid_key = m2.group(1).strip()
                            current = aid_key
                            by_actor.setdefault(aid_key, [])
                        continue
                    if current is not None and ln.lstrip().startswith(('*', '-')):
                        content = ln.lstrip()[1:].lstrip()
                        if content:
                            by_actor[current].append(content)
                return {aid: "\n".join([f"* {i}" for i in items]) for aid, items in by_actor.items() if items}

            agenda_by_actor = _parse_agenda_by_actor(agenda_text)

            # Reactions in order
            reactions_vals: List[str] = []
            for ln in (reactions_text or "").splitlines():
                m = _ACT_RE.match(ln)
                if m:
                    reactions_vals.append(m.group(2))

            # Load characters YAML
            char_yaml_by_id: Dict[str, str] = {}
            try:
                all_chars = _load_chars(ctx)
                if isinstance(all_chars, list) and all_chars:
                    for ch in all_chars:
                        cid = str(ch.get("id", "")).strip()
                        if cid:
                            char_yaml_by_id[cid.lower()] = _to_text(ch)
            except Exception:
                pass

            # Build ACTOR_LIST for helpers
            ids: List[str] = []
            seen: set[str] = set()
            for aid in actor_ids_in_order:
                low = aid.strip().lower()
                if low not in seen:
                    seen.add(low)
                    ids.append(aid)
            reps4 = dict(reps)
            reps4["[ACTOR_LINES]"] = actor_lines
            reps4["[ACTOR_LIST]"] = ", ".join(ids)

            # Delegate to the same builders used at runtime for exact parity
            kind = step_kind.strip().lower()
            if kind == "mixed":
                from .pipelines.mixed import build_mixed_batch_user as _build_mixed_batch_user
                return _build_mixed_batch_user(
                    reps4=reps4,
                    actor_lines=actor_lines,
                    body_lang=body_lang,
                    agenda_text=agenda_text,
                    reactions_text=reactions_text,
                    state=state,
                )
            else:
                from .pipelines.dialog import build_dialog_batch_user as _build_dialog_batch_user
                return _build_dialog_batch_user(
                    reps4=reps4,
                    actor_lines=actor_lines,
                    body_lang=body_lang,
                    agenda_text=agenda_text,
                    reactions_text=reactions_text,
                    state=state,
                )
    except Exception:
        # Fall through to template-based regeneration if anything goes wrong
        pass

    user = apply_template(Path("prompts") / template, reps)
    # Brainstorm resumes may include seed bullets; we intentionally rely on template only for non-batch steps
    return user


# ---------------------------
# Golden updater
# ---------------------------

def update_golden_prompts(book_base_dir: str | Path, *, prompts_dir: str | Path = "prompts", dry_run: bool = False) -> List[Dict[str, str]]:
    """Update all prompt+response logs under <book_base_dir>/iterations to use current templates.

    Keeps the RESPONSE section intact; rewrites SYSTEM (unchanged) and USER according to templates
    when a mapping exists. Returns a summary list of {file, updated, reason}.
    On success, writes the new prompt_hash into <book_base_dir>/prompt_hash.
    """
    base = Path(book_base_dir)
    iters = base / "iterations"
    results: List[Dict[str, str]] = []
    if not iters.exists():
        return results
    for log in iters.rglob("*.txt"):
        # Only consider files that look like step logs with sections
        text = log.read_text(encoding="utf-8")
        if "=== USER ===" not in text or "=== RESPONSE ===" not in text:
            continue
        try:
            system_user_resp = parse_prompt_response_file(log)
            regenerated = regenerate_prompt_for_log(log)
            if not regenerated:
                results.append({"file": str(log), "updated": "no", "reason": "no-template"})
                continue
            old_user = (system_user_resp.get("USER") or "").strip()
            new_user = (regenerated or "").strip()
            if _normalize_ws(old_user) == _normalize_ws(new_user):
                results.append({"file": str(log), "updated": "no", "reason": "unchanged"})
                continue
            if not dry_run:
                write_prompt_response_file(log, system_user_resp.get("SYSTEM", ""), regenerated, system_user_resp.get("RESPONSE", ""))
            results.append({"file": str(log), "updated": "yes", "reason": "template-updated"})
        except Exception as e:
            results.append({"file": str(log), "updated": "no", "reason": f"error:{e}"})
    # Write new prompt hash
    if not dry_run:
        write_prompt_hash(base, compute_prompts_hash(prompts_dir))
    return results


def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


# ---------------------------
# Partial copy and modification utilities
# ---------------------------

def copy_partial_book(
    src_base: str | Path,
    dest_base: str | Path,
    *,
    chapter_id: str,
    version: int,
    upto_step_dir: str,
    upto_filename: Optional[str] = None,
    clear_dest: bool = True,
    copy_root_artifacts: bool = True,
    all_versions: bool = False,
    max_version: Optional[int] = None,
) -> None:
    """Copy a partial snapshot of a golden book.

    Semantics:
    - Base metadata (SETTING.yaml, CHARACTERS.yaml, chapters/*) always copied.
    - Iteration root artifacts (story_* / draft_v* / final.txt) copied when present.
    - all_versions=True: copy all pipeline_vN (optionally bounded by max_version) and apply the upto_step_dir cutoff uniformly to each.
    - all_versions=False: treat ``version`` as the *highest* version; copy pipeline_v1..pipeline_v<version> inclusively.
        * For pipeline versions LOWER than ``version`` copy ALL steps (ignore upto_step_dir).
        * For pipeline_v<version> (the highest) apply upto_step_dir (and upto_filename within the cutoff step).
    - If upto_step_dir == 'all' treat cutoff as infinite (full copy for highest version too).
    - Files are copied preserving relative structure; final cutoff step only copies top-level files unless recursion is needed for files prior to cutoff.
    """
    src = Path(src_base)
    dst = Path(dest_base)
    if clear_dest and dst.exists():
        shutil.rmtree(dst)
    # Copy root-level files
    for name in ["SETTING.yaml", "CHARACTERS.yaml"]:
        sp = src / name
        if sp.exists():
            (dst / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(sp, dst / name)
    # Copy chapters
    s_ch = src / "chapters"
    d_ch = dst / "chapters"
    if s_ch.exists():
        shutil.copytree(s_ch, d_ch, dirs_exist_ok=True)
    # Iteration root artifacts
    iter_root_src = src / "iterations" / chapter_id
    iter_root_dst = dst / "iterations" / chapter_id
    if iter_root_src.exists():
        iter_root_dst.mkdir(parents=True, exist_ok=True)
        if copy_root_artifacts:
            for fname in [
                "story_so_far.txt",
                "story_relative_to.txt",
                "draft_v1.txt",
                "draft_v2.txt",
                "final.txt",
            ]:
                sp = iter_root_src / fname
                if sp.exists():
                    try:
                        shutil.copy2(sp, iter_root_dst / fname)
                    except Exception:
                        pass

    # Determine which pipeline directories to copy based on semantics
    pipeline_dirs: List[Path] = []
    if not iter_root_src.exists():
        return
    all_candidate_pipes = sorted(
        [p for p in iter_root_src.iterdir() if p.is_dir() and re.match(r"^pipeline_v\d+$", p.name)],
        key=lambda d: int(d.name.split("_v")[-1])
    )
    if all_versions:
        if max_version is not None:
            pipeline_dirs = [p for p in all_candidate_pipes if int(p.name.split("_v")[-1]) <= int(max_version)]
        else:
            pipeline_dirs = all_candidate_pipes
    else:
        # Inclusive of all lower versions up to 'version'
        pipeline_dirs = [p for p in all_candidate_pipes if int(p.name.split("_v")[-1]) <= int(version)]

    # Global cutoff for highest version only (unless all_versions=True then all pipelines share cutoff)
    try:
        if str(upto_step_dir).strip().lower() == "all":
            global_cutoff_idx = 10**9
        else:
            global_cutoff_idx = int(str(upto_step_dir).split("_", 1)[0])
    except Exception:
        global_cutoff_idx = 999

    highest_version = max([int(p.name.split("_v")[-1]) for p in pipeline_dirs], default=0)

    for pipe_src in pipeline_dirs:
        vnum = int(pipe_src.name.split("_v")[-1])
        # Determine per-pipeline cutoff
        if all_versions:
            cutoff_idx = global_cutoff_idx
        else:
            cutoff_idx = global_cutoff_idx if vnum == highest_version else 10**9  # lower versions full copy
        pipe_dst = iter_root_dst / pipe_src.name
        pipe_dst.mkdir(parents=True, exist_ok=True)
        # Copy any top-level files inside pipeline_vN (e.g., chapter_brainstorm_result.txt)
        try:
            for top_file in [f for f in pipe_src.iterdir() if f.is_file()]:
                try:
                    shutil.copy2(top_file, pipe_dst / top_file.name)
                except Exception:
                    pass
        except Exception:
            pass
        for step in sorted([p for p in pipe_src.iterdir() if p.is_dir() and re.match(r"^\d+_", p.name)]):
            try:
                idx = int(step.name.split("_", 1)[0])
            except Exception:
                idx = 999
            if idx > cutoff_idx:
                continue
            target_dir = pipe_dst / step.name
            target_dir.mkdir(parents=True, exist_ok=True)
            # If this step is strictly before the cutoff OR cutoff is effectively infinite, copy entire tree
            if idx < cutoff_idx or cutoff_idx >= 10**9:
                for p in step.rglob("*"):
                    if p.is_dir():
                        continue
                    rel = p.relative_to(step)
                    (target_dir / rel).parent.mkdir(parents=True, exist_ok=True)
                    try:
                        shutil.copy2(p, target_dir / rel)
                    except Exception:
                        pass
            else:
                # idx == cutoff_idx: copy top-level files (optionally bounded by upto_filename)
                files = sorted([p for p in step.iterdir() if p.is_file()])
                for f in files:
                    if upto_filename and f.name > upto_filename:
                        continue
                    try:
                        shutil.copy2(f, target_dir / f.name)
                    except Exception:
                        pass


def prepare_user_gate(dest_base: str | Path, *, chapter_id: str, version: int, tp_dir: str, strip_done: bool = True, remove_final_draft_and_suggestions: bool = True) -> None:
    """Modify a partial snapshot to simulate user-in-the-loop.

    - If strip_done: remove DONE marker from brainstorm.txt under the specified tp_dir
    - If remove_final_draft_and_suggestions: delete touch_point_draft.txt and suggestions.txt
    """
    base = Path(dest_base)
    step_dir = base / "iterations" / chapter_id / f"pipeline_v{version}" / tp_dir
    if strip_done:
        bs = step_dir / "brainstorm.txt"
        if bs.exists():
            try:
                lines = bs.read_text(encoding="utf-8").splitlines()
                while lines and not lines[-1].strip():
                    lines.pop()
                if lines and re.match(r"^(?i:done)(?:\s*[\.\-—…!])?$", lines[-1].strip()):
                    lines.pop()
                bs.write_text("\n".join(lines).rstrip() + ("\n" if lines else ""), encoding="utf-8")
            except Exception:
                pass
    if remove_final_draft_and_suggestions:
        for name in ("touch_point_draft.txt", "suggestions.txt"):
            p = step_dir / name
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass


# ---------------------------
# Current step getters
# ---------------------------

def _latest_pipeline_dir(base: Path, chapter_id: str) -> Optional[Path]:
    p = base / "iterations" / chapter_id
    if not p.exists():
        return None
    candidates = sorted([d for d in p.iterdir() if d.is_dir() and re.match(r"^pipeline_v\d+$", d.name)], key=lambda d: int(d.name.split("_v")[-1]))
    return candidates[-1] if candidates else None


def _last_step_dir(pipeline_dir: Path) -> Optional[Path]:
    dirs = sorted([d for d in pipeline_dir.iterdir() if d.is_dir() and re.match(r"^\d+_", d.name)], key=lambda d: int(d.name.split("_", 1)[0]))
    return dirs[-1] if dirs else None


def _last_log_file(step_dir: Path) -> Optional[Path]:
    files = sorted([f for f in step_dir.glob("*.txt") if f.is_file() and f.name[0:2].isdigit()])
    return files[-1] if files else None


def get_current_step_response(base_dir: str | Path) -> Optional[str]:
    base = Path(base_dir)
    # Choose latest chapter with iterations present
    it = base / "iterations"
    chapters = sorted([d for d in it.iterdir() if d.is_dir() and d.name.startswith("CHAPTER_")]) if it.exists() else []
    if not chapters:
        return None
    last_ch = chapters[-1]
    pipe = _latest_pipeline_dir(base, last_ch.name)
    if not pipe:
        return None
    step = _last_step_dir(pipe)
    if not step:
        return None
    log = _last_log_file(step)
    if not log:
        return None
    d = parse_prompt_response_file(log)
    return d.get("RESPONSE")


def get_current_step_prompt(base_dir: str | Path) -> Optional[str]:
    base = Path(base_dir)
    it = base / "iterations"
    chapters = sorted([d for d in it.iterdir() if d.is_dir() and d.name.startswith("CHAPTER_")]) if it.exists() else []
    if not chapters:
        return None
    last_ch = chapters[-1]
    pipe = _latest_pipeline_dir(base, last_ch.name)
    if not pipe:
        return None
    step = _last_step_dir(pipe)
    if not step:
        return None
    log = _last_log_file(step)
    if not log:
        return None
    d = parse_prompt_response_file(log)
    return d.get("USER")


# ---------------------------
# Golden rebuild via MockLLM (Task 10)
# ---------------------------

def _append_done_to_all_brainstorms(base_dir: Path) -> int:
    """Append DONE to any brainstorm.txt under iterations/*/* that lacks it.
    Returns the number of files modified.
    """
    count = 0
    iters = base_dir / "iterations"
    if not iters.exists():
        return 0
    for bs in iters.rglob("brainstorm.txt"):
        try:
            txt = bs.read_text(encoding="utf-8")
            if not txt.strip():
                continue
            # Reuse brainstorm DONE check from pipelines.common via a light import
            from .pipelines.common import brainstorm_has_done as _has_done
            if _has_done(txt):
                continue
            # Append DONE on its own line
            end = "\n" if txt and not txt.endswith("\n") else ""
            bs.write_text(txt + end + "DONE\n", encoding="utf-8")
            count += 1
        except Exception:
            continue
    return count


def _copy_book_roots(src: Path, dst: Path) -> None:
    """Copy SETTING.yaml, CHARACTERS.yaml, chapters/, and top-level per-chapter summaries from src into dst (overwrite).

    In addition to roots, we proactively mirror iterations/<CHAPTER>/(story_so_far.txt, story_relative_to.txt)
    so that early pipeline steps (like ordering) see the same summaries as the golden source. This helps
    MockLLM matching by keeping regenerated USER prompts aligned with the golden index.
    """
    import shutil as _sh
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("SETTING.yaml", "CHARACTERS.yaml"):
        sp = src / name
        if sp.exists():
            (dst / name).parent.mkdir(parents=True, exist_ok=True)
            _sh.copy2(sp, dst / name)
    s_ch = src / "chapters"
    d_ch = dst / "chapters"
    if s_ch.exists():
        _sh.copytree(s_ch, d_ch, dirs_exist_ok=True)
    # Ensure iterations dir exists
    (dst / "iterations").mkdir(parents=True, exist_ok=True)
    # Copy per-chapter summary files if present to ensure prompt parity during rebuild
    s_it = src / "iterations"
    d_it = dst / "iterations"
    if s_it.exists():
        for ch_dir in sorted([p for p in s_it.iterdir() if p.is_dir() and p.name.startswith("CHAPTER_")]):
            target = d_it / ch_dir.name
            target.mkdir(parents=True, exist_ok=True)
            # 1) Copy top-level per-chapter summaries used by early templates
            for fname in ("story_so_far.txt", "story_relative_to.txt"):
                sp = ch_dir / fname
                if sp.exists():
                    try:
                        _sh.copy2(sp, target / fname)
                    except Exception:
                        pass
            # 2) Also pre-copy brainstorm.txt files for all steps so ORDERING prompts
            #    see the exact same bullet lists as the golden source (avoids MockLLM mismatches)
            try:
                for pipe_dir in sorted([p for p in ch_dir.iterdir() if p.is_dir() and re.match(r"^pipeline_v\d+$", p.name)]):
                    rel_pipe = pipe_dir.relative_to(s_it)
                    dst_pipe = d_it / rel_pipe
                    dst_pipe.mkdir(parents=True, exist_ok=True)
                    # Copy any brainstorm.txt under step dirs
                    for step_dir in sorted([p for p in pipe_dir.iterdir() if p.is_dir() and re.match(r"^\d+_", p.name)]):
                        src_brain = step_dir / "brainstorm.txt"
                        if src_brain.exists():
                            dst_step = dst_pipe / step_dir.name
                            dst_step.mkdir(parents=True, exist_ok=True)
                            try:
                                _sh.copy2(src_brain, dst_step / "brainstorm.txt")
                            except Exception:
                                pass
            except Exception:
                # Best-effort; ignore any issues with copying brainstorms
                pass


def rebuild_golden_with_mock(src_base: str | Path, dest_base: str | Path, *, clear_dest: bool = True, log_progress: bool = True) -> Dict[str, str]:
    """Rebuild a golden book using MockLLM responses.

    - Copies SETTING.yaml, CHARACTERS.yaml, and chapters/ from src_base into dest_base
    - Runs pipelines for each chapter using MockLLM with GW_MOCKLLM_GOLDEN_BASE=src_base
    - Automates human-in-the-loop gates by appending DONE to brainstorm.txt files
    - Repeats runs until no gates remain and chapters finish
    - Writes prompt_hash into dest_base

    Returns a small summary dict.
    """
    src = Path(src_base)
    dst = Path(dest_base)
    import shutil as _sh
    if clear_dest and dst.exists():
        _sh.rmtree(dst, ignore_errors=True)
    _copy_book_roots(src, dst)

    # Configure environment for this rebuild
    import os as _os
    _os.environ["GW_BOOK_BASE_DIR"] = str(dst.resolve())
    _os.environ["GW_USE_MOCK_LLM"] = "1"
    _os.environ["GW_MOCKLLM_GOLDEN_BASE"] = str(src.resolve())
    # Strict MockLLM to surface mismatches; caller can relax via env if desired
    _os.environ.setdefault("GW_MOCKLLM_FALLBACK", "0")

    # Collect chapters to process (deterministic order)
    chapters_dir = dst / "chapters"
    ch_files = sorted([p for p in chapters_dir.glob("CHAPTER_*.yaml") if p.is_file()])
    processed = 0
    gates_fixed_total = 0
    # Iterate per chapter and re-run until complete (bounded safety loop)
    for ch in ch_files:
        chapter_id = _chapter_id_from_path(str(ch))
        # Determine next version number in this dest base by counting existing pipeline dirs
        pipe_root = dst / "iterations" / chapter_id
        version = 1
        try:
            if pipe_root.exists():
                pipes = [d for d in pipe_root.iterdir() if d.is_dir() and re.match(r"^pipeline_v\d+$", d.name)]
                if pipes:
                    version = max(int(d.name.split("_v")[-1]) for d in pipes) + 1
        except Exception:
            version = 1

        # Seed the new pipeline version with golden brainstorms so ORDERING sees identical [bullets]
        # Strategy: prefer the most recent source pipeline version that actually has a brainstorm.txt for each step.
        # We iterate source pipelines in descending version order and copy the first available brainstorm per step.
        try:
            src_it = Path(src) / "iterations" / chapter_id
            dest_pipe = pipe_root / f"pipeline_v{version}"
            copied_steps: set[str] = set()
            if src_it.exists():
                # Collect pipeline dirs sorted by version DESC so newer ones win
                def _ver_num(p: Path) -> int:
                    try:
                        return int(p.name.split("_v")[-1])
                    except Exception:
                        return -1
                src_pipes = [d for d in src_it.iterdir() if d.is_dir() and re.match(r"^pipeline_v\d+$", d.name)]
                for pipe in sorted(src_pipes, key=_ver_num, reverse=True):
                    # For each step dir under this pipeline, copy brainstorm.txt if present and not already copied
                    for step_dir in sorted([p for p in pipe.iterdir() if p.is_dir() and re.match(r"^\d+_", p.name)]):
                        if step_dir.name in copied_steps:
                            continue
                        src_brain = step_dir / "brainstorm.txt"
                        if not src_brain.exists():
                            continue
                        dst_step = dest_pipe / step_dir.name
                        try:
                            dst_step.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(src_brain, dst_step / "brainstorm.txt")
                            copied_steps.add(step_dir.name)
                        except Exception:
                            # Ignore copy errors for individual steps
                            pass
        except Exception:
            # Best-effort seeding; continue if anything goes wrong
            pass

        # Run until no UserActionRequired is raised and no brainstorm needs DONE
        max_cycles = 10
        for cycle in range(1, max_cycles + 1):
            try:
                _log_run(f"golden-rebuild:run chapter={chapter_id} v={version} cycle={cycle}")
            except Exception:
                pass
            try:
                _run_pipelines_for_chapter(str(ch), version, log_llm=False)
                # If run completed without raising, chapter is done
                processed += 1
                break
            except Exception as e:
                from .context import UserActionRequired as _UAR
                if isinstance(e, _UAR):
                    # Try auto-fixing gates (append DONE, etc.) and loop
                    fixed = _append_done_to_all_brainstorms(dst)
                    gates_fixed_total += fixed
                    continue
                # Unknown error: re-raise to surface
                raise
        else:  # pragma: no cover - safety
            raise RuntimeError(f"golden-rebuild: exceeded max cycles for {chapter_id}")

    # Write new prompts hash into the dest base
    write_prompt_hash(dst, compute_prompts_hash("prompts"))
    summary = {
        "chapters": str(len(ch_files)),
        "processed": str(processed),
        "gates_fixed": str(gates_fixed_total),
        "dest": str(dst.resolve()),
    }
    try:
        if log_progress:
            _log_run(f"golden-rebuild:done {summary}")
    except Exception:
        pass
    return summary
