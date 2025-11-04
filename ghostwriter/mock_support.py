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
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Local imports kept light to avoid cycles
from .templates import apply_template
from .context import RunContext
from .env import get_chapters_dir

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
    # Build replacements using the same common logic as pipelines
    from .touch_point import _build_pipeline_replacements as _build
    ctx = RunContext.from_paths(chapter_path=str(get_chapters_dir() / f"{chapter_id}.yaml"), version=version, allow_missing_chapter=False)
    state, _ = _load_tp_state(step_dir)
    # Determine touch-point dict for content/type labels
    tp_dict = {"type": step_dir.name.split("_", 1)[-1], "content": _read_touchpoint_text(step_dir)}
    reps = _build(ctx.setting, ctx.chapter, chapter_id, version, tp_dict, state, prior_paragraph="", ctx=ctx)
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
    template = _template_for_file(tp_type or step_dir.name.split("_", 1)[-1], log_p.name)
    if not template:
        return None
    reps = _build_replacements_for_step(step_dir, chapter_id, version)
    user = apply_template(Path("prompts") / template, reps)
    # Special handling: brainstorm resumes may include seed bullets; we intentionally rely on template only
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

def copy_partial_book(src_base: str | Path, dest_base: str | Path, *, chapter_id: str, version: int, upto_step_dir: str, upto_filename: Optional[str] = None, clear_dest: bool = True) -> None:
    """Copy a partial snapshot of a golden book.

    - Copies SETTING.yaml, CHARACTERS.yaml, chapters/*, and iterations/<chapter_id>/pipeline_v<version>/ up to and including upto_step_dir
    - If upto_filename is provided, include only files up to that filename (lexicographically) in the target step directory
    - If clear_dest is True, the destination directory is removed first if it exists
    - Excludes later step directories (> numeric prefix of upto_step_dir)
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
    # Copy iterations up to step
    s_it = src / "iterations" / chapter_id / f"pipeline_v{version}"
    d_it = dst / "iterations" / chapter_id / f"pipeline_v{version}"
    if not s_it.exists():
        return
    d_it.mkdir(parents=True, exist_ok=True)
    # Determine cutoff index
    try:
        cutoff_idx = int(upto_step_dir.split("_", 1)[0])
    except Exception:
        cutoff_idx = 999
    for step in sorted([p for p in s_it.iterdir() if p.is_dir()]):
        try:
            idx = int(step.name.split("_", 1)[0])
        except Exception:
            idx = 999
        if idx > cutoff_idx:
            continue
        target_dir = d_it / step.name
        target_dir.mkdir(parents=True, exist_ok=True)
        if idx < cutoff_idx:
            # Copy entire earlier step dir
            for p in step.rglob("*"):
                if p.is_dir():
                    continue
                rel = p.relative_to(step)
                (target_dir / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, target_dir / rel)
        else:
            # Copy only files up to upto_filename (if provided)
            files = sorted([p for p in step.iterdir() if p.is_file()])
            for f in files:
                if upto_filename and f.name > upto_filename:
                    continue
                shutil.copy2(f, target_dir / f.name)


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
