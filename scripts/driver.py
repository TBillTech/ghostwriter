import re
import sys
import time
import yaml
import json
import random
import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable

from dotenv import load_dotenv
import os
from yaml.loader import SafeLoader as _PySafeLoader
from yaml.dumper import SafeDumper as _PySafeDumper

try:
    # OpenAI v2 client (requirements pinned to openai==2.0.0)
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

# ---------------------------
# Helpers & Error Types
# ---------------------------

class GWError(Exception):
    """Base error for GhostWriter driver."""


class MissingFileError(GWError):
    pass


class InvalidYAMLError(GWError):
    pass

class ValidationError(GWError):
    pass

def _yaml_load_py(content: str):
    """Pure-Python YAML load using SafeLoader to avoid C-accelerated libyaml crashes."""
    return yaml.load(content, Loader=_PySafeLoader)

def _yaml_dump_py(obj) -> str:
    """Pure-Python YAML dump using SafeDumper to avoid C-accelerated libyaml."""
    return yaml.dump(obj, sort_keys=False, Dumper=_PySafeDumper)

def load_yaml(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        raise MissingFileError(f"Required file not found: {path}")
    except Exception as e:
        raise GWError(f"Unable to read file {path}: {e}")

    def _format_yaml_snippet(text: str, line: int, col: int, context: int = 2) -> str:
        try:
            lines = text.splitlines()
            if not lines:
                return ""
            idx = max(0, min(line, len(lines) - 1))
            start = max(0, idx - context)
            end = min(len(lines), idx + context + 1)
            out_lines = []
            for i in range(start, end):
                prefix = ">" if i == idx else " "
                out_lines.append(f"{prefix} {i+1:4}: {lines[i]}")
                if i == idx:
                    caret_pad = " " * (col + 8)
                    out_lines.append(f"          {caret_pad}^")
            return "\n".join(out_lines)
        except Exception:
            return ""

    try:
        return _yaml_load_py(content)
    except yaml.YAMLError as e:  # type: ignore[attr-defined]
        # Try to include line/column info
        mark = getattr(e, "problem_mark", None) or getattr(e, "context_mark", None)
        if mark is not None and hasattr(mark, "line") and hasattr(mark, "column"):
            line = int(getattr(mark, "line", 0))
            col = int(getattr(mark, "column", 0))
            snippet = _format_yaml_snippet(content, line, col)
            msg = (
                f"Invalid YAML in {path} at line {line+1}, column {col+1}: {getattr(e, 'problem', e)}"
            )
            if snippet:
                msg += f"\n{snippet}"
            raise InvalidYAMLError(msg)
        else:
            raise InvalidYAMLError(f"Invalid YAML in {path}: {e}")
    except Exception as e:
        # Catch non-YAML exceptions (e.g., IndexError from loader) and rethrow consistently
        raise InvalidYAMLError(f"Invalid YAML in {path}: {e}")

def save_text(path, content):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def read_file(path):
    p = Path(path)
    if not p.exists():
        raise MissingFileError(f"Required file not found: {path}")
    try:
        return p.read_text(encoding="utf-8")
    except Exception as e:
        raise GWError(f"Unable to read file {path}: {e}")

def load_env():
    # Load .env if present
    load_dotenv(override=False)

def get_client():
    """Return OpenAI client if API key is present; otherwise None for mock mode."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return None
    try:
        return OpenAI()
    except Exception:
        return None

def get_model() -> str:
    return os.getenv("OPENAI_MODEL", "gpt-4o-mini")

def with_backoff(fn, *, retries=3, base_delay=1.0, jitter=0.2):
    last_err = None
    for i in range(retries):
        try:
            return fn()
        except Exception as e:  # pragma: no cover
            last_err = e
            time.sleep(base_delay * (2 ** i) + random.random() * jitter)
    if last_err:
        raise last_err

def llm_complete(
    prompt: str,
    *,
    system: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 800,
    model: Optional[str] = None,
) -> str:
    """Call OpenAI if configured; else return a deterministic mock response."""
    client = get_client()
    if client is None:
        # Mock response for offline/dev usage
        head = (prompt[:220] + "...") if len(prompt) > 220 else prompt
        return f"[MOCK LLM RESPONSE]\nSystem: {system or 'n/a'}\nTemp: {temperature}\n---\n{head}"

    model_name = model or get_model()

    def _do_call():
        # Prefer chat.completions for richer prompting
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system or "You are a helpful writing assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    return with_backoff(_do_call)

def _env_int(name: str, default: int) -> int:
    """Read integer from environment with a default; ignores invalid values."""
    try:
        v = int(os.getenv(name, str(default)))
        if v <= 0:
            return default
        return v
    except Exception:
        return default

def _env_str(name: str) -> Optional[str]:
    """Read string from environment; returns None if not set or empty."""
    val = os.getenv(name)
    if val is None or str(val).strip() == "":
        return None
    return val

def _env_float(name: str, default: float) -> float:
    """Read float from environment with a default; ignores invalid values."""
    try:
        v = float(os.getenv(name, str(default)))
        return v
    except Exception:
        return default

# ---------------------------
# Deterministic pipeline scaffolding (Tasks 1, 2, 3, 5)
# ---------------------------

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
                return [{"type": k, "content": content, "raw": _yaml_dump_py(item).strip()}]
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
            return [{"type": "narration", "content": str(item), "raw": _yaml_dump_py(item).strip()}]
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

# ---- Output validators (Task 3)

def validate_text(output: str) -> Tuple[bool, str]:
    return (True, "ok")

def validate_bullet_list(output: str) -> Tuple[bool, str]:
    # Ignore headings (# ...) and empty lines when parsing
    lines = [ln.rstrip() for ln in output.splitlines() if ln.strip() and not ln.lstrip().startswith('#')]
    bullets = [ln for ln in lines if ln.lstrip().startswith("*")]
    if len(bullets) >= 2:
        return True, "ok"
    return False, "Expected a bullet list with at least 2 lines starting with '*'"

_ACTOR_LINE_RE = re.compile(r"^\s*([A-Za-z0-9_.\-]+)\s*:\s+.+")

def validate_actor_list(output: str) -> Tuple[bool, str]:
    lines = [ln for ln in output.splitlines() if ln.strip() and not ln.lstrip().startswith('#')]
    count = 0
    for ln in lines:
        if _ACTOR_LINE_RE.match(ln):
            count += 1
    if count >= 2:
        return True, "ok"
    return False, "Expected an actor list with at least 2 actor-attributed lines like 'id: ...'"

def validate_agenda_list(output: str) -> Tuple[bool, str]:
    # Heuristic: at least 2 structured lines (bullets or 'id: goal' style). Ignore headings and empties.
    lines = [ln for ln in output.splitlines() if ln.strip() and not ln.lstrip().startswith('#')]
    bullets = [ln for ln in lines if ln.lstrip().startswith(('*', '-'))]
    if len(bullets) >= 2:
        return True, "ok"
    # Fallback: check for at least two 'x: y' lines
    colon = [ln for ln in lines if re.search(r"\w\s*:\s+.+", ln)]
    if len(colon) >= 2:
        return True, "ok"
    return False, "Expected an agenda list (>=2 items as bullets or 'key: value' lines)"

def _retry_with_validation(callable_fn, validator, *, max_attempts: int = 3) -> str:
    """Call callable_fn repeatedly until validator passes or attempts exhausted.
    Raises ValidationError on persistent failure.
    """
    last_err = ""
    for attempt in range(1, max_attempts + 1):
        out = callable_fn()
        ok, reason = validator(out)
        if ok:
            return out
        last_err = reason
    raise ValidationError(f"Output validation failed after {max_attempts} attempts: {last_err}")

# ---- Brainstorm post-processing helpers

def _extract_bullet_contents(text: str) -> List[str]:
    """Extract bullet item contents from text (lines starting with * or -). Returns list of content strings without the marker."""
    contents: List[str] = []
    for ln in text.splitlines():
        s = ln.lstrip()
        if not s or s.startswith('#'):
            continue
        if s.startswith(('*', '-')):
            # remove first marker and following spaces
            after = s[1:].lstrip()
            if after:
                contents.append(after)
    return contents

def _rebuild_bullets(contents: List[str]) -> str:
    return "\n".join([f"* {c}" for c in contents])

def _filter_narration_brainstorm(text: str, min_chars: int = 24) -> str:
    """Drop bullets whose content length is < min_chars. Ensure at least 2 bullets by falling back to the longest available."""
    orig = _extract_bullet_contents(text)
    if not orig:
        return text
    filtered = [c for c in orig if len(c.strip()) >= min_chars]
    if len(filtered) < 2:
        # fallback: ensure at least two bullets by picking top-2 longest from original
        sorted_by_len = sorted(orig, key=lambda c: len(c.strip()), reverse=True)
        filtered = sorted_by_len[: max(2, len(sorted_by_len))]
        filtered = filtered[:2]
    return _rebuild_bullets(filtered)

def _truncate_brainstorm(text: str, limit: int = 10) -> str:
    contents = _extract_bullet_contents(text)
    if not contents:
        return text
    return _rebuild_bullets(contents[:limit])

def _inline_body_with_dialog(body_bullet: str, dialog: str) -> str:
    """Inline a body-language bullet as a leading clause before dialog.
    - body_bullet is expected like '* With a frown, Henry ...'
    - We remove the bullet marker and ensure it ends with a comma (if no terminal punctuation).
    - Then we concatenate with a space before the dialog.
    """
    if not body_bullet:
        return dialog.strip()
    body = body_bullet.strip()
    if body.startswith(('*', '-')):
        body = body[1:].lstrip()
    # Ensure trailing separator: keep if ends with punctuation, else add comma
    if not re.search(r"[\.,?!:;—-]\s*$", body):
        body = body + ","
    return f"{body} {dialog.strip()}".strip()

def _parse_agenda_by_actor(text: str) -> Dict[str, str]:
    """Parse an agenda text into a mapping of actor_id -> bullet list string.
    Expected format:
      ActorId:
      * item
      * item
    Repeats for each actor. We collect bullets until the next header like 'Name:'."""
    by_actor: Dict[str, List[str]] = {}
    current: Optional[str] = None
    for ln in text.splitlines():
        if not ln.strip():
            continue
        # Header line like 'ActorId:' with no leading bullet
        if not ln.lstrip().startswith(('*', '-')) and re.match(r"^\s*([A-Za-z0-9_.\-]+)\s*:\s*$", ln):
            m = re.match(r"^\s*([A-Za-z0-9_.\-]+)\s*:\s*$", ln)
            if m:
                current = m.group(1).strip()
                by_actor.setdefault(current, [])
            continue
        # Bullet line under current
        if current is not None and ln.lstrip().startswith(('*', '-')):
            content = ln.lstrip()[1:].lstrip()
            if content:
                by_actor[current].append(content)
    # Build string values
    return {aid: _rebuild_bullets(items) for aid, items in by_actor.items() if items}

def _pick_bullet_by_index(text: str, index1: int) -> str:
    """Return the Nth (1-based) bullet line as '* content', or empty if unavailable."""
    if index1 <= 0:
        return ""
    items = _extract_bullet_contents(text)
    if 1 <= index1 <= len(items):
        return f"* {items[index1-1]}"
    return ""

# ---- Parseable artifact helpers (Task 5)

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
    """Write a parseable draft file consisting of touch-point/result pairs.
    records: iterable of (tp_id, tp_type, touchpoint_text, polished_text)
    """
    out_path = iter_dir_for(chapter_id) / f"draft_v{version}.txt"
    chunks = [format_draft_record(*r) for r in records]
    save_text(out_path, "".join(chunks))
    return out_path

def read_draft_records(path: str) -> List[Dict[str, str]]:
    text = read_file(path)
    # Split by BEGIN_TP markers
    blocks = re.split(rf"(?m)^\s*{BEGIN_TP}\s+", text)
    results: List[Dict[str, str]] = []
    for blk in blocks:
        blk = blk.strip()
        if not blk:
            continue
        # Expect 'id=.. type=..\n...END_TP\nBEGIN_RESULT\n...END_RESULT'
        header_line, _, rest = blk.partition("\n")
        m = re.search(r"id=([^\s]+)\s+type=([^\s]+)", header_line)
        if not m:
            # Skip malformed block
            continue
        tp_id, tp_type = m.group(1), m.group(2)
        content, _, rest2 = rest.partition(f"\n{END_TP}\n")
        if not rest2:
            continue
        if BEGIN_RES not in rest2:
            continue
        _, _, after_begin = rest2.partition(f"{BEGIN_RES}\n")
        polished, _, _ = after_begin.partition(f"\n{END_RES}")
        results.append({
            "id": tp_id,
            "type": tp_type,
            "touchpoint": content,
            "result": polished,
        })
    return results

def format_suggestions_record(tp_id: str, tp_type: str, touchpoint_text: str, suggestions_text: str) -> str:
    # Reuse same sentinels for consistency
    return format_draft_record(tp_id, tp_type, touchpoint_text, suggestions_text)

def write_suggestions_records(chapter_id: str, version: int, records: Iterable[Tuple[str, str, str, str]]) -> Path:
    out_path = iter_dir_for(chapter_id) / f"suggestions_v{version}.txt"
    chunks = [format_suggestions_record(*r) for r in records]
    save_text(out_path, "".join(chunks))
    return out_path

def read_suggestions_records(path: str) -> List[Dict[str, str]]:
    return read_draft_records(path)

# ---------------------------
# Pipeline execution (Task 2)
# ---------------------------

def _build_pipeline_replacements(setting: dict, chapter: dict, chapter_id: str, version: int, tp: TouchPoint, state: ChapterState, *, prior_paragraph: str = "") -> Dict[str, str]:
    reps = build_common_replacements(setting, chapter, chapter_id, version)
    # touch-point variants for templates
    touch_text = tp.get("content", "")
    tp_type = tp.get("type", "")
    extra = {
        "[TOUCH_POINT]": touch_text,
        "[touch_point]": touch_text,
        "[TOUCH-POINT]": touch_text,
        "[touch-point]": touch_text,
        "[TOUCHPOINT]": touch_text,
        "[touchpoint]": touch_text,
        "[TOUCH_POINT_TYPE]": tp_type,
        # Actors/scene synonyms
        "[ACTIVE_ACTORS]": ", ".join(state.active_actors),
        "[actors]": ", ".join(state.active_actors),
        "[SCENE]": state.current_scene or "",
        "[scene]": state.current_scene or "",
        "[FORESHADOWING]": ", ".join(state.foreshadowing),
        # Prior paragraph synonyms
        "[PRIOR_PARAGRAPH]": prior_paragraph or "",
        "[prior_paragraph]": prior_paragraph or "",
    }
    # Inject selected setting/characters blocks if available
    if getattr(state, "setting_block", ""):
        extra["[SETTING]"] = state.setting_block
    else:
        extra.setdefault("[SETTING]", "")
    # Characters block: prefer explicit 'setting' touch-point; else fallback to active_actors selection
    chars_block = getattr(state, "characters_block", "")
    if not chars_block and state.active_actors:
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
                chars_block = _yaml_dump_py({"Selected-Characters": sel_chars})
        except Exception:
            chars_block = chars_block or ""
    extra["[CHARACTERS]"] = chars_block or ""
    # Provide dialog history as YAML-ish for all known actors
    try:
        dialog_map = {a: state.recent_dialog(a) for a in (state.active_actors or [])}
        extra["[DIALOG_HISTORY]"] = _yaml_dump_py(dialog_map)
    except Exception:
        extra["[DIALOG_HISTORY]"] = ""
    reps.update(extra)
    return reps

def _apply_step(template_filename: str, reps: Dict[str, str]) -> str:
    path = Path("prompts") / template_filename
    if path.exists():
        return apply_template(str(path), reps)
    # Fallback: return a generic prompt made from replacements
    return (f"Context follows.\n\n{reps.get('[SETTING.yaml]', '')}\n\n{reps.get('[CHAPTER_xx.yaml]', '')}\n\n"
            f"Touch-Point ({reps.get('[TOUCH_POINT_TYPE]', '')}): {reps.get('[TOUCH_POINT]', '')}\nState: actors={reps.get('[ACTIVE_ACTORS]', '')}, scene={reps.get('[SCENE]', '')}\n")

def _env_for(step_key: str, *, default_temp: float = 0.2, default_max_tokens: int = 800) -> Tuple[Optional[str], float, int]:
    """Resolve per-prompt env: returns (model, temperature, max_tokens). step_key examples: BRAIN_STORM, ORDERING, GENERATE_NARRATION, ACTOR_ASSIGNMENT, BODY_LANGUAGE, AGENDA, CHARACTER_DIALOG, SUBTLE_EDIT, POLISH_PROSE.
    """
    model = _env_str(f"GW_MODEL_{step_key}")
    temp = _env_float(f"GW_TEMP_{step_key}", default_temp)
    max_tokens = _env_int(f"GW_MAX_TOKENS_{step_key}", default_max_tokens)
    return model, temp, max_tokens

from typing import Callable

def _llm_call_with_validation(
    system: str,
    user: str,
    *,
    model: Optional[str],
    temperature: float,
    max_tokens: int,
    validator,
    log_maker: Optional[Callable[[int], Optional[Path]]] = None,
) -> str:
    last_reason = ""
    for attempt in range(1, 4):
        out = llm_complete(user, system=system, temperature=temperature, max_tokens=max_tokens, model=model)
        log_path = log_maker(attempt) if log_maker else None
        if log_path is not None:
            try:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("w", encoding="utf-8") as f:
                    f.write("=== SYSTEM ===\n" + (system or "") + "\n\n")
                    f.write("=== USER ===\n" + user + "\n\n")
                    f.write("=== RESPONSE ===\n" + out + "\n")
            except Exception:
                pass
        ok, reason = validator(out)
        if ok:
            return out
        last_reason = reason
    raise ValidationError(f"Output validation failed after 3 attempts: {last_reason}")

def _polish_snippet(text: str, setting: dict, chapter: dict, chapter_id: str, version: int) -> str:
    prompt = build_polish_prompt(setting, chapter, chapter_id, version, text)
    model, temp, max_tokens = _env_for("POLISH_PROSE", default_temp=0.2, default_max_tokens=2000)
    return llm_complete(prompt, system="You are a ghostwriter polishing and cleaning prose.", temperature=temp, max_tokens=max_tokens, model=model)

def _parse_actor_lines(actor_list_text: str) -> List[Tuple[str, str]]:
    """Return list of (actor_id, line_hint) from actor list text."""
    pairs: List[Tuple[str, str]] = []
    for ln in actor_list_text.splitlines():
        m = _ACTOR_LINE_RE.match(ln)
        if not m:
            continue
        actor_id = m.group(1)
        # Extract text after first ':'
        _, _, rest = ln.partition(":")
        pairs.append((actor_id.strip(), rest.strip()))
    return pairs

def run_narration_pipeline(tp: TouchPoint, state: ChapterState, *, setting: dict, chapter: dict, chapter_id: str, version: int, tp_index: int, prior_paragraph: str = "", log_dir: Optional[Path] = None) -> str:
    reps = _build_pipeline_replacements(setting, chapter, chapter_id, version, tp, state, prior_paragraph=prior_paragraph)
    # 1) Brainstorm → bullet list
    sys1 = "You are brainstorming narrative beats as concise bullet points."
    user1 = _apply_step("narration_brain_storm_prompt.md", reps)
    model, temp, max_toks = _env_for("BRAIN_STORM", default_temp=0.4, default_max_tokens=600)
    brainstorm = _llm_call_with_validation(
        sys1, user1, model=model, temperature=temp, max_tokens=max_toks, validator=validate_bullet_list,
        log_maker=(lambda attempt: (log_dir / f"{tp_index:02d}_brainstorm{'_r'+str(attempt) if attempt>1 else ''}.txt")) if log_dir else None,
    )

    # 2) Ordering → bullet list
    reps2 = dict(reps)
    # Map brainstorm output into common placeholders used by ordering templates
    reps2["[BULLET_IDEAS]"] = brainstorm
    reps2["[BULLETS]"] = brainstorm
    reps2["[bullets]"] = brainstorm
    sys2 = "You will order and refine brainstormed bullet points for coherent flow."
    user2 = _apply_step("ordering_prompt.md", reps2)
    model2, temp2, max2 = _env_for("ORDERING", default_temp=0.2, default_max_tokens=600)
    ordered = _llm_call_with_validation(
        sys2, user2, model=model2, temperature=temp2, max_tokens=max2, validator=validate_bullet_list,
        log_maker=(lambda attempt: (log_dir / f"{tp_index:02d}_ordering{'_r'+str(attempt) if attempt>1 else ''}.txt")) if log_dir else None,
    )

    # 3) Generate narration → text (fallback template name if missing)
    reps3 = dict(reps2)
    reps3["[ORDERED_BULLETS]"] = ordered
    reps3["[ordered_bullets]"] = ordered
    # Some templates may just reuse [bullets] for the ordered list
    reps3["[bullets]"] = ordered
    sys3 = "You write narrative prose from ordered bullets, keeping voice consistent."
    user3 = _apply_step("generate_narration_prompt.md", reps3)
    model3, temp3, max3 = _env_for("GENERATE_NARRATION", default_temp=0.35, default_max_tokens=1000)
    narration = _llm_call_with_validation(
        sys3, user3, model=model3, temperature=temp3, max_tokens=max3, validator=validate_text,
        log_maker=(lambda attempt: (log_dir / f"{tp_index:02d}_generate_narration{'_r'+str(attempt) if attempt>1 else ''}.txt")) if log_dir else None,
    )

    # Polish
    polished = _polish_snippet(narration, setting, chapter, chapter_id, version)
    return polished

def run_explicit_pipeline(tp: TouchPoint, state: ChapterState, *, setting: dict, chapter: dict, chapter_id: str, version: int, tp_index: int, prior_paragraph: str = "", log_dir: Optional[Path] = None) -> str:
    reps = _build_pipeline_replacements(setting, chapter, chapter_id, version, tp, state, prior_paragraph=prior_paragraph)
    # Brainstorm
    sys1 = "Brainstorm explicit dialog beats as bullet points."
    user1 = _apply_step("explicit_brain_storm_prompt.md", reps)
    m1, t1, k1 = _env_for("BRAIN_STORM", default_temp=0.45, default_max_tokens=600)
    brainstorm = _llm_call_with_validation(
        sys1, user1, model=m1, temperature=t1, max_tokens=k1, validator=validate_bullet_list,
        log_maker=(lambda attempt: (log_dir / f"{tp_index:02d}_brainstorm{'_r'+str(attempt) if attempt>1 else ''}.txt")) if log_dir else None,
    )
    # Post-process: drop overly short bullets (min 24 chars), keep at least 2
    brainstorm = _filter_narration_brainstorm(brainstorm, min_chars=24)
    # Post-process: limit to at most 10 bullets
    brainstorm = _truncate_brainstorm(brainstorm, limit=10)
    # Ordering
    reps2 = dict(reps)
    reps2["[BULLET_IDEAS]"] = brainstorm
    reps2["[BULLETS]"] = brainstorm
    reps2["[bullets]"] = brainstorm
    sys2 = "Order explicit dialog beats for flow."
    user2 = _apply_step("ordering_prompt.md", reps2)
    m2, t2, k2 = _env_for("ORDERING", default_temp=0.25, default_max_tokens=600)
    ordered = _llm_call_with_validation(
        sys2, user2, model=m2, temperature=t2, max_tokens=k2, validator=validate_bullet_list,
        log_maker=(lambda attempt: (log_dir / f"{tp_index:02d}_ordering{'_r'+str(attempt) if attempt>1 else ''}.txt")) if log_dir else None,
    )
    # Actor assignment → actor lines ("id: line" entries)
    reps3 = dict(reps2)
    reps3["[ORDERED_BULLETS]"] = ordered
    reps3["[ordered_bullets]"] = ordered
    reps3["[bullets]"] = ordered
    sys3 = "Assign dialog lines to actors as 'id: line' entries."
    user3 = _apply_step("actor_assignment_prompt.md", reps3)
    m3, t3, k3 = _env_for("ACTOR_ASSIGNMENT", default_temp=0.25, default_max_tokens=600)
    actor_lines = _llm_call_with_validation(
        sys3, user3, model=m3, temperature=t3, max_tokens=k3, validator=validate_actor_list,
        log_maker=(lambda attempt: (log_dir / f"{tp_index:02d}_actor_assignment{'_r'+str(attempt) if attempt>1 else ''}.txt")) if log_dir else None,
    )
    # Build ACTOR_LIST (ids only) from ACTOR_LINES
    pairs_for_ids = _parse_actor_lines(actor_lines)
    seen_ids: List[str] = []
    for _aid, _ in pairs_for_ids:
        aid = str(_aid).strip()
        if aid and aid not in seen_ids:
            seen_ids.append(aid)
    actor_list = ", ".join(seen_ids)
    # Parallel: body language + agenda + reactions (sequential in code but conceptually parallel)
    reps4 = dict(reps3); reps4["[ACTOR_LIST]"] = actor_list; reps4["[ACTOR_LINES]"] = actor_lines
    # Body language
    sys4a = "List body language cues as bullet points."
    user4a = _apply_step("body_language_prompt.md", reps4)
    m4a, t4a, k4a = _env_for("BODY_LANGUAGE", default_temp=0.3, default_max_tokens=300)
    body_lang = _llm_call_with_validation(
        sys4a, user4a, model=m4a, temperature=t4a, max_tokens=k4a, validator=validate_bullet_list,
        log_maker=(lambda attempt: (log_dir / f"{tp_index:02d}_body_language{'_r'+str(attempt) if attempt>1 else ''}.txt")) if log_dir else None,
    )
    # Agenda (now character-wide, grouped by actor)
    sys4b = "Produce an agenda list grouped by actor id."
    user4b = _apply_step("agenda_prompt.md", reps4)
    m4b, t4b, k4b = _env_for("AGENDA", default_temp=0.3, default_max_tokens=300)
    agenda_text = _llm_call_with_validation(
        sys4b, user4b, model=m4b, temperature=t4b, max_tokens=k4b, validator=validate_text,
        log_maker=(lambda attempt: (log_dir / f"{tp_index:02d}_agenda{'_r'+str(attempt) if attempt>1 else ''}.txt")) if log_dir else None,
    )
    agenda_by_actor = _parse_agenda_by_actor(agenda_text)
    # Reactions (per-line)
    sys4c = "Produce a reaction for each line, referencing the previous line."
    user4c = _apply_step("reaction_prompt.md", reps4)
    m4c, t4c, k4c = _env_for("REACTIONS", default_temp=0.3, default_max_tokens=400)
    reactions_text = _llm_call_with_validation(
        sys4c, user4c, model=m4c, temperature=t4c, max_tokens=k4c, validator=validate_actor_list,
        log_maker=(lambda attempt: (log_dir / f"{tp_index:02d}_reactions{'_r'+str(attempt) if attempt>1 else ''}.txt")) if log_dir else None,
    )
    # Join and render character dialog lines
    outputs: List[str] = []
    for b_index, (actor_id, line_hint) in enumerate(_parse_actor_lines(actor_lines), start=1):
        # Build a per-line prompt combining hints
        # Select the per-actor agenda block and the per-line reaction
        agenda_block = agenda_by_actor.get(actor_id, "")
        reaction_line = _pick_bullet_by_index(reactions_text, b_index)
        combined_prompt = (
            f"Line intent: {line_hint}\n"
            f"Reaction: {reaction_line}\n"
            f"Agenda: {agenda_block}\n"
            f"Body language: {body_lang}\n"
        )
        dialog_lines_ctx = state.recent_dialog(actor_id)
        resp = render_character_call(
            actor_id,
            combined_prompt,
            dialog_lines_ctx,
            temperature=0.35,
            max_tokens_line=120,
            log_file=(log_dir / f"{tp_index:02d}_b{b_index:02d}_{actor_id}.txt") if log_dir else None,
        )
        # Inline body language for this line (leading clause)
        body_for_line = _pick_bullet_by_index(body_lang, b_index)
        outputs.append(_inline_body_with_dialog(body_for_line, resp))
        if resp.strip():
            state.add_dialog_line(actor_id, resp.strip())
    text = "\n".join(outputs)
    return _polish_snippet(text, setting, chapter, chapter_id, version)

def run_implicit_pipeline(tp: TouchPoint, state: ChapterState, *, setting: dict, chapter: dict, chapter_id: str, version: int, tp_index: int, prior_paragraph: str = "", log_dir: Optional[Path] = None) -> str:
    reps = _build_pipeline_replacements(setting, chapter, chapter_id, version, tp, state, prior_paragraph=prior_paragraph)
    # Implicit brainstorm
    sys1 = "Brainstorm implicit dialog beats (indirect, subtext) as bullet points."
    user1 = _apply_step("implicit_brain_storm_prompt.md", reps)
    m1, t1, k1 = _env_for("BRAIN_STORM", default_temp=0.5, default_max_tokens=600)
    brainstorm = _llm_call_with_validation(
        sys1, user1, model=m1, temperature=t1, max_tokens=k1, validator=validate_bullet_list,
        log_maker=(lambda attempt: (log_dir / f"{tp_index:02d}_brainstorm_implicit{'_r'+str(attempt) if attempt>1 else ''}.txt")) if log_dir else None,
    )
    # Post-process: limit to at most 10 bullets
    brainstorm = _truncate_brainstorm(brainstorm, limit=10)
    # Ordering
    reps2 = dict(reps); reps2["[BULLET_IDEAS]"] = brainstorm
    sys2 = "Order implicit dialog beats for flow."
    user2 = _apply_step("ordering_prompt.md", reps2)
    m2, t2, k2 = _env_for("ORDERING", default_temp=0.25, default_max_tokens=600)
    ordered = _llm_call_with_validation(
        sys2, user2, model=m2, temperature=t2, max_tokens=k2, validator=validate_bullet_list,
        log_maker=(lambda attempt: (log_dir / f"{tp_index:02d}_ordering{'_r'+str(attempt) if attempt>1 else ''}.txt")) if log_dir else None,
    )
    # Actor assignment → actor lines ("id: line" entries)
    reps3 = dict(reps2); reps3["[ORDERED_BULLETS]"] = ordered
    sys3 = "Assign dialog lines to actors as 'id: line' entries."
    user3 = _apply_step("actor_assignment_prompt.md", reps3)
    m3, t3, k3 = _env_for("ACTOR_ASSIGNMENT", default_temp=0.25, default_max_tokens=600)
    actor_lines = _llm_call_with_validation(
        sys3, user3, model=m3, temperature=t3, max_tokens=k3, validator=validate_actor_list,
        log_maker=(lambda attempt: (log_dir / f"{tp_index:02d}_actor_assignment{'_r'+str(attempt) if attempt>1 else ''}.txt")) if log_dir else None,
    )
    # Build ACTOR_LIST (ids only) from ACTOR_LINES
    pairs_for_ids = _parse_actor_lines(actor_lines)
    seen_ids: List[str] = []
    for _aid, _ in pairs_for_ids:
        aid = str(_aid).strip()
        if aid and aid not in seen_ids:
            seen_ids.append(aid)
    actor_list = ", ".join(seen_ids)
    # Parallel helpers: body language + agenda + reactions
    reps4 = dict(reps3); reps4["[ACTOR_LIST]"] = actor_list; reps4["[ACTOR_LINES]"] = actor_lines
    sys4a = "List body language cues as bullet points."
    user4a = _apply_step("body_language_prompt.md", reps4)
    m4a, t4a, k4a = _env_for("BODY_LANGUAGE", default_temp=0.3, default_max_tokens=300)
    body_lang = _llm_call_with_validation(
        sys4a, user4a, model=m4a, temperature=t4a, max_tokens=k4a, validator=validate_bullet_list,
        log_maker=(lambda attempt: (log_dir / f"{tp_index:02d}_body_language{'_r'+str(attempt) if attempt>1 else ''}.txt")) if log_dir else None,
    )
    sys4b = "Produce an agenda list grouped by actor id."
    user4b = _apply_step("agenda_prompt.md", reps4)
    m4b, t4b, k4b = _env_for("AGENDA", default_temp=0.3, default_max_tokens=300)
    agenda_text = _llm_call_with_validation(
        sys4b, user4b, model=m4b, temperature=t4b, max_tokens=k4b, validator=validate_text,
        log_maker=(lambda attempt: (log_dir / f"{tp_index:02d}_agenda{'_r'+str(attempt) if attempt>1 else ''}.txt")) if log_dir else None,
    )
    agenda_by_actor = _parse_agenda_by_actor(agenda_text)
    sys4c = "Produce a reaction for each line, referencing the previous line."
    user4c = _apply_step("reaction_prompt.md", reps4)
    m4c, t4c, k4c = _env_for("REACTIONS", default_temp=0.3, default_max_tokens=400)
    reactions_text = _llm_call_with_validation(
        sys4c, user4c, model=m4c, temperature=t4c, max_tokens=k4c, validator=validate_actor_list,
        log_maker=(lambda attempt: (log_dir / f"{tp_index:02d}_reactions{'_r'+str(attempt) if attempt>1 else ''}.txt")) if log_dir else None,
    )
    # Join and render dialog
    outputs: List[str] = []
    for b_index, (actor_id, line_hint) in enumerate(_parse_actor_lines(actor_lines), start=1):
        agenda_block = agenda_by_actor.get(actor_id, "")
        reaction_line = _pick_bullet_by_index(reactions_text, b_index)
        combined_prompt = (
            f"(Implicit) Line intent: {line_hint}\n"
            f"Reaction: {reaction_line}\n"
            f"Agenda: {agenda_block}\n"
            f"Body language: {body_lang}\n"
        )
        dialog_lines_ctx = state.recent_dialog(actor_id)
        resp = render_character_call(
            actor_id,
            combined_prompt,
            dialog_lines_ctx,
            temperature=0.35,
            max_tokens_line=120,
            log_file=(log_dir / f"{tp_index:02d}_b{b_index:02d}_{actor_id}.txt") if log_dir else None,
        )
        body_for_line = _pick_bullet_by_index(body_lang, b_index)
        outputs.append(_inline_body_with_dialog(body_for_line, resp))
        if resp.strip():
            state.add_dialog_line(actor_id, resp.strip())
    text = "\n".join(outputs)
    return _polish_snippet(text, setting, chapter, chapter_id, version)

def run_subtle_edit_pipeline(tp: TouchPoint, state: ChapterState, *, setting: dict, chapter: dict, chapter_id: str, version: int, tp_index: int, prior_polished: str, prior_suggestions: str, prior_paragraph: str = "", log_dir: Optional[Path] = None) -> str:
    reps = _build_pipeline_replacements(setting, chapter, chapter_id, version, tp, state, prior_paragraph=prior_paragraph)
    reps["[PRIOR_POLISHED]"] = prior_polished
    reps["[SUGGESTIONS]"] = prior_suggestions
    sys1 = "Apply subtle edits to the provided prose respecting suggestions and style."
    user1 = _apply_step("subtle_edit_prompt.md", reps)
    m1, t1, k1 = _env_for("SUBTLE_EDIT", default_temp=0.2, default_max_tokens=1000)
    edited = _llm_call_with_validation(
        sys1, user1, model=m1, temperature=t1, max_tokens=k1, validator=validate_text,
        log_maker=(lambda attempt: (log_dir / f"{tp_index:02d}_subtle_edit{'_r'+str(attempt) if attempt>1 else ''}.txt")) if log_dir else None,
    )
    return _polish_snippet(edited, setting, chapter, chapter_id, version)


# ---------------------------
# Prompt builders and I/O helpers
# ---------------------------

def chapter_id_from_path(chapter_path: str) -> str:
    return Path(chapter_path).stem

def iter_dir_for(chapter_id: str) -> Path:
    return Path("iterations") / chapter_id

def list_versions(chapter_id: str, prefix: str) -> List[int]:
    d = iter_dir_for(chapter_id)
    if not d.exists():
        return []
    versions = []
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

def build_common_replacements(setting: dict, chapter: dict, chapter_id: str, current_version: int) -> Dict[str, str]:
    d = iter_dir_for(chapter_id)
    story_so_far = (d / "story_so_far.txt").read_text(encoding="utf-8") if (d / "story_so_far.txt").exists() else chapter.get("Story-So-Far", "")
    story_relative = (d / "story_relative_to.txt").read_text(encoding="utf-8") if (d / "story_relative_to.txt").exists() else _yaml_section_fallback(chapter, "Story-Relative-To")
    rep = {
        "[SETTING.yaml]": _yaml_dump_py(setting),
        "[CHAPTER_xx.yaml]": _yaml_dump_py(chapter),
        "[story_so_far.txt]": story_so_far,
        "[story_relative_to.txt]": story_relative,
    }
    # Latest artifacts, if any
    for key, pref in (
        ("[draft_v?.txt]", "draft"),
        ("[suggestions_v?.txt]", "suggestions"),
        ("[check_v?.txt]", "check"),
        ("[predraft_v?.txt]", "pre_draft"),
    ):
        rep[key] = read_latest(chapter_id, pref) or ""
    return rep

def _yaml_section_fallback(chapter: dict, key: str) -> str:
    val = chapter.get(key)
    if val is None:
        return ""
    # Represent YAML-ish blocks as YAML
    try:
        return _yaml_dump_py({key: val})
    except Exception:
        return str(val)

def apply_template(template_path: str, replacements: Dict[str, str]) -> str:
    template = read_file(template_path)
    for k, v in replacements.items():
        template = template.replace(k, v)
    return template

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

# ---------------------------
# Character template parsing and substitution
# ---------------------------

CHAR_TEMPLATE_RE = re.compile(
    r"<CHARACTER TEMPLATE>\s*\n\s*<id>(?P<id>[^<]+)</id>\s*\n(?P<body>.*?)\n\s*</CHARACTER TEMPLATE>",
    re.DOTALL | re.IGNORECASE,
)

CHAR_CALL_RE = re.compile(
    r"<CHARACTER>\s*<id>(?P<id>[^<]+)</id>\s*(?:<agenda>(?P<agenda>.*?)</agenda>\s*)?(?:<dialog>(?P<dialogn>\d+)</dialog>\s*)?<prompt>(?P<prompt>.*?)</prompt>\s*</CHARACTER>",
    re.DOTALL | re.IGNORECASE,
)

def parse_character_blocks(pre_draft_text: str) -> Tuple[Dict[str, str], List[Dict[str, str]]]:
    """Extract character templates and call sites from pre-draft text.
    Returns (templates, calls). templates maps id -> body.
    calls is a list of dicts with keys: id, prompt, full_match.
    """
    templates: Dict[str, str] = {}
    for m in CHAR_TEMPLATE_RE.finditer(pre_draft_text):
        templates[m.group("id").strip()] = m.group("body").strip()

    calls: List[Dict[str, str]] = []
    for m in CHAR_CALL_RE.finditer(pre_draft_text):
        calls.append({
            "id": m.group("id").strip(),
            "prompt": m.group("prompt").strip(),
            "agenda": (m.group("agenda") or "").strip() if "agenda" in m.groupdict() else "",
            "dialogn": int(m.group("dialogn")) if m.groupdict().get("dialogn") and m.group("dialogn") else None,
            "full_match": m.group(0),
        })
    return templates, calls

def render_character_call(
    character_id: str,
    call_prompt: str,
    dialog_lines: List[str],
    *,
    temperature: float = 0.3,
    max_tokens_line: int = 100,
    log_file: Optional[Path] = None,
    agenda: str = "",
    character_yaml: Optional[str] = None,
    dialog_n_override: Optional[int] = None,
) -> str:
    """
    CHARACTER TEMPLATE START

    <id>henry</id>
    You are role playing/acting out the following character:
    <character_yaml/>
    You are aware of or deeply care about the following details:
    <agenda/>
    The last N lines of dialog are:
    <dialog>N</dialog>
    The director now expects you to say something that matches your character, and he gives you this prompt:
    <prompt/>

    CHARACTER TEMPLATE END

    Render a character call using a Python-defined CHARACTER TEMPLATE.
    Substitutions:
    - <character_yaml/> → YAML for this character from SETTING.yaml
    - <dialog>N</dialog> → last N non-empty lines from dialog_lines
    - <prompt/> → the 'call_prompt' content
    We avoid extra wrapper meta-prompts and rely on this compact template.
    If log_file is provided, write the prompt (system+user) and the response to that file.
    """

    def _load_character_yaml(cid: str) -> str:
        try:
            setting = load_yaml("SETTING.yaml")
        except Exception:
            return ""
        chars = []
        # Support both 'Characters' top-level list and nested structures
        if isinstance(setting, dict):
            if "Characters" in setting and isinstance(setting["Characters"], list):
                chars = setting["Characters"]
            # Some schemas might nest under 'Setting' or similar; keep it simple for now
        cid_low = str(cid).strip().lower()
        for ch in chars:
            try:
                if str(ch.get("id", "")).strip().lower() == cid_low:
                    # Dump without sorting to preserve author order
                    return _yaml_dump_py(ch)
            except Exception:
                continue
        return ""

    def _last_n_lines(lines: List[str], n: int) -> str:
        if n <= 0:
            return ""
        tail = [ln for ln in lines if ln.strip()]
        chosen = tail[-n:] if tail else lines[-n:]
        return "\n".join(chosen)

    # Default minimal template if file not present (drawn from the docstring intent)
    default_template = (
        "<id/>\n"
        "You are role playing/acting out the following character:\n"
        "<character_yaml/>\n"
        "You are aware of or deeply care about the following details:\n"
        "<agenda/>\n"
        "The last N lines of dialog are:\n"
        "<dialog>N</dialog>\n"
        "The director now expects you to say something that matches your character, and he gives you this prompt:\n"
        "<prompt/>\n"
    )

    # Load external character dialog template if available
    template_path = Path("prompts") / "character_dialog_prompt.md"
    try:
        if template_path.exists():
            user = read_file(str(template_path))
        else:
            user = default_template
    except Exception:
        user = default_template

    # Prepare substitutions

    # Substitute <character_yaml/>
    if "<character_yaml/>" in user:
        cy = character_yaml if character_yaml is not None else _load_character_yaml(character_id)
        user = user.replace("<character_yaml/>", cy)

    # Substitute <id/>
    if "<id/>" in user:
        user = user.replace("<id/>", f"<id>{character_id}</id>")

    # Substitute <agenda/>
    user = user.replace("<agenda/>", agenda or "")

    # Determine number of dialog context lines to include (from call override -> template numeric -> env default)
    # Start with env default
    try:
        env_default_n = int(os.getenv("GW_DIALOG_CONTEXT_LINES", "8"))
    except Exception:
        env_default_n = 8

    chosen_n = dialog_n_override if (isinstance(dialog_n_override, int) and dialog_n_override > 0) else None
    if chosen_n is None:
        # If the template already specifies a numeric <dialog>k</dialog>, use the first one
        m_first = re.search(r"<dialog>\s*(\d+)\s*</dialog>", user)
        if m_first:
            try:
                chosen_n = int(m_first.group(1))
            except Exception:
                chosen_n = None
    if chosen_n is None:
        chosen_n = env_default_n

    # Replace the visible 'N' in the explanatory line, if present
    user = re.sub(r"The last\s+N\s+lines of dialog", f"The last {chosen_n} lines of dialog", user)

    # Substitute all <dialog>N</dialog> or <dialog>number</dialog>
    def _repl_dialog(m: re.Match) -> str:
        token = (m.group(1) or "").strip()
        if token.lower() == "n":
            n = chosen_n
        else:
            try:
                n = int(token)
            except Exception:
                n = 0
        return _last_n_lines(dialog_lines, n)

    user = re.sub(r"<dialog>\s*(\d+|[Nn])\s*</dialog>", _repl_dialog, user)

    # Substitute <prompt/>
    user = user.replace("<prompt/>", call_prompt)

    # Keep a minimal system message to constrain output style
    system = (
        f"You are the character with id '{character_id}'. "
        f"Return only the character's dialog or inner monologue; no notes or brackets."
    )
    # Allow env overrides for dialog output length and model; temperature prioritizes character/template hints
    dialog_max_tokens = _env_int("GW_MAX_TOKENS_DIALOG", max_tokens_line)
    dialog_model = _env_str("GW_MODEL_DIALOG")
    dialog_temp_env = _env_float("GW_TEMP_DIALOG", temperature)
    # Use the passed-in temperature (from template or character hints) if provided; otherwise env
    effective_temp = temperature if temperature is not None else dialog_temp_env
    response = llm_complete(
        user,
        system=system,
        temperature=effective_temp,
        max_tokens=dialog_max_tokens,
        model=dialog_model,
    )
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with log_file.open("w", encoding="utf-8") as f:
                f.write("=== SYSTEM ===\n")
                f.write(system)
                f.write("\n\n=== USER ===\n")
                f.write(user)
                f.write("\n\n=== RESPONSE ===\n")
                f.write(response)
                f.write("\n")
        except Exception:
            # Non-fatal: continue without blocking pipeline
            pass
    return response

def substitute_character_calls(
    pre_draft_text: str,
    *,
    log_dir: Optional[Path] = None,
    context_before_chars: int = 800,
    context_after_chars: int = 400,
) -> Tuple[str, Dict[str, int]]:
    """Progressively replace <CHARACTER> call sites using their templates. Returns (text, stats).
    - Processes one call at a time, rescanning after each replacement so subsequent calls
      see the newly generated dialog in context.
    - For each call, includes surrounding context (before/after slices) in the LLM prompt.
    - If log_dir is provided, each call's prompt+response will be logged into that directory.
    """
    result = pre_draft_text
    # Discover templates fresh each pass (they are optional now; Python can generate defaults)
    templates, _ = parse_character_blocks(result)

    stats = {"templates": len(templates), "calls": 0, "missing_templates": 0}
    call_index = 0

    # Preload character YAML and hints once
    char_yaml_by_id: Dict[str, str] = {}
    char_hints_by_id: Dict[str, Tuple[float, int]] = {}
    try:
        setting = load_yaml("SETTING.yaml")
        if isinstance(setting, dict) and isinstance(setting.get("Characters"), list):
            for ch in setting["Characters"]:
                cid = str(ch.get("id", "")).strip()
                if not cid:
                    continue
                try:
                    char_yaml_by_id[cid.lower()] = _yaml_dump_py(ch)
                    temp = float(ch.get("temperature_hint", 0.3))
                    max_toks = int(ch.get("max_tokens_line", 100))
                    char_hints_by_id[cid.lower()] = (temp, max_toks)
                except Exception:
                    continue
    except Exception:
        pass

    def _char_hints_from_setting(cid: str) -> Tuple[float, int]:
        """Try to read temperature_hint and max_tokens_line from SETTING.yaml for this character."""
        try:
            setting = load_yaml("SETTING.yaml")
        except Exception:
            return 0.3, 100
        chars = []
        if isinstance(setting, dict) and isinstance(setting.get("Characters"), list):
            chars = setting["Characters"]
        cid_low = str(cid).strip().lower()
        for ch in chars:
            try:
                if str(ch.get("id", "")).strip().lower() == cid_low:
                    temp = float(ch.get("temperature_hint", 0.3))
                    max_toks = int(ch.get("max_tokens_line", 100))
                    return temp, max_toks
            except Exception:
                continue
        return 0.3, 100

    while True:
        m = CHAR_CALL_RE.search(result)
        if not m:
            break

        call_index += 1
        stats["calls"] += 1
        cid = m.group("id").strip()
        call_prompt = m.group("prompt").strip()
        full_match = m.group(0)
        start, end = m.start(), m.end()

        tpl = templates.get(cid)
        # Hints: prefer hints from template if present; else from character YAML; else defaults
        if tpl:
            temp = _extract_numeric_hint(tpl, "temperature_hint", default=0.3)
            max_tokens_line = int(_extract_numeric_hint(tpl, "max_tokens_line", default=100))
        else:
            stats["missing_templates"] += 1
            temp, max_tokens_line = char_hints_by_id.get(cid.lower(), (0.3, 100))

        # Context windows
        ctx_before_start = max(0, start - context_before_chars)
        ctx_after_end = min(len(result), end + context_after_chars)
        context_before = result[ctx_before_start:start]
        # Build dialog_lines array from context_before
        dialog_lines = context_before.splitlines()

        log_file = None
        if log_dir is not None:
            safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", cid) or "character"
            log_file = log_dir / f"{call_index:02d}_{safe_id}.txt"

        replacement = render_character_call(
            cid,
            call_prompt,
            dialog_lines,
            temperature=temp,
            max_tokens_line=max_tokens_line,
            log_file=log_file,
            agenda=(m.group("agenda") or "") if "agenda" in m.re.groupindex else "",
            character_yaml=char_yaml_by_id.get(cid.lower()),
            dialog_n_override=(
                int(m.group("dialogn")) if ("dialogn" in m.re.groupindex and m.group("dialogn")) else None
            ),
        )

        # Perform the replacement at the exact span
        result = result[:start] + replacement + result[end:]

    return result, stats

def _extract_numeric_hint(text: str, key: str, default: float) -> float:
    # Look for lines like 'temperature_hint: 0.25' or 'max_tokens_line: 90'
    m = re.search(rf"{re.escape(key)}\s*[:=]\s*([0-9]*\.?[0-9]+)", text)
    if not m:
        return float(default)
    try:
        return float(m.group(1))
    except Exception:
        return float(default)

# ---------------------------
# Core pipeline
# ---------------------------

def generate_pre_draft(chapter_path: str, version_num: int) -> Tuple[str, Path, str]:
    """Generate a pre_draft_vN using master_initial or master prompt."""
    setting = load_yaml("SETTING.yaml")
    chapter = load_yaml(chapter_path)
    chapter_id = chapter_id_from_path(chapter_path)

    # Decide prompt: initial if first version, else master
    latest = get_latest_version(chapter_id)
    use_initial = latest == 0 and version_num == 1
    if use_initial:
        prompt = build_master_initial_prompt(setting, chapter, chapter_id, version_num)
    else:
        prompt = build_master_prompt(setting, chapter, chapter_id, version_num)

    pre_max_tokens = _env_int("GW_MAX_TOKENS_PRE_DRAFT", 800)
    pre_model = _env_str("GW_MODEL_PRE_DRAFT")
    pre_temp = _env_float("GW_TEMP_PRE_DRAFT", 0.2)
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
    setting = load_yaml("SETTING.yaml")
    chapter = load_yaml(chapter_path)
    chapter_id = chapter_id_from_path(chapter_path)

    # Polish only; assumes dialog substitution already applied
    polish_prompt = build_polish_prompt(setting, chapter, chapter_id, version_num, text_to_polish)
    draft_max_tokens = _env_int("GW_MAX_TOKENS_DRAFT", 2000)
    draft_model = _env_str("GW_MODEL_DRAFT")
    draft_temp = _env_float("GW_TEMP_DRAFT", 0.2)
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
    setting = load_yaml("SETTING.yaml")
    chapter = load_yaml(chapter_path)
    chapter_id = chapter_id_from_path(chapter_path)

    check_prompt = build_check_prompt(setting, chapter, chapter_id, version_num, pre_draft_text)
    check_max_tokens = _env_int("GW_MAX_TOKENS_CHECK", 1200)
    check_model = _env_str("GW_MODEL_CHECK")
    check_temp = _env_float("GW_TEMP_CHECK", 0.0)
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
        sugg_max_tokens = _env_int("GW_MAX_TOKENS_SUGGESTIONS", 800)
        # Minimal suggestions generation: reuse the check prompt to ask for actionable items only
        suggestions_prompt = (
            check_prompt
            + "\n\n---\nNow produce a concise, actionable list of suggested changes and fixes only. Do not restate the findings verbatim; output just the suggestions."
        )
        sugg_model = _env_str("GW_MODEL_SUGGESTIONS")
        sugg_temp = _env_float("GW_TEMP_SUGGESTIONS", 0.0)
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

def check_iteration_complete(check_text: str) -> bool:
    """Naive check: returns True if no 'missing' is reported (case-insensitive)."""
    return "missing" not in check_text.lower()

def generate_story_so_far_and_relative(pre_draft_text: str, chapter_path: str, version_num: int) -> None:
    """Optional helpers to evolve story_so_far.txt and story_relative_to.txt for next chapter."""
    setting = load_yaml("SETTING.yaml")
    chapter = load_yaml(chapter_path)
    chapter_id = chapter_id_from_path(chapter_path)

    # Story so far
    ssf_prompt = build_story_so_far_prompt(setting, chapter, chapter_id, version_num, pre_draft_text)
    ssf_max_tokens = _env_int("GW_MAX_TOKENS_STORY_SO_FAR", 1200)
    ssf_model = _env_str("GW_MODEL_STORY_SO_FAR")
    ssf_temp = _env_float("GW_TEMP_STORY_SO_FAR", 0.2)
    ssf = llm_complete(
        ssf_prompt,
        system="You summarize story so far.",
        temperature=ssf_temp,
        max_tokens=ssf_max_tokens,
        model=ssf_model,
    )
    save_text(iter_dir_for(chapter_id) / "story_so_far.txt", ssf)

    # Story relative to
    srt_prompt = build_story_relative_to_prompt(setting, chapter, chapter_id, version_num, pre_draft_text)
    srt_max_tokens = _env_int("GW_MAX_TOKENS_STORY_RELATIVE", 1400)
    srt_model = _env_str("GW_MODEL_STORY_RELATIVE")
    srt_temp = _env_float("GW_TEMP_STORY_RELATIVE", 0.2)
    srt = llm_complete(
        srt_prompt,
        system="You summarize story relative to each character.",
        temperature=srt_temp,
        max_tokens=srt_max_tokens,
        model=srt_model,
    )
    save_text(iter_dir_for(chapter_id) / "story_relative_to.txt", srt)

# ---------------------------
# CLI Entrypoint
# ---------------------------

def _generate_final_txt_from_records(chapter_id: str, records: List[Tuple[str, str, str, str]]) -> Path:
    """Create final.txt that concatenates only polished text results in sequence."""
    final_text = []
    for (_tpid, _tptype, _touch, polished) in records:
        if polished is None:
            continue
        final_text.append(polished.strip())
    out_path = iter_dir_for(chapter_id) / "final.txt"
    save_text(out_path, "\n\n".join([t for t in final_text if t]))
    return out_path

def run_pipelines_for_chapter(chapter_path: str, version_num: int, *, log_llm: bool = False) -> None:
    """Execute deterministic pipelines per touch-point and write artifacts for vN."""
    setting = load_yaml("SETTING.yaml")
    chapter = load_yaml(chapter_path)
    chapter_id = chapter_id_from_path(chapter_path)

    # Decide branch based on prior drafts
    latest = get_latest_version(chapter_id)
    branch_b = latest > 0  # if any prior draft/suggestions exist, treat as edit branch
    prior_draft = read_latest(chapter_id, "draft") or ""
    prior_suggestions = read_latest(chapter_id, "suggestions") or ""

    # Parse touch-points and init state
    tps = parse_touchpoints_from_chapter(chapter)
    if not tps:
        print("No Touch-Points found; nothing to do.")
        return
    try:
        dialog_ctx_n = int(os.getenv("GW_DIALOG_CONTEXT_LINES", "8"))
    except Exception:
        dialog_ctx_n = 8
    state = ChapterState(dialog_context_lines=dialog_ctx_n)

    # Logging
    base_log_dir = None
    if log_llm:
        base_log_dir = iter_dir_for(chapter_id) / f"pipeline_v{version_num}"

    # Iterate touch-points and run pipelines
    total = len(tps)
    records: List[Tuple[str, str, str, str]] = []
    prior_paragraph = ""
    for i, tp in enumerate(tps, start=1):
        tp_type = tp.get("type", "")
        tp_id = tp.get("id", str(i))
        raw = tp.get("raw") or tp.get("content") or ""
        print(f"[v{version_num}] Touch-point {i}/{total} – {tp_type}: {tp.get('content','')[:60]}")
        tp_log_dir = (base_log_dir / f"{i:02d}_{tp_type}") if base_log_dir else None

        polished_text = ""
        if tp_type == "actors":
            state.set_actors(tp.get("content", ""))
        elif tp_type == "scene":
            state.set_scene(tp.get("content", ""))
        elif tp_type == "foreshadowing":
            state.add_foreshadowing(tp.get("content", ""))
        elif tp_type == "setting":
            # Parse nested mapping: expect something like { setting: { factoids: [...], actors: [...] } }
            selected_factoids = []
            selected_characters = []
            setting_map = {}
            # Try raw block first (most reliable)
            raw_text = tp.get("raw", "")
            try:
                parsed = _yaml_load_py(raw_text) if raw_text else None
            except Exception:
                parsed = None
            if isinstance(parsed, dict) and "setting" in parsed and isinstance(parsed["setting"], dict):
                setting_map = parsed["setting"]
            else:
                # Fallback: try to parse content as YAML
                content_text = tp.get("content", "")
                try:
                    parsed2 = _yaml_load_py(content_text) if content_text else None
                except Exception:
                    parsed2 = None
                if isinstance(parsed2, dict):
                    setting_map = parsed2

            factoid_names = []
            actor_names = []
            if isinstance(setting_map, dict):
                # Accept 'factoids' and 'actors' keys (case-insensitive)
                for key in list(setting_map.keys()):
                    if str(key).strip().lower() == "factoids":
                        try:
                            factoid_names = [str(x).strip() for x in (setting_map[key] or [])]
                        except Exception:
                            factoid_names = []
                    if str(key).strip().lower() == "actors":
                        try:
                            actor_names = [str(x).strip() for x in (setting_map[key] or [])]
                        except Exception:
                            actor_names = []

            # Select from SETTING.yaml
            try:
                factoids = setting.get("Factoids") if isinstance(setting, dict) else None
                if isinstance(factoids, list) and factoid_names:
                    wanted = {n.lower() for n in factoid_names if n}
                    for f in factoids:
                        name = str(f.get("name", "")).strip()
                        if name and name.lower() in wanted:
                            selected_factoids.append(f)
                characters = setting.get("Characters") if isinstance(setting, dict) else None
                if isinstance(characters, list) and actor_names:
                    wantedc = {n.lower() for n in actor_names if n}
                    for ch in characters:
                        cid = str(ch.get("id", "")).strip().lower()
                        cname = str(ch.get("name", "")).strip().lower()
                        if cid in wantedc or cname in wantedc:
                            selected_characters.append(ch)
            except Exception:
                pass

            # Initialize active actors from setting actors (if provided)
            if actor_names:
                try:
                    state.set_actors(", ".join(actor_names))
                except Exception:
                    pass

            # Store YAML blocks for templates
            try:
                state.setting_block = _yaml_dump_py({"Selected-Factoids": selected_factoids}) if selected_factoids else ""
            except Exception:
                state.setting_block = ""
            try:
                state.characters_block = _yaml_dump_py({"Selected-Characters": selected_characters}) if selected_characters else ""
            except Exception:
                state.characters_block = ""
        elif tp_type in ("narration", "explicit", "implicit"):
            if tp_type == "narration":
                if branch_b:
                    polished_text = run_subtle_edit_pipeline(tp, state, setting=setting, chapter=chapter, chapter_id=chapter_id, version=version_num, tp_index=i, prior_polished=prior_draft, prior_suggestions=prior_suggestions, prior_paragraph=prior_paragraph, log_dir=tp_log_dir)
                else:
                    polished_text = run_narration_pipeline(tp, state, setting=setting, chapter=chapter, chapter_id=chapter_id, version=version_num, tp_index=i, prior_paragraph=prior_paragraph, log_dir=tp_log_dir)
            elif tp_type == "explicit":
                if branch_b:
                    polished_text = run_subtle_edit_pipeline(tp, state, setting=setting, chapter=chapter, chapter_id=chapter_id, version=version_num, tp_index=i, prior_polished=prior_draft, prior_suggestions=prior_suggestions, prior_paragraph=prior_paragraph, log_dir=tp_log_dir)
                else:
                    polished_text = run_explicit_pipeline(tp, state, setting=setting, chapter=chapter, chapter_id=chapter_id, version=version_num, tp_index=i, prior_paragraph=prior_paragraph, log_dir=tp_log_dir)
            else:  # implicit
                if branch_b:
                    polished_text = run_subtle_edit_pipeline(tp, state, setting=setting, chapter=chapter, chapter_id=chapter_id, version=version_num, tp_index=i, prior_polished=prior_draft, prior_suggestions=prior_suggestions, prior_paragraph=prior_paragraph, log_dir=tp_log_dir)
                else:
                    polished_text = run_implicit_pipeline(tp, state, setting=setting, chapter=chapter, chapter_id=chapter_id, version=version_num, tp_index=i, prior_paragraph=prior_paragraph, log_dir=tp_log_dir)
        else:
            # Unknown types treated as narration by default
            if branch_b:
                polished_text = run_subtle_edit_pipeline(tp, state, setting=setting, chapter=chapter, chapter_id=chapter_id, version=version_num, prior_polished=prior_draft, prior_suggestions=prior_suggestions, log_dir=tp_log_dir)
            else:
                polished_text = run_narration_pipeline(tp, state, setting=setting, chapter=chapter, chapter_id=chapter_id, version=version_num, log_dir=tp_log_dir)

        # Record (for actors/scene/foreshadowing, polished_text may be empty)
        records.append((tp_id, tp_type, raw, polished_text))
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
        generate_story_so_far_and_relative(full_text, chapter_path, version_num)
        print("Regenerated story_so_far.txt and story_relative_to.txt")

def main():
    load_env()
    if len(sys.argv) < 2:
        print("Usage: python scripts/driver.py chapters/CHAPTER_xx.yaml [vN or auto] [--show-dialog] [--log-llm]")
        sys.exit(1)

    args = [a for a in sys.argv[1:]]
    log_llm = False
    # Backcompat flag
    if "--show-dialog" in args:
        log_llm = True
        args.remove("--show-dialog")
    if "--log-llm" in args:
        log_llm = True
        args.remove("--log-llm")

    if not args:
        print("Usage: python scripts/driver.py chapters/CHAPTER_xx.yaml [vN or auto] [--show-dialog] [--log-llm]")
        sys.exit(1)

    chapter_path = args[0]
    chapter_id = chapter_id_from_path(chapter_path)

    # Determine version number
    version_num: Optional[int] = None
    if len(args) > 1 and args[1].startswith("v"):
        try:
            version_num = int(args[1][1:])
        except ValueError:
            print("Invalid version format. Use v1, v2, ...")
            sys.exit(1)
    elif len(args) > 1 and args[1] == "auto":
        version_num = get_latest_version(chapter_id) + 1
    else:
        version_num = get_latest_version(chapter_id) + 1

    # Validate required inputs early and ensure iteration directory exists
    try:
        _validate_inputs_and_prepare(chapter_path)
    except GWError as e:
        print(f"Error: {e}")
        sys.exit(2)

    # Execute the deterministic pipeline once for vN
    run_pipelines_for_chapter(chapter_path, version_num, log_llm=log_llm)

# ---------------------------
# Validation & Setup (Task 8)
# ---------------------------

# Required prompt templates for the pipeline
_REQUIRED_PROMPTS = [
    # Core polishing and summaries
    "prompts/polish_prose_prompt.md",
    "prompts/story_so_far_prompt.md",
    "prompts/story_relative_to_prompt.md",
    # Deterministic pipeline steps
    "prompts/narration_brain_storm_prompt.md",
    "prompts/explicit_brain_storm_prompt.md",
    "prompts/ordering_prompt.md",
    "prompts/implicit_brain_storm_prompt.md",
    "prompts/generate_narration_prompt.md",
    "prompts/actor_assignment_prompt.md",
    "prompts/body_language_prompt.md",
    "prompts/agenda_prompt.md",
    "prompts/reaction_prompt.md",
    "prompts/subtle_edit_prompt.md",
]


def _validate_inputs_and_prepare(chapter_path: str) -> None:
    """Validate presence of required files and create iteration directory.

    Checks:
    - SETTING.yaml exists and parses as YAML
    - Chapter file exists and parses as YAML
    - Required prompt templates exist
    - Creates iterations/<CHAPTER_ID>/ directory
    """
    # Validate chapter file early for clearer errors before any work
    ch_path = Path(chapter_path)
    if not ch_path.exists():
        raise MissingFileError(
            f"Chapter file not found: {chapter_path}. Expected a path like 'chapters/CHAPTER_001.yaml'."
        )
    # Load to validate YAML
    _ = load_yaml(str(ch_path))

    # Validate SETTING.yaml
    if not Path("SETTING.yaml").exists():
        raise MissingFileError("Missing required SETTING.yaml at project root.")
    _ = load_yaml("SETTING.yaml")

    # Validate required prompt templates
    missing = [p for p in _REQUIRED_PROMPTS if not Path(p).exists()]
    if missing:
        joined = "\n  - " + "\n  - ".join(missing)
        raise MissingFileError(
            "Missing required prompt template(s):" + joined +
            "\nPlease add these files under the 'prompts/' directory. See README.md for details."
        )

    # Character dialog prompt is optional (we fallback to a built-in template)

    # Ensure iterations directory exists for this chapter
    chapter_id = chapter_id_from_path(chapter_path)
    out_dir = iter_dir_for(chapter_id)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise GWError(f"Unable to create iterations directory '{out_dir}': {e}")

if __name__ == "__main__":
    main()