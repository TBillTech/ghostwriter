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
    allowed = {"actors", "scene", "foreshadowing", "narration", "explicit", "implicit"}

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
    lines = [ln.rstrip() for ln in output.splitlines() if ln.strip()]
    bullets = [ln for ln in lines if ln.lstrip().startswith("*")]
    if len(bullets) >= 2:
        return True, "ok"
    return False, "Expected a bullet list with at least 2 lines starting with '*'"

_ACTOR_LINE_RE = re.compile(r"^\s*([A-Za-z0-9_.\-]+)\s*:\s+.+")

def validate_actor_list(output: str) -> Tuple[bool, str]:
    lines = [ln for ln in output.splitlines() if ln.strip()]
    count = 0
    for ln in lines:
        if _ACTOR_LINE_RE.match(ln):
            count += 1
    if count >= 2:
        return True, "ok"
    return False, "Expected an actor list with at least 2 actor-attributed lines like 'id: ...'"

def validate_agenda_list(output: str) -> Tuple[bool, str]:
    # Heuristic: at least 2 structured lines (bullets or 'id: goal' style)
    lines = [ln for ln in output.splitlines() if ln.strip()]
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

def main():
    load_env()
    if len(sys.argv) < 2:
        print("Usage: python scripts/driver.py chapters/CHAPTER_xx.yaml [vN or auto] [--show-dialog]")
        sys.exit(1)

    # Simple flag parsing for --show-dialog
    args = [a for a in sys.argv[1:]]
    show_dialog = False
    if "--show-dialog" in args:
        show_dialog = True
        args.remove("--show-dialog")

    if not args:
        print("Usage: python scripts/driver.py chapters/CHAPTER_xx.yaml [vN or auto] [--show-dialog]")
        sys.exit(1)

    chapter_path = args[0]
    chapter_id = chapter_id_from_path(chapter_path)

    # Determine version number
    if len(args) > 1 and args[1].startswith("v"):
        try:
            version_num = int(args[1][1:])
        except ValueError:
            print("Invalid version format. Use v1, v2, ...")
            sys.exit(1)
    else:
        version_num = get_latest_version(chapter_id) + 1

    # Validate required inputs early and ensure iteration directory exists
    try:
        _validate_inputs_and_prepare(chapter_path)
    except GWError as e:
        # Print friendly error and exit
        print(f"Error: {e}")
        sys.exit(2)

    max_cycles = int(os.getenv("GW_MAX_ITERATIONS", "2"))

    skip_summaries = os.getenv("GW_SKIP_SUMMARIES", "0") == "1"

    for cycle in range(max_cycles):
        print(f"\n=== Iteration v{version_num} (cycle {cycle+1}/{max_cycles}) ===")
        pre_draft_text, pre_path, _ = generate_pre_draft(chapter_path, version_num)
        # Perform dialog substitution BEFORE verification
        log_dir = iter_dir_for(chapter_id) / f"dialog_prompts_v{version_num}" if show_dialog else None
        substituted_text, stats = substitute_character_calls(pre_draft_text, log_dir=log_dir)
        print(f"Character substitution: {json.dumps(stats)}")

        # Verify the substituted (rough) draft
        check_text, check_path = verify_predraft(substituted_text, chapter_path, version_num)
        if check_iteration_complete(check_text):
            print("All touch-points satisfied (no 'missing' detected). Proceeding to polish.")
            polished_text, draft_path = polish_prose(substituted_text, chapter_path, version_num)
            # Optionally evolve summaries for next chapter
            if not skip_summaries:
                generate_story_so_far_and_relative(polished_text, chapter_path, version_num)
            print("Done.")
            break
        else:
            print("Missing touch-points detected. Incrementing version and retrying master prompt.")
            version_num += 1
            # Continue loop to re-draft
    else:
        # If loop exhausted without a clean pass, still attempt polish of last pre_draft
        print("Max iteration cycles reached; polishing latest pre_draft anyway.")
        # Ensure we at least apply substitution once before polishing
        log_dir = iter_dir_for(chapter_id) / f"dialog_prompts_v{version_num}" if show_dialog else None
        substituted_text, stats = substitute_character_calls(pre_draft_text, log_dir=log_dir)
        print(f"Character substitution: {json.dumps(stats)}")
        polished_text, draft_path = polish_prose(substituted_text, chapter_path, version_num)
        if not skip_summaries:
            generate_story_so_far_and_relative(polished_text, chapter_path, version_num)

# ---------------------------
# Validation & Setup (Task 8)
# ---------------------------

# Required prompt templates for the pipeline
_REQUIRED_PROMPTS = [
    "prompts/master_initial_prompt.md",
    "prompts/master_prompt.md",
    "prompts/polish_prose_prompt.md",
    "prompts/check_prompt.md",
    "prompts/story_so_far_prompt.md",
    "prompts/story_relative_to_prompt.md",
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