import re
import sys
import time
import yaml
import json
import random
import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

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
# Helpers
# ---------------------------

def _yaml_load_py(content: str):
    """Pure-Python YAML load using SafeLoader to avoid C-accelerated libyaml crashes."""
    return yaml.load(content, Loader=_PySafeLoader)

def _yaml_dump_py(obj) -> str:
    """Pure-Python YAML dump using SafeDumper to avoid C-accelerated libyaml."""
    return yaml.dump(obj, sort_keys=False, Dumper=_PySafeDumper)

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return _yaml_load_py(content)

def save_text(path, content):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def read_file(path):
    return Path(path).read_text(encoding="utf-8")

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

def llm_complete(prompt: str, *, system: Optional[str] = None, temperature: float = 0.2, max_tokens: int = 800) -> str:
    """Call OpenAI if configured; else return a deterministic mock response."""
    client = get_client()
    if client is None:
        # Mock response for offline/dev usage
        head = (prompt[:220] + "...") if len(prompt) > 220 else prompt
        return f"[MOCK LLM RESPONSE]\nSystem: {system or 'n/a'}\nTemp: {temperature}\n---\n{head}"

    def _do_call():
        # Prefer chat.completions for richer prompting
        resp = client.chat.completions.create(
            model=get_model(),
            messages=[
                {"role": "system", "content": system or "You are a helpful writing assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    return with_backoff(_do_call)

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
    response = llm_complete(user, system=system, temperature=temperature, max_tokens=max_tokens_line)
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

    pre_draft = llm_complete(prompt, system="You are a Director creating pre-prose with character templates and call sites.")

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
    polished = llm_complete(polish_prompt, system="You are a ghostwriter polishing and cleaning prose.", temperature=0.2, max_tokens=2000)

    out_path = iter_dir_for(chapter_id) / f"draft_v{version_num}.txt"
    save_text(out_path, polished)
    print(f"draft saved to {out_path}")
    return polished, out_path

def verify_predraft(pre_draft_text: str, chapter_path: str, version_num: int) -> Tuple[str, Path]:
    setting = load_yaml("SETTING.yaml")
    chapter = load_yaml(chapter_path)
    chapter_id = chapter_id_from_path(chapter_path)

    check_prompt = build_check_prompt(setting, chapter, chapter_id, version_num, pre_draft_text)
    check_results = llm_complete(check_prompt, system="You are an evaluator checking touch-point coverage.", temperature=0.0, max_tokens=1200)

    # Save check results
    check_path = iter_dir_for(chapter_id) / f"check_v{version_num}.txt"
    save_text(check_path, check_results)
    print(f"check saved to {check_path}")

    # Save suggestions artifact as well (basic: mirror check output)
    suggestions_path = iter_dir_for(chapter_id) / f"suggestions_v{version_num}.txt"
    save_text(suggestions_path, check_results)
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
    ssf = llm_complete(ssf_prompt, system="You summarize story so far.", temperature=0.2, max_tokens=1200)
    save_text(iter_dir_for(chapter_id) / "story_so_far.txt", ssf)

    # Story relative to
    srt_prompt = build_story_relative_to_prompt(setting, chapter, chapter_id, version_num, pre_draft_text)
    srt = llm_complete(srt_prompt, system="You summarize story relative to each character.", temperature=0.2, max_tokens=1400)
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

if __name__ == "__main__":
    main()