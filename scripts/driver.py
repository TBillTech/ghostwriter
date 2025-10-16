import re
import sys
import time
import json
import random
import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable, Any

from dotenv import load_dotenv
import os
import threading
import signal
import faulthandler
try:
    import resource  # POSIX resource usage
except Exception:  # pragma: no cover
    resource = None  # type: ignore
# Force PyYAML to pure-Python mode by blocking the C extension (_yaml)
try:
    # Prevent yaml from importing the C accelerator to avoid rare native crashes
    sys.modules.setdefault("_yaml", None)
except Exception:
    pass
import yaml
from yaml.loader import SafeLoader as _PySafeLoader
from yaml.dumper import SafeDumper as _PySafeDumper

try:
    # OpenAI v2 client (requirements pinned to openai==2.0.0)
    from openai import OpenAI
    try:
        # Optional Azure client if available in this version
        from openai import AzureOpenAI  # type: ignore
    except Exception:  # pragma: no cover
        AzureOpenAI = None  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore
    AzureOpenAI = None  # type: ignore

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
    try:
        _breadcrumb("yaml:load:start:py")
        return yaml.load(content, Loader=_PySafeLoader)
    finally:
        _breadcrumb("yaml:load:end:py")

def _yaml_dump_py(obj) -> str:
    """Deprecated: avoid emitting YAML entirely to prevent libyaml/native crashes.
    Use JSON text as a stable, safe representation.
    """
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2, default=str)
    except Exception:
        return str(obj)
def _to_text(obj) -> str:
    """Safe, deterministic textual serialization for prompts and logs (JSON, unicode)."""
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2, default=str)
    except Exception:
        return str(obj)

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
        # Catch non-YAML exceptions (e.g., IndexError) and rethrow consistently without crashing native layer
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
    # Prefer .env values to ensure local config wins over shell/session leftovers
    load_dotenv(override=True)

# ---- Crash trace helpers ----
def _crash_trace_file() -> Optional[str]:
    return os.getenv("GW_CRASH_TRACE_FILE")

def _rss_kb() -> Optional[int]:
    try:
        if resource is None:
            return None
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return int(getattr(usage, "ru_maxrss", 0))
    except Exception:
        return None

def _breadcrumb(label: str) -> None:
    path = _crash_trace_file()
    if not path:
        return
    try:
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        tid = threading.get_ident()
        rss = _rss_kb()
        line = f"{ts} pid={os.getpid()} tid={tid} rss_kb={rss or 'na'} | {label}\n"
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
        try:
            sys.stderr.write(f"[crumb] {label}\n")
            sys.stderr.flush()
        except Exception:
            pass
    except Exception:
        pass

# ---- Warning helpers and token normalization ----
def _norm_token(s: Any) -> str:
    try:
        x = str(s).strip()
        # Strip matching leading/trailing single or double quotes
        if (x.startswith('"') and x.endswith('"')) or (x.startswith("'") and x.endswith("'")):
            x = x[1:-1].strip()
        return x.lower()
    except Exception:
        return str(s).lower() if s is not None else ""

def _log_warning(msg: str, log_dir: Optional[Path]) -> None:
    try:
        print(f"WARNING: {msg}")
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            with (log_dir / "warnings.txt").open("a", encoding="utf-8") as f:
                f.write(msg + "\n")
    except Exception:
        pass

_CLIENT = None  # type: ignore
_CLIENT_INFO = ""  # for diagnostics (base_url or azure endpoint)

def get_client():
    """Return OpenAI client if API key is present; otherwise None for mock mode.
    Supports native OpenAI and Azure OpenAI. Does not attempt OpenRouter.
    """
    global _CLIENT, _CLIENT_INFO
    if _CLIENT is not None:
        return _CLIENT
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return None
    try:
        # Azure OpenAI support
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("AZURE_OPENAI_API_BASE")
        if azure_endpoint and AzureOpenAI is not None:
            api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
            ak = os.getenv("AZURE_OPENAI_API_KEY") or api_key
            client = AzureOpenAI(azure_endpoint=azure_endpoint, api_version=api_version, api_key=ak)
            _CLIENT = client
            _CLIENT_INFO = f"azure:{azure_endpoint}|v={api_version}"
            _breadcrumb(f"openai:client-initialized azure_endpoint={azure_endpoint} api_version={api_version}")
            return _CLIENT

        # Normalize base URL if provided; many providers require '/v1' suffix
        base_url = (
            os.getenv("OPENAI_BASE_URL")
            or os.getenv("OPENAI_API_BASE")
        )
        client = None
        if base_url:
            bu = base_url.strip()
            # Heuristic: if not azure-like and missing '/v1' suffix, append it
            lower = bu.lower()
            is_azure = ("azure.com" in lower) or ("openai.azure" in lower)
            if (not is_azure) and not re.search(r"/v\d+/?$", bu):
                bu = bu.rstrip("/") + "/v1"
            client = OpenAI(base_url=bu, api_key=api_key)
            _CLIENT = client
            _CLIENT_INFO = f"base_url:{bu}"
            _breadcrumb(f"openai:client-initialized base_url={bu}")
        else:
            client = OpenAI(api_key=api_key)
            _CLIENT = client
            _CLIENT_INFO = "default"
            _breadcrumb("openai:client-initialized base_url=default")
        return _CLIENT
    except Exception:
        return None

def get_model() -> str:
    # Prefer GW_MODEL_DEFAULT; fall back to legacy OPENAI_MODEL for compatibility
    return os.getenv("GW_MODEL_DEFAULT") or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

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
    _breadcrumb("llm:enter")
    client = get_client()
    if client is None:
        # Mock response for offline/dev usage
        head = (prompt[:220] + "...") if len(prompt) > 220 else prompt
        out = f"[MOCK LLM RESPONSE]\nSystem: {system or 'n/a'}\nTemp: {temperature}\n---\n{head}"
        _breadcrumb("llm:mock-return")
        return out

    model_name = model or get_model()

    def _estimate_prompt_tokens(texts: List[str]) -> int:
        """Roughly estimate tokens from text length; configurable chars-per-token.
        Uses GW_CHARS_PER_TOKEN (default 4). Adds optional GW_PROMPT_TOKENS_PAD tokens.
        """
        try:
            cpt = float(os.getenv("GW_CHARS_PER_TOKEN", "4").strip() or "4")
            if cpt <= 0:
                cpt = 4.0
        except Exception:
            cpt = 4.0
        total_chars = 0
        for t in texts:
            try:
                total_chars += len(t or "")
            except Exception:
                continue
        import math
        est = int(math.ceil(total_chars / cpt))
        try:
            pad = int(os.getenv("GW_PROMPT_TOKENS_PAD", "0") or "0")
            if pad > 0:
                est += pad
        except Exception:
            pass
        return max(est, 0)

    def _do_call():
        # Prefer chat.completions for richer prompting
        _breadcrumb(f"llm:chat.create:before model={model_name} endpoint={_CLIENT_INFO}")
        # Allow env to force token param key (max_tokens vs max_completion_tokens)
        token_param = os.getenv("GW_TOKENS_PARAM", "max_tokens").strip() or "max_tokens"
        _breadcrumb(f"llm:tokens-param={token_param}")
        messages = [
            {"role": "system", "content": system or "You are a helpful writing assistant."},
            {"role": "user", "content": prompt},
        ]
        kwargs = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
        }
        # Optionally include prompt tokens estimate in the limit if provider expects total tokens
        try:
            include_prompt = os.getenv("GW_INCLUDE_PROMPT_TOKENS", "0").strip() == "1"
        except Exception:
            include_prompt = False
        effective_limit = int(max_tokens)
        if include_prompt:
            prompt_est = _estimate_prompt_tokens([
                str(messages[0]["content"] or ""),
                str(messages[1]["content"] or ""),
            ])
            # Avoid overflow if caller already specified a total-like value
            # We cap at a reasonable upper bound if user sets very high defaults
            try:
                cap = int(os.getenv("GW_MAX_TOKENS_CAP", "64000") or "64000")
            except Exception:
                cap = 64000
            effective_limit = min(max_tokens + prompt_est, cap)
            _breadcrumb(f"llm:include-prompt-tokens est={prompt_est} eff_limit={effective_limit}")
        kwargs[token_param] = effective_limit
        try:
            resp = client.chat.completions.create(**kwargs)
            _breadcrumb("llm:chat.create:after")
            out = resp.choices[0].message.content or ""
            _breadcrumb("llm:exit")
            return out
        except Exception as e:
            # Optional fallback on 404-like HTML responses from gateways
            msg = str(e)
            # Retry with alternate token parameter if provider rejects 'max_tokens'
            if "Unsupported parameter" in msg and "max_tokens" in msg:
                try:
                    _breadcrumb("llm:param-fallback:max_completion_tokens")
                    kwargs.pop("max_tokens", None)
                    kwargs["max_completion_tokens"] = max_tokens
                    resp = client.chat.completions.create(**kwargs)
                    _breadcrumb("llm:chat.create:after")
                    out = resp.choices[0].message.content or ""
                    _breadcrumb("llm:exit")
                    return out
                except Exception:
                    pass
            # Retry if provider rejects temperature values (some only allow default of 1)
            if ("Unsupported value" in msg or "unsupported_value" in msg) and "temperature" in msg:
                try:
                    _breadcrumb("llm:param-fallback:temperature-default")
                    # Some providers require omitting temperature or forcing 1
                    temp_prev = kwargs.pop("temperature", None)
                    try:
                        resp = client.chat.completions.create(**kwargs)
                    except Exception:
                        # Force to 1 and retry
                        kwargs["temperature"] = 1
                        resp = client.chat.completions.create(**kwargs)
                    _breadcrumb("llm:chat.create:after")
                    out = resp.choices[0].message.content or ""
                    _breadcrumb("llm:exit")
                    return out
                except Exception:
                    # Restore original temperature for any outer retries
                    if "temperature" not in kwargs and temp_prev is not None:
                        kwargs["temperature"] = temp_prev
                    pass
            if os.getenv("GW_LLM_FALLBACK_ON_404", "0") == "1" and ("404" in msg or "NotFound" in msg):
                _breadcrumb("llm:404-fallback:mock")
                head = (prompt[:220] + "...") if len(prompt) > 220 else prompt
                return f"[MOCK LLM RESPONSE AFTER 404]\nSystem: {system or 'n/a'}\nModel: {model_name}\n---\n{head}"
            raise

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

def _resolve_temp(step_key: str, default_temp: float) -> float:
    """Resolve temperature with precedence: GW_TEMP_{STEP} -> GW_TEMP_DEFAULT -> default_temp."""
    # Step-specific
    val = os.getenv(f"GW_TEMP_{step_key}")
    if val is not None and str(val).strip() != "":
        try:
            return float(val)
        except Exception:
            pass
    # Global default
    g = os.getenv("GW_TEMP_DEFAULT")
    if g is not None and str(g).strip() != "":
        try:
            return float(g)
        except Exception:
            pass
    return float(default_temp)

def _resolve_max_tokens(step_key: str, default_max_tokens: int) -> int:
    """Resolve max tokens with precedence: GW_MAX_TOKENS_{STEP} -> GW_MAX_TOKENS_DEFAULT -> default."""
    # Step-specific
    val = os.getenv(f"GW_MAX_TOKENS_{step_key}")
    if val is not None and str(val).strip() != "":
        try:
            v = int(val)
            return v if v > 0 else default_max_tokens
        except Exception:
            pass
    # Global default
    g = os.getenv("GW_MAX_TOKENS_DEFAULT")
    if g is not None and str(g).strip() != "":
        try:
            v = int(g)
            return v if v > 0 else default_max_tokens
        except Exception:
            pass
    return int(default_max_tokens)

# ---------------------------
# Environment snapshot helpers
# ---------------------------

def _mask_env_value(k: str, v: Optional[str]) -> str:
    """Mask secrets in environment values while retaining minimal suffix for debugging.
    Masks keys containing key/secret/token/password regardless of prefix.
    """
    try:
        if v is None:
            return ""
        kl = (k or "").lower()
        if any(s in kl for s in ("key", "secret", "token", "password")):
            s = str(v)
            if len(s) <= 8:
                return "***"
            return ("*" * (len(s) - 4)) + s[-4:]
        return str(v)
    except Exception:
        return ""

def _normalize_base_url_from_env() -> Dict[str, str]:
    """Infer the effective base URL or Azure endpoint from environment without creating a client."""
    info: Dict[str, str] = {}
    try:
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("AZURE_OPENAI_API_BASE")
        if azure_endpoint:
            info["azure_endpoint"] = azure_endpoint.strip()
            info["azure_api_version"] = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
            return info
        base_url = (
            os.getenv("OPENAI_BASE_URL")
            or os.getenv("OPENAI_API_BASE")
        )
        if base_url:
            bu = base_url.strip()
            lower = bu.lower()
            is_azure = ("azure.com" in lower) or ("openai.azure" in lower)
            if (not is_azure) and not re.search(r"/v\d+/?$", bu):
                bu = bu.rstrip("/") + "/v1"
            info["base_url"] = bu
        return info
    except Exception:
        return info

def _collect_program_env_snapshot() -> Dict[str, Any]:
    """Collect program-relevant environment settings for recording in parse_complete.txt.
    Includes GW_*, OPENAI_*, AZURE_OPENAI_* variables with secret masking,
    plus a few derived fields to aid diagnostics.
    """
    prefixes = ("GW_", "OPENAI_", "AZURE_OPENAI_")
    env_items: List[Tuple[str, str]] = []
    try:
        for k, v in os.environ.items():
            if any(k.startswith(p) for p in prefixes):
                env_items.append((k, _mask_env_value(k, v)))
    except Exception:
        pass
    env_items.sort(key=lambda kv: kv[0])
    # Derived fields
    derived: Dict[str, Any] = {}
    try:
        derived.update(_normalize_base_url_from_env())
        derived["model_default"] = get_model()
        derived["tokens_param_selected"] = (os.getenv("GW_TOKENS_PARAM", "max_tokens").strip() or "max_tokens")
        derived["include_prompt_tokens"] = (os.getenv("GW_INCLUDE_PROMPT_TOKENS", "0") == "1")
        derived["chars_per_token"] = float(os.getenv("GW_CHARS_PER_TOKEN", "4"))
        derived["crash_trace_file"] = _crash_trace_file() or ""
        # Whether .env override was intended is static in code; include a flag
        derived["dotenv_override_enabled"] = True
    except Exception:
        pass
    return {
        "env": {k: v for k, v in env_items},
        "derived": derived,
    }

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
    # New: STATE_CHARACTERS — subset of CHARACTERS limited strictly to state.active_actors
    state_chars_block = ""
    try:
        sel_state_chars = []
        # If characters_block was populated by a 'setting' touch-point, parse and filter from it first
        parsed_chars_block = None
        if getattr(state, "characters_block", ""):
            try:
                parsed = json.loads(state.characters_block)
                if isinstance(parsed, dict):
                    parsed_chars_block = parsed.get("Selected-Characters")
            except Exception:
                parsed_chars_block = None
        candidates = None
        if isinstance(parsed_chars_block, list):
            candidates = parsed_chars_block
        else:
            candidates = setting.get("Characters") if isinstance(setting, dict) else None
        if isinstance(candidates, list) and state.active_actors:
            wanted2 = {a.strip().lower() for a in state.active_actors if a.strip()}
            for ch in candidates:
                cid = str(ch.get("id", "")).strip().lower()
                cname = str(ch.get("name", "")).strip().lower()
                if cid in wanted2 or cname in wanted2:
                    sel_state_chars.append(ch)
        if sel_state_chars:
            state_chars_block = _to_text({"Selected-Characters": sel_state_chars})
    except Exception:
        state_chars_block = ""
    extra["[STATE_CHARACTERS]"] = state_chars_block
    extra["[state_characters]"] = state_chars_block
    # Provide dialog history as YAML-ish for all known actors
    try:
        dialog_map = {a: state.recent_dialog(a) for a in (state.active_actors or [])}
        extra["[DIALOG_HISTORY]"] = _to_text(dialog_map)
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
    """Resolve per-prompt env: returns (model, temperature, max_tokens) with global fallbacks.
    Precedence:
      - model: GW_MODEL_{STEP}
      - temperature: GW_TEMP_{STEP} -> GW_TEMP_DEFAULT -> default_temp
      - max_tokens: GW_MAX_TOKENS_{STEP} -> GW_MAX_TOKENS_DEFAULT -> default_max_tokens
    Step keys examples: PRE_DRAFT, DRAFT, CHECK, SUGGESTIONS, STORY_SO_FAR, STORY_RELATIVE,
    BRAIN_STORM, ORDERING, GENERATE_NARRATION, ACTOR_ASSIGNMENT, BODY_LANGUAGE, AGENDA, REACTIONS,
    CHARACTER_DIALOG, SUBTLE_EDIT, POLISH_PROSE.
    """
    model = _env_str(f"GW_MODEL_{step_key}")
    temp = _resolve_temp(step_key, default_temp)
    max_tokens = _resolve_max_tokens(step_key, default_max_tokens)
    # Back-compat: allow legacy ..._DIALOG keys for CHARACTER_DIALOG
    if step_key == "CHARACTER_DIALOG":
        if model is None:
            model = _env_str("GW_MODEL_DIALOG")
        if os.getenv("GW_TEMP_CHARACTER_DIALOG") is None and os.getenv("GW_TEMP_DIALOG") is not None:
            try:
                temp = float(os.getenv("GW_TEMP_DIALOG", str(temp)))
            except Exception:
                pass
        if os.getenv("GW_MAX_TOKENS_CHARACTER_DIALOG") is None and os.getenv("GW_MAX_TOKENS_DIALOG") is not None:
            try:
                v = int(os.getenv("GW_MAX_TOKENS_DIALOG", str(max_tokens)))
                max_tokens = v if v > 0 else max_tokens
            except Exception:
                pass
    return model, temp, max_tokens

def _prompt_key_from_filename(filename: str) -> str:
        """Normalize a prompt filename (e.g., 'narration_brain_storm_prompt.md') to an env key like 'NARRATION_BRAIN_STORM'.
        Rules: strip extension, strip trailing '_prompt', uppercase, non-alnum -> '_'.
        """
        try:
            base = Path(filename).name
            if base.lower().endswith(".md"):
                base = base[:-3]
            if base.lower().endswith("_prompt"):
                base = base[:-7]
            # Replace non-alnum with underscore, then uppercase
            key = re.sub(r"[^A-Za-z0-9]+", "_", base).strip("_").upper()
            return key
        except Exception:
            return ""

def _env_for_prompt(template_filename: str, fallback_step_key: str, *, default_temp: float, default_max_tokens: int) -> Tuple[Optional[str], float, int]:
        """Resolve env by prompt filename first, then fall back to step-level and global defaults.
        Env precedence:
          GW_MODEL_PROMPT_<KEY> | GW_TEMP_PROMPT_<KEY> | GW_MAX_TOKENS_PROMPT_<KEY>
          -> GW_MODEL_<STEP> | GW_TEMP_<STEP> | GW_MAX_TOKENS_<STEP>
          -> GW_TEMP_DEFAULT / GW_MAX_TOKENS_DEFAULT
          -> provided defaults
        """
        key = _prompt_key_from_filename(template_filename)
        model = None
        temp: Optional[float] = None
        max_tokens: Optional[int] = None
        # Prompt-level reads
        if key:
            model = _env_str(f"GW_MODEL_PROMPT_{key}") or None
            t = os.getenv(f"GW_TEMP_PROMPT_{key}")
            if t is not None and str(t).strip() != "":
                try:
                    temp = float(t)
                except Exception:
                    temp = None
            mt = os.getenv(f"GW_MAX_TOKENS_PROMPT_{key}")
            if mt is not None and str(mt).strip() != "":
                try:
                    v = int(mt)
                    max_tokens = v if v > 0 else None
                except Exception:
                    max_tokens = None

        # Fallback to step-level resolver
        step_model, step_temp, step_max = _env_for(fallback_step_key, default_temp=default_temp, default_max_tokens=default_max_tokens)
        model = model or step_model
        if temp is None:
            temp = step_temp
        if max_tokens is None:
            max_tokens = step_max
        return model, float(temp), int(max_tokens)

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
    model, temp, max_tokens = _env_for_prompt("polish_prose_prompt.md", "POLISH_PROSE", default_temp=0.2, default_max_tokens=2000)
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
    # Clear last appended dialog; narration adds none
    state.last_appended_dialog = {}
    reps = _build_pipeline_replacements(setting, chapter, chapter_id, version, tp, state, prior_paragraph=prior_paragraph)
    # 1) Brainstorm → bullet list
    sys1 = "You are brainstorming narrative beats as concise bullet points."
    tpl1 = "narration_brain_storm_prompt.md"
    user1 = _apply_step(tpl1, reps)
    model, temp, max_toks = _env_for_prompt(tpl1, "BRAIN_STORM", default_temp=0.4, default_max_tokens=600)
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
    tpl2 = "ordering_prompt.md"
    user2 = _apply_step(tpl2, reps2)
    model2, temp2, max2 = _env_for_prompt(tpl2, "ORDERING", default_temp=0.2, default_max_tokens=600)
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
    tpl3 = "generate_narration_prompt.md"
    user3 = _apply_step(tpl3, reps3)
    model3, temp3, max3 = _env_for_prompt(tpl3, "GENERATE_NARRATION", default_temp=0.35, default_max_tokens=1000)
    narration = _llm_call_with_validation(
        sys3, user3, model=model3, temperature=temp3, max_tokens=max3, validator=validate_text,
        log_maker=(lambda attempt: (log_dir / f"{tp_index:02d}_generate_narration{'_r'+str(attempt) if attempt>1 else ''}.txt")) if log_dir else None,
    )

    # Polish
    polished = _polish_snippet(narration, setting, chapter, chapter_id, version)
    return polished

def run_explicit_pipeline(tp: TouchPoint, state: ChapterState, *, setting: dict, chapter: dict, chapter_id: str, version: int, tp_index: int, prior_paragraph: str = "", log_dir: Optional[Path] = None) -> str:
    appended: Dict[str, List[str]] = {}
    reps = _build_pipeline_replacements(setting, chapter, chapter_id, version, tp, state, prior_paragraph=prior_paragraph)
    # Brainstorm
    sys1 = "Brainstorm explicit dialog beats as bullet points."
    tpl1 = "explicit_brain_storm_prompt.md"
    user1 = _apply_step(tpl1, reps)
    m1, t1, k1 = _env_for_prompt(tpl1, "BRAIN_STORM", default_temp=0.45, default_max_tokens=600)
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
    tpl2 = "ordering_prompt.md"
    user2 = _apply_step(tpl2, reps2)
    m2, t2, k2 = _env_for_prompt(tpl2, "ORDERING", default_temp=0.25, default_max_tokens=600)
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
    tpl3 = "actor_assignment_prompt.md"
    user3 = _apply_step(tpl3, reps3)
    m3, t3, k3 = _env_for_prompt(tpl3, "ACTOR_ASSIGNMENT", default_temp=0.25, default_max_tokens=600)
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
    tpl4a = "body_language_prompt.md"
    user4a = _apply_step(tpl4a, reps4)
    m4a, t4a, k4a = _env_for_prompt(tpl4a, "BODY_LANGUAGE", default_temp=0.3, default_max_tokens=300)
    body_lang = _llm_call_with_validation(
        sys4a, user4a, model=m4a, temperature=t4a, max_tokens=k4a, validator=validate_bullet_list,
        log_maker=(lambda attempt: (log_dir / f"{tp_index:02d}_body_language{'_r'+str(attempt) if attempt>1 else ''}.txt")) if log_dir else None,
    )
    # Agenda (now character-wide, grouped by actor)
    sys4b = "Produce an agenda list grouped by actor id."
    tpl4b = "agenda_prompt.md"
    user4b = _apply_step(tpl4b, reps4)
    m4b, t4b, k4b = _env_for_prompt(tpl4b, "AGENDA", default_temp=0.3, default_max_tokens=300)
    agenda_text = _llm_call_with_validation(
        sys4b, user4b, model=m4b, temperature=t4b, max_tokens=k4b, validator=validate_text,
        log_maker=(lambda attempt: (log_dir / f"{tp_index:02d}_agenda{'_r'+str(attempt) if attempt>1 else ''}.txt")) if log_dir else None,
    )
    agenda_by_actor = _parse_agenda_by_actor(agenda_text)
    # Reactions (per-line)
    sys4c = "Produce a reaction for each line, referencing the previous line."
    tpl4c = "reaction_prompt.md"
    user4c = _apply_step(tpl4c, reps4)
    m4c, t4c, k4c = _env_for_prompt(tpl4c, "REACTIONS", default_temp=0.3, default_max_tokens=400)
    reactions_text = _llm_call_with_validation(
        sys4c, user4c, model=m4c, temperature=t4c, max_tokens=k4c, validator=validate_actor_list,
        log_maker=(lambda attempt: (log_dir / f"{tp_index:02d}_reactions{'_r'+str(attempt) if attempt>1 else ''}.txt")) if log_dir else None,
    )
    # Join and render character dialog lines
    # Preload character YAML (JSON text) from in-memory setting to avoid file I/O and YAML parsing
    char_yaml_by_id: Dict[str, str] = {}
    try:
        all_chars = setting.get("Characters") if isinstance(setting, dict) else None
        if isinstance(all_chars, list):
            for ch in all_chars:
                try:
                    cid = str(ch.get("id", "")).strip()
                    if cid:
                        char_yaml_by_id[cid.lower()] = _to_text(ch)
                except Exception:
                    continue
    except Exception:
        pass
    outputs: List[str] = []
    for b_index, (actor_id, line_hint) in enumerate(_parse_actor_lines(actor_lines), start=1):
        # Build a per-line prompt combining hints
        # Select the per-actor agenda block and the per-line reaction
        agenda_block = agenda_by_actor.get(actor_id, "")
        reaction_line = _pick_bullet_by_index(reactions_text, b_index)
        # Use only the body-language bullet for this specific line
        body_for_line = _pick_bullet_by_index(body_lang, b_index)
        combined_prompt = (
            f"Line intent: {line_hint}\n"
            f"Reaction: {reaction_line}\n"
            f"Agenda: {agenda_block}\n"
            f"Body language: {body_for_line}\n"
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
        )
        # Inline body language for this line (leading clause)
        outputs.append(_inline_body_with_dialog(body_for_line, resp))
        if resp.strip():
            state.add_dialog_line(actor_id, resp.strip())
            appended.setdefault(actor_id, []).append(resp.strip())
    text = "\n".join(outputs)
    out = _polish_snippet(text, setting, chapter, chapter_id, version)
    state.last_appended_dialog = appended
    return out

def run_implicit_pipeline(tp: TouchPoint, state: ChapterState, *, setting: dict, chapter: dict, chapter_id: str, version: int, tp_index: int, prior_paragraph: str = "", log_dir: Optional[Path] = None) -> str:
    appended: Dict[str, List[str]] = {}
    reps = _build_pipeline_replacements(setting, chapter, chapter_id, version, tp, state, prior_paragraph=prior_paragraph)
    # Implicit brainstorm
    sys1 = "Brainstorm implicit dialog beats (indirect, subtext) as bullet points."
    tpl1 = "implicit_brain_storm_prompt.md"
    user1 = _apply_step(tpl1, reps)
    m1, t1, k1 = _env_for_prompt(tpl1, "BRAIN_STORM", default_temp=0.5, default_max_tokens=600)
    brainstorm = _llm_call_with_validation(
        sys1, user1, model=m1, temperature=t1, max_tokens=k1, validator=validate_bullet_list,
        log_maker=(lambda attempt: (log_dir / f"{tp_index:02d}_brainstorm_implicit{'_r'+str(attempt) if attempt>1 else ''}.txt")) if log_dir else None,
    )
    # Post-process: limit to at most 10 bullets
    brainstorm = _truncate_brainstorm(brainstorm, limit=10)
    # Ordering
    reps2 = dict(reps); reps2["[BULLET_IDEAS]"] = brainstorm
    sys2 = "Order implicit dialog beats for flow."
    tpl2 = "ordering_prompt.md"
    user2 = _apply_step(tpl2, reps2)
    m2, t2, k2 = _env_for_prompt(tpl2, "ORDERING", default_temp=0.25, default_max_tokens=600)
    ordered = _llm_call_with_validation(
        sys2, user2, model=m2, temperature=t2, max_tokens=k2, validator=validate_bullet_list,
        log_maker=(lambda attempt: (log_dir / f"{tp_index:02d}_ordering{'_r'+str(attempt) if attempt>1 else ''}.txt")) if log_dir else None,
    )
    # Actor assignment → actor lines ("id: line" entries)
    reps3 = dict(reps2); reps3["[ORDERED_BULLETS]"] = ordered
    sys3 = "Assign dialog lines to actors as 'id: line' entries."
    tpl3 = "actor_assignment_prompt.md"
    user3 = _apply_step(tpl3, reps3)
    m3, t3, k3 = _env_for_prompt(tpl3, "ACTOR_ASSIGNMENT", default_temp=0.25, default_max_tokens=600)
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
    tpl4a = "body_language_prompt.md"
    user4a = _apply_step(tpl4a, reps4)
    m4a, t4a, k4a = _env_for_prompt(tpl4a, "BODY_LANGUAGE", default_temp=0.3, default_max_tokens=300)
    body_lang = _llm_call_with_validation(
        sys4a, user4a, model=m4a, temperature=t4a, max_tokens=k4a, validator=validate_bullet_list,
        log_maker=(lambda attempt: (log_dir / f"{tp_index:02d}_body_language{'_r'+str(attempt) if attempt>1 else ''}.txt")) if log_dir else None,
    )
    sys4b = "Produce an agenda list grouped by actor id."
    tpl4b = "agenda_prompt.md"
    user4b = _apply_step(tpl4b, reps4)
    m4b, t4b, k4b = _env_for_prompt(tpl4b, "AGENDA", default_temp=0.3, default_max_tokens=300)
    agenda_text = _llm_call_with_validation(
        sys4b, user4b, model=m4b, temperature=t4b, max_tokens=k4b, validator=validate_text,
        log_maker=(lambda attempt: (log_dir / f"{tp_index:02d}_agenda{'_r'+str(attempt) if attempt>1 else ''}.txt")) if log_dir else None,
    )
    agenda_by_actor = _parse_agenda_by_actor(agenda_text)
    sys4c = "Produce a reaction for each line, referencing the previous line."
    tpl4c = "reaction_prompt.md"
    user4c = _apply_step(tpl4c, reps4)
    m4c, t4c, k4c = _env_for_prompt(tpl4c, "REACTIONS", default_temp=0.3, default_max_tokens=400)
    reactions_text = _llm_call_with_validation(
        sys4c, user4c, model=m4c, temperature=t4c, max_tokens=k4c, validator=validate_actor_list,
        log_maker=(lambda attempt: (log_dir / f"{tp_index:02d}_reactions{'_r'+str(attempt) if attempt>1 else ''}.txt")) if log_dir else None,
    )
    # Join and render dialog
    # Preload character YAML (JSON text) from in-memory setting
    char_yaml_by_id: Dict[str, str] = {}
    try:
        all_chars = setting.get("Characters") if isinstance(setting, dict) else None
        if isinstance(all_chars, list):
            for ch in all_chars:
                try:
                    cid = str(ch.get("id", "")).strip()
                    if cid:
                        char_yaml_by_id[cid.lower()] = _to_text(ch)
                except Exception:
                    continue
    except Exception:
        pass
    outputs: List[str] = []
    for b_index, (actor_id, line_hint) in enumerate(_parse_actor_lines(actor_lines), start=1):
        agenda_block = agenda_by_actor.get(actor_id, "")
        reaction_line = _pick_bullet_by_index(reactions_text, b_index)
        body_for_line = _pick_bullet_by_index(body_lang, b_index)
        combined_prompt = (
            f"(Implicit) Line intent: {line_hint}\n"
            f"Reaction: {reaction_line}\n"
            f"Agenda: {agenda_block}\n"
            f"Body language: {body_for_line}\n"
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
        )
        outputs.append(_inline_body_with_dialog(body_for_line, resp))
        if resp.strip():
            state.add_dialog_line(actor_id, resp.strip())
            appended.setdefault(actor_id, []).append(resp.strip())
    text = "\n".join(outputs)
    out = _polish_snippet(text, setting, chapter, chapter_id, version)
    state.last_appended_dialog = appended
    return out

def run_subtle_edit_pipeline(tp: TouchPoint, state: ChapterState, *, setting: dict, chapter: dict, chapter_id: str, version: int, tp_index: int, prior_polished: str, prior_suggestions: str, prior_paragraph: str = "", log_dir: Optional[Path] = None) -> str:
    reps = _build_pipeline_replacements(setting, chapter, chapter_id, version, tp, state, prior_paragraph=prior_paragraph)
    reps["[PRIOR_POLISHED]"] = prior_polished
    reps["[SUGGESTIONS]"] = prior_suggestions
    sys1 = "Apply subtle edits to the provided prose respecting suggestions and style."
    tpl_se = "subtle_edit_prompt.md"
    user1 = _apply_step(tpl_se, reps)
    m1, t1, k1 = _env_for_prompt(tpl_se, "SUBTLE_EDIT", default_temp=0.2, default_max_tokens=1000)
    edited = _llm_call_with_validation(
        sys1, user1, model=m1, temperature=t1, max_tokens=k1, validator=validate_text,
        log_maker=(lambda attempt: (log_dir / f"{tp_index:02d}_subtle_edit{'_r'+str(attempt) if attempt>1 else ''}.txt")) if log_dir else None,
    )
    out = _polish_snippet(edited, setting, chapter, chapter_id, version)
    state.last_appended_dialog = {}
    return out


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
    # Avoid large YAML dumps by default; enable via GW_INCLUDE_RAW_YAML=1 when needed
    include_raw = os.getenv("GW_INCLUDE_RAW_YAML", "0") == "1"
    rep = {
        "[SETTING.yaml]": _to_text(setting) if include_raw else "",
        "[CHAPTER_xx.yaml]": _to_text(chapter) if include_raw else "",
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

    # Character YAML is now provided by caller to avoid runtime YAML parsing

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

    # Warn if caller did not provide character YAML (may reduce output quality)
    try:
        warn_dir = log_file.parent if log_file is not None else None
        if not (character_yaml or "").strip():
            _log_warning(f"CHARACTER: Missing character_yaml for id '{character_id}'.", warn_dir)
    except Exception:
        pass

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
        cy = character_yaml or ""
        if not cy.strip():
            try:
                _log_warning(f"CHARACTER: <character_yaml/> placeholder empty for id '{character_id}'.", warn_dir)
            except Exception:
                pass
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
    dialog_model, dialog_temp_env, dialog_max_tokens = _env_for("CHARACTER_DIALOG", default_temp=temperature if temperature is not None else 0.3, default_max_tokens=max_tokens_line)
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
                    char_yaml_by_id[cid.lower()] = _to_text(ch)
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
    setting = load_yaml("SETTING.yaml")
    chapter = load_yaml(chapter_path)
    chapter_id = chapter_id_from_path(chapter_path)

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
    setting = load_yaml("SETTING.yaml")
    chapter = load_yaml(chapter_path)
    chapter_id = chapter_id_from_path(chapter_path)

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
    ssf_model, ssf_temp, ssf_max_tokens = _env_for_prompt("story_so_far_prompt.md", "STORY_SO_FAR", default_temp=0.2, default_max_tokens=1200)
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
    srt_model, srt_temp, srt_max_tokens = _env_for_prompt("story_relative_to_prompt.md", "STORY_RELATIVE", default_temp=0.2, default_max_tokens=1400)
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

# ---- Resume/Checkpoint helpers ----
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
        save_text(tp_dir / "touch_point_state.json", _to_text(cp))
    except Exception:
        pass

def _resume_scan(base_log_dir: Optional[Path]) -> Dict[int, Dict[str, str]]:
    """Return map of completed tp_index -> info with keys: 'type', 'draft_path', 'state_path'."""
    done: Dict[int, Dict[str, str]] = {}
    if base_log_dir is None or not base_log_dir.exists():
        return done
    try:
        for p in sorted(base_log_dir.iterdir()):
            if not p.is_dir():
                continue
            m = re.match(r"^(\d{2})_([A-Za-z0-9_\-]+)$", p.name)
            if not m:
                continue
            idx = int(m.group(1))
            draft = p / "touch_point_draft.txt"
            statep = p / "touch_point_state.json"
            if draft.exists():
                done[idx] = {
                    "type": m.group(2),
                    "draft_path": str(draft),
                    "state_path": str(statep) if statep.exists() else "",
                }
    except Exception:
        return done
    return done

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
    setting = load_yaml("SETTING.yaml")
    chapter = load_yaml(chapter_path)
    chapter_id = chapter_id_from_path(chapter_path)
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

    # Validate Characters parsing
    try:
        characters = setting.get("Characters") if isinstance(setting, dict) else None
        if not isinstance(characters, list):
            _log_warning("SETTING.yaml: Characters not parsed as a list.", warn_log_dir)
            characters = []
        if len(characters) == 0:
            _log_warning("SETTING.yaml: Characters length is 0.", warn_log_dir)
    except Exception:
        _log_warning("SETTING.yaml: Error reading Characters.", warn_log_dir)
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
            # Build lookup sets from SETTING.yaml
            fact_set = { _norm_token(f.get("name", "")) for f in factoids if isinstance(f, dict) }
            char_id_set = { _norm_token(c.get("id", "")) for c in characters if isinstance(c, dict) }
            # Compare, ignoring case and leading/trailing quotes
            for x in (factoid_names or []):
                if _norm_token(x) and _norm_token(x) not in fact_set:
                    _log_warning(f"CHAPTER setting: factoid '{x}' does not match any Factoids.name in SETTING.yaml.", warn_log_dir)
            for a in (actor_names or []):
                if _norm_token(a) and _norm_token(a) not in char_id_set:
                    _log_warning(f"CHAPTER setting: actor '{a}' does not match any Characters.id in SETTING.yaml.", warn_log_dir)
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
                    snapshot = _collect_program_env_snapshot()
                    f.write("\n[ENVIRONMENT]\n")
                    f.write(_to_text(snapshot))
                    f.write("\n")
                except Exception:
                    pass
    except Exception:
        pass

    for i, tp in enumerate(tps, start=1):
        tp_type = tp.get("type", "")
        tp_id = tp.get("id", str(i))
        tp_text = tp.get("content", "")
        print(f"[v{version_num}] Touch-point {i}/{total} – {tp_type}: {tp_text[:60]}")
        tp_log_dir = (base_log_dir / f"{i:02d}_{tp_type}") if base_log_dir else None

        # Skip already completed touch-points by checkpoint presence
        if i in completed:
            _log_warning(f"RESUME: Skipping completed touch-point {i} ({tp_type}).", base_log_dir)
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
                all_chars = setting.get("Characters") if isinstance(setting, dict) else None
                if isinstance(all_chars, list) and state.active_actors:
                    wanted = {a.strip().lower() for a in state.active_actors if a.strip()}
                    for ch in all_chars:
                        cid = str(ch.get("id", "")).strip().lower()
                        cname = str(ch.get("name", "")).strip().lower()
                        if cid in wanted or cname in wanted:
                            sel_chars.append(ch)
                state.characters_block = _to_text({"Selected-Characters": sel_chars}) if sel_chars else ""
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
        records.append((tp_id, tp_type, tp_text, polished_text))
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

    # Optional: enable crash tracing
    crash_trace = os.getenv("GW_CRASH_TRACE", "0") == "1"

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

    # Setup crash tracing/stack dumps if requested
    if crash_trace:
        try:
            p = iter_dir_for(chapter_id)
            crash_log_dir = p / f"pipeline_v{version_num}"
            crash_log_dir.mkdir(parents=True, exist_ok=True)
            crash_log_file = crash_log_dir / "crash_trace.log"
            os.environ["GW_CRASH_TRACE_FILE"] = str(crash_log_file)
            fh = open(crash_log_file, "a", encoding="utf-8")
            faulthandler.enable(file=fh, all_threads=True)
            for sig in (signal.SIGSEGV, signal.SIGABRT):
                try:
                    faulthandler.register(sig, file=fh, all_threads=True, chain=True)
                except Exception:
                    pass
            _breadcrumb("main:crash-trace-enabled")
        except Exception:
            pass

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