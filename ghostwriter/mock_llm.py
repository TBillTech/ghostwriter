"""Deterministic MockLLM (Task 9).

Provides a drop-in completion function backed by a golden book base
(e.g., testdata/LittleRedRidingHood). It maps the current templates and
replacements for each golden log to a regenerated USER prompt and returns the
stored RESPONSE when the incoming prompt matches (ignoring whitespace).

Usage:
- Set GW_USE_MOCK_LLM=1 and GW_BOOK_BASE_DIR=<path to golden book>
- Call ghostwriter.llm.complete(...) as usual

Strictness:
- By default, prompts must match a golden prompt after whitespace normalization.
  If no match is found, a ValueError is raised. This helps surface template
  drift. You can set GW_MOCKLLM_FALLBACK="1" to allow a soft fallback to a
  generic mock response instead of raising.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, List
import os
import re

from .mock_support import regenerate_prompt_for_log, parse_prompt_response_file, compute_prompts_hash, read_prompt_hash

# Cache: (book_base_dir, prompts_hash) -> { normalized_user_prompt: [responses in order] }
_CACHE: Dict[Tuple[str, str], Dict[str, List[str]]] = {}
# Per-process counters for sequential multi-round behavior
_COUNTS: Dict[Tuple[str, str, str], int] = {}


def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _build_index(book_base_dir: str | Path) -> Dict[str, List[str]]:
    base = str(Path(book_base_dir).resolve())
    # Use the book's prompt_hash if present, else current prompts hash
    ph = read_prompt_hash(book_base_dir) or compute_prompts_hash("prompts")
    key = (base, ph)
    if key in _CACHE:
        return _CACHE[key]
    index: Dict[str, List[str]] = {}
    iters = Path(book_base_dir) / "iterations"
    if not iters.exists():
        _CACHE[key] = index
        return index
    # Deterministic ordering: lexicographic by path ensures base files (e.g., foo.txt)
    # come before suffixed retries (e.g., foo_42.txt, foo_r2.txt)
    logs = sorted(iters.rglob("*.txt"), key=lambda p: str(p))
    for log in logs:
        try:
            text = log.read_text(encoding="utf-8")
            if "=== USER ===" not in text or "=== RESPONSE ===" not in text:
                continue
            regen = regenerate_prompt_for_log(log)
            if not regen:
                continue
            parts = parse_prompt_response_file(log)
            user_norm = _normalize_ws(regen)
            # Append in order; consumers can choose first (baseline) then retries
            resp = parts.get("RESPONSE", "")
            index.setdefault(user_norm, []).append(resp)
        except Exception:
            continue
    _CACHE[key] = index
    return index


def complete(prompt: str, *, system: Optional[str] = None, temperature: float = 0.0, max_tokens: int = 0, model: Optional[str] = None, reasoning_effort: Optional[str] = None) -> str:
    # Allow reading golden responses from a different base than the active run base
    base = os.getenv("GW_MOCKLLM_GOLDEN_BASE") or os.getenv("GW_BOOK_BASE_DIR")
    if not base:
        raise ValueError("GW_BOOK_BASE_DIR is required for MockLLM")
    idx = _build_index(base)
    key = _normalize_ws(prompt)
    if key in idx:
        base_path = str(Path(base).resolve())
        ph = read_prompt_hash(base) or compute_prompts_hash("prompts")
        seq_key = (base_path, ph, key)
        count = _COUNTS.get(seq_key, 0)
        options = idx[key]
        # Choose based on call count; saturate at last element
        choice = options[min(count, len(options) - 1)] if options else ""
        _COUNTS[seq_key] = count + 1
        return choice
    # Strict by default to surface mismatches
    if os.getenv("GW_MOCKLLM_FALLBACK", "0") == "1":
        head = (prompt[:220] + "...") if len(prompt) > 220 else prompt
        return f"[MOCK LLM RESPONSE]\nSystem: {system or 'n/a'}\nTemp: {temperature}\n---\n{head}"
    # Optional debug dump to help diagnose mismatches in CI/tests
    if os.getenv("GW_MOCKLLM_DEBUG", "0") == "1":
        try:
            base_path = Path(base)
            # Determine optional external debug dir for easier inspection in tests/CI
            dbg_dir_env = os.getenv("GW_MOCKLLM_DEBUG_DIR")
            dbg_path = Path(dbg_dir_env).resolve() if dbg_dir_env else None
            # Prepare payloads
            sample = "\n".join(list(idx.keys())[:20])
            # Always try to write into the book base (tmp sandbox)
            (base_path / ".mockllm_last_prompt_user.txt").write_text(prompt, encoding="utf-8")
            (base_path / ".mockllm_last_prompt_norm.txt").write_text(key, encoding="utf-8")
            (base_path / ".mockllm_index_keys.txt").write_text(sample, encoding="utf-8")
            # Optionally mirror into a fixed debug dir inside the repo/workspace for visibility
            if dbg_path:
                try:
                    dbg_path.mkdir(parents=True, exist_ok=True)
                    (dbg_path / "last_prompt_user.txt").write_text(prompt, encoding="utf-8")
                    (dbg_path / "last_prompt_norm.txt").write_text(key, encoding="utf-8")
                    (dbg_path / "index_keys_sample.txt").write_text(sample, encoding="utf-8")
                    # Include a tiny meta file with index length and prompts hash
                    from .mock_support import compute_prompts_hash as _cph, read_prompt_hash as _rph
                    ph = _rph(base) or _cph("prompts")
                    (dbg_path / "meta.txt").write_text(f"index_len={len(idx)}\nprompts_hash={ph}\nbase={str(base_path)}\n", encoding="utf-8")
                except Exception:
                    pass
        except Exception:
            pass
        # Also emit a compact stderr hint with sizes and a fuzzy nearest head to aid debugging in CI logs
        try:
            import sys, hashlib, difflib
            key_sha = hashlib.sha1(key.encode("utf-8")).hexdigest() if key else ""
            idx_len = len(idx)
            # Find a fuzzy closest key (by difflib) among a small sample to avoid O(n)
            candidates = list(idx.keys())
            head = key[:160].replace("\n", " ")
            closest = ""
            ratio = 0.0
            # Sample up to first 200 keys deterministically
            for cand in candidates[:200]:
                r = difflib.SequenceMatcher(a=key, b=cand).ratio()
                if r > ratio:
                    ratio = r
                    closest = cand
            clo_sha = hashlib.sha1(closest.encode("utf-8")).hexdigest() if closest else ""
            clo_head = closest[:160].replace("\n", " ") if closest else ""
            sys.stderr.write(
                f"[mockllm] miss: idx_keys={idx_len} key_sha={key_sha} key_head={head!r} closest_ratio={ratio:.4f} closest_sha={clo_sha} closest_head={clo_head!r}\n"
            )
        except Exception:
            pass
    raise ValueError("MockLLM: prompt not found in golden index. Consider running golden-update on the book base, or disable strict mode via GW_MOCKLLM_FALLBACK=1.")
