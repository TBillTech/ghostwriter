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
    base = os.getenv("GW_BOOK_BASE_DIR")
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
    raise ValueError("MockLLM: prompt not found in golden index. Consider running golden-update on the book base, or disable strict mode via GW_MOCKLLM_FALLBACK=1.")
