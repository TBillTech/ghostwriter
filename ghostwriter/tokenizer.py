"""Token counting helpers using tiktoken when available.

- count_text_tokens(text: str, model: str) -> int
- count_chat_tokens(messages: list[dict], model: str) -> int

If tiktoken is unavailable, falls back to a rough chars-per-token estimate
controlled by GW_CHARS_PER_TOKEN (default 4).
"""
from __future__ import annotations

from typing import List, Dict
import os

try:
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover
    tiktoken = None  # type: ignore


def _encoding_for_model(model: str):
    if tiktoken is None:
        return None
    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        # Fallbacks for newer families
        try:
            # Many GPT-4o family models use o200k_base
            return tiktoken.get_encoding("o200k_base")
        except Exception:
            try:
                return tiktoken.get_encoding("cl100k_base")
            except Exception:
                return None


def count_text_tokens(text: str, model: str) -> int:
    enc = _encoding_for_model(model)
    if enc is None:
        # Estimate using chars-per-token heuristic
        try:
            cpt = float(os.getenv("GW_CHARS_PER_TOKEN", "4") or "4")
        except Exception:
            cpt = 4.0
        return int((len(text or "") / cpt) + 0.5)
    try:
        return len(enc.encode(text or ""))
    except Exception:
        return 0


def count_chat_tokens(messages: List[Dict[str, str]], model: str) -> int:
    """Approximate tokens for chat messages according to ChatML-like rules.

    This follows OpenAI's guidance for GPT-3.5/4 families where each message carries
    an overhead; values vary by model snapshot. We use a conservative default of:
      tokens_per_message = 3, tokens_per_name = 1, plus 3 for assistant priming.
    """
    enc = _encoding_for_model(model)
    if enc is None:
        # Fallback: sum text tokens with heuristic
        total = 0
        for m in messages:
            total += count_text_tokens(str(m.get("role", "")), model)
            total += count_text_tokens(str(m.get("content", "")), model)
            if m.get("name"):
                total += count_text_tokens(str(m.get("name")), model)
        # Add rough overhead
        return total + 6
    # Defaults per OpenAI docs for many GPT-3.5/4 variants
    tokens_per_message = 3
    tokens_per_name = 1
    try:
        # Empirical: newer o-series may have different overheads; we keep defaults
        pass
    except Exception:
        pass
    total = 0
    for m in messages:
        total += tokens_per_message
        total += len(enc.encode(str(m.get("role", ""))))
        total += len(enc.encode(str(m.get("content", ""))))
        if m.get("name"):
            total += tokens_per_name
            total += len(enc.encode(str(m.get("name"))))
    # Every reply is primed with <im_start>assistant
    total += 3
    return total
