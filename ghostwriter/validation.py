"""Validation utilities used by pipelines."""
from __future__ import annotations
import re
from typing import Tuple


def validate_text(output: str) -> Tuple[bool, str]:
    """Text must be non-empty (after stripping). Used for prose-like prompts.

    We intentionally keep this permissive: any non-empty content is acceptable.
    Downstream checks enforce shape/quality as needed.
    """
    if output is None:
        return False, "No output returned (None)"
    if str(output).strip() == "":
        return False, "Empty output not allowed"
    return True, "ok"


_ACTOR_LINE_RE = re.compile(r"^\s*([A-Za-z0-9_.\-]+)\s*:\s+.+")


def validate_bullet_list(output: str) -> Tuple[bool, str]:
    lines = [ln.rstrip() for ln in output.splitlines() if ln.strip() and not ln.lstrip().startswith('#')]
    bullets = [ln for ln in lines if ln.lstrip().startswith("*")]
    if len(bullets) >= 2:
        return True, "ok"
    return False, "Expected a bullet list with at least 2 lines starting with '*'"


def validate_actor_list(output: str) -> Tuple[bool, str]:
    lines = [ln for ln in output.splitlines() if ln.strip() and not ln.lstrip().startswith('#')]
    count = 0
    for ln in lines:
        if _ACTOR_LINE_RE.match(ln):
            count += 1
    if count >= 2:
        return True, "ok"
    return False, "Expected an actor list with at least 2 actor-attributed lines like 'id: ...'"
