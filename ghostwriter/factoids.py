"""
Factoids expansion helpers for SETTING.yaml integration.

These helpers extract and format Factoids as a compact JSON-ish text block
and merge it with any existing [SETTING] block text. Selection is filtered by
explicit chapter-provided names when available.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional
import ast
import json

from .utils import to_text as _gw_to_text


def factoids_block_from_setting(setting: Dict[str, Any], selected_names: Optional[Iterable[str]] = None) -> str:
    """Return a compact text block containing Factoids from SETTING.yaml.
    Format as JSON text: {"Factoids": [{"name": ..., "description": ...}, ...]}
    Returns empty string if no factoids parsed.
    """
    try:
        if not isinstance(setting, dict):
            return ""
        facts = setting.get("Factoids")
        if not isinstance(facts, list):
            return ""
        selected: Optional[set] = None
        if selected_names is not None:
            try:
                selected = {str(n).strip().lower() for n in selected_names if str(n).strip()}
            except Exception:
                selected = None
        items = []
        for f in facts:
            if not isinstance(f, dict):
                continue
            name = f.get("name")
            desc = f.get("description")
            if not (isinstance(name, str) and name.strip()):
                continue
            if selected and str(name).strip().lower() not in selected:
                continue
            items.append({
                "name": str(name).strip(),
                "description": str(desc).strip() if isinstance(desc, str) else "",
            })
        if not items:
            return ""
        return _gw_to_text({"Factoids": items})
    except Exception:
        return ""


def selected_factoid_names_from_sources(chapter: Dict[str, Any], setting_block_text: str) -> Optional[Iterable[str]]:
    """Try to extract a list of desired factoid names.
    Prefers chapter['setting']['factoids'] when present; otherwise attempts to parse from setting_block_text.
    Returns None if no explicit selection was found.
    """
    # From chapter.setting.factoids
    try:
        ch_setting = chapter.get("setting") if isinstance(chapter, dict) else None
        if isinstance(ch_setting, dict) and isinstance(ch_setting.get("factoids"), list):
            names = [str(x) for x in ch_setting.get("factoids") if isinstance(x, (str, int, float))]
            if names:
                return names
    except Exception:
        pass
    # Attempt to parse from textual setting block (e.g., "{'factoids': ['A','B'], 'actors': [...]}" )
    try:
        text = (setting_block_text or "").strip()
        if text:
            # Try JSON first
            try:
                obj = json.loads(text)
            except Exception:
                # Fallback to Python literal eval for single-quoted dicts
                obj = ast.literal_eval(text)
            if isinstance(obj, dict):
                f = obj.get("factoids")
                if isinstance(f, list):
                    names = [str(x) for x in f if isinstance(x, (str, int, float))]
                    if names:
                        return names
    except Exception:
        pass
    return None


def merge_setting_with_factoids(setting_block_text: str, setting: Dict[str, Any], chapter: Optional[Dict[str, Any]] = None) -> str:
    """Merge any existing [SETTING] block text with expanded Factoids from SETTING.yaml.
    If either part is empty, return the other; otherwise join with a separating blank line.
    """
    try:
        selected = None
        if chapter is not None:
            selected = selected_factoid_names_from_sources(chapter, setting_block_text)
        facts_text = factoids_block_from_setting(setting, selected_names=selected)
        base = (setting_block_text or "").strip()
        if base and facts_text:
            return base + "\n\n" + facts_text
        return base or facts_text or ""
    except Exception:
        return setting_block_text or ""
