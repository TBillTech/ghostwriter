"""Artifact I/O helpers for drafts, suggestions, and final outputs."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Dict, List, Tuple

from .utils import read_text, save_text

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


def write_records(path: str | Path, records: Iterable[Tuple[str, str, str, str]]) -> Path:
    chunks = [format_draft_record(*r) for r in records]
    save_text(path, "".join(chunks))
    return Path(path)


def read_records(path: str | Path) -> List[Dict[str, str]]:
    text = read_text(path)
    blocks = re.split(rf"(?m)^\s*{BEGIN_TP}\s+", text)
    results: List[Dict[str, str]] = []
    for blk in blocks:
        blk = blk.strip()
        if not blk:
            continue
        header_line, _, rest = blk.partition("\n")
        m = re.search(r"id=([^\s]+)\s+type=([^\s]+)", header_line)
        if not m:
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
