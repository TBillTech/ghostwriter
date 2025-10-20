"""Checkpoint/resume helpers (skeleton)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict


def touchpoint_dir(base: Path, index: int, tp_type: str) -> Path:
    return base / f"{index:02d}_{tp_type}"


def scan_completed(base: Path) -> Dict[int, Dict[str, str]]:
    done: Dict[int, Dict[str, str]] = {}
    if not base.exists():
        return done
    for p in sorted(base.iterdir()):
        if not p.is_dir():
            continue
        name = p.name
        if len(name) < 3 or not name[:2].isdigit():
            continue
        idx = int(name[:2])
        draft = p / "touch_point_draft.txt"
        if draft.exists():
            done[idx] = {
                "type": name[3:],
                "draft_path": str(draft),
                "state_path": str(p / "touch_point_state.json"),
            }
    return done
