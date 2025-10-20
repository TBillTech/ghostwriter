"""Logging helpers: crash tracing breadcrumbs and warnings.

This module centralizes lightweight logging utilities used across the project.

Public API:
- crash_trace_file() -> Optional[str]
- rss_kb() -> Optional[int]
- breadcrumb(label: str) -> None
- log_warning(msg: str, log_dir: Optional[Path]) -> None
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import os
import sys
import time
import threading

try:  # POSIX resource usage for RSS
    import resource  # type: ignore
except Exception:  # pragma: no cover
    resource = None  # type: ignore


def crash_trace_file() -> Optional[str]:
    return os.getenv("GW_CRASH_TRACE_FILE")


def rss_kb() -> Optional[int]:
    try:
        if resource is None:
            return None
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return int(getattr(usage, "ru_maxrss", 0))
    except Exception:
        return None


def breadcrumb(label: str) -> None:
    path = crash_trace_file()
    if not path:
        return
    try:
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        tid = threading.get_ident()
        mem = rss_kb()
        line = f"{ts} pid={os.getpid()} tid={tid} rss_kb={mem or 'na'} | {label}\n"
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


def log_warning(msg: str, log_dir: Optional[Path]) -> None:
    try:
        print(f"WARNING: {msg}")
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            with (log_dir / "warnings.txt").open("a", encoding="utf-8") as f:
                f.write(msg + "\n")
    except Exception:
        pass
