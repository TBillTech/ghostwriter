"""Logging helpers: crash tracing breadcrumbs and warnings.

This module centralizes lightweight logging utilities used across the project.

Public API:
- crash_trace_file() -> Optional[str]
- rss_kb() -> Optional[int]
- breadcrumb(label: str) -> None
- log_warning(msg: str, log_dir: Optional[Path]) -> None
 - log_error_base(msg: str) -> None
 - log_run(msg: str) -> None
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
    try:
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        tid = threading.get_ident()
        mem = rss_kb()
        line = f"{ts} pid={os.getpid()} tid={tid} rss_kb={mem or 'na'} | {label}\n"
        # Write to crash trace file if enabled
        if path:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
        # Also append all breadcrumbs to base run.log
        try:
            log_run(f"BREADCRUMB | {label}")
        except Exception:
            pass
        try:
            sys.stderr.write(f"[crumb] {label}\n")
            sys.stderr.flush()
        except Exception:
            pass
    except Exception:
        pass


def log_warning(msg: str, log_dir: Optional[Path]) -> None:
    """Log a warning message to stdout and base run.log (ignores log_dir)."""
    try:
        text = f"WARNING: {msg}"
        print(text)
        log_run(text)
    except Exception:
        pass


def log_error_base(msg: str) -> None:
    """Append an error message to the base book directory (run_error.log)."""
    try:
        from .env import get_book_base_dir  # lazy import to avoid cycles
        base = get_book_base_dir()
        base.mkdir(parents=True, exist_ok=True)
        path = base / "run_error.log"
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        with path.open("a", encoding="utf-8") as f:
            f.write(f"[{ts}] {msg}\n")
        # Also append to unified run.log
        try:
            log_run(f"ERROR: {msg}")
        except Exception:
            pass
        try:
            print(f"ERROR: {msg}")
        except Exception:
            pass
    except Exception:
        # Last resort: print only
        try:
            print(f"ERROR: {msg}")
        except Exception:
            pass


def log_run(msg: str) -> None:
    """Append a message to the unified base run.log file."""
    try:
        from .env import get_book_base_dir  # lazy import
        base = get_book_base_dir()
        base.mkdir(parents=True, exist_ok=True)
        path = base / "run.log"
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        with path.open("a", encoding="utf-8") as f:
            f.write(f"[{ts}] {msg}\n")
    except Exception:
        pass
