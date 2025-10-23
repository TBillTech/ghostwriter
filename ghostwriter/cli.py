"""GhostWriter CLI entrypoint.

Usage:
  ghostwriter.cli run <chapter_path> [vN|auto] [--log-llm]

This CLI delegates to functions provided by scripts/driver.py via
ghostwriter.commands to avoid import cycles during the refactor.
"""
from __future__ import annotations

import argparse
import sys

from pathlib import Path
from .commands import (
    run_pipelines_for_chapter,
    validate_and_prepare,
)
from .env import load_env
from .env import resolve_chapter_path
from .context import load_yaml
from .pipelines import run_chapter_brainstorm, run_character_brainstorm, run_content_table_brainstorm
from .context import RunContext
from .utils import _norm_token
from .env import get_chapters_dir
from .templates import iter_dir_for, get_latest_version


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="ghostwriter", description="GhostWriter CLI")
    sub = parser.add_subparsers(dest="cmd")

    p_run = sub.add_parser("run", help="Run pipelines for a chapter")
    p_run.add_argument("chapter_path", help="Path to chapter yaml, e.g., chapters/CHAPTER_001.yaml")
    p_run.add_argument("version", nargs="?", help="vN or 'auto' (default: auto)")
    p_run.add_argument("--log-llm", action="store_true", dest="log_llm", help="Log LLM prompts/responses")
    p_run.add_argument("--book-base", dest="book_base", help="Override GW_BOOK_BASE_DIR for this run")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    load_env()
    argv_list = list(sys.argv[1:] if argv is None else argv)
    ns = _parse_args(argv_list)

    # Legacy mode: invoked as `scripts/driver.py chapters/CHAPTER_xx.yaml [vN|auto] [--log-llm]`
    if ns.cmd is None and argv_list:
        # Simple legacy parse
        chapter_path = None
        version_token = None
        log_llm = False
        for a in list(argv_list):
            if a == "--log-llm" or a == "--show-dialog":
                log_llm = True
        # Legacy parse for --book-base
        book_base = None
        if "--book-base" in argv_list:
            try:
                idx = argv_list.index("--book-base")
                if idx + 1 < len(argv_list):
                    book_base = argv_list[idx + 1]
            except Exception:
                book_base = None
        # chapter path is first non-flag
        toks = [t for t in argv_list if not t.startswith("--")]
        if toks:
            chapter_path = toks[0]
            if len(toks) > 1:
                version_token = toks[1]
        if chapter_path:
            ns = argparse.Namespace(cmd="run", chapter_path=chapter_path, version=version_token, log_llm=log_llm, book_base=book_base)
        else:
            # Fall back to showing help
            print("Usage: ghostwriter run <chapters/CHAPTER_xx.yaml> [vN|auto] [--log-llm] [--book-base <dir>]")
            return 1

    if ns.cmd == "run":
        # Apply --book-base override if provided
        if getattr(ns, "book_base", None):
            import os
            os.environ["GW_BOOK_BASE_DIR"] = str(ns.book_base)
        # Iterations dir is always resolved from configuration (see templates.iter_dir_for)
        chapter_path = ns.chapter_path
        def _chapter_id_from_path(chapter_path: str) -> str:
            return Path(chapter_path).stem
        chapter_id = _chapter_id_from_path(chapter_path)
        # Determine version number
        if ns.version and str(ns.version).startswith("v"):
            try:
                version_num = int(str(ns.version)[1:])
            except ValueError:
                print("Invalid version format. Use v1, v2, ...")
                return 1
        elif ns.version == "auto" or ns.version is None:
            version_num = get_latest_version(chapter_id) + 1
        else:
            version_num = get_latest_version(chapter_id) + 1

        # Run CONTENT_TABLE.yaml brainstorming FIRST if '???' placeholder exists, regardless of chapter arg
        try:
            ct_path = get_chapters_dir() / "CONTENT_TABLE.yaml"
            if ct_path.exists():
                ct_yaml = load_yaml(str(ct_path))
                toc = ct_yaml.get("TABLE_OF_CONTENTS") if isinstance(ct_yaml, dict) else None
                if isinstance(toc, list):
                    has_placeholder = any(isinstance(item, dict) and ("???" in item) for item in toc)
                    if has_placeholder:
                        run_content_table_brainstorm()
                        return 0
        except Exception:
            pass

        # Branch: CONTENT_TABLE.yaml brainstorming
        if Path(chapter_path).name.upper() == "CONTENT_TABLE.YAML":
            run_content_table_brainstorm()
            return 0

        # Branch: Chapter brainstorming conditions
        # 1) If chapter file is missing, trigger brainstorming pipeline
        resolved = resolve_chapter_path(chapter_path)
        if not resolved.exists():
            run_chapter_brainstorm(chapter_path, version_num, log_llm=bool(ns.log_llm))
            return 0

        # 2) If chapter contains a touch-point with `brainstorming: True`, trigger brainstorming
        try:
            ch_yaml = load_yaml(str(resolved))
            if isinstance(ch_yaml, dict):
                tps = ch_yaml.get("Touch-Points") or ch_yaml.get("TouchPoints") or []
                if isinstance(tps, list):
                    for it in tps:
                        if isinstance(it, dict) and it.get("brainstorming") is True:
                            run_chapter_brainstorm(chapter_path, version_num, log_llm=bool(ns.log_llm))
                            return 0
        except Exception:
            pass

        # Character brainstorming trigger (Task 2): missing actor in chapter setting OR existing character with brainstorming: True
        try:
            resolved2 = resolve_chapter_path(chapter_path)
            if resolved2.exists():
                ctx = RunContext.from_paths(chapter_path=str(resolved2), version=version_num)
                # 1) existing character with brainstorming: True
                target_name: str | None = None
                for ch in (ctx.characters or []):
                    try:
                        if ch.get("brainstorming") is True:
                            # Prefer id; fallback to name
                            target_name = str(ch.get("id") or ch.get("name") or "").strip()
                            if target_name:
                                break
                    except Exception:
                        continue
                # 2) missing actor referenced in chapter setting
                if target_name is None:
                    chs = ctx.chapter.get("setting") if isinstance(ctx.chapter, dict) else None
                    actor_list = chs.get("actors") if isinstance(chs, dict) and isinstance(chs.get("actors"), list) else []
                    # Build lookup set of known ids/names
                    known: set[str] = set()
                    for c in (ctx.characters or []):
                        try:
                            if c.get("id"):
                                known.add(_norm_token(c.get("id")))
                            if c.get("name"):
                                known.add(_norm_token(c.get("name")))
                        except Exception:
                            continue
                    for a in actor_list:
                        if _norm_token(a) and _norm_token(a) not in known:
                            target_name = str(a)
                            break
                if target_name:
                    # For existing 'brainstorming: True' character, seed description from its YAML in pipeline
                    run_character_brainstorm(ctx=ctx, target_name=target_name, version_num=version_num, user_description=None)
                    return 0
        except Exception:
            # Non-fatal; proceed to normal validation
            pass

        # Validate early and ensure iteration dir exists
        try:
            validate_and_prepare(chapter_path)
        except Exception as e:
            print(f"Error: {e}")
            return 2

        # Enable crash tracing if requested (match legacy driver behavior)
        try:
            import os as _os
            if _os.getenv("GW_CRASH_TRACE", "0") == "1":
                import faulthandler as _faulthandler
                import signal as _signal
                base_log_dir = iter_dir_for(chapter_id) / f"pipeline_v{version_num}"
                base_log_dir.mkdir(parents=True, exist_ok=True)
                crash_log_file = base_log_dir / "crash_trace.log"
                _os.environ["GW_CRASH_TRACE_FILE"] = str(crash_log_file)
                fh = open(crash_log_file, "a", encoding="utf-8")
                _faulthandler.enable(file=fh, all_threads=True)
                for sig in (getattr(_signal, "SIGSEGV", None), getattr(_signal, "SIGABRT", None)):
                    try:
                        if sig is not None:
                            _faulthandler.register(sig, file=fh, all_threads=True, chain=True)
                    except Exception:
                        pass
        except Exception:
            pass

        run_pipelines_for_chapter(chapter_path, version_num, log_llm=bool(ns.log_llm))
        return 0

    print("No command executed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
