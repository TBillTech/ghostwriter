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
        chapter_path = ns.chapter_path
        def _chapter_id_from_path(chapter_path: str) -> str:
            return Path(chapter_path).stem
        def _iter_dir_for(cid: str) -> Path:
            return Path("iterations") / cid
        def _list_versions(cid: str, prefix: str) -> list[int]:
            d = _iter_dir_for(cid)
            if not d.exists():
                return []
            versions = []
            for p in d.glob(f"{prefix}_v*.txt"):
                import re as _re
                m = _re.search(r"_v(\d+)\.txt$", p.name)
                if m:
                    versions.append(int(m.group(1)))
            return sorted(set(versions))
        def _get_latest_version(cid: str) -> int:
            latest = 0
            for pref in ("draft", "check", "suggestions", "pre_draft"):
                vs = _list_versions(cid, pref)
                if vs:
                    latest = max(latest, vs[-1])
            return latest
        chapter_id = _chapter_id_from_path(chapter_path)
        # Determine version number
        if ns.version and str(ns.version).startswith("v"):
            try:
                version_num = int(str(ns.version)[1:])
            except ValueError:
                print("Invalid version format. Use v1, v2, ...")
                return 1
        elif ns.version == "auto" or ns.version is None:
            version_num = _get_latest_version(chapter_id) + 1
        else:
            version_num = _get_latest_version(chapter_id) + 1

        # Validate early and ensure iteration dir exists
        try:
            validate_and_prepare(chapter_path)
        except Exception as e:
            print(f"Error: {e}")
            return 2

        run_pipelines_for_chapter(chapter_path, version_num, log_llm=bool(ns.log_llm))
        return 0

    print("No command executed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
