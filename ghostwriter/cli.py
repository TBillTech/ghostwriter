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
from .pipelines import run_chapter_brainstorm, run_character_brainstorm, run_content_table_brainstorm
from .context import RunContext
from .utils import _norm_token
from .env import get_chapters_dir
from .logging import breadcrumb as _breadcrumb
from .logging import init_run_logs as _init_run_logs, log_run as _log_run
from .templates import iter_dir_for, get_latest_version
from .mock_support import (
    update_golden_prompts as _update_golden_prompts,
    compute_prompts_hash as _compute_prompts_hash,
    write_prompt_hash as _write_prompt_hash,
    copy_partial_book as _copy_partial_book,
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="ghostwriter", description="GhostWriter CLI")
    sub = parser.add_subparsers(dest="cmd")

    p_run = sub.add_parser("run", help="Run pipelines for a chapter")
    p_run.add_argument("chapter_path", help="Path to chapter yaml, e.g., chapters/CHAPTER_001.yaml")
    p_run.add_argument("version", nargs="?", help="vN or 'auto' (default: auto)")
    p_run.add_argument("--log-llm", action="store_true", dest="log_llm", help="Log LLM prompts/responses")
    p_run.add_argument("--book-base", dest="book_base", help="Override GW_BOOK_BASE_DIR for this run")

    # Task 8 helpers: golden update and prompt hash utilities
    p_golden = sub.add_parser("golden-update", help="Update golden prompt logs to match current templates")
    p_golden.add_argument("--book-base", dest="book_base", help="Book base directory (defaults to GW_BOOK_BASE_DIR)")
    p_golden.add_argument("--dry-run", action="store_true", dest="dry_run", help="Scan and report without writing changes")

    # Task 10: Rebuild goldens via MockLLM into a new directory
    p_golden_rebuild = sub.add_parser("golden-rebuild", help="Rebuild a golden book using MockLLM into a destination directory")
    p_golden_rebuild.add_argument("--src-base", dest="src_base", required=True, help="Source golden base directory to read responses from")
    p_golden_rebuild.add_argument("--dest-base", dest="dest_base", required=True, help="Destination directory to rebuild into")
    p_golden_rebuild.add_argument("--no-clear", action="store_true", dest="no_clear", help="Do not clear destination before rebuild")
    # Partial rebuild / snapshot parameters (Task: add restriction up to a given step)
    # If --chapter-id and --upto-step are provided, golden-rebuild performs a partial copy only
    # (no MockLLM pipeline execution) using the existing copy_partial_book helper.
    p_golden_rebuild.add_argument("--chapter-id", dest="chapter_id", help="Restrict to a single chapter id for partial snapshot (e.g., CHAPTER_001)")
    p_golden_rebuild.add_argument("--version", dest="version", type=int, help="Highest pipeline version number to snapshot (includes all lower versions). Omit when using --all-versions to copy all.")
    p_golden_rebuild.add_argument("--upto-step", dest="upto_step", help="Inclusive step directory name cutoff (e.g., 05_narration)")
    p_golden_rebuild.add_argument("--upto-filename", dest="upto_filename", help="Optional filename cutoff within the final step directory")
    p_golden_rebuild.add_argument("--apply-templates", action="store_true", dest="apply_templates", help="(Deprecated - templates now always applied) After partial copy, update USER prompts to current templates in the destination")
    p_golden_rebuild.add_argument("--all-versions", action="store_true", dest="all_versions", help="Copy all pipeline_vN directories when performing a partial snapshot")

    p_phash = sub.add_parser("prompt-hash", help="Compute current prompts hash; optionally write to book base")
    p_phash.add_argument("--book-base", dest="book_base", help="If provided, write prompt_hash file to this directory")

    # Tutorial helper: create a minimal starter book in an empty directory
    p_new = sub.add_parser("new-book", help="Create a starter book in an empty directory")
    p_new.add_argument("book_base", help="Destination directory for the new book (must be empty or not exist)")

    return parser.parse_args(argv)


# --- Golden snapshot helpers to reduce duplication ---
def _copy_roots(src_base: Path, dest_base: Path) -> None:
    """Copy root files and chapters folder required to perform snapshots.

    Copies SETTING.yaml, CHARACTERS.yaml, and chapters/*.
    """
    import shutil
    dest_base.mkdir(parents=True, exist_ok=True)
    for name in ["SETTING.yaml", "CHARACTERS.yaml"]:
        sp = Path(src_base) / name
        if sp.exists():
            (dest_base / name).parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(sp, dest_base / name)
            except Exception:
                pass
    chapters_src = Path(src_base) / "chapters"
    if chapters_src.exists():
        shutil.copytree(chapters_src, dest_base / "chapters", dirs_exist_ok=True)


def _iter_chapter_files(base_with_chapters: Path) -> list[Path]:
    """Return sorted list of CHAPTER_*.yaml paths under base/chapters."""
    chapters_dir = Path(base_with_chapters) / "chapters"
    if not chapters_dir.exists():
        return []
    return sorted(chapters_dir.glob("CHAPTER_*.yaml"))


def _snapshot_copy_chapter(
    *,
    src_base: Path,
    dest_base: Path,
    chapter_id: str,
    upto_step_dir: str = "all",
    upto_filename: str | None = None,
    all_versions: bool = True,
    max_version: int | None = None,
    clear_dest: bool = False,
    effective_version: int = 1,
) -> None:
    """Perform snapshot copy for a single chapter via copy_partial_book.

    Mirrors semantics used in golden-rebuild/partial and golden-update.
    """
    _copy_partial_book(
        src_base=Path(src_base),
        dest_base=Path(dest_base),
        chapter_id=str(chapter_id),
        version=effective_version,
        upto_step_dir=str(upto_step_dir),
        upto_filename=upto_filename,
        clear_dest=bool(clear_dest),
        copy_root_artifacts=True,
        all_versions=bool(all_versions),
        max_version=max_version,
    )


def _snapshot_copy_all_chapters(
    *,
    src_base: Path,
    dest_base: Path,
    clear_dest: bool = True,
    upto_step_dir: str = "all",
    upto_filename: str | None = None,
    all_versions: bool = True,
    max_version: int | None = None,
) -> int:
    """Prepare destination roots and snapshot all chapters; returns count processed."""
    import shutil
    dest_base = Path(dest_base)
    if clear_dest and dest_base.exists():
        shutil.rmtree(dest_base, ignore_errors=True)
    _copy_roots(Path(src_base), dest_base)
    processed = 0
    for ch_file in _iter_chapter_files(dest_base):
        _snapshot_copy_chapter(
            src_base=Path(src_base),
            dest_base=dest_base,
            chapter_id=ch_file.stem,
            upto_step_dir=upto_step_dir,
            upto_filename=upto_filename,
            all_versions=all_versions,
            max_version=max_version,
            clear_dest=False,  # never clear per-chapter during bulk copy
            effective_version=1,  # ignored when all_versions=True
        )
        processed += 1
    return processed


def _apply_templates_and_counts(base_dir: Path) -> tuple[list[dict], int, int, int]:
    """Apply templates to USER prompts and return (results, updated, unchanged, no-template)."""
    results = _update_golden_prompts(Path(base_dir), dry_run=False)
    updated = sum(1 for r in results if r.get("updated") == "yes")
    unchanged = sum(1 for r in results if r.get("reason") == "unchanged")
    no_tpl = sum(1 for r in results if r.get("reason") == "no-template")
    return results, updated, unchanged, no_tpl


def _write_current_prompt_hash_to(base_path: Path) -> str:
    """Compute current prompts hash relative to 'prompts' directory and write to base_path."""
    hv = _compute_prompts_hash("prompts")
    _write_prompt_hash(Path(base_path), hv)
    return hv


def _replace_iterations(dest_base: Path, src_iterations: Path) -> None:
    """Replace iterations/<CHAPTER_*> dirs in dest_base with those under src_iterations."""
    import shutil
    src_iterations = Path(src_iterations)
    dest_iterations = Path(dest_base) / "iterations"
    if not src_iterations.exists():
        return
    dest_iterations.mkdir(parents=True, exist_ok=True)
    for ch_dir in sorted([d for d in src_iterations.iterdir() if d.is_dir() and d.name.startswith("CHAPTER_")]):
        target = dest_iterations / ch_dir.name
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        shutil.copytree(ch_dir, target)


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

        # Now that GW_BOOK_BASE_DIR is resolved (env or --book-base), trim logs for that base
        try:
            _init_run_logs()
        except Exception:
            pass

        # Mark the start of a run clearly in run.log once chapter/version are known
        try:
            _log_run(f"=== START RUN === chapter_id={chapter_id} version={version_num}")
        except Exception:
            pass

        # Enable crash tracing early, so breadcrumbs below are captured to base/crash_trace.log
        try:
            import os as _os
            if _os.getenv("GW_CRASH_TRACE", "0") == "1":
                import faulthandler as _faulthandler
                import signal as _signal
                from .env import get_book_base_dir
                crash_log_file = get_book_base_dir() / "crash_trace.log"
                crash_log_file.parent.mkdir(parents=True, exist_ok=True)
                _os.environ["GW_CRASH_TRACE_FILE"] = str(crash_log_file)
                fh = open(crash_log_file, "a", encoding="utf-8")
                _faulthandler.enable(file=fh, all_threads=True)
                for sig in (getattr(_signal, "SIGSEGV", None), getattr(_signal, "SIGABRT", None)):
                    try:
                        if sig is not None:
                            _faulthandler.register(sig, file=fh, all_threads=True, chain=True)
                    except Exception:
                        pass
                _breadcrumb("crash:enabled")
        except Exception:
            pass

        # Build context up-front (allow missing chapter if needed)
        # We need chapter id for versioning; create context even if file is missing
        try:
            ctx = RunContext.from_paths(chapter_path=chapter_path, version=0, allow_missing_chapter=True)
        except Exception:
            ctx = None

        # Run CONTENT_TABLE.yaml brainstorming FIRST if a brainstorm placeholder exists, regardless of chapter arg
        try:
            def _toc_has_brainstorm_placeholder(toc_obj) -> bool:
                try:
                    if not isinstance(toc_obj, list):
                        return False
                    for it in toc_obj:
                        # Map form: check key '???' or value '???' or value starting with 'Brainstorm'
                        if isinstance(it, dict):
                            for k, v in it.items():
                                ks = str(k).strip()
                                vs = str(v).strip() if v is not None else ""
                                if ks == "???" or vs == "???" or vs.lower().startswith("brainstorm"):
                                    return True
                        # String form: contains ???
                        elif isinstance(it, str) and "???" in it:
                            return True
                    return False
                except Exception:
                    return False

            if ctx and isinstance(ctx.content_table, dict):
                _breadcrumb("content_table:check:start")
                toc = ctx.content_table.get("TABLE_OF_CONTENTS") if isinstance(ctx.content_table, dict) else None
                try:
                    _breadcrumb(f"content_table:toc_type={type(toc).__name__} size={len(toc) if isinstance(toc, list) else 'na'}")
                except Exception:
                    pass
                if _toc_has_brainstorm_placeholder(toc):
                    _breadcrumb("content_table:brainstorm:detected")
                    run_content_table_brainstorm(ctx=ctx)
                    return 0
                else:
                    _breadcrumb("content_table:brainstorm:not_detected")
        except Exception:
            pass

        # Branch: CONTENT_TABLE.yaml brainstorming
        if Path(chapter_path).name.upper() == "CONTENT_TABLE.YAML":
            if ctx is None:
                ctx = RunContext.from_paths(chapter_path=chapter_path, version=0, allow_missing_chapter=True)
            run_content_table_brainstorm(ctx=ctx)
            return 0

        # Branch: Chapter brainstorming conditions
        # 1) If chapter file is missing, trigger brainstorming pipeline
        resolved = resolve_chapter_path(chapter_path)
        if not resolved.exists():
            if ctx is None:
                ctx = RunContext.from_paths(chapter_path=chapter_path, version=version_num, allow_missing_chapter=True)
            run_chapter_brainstorm(ctx=ctx, log_llm=bool(ns.log_llm))
            return 0

        # 2) If chapter contains a touch-point with `brainstorming: True`, trigger brainstorming
        try:
            if ctx is None:
                ctx = RunContext.from_paths(chapter_path=str(resolved), version=version_num)
            ch_yaml = ctx.chapter
            if isinstance(ch_yaml, dict):
                tps = ch_yaml.get("Touch-Points") or ch_yaml.get("TouchPoints") or []
                if isinstance(tps, list):
                    for it in tps:
                        if isinstance(it, dict) and it.get("brainstorming") is True:
                            run_chapter_brainstorm(ctx=ctx, log_llm=bool(ns.log_llm))
                            return 0
        except Exception:
            pass

        # Character brainstorming trigger (Task 2): missing actor in chapter setting OR existing character with brainstorming: True
        try:
            resolved2 = resolve_chapter_path(chapter_path)
            if resolved2.exists():
                if ctx is None:
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
                # 2) missing actor referenced in the chapter (top-level setting, a 'setting' touch-point, or 'actors' touch-points)
                if target_name is None:
                    actor_list: list[str] = []
                    # a) Top-level chapter setting
                    chs = ctx.chapter.get("setting") if isinstance(ctx.chapter, dict) else None
                    if isinstance(chs, dict):
                        actors_val = chs.get("actors")
                        if isinstance(actors_val, list):
                            actor_list.extend([str(a) for a in actors_val])
                    # b) Touch-Points: collect actors from any 'setting' or 'actors' entries
                    try:
                        tps = []
                        if isinstance(ctx.chapter, dict):
                            tps = ctx.chapter.get("Touch-Points") or ctx.chapter.get("TouchPoints") or []
                        if isinstance(tps, list):
                            for it in tps:
                                # setting touch-point with actors list
                                if isinstance(it, dict):
                                    st = it.get("setting")
                                    if isinstance(st, dict):
                                        st_actors = st.get("actors")
                                        if isinstance(st_actors, list):
                                            actor_list.extend([str(a) for a in st_actors])
                                # explicit actors touch-point: actors: [..]
                                if isinstance(it, dict):
                                    it_actors = it.get("actors")
                                    if isinstance(it_actors, list):
                                        actor_list.extend([str(a) for a in it_actors])
                    except Exception:
                        pass
                    # Deduplicate while preserving order
                    seen: set[str] = set()
                    dedup_actors: list[str] = []
                    for a in actor_list:
                        tok = _norm_token(a)
                        if tok and tok not in seen:
                            seen.add(tok)
                            dedup_actors.append(a)
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
                    for a in dedup_actors:
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

        # Crash tracing already enabled above if requested

        try:
            run_pipelines_for_chapter(chapter_path, version_num, log_llm=bool(ns.log_llm))
            return 0
        except Exception as e:
            from .context import UserActionRequired as _UAR
            if isinstance(e, _UAR):
                msg = str(e).strip() or "Waiting for user suggestions on first draft."
                print(msg)
                return 0
            raise

    if ns.cmd == "golden-update":
        """Refactored golden-update: perform a full snapshot rebuild into a temp dir then copy back.

        Steps:
          1. Create temporary rebuild destination.
          2. For every chapter CHAPTER_*.yaml under --book-base/chapters run snapshot equivalent of:
             golden-rebuild --chapter-id CHAPTER_N --all-versions --upto-step all (reusing copy_partial_book).
          3. Apply templates in the temp destination to ensure USER prompts match current templates.
          4. If --dry-run: report counts and leave temp directory in place.
             Else: copy rebuilt iteration artifacts back into --book-base (replace iterations/* for processed chapters) and delete temp.
          5. Write/refresh prompt_hash in --book-base.
        """
        import os, tempfile, shutil
        base = ns.book_base or os.getenv("GW_BOOK_BASE_DIR")
        if not base:
            print("Error: --book-base not provided and GW_BOOK_BASE_DIR is not set.")
            return 2
        base_path = Path(base)
        if not base_path.exists():
            print(f"Error: book base does not exist: {base_path}")
            return 2
        # Build temp destination
        tmp_dir = Path(tempfile.mkdtemp(prefix="gw_golden_update_"))
        # Copy roots and snapshot all chapters (all versions, all steps)
        _copy_roots(base_path, tmp_dir)
        processed = _snapshot_copy_all_chapters(
            src_base=base_path,
            dest_base=tmp_dir,
            clear_dest=False,
            upto_step_dir="all",
            upto_filename=None,
            all_versions=True,
            max_version=None,
        )
        # Apply templates (update USER sections) in temp dir
        _, updated, unchanged, no_tpl = _apply_templates_and_counts(tmp_dir)
        dry_run = bool(getattr(ns, "dry_run", False))
        # Compute and write prompt hash for current templates into base (same as old behavior)
        hv = _write_current_prompt_hash_to(base_path)
        if dry_run:
            print(
                "Golden update (dry-run snapshot) complete.\n"
                f"  Base: {base_path}\n"
                f"  Temp rebuild: {tmp_dir}\n"
                f"  Chapters processed: {processed}\n"
                f"  Templates applied: updated={updated}, unchanged={unchanged}, no-template={no_tpl}\n"
                f"  prompt_hash={hv}\n"
                "No files copied back (dry-run)."
            )
        else:
            # Replace iterations/<CHAPTER> directories with rebuilt ones
            _replace_iterations(base_path, tmp_dir / "iterations")
            # Optional cleanup of temp dir
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass
            print(
                "Golden update complete.\n"
                f"  Base: {base_path}\n"
                f"  Chapters processed: {processed}\n"
                f"  Templates applied: updated={updated}, unchanged={unchanged}, no-template={no_tpl}\n"
                f"  prompt_hash={hv}\n"
                "Iterations replaced with rebuilt snapshot."
            )
        return 0

    if ns.cmd == "golden-rebuild":
        # Branch: partial snapshot only (no pipeline execution)
        if getattr(ns, "chapter_id", None) and getattr(ns, "upto_step", None):
            clear = not bool(getattr(ns, "no_clear", False))
            # Determine version semantics:
            # --all-versions with no --version -> copy all pipeline_vN found (uniform cutoff)
            # --version N (without --all-versions) -> copy pipeline_v1..vN (lower versions full, highest uses cutoff)
            # --version N with --all-versions -> copy pipeline_v1..vN (uniform cutoff across all)
            all_versions_flag = bool(getattr(ns, "all_versions", False))
            highest_version = ns.version if ns.version and ns.version > 0 else None
            # Pass effective_version as the highest for copy semantics
            effective_version = highest_version or 1
            _snapshot_copy_chapter(
                src_base=Path(ns.src_base),
                dest_base=Path(ns.dest_base),
                chapter_id=str(ns.chapter_id),
                upto_step_dir=str(ns.upto_step),
                upto_filename=getattr(ns, "upto_filename", None),
                all_versions=all_versions_flag,
                max_version=highest_version if all_versions_flag else None,
                clear_dest=clear,
                effective_version=effective_version,
            )
            # Always apply templates post-copy (flag retained for backward compatibility)
            _, updated, unchanged, no_tpl = _apply_templates_and_counts(Path(ns.dest_base))
            parts = [
                "Golden partial snapshot complete.\n",
                f"  Source: {ns.src_base}\n",
                f"  Destination: {ns.dest_base}\n",
                f"  Chapter: {ns.chapter_id} " + (
                    (
                        (f"pipeline_v1..v{effective_version} (lower versions ALL steps, highest up to {ns.upto_step})")
                        if not all_versions_flag and highest_version else
                        (f"pipeline_v1..v{highest_version} up to {ns.upto_step}" if all_versions_flag and highest_version else "ALL pipeline versions up to {ns.upto_step}")
                    )
                ) + "\n",
                f"  Clear destination: {'no' if ns.no_clear else 'yes'}\n",
            ]
            parts.append(
                f"  Templates applied: updated={updated}, unchanged={unchanged}, no-template={no_tpl}\n"
            )
            parts.append("You can now inspect the partial book state for targeted tests.")
            print("".join(parts))
            return 0
        # Full rebuild path (original behavior)
        # Refactored full rebuild: reuse shared helpers
        clear = not bool(getattr(ns, "no_clear", False))
        dest_base = Path(ns.dest_base)
        src_base = Path(ns.src_base)
        processed = _snapshot_copy_all_chapters(
            src_base=src_base,
            dest_base=dest_base,
            clear_dest=clear,
            upto_step_dir="all",
            upto_filename=None,
            all_versions=True,
            max_version=None,
        )
        # Apply templates across destination
        _, updated, unchanged, no_tpl = _apply_templates_and_counts(dest_base)
        print(
            "Golden rebuild (snapshot mode) complete.\n"
            f"  Source: {ns.src_base}\n"
            f"  Destination: {ns.dest_base}\n"
            f"  Chapters processed: {processed}\n"
            f"  Templates applied: updated={updated}, unchanged={unchanged}, no-template={no_tpl}\n"
            "All pipeline versions and steps copied per chapter."
        )
        return 0

    if ns.cmd == "prompt-hash":
        import os
        hv = _compute_prompts_hash("prompts")
        base = ns.book_base or os.getenv("GW_BOOK_BASE_DIR")
        if base:
            _write_prompt_hash(base, hv)
            print(f"prompt_hash updated at {base}: {hv}")
        else:
            print(hv)
        return 0

    if ns.cmd == "new-book":
        # Create a starter book structure by copying from tutorial/MyFirstBook
        import os, shutil
        dest = Path(getattr(ns, "book_base", "")).expanduser()
        if not str(dest):
            print("Error: You must provide a destination directory for the new book.")
            return 2
        # Check destination state
        if dest.exists():
            if not dest.is_dir():
                print(f"Error: Destination exists and is not a directory: {dest}")
                return 2
            # Directory must be empty
            try:
                if any(dest.iterdir()):
                    print(
                        "Error: Destination directory is not empty. Please specify an empty directory for creating a new book template."
                    )
                    return 2
            except Exception:
                print(
                    "Error: Unable to read destination directory. Choose a different path or fix permissions."
                )
                return 2
        else:
            try:
                dest.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(f"Error: Could not create destination directory: {dest} ({e})")
                return 2

        # Source template location
        src = Path(__file__).resolve().parents[1] / "tutorial" / "MyFirstBook"
        if not src.exists():
            print(
                f"Error: Tutorial source not found at {src}. Please ensure 'tutorial/MyFirstBook' exists in the repository."
            )
            return 2

        # Files to copy
        try:
            # Ensure chapters directory exists
            (dest / "chapters").mkdir(parents=True, exist_ok=True)
            # Copy core files
            for name in ["SETTING.yaml", "CHARACTERS.yaml"]:
                sp = src / name
                if not sp.exists():
                    print(f"Error: Missing template file: {sp}")
                    return 2
                shutil.copy2(sp, dest / name)
            # Copy chapters files
            for rel in ["chapters/CONTENT_TABLE.yaml", "chapters/CHAPTER_001.yaml"]:
                sp = src / rel
                if not sp.exists():
                    print(f"Error: Missing template file: {sp}")
                    return 2
                (dest / Path(rel)).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(sp, dest / Path(rel))
        except Exception as e:
            print(f"Error: Failed to copy template files: {e}")
            return 2

        # Success message and next steps
        print(
            "New book template created.\n"
            f"  Location: {dest}\n"
            "Created files:\n"
            "  - SETTING.yaml\n"
            "  - CHARACTERS.yaml\n"
            "  - chapters/CONTENT_TABLE.yaml\n"
            "  - chapters/CHAPTER_001.yaml\n\n"
            "Next steps:\n"
            "  1) Run: python -m ghostwriter.cli run CHAPTER_001.yaml --book-base "
            f"{dest}\n"
            "  2) Edit the YAML files to make the story yours and run again."
        )
        return 0

    print("No command executed.")
    return 1


if __name__ == "__main__":
    code = main()
    raise SystemExit(code)
