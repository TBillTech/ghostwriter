import re
import ast
import sys
import time
import json
import random
import hashlib
import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable, Any

from dotenv import load_dotenv
import os
import threading
import signal
import faulthandler
try:
    import resource  # POSIX resource usage
except Exception:  # pragma: no cover
    resource = None  # type: ignore
# Force PyYAML to pure-Python mode by blocking the C extension (_yaml)
try:
    # Prevent yaml from importing the C accelerator to avoid rare native crashes
    sys.modules.setdefault("_yaml", None)
except Exception:
    pass

from ghostwriter.openai import llm_complete, get_model, with_backoff

"""
Use error types and context utilities from the ghostwriter library.
"""
from ghostwriter.context import (
    GWError,
    MissingFileError,
    InvalidYAMLError,
    RunContext,
    chapter_id_from_path,
)
from ghostwriter.utils import to_text as _gw_to_text, save_text, read_file, load_env, _norm_token, _extract_numeric_hint
from ghostwriter.templates import (
    apply_template,
    iter_dir_for,
    list_versions,
    get_latest_version,
    read_latest,
    build_common_replacements,
    build_master_initial_prompt,
    build_master_prompt,
    build_polish_prompt,
    build_check_prompt,
    build_story_so_far_prompt,
    build_story_relative_to_prompt,
)
from ghostwriter.logging import breadcrumb as _breadcrumb, log_warning as _log_warning, crash_trace_file as _crash_trace_file
from ghostwriter.touch_point import (
    validate_text,
    validate_bullet_list,
    validate_actor_list,
    validate_agenda_list,
    _retry_with_validation,
    _extract_bullet_contents,
    _rebuild_bullets,
    _filter_narration_brainstorm,
    _truncate_brainstorm,
    _brainstorm_has_done,
    _strip_trailing_done,
    _ensure_brainstorm_done_on_resume,
    _inline_body_with_dialog,
    _apply_body_template,
    _parse_agenda_by_actor,
    _pick_bullet_by_index,
    _build_pipeline_replacements,
    _apply_step,
    _env_for,
    _env_for_prompt,
    _llm_call_with_validation,
    _polish_snippet,
    _parse_actor_lines,
)
from ghostwriter.chapter import (
    format_draft_record,
    write_draft_records,
    read_draft_records,
    format_suggestions_record,
    write_suggestions_records,
    read_suggestions_records,
    _generate_final_txt_from_records,
    check_iteration_complete,
    generate_story_so_far_and_relative,
    parse_touchpoints_from_chapter,
    ChapterState,
    run_pipelines_for_chapter,
    generate_pre_draft,
    polish_prose,
    verify_predraft,
)
from ghostwriter.factoids import (
    merge_setting_with_factoids as _merge_setting_with_factoids,
    factoids_block_from_setting as _factoids_block_from_setting,
    selected_factoid_names_from_sources as _selected_factoid_names_from_sources,
)
from ghostwriter.env import (
    env_int as _env_int,
    env_str as _env_str,
    env_float as _env_float,
    resolve_temp as _resolve_temp,
    resolve_max_tokens as _resolve_max_tokens,
    mask_env_value as _mask_env_value,
    normalize_base_url_from_env as _normalize_base_url_from_env,
    collect_program_env_snapshot as _collect_program_env_snapshot,
    get_setting_path as _get_setting_path,
    get_characters_path as _get_characters_path,
    resolve_chapter_path as _resolve_chapter_path,
)

from ghostwriter.touch_point import ValidationError

# Prefer library character accessors
from ghostwriter.characters import load_characters_list


# ---- Output validators (Task 3)

# ---------------------------
# Pipeline execution (Task 2)
# ---------------------------


from typing import Callable
# Prefer new package pipelines for execution
try:
    from ghostwriter.pipelines import (
        run_narration_pipeline as gw_run_narration_pipeline,
        run_dialog_pipeline as gw_run_dialog_pipeline,
        run_implicit_pipeline as gw_run_implicit_pipeline,
        run_subtle_edit_pipeline as gw_run_subtle_edit_pipeline,
    )
except Exception:
    gw_run_narration_pipeline = None  # type: ignore
    gw_run_dialog_pipeline = None  # type: ignore
    gw_run_implicit_pipeline = None  # type: ignore
    gw_run_subtle_edit_pipeline = None  # type: ignore

# Begin migration imports (Tasks 5–7)
try:
    from ghostwriter.artifacts import (
        format_draft_record as _gw_format_draft_record,
        write_records as _gw_write_records,
        read_records as _gw_read_records,
    )
    from ghostwriter.resume import scan_completed as _gw_scan_completed
except Exception:
    # Fallback to local implementations during transition
    _gw_format_draft_record = None  # type: ignore
    _gw_write_records = None  # type: ignore
    _gw_read_records = None  # type: ignore
    _gw_scan_completed = None  # type: ignore



# ---------------------------
# Character template parsing and substitution
# ---------------------------

from ghostwriter.characters import parse_character_blocks, render_character_call, substitute_character_calls

## numeric hint extractor moved to ghostwriter.utils._extract_numeric_hint


# ---------------------------
# CLI Entrypoint
# ---------------------------

def main():
    load_env()
    if len(sys.argv) < 2:
        print("Usage: python scripts/driver.py chapters/CHAPTER_xx.yaml [vN or auto] [--show-dialog] [--log-llm]")
        sys.exit(1)

    args = [a for a in sys.argv[1:]]
    log_llm = False
    # Backcompat flag
    if "--show-dialog" in args:
        log_llm = True
        args.remove("--show-dialog")
    if "--log-llm" in args:
        log_llm = True
        args.remove("--log-llm")

    if not args:
        print("Usage: python scripts/driver.py chapters/CHAPTER_xx.yaml [vN or auto] [--show-dialog] [--log-llm]")
        sys.exit(1)

    chapter_path = args[0]
    chapter_id = chapter_id_from_path(chapter_path)

    # Optional: enable crash tracing
    crash_trace = os.getenv("GW_CRASH_TRACE", "0") == "1"

    # Determine version number
    version_num: Optional[int] = None
    if len(args) > 1 and args[1].startswith("v"):
        try:
            version_num = int(args[1][1:])
        except ValueError:
            print("Invalid version format. Use v1, v2, ...")
            sys.exit(1)
    elif len(args) > 1 and args[1] == "auto":
        version_num = get_latest_version(chapter_id) + 1
    else:
        version_num = get_latest_version(chapter_id) + 1

    # Validate required inputs early and ensure iteration directory exists
    try:
        _validate_inputs_and_prepare(chapter_path)
    except GWError as e:
        print(f"Error: {e}")
        sys.exit(2)

    # Setup crash tracing/stack dumps if requested
    if crash_trace:
        try:
            p = iter_dir_for(chapter_id)
            crash_log_dir = p / f"pipeline_v{version_num}"
            crash_log_dir.mkdir(parents=True, exist_ok=True)
            crash_log_file = crash_log_dir / "crash_trace.log"
            os.environ["GW_CRASH_TRACE_FILE"] = str(crash_log_file)
            fh = open(crash_log_file, "a", encoding="utf-8")
            faulthandler.enable(file=fh, all_threads=True)
            for sig in (signal.SIGSEGV, signal.SIGABRT):
                try:
                    faulthandler.register(sig, file=fh, all_threads=True, chain=True)
                except Exception:
                    pass
            _breadcrumb("main:crash-trace-enabled")
        except Exception:
            pass

    # Execute the deterministic pipeline once for vN
    run_pipelines_for_chapter(chapter_path, version_num, log_llm=log_llm)

# ---------------------------
# Validation & Setup (Task 8)
# ---------------------------

from ghostwriter.config import REQUIRED_PROMPTS as _REQUIRED_PROMPTS

def _validate_inputs_and_prepare(chapter_path: str) -> None:
    """Validate presence of required files and create iteration directory.

    Checks:
    - SETTING.yaml exists and parses as YAML
    - Chapter file exists and parses as YAML
    - Required prompt templates exist
    - Creates iterations/<CHAPTER_ID>/ directory
    """
    # Validate chapter file early for clearer errors before any work
    # Resolve chapter path using environment-aware resolver
    ch_path = _resolve_chapter_path(chapter_path)
    if not ch_path.exists():
        raise MissingFileError(
            f"Chapter file not found: {chapter_path}. Resolved to '{ch_path}'."
        )
    # Do not parse YAML here—RunContext is the single source of YAML loading.
    # Just check required files exist for clearer early errors.
    if not _get_setting_path().exists():
        raise MissingFileError(f"Missing required SETTING.yaml: {_get_setting_path()}")
    if not _get_characters_path().exists():
        raise MissingFileError(f"Missing required CHARACTERS.yaml: {_get_characters_path()}")

    # Validate required prompt templates
    missing = [p for p in _REQUIRED_PROMPTS if not Path(p).exists()]
    if missing:
        joined = "\n  - " + "\n  - ".join(missing)
        raise MissingFileError(
            "Missing required prompt template(s):" + joined +
            "\nPlease add these files under the 'prompts/' directory. See README.md for details."
        )

    # Character dialog prompt is optional (we fallback to a built-in template)

    # Ensure iterations directory exists for this chapter
    chapter_id = chapter_id_from_path(str(ch_path))
    out_dir = iter_dir_for(chapter_id)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise GWError(f"Unable to create iterations directory '{out_dir}': {e}")

if __name__ == "__main__":
    # Prefer the new CLI entrypoint for argument parsing and consistency.
    try:
        from ghostwriter.cli import main as cli_main
        raise SystemExit(cli_main())
    except Exception:
        # Fallback to legacy main() if CLI import fails
        main()