from pathlib import Path
import re

from ghostwriter.characters import substitute_character_calls
from ghostwriter.context import RunContext
from ghostwriter.env import get_iterations_dir


def test_substitute_character_calls_warns_on_malformed_blocks(use_lr_book_env, lr_book_dir: Path):
    # Malformed: missing closing </CHARACTER>
    pre = (
        "Before...\n"
        "<CHARACTER>\n"
        "<id>red</id>\n"
        "<agenda>be kind</agenda>\n"
        "<dialog>2</dialog>\n"
        "<prompt>Greet the wolf.</prompt>\n"
        # intentionally no closing tag
        "...After\n"
    )
    ctx = RunContext.from_paths(chapter_path=str(lr_book_dir / "chapters/CHAPTER_001.yaml"), version=1)
    # Use a predictable log_dir under the sandboxed book base
    log_dir = get_iterations_dir() / ctx.chapter_id / "pipeline_v1" / "00_test"
    out, stats = substitute_character_calls(pre, ctx=ctx, log_dir=log_dir)
    assert out.startswith("Before...") and "Greet the wolf" in out  # unchanged, no substitution
    # Check run.log contains the warning
    run_log = lr_book_dir / "run.log"
    assert run_log.exists()
    text = run_log.read_text(encoding="utf-8")
    assert re.search(r"WARNING: .*Malformed CHARACTER block", text)
