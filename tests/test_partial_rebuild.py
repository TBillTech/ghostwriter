import json
import os
from pathlib import Path

from ghostwriter.cli import main


def _resolve_dest(tmp_path: Path) -> Path:
    """Allow overriding tmp destination via env GW_TEST_PARTIAL_DEST for easier manual inspection."""
    override = os.getenv("GW_TEST_PARTIAL_DEST")
    if override:
        p = Path(override)
        p.mkdir(parents=True, exist_ok=True)
        return p
    return tmp_path / "partial_book"


def test_golden_partial_rebuild_setting(tmp_path, lr_source_dir):
    """Verify partial golden-rebuild creates pipeline_v1 and 01_setting with expected files.

    Env override GW_TEST_PARTIAL_DEST lets a developer inspect the copied snapshot after the test.
    Run with `pytest -s` to see printed paths and step listing.
    """
    dest = _resolve_dest(tmp_path)
    args = [
        "golden-rebuild",
        "--src-base", str(lr_source_dir),
        "--dest-base", str(dest),
        "--chapter-id", "CHAPTER_001",
        "--version", "1",
        "--upto-step", "05_narration",
        "--no-clear",
    ]
    code = main(args)
    assert code == 0, "CLI should exit successfully for partial rebuild"

    # Paths
    setting_dir = dest / "iterations" / "CHAPTER_001" / "pipeline_v1" / "01_setting"
    assert setting_dir.exists(), f"Expected setting dir: {setting_dir}"

    draft_file = setting_dir / "touch_point_draft.txt"
    state_file = setting_dir / "touch_point_state.json"
    assert draft_file.exists(), "touch_point_draft.txt should be copied in partial snapshot"
    assert state_file.exists(), "touch_point_state.json should be copied in partial snapshot"

    data = json.loads(state_file.read_text(encoding="utf-8"))
    # Minimal structural assertions
    assert data.get("id") == "1"
    assert data.get("type") == "setting"
    assert isinstance(data.get("setting_block"), str) and data.get("setting_block"), "setting_block should be a non-empty string"
    # Ensure no dialog appended yet
    assert data.get("appended_dialog") == {}, "appended_dialog should be empty at initial setting step"

    # Developer visibility: print destination and top-level structure (requires -s to show)
    print(f"[partial-rebuild] destination={dest}")
    pipeline_root = dest / "iterations" / "CHAPTER_001" / "pipeline_v1"
    if pipeline_root.exists():
        steps = sorted([d.name for d in pipeline_root.iterdir() if d.is_dir()])
        print(f"[partial-rebuild] steps_present={steps}")


def test_golden_partial_rebuild_version2_upto_narration(tmp_path, lr_source_dir):
    """Verify partial snapshot for --version 2 includes v1 (ALL steps) and v2 up to 05_narration; root artifacts copied.

    Ensures pipeline_v3 is NOT copied when --version 2 is provided without --all-versions.
    """
    dest = tmp_path / "partial_v2_upto5"
    args = [
        "golden-rebuild",
        "--src-base", str(lr_source_dir),
        "--dest-base", str(dest),
        "--chapter-id", "CHAPTER_001",
        "--version", "2",
        "--upto-step", "05_narration",
    ]
    code = main(args)
    assert code == 0

    iter_root = dest / "iterations" / "CHAPTER_001"
    # Root artifacts
    for fname in ["draft_v1.txt", "draft_v2.txt", "final.txt", "story_so_far.txt", "story_relative_to.txt"]:
        p = iter_root / fname
        assert p.exists(), f"Expected root artifact {fname}"

    # pipeline_v1 and pipeline_v2 should exist (v1..v2 inclusive); no pipeline_v3
    assert (iter_root / "pipeline_v1").exists(), "pipeline_v1 should be copied for inclusive range"
    assert (iter_root / "pipeline_v2").exists(), "pipeline_v2 should be copied"
    assert not (iter_root / "pipeline_v3").exists(), "pipeline_v3 should not be copied for version-specific snapshot"

    src_pipe_v2 = lr_source_dir / "iterations" / "CHAPTER_001" / "pipeline_v2"
    assert src_pipe_v2.exists(), "Source pipeline_v2 must exist in golden data"
    # Determine expected step dirs <=5 and >5 from source
    src_steps = [d.name for d in src_pipe_v2.iterdir() if d.is_dir()]
    le5 = [d for d in src_steps if d.split("_",1)[0].isdigit() and int(d.split("_",1)[0]) <= 5]
    gt5 = [d for d in src_steps if d.split("_",1)[0].isdigit() and int(d.split("_",1)[0]) > 5]

    dest_pipe_v2 = iter_root / "pipeline_v2"
    present = [d.name for d in dest_pipe_v2.iterdir() if d.is_dir()]
    for d in le5:
        assert d in present, f"Expected step {d} to be copied"
    for d in gt5:
        assert d not in present, f"Did not expect step {d} (beyond upto-step)"

    # For pipeline_v1 all steps should be present; also verify some files exist in representative steps
    dest_pipe_v1 = iter_root / "pipeline_v1"
    src_pipe_v1 = lr_source_dir / "iterations" / "CHAPTER_001" / "pipeline_v1"
    assert dest_pipe_v1.exists() and src_pipe_v1.exists()
    src_steps_v1 = sorted([d for d in src_pipe_v1.iterdir() if d.is_dir()])
    dest_steps_v1 = {d.name for d in dest_pipe_v1.iterdir() if d.is_dir()}
    for s in src_steps_v1:
        assert s.name in dest_steps_v1, f"Expected all steps in pipeline_v1 including {s.name}"
    # File-level checks
    # v2/05_narration expected files
    for fname in ["touch_point_state.json", "touch_point_draft.txt", "check.txt", "05_subtle_edit.txt"]:
        assert (dest_pipe_v2 / "05_narration" / fname).exists(), f"Missing file in v2 upto-step: {fname}"
    # v1/04_foreshadowing expected files
    for fname in ["touch_point_state.json", "touch_point_draft.txt"]:
        assert (dest_pipe_v1 / "04_foreshadowing" / fname).exists(), f"Missing file in v1 full copy: {fname}"


def test_golden_partial_rebuild_all_versions_all_steps(tmp_path, lr_source_dir):
    """Verify snapshot with --all-versions and --upto-step all copies every pipeline_vN and all steps."""
    dest = tmp_path / "partial_all_versions"
    args = [
        "golden-rebuild",
        "--src-base", str(lr_source_dir),
        "--dest-base", str(dest),
        "--chapter-id", "CHAPTER_001",
        "--upto-step", "all",
        "--all-versions",
    ]
    code = main(args)
    assert code == 0

    iter_root = dest / "iterations" / "CHAPTER_001"
    for fname in ["draft_v1.txt", "draft_v2.txt", "final.txt", "story_so_far.txt", "story_relative_to.txt"]:
        assert (iter_root / fname).exists(), f"Expected root artifact {fname}"

    # Collect all pipeline dirs from source
    src_iter_root = lr_source_dir / "iterations" / "CHAPTER_001"
    src_pipes = sorted([p for p in src_iter_root.iterdir() if p.is_dir() and p.name.startswith("pipeline_v")], key=lambda d: int(d.name.split("_v")[-1]))
    for pipe in src_pipes:
        dest_pipe = iter_root / pipe.name
        assert dest_pipe.exists(), f"Expected pipeline directory {pipe.name}"
        # All step dirs should be present
        src_steps = sorted([d for d in pipe.iterdir() if d.is_dir()])
        dest_steps = {d.name for d in dest_pipe.iterdir() if d.is_dir()}
        for s in src_steps:
            assert s.name in dest_steps, f"Expected step {s.name} in {pipe.name}"
        # Spot-check files in a couple of steps for existence parity
        check_pairs = [
            ("05_narration", ["touch_point_state.json", "touch_point_draft.txt"]),
            ("04_foreshadowing", ["touch_point_state.json", "touch_point_draft.txt"]),
        ]
        for step_name, files in check_pairs:
            sp = dest_pipe / step_name
            if sp.exists():
                for fname in files:
                    assert (sp / fname).exists(), f"Expected {fname} in {pipe.name}/{step_name}"

    # No extraneous pipeline directories
    dest_extra = [d.name for d in iter_root.iterdir() if d.is_dir() and d.name.startswith("pipeline_v") and d.name not in {p.name for p in src_pipes}]
    assert not dest_extra, f"Unexpected extra pipeline directories: {dest_extra}"
