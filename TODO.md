# GhostWriter Project TODO

This document outlines the tasks currently being worked for this project for new feature developent, feature refinement, and testing.

## High Level Tasks

1. [x] Update the music_design.md and TODO.md for 10-measure group feature
    The current implementation tries to prompt the LLM to do the entire 120 (or 150 or 180) measures in one music_first_score_prompt.md request. The LLM seems to be resistant to this request, possibly due to internal anti-repetition utility in training. There is a way we can work around this though: We can do the music_first_score_prompt.md in groups of 10 measures.
    - [x] Rewrite `music_design.md` to reflect the current code base and introduce the grouped flow.
    - [x] Document how successive 10-measure prompts work (motivation, melody slicing, nth block rules, prior-block injection, module updates, artifact/debug plan).
    - [x] Update this TODO with the implementation plan below.

2. Implement the 10-measure grouped first-score flow end to end
    - [ ] Update `prompts/music_first_score_prompt.md` and `ghostwriter.music.prompts` with block-aware placeholders and guidance.
    - [ ] Teach `ghostwriter.music.context` & pipeline helpers to slice melody CSVs, surface prior blocks, and expose progress metadata.
    - [ ] Extend `ghostwriter.music.pipeline` to iterate per voice/variant in 10-measure blocks, manage `music_progress_<variant>.json`, and emit per-block note CSVs.
    - [ ] Persist the required first/last block attempt logs plus per-block scratch artifacts while keeping resume idempotent.
    - [ ] Update resume/export glue, README, and tests (unit + integration) to cover the new grouped behavior.


## Session Summary (Oct 24, 2025)

- Implemented Previous Task: pre-draft user-in-the-loop across narration, dialog, and implicit.
    - Writes `touch_point_first_draft.txt` and `first_suggestions.txt`, then stops gracefully.
    - On resume, reads edited first draft + suggestions and applies `subtle_edit_prompt.md` → `touch_point_draft.txt`; regenerates `suggestions.txt`.
    - Removed per–touch-point `check.txt` generation.
    - Ensured `touch_point_state.json` remains compatible and is written at both phases.
- Added `UserActionRequired` exception to signal human-gated stops and caught it in the driver for graceful exit.
- Updated README with the new flow.

## Session Summary (Nov 10, 2025)

- Completed Slice 1 (Importer & Sanitizer): added `ghostwriter.music.importer` and `ghostwriter.music.sanitizer` utilities, new unit tests, and documented dependencies/workflow touchpoints.
- Completed Slice 2 (Voice Metadata & Context Builder): introduced `ghostwriter.music.context`, emitted prompt-ready payloads, voice/entity matching, and score summarization tests.
- Completed Slice 3 (First-Score Gate Integration): added music prompt templates, `ghostwriter.music.prompts`, first-score pipeline hook in `ghostwriter.chapter`, and unit tests covering gate orchestration.
- Completed Slice 4 (Subtle-Score Refinement): implemented resume subtle pass, feedback ingestion, new suggestions artifacts, importer flatten fix, and updated full-suite tests (`YAML_CEXT_DISABLED=1 python -m pytest -q`).
- Completed Slice 5 (Export & Packaging): aggregated touch-point scores into `score_vN.musiccsv`, rendered score/per-voice MIDI, generated manifests + zip bundles via `ghostwriter.music.exporter`, wired chapter pipeline hook, and added exporter unit coverage.

