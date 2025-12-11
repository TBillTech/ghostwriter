# GhostWriter Project TODO

This document outlines the tasks currently being worked for this project for new feature developent, feature refinement, and testing.

## High Level Tasks

### Import-driven melody workflow (Dec 2025)

- [ ] Detect per-voice import folders (`<tp_index>_track_<voice>_import/`) when running the music pipeline. Ignore `*.Zone.Identifier` sidecar files and map each folder to the matching chapter voice so imported MIDI content is available before prompting.
- [ ] When an import exists for the melody voice, skip `music_melody_emotion_prompt.md` and `music_emotion_chord_prompt.md`. Instead, run a new `music_import_sanitize_prompt.md` that consumes the sanitized `score.musiccsv` (or freshly generated `import.musiccsv`), quantizes timestamps, trims leading/trailing empty measures, and emits CORE-ready rows (`measure,beat,pitch,duration`). The pipeline will enforce `semi_tones=(0)` and derives the `transition` column directly from the provided pitch values so the dwell/root logic follows the imported contour.
- [ ] Feed the sanitized import output directly into `CORE_MELODY.csv` generation, ensuring the rest of the pipeline (melody edges, reduced grids, score prompts) treat the imported melody as authoritative rather than re-synthesizing it.
- [ ] Add regression coverage for (a) Zone.Identifier filtering, (b) import detection across melody/bass/beat voices, and (c) the new prompt path so future refactors do not regress the import workflow.


    
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

