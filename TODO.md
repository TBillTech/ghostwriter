# GhostWriter Project TODO

This document outlines the tasks currently being worked for this project for new feature developent, feature refinement, and testing.

## High Level Tasks

1. **music track output feature design and Tasks***
    Requirements for the new music track output feature: See MUSIC_REQUIREMENTS.md
    Initial design tasks:
    - [x] Write a `music_design.md` design document with the following sections.
    - [x] Add a high level feature story description to `music_design.md`, including user-in-the-loop workflow.
    - [x] Add details on which steps of the workflow will be performed by prompting an LLM using a filled-in prompting template.
    - [x] Add details on MIDI input and output formats to `music_design.md`.
    - [x] Add details on what Python modules will be created to `music_design.md`.
    - [x] Add details on what tests will be executed against Python modules to `music_design.md`.
    - [x] Based on `music_design.md`, create high level tasks for each vertical slice of work and insert into this TODO.md.

2. [x] **Slice 1 — Importer & Sanitizer**: MIDI ingestion → MusicCSV normalization and monitor MIDI generation.

3. [x] **Slice 2 — Voice Metadata & Context Builder**: Chapter YAML parsing, voice-to-entity mapping, tempo/key summaries.

4. [x] **Slice 3 — First-Score Gate Integration**: LLM prompt templates, `touch_point_first_score.musiccsv` gate, suggestion handling.

5. [x] **Slice 4 — Subtle-Score Refinement**: Resume flow, `touch_point_score.musiccsv`, `score_suggestions.txt`, reconciliation.

6. [x] **Slice 5 — Export & Packaging**: `score_vN.musiccsv`, final/per-voice MIDI renders, manifest and zip bundle.

7. **Implement musicCSV format**
    musicXML has limitations that we wish to exceed. Therefore, we have developed the musicCSV format.  Detailed requirements are desribed in MusicCSVSpecification.md. When implementing tasks below, including the tasks from 7.A to 7.E below, use the MusicCSVSpecification.md as a reference.

    ### 📦 Module name suggestion
    `musiccsv`

    ### 🧱 Core functionality

    #### 7.A. File I/O
    - [x] Implement `read_musiccsv(path)` → loads all CSVs and JSON into a structured object.
    - [x] Implement `write_musiccsv(path, data)` → saves from object back to files.
    - [x] Add pretty summary / printing and reading for LLM and human-in-the-loop I/O.
    - [x] Unit test which proves the invariant: read_musiccsv(write_musiccsv(pretty_data)) == pretty_data, and the inverse test.
    - [x] Implement support for reading/writing `.musiccsv` (zip archive).

    #### 7.B. MIDI conversion
        - [x] Implement `from_midi(mid_path)` → returns `MusicCSV` object.
        - [x] Implement `to_midi(data, out_path)` → writes `.mid`.
        - [x] Handle channel/program mapping from `tracks.csv`.
        - [x] Translate tempo and time signatures from `measures.csv` to MIDI meta events.
        - [x] Unit test which proves the invariant: from_midi(to_midi(pretty_data)) == pretty_data, and the inverse test.

    #### 7.C. Utility / Validation
        - [x] Validate schema compliance (required columns, data types).
        - [x] Resolve derived fields (e.g. `beat` → absolute tick).
        - [x] Optional: Provide JSON Schema for validation of `metadata.json`.

    #### 7.D. API design
    ```python
    from musiccsv import MusicCSV

    score = MusicCSV.from_midi("input.mid")
    score.write("output.musiccsv")
    new_mid = MusicCSV.read("output.musiccsv").to_midi("reexport.mid")
    ```
        - [x] Ensure class-level helpers (`read`, `write`, `from_midi`, `to_midi`) support chaining.
        - [x] Add unit test covering the documented API flow.

    #### 7.E. Additional feature
        - [x] Add support for grace notes, repeats, tuplets.

8. ** Replace musicXML with musicCSV **
    - [x] Replace all score outputs (formerly `.musicxml` artifacts such as `touch_point_first_score.musicxml`, `touch_point_score.musicxml`, `score_vN.musicxml`) with MusicCSV equivalents and keep filenames predictable (e.g., `.musiccsv` + zipped bundle).
    - [x] Update exporter/sanitizer to call `MusicCSV.to_midi` internally and regenerate `monitor.mid` / per-voice MIDIs without MusicXML intermediates.
    - [x] Rewrite pipeline prompts and templates to reference `musiccsv_format_prompt.txt`, ensuring LLM instructions match the new format.
    - [x] Remove MusicXML-specific modules, schemas, and tests that are superseded by MusicCSV to avoid dual-maintenance paths.
    - [x] Migrate integration tests to assert against MusicCSV outputs and adjust fixtures accordingly.

9. ** Refactor to make one song for a music touch-point **
    Upon further reflection, songs may or may not be 1 to 1 with touch points. Plus, all the files building up in the same directory are getting messy.  Let's refactor the program to do something a little more controlled and independent for music tracks.  Let's do the music if and only if there is a music touch-point in the chapter.  Let's do the following:
    - [x] Do the music touch-point process if and only if the touch-point is exactly music. This means, music won't be done in narration, dialog, or mixed anymore.
    - [x] Restructure the music and voices in the yaml so that the music touch-point looks like this example:
        - music: 
            - title: "The Forest Path"
            - description: a cinematic song; fast tempo, haunting melody, steady bass, rapid percussion. Target typical song length of 3 minutes.
            - voices: [major.tenor.base.wolf.harmony, major.alto.flute.red.melody, major.beat.drum.axe, major.bass.soundscape.forest_path]
    - [x] Make sure that most recent paragraph is avialable when substituting in to music prompt templates.

10. ** Break music generation down into multiple steps **
    The music previously generated was naive and kind of random. We can do better by focusing on building more thought out and recursive melodies. Overall, the algorithm for a `music` touch-point breaks down like this:
    - [x] Prompt the LLM with the previous paragraph, story-relative summary, and factoids, and ask it to generate `metadata.json` and `tracks.csv` using `music_metadata_tracks_prompt.md`. These artifacts are stored alongside the prompt + response log (`metadatatracks.txt`).
    - [x] Prompt the LLM with `music_melody_edges_prompt.md`, which includes the previous paragraph, story-relative and factoids blocks, `metadata.json`, and the `melody_elements_instructions.txt` template. Write the raw dwell-notes + melodic-edges description to `melody_edges.txt` and pause for human editing before continuing.
    - [x] Construct a melody using `music_melody_prompt.md`. This prompt includes context (previous paragraph, story-relative, factoids, touch-point title/description, metadata JSON) and an expanded `melody_instructions.txt` where `[dwell_notes]`, `[melodic_edges]`, and `[additional_rules]` have been substituted.
    - [x] Repeat the melody construction three times by calling `run_melody_construction_step` with variants:
        - `standard` — no extra variant rules.
        - `complimentary` — injects `additional_rules` that describe inverting dwell weights to create a complimentary line.
        - `reprise` — injects `additional_rules` that describe stretching edges by another measure and filling with additional notes.
      Each run writes `melody_{variant}.txt` (prompt + response log) and `melody_{variant}.csv`.
    Music construction instructions (to be repeated for each type of melody):
    - [x] Construct a melody by prompting the LLM with the `melody_instructions.txt` template (after substituting dwell notes, melodic edges, and any variant-specific `additional_rules`).
    - [x] Save the melody artifact from the output of the prompt as `melody_standard.csv`, `melody_complimentary.csv`, or `melody_reprise.csv` under the touch-point directory.
    - [X] Generate `measures.csv` from the melody artifact (ideally in Python without an LLM in the loop). The `measures.csv` is the summary csv of measures that is needed to complete the musiccsv format.
    - [ ] For each voice in the music touch-point, implement a multi-voice first-score pipeline without extra user feedback:
        - [ ] **Prompt design** (moved to Task 11)
            - [X] Modify `music_first_score_prompt.md` and its builder so each call focuses on exactly one voice track to be generated.
            - [X] Add support for passing full, reduced-note grids (measure, beat, pitch, duration) for melody and already-scored voices into the prompt.
            - [X] Extend the prompt instructions so the model:
                - [X] Treats the reduced-note grids as the authoritative “music so far” timeline.
                - [X] Writes only the current voice’s `notes.csv` rows, assuming shared `metadata.json` and `measures.csv`.
        - [ ] **Pipeline flow from an empty music directory (v1)** (moved to Task 12)
            - [ ] Ensure `run_pipelines_for_chapter` performs the following for a fresh music touch-point:
                - [X] Step 1: generate `metadata.json`, `tracks.csv`, `metadatatracks.txt`, and `music_touch_point.json`.
                - [X] Step 2: generate `melodyelements.txt` and `melody_edges.txt`, then pause for user review/edit of `melody_edges.txt`.
                - [X] Step 3: on restart, generate all `melody_*.csv` and `measures_*.csv` without a pause.
                - [ ] Step 4: immediately call a per-voice first-score pipeline (no user feedback at this stage) that:
                    - [ ] For the melody voice, uses the standard melody variant and produces a complete `notes_melody_standard.csv`.
                    - [ ] For other voices, uses the full reduced melody grid and any existing reduced-note grids to write harmonized parts, each to its own `notes_<voice>_<variant>.csv` .
                    - [ ] Accumulates “music so far” by including all previously generated voices in later prompts.
        - [ ] **Aggregation and exports** (moved to Task 13)
            - [ ] Combine the `metadata.json`, `tracks.csv`, the appropriate `measures_*.csv`, and all `notes_<voice>_<variant>.csv` merged to form `first_<title>_<variant>.musiccsv` and `first_monitor_<title>_<variant>.mid` for:
                - [ ] Standard melody variant.
                - [ ] Complimentary melody variant.
                - [ ] Reprise melody variant.
            - [ ] Ensure `finalize_music_exports` (or a new helper) writes `first_monitor_<title>_<variant>.mid` for each `first_<title>_<variant>.musiccsv`.
        - [ ] **Check and feedback loop** (moved to Tasks 14–15)
            - [ ] Automatically run `music_check_prompt.md` against each `first_<title>_<variant>.musiccsv`, writing variant-specific suggestions files (e.g., `first_score_suggestions_standard.txt`).
            - [ ] Do **not** pause for user input at this stage; allow v1 to complete with first-pass scores and suggestions.
            - [ ] On the next run (final v1), detect the presence of first score suggestions and:
                - [ ] Re-use per-voice first-score pipeline to re-generate all voices for each variant, but this time including and incorporating user-edited suggestions.
                - [ ] Outputs of this loop should be `<title>_<variant>.musiccsv` and `monitor_<title>_<variant>.mid` completion as the final artifact for that chapter version.
    - [x] When the `pipeline_vN` run is complete, similar to how `final.txt` is collated and regenerated, create a `sound_track` directory. Copy each of the `<title>.mid` (or `<title>_<variant>.mid`) files from the `pipeline_vN` music touch-points, using the normalized title from the music touch-point as the file name.

11. **Per-voice / per-variant prompt and artifact model**
    - [x] Define the canonical artifact naming for the refactored pipeline:
        - [x] Per-voice, per-variant note CSVs: `notes_<voice_token>_<variant>.csv`.
        - [x] Per-variant assembled first-pass scores: `first_<title>_<variant>.musiccsv`.
        - [x] Per-variant monitor MIDIs: `first_monitor_<title>_<variant>.mid`.
        - [x] Final refined scores: `<title>_<variant>.musiccsv`.
        - [x] Final monitor MIDIs: `monitor_<title>_<variant>.mid`.
    - [x] Update `music_design.md` and any inline comments to reflect the per-voice/per-variant artifact layout.
    - [x] Confirm that the prompt-level requirements in Task 10 (**Prompt design** subtasks) are satisfied and reference this artifact model. (Task 10 prompt-design subtasks moved here.)

12. **Multi-voice, multi-variant first-pass composer (Step 4)**
    - [x] Introduce a new helper in `ghostwriter/music/pipeline.py` that:
        - [x] Takes `VoiceContext`, `prompt_payload`, and a target variant (standard/complimentary/reprise).
        - [x] Iterates over `voice_context.voices` in a stable order.
        - [x] For each voice, builds music-so-far reduced-note grids from any previously written `notes_<voice_token>_<variant>.csv` (and/or normalized MusicCSV) and calls the per-voice first-pass prompt.
        - [x] Writes `notes_<voice_token>_<variant>.csv` for the new voice and keeps the shared `metadata.json` + `measures_*.csv` untouched.
    - [x] Refactor or wrap `ensure_first_score_gate` so that it becomes an internal per-voice composer used by this new helper, rather than writing a single `touch_point_first_score.musiccsv`.
    - [x] Ensure the helper can be called three times (for standard/complimentary/reprise) without clobbering per-voice artifacts.
    - [x] Update Task 10 “Pipeline flow from an empty music directory (v1)” subtasks to reference this helper instead of the legacy monolithic gate. (Subtasks moved conceptually here.)

13. **Aggregation and exports for first-pass per variant**
    - [x] Implement a helper (or extend `finalize_music_exports`) to:
        - [x] Load `metadata.json`, `tracks.csv`, and `measures_<variant>.csv`.
        - [x] Merge all `notes_<voice_token>_<variant>.csv` into a single in-memory `MusicCSV` object per variant.
        - [x] Write `first_<title>_<variant>.musiccsv` for each of standard, complimentary, and reprise.
        - [x] Render `first_monitor_<title>_<variant>.mid` via `MusicCSV.to_midi`.
    - [x] Ensure this aggregation step does not require additional LLM calls and can be repeated idempotently.
    - [x] Mark Task 10 “Aggregation and exports” subtasks as implemented via this helper. (Task 10 aggregation subtasks moved here.)
    - [x] Note: this helper should **not** be used as the primary location for applying LLM-driven musical suggestions; it is a pure assembly/export step.

14. **Chapter pipeline integration: non-pausing v1 flow**
    - [x] Update the music branch in `run_pipelines_for_chapter` so that, after all `melody_*.csv` and `measures_*.csv` are present:
        - [x] It calls the multi-voice first-pass composer (Task 12) for each variant **without** incorporating suggestion text yet.
        - [x] It invokes the aggregation/export helper (Task 13) to produce `first_<title>_<variant>.musiccsv` and `first_monitor_<title>_<variant>.mid`.
    - [x] Remove the legacy `ensure_first_score_gate`-driven “Music touch-point still awaiting refinement…” pause from the v1 path.
    - [x] Ensure that v1 runs to completion for a music touch-point, leaving all first-pass per-variant artifacts and suggestions in place without HIL pauses.
    - [x] Keep subtle-edit/v2 integration stubs in place but no longer block v1 on them.
    - [x] Mark Task 10 “Pipeline flow from an empty music directory (v1) Step 4” as implemented here. (Subtasks moved.)

15. **Variant checks and v2 refinement loop**
    - [x] Extend `music_check_prompt.md` usage so that:
        - [x] It runs against each `first_<title>_<variant>.musiccsv`.
        - [x] Writes variant-specific suggestion files (e.g., `first_score_suggestions_standard.txt`).
    - [x] Ensure these checks are invoked from the chapter pipeline after first-pass aggregation, but do **not** raise `UserActionRequired` for v1; they should be best-effort diagnostics.
    - [x] Design the v2 subtle-edit flow for music so that:
        - [x] The presence of first-pass suggestions triggers a **second-pass** call to the multi-voice, multi-variant composer (Task 12), running at the per-voice, per-variant level rather than at the aggregated full-score level.
        - [x] Feed suggestion text into these second-pass prompts as additional conditioning, alongside the existing reduced-note grids and context.
        - [x] The outputs of this second pass are `<title>_<variant>.musiccsv` and `monitor_<title>_<variant>.mid` per variant, assembled via the same aggregation/export helper as first-pass.
    - [x] Mark Task 10 “Check and feedback loop” subtasks as implemented via this v2 flow. (Subtasks moved here.)

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

