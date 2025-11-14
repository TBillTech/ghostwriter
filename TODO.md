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
    - [ ] Do the music touch-point process if and only if the touch-point is exactly music. This means, music won't be done in narration, dialog, or mixed anymore.
    - [ ] Restructure the music and voices in the yaml so that the music touch-point looks like this example:
        - music: 
            - title: "The Forest Path"
            - description: a cinematic song; fast tempo, haunting melody, steady bass, rapid percussion. Target typical song length of 3 minutes.
            - voices: [major.tenor.base.wolf.harmony, major.alto.flute.red.melody, major.beat.drum.axe, major.bass.soundscape.forest_path]
    - [ ] Make sure that most recent paragraph is avialable when substituting in to music prompt templates.

10. ** Break music generation down into multiple steps **
    The music previously generated was naive and kind of random. I think we can do better, by focusing on building more thought out and recursive melodies. I have an idea I would like to implement, and overall, the algorithm breaks down like this:
    - [ ] Prompt the LLM with the previous paragraph, relative character and factoids, and ask it to generate the metadata.json, and tracks.csv. These two artifacts can be stored in the prompt + response log file for this prompt.
    - [ ] Prompt the LLM with melody_elements_instructions_prompt.txt.  This prompt needs to prompt the LLM with the previous paragraph, and relative character and factoids like is done previously first_score_suggestions.txt, but this time, append metadat.json, and then append melody_elements_instructions.txt. Read out and parse the artifacts which are the 4 dwell notes and the 9 melodic edges.
    - [ ] Repeat the music construction three times, once with the following additional rules substituted into melody_instructions.txt:
        - standard additional_rules: none
        - complimentary additoinal_rules: * create a complimentary melody by inverting the weights, for example, (1.0-0.5,1.0-0.25,1.0-0.125,1.0-0.125) = (0.5, 0.75, 0.875, 0.857)
        - reprise additional_rules: * Before following the above steps, adjust the musical edges by stretching them out by another measure, and adding appropriate additional notes to makes sense to fill out the new timing.
    Music Constructions instructions (to be repeated for each type of melody):
    - [ ] Construct a melody by prompting the LLM with the melody_instructions.txt template.
    - [ ] Save off the melody artifact from the output of the prompt and save it in a file: melody.csv (or complimentary_melody.csv or reprise_melody.csv)
    - [ ] Generate the measures.csv from the melody artifact (Hopefully this can be done in Python without LLM in the loop).
    - [ ] For each voice in the music touch-point, do the following:
        - [ ] Modify first_score_suggesions.txt to focus on exactly the voice to be generated
        - [ ] For the melody voice, substitute the melody artifact into the first_score_suggestions.txt, instructing the LLM to convert the given melody into a track (all syntax for the output artifact of the track should remain as before), and save as melody.musiccsv
        - [ ] For the other voices, substitute the melody track AND any other already generated tracks into the first_score_suggesstions.txt, instructing the LLM to build the given harmony, beat, base, or other track to match and harmonize with the given music so far.
        - [ ] NO LONGER ask the first_score_suggestions.txt to generate the metadata.json, tracks.csv, measures.csv.
    - [ ] Combine the outputs of all the voices with the metadata.json, tracks.csv, and measures.csv to form the <title>.musiccsv and the <title>.mid (or <title>.complimentary.musiccsv and <title>.complimentary.mid or <title>.reprise.musiccsv and <title>.reprise.mid)
    - [ ] When the pipeline_vN is complete, similiar to how final.txt is collated and regenerated, create a sound_track directory. Copy each of the <title>.mid files from the vN pipeline music touch_points, using the normalized title from the music touch-point as the file name.

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

