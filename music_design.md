# Music Track Output Feature — Technical Design

## 1. Context and Goals
- Extend GhostWriter so that every chapter Touch-Point can also produce symbolic music artifacts aligned with the prose pipelines.
- Support ingestion of user-supplied MIDI that is normalized into MusicCSV, plus AI-assisted refinement loops and DAW-friendly MIDI exports.
- Maintain compatibility with the existing iteration/version model (`pipeline_vN/`, `draft_vN.txt`, `final.txt`) while introducing analogous score artifacts.
- Provide a clear developer roadmap for modules, prompts, tests, and deliverables derived from `MUSIC_REQUIREMENTS.md`.

## 2. User Story and Human-in-the-Loop Workflow
**Feature Story**: “As an author-composer, I want GhostWriter to help me co-create music that mirrors my chapter beats, so I can iterate on both prose and soundtrack together.”

**Workflow Overview**
1. Author prepares chapter YAML with new `voices` metadata and optional `music:` directive. Note: if either voices or the music directive is missing from the chapter YAML, skip the music portion of the workflow.
2. Optional: Author drops raw MIDI assets into `iterations/CHAPTER_xxx/pipeline_vN/NN_<type>/NN_track_<voice>_import/`.
3. When raw assets are present, the importer converts them into normalized `import.musiccsv`; the sanitizer validates user edits and maintains `score.musiccsv` plus `monitor.mid`.
4. If an import directory is absent, or after a sanitized score exists, GhostWriter runs the content pipelines using a **per-voice / per-variant** model:
  - Uses prose context (touch-point text, brainstorm bullets) and voice metadata to prompt the LLM for **per-voice, per-variant note CSVs** named `notes_<voice_token>_<variant>.csv` (e.g., `notes_major.tenor.flute.red.melody_standard.csv`).
  - These calls are made via a multi-voice first-pass composer that iterates voices in a stable order, passing reduced-note grids for the melody and already-scored voices.
  - A helper then assembles **per-variant first-pass scores** `first_<title>_<variant>.musiccsv` and per-variant monitor MIDIs `first_monitor_<title>_<variant>.mid` from shared `metadata.json`, `tracks.csv`, `measures_<variant>.csv`, and all `notes_<voice_token>_<variant>.csv`.
  - A second-pass composer reuses the same per-voice/per-variant flow but includes suggestion text as additional context, producing final `<title>_<variant>.musiccsv` and `monitor_<title>_<variant>.mid`.
5. On version completion, GhostWriter aggregates the final per-variant scores into `score_vN.musiccsv`, renders consolidated `score/final.musiccsv`, `score/final.mid`, and per-voice MIDI files.
6. Bundle is packaged for sharing (zip). User can iterate (vN+1) with updated assets.

Note: In order to gracefully handle the case when the user wants to refine either the prose or the music separately, and not the other, well will add a minor enhancement: If the prior version (vN-1) suggestions.txt is only whitespace, then don't run the subtle_edit, but only copy the touch_point_* file, and create an empty suggestions.txt for vN.

**Human Gates**
- Optional import normalization: author edits `import.musiccsv` before the system refreshes `score.musiccsv` and `monitor.mid`.
- First-score gate: analogous to the prose first-draft gate (edit `touch_point_first_score.musiccsv`, `first_score_suggestions.txt`).
- Subtle-score gate: author tweaks `touch_point_score.musiccsv` or global `score_vN.musiccsv`; reconciliation regenerates per-voice suggestions.

## 3. Architecture Overview
```
chapters/CHAPTER_XXX.yaml
  ├─ voices: ["major.beat.drum.engine", ...]
  └─ music: "cinematic track in D minor; slow tempo ..."

iterations/CHAPTER_XXX/pipeline_vN/NN_<type>/
  ├─ NN_track_<voice>_import/
  │   ├─ raw midi files (.mid, .mid2)
  │   ├─ import.musiccsv (round-trip editable)
  │   ├─ score.musiccsv (sanitized)
  │   └─ monitor.mid (preview)
  ├─ metadata.json
  ├─ tracks.csv
  ├─ measures_standard.csv
  ├─ measures_complimentary.csv
  ├─ measures_reprise.csv
  ├─ notes_<voice_token>_standard.csv
  ├─ notes_<voice_token>_complimentary.csv
  ├─ notes_<voice_token>_reprise.csv
  ├─ first_<title>_standard.musiccsv
  ├─ first_<title>_complimentary.musiccsv
  ├─ first_<title>_reprise.musiccsv
  ├─ first_monitor_<title>_standard.mid
  ├─ first_monitor_<title>_complimentary.mid
  ├─ first_monitor_<title>_reprise.mid
  ├─ first_score_suggestions_standard.txt
  ├─ first_score_suggestions_complimentary.txt
  ├─ first_score_suggestions_reprise.txt
  ├─ <title>_standard.musiccsv
  ├─ <title>_complimentary.musiccsv
  ├─ <title>_reprise.musiccsv
  ├─ monitor_<title>_standard.mid
  ├─ monitor_<title>_complimentary.mid
  └─ monitor_<title>_reprise.mid

iterations/CHAPTER_XXX/
  ├─ score_vN.musiccsv
  └─ score/
      ├─ final.musiccsv
      ├─ final.mid
      ├─ <voice>.mid (per voice)
      └─ manifest.json
```

## 4. Chapter YAML Additions
- Introduce optional `voices:` list under chapter root.
  - Token format: `<chords>.<register>.<instrument>.<idea>[.<role>]`
  - Loosely validate components against enumerations: the exact strings do not need to be enforced, but the required fields should be present (major, minor, diminished, augmented, chromatic / bass, tenor, alto, beat / free-form instrument string / story entity or mood / optional melody|harmony). T
- `Music:` scalar string describing global direction (tempo, key, mood).
- Parser updates ensure `RunContext` exposes `chapter.get("voices", [])` and `chapter.get("Music")`.

## 5. LLM Prompt Integration
For each content touch-point that generates music:
1. Collect context:
   - Touch-point type + text, brainstorm bullets, prior `touch_point_score.musiccsv` (last two measures) if available.
   - Voice-specific assets: voice components, sanitized `score.musiccsv` from import dir if present, character/factoid lookup based on voice idea.
   - Global music directive, tempo/time signature summary, and aggregated metadata (tempo map, key).
2. Prompt templates:
   - `prompts/music_first_score_prompt.md` → Produces `touch_point_first_score.musiccsv` and `first_score_suggestions.txt`.
   - `prompts/music_subtle_edit_prompt.md` → Applies refinements using edited first score and suggestions.
   - `prompts/music_check_prompt.md` → Generates actionable `score_suggestions.txt`.
3. Validation: ensure MusicCSV output parses; quantize measure alignment to chapter tempo grid.
4. Reuse `UserActionRequired` gates mirroring prose pipeline.

## 6. MIDI Input and Output Handling
### Input
- Accept `.mid`, `.midi`, `.mid2` (MIDI 2.0), optionally zipped packages.
- Use `mido`/`pretty_midi` for event extraction → canonical representation (tempo map, time signature changes, key detection heuristics).
- Normalize directly into MusicCSV via `MusicCSV.from_midi`, preserving:
  - Measures aligned by computed beats.
  - Instrument names (map GM program numbers to human-readable strings).
  - Channel, velocity (mapped to dynamics), articulation metadata, tuplets, repeats, and grace notes.
- Persist `import.musiccsv` plus a text summary suitable for LLM inspection.
- When the user edits `import.musiccsv`, run the sanitizer pass: schema validation, derived-field resolution, measure renumbering, consistent divisions.
- Generate `score.musiccsv` + `monitor.mid` by validating sanitized MusicCSV and calling `MusicCSV.to_midi`.
- Surface the canonical MusicCSV format instructions to the LLM by embedding `prompts/musiccsv_format_prompt.txt` in every generation prompt.

### Output
- Aggregate `touch_point_score.musiccsv` into `score_vN.musiccsv` by concatenating parts/measure sequences.
- Derive `score/final.musiccsv` (latest version) plus renderings:
  - `score/final.mid`: multi-track, default to MIDI 2.0 when `GW_MIDI2_ENABLED=1` (default).
  - Per-voice MIDI files (single track each) named after voice token.
- Ensure metadata (track names, tempo map) preserved.
- Generate `score/manifest.json` summarizing tempo, key, track metadata.

## 7. Module Plan
| Module | Responsibility | Notes |
|--------|----------------|-------|
| `ghostwriter/music/importer.py` | Detect and convert incoming MIDI into normalized MusicCSV | Wraps `mido`, `pretty_midi`, `MusicCSV.from_midi`; exposes `generate_import_musiccsv` |
| `ghostwriter/music/sanitizer.py` | Validate/clean user-edited MusicCSV, enforce measure alignment, regenerate monitor MIDI | Provides `sanitize_import(import_path, score_path, monitor_path)` |
| `ghostwriter/music/context.py` | Derive tempo, key, voice metadata; map voices to characters/factoids | Shared utilities for prompt-building |
| `ghostwriter/music/pipeline.py` | Orchestrate per-touch-point gates (first-score, subtle edit) | Mirrors prose pipeline structure |
| `ghostwriter/music/exporter.py` | Build `score_vN.musiccsv`, render final/per-voice MIDI, write manifest | Uses `MusicCSV.to_midi` |
| `ghostwriter/music/prompts.py` | Helpers to format LLM prompt payloads | Provides MusicCSV format reference |
| Updates to `ghostwriter/pipelines/narration.py` (and dialog/mixed) | Hook into music pipelines post prose generation | Ensure gating order maintained |
| Updates to `ghostwriter/resume.py` | Include music checkpoints in resume flow | |

## 8. Testing Strategy
- **Unit Tests**
  - `tests/music/test_importer.py`: MIDI → MusicCSV conversion (fixtures under `testdata/music/`).
  - `tests/music/test_sanitizer.py`: user-edited MusicCSV sanitized correctly; monitor MIDIs regenerated.
  - `tests/music/test_context.py`: voice-to-character/factoid mapping, tempo summaries.
  - `tests/music/test_exporter.py`: aggregation of touch-point scores into final outputs.
- **Golden Integration Tests**
  - Extend LittleRedRidingHood testbed with sample voice definitions and short MIDI fixtures; assert deterministic exports.
  - Compare generated `score_vN.musiccsv` and per-voice MIDIs against golden files (using tolerant diff for CSV/JSON components).
- **Prompt Regression Tests**
  - Mock LLM responses for `music_first_score_prompt.md` and `music_subtle_edit_prompt.md` to confirm gate transitions.
  - Validate schema of outputs after mock responses.
- **Schema Validation**
  - Validate generated artifacts with `ghostwriter.musiccsv.validate_musiccsv` and round-trip MIDI conversions.

## 9. Open Questions / Risks
- MusicCSV size & diffing: ensure summaries stay readable for suggestions, possibly add helper diff utilities.
- MIDI 2.0 support depth: initial export via `pretty_midi` (MIDI 1.0) with optional upgrade path.
- LLM token limits for larger scores—may need summarization of prior measures beyond last two.

## 10. Incremental Delivery Plan
1. Land importer + sanitizer with CLI hooks and basic tests.
2. Add voice metadata parsing and prompt context builders.
3. Integrate first-score gate and mock LLM flow.
4. Implement subtle-score refinements and suggestion regeneration.
5. Finish exporter + packaging bundle.
6. Final polish: documentation, CLI options, env toggles, golden refresh.
