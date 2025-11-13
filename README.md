# Introduction to GhostWriter
GhostWriter helps you turn rough ideas into clean, consistent chapters—fast. It brings structure to creative writing by guiding an LLM with your Setting, Characters, and per‑chapter Touch‑Points, so you stay in control of plot, voice, and pacing. Unlike “one‑shot” story generators, GhostWriter separates brainstorming from prose, preserves the parts you already like, and lets you iterate safely on just the sections you want. Human‑in‑the‑loop gates make revision deliberate, while deterministic pipelines keep runs repeatable and testable.

GhostWriter is an LLM centric writing engine that uses structured YAML (Setting + Chapter logs) to guide an AI “ghostwriter” to produce publishable prose while honoring your stories structure, flavor, and creative goals.

It focuses on:
- Feeding the LLM the right context (Setting + Chapter logs)
- Separating word smithing from creative descisions (brainstorming vs. active voice and character voices)
- Controlling detail level and story progression (Touch-Points = must-hit beats, narration vs. dialog vs. implicit)
- Leaving room for creativity (user-in-the-loop at many levels, especially creative idea level, and editing)

Examples of what you can do

- Brainstorm an outline for a missing chapter: If `chapters/CHAPTER_00X.yaml` doesn’t exist (or contains `brainstorming: true`), GhostWriter creates a structured outline from your Setting, Characters, and Content Table, then stops—so you can review before drafting.
- Shape a single scene without rewriting the whole chapter: Add a Touch‑Point like `- narration: Henry crosses the field under fire`. Run v1, review the first draft and suggestions, then mark up only that section. The rest of the chapter stays untouched.
- Iterate safely on dialog tone: For a `dialog` Touch‑Point, tweak `touch_point_first_draft.txt` and re‑run. GhostWriter applies a subtle edit and regenerates `suggestions.txt`, preserving other scenes and voices.
- Keep summaries in sync: If `story_so_far.txt` or `story_relative_to.txt` are missing, they’re regenerated from the latest chapter content—useful when you copy books or branch experiments.
- Edit globally, reconcile locally: Make sweeping edits in `draft_vN.txt`. On the next run, GhostWriter syncs the per‑touch‑point drafts and refreshes suggestions only where you changed something.

Why this matters: LLMs are great at rewriting everything—but that can erase good prose you already like. GhostWriter tracks each Touch‑Point and lets you regenerate just the parts you choose, keeping the rest stable. This lowers the impedance between fast idea sessions and publishable, consistent prose.

## Installation

1) Create and activate a virtual environment

- Linux/macOS
    - Create: `python3 -m venv venv`
    - Activate: `source venv/bin/activate`

- Windows (PowerShell)
    - Create: `python -m venv venv`
    - Activate: `venv\Scripts\Activate.ps1`

2) Install dependencies

- With the venv active: `pip install -r requirements.txt`

3) (Optional) Configure environment variables

- Create a `.env` file (recomended to copy `.env.example`) and set:
  - `OPENAI_API_KEY=...` (required for live LLM calls; if absent, mock mode is used)

## Quickstart

Activate your virtual environment, then run one of the following:

- Recommended:
  - python -m ghostwriter.cli run chapters/CHAPTER_001.yaml v1
  - python -m ghostwriter.cli run chapters/CHAPTER_001.yaml   # picks next version automatically

- With a custom book base directory (Task 11):
  - python -m ghostwriter.cli run chapters/CHAPTER_001.yaml v1 --book-base BecomingDjinn
  - python -m ghostwriter.cli run CHAPTER_001.yaml --book-base BecomingDjinn  # chapter resolved under base/chapters

- Legacy-compatible (delegates to the CLI under the hood):
  - python scripts/driver.py chapters/CHAPTER_001.yaml v1
  - python scripts/driver.py chapters/CHAPTER_001.yaml        # picks next version automatically

Optional flags:
- --log-llm to save per-step prompts and responses (legacy alias --show-dialog also works)

Notes:
- The program runs the pipelines exactly once for the selected version; there is no auto-loop. To iterate, edit inputs and run the next version.
- If OPENAI_API_KEY is not set, the system operates in offline/mock mode and returns deterministic mock outputs (good for dry runs and tests).

## Configurable file locations

You can control where the book’s working files live via environment variables. Set these in your shell or in a `.env` file (see `.env.example`). `.env.example` contains numerous environment variable settings for controlling the program, including per prompt model, max tokens, and thinking depth.

- GW_BOOK_BASE_DIR — Base directory containing your book assets (defaults to current working directory). When set, other paths default relative to it.
- GW_SETTING_PATH — Path to SETTING.yaml; defaults to `<base>/SETTING.yaml`.
- GW_CHARACTERS_PATH — Path to CHARACTERS.yaml; defaults to `<base>/CHARACTERS.yaml`.
- GW_CHAPTERS_DIR — Directory containing chapter YAML files; defaults to `<base>/chapters`.
- GW_ITERATIONS_DIR — Output directory for per-chapter artifacts; defaults to `<base>/iterations`.

CLI override:
- Pass `--book-base <dir>` to temporarily override `GW_BOOK_BASE_DIR` for a single run.

Examples:

```
# Use the included Little Red Riding Hood testbed
cp .env.example .env
# (optional) edit .env if needed
python -m ghostwriter.cli run CHAPTER_001.yaml --book-base testdata/LittleRedRidingHood
```

## User Guide

This `README.md` is mostly the Developer Guide, but also as seen above explains how to install and get the program running in the quick-start.  Once you have customized the book directory in .env, then you are ready to start writing! You will, however, wish to understand how the program works, and how to get the most out of a cli execution stage.  You can find this information in the `USER_GUIDE.md`.

The user guide covers concepts, user stories, and examples for using GhostWriter. It shows how to brainstorm at several levels (table of contents, chapter outline, character), and how to refine prose while preserving voice. `USER_GUIDE.md` includes:

### What the User Guide Contains

* Core concepts: `CONTENT_TABLE.yaml`, `SETTING.yaml`, `CHARACTERS.yaml`, `CHAPTER_XXX.yaml`, Touch‑Points, and the iterations/pipeline folder structure.
* CONTENT_TABLE brainstorming: how placeholders like `???` trigger a single synopsis insertion, with an end‑to‑end example.
* Chapter outline brainstorming: adding `brainstorming: true` or generating a missing chapter file; backing up prior versions.
* Character brainstorming: handling missing actors or `brainstorming: true` on a character; appending the new YAML entry.
* v1 narration/dialog/mixed brainstorming flow: adding bullets → human gate (`DONE`) → first draft suggestions pause.
* First draft gate details: editing `touch_point_first_draft.txt` + `first_suggestions.txt` before subtle refinement.
* Subtle edit phase: producing `touch_point_draft.txt` and new `suggestions.txt` after your edits.
* vN (edit branch) iteration: using prior drafts and suggestions to apply targeted subtle edits across the chapter.
* Global chapter editing: modifying `draft_vN.txt` directly and letting the next run reconcile changes back into per‑touch‑point files.
* Practical user stories illustrating each phase and when to re‑run versus when to advance the version number.

# Iteration model

The following discussion is more about how the program is designed and how it operates.  The following content is more aligned to the Developer Guide. Conceptually, each run executes the deterministic pipelines once for a specific version (vN). You iterate manually by making changes (to YAML, prompts, etc.) and running the next version.

## Prompting System

The current system relies on specialized prompts per pipeline step and per touch-point (narration, dialog, implicit). Prompt Templates can be modified and fine tuned by "developers" who know how to get the most out of LLM prompts. Each prompt is generated using a template under `prompts/` and supports the following substitution tokens (filled in at runtime):

Template substitutions (tokens)

- [SETTING] — The current chapter’s setting block merged with global `SETTING.yaml` factoids (or empty if none selected).
- [CHARACTERS] — YAML block of selected characters (by active actors), or all characters if none selected.
- [TOUCH_POINT] (aliases: [touch_point], [TOUCH-POINT], [touch-point], [TOUCHPOINT], [touchpoint]) — The raw text for the current touch‑point.
- [TOUCH_POINT_TYPE] — One of: narration, dialog, mixed.
- [ACTIVE_ACTORS] (alias: [actors]) — Comma‑separated list of actors active at this point.
- [SCENE] (alias: [scene]) — Current scene name/short description.
- [FORESHADOWING] — Comma‑separated foreshadowing tags accumulated so far.
- [PRIOR_PARAGRAPH] (alias: [prior_paragraph]) — The full polished text of the previous contentful touch‑point (context for continuity).
- [context] — Recent polished context window assembled for evaluation/suggestions.
- [prose] — The polished prose under evaluation (used by check/suggestions prompts).
- [story_so_far.txt] (alias: [STORY_SO_FAR]) — Current chapter’s story‑so‑far text (or prior chapter’s if current missing).
- [story_relative_to.txt] (alias: [STORY_RELATIVE_TO]) — Current chapter’s story‑relative‑to block (or prior chapter’s if missing).
- [draft_v?.txt] — Latest `draft_vN.txt` contents for this chapter (if present).
- [suggestions_v?.txt] — Latest `suggestions_vN.txt` contents (if present).
- [check_v?.txt] — Latest `check_vN.txt` contents (legacy; may be empty in new flow).
- [predraft_v?.txt] — The pre‑draft text for the current version when used by summary/check prompts.
- [rough_draft] — Freeform rough text being polished (used by `polish_prose_prompt.md`).
- [SETTING.yaml], [CHAPTER_xx.yaml] — Raw YAML text included only when `GW_INCLUDE_RAW_YAML=1` (debugging).

Notes:
- Tokens are literal (including brackets) and case‑sensitive unless an alias is listed.
- Not all tokens appear in every template; unused tokens are simply ignored.

Currently the system uses the following prompts:
### Prompt catalog and where each is used

- Brainstorm (first step for contentful touch‑points)
  - `narration_brain_storm_prompt.md` — produce narrative beat bullets.
    - Before: start of narration touch‑point
    - After: `ordering_prompt.md`
  - `dialog_brain_storm_prompt.md` — produce dialog beat bullets.
    - Before: start of dialog touch‑point
    - After: `ordering_prompt.md` (+ actor/agenda/body‑language helpers)
  - `implicit_brain_storm_prompt.md` — produce mixed dialog + brief action beats.
    - Before: start of mixed touch‑point
    - After: `ordering_prompt.md`

- Structure/ordering
  - `ordering_prompt.md` — sort and prioritize brainstorm bullets into a working sequence for this touch‑point.
    - After (narration): `generate_narration_prompt.md`
    - After (dialog/mixed): `actor_assignment_prompt.md`, `agenda_prompt.md`, `body_language_prompt.md`

- Narration generation
  - `generate_narration_prompt.md` — turn the ordered beats into polished narration for this touch‑point.
    - After: first‑draft gate (write `touch_point_first_draft.txt` + `first_suggestions.txt`), then `subtle_edit_prompt.md` on resume; suggestions via `check_narration_prompt.md`.

- Dialog/mixed generation helpers
  - `actor_assignment_prompt.md` — attribute lines to the correct characters for this touch‑point.
    - After: `character_dialog_prompt.md`
  - `agenda_prompt.md` — generate concise per‑actor goals/tactics for the scene; used as context.
  - `body_language_prompt.md` — generate non‑verbal cues; used as context.
  - `character_dialog_prompt.md` — generate the dialog text, incorporating assignment, agenda, and body language.
    - After: first‑draft gate → `subtle_edit_prompt.md` on resume; suggestions via `check_dialog_prompt.md`.

- Editing, evaluation, and suggestions
  - `subtle_edit_prompt.md` — apply targeted edits using the first draft and suggestions (or prior version artifacts on vN runs).
  - `check_narration_prompt.md`, `check_dialog_prompt.md`, `check_implicit_prompt.md` — evaluate the prose and produce actionable suggestions (saved to `suggestions.txt`).
  - `polish_prose_prompt.md` — optional polishing pass used by some flows to clean style and consistency.

- Chapter summaries
  - `story_so_far_prompt.md` — generate or refresh `story_so_far.txt`.
  - `story_relative_to_prompt.md` — generate or refresh `story_relative_to.txt`.

Notes:
- Dialog and mixed flows may run helper prompts in parallel before assembling context for `character_dialog_prompt.md`.
- The first‑draft gate applies to narration, dialog, and mixed; on resume the system runs `subtle_edit_prompt.md` and then regenerates suggestions.
### How `character_dialog_prompt.md` batches dialog

The dialog (and mixed) pipeline produces a single contiguous block of scene dialog by calling `character_dialog_prompt.md` once per actor in a controlled sequence. Here is how those individual per‑actor generations are assembled into the final batch:

1. Actor ordering source
  - `ordering_prompt.md` structures raw brainstorm bullets.
  - `actor_assignment_prompt.md` assigns each ordered beat (or portion of a beat) to a specific actor (character id) or sometimes to the narrator.
  - The resulting ordered list of actor IDs becomes the iteration order for dialog generation.

2. Context construction per call
  - Before each actor call, the system gathers the most recent dialog lines that have already been generated in this touch‑point plus any retained historical lines (tracked per actor across earlier touch‑points for continuity).
  - The number of lines injected is controlled by either an explicit `<dialog>N</dialog>` tag inside the template or, if omitted/left as `<dialog>N</dialog>`, by the environment variable `GW_DIALOG_CONTEXT_LINES` (default: 8).
  - If the template only contains the phrase “The last N lines of dialog” without a numeric `<dialog>` tag, GhostWriter rewrites the phrase to the chosen number but does not inject verbatim prior lines (this lets you show the count without flooding the prompt).

3. Template token substitution
  - `<id/>` → `<id>{character_id}</id>` (the canonical id from `CHARACTERS.yaml`)
  - `<character_yaml/>` → The full YAML snippet for that character (traits, cadence, lexicon, mannerisms, samples, etc.). If missing, the system logs a warning but proceeds.
  - `<agenda/>` → Actor‑specific goal/tactics text taken from `agenda_prompt.md` (or blank if none produced or disabled).
  - `<dialog>N</dialog>` → Replaced by the last N non‑empty dialog lines for contextual continuity (only if the tag exists).
  - `<prompt/>` → The beat‑specific directive for this actor call (often a merged or refined version of the assigned bullet, possibly decorated with body language cues from `body_language_prompt.md`).

4. Per‑actor model call
  - The filled template becomes the USER message; a concise SYSTEM message reinforces “return only dialog or inner monologue”.
  - Temperature and max tokens may be overridden by character hints (`temperature_hint`, `max_tokens_line`) or per‑prompt/env overrides.
  - The model output should be a single character utterance (may be one or multiple sentences). Validation runs; on format issues it retries up to 3 attempts.

5. Aggregation
  - The returned line(s) are appended to the growing dialog block in order. Subsequent actor calls see these newly added lines in their `<dialog>` context window.
  - After all actor calls for the touch‑point complete, the pipeline writes the combined text into the touch‑point’s draft artifact (`touch_point_first_draft.txt` during first‑draft phase, later refined by `subtle_edit_prompt.md` into `touch_point_draft.txt`).
  - Suggestions (`check_dialog_prompt.md`) evaluate the entire assembled dialog block, not each line individually, so inter‑line cohesion is considered.

6. Edge cases & fallbacks
  - Missing template file: If `prompts/character_dialog_prompt.md` is absent, a built‑in default template (shipping in code) is used so tests and offline runs remain deterministic.
  - Actor templates with embedded numeric `<dialog>12</dialog>` override both environment and default counts for that actor only.
  - If a character lacks YAML definition (e.g., newly referenced in a scene), a warning is logged (still proceeds—useful for exploratory drafting).
  - Narrator lines: Special id `Narrator` bypasses missing‑YAML warnings and is treated as neutral voice for exposition within the dialog sequence.

7. Why this design
  - Guarantees deterministic ordering (replayable in tests).
  - Allows tight control of voice per character while sharing immediate conversational context.
  - Makes incremental edits cheap: re‑running the version only regenerates the portion you changed, preserving other lines and their suggestions history.

Tuning tips:
  - Reduce `GW_DIALOG_CONTEXT_LINES` for snappier, less self‑referential dialog; increase it for stronger long‑range continuity.
  - Add character‑specific `<dialog>4</dialog>` tags in the template if some characters should pay attention only to the last few lines (e.g., aloof or distracted personas).
  - Incorporate more structured guidance into `<agenda/>` (e.g., “Goal: persuade Henry to advance; Tactic: gentle reassurance, avoid military jargon”) to sharpen intent nuances.

### Music touch-point gates (Slices 1–5)

The music track feature now ships with the first two interactive gates:

- **Voice context assembly** (`ghostwriter.music.context`): parses chapter `voices:` directives, links each voice token to matching characters/factoids, and summarizes any sanitized `score.musiccsv` assets already in the iteration folder. The resulting payload (`build_music_prompt_context`) fuels all music prompts.
- **First-score gate** (`ghostwriter.music.pipeline.ensure_first_score_gate`): whenever a prose touch-point creates a first-draft gate, the system also generates `touch_point_first_score.musiccsv`, `first_score_suggestions.txt`, and a quick-audition `first_monitor.mid` when voices are defined. This uses `prompts/music_first_score_prompt.md` plus `prompts/music_check_prompt.md` for immediate feedback.
  - Debug traces: `first_score.txt` (SYSTEM/USER/RESPONSE for score generation) and `score_check.txt` (SYSTEM/USER/RESPONSE for score suggestions) are written alongside the artifacts for inspection; `first_monitor.mid` mirrors the generated score so authors can listen without opening a DAW.
- **Subtle-score refinement** (`ghostwriter.music.pipeline.run_subtle_score_pass`): on resume (or any vN subtle-edit branch) the edited first score and feedback are refined into `touch_point_score.musiccsv` with fresh `score_suggestions.txt`, via `prompts/music_subtle_edit_prompt.md` and the shared check prompt.
  - Debug traces: `subtle_score.txt` captures the refinement prompt/response; `score_check.txt` is overwritten with the latest check trace after subtle pass.
- **Sanitized imports**: Slice 1’s importer/sanitizer continues to normalize optional raw MIDI drops into `import.musiccsv`, `score.musiccsv`, and `monitor.mid`; the voice context automatically surfaces those summaries to the gates above.
- **Exports & packaging** (`ghostwriter.music.exporter.finalize_music_exports`): every completed run assembles available touch-point scores into `score_vN.musiccsv`, mirrors the latest build under `iterations/<chapter>/score/`, renders `final.mid` plus per-voice MIDIs, emits a structured `manifest.json`, and creates `score_bundle_vN.zip` for easy sharing.

Character outlines in music prompts

- When a voice references known characters (e.g., idea "wolf" linked to `id: wolf`), the first-score prompt now includes a compact `character_outlines` block with the fields most useful to composition (id, name, background, traits, cadence, lexicon, prefer/avoid, mannerisms, sample/common/rare lines, forbidden). This gives the model concrete persona guidance instead of only names.
- You can toggle this with the environment variable `GW_INCLUDE_CHARACTER_OUTLINES_MUSIC` (default: 1). Set to `0` to omit character outlines from the prompt if you prefer lighter requests.

These steps mirror the prose workflow: first-score artifacts pause for author edits, then subtle refinement resumes deterministically. See `tests/music/test_pipeline.py` for mocked examples of both gates, and ensure the music dependencies (`mido`, `pretty_midi`, `numpy`) are installed via `pip install -r requirements.txt`.

If you substantially change the template, run the test suite to ensure formatting validators still pass, then perform a golden update (see instructions below—added in the next section) so example artifacts match the new prompt wording.

If you do change the prompts, it is possible the unit tests may no longer pass.  Also, it is confusing for the LRRH golden example to have prompts that do not match what the program outputs. If you do change the prompts, you probably will want to run the golden-update on the book like so:

### Golden update: refresh the example (“LRRH”) artifacts

When you change prompt templates, regenerate the example outputs so the repository’s golden artifacts (under `testdata/LittleRedRidingHood/iterations/`) match the new behavior.

Two safe ways to do it:

Option A — using `--book-base` (no `.env` changes)

```bash
# From project root, run one chapter at a time; the next version is picked automatically
python -m ghostwriter.cli run CHAPTER_001.yaml --book-base testdata/LittleRedRidingHood
python -m ghostwriter.cli run CHAPTER_002.yaml --book-base testdata/LittleRedRidingHood
python -m ghostwriter.cli run CHAPTER_003.yaml --book-base testdata/LittleRedRidingHood
python -m ghostwriter.cli run CHAPTER_004.yaml --book-base testdata/LittleRedRidingHood
```

Option B — point your `.env` at the test book

```bash
cp .env.example .env
printf "\nGW_BOOK_BASE_DIR=testdata/LittleRedRidingHood\n" >> .env
# Then run with chapter paths resolved via GW_CHAPTERS_DIR
python -m ghostwriter.cli run CHAPTER_001.yaml
python -m ghostwriter.cli run CHAPTER_002.yaml
python -m ghostwriter.cli run CHAPTER_003.yaml
python -m ghostwriter.cli run CHAPTER_004.yaml
```

Notes and tips
- Versioning: Each run executes exactly once and writes `draft_vN.txt`, `suggestions_vN.txt`, and `final.txt`. If prior drafts exist, the CLI will choose the next N automatically when you omit `vN`.
- Human gates: If a run pauses at a brainstorm gate, open the `brainstorm.txt` shown in the message, add a final line `DONE`, and re‑run the same command. For v2+ edit runs, gate files are not reconstructed; the pipeline proceeds with subtle edit and suggestions regeneration.
- Clean rebuild (optional): To force a full regeneration, back up and remove old iteration folders first. Example:

  ```bash
  # Backup then clear old artifacts (optional, destructive)
  ts=$(date +%Y%m%d_%H%M%S)
  mkdir -p backups && cp -r testdata/LittleRedRidingHood/iterations backups/iterations_$ts
  rm -rf testdata/LittleRedRidingHood/iterations/*
  ```

- Mock vs live LLM: If `OPENAI_API_KEY` is unset, runs use deterministic mock outputs (recommended for tests and CI). Set the key in `.env` for live generations.
- Verify: After refreshing, run the test suite to ensure repo state and prompts are consistent:

  ```bash
  YAML_CEXT_DISABLED=1 python -m pytest -q
  ```

  Setting `YAML_CEXT_DISABLED=1` sidesteps intermittent PyYAML segmentation faults observed on some Python builds.

This keeps the Little Red Riding Hood golden examples aligned with the current prompt templates and validators.

### Brainstorm (human-in-the-loop)

For narration/dialog/implicit touch-points, the pipeline begins with a brainstorm step that produces a bullet list. This brainstorm is intentionally human-gated:

- The LLM is never instructed to add DONE. It only produces bullets.
- The program writes or appends brainstorm bullets to `iterations/CHAPTER_xxx/pipeline_vN/NN_<type>/brainstorm.txt`.
- To proceed, you must manually open that brainstorm.txt and add a final line containing exactly: `DONE` (all caps) on a line by itself.
- Re-run the same command. The pipeline will pick up from there using all bullets up to (but not including) the DONE marker.
- Bullets are cumulative across runs; previous bullets are silently included in the prompt for continuation.

# Iteration workflow and brainstorming rules

This project uses deterministic, single-pass pipelines with explicit human gates. Each run performs exactly one update and then exits. You re-run to proceed. There are four kinds of brainstorming the system can initiate, each with clear trigger and stop rules.

## 1) CONTENT_TABLE brainstorming (table of contents)

When it runs
- Automatically runs first if a brainstorm placeholder exists in `CONTENT_TABLE.yaml` regardless of the chapter you asked to run.
  - Placeholders detected in the table list include:
    - A map item with key `???` or value `???`
    - A string list item containing `???`
    - A numeric chapter entry whose value starts with `???` (e.g., `6: "??? …"`) or with `Brainstorm…` (case-insensitive)

What it targets
- If any numeric placeholder exists, it selects the smallest such chapter number and fills that slot.
- Otherwise it selects “next” = max existing numeric chapter + 1.

What the LLM sees and writes
- Prompt includes the entire `CONTENT_TABLE.yaml` text and a chapter-settings index derived from parsed chapters.
- The synopsis is written back to `CONTENT_TABLE.yaml` as a numeric key (e.g., `6:`), using YAML literal block style (`|`) for multi-line text.
- A copy of system+prompt+response is saved to `chapters/CONTENT_TABLE_brainstorm.txt`.

How it stops
- It always performs exactly one synopsis insertion (fill placeholder or append next) and then exits.

## 2) CHAPTER brainstorming (outline file)

When it runs
- If the chapter file you requested is missing, or
- If the chapter YAML contains any Touch-Point dictionary with `brainstorming: true`.

What the LLM sees and writes
- Prompt includes: CONTENT_TABLE, story-so-far/relative-to from the previous completed chapter, and Setting/Characters excerpts.
- It writes or overwrites `chapters/CHAPTER_XXX.yaml` with a structured outline (Touch-Points, optional setting with factoids/actors/scene). Story sections are omitted on purpose.
- The prior chapter YAML (if any) is backed up as `CHAPTER_XXX.N.yaml` before overwrite.
- A prompt+response trace is saved next to the chapter as `chapters/CHAPTER_XXX.txt`.

How it stops
- It performs one outline update and then exits.

## 3) CHARACTER brainstorming (single character entry)

When it runs
- If any `CHARACTERS.yaml` entry has `brainstorming: true`, or
- If the current chapter setting references an actor not present in `CHARACTERS.yaml` (missing character).

What the LLM sees and writes
- It uses the current chapter YAML, setting dereferences, and optionally an example character when none are referenced.
- If the character exists and has `brainstorming: true`, that YAML is used as the seed text; otherwise you’re prompted in the terminal for a brief description.
- The generated YAML entry starting with `- id:` is appended to `CHARACTERS.yaml`.
- A prompt+response trace is written to `<base>/character_brainstorm.txt`.

How it stops
- It appends exactly one character outline and then exits.

## 4) Touch-point pipelines: narration, dialog, implicit (human-gated brainstorm)

When they run
- For each touch-point that produces prose, the pipeline begins with a “brainstorm bullets” step. The rest of the pipeline (ordering → generation/assignment → polish) only proceeds after you mark the brainstorm as done.

Where the brainstorm lives
- For each chapter version `vN`, brainstorm bullets are persisted under:
  - `iterations/CHAPTER_xxx/pipeline_vN/04_narration/brainstorm.txt` (narration example)
  - `iterations/CHAPTER_xxx/pipeline_vN/05_dialog/brainstorm.txt` (dialog example)
  - `iterations/CHAPTER_xxx/pipeline_vN/implicit/.../brainstorm.txt` (implicit example)
  - Exact folder numbers can vary with pipeline layout; look for a `brainstorm.txt` in the step folder.

Human gate: how to control “done”
- The LLM never writes `DONE`. It only produces more bullets.
- The program appends newly generated bullets to `brainstorm.txt` and then stops with: “Brainstorming still in progress.”
- To proceed to ordering/generation:
  1) Open the `brainstorm.txt` for that step.
  2) Add a final line with exactly: `DONE` (all caps) on its own line.
  3) Re-run the same command. The pipeline will consume all bullets up to (but not including) the `DONE` line and continue.
- Bullets are cumulative across runs; you can add, edit, or reorder them before placing `DONE`.

Other controls
- Dialog/implicit ordering can be disabled with env vars (see `.env.example`):
  - `GW_DISABLE_ORDERING_DIALOG=1`
  - `GW_DISABLE_ORDERING_IMPLICIT=1`

# Architecture and Design

Instead of writing freeform prose from scratch, this system separates story elements into structured YAML files:

SETTING.yaml – A declarative set of notes of the novel’s background.

CHARACTERS.yaml - A declarative set of notes of the novel's characters.

The system is a deterministic, sequential pipeline driven by touch-points currently with zero automatic looping. Two operating branches exist depending on whether a previous draft is present.

- Removed from sequence for now: master_initial_prompt, master_prompt, and the single monolithic check prompt. Instead, the system uses specialized prompts per step and per touch-point.
- All steps are strictly sequential and deterministic (no auto-iterate), except for the implied per–touch-point loop and a local retry mechanism for output-format validation.

### Two branches

1) When no prior draft exists (no `draft_v1.txt`):
  - Read and parse chapter touch-points as commands: `actors`, `scene`, `foreshadowing`, `narration`, `dialog`, `implicit`.
   - Maintain state across touch-points: active actors, current scene, foreshadowing flags, and dialog history per actor (latest few lines).
   - For each touch-point:
     - `actors`/`scene`/`foreshadowing`: update state only.
     - `narration`: run Narration pipeline.
  - `dialog`: run Dialog pipeline.
     - `implicit`: run Implicit pipeline.
   - Write `draft_v1.txt` as a parseable sequence of pairs: original touch-point + polished output.
   - Generate `story_relative_to.txt` and `story_so_far.txt`.
   - Write `final.txt` as a clean, publishable text containing only polished prose (no touch-points/markdown).

2) When a prior version exists (largest N where `draft_vN.txt` and `suggestions_vN.txt` are present):
  - Load prior polished texts and suggestions into state so the edit pipeline can reference them by touch-point.
  - For each touch-point that yields prose (`narration`, `dialog`, `implicit`), run the Subtle Edit pipeline instead of generating from scratch.
   - Re-run per-touch-point checks and write a parseable `suggestions_v(N+1).txt`.
   - Regenerate and overwrite `story_relative_to.txt`, `story_so_far.txt`, and `final.txt`.

### Pipelines
  - `(narration_brain_storm_prompt.md → bullet list)`
  - `(ordering_prompt.md → bullet list)`
  - `(generate_narration_prompt.md → text)`

- Dialog pipeline
  - `(ordering_prompt.md → bullet list)`
  - `(actor_assignment_prompt.md → actor list)`
  - In parallel: `(body_language_prompt.md → bullet list)` and `(agenda_prompt.md → agenda list)`
  - The actor list is stored in state and re-used for downstream templates.

- Implicit pipeline
  - `(implicit_brain_storm_prompt.md → bullet list)`
  - `(ordering_prompt.md → bullet list)`
  - In parallel: `(body_language_prompt.md → bullet list)` and `(agenda_prompt.md → agenda list)`
  - Join as in Dialog pipeline; produce output text via `character_dialog_prompt.md` per actor line.

  - `(subtle_edit_prompt.md → text)`

### Pre-draft user-in-the-loop (Task 3)

For narration, dialog, and implicit touch-points, each run now pauses for a human edit pass before the final draft is recorded:

- Phase 1 (first draft):
  - The pipeline generates prose and writes `touch_point_first_draft.txt` and `first_suggestions.txt` under `iterations/CHAPTER_xxx/pipeline_vN/NN_<type>/`.
  - The program stops gracefully with: “Waiting for user suggestions on first draft.”
  - Edit `touch_point_first_draft.txt` directly as needed.

- Phase 2 (resume):
  - Re-run the same command. The system reads your edited `touch_point_first_draft.txt` plus `first_suggestions.txt` and applies `subtle_edit_prompt.md`.
  - Outputs are saved as `touch_point_draft.txt` and a fresh `suggestions.txt`.

Notes:
- The old `check.txt` per–touch-point file is no longer produced.
- `touch_point_state.json` is still written to allow resuming and to preserve state.

### Output formats and validation

Each pipeline step has an expected output format. The framework validates output; if invalid, it retries up to two more times (total 3 attempts). On third failure, the program stops and reports the error.


### State tracking and dialog history

- Track current `actors`, `scene`, and `foreshadowing` (updated by their commands).
- Track recent dialog per actor (last N lines) so `dialog`/`implicit` pipelines can feed rich context to `character_dialog_prompt.md`.

Per chapter iteration folder: `iterations/CHAPTER_xxx/`

- `draft_vN.txt` — parseable list of (touch-point, polished output) pairs.
- `suggestions_vN.txt` — parseable list of (touch-point, per-touch-point checks) results.
- `final.txt` — stripped, publish-ready prose (polished only).
- LLM logs — every prompt+response round-trip is saved under a subdirectory with descriptive filenames; logs include fully substituted prompts and raw outputs for traceability.

Environment configuration

For detailed environment variable settings (global defaults, per-step overrides, and per-prompt overrides for model/temperature/token budgeting, plus reasoning-effort controls), see `.env.example`.


These YAML files serve as inputs to an LLM-powered ghostwriter, which produces continuous prose chapters while ensuring that key narrative elements are included.

File Structure
/project-root
  ├── SETTING.yaml
  ├── chapters/
  │    ├── CHAPTER_001.yaml
  │    ├── CHAPTER_002.yaml
  │    └── ...
  ├── prompts/
  |    ├── actor_assignment_prompt.md
  |    ├── agenda_prompt.md
  |    ├── body_language_prompt.md
  |    ├── narration_brain_storm_prompt.md
  |    ├── dialog_brain_storm_prompt.md
  |    ├── implicit_brain_storm_prompt.md
  |    ├── character_dialog_prompt.md
  |    ├── check_dialog_prompt.md
  |    ├── check_implicit_prompt.md
  |    ├── check_narration_prompt.md
  |    ├── generate_narration_prompt.md
  |    ├── ordering_prompt.md
  |    ├── polish_prose_prompt.md
  |    ├── reaction_prompt.md
  |    ├── story_relative_to_prompt.md
  |    ├── story_so_far_prompt.md
  |    └── subtle_edit_prompt.md
  ├── iterations/
  │    └── CHAPTER_01/
  |         ├── story_so_far.txt
  |         ├── story_relative_to.txt
  |         ├── suggestions_v1.txt
  │         ├── draft_v1.txt
  |         ├── suggestions_v2.txt 
  │         └── draft_v2.txt
  ├── README.md
  └── scripts/ (Python helpers)

  ### Tuning the character dialog template

  The template used to generate each character’s dialog is now editable at:

  `prompts/character_dialog_prompt.md`

  Supported tokens inside this template:

  - `<id/>` — replaced with `<id>{character_id}</id>`
  - `<character_yaml/>` — inlined YAML for the character from `SETTING.yaml`
  - `<agenda/>` — optional per-call agenda text (or blank)
  - `<dialog>N</dialog>` — replaced by the last N lines of surrounding dialog context
    - Default N is controlled by env var `GW_DIALOG_CONTEXT_LINES` (default: 8)
  - `<prompt/>` — the per-call prompt content

  Notes:
  - If your visible text contains the phrase `The last N lines of dialog`, the `N` will be replaced by the actual number chosen for that call.
  - If `prompts/character_dialog_prompt.md` is missing, the code falls back to a sensible built-in default.
- The driver is being updated to the deterministic per–touch-point pipelines described above. Some legacy sections are retained in the README for context; the new design takes precedence.

## Package layout

    GhostWriter is a maintainable, testable package. Note that all YAML access must go through `RunContext` as the single point of YAML loading/writing.

    ```
    ghostwriter/
      __init__.py
      context.py            # RunContext.from_paths, load_yaml (only place that reads YAML)
      env.py                # Environment helpers, model/temp/max token resolution
      llm.py                # LLM client, backoff, completions
      utils.py              # File I/O, text formatting, warnings/logging, breadcrumbs
      validation.py         # Output validators and retry wrapper
      templates.py          # Template application and prompt-key helpers
      artifacts.py          # Draft/suggestions record I/O and parseable formats
      resume.py             # Checkpointing and resume helpers
      characters.py         # Character list access (via ctx), rendering and substitution
      pipelines/
        __init__.py
        common.py           # Shared pipeline helpers (replacements, polish, DONE gating)
        narration.py        # run_narration_pipeline
        dialog.py           # run_dialog_pipeline
        implicit.py         # run_implicit_pipeline
        subtle_edit.py      # run_subtle_edit_pipeline
      commands.py           # High-level flows (run_pipelines_for_chapter, etc.)
      cli.py                # Argument parsing and entrypoint
    ```

    - `scripts/driver.py` will become a thin wrapper that imports `ghostwriter.cli` and calls `main()`.
    - Only `ghostwriter.context` performs YAML reads. All other modules accept a `RunContext` and avoid file I/O during pipelines.

# Book Directory Files

This section provides precise YAML schemas and file locations for developers. For conceptual understanding and usage guidance, see `USER_GUIDE.md`.

## File Locations (configurable via environment)

Default paths relative to `GW_BOOK_BASE_DIR` (defaults to current working directory):

- `SETTING.yaml` — Controlled by `GW_SETTING_PATH`; defaults to `<base>/SETTING.yaml`
- `CHARACTERS.yaml` — Controlled by `GW_CHARACTERS_PATH`; defaults to `<base>/CHARACTERS.yaml`
- `chapters/` — Controlled by `GW_CHAPTERS_DIR`; defaults to `<base>/chapters`
  - `CONTENT_TABLE.yaml` — Table of contents (optional)
  - `CHAPTER_*.yaml` — Per-chapter outlines (e.g., `CHAPTER_001.yaml`, `CHAPTER_002.yaml`)
- `iterations/` — Controlled by `GW_ITERATIONS_DIR`; defaults to `<base>/iterations`
  - `CHAPTER_*/` — Per-chapter output artifacts (drafts, suggestions, summaries, pipeline logs)

## SETTING.yaml

Schema (top-level dict):

```yaml
Title: string (optional)

Factoids:
  - name: string (required, unique identifier)
    description: string (required, narrative detail)
```

- `Factoids` is a list of dicts; each dict must have `name` and `description` keys.
- Referenced from chapter YAML via `factoids: ["name1", "name2"]` in `setting` or `scene` touch-points.
- Used by prompts via `[SETTING]` token after dereferencing selected factoids.

Example:

```yaml
Title: Little Red Riding Hood

Factoids:
  - name: Basket of Goodies
    description: A small wicker basket filled with bread and jam.
  - name: Red Hood
    description: A bright red cloak and hood, a gift from Grandmother.
```

## CHARACTERS.yaml

Schema (top-level dict or list):

Option A — top-level `Characters` key (preferred):

```yaml
Characters:
  - id: string (required, lowercase, unique identifier; used in actors lists)
    name: string (required, display name)
    background: string (optional, multi-line description)
    traits: [string, ...] (optional list)
    cadence: string (optional, voice style)
    lexicon: string (optional, word choice guidance)
    prefer: [string, ...] (optional list of favored words/phrases)
    avoid: [string, ...] (optional list of forbidden words/phrases)
    mannerisms: [string, ...] (optional list of physical/behavioral cues)
    sample_lines: [string, ...] (optional list of example dialog)
    common_lines: [string, ...] (optional list of frequent phrases)
    rare_lines: [string, ...] (optional list of uncommon phrases)
    forbidden: [string, ...] (optional list of prohibited behaviors/topics)
    temperature_hint: float (optional, LLM temperature override for this character)
    max_tokens_line: int (optional, max tokens per dialog line for this character)
    brainstorming: bool (optional, triggers character refresh on next run if true)
```

Option B — top-level list (legacy):

```yaml
- id: red
  name: Little Red Riding Hood
  traits: [curious, kind, naive]
  # ... (same fields as above)
```

- `id` is the canonical identifier used in `actors` touch-points and template substitutions.
- `name` is the display name used in prose and logs.
- All fields except `id` and `name` are optional; they guide LLM prompts for dialog/mixed/implicit touch-points.
- Character entries with `brainstorming: true` trigger automatic character refresh on the next run.
- Missing actors referenced in chapter YAML will trigger automatic character generation (pauses for user confirmation).

## CONTENT_TABLE.yaml

Schema (top-level dict):

```yaml
TABLE_OF_CONTENTS:
  - Author: string (optional)
  - Title: string (optional)
  - <chapter_number>: |
      Multi-line synopsis text for chapter N.
  - <chapter_number>: string (single-line synopsis)
```

Alternatively, a flat dict:

```yaml
Author: "..."
Title: "..."
1: "Synopsis for chapter 1"
2: |
  Multi-line synopsis
  for chapter 2.
```

Brainstorm placeholders (triggers automatic synopsis generation):

- Key or value containing `???`
- Numeric chapter value starting with `???` or `Brainstorm…` (case-insensitive)

Example:

```yaml
TABLE_OF_CONTENTS:
  - Author: "Collected Fairy Tale"
  - Title: "Little Red Riding Hood"
  - 001: |
      A Meeting on the Path. Red meets the Wolf along the forest trail.
  - 002: |
      The Cottage Deception. The Wolf tricks his way in.
  - 003: "???"
```

On next run, the system will generate a synopsis for chapter 3 and exit.

## CHAPTER_*.yaml

Schema (top-level dict):

```yaml
Touch-Points:
  - setting:
      factoids: [string, ...] (optional, references SETTING.yaml Factoids by name)
      actors: [string, ...] (optional, references CHARACTERS.yaml by id)
  - scene:
      name: string (optional, scene identifier)
      description: string (optional, multi-line scene details)
      props: (optional list)
        - name: string
          significance: string (optional, narrative role)
          description: string (optional, visual/sensory detail)
  - actors: [string, ...] (state update; activates these actors for subsequent touch-points)
  - foreshadowing: string | [string, ...] (state update; tags to avoid revealing prematurely)
  - narration: string (prose beat directive; triggers narration pipeline)
  - dialog: string (dialog beat directive; triggers dialog pipeline)
  - mixed: string (action + dialog beat; triggers mixed pipeline)
  - implicit: string (subtext/implicit dialog beat; triggers implicit pipeline)
  - brainstorming: bool (if true, triggers outline refresh and exits)

Story-So-Far: |
  Optional multi-line summary of prior chapters.

Story-Relative-To:
  <entity_name>: "Contextual state description for this chapter"
```

Touch-Point types and behavior:

- `setting`, `scene`, `actors`, `foreshadowing` — state-only; update context for subsequent content touch-points.
- `narration`, `dialog`, `mixed`, `implicit` — content touch-points; each runs a full pipeline (brainstorm → ordering → generation → first-draft gate → subtle edit).
- `brainstorming: true` — triggers chapter outline regeneration; backs up existing YAML as `CHAPTER_XXX.N.yaml`; exits after writing new outline.

Validation rules:

- `Touch-Points` must be a list.
- Each touch-point is either a string (shorthand for `narration: "..."`) or a dict with exactly one key (the touch-point type).
- `actors` references must match character `id` values in `CHARACTERS.yaml` (case-sensitive).
- `factoids` references must match factoid `name` values in `SETTING.yaml`.

Example:

```yaml
Touch-Points:
  - setting:
      factoids: ["Forest Path", "Red Hood"]
      actors: ["red", "wolf"]
  - scene:
      name: "Forest Path"
      description: |
        A winding dirt trail beneath tall, shadowed pines.
  - narration: "Red walks the forest path, humming, basket in hand."
  - dialog: "The Wolf greets Red and asks where she's going."
  - mixed: "Danger hinted at in the dark trees and Wolf's smile."

Story-So-Far: |
  Red promised to bring goodies to her ailing grandmother.

Story-Relative-To:
  red: "Eager to help, trusts strangers."
  wolf: "Hungry, looking for an opportunity."
```

## Iterations Directory Structure

Output artifacts per chapter under `iterations/CHAPTER_*/`:

- `draft_vN.txt` — Parseable pairs of (touch-point, polished prose) for version N
- `suggestions_vN.txt` — Per-touch-point evaluation and actionable suggestions for version N
- `final.txt` — Clean, publish-ready prose (polished text only; no touch-point markers)
- `story_so_far.txt` — Chapter summary (cumulative narrative state)
- `story_relative_to.txt` — Entity-specific context for this chapter
- `pipeline_vN/` — Per-version pipeline logs:
  - `NN_<type>/` — Per-touch-point step folders (e.g., `05_narration/`, `06_dialog/`)
    - `brainstorm.txt` — Bullet list (human-editable; append `DONE` to proceed)
    - `*_ordering.txt` — Ordered bullet list
    - `touch_point_first_draft.txt` — First draft (v1 gate; edit before resume)
    - `first_suggestions.txt` — Initial suggestions (v1 gate; edit before resume)
    - `touch_point_draft.txt` — Final polished prose for this touch-point
    - `suggestions.txt` — Updated suggestions after refinement
    - `touch_point_state.json` — State snapshot (actors, scene, foreshadowing, dialog history)
    - Additional logs: `*_actor_assignment.txt`, `*_agenda.txt`, `*_body_language.txt`, `*_dialog_batch.txt` (dialog/mixed/implicit only)

Version flow:

- v1: Full generation with brainstorm gates (append `DONE` to `brainstorm.txt`) and first-draft gates (edit `touch_point_first_draft.txt` + `first_suggestions.txt`, then re-run).
- v2+: Subtle edit iteration using prior version's `touch_point_draft.txt` and `suggestions.txt` as inputs; no gates reconstructed.


# Future Extensions

Tasks: Add a VS Code task (.vscode/tasks.json) to run the script with one click (e.g. Run Draft Generator).

CodeLens / Comments: You can write custom commands to highlight missing touch-points after verification.

Visualization tools (e.g. graph of character/prop interactions).

Integration with GitHub Actions for automated iteration loops.

# Goals

Maintain creative flexibility while enforcing structural discipline.

- Ensure you have the expected inputs: `SETTING.yaml`, `CHARACTERS.yaml`, a chapter file like `chapters/CHAPTER_001.yaml`, and the required prompt templates under `prompts/` (see below).

- Use the project virtual environment when running. Either activate it first or call the venv’s Python directly:

  - source venv/bin/activate && python -m ghostwriter.cli run chapters/CHAPTER_001.yaml v1
  - ./venv/bin/python -m ghostwriter.cli run chapters/CHAPTER_001.yaml v1
- Python 3.10+ (project verified with Python 3.13)
- Git (optional but recommended)


## Dependencies

Runtime/testing dependencies are pinned in `requirements.txt`. Key libraries:

- PyYAML — YAML loading/writing
- openai — LLM client (to be used when core integration is implemented)
- python-dotenv — environment variable loading from `.env`
- pytest, pytest-mock — test framework and mocking utilities
- numpy — required by musical tooling
- mido, pretty_midi — MIDI inspection and rendering helpers

### Music importer utilities (Slice 1)

- `ghostwriter.music.process_import_directory(import_dir)` reads raw `.mid`, `.midi`, or `.mid2` files under a voice import folder and writes a normalized `import.musiccsv` with tempo/time-signature/instrument metadata embedded as JSON.
- `ghostwriter.music.ensure_sanitized(import_dir)` (invoked automatically by `process_import_directory`) reparses `import.musiccsv`, emits a sanitized `score.musiccsv`, and keeps a playable `monitor.mid` in sync so authors can audition the track.
- Dependencies listed above are required for these helpers. Install via `pip install -r requirements.txt` before running the music workflow.

## Usage

Basic CLI to run the deterministic pipelines for a chapter:

- Ensure you have the expected inputs: `SETTING.yaml`, `CHARACTERS.yaml`, a chapter file like `chapters/CHAPTER_001.yaml`, and prompt templates under `prompts/`.
- Activate your venv, then run:

```
python -m ghostwriter.cli run chapters/CHAPTER_001.yaml v1
```

- Outputs will be written under `iterations/CHAPTER_001/`:
  - `pipeline_v1/` — per–touch-point logs, checks, suggestions, brainstorm files
  - `draft_v1.txt` — parseable pairs of touch-point and polished result
  - `final.txt` — clean, publish-ready prose (polished only)

Notes:

- The current implementation uses placeholders for LLM calls; tasks 4–8 in `TODO.md` will wire up real API usage and the full iteration loop.
- For Windows, replace `python`/activation commands per the Installation section above.

### File validation and auto-directories

- The driver now validates required inputs up front and exits with a clear message if something is missing:
  - `SETTING.yaml` (resolved via GW_SETTING_PATH or `<base>/SETTING.yaml`)
  - `CHARACTERS.yaml` (resolved via GW_CHARACTERS_PATH or `<base>/CHARACTERS.yaml`)
  - A chapter file under your chapters dir (GW_CHAPTERS_DIR), e.g. `CHAPTER_001.yaml` or `chapters/CHAPTER_001.yaml`
  - Prompt templates under `prompts/`:
    - `narration_brain_storm_prompt.md`
  - `dialog_brain_storm_prompt.md`
    - `implicit_brain_storm_prompt.md`
    - `ordering_prompt.md`
    - `generate_narration_prompt.md`
    - `actor_assignment_prompt.md`
    - `body_language_prompt.md`
    - `agenda_prompt.md`
    - `reaction_prompt.md`
    - `subtle_edit_prompt.md`
    - `polish_prose_prompt.md`
    - `story_so_far_prompt.md`
    - `story_relative_to_prompt.md`
  - `prompts/character_dialog_prompt.md` is optional; a built-in default is used if missing.

- On first run for a chapter, the driver auto-creates `<iterations>/CHAPTER_xxx/` (and dialog log folders when `--log-llm` is used).

### Version selection

- You can specify `vN` explicitly (e.g., `v1`), omit it to pick the next version automatically, or pass `auto` as an alias for “next available version”.
- There is no auto-looping mode in the new pipelines; run again for subsequent iterations.

### Prompt logging (--log-llm)

To debug per-step prompts and responses, use the `--log-llm` flag. The legacy alias `--show-dialog` is also accepted.

Where logs go:

- `iterations/CHAPTER_xxx/dialog_prompts_vN/`
- Each call is saved as `NN_<characterId>.txt`, where `NN` is the call index for that version.
- Each file contains labeled sections:
  - `=== SYSTEM ===`
  - `=== USER ===`
  - `=== RESPONSE ===`

Example:

```
python -m ghostwriter.cli run chapters/CHAPTER_001.yaml v1 --log-llm
```

This is useful for reviewing exactly how prompts were presented to the LLM and what responses were returned.

### Offline/mock mode

- If `OPENAI_API_KEY` is not set (or the OpenAI client cannot be initialized), the program will return deterministic mock outputs. This is helpful for local testing without incurring API calls.
- You can control behavior via a `.env` file in the project root; environment variables from `.env` override the shell by default.

## Running Tests

Always run tests using the project's virtual environment so the correct dependencies are used.

```bash
# Option A: Activate the venv first (recommended)
source venv/bin/activate
python -m pytest -q

# Option B: Without activating, call the venv's Python explicitly
./venv/bin/python -m pytest -q
```

Notes:
- Ensure you've installed dependencies inside the venv: `pip install -r requirements.txt` (see Installation above).
- Using the venv avoids "pytest: command not found" and version mismatches.

### Test environment (.env.test)

Pytest automatically loads a test-specific environment file `.env.test` (wired in `tests/conftest.py`). This keeps tests deterministic and independent of your personal `.env` settings.

- Default test book base: `.env.test` sets `GW_BOOK_BASE_DIR=sandbox/LittleRedRidingHood`.
- Your personal `.env` can continue to point to your active book (e.g., `BecomingDjinn`); tests will still use `.env.test`.
- VS Code Testing tab will also respect this because `.env.test` is loaded by pytest itself.

## Troubleshooting

- If `python` isn’t found, try `python3`.
- If packages fail to install, upgrade tooling inside the venv:

```bash
# With venv activated
python -m pip install --upgrade pip setuptools wheel

# Or directly via the venv interpreter
./venv/bin/python -m pip install --upgrade pip setuptools wheel
```
