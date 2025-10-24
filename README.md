# Introduction to GhostWriter
GhostWriter is a domain-specific writing engine that uses structured YAML (Setting + Chapter logs) to guide an AI “ghostwriter” to produce publishable prose while honoring story structure.

It focuses on:
- Feeding the LLM the right context (Setting + Chapter logs)
- Separating structure from prose (what’s required vs. what’s creative)
- Controlling emphasis (Touch-Points = must-hit beats, narration vs. dialog vs. implicit)
- Leaving room for creativity (avoid mechanical lists; keep a strong voice)

GhostWriter has been refactored into deterministic per–touch-point pipelines with a modern CLI.

## Quickstart (new CLI)

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

## Configurable file locations (Task 11)

You can control where the book’s working files live via environment variables. Set these in your shell or in a `.env` file (see `.env.example`).

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

## Emphasis & Importance

The key to making the LLM honor the structure is hierarchy. In the instructions:

Put Touch-Points at the top of the rules section (highest priority).

Put continuity (Story-So-Far + Story-Relative-To) second.

Put Setting third (context, but less “must do”).

Put Suggestions last (soft guidance).

## Iteration model

Each run executes the deterministic pipelines once for a specific version (vN). You iterate manually by making changes (to YAML, prompts, etc.) and running the next version.

## Meta-Prompting System

- Pre-processor reads Setting.yaml + Chapter.yaml and derives per–touch-point context.
- Specialized prompt templates are applied per step (brainstorm, ordering, dialog generation, checks, etc.).
- Output validators enforce expected shapes (bullets, actor lines) with local retry on validation failure.

# Architecture

Instead of writing freeform prose from scratch, this system separates story elements into structured YAML files:

SETTING.yaml – A declarative set of notes of the novel’s background.

CHARACTERS.yaml - A declarative set of notes of the novel's characters.

The system is moving to a deterministic, sequential pipeline driven by touch-points with zero automatic looping. Two operating branches exist depending on whether a previous draft is present.

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

## YAML Format
    ## Planned package layout (refactor)

    We are refactoring to a maintainable, testable package while keeping `RunContext` as the single point of YAML access.

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
## Setting Log (SETTING.yaml)
```yaml
Factoids:
    - name: "Time Period"
      description: "The Civil war was a time of industrialization and strained social fabrics"
```

## Characters (CHARACTERS.yaml)
```yaml
Characters:
    - id: "henry"
      name: "Henry"
      traits:
        - fearful
        - sentimental
        - avoids direct confrontation
      cadence: "short sentences with an inward stutter"
      lexicon: "Poor, uneducated, 1860 background"
      prefer: ["shudder", "ashamed", "light"]
      avoid: ["cool", "awesome"]
      mannerisms:
        - "touches his left sleeve when unsure"
      sample_lines:
        - "I— I don't know if I can do it."
        - "The light hit the field like a wound."
      forbidden:
        - "use military jargon beyond what's established"
      temperature_hint: 0.25
      max_tokens_line: 90
```

Factoids: General statements or explanations that do not fit into the remaining categories.

Characters: Dialog generators with traits and personality anchors.  This system works with a higher level more intelligent Director agent, which spawns off calls to individual sub-agent calls for the dialog.  The sub agents therefore need a full description of the voice, lexicon, traits, etc. for the best personality prompting. 

## Chapter Log (CHAPTER_xxx.yaml)
```yaml
Touch-Points:
  - setting:
      factoids:
        - "Time Period"
      actors:
        - "Henry"
        - "Seargent"
  - scene:
      name: "Battlefield"
      description: "A terrifying field of death and artillery explosions"
      props:
        - name: "Red Cloth"
          significance: "A symbol of courage, sacrifice, or tragedy"
          description: "A scrap of what was once a white soldier's shirt"
  - dialog: "Henry sees the wounded soldier with the bloody shirt"
  - narration: The horror of a falling artillery shell landing 100 meters away.
  - implicit: "The battlefield described only through Henry’s impressions"
  - dialog: "The ridiculousness of shelling an entire continent"

Story-So-Far: |
    Henry has fled battle once, ashamed of his cowardice.

Story-Relative-To:
    Henry: "Haunted by his flight, eager to prove himself."
    Battlefield: "Still chaotic, still alive with artillery."
    Bloody Shirt: "About to be discovered."
```

Touch-Points: Mandatory beats the chapter must include.

Story-So-Far: Summary of past chapters.

Story-Relative-To: Contextualizes each character/scene/prop for this chapter.

## Prompting System (modernized)

The current system relies on specialized prompts per pipeline step and per touch-point (narration, dialog, implicit). The legacy master prompts are no longer part of the default flow.

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

- Create a `.env` file and set:
  - `OPENAI_API_KEY=...` (required for live LLM calls; if absent, mock mode is used)

## Dependencies

Runtime/testing dependencies are pinned in `requirements.txt`. Key libraries:

- PyYAML — YAML loading/writing
- openai — LLM client (to be used when core integration is implemented)
- python-dotenv — environment variable loading from `.env`
- pytest, pytest-mock — test framework and mocking utilities

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
