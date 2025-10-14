# Introduction to GhostWriter
This project is a system for writing a novel using structured YAML logs and AI ghostwriting assistance.
It provides a meta-prompting system and an iteration loop to generate, verify, and refine prose chapters.

GhostWriter is a domain-specific writing engine with YAML as the structured source and LLM as the prose generator. The challenge is to:

Feed the LLM the right context (Setting + Chapter logs).

Tell the LLM what’s structural vs. what’s prose.

Control emphasis (Touch-Points = must-hit beats, implicit vs. explicit).

Leave room for creativity (so the LLM doesn’t just mechanically list things).

## Emphasis & Importance

The key to making the LLM honor the structure is hierarchy. In the instructions:

Put Touch-Points at the top of the rules section (highest priority).

Put continuity (Story-So-Far + Story-Relative-To) second.

Put Setting third (context, but less “must do”).

Put Suggestions last (soft guidance).

## Iteration Loop

Thu user will kick off the following iteration loop sequence (for a chapter):

1. Construct prompt for next Draft (using the prompt as described in a later section), and write the next draft.

2. Run a Check Prompt afterward, e.g.:

Check the following prose against the "Touch-Points".  
For each touch-point, state whether it was satisfied explicitly, implicitly, or missing.  
Suggest improvements where missing.

3. Feed that back in as suggestions, and if anything is missing, return to step 1.

4. Write the the resulting draft version to the next .yaml chapter version.

5. Report status and overview of changes to the User.

## Meta-Prompting System

Pre-processor merges Setting.yaml + Chapter.yaml → one big input.

Standard prompt template applied.

Post-processor checks touch-point coverage (also using a prompt template).

# Architecture

Instead of writing freeform prose from scratch, this system separates story elements into structured YAML files:

SETTING.yaml – A declarative log of the novel’s background.

CHAPTER_xxx.yaml – Sequential logs for each chapter that guide prose generation.

These YAML files serve as inputs to an LLM-powered ghostwriter, which produces continuous prose chapters while ensuring that key narrative elements are included.

File Structure
/project-root
  ├── SETTING.yaml
  ├── chapters/
  │    ├── CHAPTER_001.yaml
  │    ├── CHAPTER_002.yaml
  │    └── ...
  ├── prompts/
  │    ├── master_prompt.md
  |    ├── story_so_far_prompt.md
  |    ├── story_relative_to_prompt.md
  │    └── check_prompt.md
  ├── iterations/
  │    └── CHAPTER_01/
  |         ├── story_so_far.txt
  |         ├── story_relative_to.txt
  |         ├── suggestions_v1.txt
   - `character_dialog_prompt.md` — template used for each CHARACTER dialog call (see below)
  │         ├── draft_v1.txt
  │         ├── check_v1.txt
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
    - You can also write a number instead of `N`, e.g. `<dialog>6</dialog>`
    - If the call includes `<dialog>k</dialog>`, that overrides the default N for that call
    - Default N is controlled by env var `GW_DIALOG_CONTEXT_LINES` (default: 8)
  - `<prompt/>` — the per-call prompt content

  Notes:
  - If your visible text contains the phrase `The last N lines of dialog`, the `N` will be replaced by the actual number chosen for that call.
  - If `prompts/character_dialog_prompt.md` is missing, the code falls back to a sensible built-in default.

  ## Configuration: token limits and models

  You can control output lengths and models per stage via environment variables (copy `.env.example` to `.env`).

  Token limits (max_tokens) per stage:
  - `GW_MAX_TOKENS_DIALOG` — each CHARACTER call
  - `GW_MAX_TOKENS_PRE_DRAFT` — pre-draft generation
  - `GW_MAX_TOKENS_CHECK` — touch-point check
  - `GW_MAX_TOKENS_SUGGESTIONS` — suggestions list
  - `GW_MAX_TOKENS_DRAFT` — polished draft
  - `GW_MAX_TOKENS_STORY_SO_FAR` — story_so_far summary
  - `GW_MAX_TOKENS_STORY_RELATIVE` — story_relative_to summary

  Model overrides per stage (optional):
  - `GW_MODEL_DIALOG`
  - `GW_MODEL_PRE_DRAFT`
  - `GW_MODEL_CHECK`
  - `GW_MODEL_SUGGESTIONS`
  - `GW_MODEL_DRAFT`
  - `GW_MODEL_STORY_SO_FAR`
  - `GW_MODEL_STORY_RELATIVE`

  If a per-stage model is not set, `OPENAI_MODEL` is used.

# YAML Format
## Setting Log (SETTING.yaml)
```yaml
Factoids:
    - name: "Time Period"
      description: "The Civil war was a time of industrialization and strained social fabrics"

Scenes:
    - name: "Battlefield"
      description: "Rolling fields, littered with smoke and the echoes of artillery."

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

Props:
    - name: "Bloody Shirt"
      active: true
      significance: "Symbol of sacrifice; implicit metaphor for courage"
```

Factoids: General statements or explanations that do not fit into the remaining categories.

Scenes: Backgrounds for story action. Should usually be described indirectly (e.g. through character impressions), but some narrator expositions are allowable if it doesn't easily fit into dialog.

Characters: Dialog generators with traits and personality anchors.  This system works with a higher level more intelligent Director agent, which spawns off calls to individual sub-agent calls for the dialog.  The sub agents therefore need a full description of the voice, lexicon, traits, etc. for the best personality prompting. 

Props: Active (interactable) or inactive (background/foreshadowing).

## Chapter Log (CHAPTER_xxx.yaml)
```yaml
Touch-Points:
    - explicit: "Henry sees the wounded soldier with the bloody shirt"
    - implicit: "The battlefield described only through Henry’s impressions"
    - explicit: "The ridiculousness of shelling an entire continent"

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

# Prompting System

# Main Prompts

1. Master Initial Prompt

The Master initial prompt is constructed from the master initial prompt template at ```prompts/master_initial_prompt.md```

The template has a placeholder for the SETTING.yaml, story_so_far.txt, story_relative_to.txt, the CHAPTER_xx.yaml, but does not have the most recent suggestions_v0.txt, and draft_v0.txt because this prompt is designed to write the initial v1 files to seed the following process. 

This actually will output a pre-prose document with CHARACTER TEMPLATES and CHARACTER "calls" to generate dialog progressviely via an LLM.

2. Master Prompt

The Master prompt is constructed from the master prompt template at ```prompts/master_prompt.md```

The template has a placeholder for the SETTING.yaml, story_so_far.txt, story_relative_to.txt, the CHAPTER_xxx.yaml, and the most recent suggestions_vn.txt, and draft_vn.txt (where n is the integer sequence number of the version being currently generated). 

This actually will output a pre-prose document with CHARACTER TEMPLATES and CHARACTER "calls" to generate dialog progressviely via an LLM.

3. Polish Prose Prompt

The result of the Master prompt will be processed by substituting in CHARACTER "calls" with the response from an LLM prompt.  Thus, the results might be coarse, mal-formatted, or otherwise not polished.

The Polish Prose Prompt is constructed from the polish_prose_prompt template at ```prompts/polish_prose_prompt.md```.

3. Verification Prompt 

The Verification Prompt is constructed from the verification prompt template at ```prompts/check_prompt.md```.

The template has a placeholder for the SETTING.yaml, CHAPTER_xx.yaml, and the predraft_v? text as generated by the master prompt.

4. Story-So-Far prompt

The Story-So-Far Prompt is constructed from the story-so-far template at ```prompts/story_so_far_prompt.md```.

The results of this prompt will be used to create the story_so_far for the next chapter.

5. Story-Relative-To prompt

The Story-Relative-To Prompt is constructed from the story-relative-to template at ```prompts/story_relative_to_prompt.md```.

The results of this prompt will be used to create the story_relative_to for the next chapter.

# Iteration Loop

The system supports an iteration workflow. For a given Chapter_xx:

EITHER: 
1.1. Version 1 Draft Generation will use Master Initial Prompt Template ```prompts/master_initial_prompt.md```.

1.2. Feed files into prompt template.

1.3. Send to LLM and save output as pre_draft_v1.txt.

OR: 
1.1. If Version N Draft Generation has already been completed, use Master Prompt Template ```prompts/master_prompt.md```.

1.2. Feed files into prompt template

1.3. Send to LLM and save output as pre_draft_vn.txt, where N is the current version.

Template Substitution

2.1. The program will load pre_draft_vn.txt, where N is the current version, and load in each <CHARACTER TEMPLATE> structure (see master_prompt.md for detailed structure).  Then, for each <CHARACTER> template "call", the framework will send a LLM prompt for that character to get a narrowed and accurate voice and dialog response.  The reponse will be substituted in.

2.2. The program will insert the fully substituted <CHARACTER> template response text into the polish_prose_prompt template, and write the response of this prompt to the file draft_vn.txt, where N is the current version.

Verification

2.1. Verification will use ```prompts/check_prompt.md``` Prompt Template.

2.2. Feed files into prompt template.

2.3. Send to LLM, and save results as check_vN.txt, where N is the current version.

Verification Check

3.1. If Verification resulted in a "missing" case, go to step 1.1 where the check_vN will be added into a re-draft. Repeat until all touch-points are satisfied.

# Future Extensions

Tasks: Add a VS Code task (.vscode/tasks.json) to run the script with one click (e.g. Run Draft Generator).

CodeLens / Comments: You can write custom commands to highlight missing touch-points after verification.

Visualization tools (e.g. graph of character/prop interactions).

Integration with GitHub Actions for automated iteration loops.

# Goals

Maintain creative flexibility while enforcing structural discipline.

- Ensure you have the expected inputs: `SETTING.yaml`, a chapter file like `chapters/CHAPTER_001.yaml`, and prompt templates under `prompts/`.

- Use the project virtual environment when running. Either activate it first:

```bash
source venv/bin/activate
python scripts/driver.py chapters/CHAPTER_001.yaml v1
```

- Or, without activating, call the venv’s Python directly:
Provide a framework that scales to a full novel-length work.
```bash
./venv/bin/python scripts/driver.py chapters/CHAPTER_001.yaml v1
```
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

- Create a `.env` file (use `.env.example` as a starting point) and set:
    - `OPENAI_API_KEY=...` (required when LLM integration is implemented in later tasks)

## Dependencies

Runtime/testing dependencies are pinned in `requirements.txt`. Key libraries:

- PyYAML — YAML loading/writing
- openai — LLM client (to be used when core integration is implemented)
- python-dotenv — environment variable loading from `.env`
- pytest, pytest-mock — test framework and mocking utilities

## Usage

Basic CLI to generate a draft and run a check for a chapter:

- Ensure you have the expected inputs: `SETTING.yaml`, a chapter file like `chapters/CHAPTER_001.yaml`, and prompt templates under `prompts/`.
- Activate your venv, then run:

```
python scripts/driver.py chapters/CHAPTER_01.yaml v1
```

- Outputs will be written under `iterations/CHAPTER_001/` as `draft_v1.txt` and `check_v1.txt`.

Notes:

- The current implementation uses placeholders for LLM calls; tasks 4–8 in `TODO.md` will wire up real API usage and the full iteration loop.
- For Windows, replace `python`/activation commands per the Installation section above.

### File validation and auto-directories

- The driver now validates required inputs up front and exits with a clear message if something is missing:
  - `SETTING.yaml` at project root
  - A chapter file under `chapters/`, e.g. `chapters/CHAPTER_001.yaml`
  - Prompt templates under `prompts/`:
    - `master_initial_prompt.md`
    - `master_prompt.md`
    - `polish_prose_prompt.md`
    - `check_prompt.md`
    - `story_so_far_prompt.md`
    - `story_relative_to_prompt.md`
  - `prompts/character_dialog_prompt.md` is optional; a built-in default is used if missing.

- On first run for a chapter, the driver auto-creates `iterations/CHAPTER_xxx/` (and dialog log folders when `--show-dialog` is used).

### Iteration mode (auto) and limits

- Instead of specifying an explicit version like `v1`, you can pass `auto` to let the driver iterate drafts until all touch-points are satisfied (i.e., no "missing" found in the verification step) or until the maximum cycles is reached.
- Maximum cycles are controlled by the `GW_MAX_ITERATIONS` environment variable (default: `2`).

Example:

```
python scripts/driver.py chapters/CHAPTER_001.yaml auto
```

Artifacts produced per version include:

- `pre_draft_vN.txt` — pre-prose with CHARACTER TEMPLATES and <CHARACTER> call sites
- `check_vN.txt` — verification output for touch-points
- `suggestions_vN.txt` — suggestions derived from the check output
- `draft_vN.txt` — polished prose after dialog substitution and cleanup

### Dialog prompt logging (--show-dialog)

To debug per-character dialog generation, use the `--show-dialog` flag. When enabled, every <CHARACTER> call made during the polishing phase is captured (system prompt, user prompt, and the LLM response) as plain text files.

Where logs go:

- `iterations/CHAPTER_xxx/dialog_prompts_vN/`
- Each call is saved as `NN_<characterId>.txt`, where `NN` is the call index for that version.
- Each file contains labeled sections:
  - `=== SYSTEM ===`
  - `=== USER ===`
  - `=== RESPONSE ===`

Example:

```
python scripts/driver.py chapters/CHAPTER_001.yaml auto --show-dialog
```

This is useful for reviewing exactly how the character template and call prompt were presented to the LLM and what response was returned.

## Running Tests

When tests are added (tasks 12–15), run them with:

```bash
# If activated
pytest -q

# Or without activating
./venv/bin/python -m pytest -q
```

## Troubleshooting

- If `python` isn’t found, try `python3`.
- If packages fail to install, upgrade tooling inside the venv:

```bash
# With venv activated
python -m pip install --upgrade pip setuptools wheel

# Or directly via the venv interpreter
./venv/bin/python -m pip install --upgrade pip setuptools wheel
```
