# GhostWriter — User Guide

Welcome! This guide teaches you how to use GhostWriter step by step. We keep the language simple and the directions clear. You’ll learn the core ideas first, then practice with short stories and examples.

---

## Chapter 1 — Core Concepts

Big idea: GhostWriter helps you turn notes into a polished story. You write small pieces of information in easy YAML files. GhostWriter reads those files and uses an AI to write smooth, connected chapters for you.

### The six important pieces

We’ll use these files and folders again and again. Learn them once, and everything gets easier.

1) CONTENT_TABLE.yaml — your book’s table of contents
- What it is: A simple list of chapters and short summaries.
- Why it matters: It tells GhostWriter what chapters exist and the big idea of each one.
- Where it lives: In your book’s base folder.
- What it looks like:
```yaml
1: | 
  Red meets the Wolf on the path. 
2: |
  The Wolf reaches Grandma’s house first.
```
- Tip: If you put "???" as a placeholder, GhostWriter can brainstorm a single new synopsis for you.

2) SETTING.yaml — the world and facts
- What it is: Notes about time, place, and special rules in your story world.
- Why it matters: The AI uses these facts to keep the story consistent.
- Where it lives: In your book’s base folder.
- What it looks like:
```yaml
Factoids:
  - name: "Time Period"
    description: "A forest village long ago"
```

3) CHARACTERS.yaml — people who speak and act
- What it is: A list of characters with traits and voice.
- Why it matters: The AI uses these to make dialog sound right for each person.
- Where it lives: In your book’s base folder.
- What it looks like:
```yaml
Characters:
  - id: "red"
    name: "Little Red"
    traits: [brave, kind]
    cadence: "short, friendly sentences"
    sample_lines:
      - "I can do this."
```

4) CHAPTER_XXX.yaml — the outline of one chapter
- What it is: A list of Touch‑Points (beats) the chapter should hit.
- Why it matters: You stay in control of the plan; the AI fills in the prose.
- Where it lives: Inside your `chapters/` folder.
- What it looks like:
```yaml
Touch-Points:
  - setting:
      factoids: ["Time Period"]
      actors: ["Little Red", "Wolf"]
  - scene:
      name: "Forest Path"
      description: "Tall trees. Dappled light. A quiet breeze."
  - dialog: "Red meets the Wolf and says hello"
  - narration: "The path bends toward Grandma’s house."
  - mixed: "The Wolf baits red with a sly smile"
```

5) Touch‑Points — the tiny steps inside a chapter
- What they are: Small instructions like narration, dialog, or mixed action.
- Why they matter: Each Touch‑Point turns into a piece of polished text.
- Types you’ll use:
  - `actors` — who is active now
  - `scene` — where we are and what it’s like
  - `foreshadowing` — hints about what NOT to say in the story (yet). What do you, the author know: but the audience must guess (for now)?
  - `narration` — descriptive writing
  - `dialog` — characters talking
  - `mixed` — short actions mixed with quick lines

6) iterations/ — where the results go
- What it is: A folder the program fills with drafts and logs for each chapter and version.
- Why it matters: You can read what the AI wrote, see suggestions, and try again.
- Where it lives: `<base>/iterations/CHAPTER_xxx/`
- What you’ll see:
  - `draft_v1.txt` — the first full draft for that chapter
  - `suggestions_v1.txt` — helpful notes on what to improve
  - `final.txt` — the clean story (no extra markers)
  - `pipeline_v1/` — step‑by‑step files, like brainstorm notes and first‑draft gates

### A tiny story: Meet Alex

Alex wants to write Little Red Riding Hood. Alex adds a few Touch‑Points: a scene in the forest, a line of dialog, and a bit of narration. Alex runs GhostWriter. The program writes a first draft and some suggestions. Alex tweaks a line, runs it again, and gets a cleaner version—without losing the parts Alex liked.

### Try it now (make your own tiny starter book)

Instead of using the full Little Red Riding Hood example (which is already complete), create a fresh, empty tutorial book. The CLI now has a helper command:

1) Pick an empty folder path (or one that doesn’t exist yet), for example `my_first_book`.
2) Run the new command:

```bash
python -m ghostwriter.cli new-book my_first_book
```

What this does:
- Checks the folder is empty (or creates it).
- Copies simple starter files: `SETTING.yaml`, `CHARACTERS.yaml`, `chapters/CONTENT_TABLE.yaml`, `chapters/CHAPTER_001.yaml`.
- Names things “MyFirstBook” with a very small plot idea.

3) Run your first chapter draft:

```bash
python -m ghostwriter.cli run CHAPTER_001.yaml --book-base my_first_book
```

If the program pauses for brainstorming: open the shown `brainstorm.txt`, add a new line with just `DONE`, save, and run the same command again. This tells the program you’re ready to move on.

Feel free to edit the YAML files (change a character name, update a Touch‑Point) and re‑run to see the difference.

### Quick checklist
- Do I have SETTING.yaml and CHARACTERS.yaml?
- Did I write a few Touch‑Points in `chapters/CHAPTER_XXX.yaml`?
- Do I know which chapter I’m running (like CHAPTER_001)?
- Am I okay with the program creating or updating `iterations/CHAPTER_xxx/`?

---

## Chapter 2 — CONTENT_TABLE brainstorming

Goal: Let GhostWriter write one new chapter synopsis in your `CONTENT_TABLE.yaml` when you leave a simple placeholder.

### What is `CONTENT_TABLE.yaml`?
It’s your table of contents. Each item is one chapter number with a short summary (a few sentences). GhostWriter looks here first to understand your book’s plan.

### When does brainstorming run?
If GhostWriter finds a placeholder, it will brainstorm exactly one new synopsis and then stop. Placeholders include:
- A map item with key `???` or value `???`
- A list item that is just `???`
- A numeric chapter whose summary starts with `???` or with `Brainstorm…` (not case‑sensitive)

How it chooses where to write:
- If any numbered chapter has a placeholder, it picks the smallest chapter number with a placeholder and fills that one.
- Otherwise, it adds a new chapter at the end: next number = largest current number + 1.

What gets written:
- GhostWriter writes the new synopsis directly into `CONTENT_TABLE.yaml` using a neat block style (`|`).
- It also saves a trace file: `chapters/CONTENT_TABLE_brainstorm.txt` so you can see the exact prompt and response.
- It does only one insertion per run, then exits so you can review the change.

### Step‑by‑step: Add a placeholder and generate one synopsis

1) Open `CONTENT_TABLE.yaml` and add a placeholder. Examples:

```yaml
# Example A: Replace a chapter’s summary with ???
2: "???"

# Example B: Add a new slot at the end
6: "???"

# Example C: A list form (also supported)
Chapters:
  - 1: |
      Red meets the Wolf on the path.
  - 2: "???"
```

2) Run GhostWriter for any chapter. The CONTENT_TABLE step will run first and fill one synopsis.

```bash
python -m ghostwriter.cli run CHAPTER_001.yaml --book-base my_first_book
```

3) Check the result:
- Open `CONTENT_TABLE.yaml` and read the new text GhostWriter added.
- Look at `chapters/CONTENT_TABLE_brainstorm.txt` to see the prompt, system message, and the model’s response.

4) Decide what’s next:
- Happy with it? Keep writing or run again to fill the next placeholder.
- Want to try again? Edit the new synopsis yourself or put `???` back and run once more (it will add or update one synopsis each run).

### Try it now (safe example)

Use the included Little Red Riding Hood book so you don’t affect your own files.

```bash
cp .env.example .env
# Add a placeholder in my_first_book/CONTENT_TABLE.yaml (open and edit manually)
python -m ghostwriter.cli run CHAPTER_001.yaml --book-base my_first_book
```

Tip: CONTENT_TABLE brainstorming never loops. It writes one synopsis and exits so you can review.

---

## Chapter 3 — CHAPTER outline brainstorming

Goal: Create or improve a chapter outline (its list of Touch‑Points) before any prose is written.

### When does chapter brainstorming run?
GhostWriter runs the chapter outline brainstorming step automatically in two cases:
1. The chapter file you ask for does not exist yet (e.g., you run `CHAPTER_002.yaml` but only `CHAPTER_001.yaml` is present). It will create a brand‑new outline file.
2. The existing chapter YAML contains a Touch‑Point dictionary with `brainstorming: true`. That tells GhostWriter: “Replace or expand this outline.”

After the outline is generated, the program stops so you can read and edit it. No prose is produced yet.

### What does the outline include?
The chapter YAML will contain:
- A list of Touch‑Points (setting, scene changes, narration beats, dialog beats, mixed beats).
- Optional setting block with `factoids` and `actors` for the opening context.
- It will not include finished story paragraphs; only the plan.

### Backups for safety
If a chapter file already exists and you trigger brainstorming (with `brainstorming: true`), GhostWriter backs up the old file first:
`chapters/CHAPTER_XXX.N.yaml` (where N is a number). This keeps previous outlines safe.

### Adding a brainstorming trigger
Open an existing chapter and add `brainstorming: true` to a Touch‑Point dictionary. Example:

```yaml
Touch-Points:
  - setting:
      factoids: ["Time Period"]
      actors: ["The Hero", "The Guide"]
  - dialog: "The Hero meets the Guide and asks what to do first."
  - narration: "They start walking together toward a small goal."
  - implicit: "The Guide points to a flower and smiles."
  - brainstorming: true
```

When GhostWriter sees that final dictionary `{brainstorming: true}` it replaces the Touch‑Point list with a freshly generated outline.

### Step‑by‑step: Create a missing chapter
1) Make sure your `CONTENT_TABLE.yaml` lists the chapter number you want (optional but helpful).
2) Run the CLI for a chapter that doesn’t exist yet:
```bash
python -m ghostwriter.cli run CHAPTER_002.yaml --book-base my_first_book
```
3) GhostWriter creates `chapters/CHAPTER_002.yaml` with a structured list of Touch‑Points and stops.
4) Open the new file and edit any lines you want. Remove or rearrange beats. Keep them short.
5) Run the chapter again (now prose generation will begin unless you add a brainstorm gate elsewhere).

### Step‑by‑step: Refresh an existing outline
1) Open `chapters/CHAPTER_001.yaml`.
2) Add a dictionary Touch‑Point: `- brainstorming: true` at the end (or anywhere as its own list item).
3) Save the file.
4) Run:
```bash
python -m ghostwriter.cli run CHAPTER_001.yaml --book-base my_first_book
```
5) The old outline is backed up as `CHAPTER_001.1.yaml` (or higher number). The new outline appears in `CHAPTER_001.yaml`.
6) Remove the `brainstorming: true` entry so future runs produce prose instead of regenerating outline.

### Tips for a strong outline
- Keep each Touch‑Point short and above all _focused_ on one conceptual idea.
- Use a mix of narration and dialog beats so the chapter alternates description and character interaction.
- Add an early setting Touch‑Point for context.
- If your characters seem to want to "spill the beans", add a `foreshadowing` Touch‑Point with the spoilers to avoid.

### Try it now (outline refresh)
Edit `CHAPTER_001.yaml` in `my_first_book` and append:
```yaml
  - brainstorming: true
```
Run the command. Compare the old backup file to the new outline. Pick the beats you like and tweak them.

### Stop rule
Chapter outline brainstorming always performs exactly one update then stops. You are in control before any prose exists.

---

More chapters coming next, one at a time:
## Chapter 4 — CHARACTER brainstorming

Goal: Add or improve a character entry so dialog and narration match the right voice.

### When does character brainstorming run?
GhostWriter runs character brainstorming in two cases:
1. A character in `CHARACTERS.yaml` has `brainstorming: true`. This asks the system to expand or refresh that character.
2. Your current chapter references an actor that does not exist in `CHARACTERS.yaml` (for example, in the chapter’s top‑level setting, a `setting` touch‑point, or an `actors` touch‑point).

After one character entry is generated, the program stops so you can check and edit it.

### What gets written
- A new YAML entry starting with `- id:` is appended to `CHARACTERS.yaml` (for missing actors). For `brainstorming: true`, the existing YAML is used as a seed for improvement; the system may add a refreshed entry.
- A prompt/response trace is saved to `<base>/character_brainstorm.txt`.
- Only one character is processed per run.

### Step‑by‑step: Add a missing actor from a chapter
1) Open `chapters/CHAPTER_001.yaml` in your `my_first_book` folder.
2) Add an actor not in `CHARACTERS.yaml`, for example `Friend`:
```yaml
Touch-Points:
  - setting:
      factoids: ["Time Period", "Path"]
      actors: ["The Hero", "Friend"]
  - dialog: "The Hero says hi to a Friend."
  - narration: "They walk a few steps and look around."
```
3) Run:
```bash
python -m ghostwriter.cli run CHAPTER_001.yaml --book-base my_first_book
```
4) GhostWriter will create a new character entry for `Friend` and append it to `CHARACTERS.yaml`. If prompted in the terminal, type a description (for example: "A cheerful classmate who encourages the Hero").
5) Open `CHARACTERS.yaml` and tweak traits, cadence, and sample_lines to match your taste.

### Step‑by‑step: Refresh an existing character
1) Open `CHARACTERS.yaml` and add `brainstorming: true` to the character you want to improve:
```yaml
- id: hero
  name: The Hero
  traits: [curious, kind, brave]
  cadence: simple, friendly
  brainstorming: true
```
2) Run:
```bash
python -m ghostwriter.cli run CHAPTER_001.yaml --book-base my_first_book
```
3) GhostWriter uses the existing YAML as a seed and writes an improved entry (saved in `CHARACTERS.yaml`).
4) Remove `brainstorming: true` so future runs don’t keep refreshing.

### Tips for strong character entries
- Keep `id` short and lowercase and _unique_ (e.g., `hero`, `guide`). Use `name` for the display name.
- Add 2–5 traits and a simple `cadence` (how the character sounds).
- Include a few `sample_lines` so the voice is clear.
- Use `avoid` words to prevent out‑of‑style phrases.

### Try it now (new character)
Add `Friend` to the chapter’s `actors` list as shown above and run the command. Open `CHARACTERS.yaml`, find the new entry, and edit it to match the story you want to tell.

### Stop rule
Character brainstorming adds or refreshes exactly one entry per run and then exits so you can review.

---

## Chapter 5 — v1 brainstorming flow (narration, dialog, mixed, implicit)

Goal: Understand how GhostWriter turns each content Touch‑Point into polished prose or dialog during the very first (v1) authoring pass.

### The core pattern (every content Touch‑Point)
For each Touch‑Point of type `narration`, `dialog`, `mixed`, or `implicit`, the pipeline follows the same high‑level pattern:

1. Brainstorm bullet ideas (append‑only list in `brainstorm.txt`).
2. Stop and wait for you to add a final `DONE` line.
3. Order the bullet list (unless ordering is disabled with an environment flag).
4. (Dialog / Mixed / Implicit only) Expand with actor assignment, body language, per‑actor agenda, and reactions.
5. Generate the prose or dialog lines.
6. Create the first‑draft gate (files `touch_point_first_draft.txt` + `first_suggestions.txt`) and pause again. (Full editing instructions appear in Chapter 6.)

You control the pace by deciding when to add `DONE` and when to resume after reviewing the first draft. Pease DO edit the brainstorm.txt to your heart's content! In many cases, some of the ideas generated by the LLM are strong, and many are weak. You should Keep to the bullet point format, but other than that, definitely delete bullets, modify them, or even add your own! For dialog and mixed, the order of the bullets will be the order the characters will say things. For narration, don't worry about order, because the LLM will try to put the bullets into the most logical flow for you! 

### Artifact locations
Per Touch‑Point artifacts live under:
```
iterations/CHAPTER_xxx/pipeline_v1/NN_<type>/
```
Where `NN_<type>` is a zero‑padded index like `04_narration`, `05_dialog`, etc. Inside you’ll find:
- `brainstorm.txt` — cumulative bullet list you edit directly.
- `touch_point_first_draft.txt` — first draft prose or dialog (created after generation).
- `first_suggestions.txt` — initial suggestions for improvements (paired with the first draft).
- Additional step logs like `*_ordering.txt`, `*_actor_assignment.txt`, `*_dialog_batch.txt` depending on the pipeline.

### Brainstorm step details
When you first run a chapter, the program creates or appends bullets in `brainstorm.txt` for the current Touch‑Point and then prints: “Brainstorming still in progress.” It exits so you can:
- Delete weak or redundant bullets.
- Add your own bullet lines (start them with `*` or `-`).
- Keep bullets short but meaningful (one clear intent per bullet).
- Finally add a line containing only: `DONE` (all caps). That marks the brainstorm complete.

Re‑run the same command. The pipeline detects the trailing `DONE`, strips it internally, and moves to ordering.

Tip: If you re‑run before adding `DONE`, the model appends NEW bullets below the existing list. Use this to refine variety, but keep the list focused—too many bullets slow ordering and make the chapter meander. If the chapter needs to be longer, and touch on new ideas or situtations, add it is better to add more touch-points instead of dozens of bullets.

Edge case: `DONE.` (with punctuation) is accepted. The program normalizes a few common trailing marks.

### Ordering stage
Ordering rewrites the bullet list into a sensible sequence and may lightly tighten phrasing. If ordering is disabled (see Environment Flags below), the original brainstorm order is used as‑is.

Environment flags (set to "1" to skip ordering for that type):
- `GW_DISABLE_ORDERING_DIALOG` — skips ordering for dialog AND also applies to other types in some paths.
- `GW_DISABLE_ORDERING_MIXED`
- `GW_DISABLE_ORDERING_IMPLICIT`

### Narration pipeline specifics
Sequence:
1. Brainstorm bullets (`narration_brain_storm_prompt.md`).
2. Ordering (`ordering_prompt.md`).
3. Generate prose directly (`generate_narration_prompt.md`).
4. Produce first‑draft gate files and pause.

No actor assignment or body language is needed; it’s pure descriptive prose. Dialog history isn’t updated for narration.

### Dialog pipeline specifics
Sequence:
1. Brainstorm dialog intent bullets (`dialog_brain_storm_prompt.md`). Short filler bullets may be filtered—keep each bullet specific (e.g., motivation, emotional beat, or reveal).
2. Ordering.
3. Actor assignment: each bullet turned into `id: line‑intent` pairs.
4. Body language: generic physical cues list.
5. Agenda: per‑actor goals (grouped under actor names).
6. Reactions: a reaction for each line referencing the previous one.
7. Batch generation: all lines produced together; narrator lines become prose without quotes; character lines get body language merged inline.
8. First‑draft gate pause.

Dialog continuity: The system keeps a short history per actor so later dialog Touch‑Points can echo tone without repeating lines.

### Mixed pipeline specifics
Mixed Touch‑Points interleave brief narration/action beats and dialog lines.
Sequence mirrors the dialog pipeline with wording tuned for “mixed” intents:
1. Brainstorm mixed action + dialog bullets (`mixed_brain_storm_prompt.md`).
2. Ordering.
3. Actor assignment (Narrator allowed for small action prose beats).
4. Body language.
5. Agenda.
6. Reactions.
7. Batch generation (Narrator lines become unquoted prose; others merge body language).
8. First‑draft gate pause.

### Implicit pipeline specifics
Implicit dialog focuses on subtext and indirect speech.
Sequence:
1. Brainstorm implicit/subtext bullets (`implicit_brain_storm_prompt.md`).
2. Ordering.
3. Actor assignment (Narrator allowed for subtle narrative interjections).
4. Body language (often more important—nonverbal hints matter here).
5. Agenda.
6. Reactions (subtextual interplay; references prior line).
7. Generation (may run per‑line or batch depending on `GW_DIALOG_BATCH_MODE`).
8. First‑draft gate pause.

Batch mode: Setting `GW_DIALOG_BATCH_MODE=1` can cause implicit dialog to generate all lines in one structured batch for tighter cohesion.

### First‑draft gate (preview)
After generation the pipeline writes:
- `touch_point_first_draft.txt`
- `first_suggestions.txt`
Then it stops. You’ll learn how to edit these files and resume in Chapter 6. For now: just know the pause is intentional.

### Working cycle example (narration Touch‑Point)
1. Run:
   ```bash
   python -m ghostwriter.cli run CHAPTER_001.yaml --book-base my_first_book
   ```
   Program halts at narration brainstorm.
2. Open `iterations/CHAPTER_001/pipeline_v1/NN_narration/brainstorm.txt`.
3. Edit bullets; add your own; final line: `DONE`.
4. Re‑run the same command: ordering + generation occur; program pauses again with first‑draft gate.
5. (Stop here—Chapter 6 tells you what to do next.)

### Quality of bullets: quick advice
- One concept per bullet; avoid multi‑sentence rambling.
- Prefer active phrasing: “Hero notices claw marks” not “There are claw marks the hero might notice.”
- For dialog intent bullets: write what the speaker tries to achieve (“Wolf redirects topic toward Grandma”).
- For implicit: hint at motive (“Hero deflects question; hides worry”).

### Troubleshooting
- Program keeps saying “Brainstorming still in progress”: You forgot the final `DONE` line.
- Ordering feels identical to brainstorm: Possibly an ordering flag disabled, or bullet list too small.
- Body language seems mismatched: Edit the final prose later using the first‑draft gate (Chapter 6) rather than forcing exact phrasing in brainstorm bullets.
- A narrator line appears quoted: Rare—remove quotes when editing first draft; subtle edit will maintain your change.

### Try it now (mixed Touch‑Point)
In `my_first_book/chapters/CHAPTER_001.yaml`, add a new mixed Touch‑Point at the end:
```yaml
  - mixed: "Hero and Guide trade quick lines while walking past a strange tree"
```
Run the chapter. Handle the brainstorm pause, add `DONE`, re‑run, and observe the created `touch_point_first_draft.txt` and `first_suggestions.txt` for that mixed step. Don’t edit them yet—Chapter 6 is next.

---

## Chapter 6 — First draft gate

Goal: Make quick, human edits to the just‑generated text and focus the AI with your own suggestions before refinement.

When this happens
- After a content Touch‑Point (narration/dialog/mixed/implicit) generates text in v1, the program stops and prints: “Waiting for user suggestions on first draft.”
- Two files are created in the Touch‑Point’s folder:
  - `touch_point_first_draft.txt` — the raw first draft for this one beat.
  - `first_suggestions.txt` — a concise list of fixes or improvements the system proposes.

Where to find them
```
iterations/CHAPTER_xxx/pipeline_v1/NN_<type>/
```
Examples: `04_narration`, `05_dialog`, `06_mixed`, `07_implicit` (numbers vary by chapter outline).

What to edit
- In `touch_point_first_draft.txt`:
  - Fix phrasing, tone, or small plot details.
  - Add or delete lines, but keep the scope of the beat the same.
  - Dialog: keep character voice; the system tracks recent lines to maintain continuity later.
- In `first_suggestions.txt`:
  - Trim or rewrite to be actionable (“tighten pacing in the middle; cut repetition about the tree”).
  - Add your own suggestions; delete those you don’t want.
  - Keep it short. 3–7 bullets is ideal.

How to resume
1) Save both files.
2) Re‑run the same command you used for this chapter and version:
```bash
python -m ghostwriter.cli run CHAPTER_001.yaml --book-base my_first_book
```
3) The system will read your edited first draft + suggestions and move into the subtle edit step (Chapter 7).

Tips
- Prefer editing the first draft file at this stage, not the downstream `touch_point_draft.txt` (that file is produced after the subtle edit pass).
- If you accidentally remove the gate files during v1: the program can reconstruct them from the latest `touch_point_draft.txt`/`suggestions.txt` and pause again. On later versions (v2+), gates are not reconstructed.

### Try it now (tighten a dialog beat)
1) Run your chapter until it pauses on a dialog step.
2) Open that step’s folder and edit `touch_point_first_draft.txt` — remove a filler line; sharpen one reply.
3) Edit `first_suggestions.txt` — add “avoid repeating the tree detail; keep subtext playful.”
4) Re‑run. You’ll see the refined result in Chapter 7’s outputs.

---

## Chapter 7 — Subtle edit phase

Goal: Let GhostWriter apply targeted edits using your first‑draft changes and your curated suggestions to produce the beat’s refined text and fresh suggestions.

Inputs it uses
- Your edited `touch_point_first_draft.txt` and `first_suggestions.txt`.
- Local context captured at generation time: active actors, scene, foreshadowing; selected character YAML; and a small window of nearby polished text.

What it produces
- `touch_point_draft.txt` — the refined result for this one Touch‑Point.
- `suggestions.txt` — an updated list of actionable suggestions after refinement.

What happens next
- The program saves a checkpoint and proceeds to the next Touch‑Point. When all beats are done, it writes chapter‑level files:
  - `draft_v1.txt` — parseable records of each Touch‑Point’s final text.
  - `final.txt` — the clean chapter (polished text only).
  - `story_so_far.txt` and `story_relative_to.txt` — summaries regenerated from the chapter.

How to iterate further on this beat
- If you still want changes scoped to this beat during v1, return to the gate: edit `touch_point_first_draft.txt` and/or `first_suggestions.txt` again and re‑run. The subtle edit step will re‑apply.
- For broader or repeated edits across the whole chapter, use the vN edit flow (Chapter 8).

Troubleshooting
- I don’t see `suggestions.txt`: the system attempts to create it from the check artifacts; if missing, re‑run once. You can always edit the gate files and run again to regenerate.
- My exact wording change didn’t stick: make that wording explicit in the first draft, and add a clear bullet in `first_suggestions.txt` (“preserve this exact line”). Then re‑run.

### Try it now (apply your suggestions)
After finishing Chapter 6’s try‑it, re‑run the same command. Open the same step folder and compare:
- `touch_point_first_draft.txt` (your edited input)
- `touch_point_draft.txt` (refined output)
- `first_suggestions.txt` → `suggestions.txt` (updated guidance)

---

## Chapter 8 — vN edit iteration

Goal: Apply targeted improvements across a chapter by building on the previous version’s results without re‑doing v1 gates.

When to use it
- You’ve completed a first pass (v1) and want to refine specific beats (Touch‑Points) or the overall flow.
- You want the system to respect your last version and make subtle, guided edits.

How it works under the hood
- When a prior version exists, GhostWriter switches to the edit branch.
- For each content Touch‑Point (narration/dialog/mixed/implicit), it runs a subtle‑edit step instead of brainstorming/generation.
- Inputs for each beat come from the previous version:
  - Preferred: that beat’s `touch_point_draft.txt` and `suggestions.txt` under `pipeline_v(N-1)/NN_<type>/`.
  - Fallback: the chapter‑level `draft_v(N-1).txt` and `suggestions_v(N-1).txt`.
- Outputs for the new version (vN):
  - `iterations/CHAPTER_xxx/pipeline_vN/NN_<type>/touch_point_draft.txt`
  - `iterations/CHAPTER_xxx/pipeline_vN/NN_<type>/suggestions.txt`
  - Chapter aggregates: `draft_vN.txt`, `final.txt`, and refreshed summaries.
- Note: vN runs do not recreate v1 gates or brainstorm steps; they apply edits directly.

Where to focus edits before starting vN
- Per‑beat (recommended): edit the previous version’s files for just the beats you care about:
  - `.../pipeline_v(N-1)/NN_<type>/touch_point_draft.txt` (change the text you want preserved/shaped)
  - `.../pipeline_v(N-1)/NN_<type>/suggestions.txt` (add or refine bullets; keep them actionable)
- These edits guide the subtle‑edit pass for only those beats; untouched beats are lightly preserved.

Running the next version
- Omit the version to auto‑pick the next, or specify it:
```bash
python -m ghostwriter.cli run CHAPTER_001.yaml --book-base my_first_book
# or explicitly
python -m ghostwriter.cli run CHAPTER_001.yaml v2 --book-base my_first_book
```

What to expect
- The system writes new per‑beat drafts and suggestions under `pipeline_vN/`.
- Chapter‑level `draft_vN.txt` and `final.txt` are updated.
- No brainstorm pauses; no first‑draft gate in vN.

### Try it now (target one beat)
1) Open `iterations/CHAPTER_001/pipeline_v1/NN_dialog/touch_point_draft.txt` and tighten one reply. Also add a bullet in that folder’s `suggestions.txt` (e.g., “keep subtext playful; trim filler”).
2) Run the chapter again without specifying a version (it will pick v2). 
3) Compare `pipeline_v2/NN_dialog/touch_point_draft.txt` to see the refined line, and check the updated `suggestions.txt`.

Tips
- Keep suggestions short and concrete (3–7 bullets). You can iterate v3, v4… repeating the pattern.
- If you need broad, multi‑beat rewrites, consider Chapter 9’s global editing instead of many tiny beat tweaks.

---

## Chapter 9 — Global chapter editing

Goal: Make sweeping edits in one place (`draft_vN.txt`) and let GhostWriter reconcile changes back into each beat with fresh suggestions.

What you edit
- Open the latest `iterations/CHAPTER_xxx/draft_vN.txt` and change any paragraphs you want. Treat it as the “whole chapter” edit surface.

What happens on the next run
- GhostWriter detects your global edits and distributes them back into the per‑Touch‑Point files:
  - Updates each affected `.../pipeline_vN/NN_<type>/touch_point_draft.txt`.
  - Regenerates `.../pipeline_vN/NN_<type>/suggestions.txt` for those beats.
- It then pauses with a clear message so you can review the reconciled suggestions. Re‑run to continue the iteration.
- `final.txt` is rebuilt from the reconciled per‑beat drafts to keep everything consistent.
- If chapter summaries are missing, they’re regenerated from the reconciled text.

Workflow
1) Edit `draft_vN.txt` directly (large‑scale improvements are welcome).
2) Run the same chapter again (auto‑selects vN+1 unless you specify).
3) Read the updated per‑beat suggestions; if needed, tweak any `touch_point_draft.txt` or suggestions and run once more.

### Try it now (global tone pass)
1) Open `iterations/CHAPTER_001/draft_v1.txt` and adjust a few paragraphs for tone.
2) Run the chapter again to v2:
```bash
python -m ghostwriter.cli run CHAPTER_001.yaml --book-base my_first_book
```
3) Notice the pause after reconciliation. Inspect any `pipeline_v2/NN_<type>/suggestions.txt` that changed. Re‑run to proceed when satisfied.

Notes
- v2+ runs do not recreate v1 brainstorm or first‑draft gates; they respect your latest text and suggestions.
- The reconcile step only updates beats whose text actually changed in `draft_vN.txt`.

---

## Chapter 10 — Practical user stories

Let’s put everything together with real scenarios. Each story shows what to edit, what to run, and why it works. Keep edits small and focused unless you’re doing a global pass.

### Scenario 1 — Speed‑fix a chatty dialog beat

Situation
- One dialog Touch‑Point rambles. You want quicker replies and less repetition.

Do this
1) Open the beat’s first‑draft gate files from v1:
   - `iterations/CHAPTER_001/pipeline_v1/NN_dialog/touch_point_first_draft.txt`
   - `iterations/CHAPTER_001/pipeline_v1/NN_dialog/first_suggestions.txt`
2) Edit the first draft: delete filler lines, sharpen one reply, keep voices intact.
3) Edit `first_suggestions.txt`: add “trim filler; keep subtext playful; avoid repeating the tree detail.”
4) Re‑run the chapter. The subtle edit step (Chapter 7) applies your changes.

Why this works
- The v1 gate is the fastest way to nudge one beat without touching the outline or other beats.

Tip
- If you already finished v1, do the same but on v2+ by editing the previous version’s per‑beat files (`pipeline_v1/.../touch_point_draft.txt` and `suggestions.txt`) before running v2 (see Chapter 8).

### Scenario 2 — Add a new scene mid‑chapter

Situation
- Your chapter needs a short scene between two beats: a quick mixed action + a couple lines.

Do this
1) Edit `chapters/CHAPTER_001.yaml` and insert a new Touch‑Point where it belongs, for example:
   ```yaml
   - mixed: "They walk past a strange tree; a quick joke breaks the tension"
   ```
   (Optionally add a `scene` touch‑point first if the location changes.)
2) Run v1 for this chapter. You’ll hit the brainstorm gate for the new beat.
3) Edit `brainstorm.txt`, add `DONE`, re‑run, then use the first‑draft gate to refine.

Why this works
- New beats are planned in the chapter YAML (Chapter 3) and flow through the v1 pipeline (Chapter 5–7) without disturbing finished beats.

Tip
- If adding several new beats, consider a quick outline refresh with a `{ brainstorming: true }` touch‑point to restructure the list, then remove it and proceed.

### Scenario 3 — Fix continuity across three beats

Situation
- A nickname and a small fact drift across three beats. You want one edit to unify all mentions.

Do this
1) Open `iterations/CHAPTER_001/draft_v1.txt` (or latest `draft_vN.txt`).
2) Make the global fixes directly in this file: set the nickname once, correct the fact everywhere.
3) Run the chapter. GhostWriter reconciles the global changes back into each beat and regenerates suggestions for affected beats.
4) Review the updated `pipeline_v(N+1)/NN_<type>/suggestions.txt` and re‑run to accept.

Why this works
- Global edits (Chapter 9) are best for multi‑beat consistency. Reconciliation updates only the beats you changed.

Tip
- If a beat still needs stylistic polish, tweak its per‑beat `touch_point_draft.txt` and re‑run for a clean vN.

### Scenario 4 — Big tone overhaul after feedback

Situation
- An editor asks for a warmer narrator voice and lighter pacing across the chapter.

Do this
1) Open `draft_vN.txt` and rewrite key paragraphs to demonstrate the target tone.
2) Run the chapter to reconcile. Read the regenerated per‑beat suggestions.
3) If some beats still miss the tone, edit those beats’ `touch_point_draft.txt` and add explicit suggestion bullets (e.g., “keep warmth, avoid cynicism”). Run again.

Why this works
- Mixing a global pass (to set tone) with a short targeted per‑beat pass (to lock tone) is faster than trying to push everything through brainstorm.

Tip
- If the structure itself feels wrong, revisit Chapter 3 to refresh the outline before more prose edits.

### Quick decision guide — which knob to turn?

- Change the plan: Edit chapter YAML; use `{ brainstorming: true }` for outline refresh (Chapter 3).
- Add brand‑new prose: Insert a new beat and run v1 (Chapters 5–7).
- Nudge one beat’s wording in v1: Edit first‑draft gate files and re‑run (Chapter 6–7).
- Refine a few beats after v1: Edit prior version’s per‑beat `touch_point_draft.txt` + `suggestions.txt`, then run vN (Chapter 8).
- Fix chapter‑wide tone/consistency: Edit `draft_vN.txt`, run, then review reconciled suggestions (Chapter 9).

### Try‑its

1) Targeted dialog tighten
   - In `pipeline_v1/NN_dialog/`, remove one filler line from `touch_point_first_draft.txt` and add a bullet to `first_suggestions.txt`.
   - Re‑run and compare to `touch_point_draft.txt`.

2) Global nickname sweep
   - In `draft_v1.txt`, change a character’s nickname everywhere it appears.
   - Run once to reconcile; inspect the changed beats’ `suggestions.txt` and re‑run to finish.

### Troubleshooting

- I edited `draft_vN.txt` but nothing changed.
  - You must run the chapter to trigger reconciliation. Also ensure you edited the latest version’s draft file.
- My manual wording disappeared.
  - Edits in `draft_vN.txt` or per‑beat `touch_point_draft.txt` are preserved and reconciled forward. Make sure you didn’t edit an older version by mistake.
- Ordering feels off for a small beat.
  - In v1, add clearer bullets then `DONE`. If needed, disable ordering with env flags for experimentation; restore defaults afterward (Chapter 5).
- Dialog lines show quotes on narrator lines.
  - Remove the quotes in your beat draft; the system will keep them as prose on future runs.

---

You’ve finished the user guide! You can now plan chapters, run v1 with human gates, iterate with vN, and use global reconciliation to keep everything tidy. Happy writing.
