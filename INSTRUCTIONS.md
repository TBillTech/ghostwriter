# INSTRUCTIONS for AI collaborator

Use this document to guide your work in this repository. Keep your responses short, concrete, and actionable.

The user can invoke following these instructions, which means to perfectly follow the details under the Next steps section below by saying: 

Follow instructions

## Next steps

This program now works after a fashion, but I am dissapointed by the inconsistency, lack of creativity, and minimal dialog as opposed to too much narration in the result.  We need to update the README.md with an explanation of the following algorithm. We also need to add all these requirements to the TODO.md, and then create tasks in the TODO.md to implement these requirements.  I have already created all the prompt template files under the prompts directory, so please examine these for how to fill the templates out.

New direction:
* For now, remove the check_prompt, the master_intial_prompt and the master_prompt from the sequence.
* For now, the sequence will be deterministic and the steps sequential. We will not loop automatically, except for the implied loops over the touch-points as described below.
* There are going to be two operating branches for the program.  One, is when there is NOT already a draft_v1.txt in the iterations/CHAPTER_xxx directory.  The other state is when there IS the draft_v1.txt .
* There are now even more prompts in the prompts directory, and they need corresponding environment variables for LLM model, temperature, and max tokens, just like the previous prompts had.

### New Requirements

WHEN draft_v1.txt is missing the sequence needs to be as follows: 
1) Read the touch-points from the chapter into a list, parse each touch point as one of several commands: "actors", "narration", "explicit", "implicit".  "actors" sets which actors are active until the next actors command, "narration" creates narrative prose, "explicit" creates dialog where characters explicitly discuss and tell each other things, and "implicit" creates dialog where characters usually dance around an idea and don't always seem to address each other as they expect.   
2) For each touch-point in the chapter do the following:
   2.1) If "actors" command, update the state, and continue with the next touch-point
   2.2) If "scene" command, update the state, and continue wiht the next touch-point
   2.3) If "foreshadowing" command, update the state, and continue with the next touch-point
   2.4) If "narration", feed the state to the new "narration" pipeline
   2.5) If "explicit", feed the state to the new "explicit" pipeline
   2.6) If "implicit", feed the state to the new "implicit" pipeline
   Note: For "explicit" and "implicit" pipelines, the program state needs to track the lastest few lines of dialog coming from each actor.
3) Use the polish_prose_prompt.md on each pipeline result
4) Combine the original touch-point and the result of each polish prompt from each touch-point and combine them all to the total draft_v1.txt file. Make sure the format is such that the file can be parsed to recover the touch-point and the polished result which it ultimately generated.
5) For each touch-point, load the corresponding check_xxxxxx_prompt.md and fill in the template, then concatenate all the results from sending these prompts to the LLM. Make it parseable just like the draft_v1.txt file, and save this in the suggestions_v1.txt file.
6) Generate the story_relative_to.txt and the story_so_far.txt
7) Generate a stripped down version of the whole text which includes only the polished text without touch-point or markdown suitable for publishing. Write this to final.txt

WHEN draft_vN.txt (and therefore suggestions_vN.txt) is present, look at the largest N:
1) Read the touch-points from the chapter into a list, parse each touch point as one of several commands: "actors", "narration", "explicit", "implicit".  "actors" sets which actors are active until the next actors command, "narration" creates narrative prose, "explicit" creates dialog where characters explicitly discuss and tell each other things, and "implicit" creates dialog where characters usually dance around an idea and don't always seem to address each other as they expect.
2) Read the draft_vN.txt and the suggestions_vN.txt, and add the polished texts and the suggestions to the state so that the edit pipeline can reference them.   
2) For each touch-point in the chapter do the following:
   2.1) If "actors" command, update the state, and continue with the next touch-point
   2.2) If "scene" command, update the state, and continue wiht the next touch-point
   2.3) If "foreshadowing" command, update the state, and continue with the next touch-point
   2.4) If "narration", "explicit" or "implicit" feed the state to the subtle edit pipeline
3) Use the polish_prose_prompt.md on each pipeline result
4) Combine the original touch-point and the result of each polish prompt from each touch-point and combine them all to the total draft_v(N+1).txt file. Make sure the format is such that the file can be parsed to recover the touch-point and the polished result which it ultimately generated.
5) For each touch-point, load the corresponding check_xxxxxx_prompt.md and fill in the template, then concatenate all the results from sending these prompts to the LLM. Make it parseable just like the draft_v1.txt file, and save this in the suggestions_v(N+1).txt file.
6) Re-Generate and overwrite the story_relative_to.txt and the story_so_far.txt
7) Re-Generate and overwrite the stripped down version of the whole text which includes only the polished text without touch-point or markdown suitable for publishing. Write this to final.txt

Each pipeline is a sequence of (template, output format) pairs. The expected output formats are:
* text (no parsing)
* bullet list (Each bullet is a line beginning with the '*' character, MUST have at least 2 bullets)
* actor list (Dialog lines start with the id of a character, MUST have at least 2 actors, not necessarily distinct.  Non-dialog identifying narrative bits in between dialog lines are allowed.)
* agenda list (See agenda_prompt.md for example)
To keep the algorithm stable, we will verify that the output formats are met, and if not, repeat up to twice. The third time, stop the program and report the error. We are also going to be well served if we log the fully substituted literal LLM prompts and outputs anytime we have a round trip with the llm.  Please use descriptive file names and a sub directory for these LLM logs.

New Narration pipeline:
* (brain_storm_prompt, bullet list)
* (ordering_prompt, bullet list)
* (generate_narration_prompt, text)

New Explicit pipeline:
* (brain_storm_prompt, bullet list)
* (ordering_prompt, bullet list)
* (actor_assignment_prompt, actor list)
* In parallel: (body_language_prompt, bullet list), (agenda_prompt, agenda list)
* Then, join the results of the actor assignment prompt, body_language_prompt, and agenda_prompt.  Using the joined data, for each actor line, fill in the character_dialog_prompt template, and collect the results into the text for the output of the new explicit pipeline.  Note that the actor list needs to be stored in the state to provide to many of the templates as a stream of consciousness dialog_lines.

New Implicit pipeline:
* (implicit_brain_storm_prompt, bullet list)
* (ordering_prompt, bullet list)
* (actor_assignment_prompt, actor list)
* In parallel: (body_language_prompt, bullet list), (agenda_prompt, agenda list)
* Then, join the results of the actor assignment prompt, body_language_prompt, and agenda_prompt.  Using the joined data, for each actor line, fill in the character_dialog_prompt template, and collect the results into the text for the output of the new explicit pipeline.  Note that the actor list needs to be stored in the state to provide to many of the templates as a stream of consciousness dialog_lines.

Subtle Edit pipeline:
* (subtle_edit_prompt, text)

Relevant `TODO.md` is in the project root
Relevant `README.md` is in the project root

## Before you start
- Carefully read the relevant `TODO.md` for open tasks (check root and relevant subfolders).
- Carefully read the relevant `README.md` (or `Readme.md`) for project details that aren’t obvious from code.

## How to work
- Extract explicit requirements into a small checklist and keep it updated until done.
- Prefer doing over asking; ask only when truly blocked. Make 1–2 reasonable assumptions and proceed if safe.
- Make minimal, focused changes. Don’t reformat unrelated code or change public APIs without need.
- Validate changes: build, run tests/linters, and report PASS/FAIL succinctly with key deltas only.
- After ~3–5 tool calls or when editing >3 files, post a compact progress checkpoint (what changed, what’s next).
- Use delta updates in conversation—avoid repeating unchanged plans.
- Always run scripts and tests using the repository virtual environment:
	- Activate first: `source venv/bin/activate` then `python ...`
	- Or call directly: `./venv/bin/python ...`

## Prioritization
- Prioritize items in `TODO.md` matching what we are working on during this session. If unclear, suggest small, high-impact fixes or docs/tests that clarify behavior, and get confirmation from the user. 

## Deliverables
- Provide complete, runnable edits (code + small test/runner if needed). Update docs when behavior changes.
- When commands are required, run them yourself and summarize results. Offer optional copyable commands.
- Wrap filenames in backticks, use Markdown headings and bullets, and keep explanations brief.

## Quality gates
- Build: no new compile errors.
- Lint/Typecheck: clean or noted exceptions.
- Tests: add/adjust minimal tests; ensure green.

## Style
- Keep it concise and impersonal. Use clear bullets and short paragraphs. Avoid filler.

## After you Finish

- If you made any code changes, ALWAYS run linter, and fix all errors and warnings UNLESS the prior instruction details specifically say otherwise.
- If you made any code changes, ALWAYS run unit tests.
- Run tests via the venv:
	- Activated: `pytest -q`
	- Or direct: `./venv/bin/python -m pytest -q`
- If there are ANY unit test failures, try hard to fix them all. If this seems too difficult consult with the user and get detailed about debugging.
- If you made any code changes, update the TODO.md and check all completed tasks.
- Update the session conversation summary at the end of TODO.md.
- Update the README.md with any findings that appeared during the session which are worth remarking on.  Be sure to preserve any solutions to command line issues, so we don't have to repeat broken command lines in the future.
- Update these `INSTRUCTIONS.md` by setting "Focus" to the next actionable tasks in `TODO.md` (e.g., move from Environment Setup to Core Functionality Implementation).
	- Current: Task 8 done; focus moved to Task 9.
- Finally, commit all file changes.