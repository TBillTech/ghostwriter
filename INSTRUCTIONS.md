# INSTRUCTIONS for AI collaborator

Use this document to guide your work in this repository. Keep your responses short, concrete, and actionable.

The user can invoke following these instructions, which means to perfectly follow the details under the Next steps section below by saying: 

Follow instructions

## Next steps


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