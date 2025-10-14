# GhostWriter Project TODO

This document outlines the tasks necessary to get the driver.py working correctly and establish a complete development environment for the GhostWriter novel generation system.



## New Direction: Deterministic pipelines and two-branch workflow

The following tasks capture requirements introduced in INSTRUCTIONS.md and extend the plan for the next implementation phase.

1. Architecture and State Machine
    - [x] Parse touch-points into commands: `actors`, `scene`, `foreshadowing`, `narration`, `explicit`, `implicit`
    - [x] Implement chapter processing state:
         - `active_actors: list[str]`
         - `current_scene: Optional[str]`
         - `foreshadowing: set[str]`
         - `dialog_history: dict[str, deque[str]]` (size N)
         - `prior_context: dict[touchpoint_id, {polished_text, suggestions}]` when N>0
    - [ ] Implement two branches:
         - Branch A (no `draft_v1.txt`): run Narration/Explicit/Implicit pipelines
         - Branch B (has `draft_vN.txt`): run Subtle Edit pipeline for prose-producing touch-points
    - [ ] Generate artifacts per branch: `draft_vN.txt`, `suggestions_vN.txt`, `story_so_far.txt`, `story_relative_to.txt`, `final.txt`

2. Pipelines
    - [x] Narration pipeline: Brainstorm → Ordering → Generate Narration → Polish (functions implemented)
    - [x] Explicit pipeline: Brainstorm → Ordering → Actor Assignment → (Body Language || Agenda) → Join → Character Dialog per actor line → Polish (functions implemented)
    - [x] Implicit pipeline: Implicit Brainstorm → Ordering → Actor Assignment → (Body Language || Agenda) → Join → Character Dialog per actor line → Polish (functions implemented)
    - [x] Subtle Edit pipeline: Subtle Edit → Polish (functions implemented)

3. Output format validators and retries
    - [x] bullet list: each line starts with `*`, min 2 bullets
    - [x] actor list: lines begin with `<actor_id>:`; min 2 actor attributions; allow narrative lines
    - [x] agenda list: conform to `prompts/agenda_prompt.md`
    - [x] text: accept any string
    - [x] Retry invalid outputs up to 3 attempts; on third failure, abort with error

4. Per-touch-point checking and suggestions
    - [ ] Use `check_narration_prompt.md`, `check_explicit_prompt.md`, `check_implicit_prompt.md`
    - [ ] Build a parseable `suggestions_vN.txt` with one record per touch-point

5. Parseable artifact formats
    - [x] Define sentinel-based records:
         - `BEGIN_TOUCHPOINT id=... type=...` / `END_TOUCHPOINT`
         - `BEGIN_RESULT` / `END_RESULT`
    - [x] Implement read/write helpers for `draft_vN.txt` and `suggestions_vN.txt`
    - [ ] Implement `final.txt` generator (polished-only)

6. LLM logging
    - [ ] Log fully substituted prompts and raw outputs for every call
    - [ ] For every call, also print brief statistics of input and output, and log this to console.
    - [ ] When logging to console, also log completion status of current check-point index out of total check-points.
    - [ ] Organize logs per chapter/version/touch-point/step
    - [ ] Redact secrets; ensure `.gitignore` covers logs if needed

7. Environment variables per prompt
    - [ ] Add per-prompt model/temperature/max_tokens for:
         - Brainstorm, Ordering, Generate Narration, Actor Assignment, Body Language, Agenda, Character Dialog
         - Subtle Edit, Polish Prose
         - Checks: Narration/Explicit/Implicit
         - Summaries: Story-So-Far, Story-Relative-To
    - [ ] Fallback to global defaults when unset

8. Driver integration and CLI
    - [x] Update `scripts/driver.py` to implement branch selection and pipelines
    - [x] Add `--log-llm` (or expand `--show-dialog`) to capture logs for all steps
    - [x] Keep existing input validation and directory scaffolding

9. Documentation
    - [ ] README: add deterministic architecture section, pipelines, formats, env vars (done)
    - [ ] Add `.env.example` entries for new variables
    - [ ] Document log directory structure and file naming

10. Testing
    - [ ] Unit: touch-point parser, state transitions, validators, file formats, env var resolution
    - [ ] Unit: pipelines with Mock LLM — success and retry/failure paths
    - [ ] Integration: Branch A and B end-to-end with Mock LLM
    - [ ] Finalization: `final.txt`, `story_so_far.txt`, `story_relative_to.txt`

11. Quality gates
    - [ ] Lint/typecheck for new modules
    - [ ] Deterministic test runs with fixed Mock LLM outputs


## Testing and Quality Assurance

12. **Create Unit Tests for driver.py**
    - Test YAML loading and validation functions
    - Test prompt template building with mock data
    - Test file I/O operations with temporary directories
    - Test error handling for missing files and invalid YAML
    - Test version numbering and iteration logic
    - Test CHARACTER parsing and substitution logic (happy path and malformed blocks)
    - Test polish step integration: given substituted text, ensure polish prompt is built and called correctly

13. **Create Integration Tests**
    - Test complete workflow from YAML to draft generation (with mocked LLM)
    - Test iteration loop with simulated verification results
    - Test story-so-far and story-relative-to generation
    - Test multi-chapter workflow
    - Test the two-phase pre-draft → substitution → polish pipeline end-to-end with a Mock LLM

14. **Add Mock LLM for Testing**
    - Create MockLLM class that returns predictable responses
    - Implement different response scenarios (successful, missing touch-points, API errors)
    - Add per-character response fixtures to simulate distinct voices/cadences
    - Use for unit testing without requiring actual API calls

15. **Create Test Data**
    - Create test SETTING.yaml and CHAPTER files in tests/ directory
    - Create expected output files for verification
    - Set up pytest fixtures for consistent test data

## Documentation and Developer Experience

16. **Add Comprehensive Error Handling**
    - Implement specific exception classes for different error types
    - Add user-friendly error messages with suggested solutions
    - Create error recovery strategies where possible

17. **Add Logging and Debugging**
    - Implement structured logging throughout the application
    - Add debug mode for verbose output
    - Log API calls, file operations, and iteration progress

18. **Create Developer Documentation**
    - Document the complete system architecture
    - Add code comments explaining complex logic
    - Create troubleshooting guide for common issues
    - Document the prompt engineering approach
    - Document Character Template schema and CHARACTER call syntax with examples
    - Add a README section for the two-phase generation and how to add new characters

19. **Security and Safety Considerations**
    - Ensure `.env` is ignored (already in .gitignore) and provide `.env.example`
    - Avoid leaking API keys in logs; redact sensitive fields in debug mode
    - Validate CHARACTER blocks to prevent prompt injection via templates or call sites
    - Add rate limit/backoff strategies for per-character calls; batch when possible

## Future Enhancements (Optional)

19. **Add Command-Line Interface Improvements**
    - Add `--dry-run` flag to preview operations without executing
    - Add `--continue` flag to resume interrupted iterations
    - Add `--chapter-range` to process multiple chapters
    - Add progress bars for long-running operations

20. **Add Validation and Quality Checks**
    - Validate YAML structure against expected schema
    - Check for required fields in SETTING.yaml and CHAPTER files
    - Warn about potential issues (empty touch-points, missing characters)
    - Add prose quality metrics (word count, readability scores)
    - Add style/voice conformance checks per-character (e.g., lexicon hits/misses)

## Completion Criteria

- [ ] All core functions implemented and working
- [ ] Complete test suite with >90% code coverage
- [ ] Documentation updated and comprehensive
- [ ] Example files created and tested
- [ ] Virtual environment and dependencies properly configured
- [ ] Error handling robust and user-friendly
- [ ] Iteration loop fully functional with real LLM integration

---

**Priority Order**: Complete tasks 1-8 first for basic functionality, then 9-15 for robustness, and finally 16-20 for polish and developer experience.

---

Session summary (env setup):
- Created and validated a Python virtual environment under `venv/`.
- Installed and pinned dependencies to `requirements.txt` (PyYAML, openai, python-dotenv, pytest, pytest-mock, and transitive deps).
- Updated `README.md` with environment setup, usage, and troubleshooting.
- Verified `.gitignore` already ignores `venv/` and related files.

---

Session status (2025-10-09):
- SETTING.yaml parsing errors resolved (converted long scalars to block scalars; normalized quotes/indentation).
- Driver end-to-end smoke test passes with mock LLM; artifacts generated:
    - pre_draft_v1.txt, check_v1.txt, draft_v1.txt, story_so_far.txt, story_relative_to.txt, suggestions_v1.txt
- Character substitution wired with template/call parsing; mock run shows missing templates when calls exist without preceding templates in the pre_draft.
- Iteration loop (initial → verify → polish) operational; touch-point check uses 'missing' heuristic.

---

Session status (2025-10-13):
- Implemented Task 8 validations and directory management in `scripts/driver.py`:
    - Early validation for `SETTING.yaml`, chapter YAML, and required prompt templates.
    - Automatic creation of `iterations/CHAPTER_xxx/` directory.
    - Clear, user-friendly error messages with custom exceptions.
    - Guarded file reads to avoid confusing stack traces for common issues.