# GhostWriter Project TODO

This document outlines the tasks necessary to get the driver.py working correctly and establish a complete development environment for the GhostWriter novel generation system.

## Environment Setup

1. **Create Python Virtual Environment**
    - [x] Create a virtual environment in the project root: `python -m venv venv` (created with `python3` where needed)
    - [x] Document activation commands for different platforms (Linux/macOS: `source venv/bin/activate`, Windows: `venv\Scripts\activate`)
    - [x] `.gitignore` already excludes common env directories including `venv/`

2. **Install Required Python Dependencies**
    - [x] Install PyYAML for YAML file processing: `pip install PyYAML`
    - [x] Install OpenAI Python client for LLM integration: `pip install openai`
    - [x] Install python-dotenv for environment variable management: `pip install python-dotenv`
    - [x] Install pytest for unit testing: `pip install pytest pytest-mock`
    - [x] Create `requirements.txt` file with all dependencies and pinned versions

3. **Update README.md with Python Environment Section**
    - [x] Add "Prerequisites" section with Python version requirements
    - [x] Add "Installation" section with virtual environment setup instructions
    - [x] Add "Dependencies" section listing all required Python libraries
    - [x] Add usage examples and command-line interface documentation

## Core Functionality Implementation

4. **Implement LLM Integration in driver.py**
   - Replace placeholder comments with actual OpenAI API calls
   - Add API key configuration via environment variables (.env file)
   - Implement error handling for API failures, rate limits, and network issues
   - Add retry logic with exponential backoff for robust API interactions

5. **Fix Prompt Template System (with Templating Loop + Polish)**
     - Update `build_prompt()` function to handle all placeholder replacements:
         - `[story_so_far.txt]` - read from iterations/CHAPTER_XXX/story_so_far.txt
         - `[story_relative_to.txt]` - read from iterations/CHAPTER_XXX/story_relative_to.txt
         - `[draft_v?.txt]` - read most recent draft version
         - `[suggestions_v?.txt]` - read most recent suggestions
         - `[check_v?.txt]` - read most recent check results
         - `[predraft_v?.txt]` - current draft being evaluated
     - Create separate functions for each prompt type (initial, master, polish, check, story-so-far, story-relative-to)
     - Ensure master prompts output a pre-prose document (pre_draft) that includes:
         - A top section of CHARACTER TEMPLATE blocks for all characters needed this chapter
         - In-line CHARACTER call placeholders embedded in the pre-prose body
     - Implement the templating loop:
         - Parse CHARACTER TEMPLATE definitions and scan the pre-draft for CHARACTER call sites
         - For each call, build a focused character prompt (using cadence, lexicon, prefer/avoid, mannerisms, sample_lines, max_tokens_line, temperature_hint) and get the dialog via LLM
         - Substitute each CHARACTER call with the generated dialog, preserving surrounding narrative formatting
         - Use robust, explicit delimiters for templates and calls to avoid accidental matches and reduce prompt-injection risk (match README syntax)
         - Handle missing/unknown character templates gracefully (configurable: warn/skip/fail)
     - Add the polish prose step:
         - Feed the fully substituted text into `prompts/polish_prose_prompt.md`
         - Return a cleaned, well-formatted `draft_vN.txt` with consistent style and transitions
     - Logging and observability:
         - Log discovered templates and number of call sites; optionally provide a `--dry-run` to preview substitutions
         - Support configurable parallelism and timeouts for per-character calls

6. **Implement Complete Iteration Loop**
    - Add logic to determine if this is the first iteration (use master_initial_prompt.md) or subsequent (use master_prompt.md)
    - Implement version numbering system (v1, v2, v3, etc.)
    - Add automatic iteration based on verification results ("missing" touch-points trigger re-draft)
    - Create functions for story-so-far and story-relative-to generation between chapters
    - Insert a two-phase generation flow:
      1) Generate `pre_draft_vN.txt` from master prompt containing CHARACTER TEMPLATEs and CHARACTER calls.
      2) Execute per-character LLM calls and substitute results; then run a Polish Prose prompt to produce `draft_vN.txt`.

7. **Add Missing Core Functions**
    - `generate_story_so_far()` - uses story_so_far_prompt.md to create summary
    - `generate_story_relative_to()` - uses story_relative_to_prompt.md for character perspectives
    - `check_iteration_complete()` - determines if all touch-points are satisfied
    - `get_latest_version()` - finds the highest version number for a chapter
    - `parse_character_blocks()` - extracts CHARACTER TEMPLATE definitions and in-text CHARACTER call sites from `pre_draft_vN.txt`.
    - `render_character_call(character_id, call_context)` - constructs and sends a focused LLM prompt using the character’s template.
    - `substitute_character_calls(pre_draft_text, responses)` - replaces call sites with generated dialog.
    - `polish_prose(text)` - sends the substituted text to `polish_prose_prompt.md` and returns polished prose.

8. **Improve File and Directory Management**
   - Add validation for required input files (SETTING.yaml, CHAPTER_XX.yaml)
   - Create iterations directory structure automatically
   - Add file existence checks before attempting to read files
   - Implement proper error messages for missing files

## Data Structure and Configuration

9. **Create Example SETTING.yaml**
   - Create a sample SETTING.yaml file with proper structure (Factoids, Scenes, Characters, Props)
   - Include comprehensive examples for each section
   - Document the YAML schema and validation rules

10. **Create Example Chapter Files**
    - Create CHAPTER_01.yaml with proper Touch-Points, Story-So-Far, and Story-Relative-To structure
    - Create CHAPTER_02.yaml to demonstrate chapter progression
    - Include examples of explicit and implicit touch-points

11. **Add Configuration Management**
    - Create config.yaml for system settings (API endpoints, model parameters, retry limits)
    - Add command-line argument parsing for better CLI interface
    - Implement logging configuration for debugging and monitoring
    - Add character-render settings (max parallel calls, per-character token/temperature caps, refusal policy if template is missing)
    - Add safety filters for dialog substitution (max line length, allowed character set, profanity filter toggle)

11a. **Add Prompt Templates for New Flow**
    - Create `prompts/polish_prose_prompt.md` (used after substitution)
    - Update `prompts/master_initial_prompt.md` and `prompts/master_prompt.md` to emit:
      - A top section with all CHARACTER TEMPLATE blocks (for any characters needed in the chapter)
      - In-line CHARACTER call placeholders in the pre-prose body
    - Document the exact placeholder syntax in README and ensure unit tests cover parsing.

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