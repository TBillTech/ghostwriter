# GhostWriter Project TODO

This document outlines the tasks necessary to get the driver.py working correctly and establish a complete development environment for the GhostWriter novel generation system.

## New Features

1. **CHAPTER_xxx brainstorming pipeline***
    - [x] If the ghostwriter.cli is run with CHAPTER_XYZ.yaml, but the CHAPTER_XYZ.yaml file is missing (where XYZ is the chapter number), then trigger this new pipeline
    - [x] If the ghostwriter.cli is run with CHAPTER_XYZ.yaml, and if there is a touch-point called : "- brainstorming: True", then trigger this new pipeline
    - [x] The new CHAPTER_xxx brainstorming pipeline should use a new prompt template called brainstorm_chapter_outline.md
    - [x] The new CHAPTER_xxx brainstorming pipeline prompt template should provide the following information to the LLM:
        - The dereferenced factoids and characters if the CHAPTER_XYZ.yaml is present and has a setting block
        - If the setting block is absent, or CHAPTER_XYZ.yaml is not present, then instead provide the entire list of character names, and the entire list of factoid names (But not the full character data or the full factoid data). This is so that the brainstorming session can pick up likely factoids and characters to put into a new brainstormed setting block.
        - The story_so_far
        - [x] If the CHAPTER_XYZ.yaml is not present, then careful instructions on the format desired for the CHAPTER_XYZ.yaml. This should not include the "Story-So-Far" or "Story-Relative-To" parts
        - CONTENT_TABLE.yaml
        - [x] The current version of CHAPTER_XYZ.yaml if present. Do NOT include the "Story-So-Far" or "Story-Relative-To" parts with the CHAPTER_XYZ.yaml provided in the prompt
        - [x] Explicit instructions to look at the synopsis in for version XYZ in the CONTENT_TABLE.yaml, and brainstorm an exciting and engaging chapter for the book, at the high level direction level.
    - [x] The old version of the CHAPTER_XYZ.yaml should be copied to a prior version like CHAPTER_XYZ.9.yaml, if this was the ninth brainstorm cycle.
    - [x] Save the prompt + result in CHAPTER_XYZ.txt
    - [x] Save result from the LLM plus the story so far and story relative to from the prior completed chapter in iterations/CHAPTER_(XYZ-1)/pipeline_v2/ 
    - [x] After the pipeline is done, the program should stop. This will result in exactly one chapter outline being updated.

2. **CHARACTER brainstorming pipeline**
    - [x] When ghostwriter.cli is run with CHAPTER_XYZ.yaml, test the contents of the actors in the setting block.
    - [x] If an actor name is present in the setting block that is NOT in in the CHARACTERS context, then kick off this new brainstorming pipeline focussed on that character. In addition, prompt the user for a text description of the character they should type in the terminal and save this text to a user_description variable.
    - [x] Alternatively, if in the CHARACTERS context, there is a character with the field "brainstorming: True", then kick off this new brainstorming pipeline focussed on that character. Use the data already provided from the CHARACTERS context to initialize the user_description variable without prompting the user in the terminal. 
    - [x] The new CHARACTER brainstorming pipeline should use a new prompt template called brainstorm_character_outline.md
    - [x] The new CHARACTER brainstorming pipeline prompt template should provide the following information to the LLM:
        - The dereferenced factoids and characters from the CHAPTER_XYZ.yaml setting block (except of course for other characters that are missing.)
        - The CHAPTER_XYZ.yaml
        - Clear instructions on the output format of the CHARACTER which should match that in LRRH book CHARACTERS.yaml. You can use Red's character as an example.
        - The user_description from either the terminal or the existing CHARACTER context, followed by "Now, for the character <NAME>,  brainstorm an interesting character outline from the prior description".  Note that <NAME> sould be replaced with the character actor name being brainstormed.
    - [x] Save the LLM prompt + LLM result in character_brainstorm.txt in the book base operating directory (overwrite if necessary)
    - [x] Simply append the LLM result to the CHARACTERS.yaml, and make sure it is properly tabbed to be correct yaml.
    - [x] After the pipeline is done, the program should stop. This will result in exactly one character outline being updated.

## Testing and Quality Assurance

11. **Fully configurable file locations**
    - [x] Add a new ENV for the book base operating directory (GW_BOOK_BASE_DIR)
    - [x] Add a new optional argument to override this (`--book-base` in CLI)
    - [x] Define ENV vars for SETTING.yaml path, CHARACTERS.yaml path, chapters path, and iterations path
    - [x] Make these ENV vars default to expected locations in the base operating directory 
    - [x] Support moving these four paths into `BecomingDjinn/` (or any base) via GW_BOOK_BASE_DIR
    - [x] Create new testbed directory for later unit testing and integration testing steps
    - [x] Create a set of test files in this directory (Little Red Riding Hood book)
    - [x] Modify unit test scaffolding to allow copying LRRH into a sandbox (pytest fixtures in `tests/conftest.py`)
    - [x] Update README.md to document the new path variables, and add `.env.example` pointing to LittleRedRidingHood

12. **Create Unit Tests**
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
    - Rely on the fact that for testing, we will use the Little Red Riding Hood book
    - Use the LRRH text as the return values for the MockLLM.  This way, the outputs can be tested in detail.
    - Create MockLLM class that returns predictable responses
    - Implement different response scenarios (successful, missing touch-points, API errors)
    - Add per-character response fixtures to simulate distinct voices/cadences
    - Use for unit testing without requiring actual API calls

15. **Create Test Data**
    - Create test SETTING.yaml and CHAPTER files in tests/ directory
    - Create expected output files for verification
    - Set up pytest fixtures for consistent test data


## Completion Criteria

- [ ] All core functions implemented and working
- [ ] Complete test suite with >90% code coverage
- [x] Documentation updated and comprehensive (paths + example)
- [x] Example files created and tested (Little Red Riding Hood)
- [ ] Virtual environment and dependencies properly configured
- [ ] Error handling robust and user-friendly
- [ ] Iteration loop fully functional with real LLM integration

