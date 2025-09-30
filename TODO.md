# GhostWriter Project TODO

This document outlines the tasks necessary to get the driver.py working correctly and establish a complete development environment for the GhostWriter novel generation system.

## Environment Setup

1. **Create Python Virtual Environment**
   - Create a virtual environment in the project root: `python -m venv venv`
   - Document activation commands for different platforms (Linux/macOS: `source venv/bin/activate`, Windows: `venv\Scripts\activate`)
   - Update .gitignore to exclude venv/ directory

2. **Install Required Python Dependencies**
   - Install PyYAML for YAML file processing: `pip install PyYAML`
   - Install OpenAI Python client for LLM integration: `pip install openai`
   - Install python-dotenv for environment variable management: `pip install python-dotenv`
   - Install pytest for unit testing: `pip install pytest pytest-mock`
   - Create requirements.txt file with all dependencies and versions

3. **Update README.md with Python Environment Section**
   - Add "Prerequisites" section with Python version requirements
   - Add "Installation" section with virtual environment setup instructions
   - Add "Dependencies" section listing all required Python libraries
   - Add usage examples and command-line interface documentation

## Core Functionality Implementation

4. **Implement LLM Integration in driver.py**
   - Replace placeholder comments with actual OpenAI API calls
   - Add API key configuration via environment variables (.env file)
   - Implement error handling for API failures, rate limits, and network issues
   - Add retry logic with exponential backoff for robust API interactions

5. **Fix Prompt Template System**
   - Update `build_prompt()` function to handle all placeholder replacements:
     - `[story_so_far.txt]` - read from iterations/CHAPTER_XX/story_so_far.txt
     - `[story_relative_to.txt]` - read from iterations/CHAPTER_XX/story_relative_to.txt
     - `[draft_v?.txt]` - read most recent draft version
     - `[suggestions_v?.txt]` - read most recent suggestions
     - `[check_v?.txt]` - read most recent check results
     - `[predraft_v?.txt]` - current draft being evaluated
   - Create separate functions for each prompt type (initial, master, check, story-so-far, story-relative-to)

6. **Implement Complete Iteration Loop**
   - Add logic to determine if this is the first iteration (use master_initial_prompt.md) or subsequent (use master_prompt.md)
   - Implement version numbering system (v1, v2, v3, etc.)
   - Add automatic iteration based on verification results ("missing" touch-points trigger re-draft)
   - Create functions for story-so-far and story-relative-to generation between chapters

7. **Add Missing Core Functions**
   - `generate_story_so_far()` - uses story_so_far_prompt.md to create summary
   - `generate_story_relative_to()` - uses story_relative_to_prompt.md for character perspectives
   - `check_iteration_complete()` - determines if all touch-points are satisfied
   - `get_latest_version()` - finds the highest version number for a chapter

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

## Testing and Quality Assurance

12. **Create Unit Tests for driver.py**
    - Test YAML loading and validation functions
    - Test prompt template building with mock data
    - Test file I/O operations with temporary directories
    - Test error handling for missing files and invalid YAML
    - Test version numbering and iteration logic

13. **Create Integration Tests**
    - Test complete workflow from YAML to draft generation (with mocked LLM)
    - Test iteration loop with simulated verification results
    - Test story-so-far and story-relative-to generation
    - Test multi-chapter workflow

14. **Add Mock LLM for Testing**
    - Create MockLLM class that returns predictable responses
    - Implement different response scenarios (successful, missing touch-points, API errors)
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