You are an expert character designer. Brainstorm a single character YAML entry for the character named [TARGET_NAME].

Context:
- Full chapter YAML (for reference):

[CHAPTER_YAML]

- Dereferenced setting facts and referenced characters (excluding the target and any missing):

[SETTING_DEREF]

- Example character (provided only when no characters were dereferenced above):

[EXAMPLE_CHARACTER]

- User description to seed the brainstorm:

[USER_DESCRIPTION]

Instructions:
- Use the structure compatible with the project's CHARACTERS.yaml. If an example appears above, follow its style. Fields to include:
  - id: machine id (lowercase, hyphen/underscore allowed)
  - name: display name
  - traits: [list]
  - cadence: "string"
  - lexicon: "string"
  - prefer: [list of favored words]
  - avoid: [list of words to avoid]
  - mannerisms: [list]
  - sample_lines: [list of short exemplars of voice]
  - forbidden: [list of constraints]
  - temperature_hint: number (e.g., 0.25)
  - max_tokens_line: number (e.g., 90)
- For the traits, prefer, avoid, please use the condensed yaml list with "[" and "]" characters on a single line.
- For the character [TARGET_NAME], brainstorm an interesting character outline from the prior description.
- Output format: RETURN EXACTLY ONE top-level YAML item that starts with '- id:'. No surrounding list markers beyond the single item. No additional commentary.
