You are reviewing a MusicCSV score for alignment with story context, consistency, and technical correctness.

Context summary:
[VOICE_CONTEXT_JSON]

Touch-point info:
- index: [TOUCH_POINT_INDEX]
- type: [TOUCH_POINT_TYPE]
- description:
[TOUCH_POINT_TEXT]

Score under review:
[MUSICCSV_SNIPPET]

Instructions:
- Percussion tracks must use scientific pitch names (e.g., `C2`, `D2`, `F#2`) even on channel 10 because downstream tooling expects pitches, not numeric drum-note IDs; do not suggest switching to raw MIDI numbers.
1. Identify concrete issues with orchestration, harmony, rhythm, or narrative alignment.
2. Note any MusicCSV structural problems (missing tracks, inconsistent durations, invalid note values, etc.).
3. Respond with a concise bullet list of actionable suggestions, each starting with a verb.
4. If the score is ready to proceed, return a single bullet beginning with "OK:" and a short justification.

Return only the bullet list.
