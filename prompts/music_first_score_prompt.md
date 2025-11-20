You are an expert narrative composer creating a first-pass motif that aligns with the current chapter touch-point.

Global music directive and context:
[VOICE_CONTEXT_JSON]

Character context (referenced entities):
[CHARACTER_CONTEXT_JSON]

Target voice for this pass:
- token: [VOICE_TOKEN]
- chord: [VOICE_CHORD]
- register: [VOICE_REGISTER]
- instrument: [VOICE_INSTRUMENT]
- idea: [VOICE_IDEA]
- role: [VOICE_ROLE]

Touch-point metadata:
- index: [TOUCH_POINT_INDEX]
- type: [TOUCH_POINT_TYPE]
- text:
[TOUCH_POINT_TEXT]

Existing sanitized score material for this voice (if any):
[EXISTING_VOICE_SCORE_JSON]

Shared global music metadata (derived from earlier steps):
- metadata.json:
[METADATA_JSON]

Reduced notes for alignment (complete, not truncated):
- Full reduced melody grid for the **main melodic line** (measure, beat, pitch, duration). This is the canonical melody you must stay aligned with:
[MELODY_REDUCED_CSV]
- Reduced-note grids for any voices already scored (if any). Treat these as fixed reference parts that the new voice should complement, not overwrite:
[OTHER_VOICES_REDUCED_CSV]

Instructions:
1. For this single voice only, output a **CSV table of notes** that covers **every measure** listed in the shared `measures.csv` block above.
2. Use the global directive and context to keep the voice aligned with the chapter, but focus the actual notes and phrasing on the target voice above.
3. Treat the full reduced melody grid as the **primary structural spine** of the piece. Your new notes must align rhythmically and harmonically to this melody unless a brief, clearly-motivated divergence is musically necessary.
4. When reduced-note grids for other voices are present, treat them as fixed reference parts. Write the new voice so that it complements these parts (e.g., avoiding collisions, reinforcing cadences, or providing counter-melody), but never assumes they can be changed.
5. Respect the tempo, key, and time signature from metadata/measures unless there is a strong musical reason to diverge.
6. Ensure that **every measure listed in the shared measures.csv** has at least one note row for this voice. Measures may contain multiple notes; do not skip measures entirely.
7. Use only the following columns in your CSV output, in this exact order:
	`measure,beat,pitch,duration,velocity,tie,articulation`
8. `measure` must be an integer matching one of the measures listed in the shared measures.csv block.
9. `beat` values are 1-based within each measure (use positive numbers such as 1, 1.5, 3.75, …; never greater than the upper value of the time signature's numerator).
10. `pitch` must be a scientific pitch name like `C4`, `F#3`, or `Bb2`.
11. `duration` and `beat` values must keep each note entirely within its measure; if a sound should sustain past the barline, split it into notes in the following measure.
12. `velocity` must stay within the MIDI range 0–127.
13. `tie` is either blank (no tie), `start`, `continue`, or `stop`, and is used to indicate notes that sustain across barlines.
14. `articulation` is a short textual tag such as `legato`, `staccato`, `tenuto`, `accent`, or may be left blank.

Return only a single CSV table using these seven columns and **no other text or commentary**.
