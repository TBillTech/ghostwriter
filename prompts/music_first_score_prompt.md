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

Ten-measure block focus (leave blank fields if composing the entire score at once):
- Measure window: [MEASURE_WINDOW_DESCRIPTION]
- Block position (index / total): [BLOCK_INDEX]/[BLOCK_COUNT]

Melody guidance for this block:
- Current block slice (10 measures):
[MELODY_BLOCK_CSV]
- Previous block slice (for continuity, if available):
[MELODY_PREVIOUS_BLOCK_CSV]

Current voice continuity (previous block rows, do not rewrite unless blanks are supplied):
[VOICE_PREVIOUS_BLOCK_CSV]

Other voices locked for this window:
[OTHER_VOICES_BLOCK_CSV]

Other voices from the immediately preceding block:
[OTHER_VOICES_PREVIOUS_BLOCK_CSV]

Reduced notes for alignment (complete, not truncated):
- Full reduced melody grid for the **main melodic line** (measure, beat, pitch, duration). This may be omitted when the block slice above already covers the needed measures:
[MELODY_REDUCED_CSV]
- Reduced-note grids for any voices already scored (if any). Treat these as fixed reference parts that the new voice should complement, not overwrite:
[OTHER_VOICES_REDUCED_CSV]

Instructions:
1. For this single voice only, output a **CSV table of notes** that copies the provided melody guidance exactly when this voice is the melody, and otherwise meshes cleanly with it when this voice is an accompaniment.
2. When a measure window is provided, **emit notes only for that window** and leave prior windows untouched. Extend ties with new rows instead of rewriting earlier measures.
3. If this voice is the melody, do **not** invent or rephrase the line: for every row in the melody guidance, emit a matching note with the identical measure, beat, pitch, and duration. No substitutions, omissions, or reordering are allowed.
4. When this voice is not the melody, compose material that sounds good with the guided melody, but still treat the guidance as fixed.
5. Never split a guided note into multiple notes, tie it across a new boundary, or alter its duration unless the guidance itself already does so. Each provided row should map to one output row starting at the exact beat and lasting the exact duration.
6. When reduced-note grids for other voices are present, treat them as fixed reference parts. Write the new voice so that it complements these parts (e.g., avoiding collisions, reinforcing cadences, or providing counter-melody), but never assumes they can be changed.
7. Respect the tempo, key, and time signature from metadata/measures unless there is a strong musical reason to diverge.
8. Use only the following columns in your CSV output, in this exact order:
	`measure,beat,pitch,duration,velocity,tie,articulation`
9. `measure` must be an integer counting up like the measures listed in the shared **main melodic line** block, and it must land within the requested window when one is supplied.
10. `beat` values are 1-based within each measure (use positive numbers such as 1, 1.5, 3.75, …; never greater than the upper value of the time signature's numerator).
11. `pitch` must be a scientific pitch name like `C4`, `F#3`, or `Bb2`, or `rest` for no pitch.
12. `duration` is the number of beats the note should continue for, starting at the `beat` point of the note.
13. `velocity` must stay within the MIDI range 0–127.
14. `tie` is either blank (no tie), `start`, `continue`, or `stop`, and is used to indicate notes that sustain across barlines.
15. `articulation` is a short textual tag such as `legato`, `staccato`, `tenuto`, `accent`, or may be left blank.

Return only a single CSV table using these seven columns and **no other text or commentary**.
