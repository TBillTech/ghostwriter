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

Shared measures and global music metadata (derived from earlier steps):
- metadata.json:
[METADATA_JSON]
- measures.csv for the selected melody variant:
[MEASURES_CSV]

Reduced notes for alignment (complete, not truncated):
- Full reduced melody grid for the **main melodic line** (measure, beat, pitch, duration). This is the canonical melody you must stay aligned with:
[MELODY_REDUCED_CSV]
- Reduced-note grids for any voices already scored (if any). Treat these as fixed reference parts that the new voice should complement, not overwrite:
[OTHER_VOICES_REDUCED_CSV]

Instructions:
1. Produce a valid MusicCSV score **for this single voice only**, using the format reference below.
2. Use the global directive and context to keep the voice aligned with the chapter, but focus the actual notes and phrasing on the target voice above.
3. Treat the full reduced melody grid as the **primary structural spine** of the piece. Your new notes must align rhythmically and harmonically to this melody unless a brief, clearly-motivated divergence is musically necessary.
4. When reduced-note grids for other voices are present, treat them as fixed reference parts. Write the new voice so that it complements these parts (e.g., avoiding collisions, reinforcing cadences, or providing counter-melody), but never assumes they can be changed.
5. Respect the tempo, key, and time signature from metadata/measures unless there is a strong musical reason to diverge.
6. Encode instrument names, articulations, dynamics, and expressive markings where appropriate for this voice.
7. Keep the part complete, and musically coherent with the entire melody in the shared measures grid.
8. Do **not** regenerate `metadata.json`, `tracks.csv`, or any `measures_*.csv`; assume they are shared across voices and already established by earlier steps. Only emit or update notes.csv relevant to this voice.

MusicCSV format reference:
[MUSICCSV_FORMAT_PROMPT]

Return only valid MusicCSV text for a complete MusicCSV document. Do not include commentary.
