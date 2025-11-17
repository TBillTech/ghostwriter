You are designing the high-level structure of a single song for a story touch-point.

Context (JSON, sanitized for length):
[VOICE_CONTEXT_JSON]

Touch-point info:
- index: [TOUCH_POINT_INDEX]
- type: [TOUCH_POINT_TYPE]
- title: [TOUCH_POINT_TITLE]
- description:
[TOUCH_POINT_DESCRIPTION]
- prior paragraph (most recent polished prose for context):
[TOUCH_POINT_PRIOR_PARAGRAPH]

Instructions:
1. First, decide on global score metadata: tempo, time signature, key signature, divisions_per_quarter, and any brief freeform tags or notes.
2. Then, design the tracks: one row per musical voice or part. For each track, specify at least: track id, label, part, instrument, MIDI channel, MIDI program, and an optional short role description.
3. Keep channel assignments consistent with these rules:
   - Use channel 10 for percussion / drum kit tracks only.
   - Use channels 1 for all pitched instruments (melody, harmony, bass, pads, etc.).
   - Reserve program 0 for unspecified / generic, otherwise choose a reasonable General MIDI program.
4. Ensure every voice token mentioned in the context is mapped to exactly one track.
5. Do not generate any measures or notes yet; this step is for metadata and track layout only.

Output format:
- First, a `metadata.json` block in pretty-printed JSON that includes at minimum: title, tempo, time_signature, key_signature, divisions_per_quarter, and optional descriptive fields.
- Second, a `tracks.csv` block with a header row and one row per track, with columns: track,label,part,instrument,channel,program,role,voice_token.

Example skeleton (illustrative only):

metadata.json
{
  "title": "Example Song",
  "tempo": 96,
  "time_signature": "4/4",
  "key_signature": "D minor",
  "divisions_per_quarter": 480,
  "tags": ["brooding", "cinematic"]
}

tracks.csv
track,label,part,instrument,channel,program,role,voice_token
1,Red Melody,Melody,Flute,1,73,lead,major.alto.flute.red.melody
2,Wolf Harmony,Harmony,French Horn,3,61,inner-voice,major.tenor.bass.wolf.harmony
3,Drums,Drums,Standard Kit,10,1,groove,major.beat.drum.axe

Return exactly two blocks in this order:
1) The `metadata.json` block
2) The `tracks.csv` block

Do not include commentary or any other text outside those two blocks.
