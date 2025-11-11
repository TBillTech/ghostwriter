You are an expert narrative composer creating a first-pass motif that aligns with the current chapter touch-point.

[VOICE_CONTEXT_JSON]

Character context (referenced entities):
[CHARACTER_CONTEXT_JSON]

Touch-point metadata:
- index: [TOUCH_POINT_INDEX]
- type: [TOUCH_POINT_TYPE]
- text:
[TOUCH_POINT_TEXT]

Existing sanitized score material (if any):
[EXISTING_SCORES_JSON]

Instructions:
1. Produce a complete MusicXML score (partwise) named for this touch-point.
2. Follow the directive and voice definitions above; map each voice token to a clear, distinct musical track.
3. Respect the tempo, key, and time signature summaries unless there is a strong musical reason to diverge.
4. Encode instrument names, articulations, and dynamics where appropriate.
5. Keep the score concise (4–16 measures) but musically coherent.

Return only valid MusicXML for the score. Do not include commentary.
