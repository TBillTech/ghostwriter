You are an expert story architect. Draft or revise a chapter outline for [CHAPTER_ID].

Context:
- CONTENT_TABLE.yaml (full):

[CONTENT_TABLE.yaml]

- Story-So-Far (from the prior completed chapter):

[STORY_SO_FAR]

- Current chapter YAML (if present), with Story-* sections removed (you may revise its Touch-Points and setting):

[CURRENT_CHAPTER_NO_STORY]

- Setting/Characters context:

[SETTING_DEREF]

- If the above is empty, only names are provided; use these to propose an appropriate setting block:

[NAMES_ONLY]

Instructions:
- Look at the synopsis for [CHAPTER_NUM] in CONTENT_TABLE.yaml and brainstorm an exciting, engaging chapter at a high directional level.
- Output must be VALID YAML for chapters/[CHAPTER_ID].yaml.
- Do NOT include Story-So-Far or Story-Relative-To sections.
- The YAML SHOULD include:
  - Touch-Points: a list of items (narration/dialog/implicit/actors/scene/foreshadowing/setting) in the order they should be executed.
  - Optional 'setting' block with:
    - factoids: [list of factoid names]
    - actors: [list of character ids or names]
    - scene: string
- Keep Touch-Points specific enough to guide pipelines, but concise (single sentences or short phrases).
- Prefer clarity and coherence over exhaustiveness.

If the current chapter is absent, here is a formatting hint.
[FORMAT_HINT]

Now, provide touch points creatively for this chapter outline from 2 or 3 sigma in the sampled distribution:


