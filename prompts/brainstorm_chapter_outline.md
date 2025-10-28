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
  - Touch-Points: a list of items (narration/dialog/mixed/actors/scene/foreshadowing/setting) in the order they should be executed.
  - The first touch-point should be a 'setting' block with:
    - factoids: [list of factoid names]
    - actors: [list of character ids or names]
  - The second touch-point should be a 'scene' block with:
    - name: the name of the scene
    - description: The 1 or two paragraph description of the scene
    - props: a list of 'prop' blocks
  - The third touch-point should be an 'actors: [list of character ids or names]'  
  - You may include another 'scene' block touch-point if the scene changes significantly during the chapter.
  - You may change the on-scene actors by using another actors touch-point.
- Keep Touch-Points specific enough to guide pipelines. Each touch-point should dwell on a single idea or narrative link; be focussed.
- Prefer clarity and coherence over exhaustiveness.

If the current chapter is absent, here is a formatting hint.
[FORMAT_HINT]

Now, provide touch points creatively for this chapter outline from 2 or 3 sigma in the sampled distribution:


