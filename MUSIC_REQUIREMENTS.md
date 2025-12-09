🎼 Symbolic Music Collaboration Feature — Requirements Outline
🧭 Overall Goals

Create a music collaboration feature that allows the user and AI to share, understand, and generate musical tracks at the note/measure level, correlated with the touch-point bullets and iterations of the book. The book story will be the inspiration for a parallel song, when this is working.

Support interoperability with common DAWs via MIDI/MIDI 2.0.

Use MusicCSV internally to allow the LLM (and human users) to reason about structure, measures, instruments, harmony, and timing.

Enable AI-assisted music authoring, where users can tweak results interactively (“user-in-the-loop”).

1️⃣ Input & Ingestion Layer
Purpose:

Convert incoming musical data (e.g., .mid, .mid2) into a normalized, enriched internal MusicCSV representation.

Detect incoming musical data 

Functional Requirements:

MIDI Import

Read MIDI 1.0 and MIDI 2.0 files.

Detect and extract, via LLM prompting:

Global tempo(s)

Time signatures

Key signature(s)

Track-level instrument names or GM program numbers (a best effort basis, approximations acceptable like "electric piano", prefer human readable short descriptions to opaque program numbers)

Channel assignments

Parse note events into measure-based structure using tempo and time signature.

MusicCSV Conversion

Generate an initial MusicCSV archive representing the parsed data.

Ensure proper encoding of:

Measures and beats

Voice/instrument mapping

Articulations (if available)

Tempo and dynamics

Emit a human-readable text form (for LLM review) stored alongside the archive so edits remain straightforward.

User-in-the-Loop Refinement

Look for directories under iterations/CHAPTER_XXX/pipeline_vN/xx_naration (or xx_dialog or xx_mixed) called xx_track_[voice]_import/ , where [voice] is one of the voices defined in the new voice touch-point in the chapter. (see Internal Representation & Processing Layer requirements for more on the chapter voices.)

Provide a text-edit round-trip based files under xx_track_[voice]_import/

If the file xx_track_[voice]_import/import.musiccsv file is missing, then generate it from the importable files.

Any importable files such as .midi, etc... (not including the monitor.mid file) should be read from this directory to create the import.musiccsv file.

The user then reviews the MusicCSV text summary, confirms or edits detected tempo/time signature, instrument metadata (name, type, range, MIDI program), and adjusts measure alignment (barline correction) via the CSV/JSON payloads.

IF the output file (score.musiccsv) is absent, then sanitize and fix user-edited import.musiccsv to ensure structural integrity, and write this output to the xx_track_[voice]_import/score.musiccsv

Also, anytime you write out score.musiccsv, overwrite the monitor.mid.  The monitor.mid file should be a reasonable approximation of the score which can be played by the user to hear the notes and/or beats. 

### Melody import sanitization path (Dec 2025)

- Ignore any `*.Zone.Identifier` companions that appear beside imported `.mid` files (e.g., Windows ADS metadata). Only the actual `.mid/.midi/.mid2` payloads should participate in normalization.
- When the melody voice has an import folder, the pipeline must route through a dedicated `music_import_sanitize_prompt.md` instead of the `music_melody_emotion`/`music_emotion_chord` prompts. Provide that prompt with the sanitized `score.musiccsv` (or freshly generated `import.musiccsv`) for the voice, plus the surrounding metadata so it can quantize events and drop leading/trailing empty measures.
- The prompt output should be a deterministic table supplying the simplified columns (`measure,pitch,duration`). Because the import contains single-note rows, mandate `semi_tones=(0)`; the pipeline converts the pitch column into `transition` intervals automatically when producing `CORE_MELODY.csv`, so no extra math is required in the prompt response.
- Feed the resulting rows directly into the core-melody builder so subsequent steps (melody edges, reduced grids, voice prompts) treat the imported line as authoritative. Ensure this bypass honours any accompanying imports for bass/beat voices as well, wiring each `<tp_index>_track_<voice>_import/` folder to the matching track metadata.

2️⃣ Internal Representation & Processing Layer
Purpose:

Maintain a semantically rich symbolic music representation (MusicCSV) for reasoning, analysis, and generation. It is expected to track the touch-point bullets in a novel and interesting isomorphism between the story and the music.

Functional Requirements:

Voices will be defined in the chapter yaml, similiar to how actors are done.  Each voice has four parts, the "chords", the "register", the "instrument", and the "idea" in the story. The "chords" will normally be one of major, minor, diminished, augmented, or chromatic. The "register" will normally be one of base, tenor, alto, or beat. The "instrument" is a simple instrument description like electric-piano, strings, or soundscape. Ideas can be actors, props, or atomosphere. Optionally, the voice may have a ".melody" or ".harmony" as an additional final specification. Here are a few examples of voices: "augmented.base.base.wolf.harmony", "major.beat.drum.engine", "diminished.tenor.soundscape.moon", "minor.alto.strings.mob.melody", "major.tenor.cello.Ralph".

Music will be a description defined in the chapter yaml as well. It doesn't adhere to an exact format, but instructs the LLM about the song globally.  For example: "music: a cinematic track in D minor; slow tempo, haunting melody, steady bass, light percussion." 

Create tracks in the MusicCSV score for each voice defined in the chapter, similar to how actors are tracked. 

When prompting the LLM to produce the xx_narration, xx_dialog, or xx_mixed /first_score.musiccsv, be sure to provide the last two measures from the prior touch_point, if it exists.

The workflow for touch_point_first_score.musiccsv, first_score_suggestions.txt, feeding touch_point_score.musiccsv and score_suggestions.txt should echo the one already built for touch_point_first_draft.txt, first_suggestions.txt, touch_point_draft.txt, and suggestions.txt.

MusicCSV Parser/Serializer

Implement or integrate a library (e.g., the internal `ghostwriter.musiccsv` helpers) to parse and manipulate MusicCSV archives.

Allow in-memory modification of structure (notes, measures, instruments, etc.).

Serialize back to valid MusicCSV conforming to the specification.

Musical Knowledge Model

Abstract model for:

Score

Track / Part

Measure

Note event

Instrument / Voice metadata

Maintain synchronization of beats and measures across all tracks.

Tempo & Rhythm Engine

Provide consistent internal clock for quantization and alignment.

Handle tempo curves (ritardando, accelerando).

AI Reasoning Interface

LLM integration layer that exposes the internal MusicCSV as text for analysis or generation.

LLM prompts should include:

Any provided xx_track_[voice]_import/score.musiccsv, but make sure to clearly link the [voice] title with the CSV snippet.

Story Context, including the touch_point, brainstorm.txt bullets, and voice context. For voice context, voices should be matched to props, actors, and factoids in a best effort bag-of-words way. For example, if the voice is "base.wolf", then the prompt for generating the "base.wolf" track should provide the character outline for wolf. On the other hand, if the voice were "soundscape.moon", then factoids with moon in the name in the setting on should be provided in the prompt instead. If no factoids, props, or characters can be found, then just tell the LLM to do it's best with general knowledge of the voice.

Context summary (tempo, time signature, scale)

Current structure (number of measures, instruments)

Musical intent or requested operation (e.g., “generate a harmonic accompaniment”)

Results parsed back into MusicCSV delta or new track.

Harmonic and Rhythmic Coherence

Ensure generated parts are synchronized measure-by-measure with the base composition.

Support transposition, reharmonization, and rhythmic quantization.

Semantic Metadata

Maintain attributes for:

Instrument family

Instrument register (range)

Intended role (melody, harmony, bass, percussion)

Dynamic and articulation defaults

Allow AI or user to modify roles and regenerate related parts accordingly.

3️⃣ Output & Export Layer
Purpose:

Convert enriched MusicCSV compositions back into standard formats for interoperability.

Similar to how draft_vN.txt is a concatenation of the touch_point_draft.txt files, there should be a score_vN.musiccsv which concatenates the touch_point_score.musiccsv files from the pipeline_vN.

Similar to how the final.txt is condensed from the latest draft_vN.txt, a score/ directory should be regenerated by condensing the score_vN.musiccsv into a score/final.musiccsv and also broken out into each individual voice track in the same directory, for example: `score/augmented.base.base.wolf.harmony.mid`, `score/diminished.tenor.soundscape.moon.mid`, etc..

Functional Requirements:

MusicCSV to MIDI Conversion

Translate note events, dynamics, and tempo changes into MIDI 1.0 or 2.0 events.

Support both:

Single multi-track score/final.mid file

Multiple single-track .mid files (`score/augmented.base.base.wolf.harmony.mid`, `score/diminished.tenor.soundscape.moon.mid`, etc.)

MIDI 2.0 Export by default (use env variable to disable)

Where supported, include high-resolution per-note control (if the DAW supports it).

Metadata Export

Preserve voice names in MIDI track names.

Optional: export a JSON manifest summarizing the composition.

{
  "title": "Haunting Waltz",
  "tempo": 90,
  "key": "D minor",
  "tracks": [
    {"name": "Melody", "instrument": "Violin"},
    {"name": "Harmony", "instrument": "Piano"},
    {"name": "Bass", "instrument": "Cello"},
    {"name": "Percussion", "instrument": "Brush Kit"}
  ]
}


DAW Compatibility Testing

Ensure exported MIDI loads cleanly in:

Reaper

4️⃣ Collaborative Workflow Layer
Purpose:

Facilitate easy exchange and shared understanding between users, even if they don’t use the same tools.

Functional Requirements:

Track Packaging

As described above, each draft_vN will regenerate a bundle:

CHAPTER_XXX/pipeline_vN/score/
  ├── manifest.json
  ├── final.musiccsv
  ├── final.mid
  ├── diminished.tenor.soundscape.moon.mid
  ├── augmented.base.base.wolf.harmony.mid
  ├── minor.alto.flute.red.melody.mid
  └── major.beat.drum.axe.mid


Compress to .zip for sharing.

Human-AI Round-Trip

Workflow should allow:

User edits MusicCSV (by hand or DAW tooling that understands the CSV format) (see discussion above).

AI reads updated MusicCSV, interprets changes, and generates follow-up suggestions (see discussion above).

5️⃣ Auxiliary & Developer Requirements
Libraries and Tooling

Core parsing & conversion:

mido, pretty_midi, and the internal MusicCSV helpers

Serialization:

Standard XML parser (lxml, xml.etree.ElementTree)

LLM Integration:

Input/output structured prompt templates for musical context

Testing:

Use reference .mid and .musiccsv samples for regression testing

🧩 Summary of Major Components
Layer	Primary Role	Key Technologies
Input/Ingestion	Parse MIDI → MusicCSV	mido, ghostwriter.musiccsv
Representation/Processing	Maintain internal enriched structure	custom MusicCSV classes
AI Integration	Reason about structure and generate new parts	LLM with prompt templates
Output/Export	MusicCSV → MIDI	pretty_midi, ghostwriter.musiccsv
Collaboration Layer	Packaging, metadata, versioning	JSON, ZIP, diff tools