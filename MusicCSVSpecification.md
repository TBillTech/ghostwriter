# 🎼 MusicCSV: Lightweight Structured Music Format
**Specification v0.1**

---

## 1. Overview

**MusicCSV** is a compact, human- and machine-readable data format representing symbolic musical scores.  
It is conceptually equivalent to a subset of **MusicXML**, but organized into simple tabular CSVs and a single JSON metadata file.

MusicCSV consists of four main components:

1. **`metadata.json`** – global score metadata (title, composer, tempo map, key/time defaults, etc.)
2. **`tracks.csv`** – defines tracks and their descriptive attributes.
3. **`measures.csv`** – defines measure-level attributes and timing information.
4. **`notes.csv`** – defines individual musical events (notes, articulations, etc.)

A single `.musiccsv` file is simply these four files concatenated into a single human/LLM readable file. See musiccsv_format_prompt.txt for an example of how they should be concatenated.

---

## 2. File Definitions

### 2.1 `metadata.json`

Stores global musical context and score-level metadata.

#### Example
```json
{
  "title": "Prelude in C Major",
  "composer": "Johann Sebastian Bach",
  "tempo": 120,
  "time_signature": "4/4",
  "key_signature": "C",
  "divisions_per_quarter": 480,
  "encoding_software": "musiccsv-encoder 0.1",
  "created": "2025-11-12T14:00:00Z",
  "description": "Example piece in MusicCSV format",
  "version": "0.1",
  "copyright": "Public Domain"
}
```

#### Required fields
| Field | Type | Description |
|--------|------|-------------|
| `title` | string | Name of the work |
| `tempo` | number | BPM for the primary tempo |
| `time_signature` | string | Default time signature (e.g. `"3/4"`) |
| `key_signature` | string | Default key (e.g. `"Gm"`, `"C"`) |
| `divisions_per_quarter` | integer | MIDI-compatible resolution |
| `version` | string | Schema version tag |

#### Optional fields
`composer`, `description`, `encoding_software`, `created`, `copyright`, etc.

---

### 2.2 `tracks.csv`

Maps **track IDs** to their attributes.

#### Columns
| Column | Type | Description |
|---------|------|-------------|
| `track` | integer | Unique track ID (1-N) |
| `label` | string | Human-readable label or name |
| `part` | string | Logical part name (e.g. `"Right Hand"`) |
| `instrument` | string | Instrument name (e.g. `"Acoustic Grand Piano"`) |
| `channel` | integer | Optional MIDI channel number |
| `program` | integer | Optional MIDI program number |
| `volume` | integer | Optional initial volume (0-127) |

#### Example
```csv
track,label,part,instrument,channel,program,volume
1,Piano RH,Right Hand,Acoustic Grand Piano,1,0,100
2,Piano LH,Left Hand,Acoustic Grand Piano,2,0,100
```

---

### 2.3 `measures.csv`

Represents attributes of each measure.

#### Columns
| Column | Type | Description |
|---------|------|-------------|
| `measure` | integer | Measure number (1-N) |
| `time_signature` | string | `"4/4"`, `"3/8"`, etc. |
| `key_signature` | string | `"C"`, `"Gm"`, etc. |
| `tempo` | number | BPM for this measure |
| `start_beat` | float | Cumulative starting beat of this measure |
| `pickup` | boolean | `true` if anacrusis (partial measure) |

#### Example
```csv
measure,time_signature,key_signature,tempo,start_beat,pickup
1,4/4,C,120,0,false
2,4/4,C,120,4,false
```

---

### 2.4 `notes.csv`

Represents all musical events (notes and expressive data).

#### Columns
| Column | Type | Description |
|---------|------|-------------|
| `track` | integer | Reference to `tracks.csv.track` |
| `measure` | integer | Reference to `measures.csv.measure` |
| `beat` | float | Starting beat within measure (1-based) |
| `pitch` | string | Scientific pitch notation (e.g. `C#4`, `Bb3`) |
| `duration` | float | Duration in beats |
| `velocity` | integer | MIDI velocity (0–127) |
| `tie` | string | `"start"`, `"stop"`, or `"none"` |
| `articulation` | string | Optional (e.g. `"staccato"`, `"accent"`) |
| `pedal` | boolean | `true` if sustain pedal active during note |
| `lyric` | string | Optional lyric syllable |
| `ornament` | string | Optional (e.g. `"trill"`, `"grace"`) |
| `comment` | string | Optional text annotation |

#### Example
```csv
track,measure,beat,pitch,duration,velocity,tie,articulation,pedal,lyric,ornament,comment
1,1,1.0,C4,1.0,90,none,legato,false,,,
1,1,2.0,D4,1.0,88,none,legato,false,,,
1,1,3.0,E4,1.0,85,none,staccato,false,,,
1,2,1.0,G4,2.0,92,none,legato,true,,,
2,1,1.0,C3,4.0,80,none,,false,,bass_note,
```

---

## 3. Example File Set

**metadata.json**
```json
{
  "title": "Simple Duet",
  "composer": "Anonymous",
  "tempo": 120,
  "time_signature": "4/4",
  "key_signature": "C",
  "divisions_per_quarter": 480,
  "version": "0.1"
}
```

**tracks.csv**
```csv
track,label,part,instrument,channel,program,volume
1,Piano RH,Right Hand,Acoustic Grand Piano,1,0,100
2,Piano LH,Left Hand,Acoustic Grand Piano,2,0,100
```

**measures.csv**
```csv
measure,time_signature,key_signature,tempo,start_beat,pickup
1,4/4,C,120,0,false
2,4/4,C,120,4,false
```

**notes.csv**
```csv
track,measure,beat,pitch,duration,velocity,tie,articulation,pedal,lyric,ornament,comment
1,1,1.0,C4,1.0,90,none,legato,false,,,
1,1,2.0,D4,1.0,88,none,legato,false,,,
1,2,1.0,E4,2.0,92,none,legato,true,,,
2,1,1.0,C3,4.0,80,none,,false,,bass_note,
```

---

## 4. Design Goals

- **Readable:** CSV + JSON, no XML tags or indentation bloat.
- **Concise:** ~90% smaller token footprint than MusicXML.
- **Loss-minimized:** Retains core musical and expressive semantics.
- **Round-trippable:** Can be regenerated from MIDI and re-exported as MusicXML.
- **LLM-friendly:** Each event is a single line of structured data.

---

## 5. Prompt Template for LLMs

When prompting the LLM to generate MusicCSV output, append the following to the prompt:
prompts/musiccsv_format_prompt.txt