from ghostwriter.chapter import parse_touchpoints_from_chapter


def test_parse_music_touchpoint_preserves_payload():
    chapter = {
        "Touch-Points": [
            {
                "music": {
                    "title": "The Forest Path",
                    "description": "A cinematic song with a haunting melody.",
                    "directive": "Lean into tension",
                    "voices": [
                        "major.tenor.bass.wolf.harmony",
                        {
                            "token": "major.alto.flute.red.melody",
                            "role": "lead"
                        },
                    ],
                }
            }
        ]
    }

    tps = parse_touchpoints_from_chapter(chapter)

    assert len(tps) == 1
    music_tp = tps[0]
    assert music_tp["type"] == "music"
    assert "haunting" in music_tp["content"]
    payload = music_tp.get("payload")
    assert isinstance(payload, dict)
    assert payload.get("title") == "The Forest Path"
    assert payload.get("directive") == "Lean into tension"
    voices = payload.get("voices")
    assert isinstance(voices, list)
    assert "major.tenor.bass.wolf.harmony" in voices
    assert any(isinstance(entry, dict) and entry.get("token") == "major.alto.flute.red.melody" for entry in voices)
