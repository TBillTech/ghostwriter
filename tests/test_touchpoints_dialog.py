import pytest

from ghostwriter.chapter import parse_touchpoints_from_chapter


def test_parse_touchpoints_accepts_dialog_and_rejects_explicit():
    chapter = {
        "Touch-Points": [
            {"dialog": "Henry greets Jim."},
            "dialog: Jim replies with a nod.",
            {"explicit": "This should not be a valid key."},
            "explicit: This should be treated as narration",
            {"narration": "A cool breeze passes through the forest."},
        ]
    }

    tps = parse_touchpoints_from_chapter(chapter)

    # Ensure we parsed the same number of items
    assert len(tps) == 5

    # First two are dialog
    assert tps[0]["type"] == "dialog"
    assert "Henry greets Jim" in tps[0]["content"]
    assert tps[1]["type"] == "dialog"
    assert "Jim replies" in tps[1]["content"]

    # Third item had an unknown dict key 'explicit' -> should default to narration
    assert tps[2]["type"] == "narration"
    # Content will be a stringified dict; ensure keyword is present
    assert "explicit" in tps[2]["content"].lower()

    # Fourth was a string with 'explicit: ...' -> should also default to narration
    assert tps[3]["type"] == "narration"
    assert "explicit:" in tps[3]["raw"].lower()

    # Fifth is a normal narration
    assert tps[4]["type"] == "narration"

    # No touch-point should have type 'explicit'
    assert all(tp.get("type") != "explicit" for tp in tps)
