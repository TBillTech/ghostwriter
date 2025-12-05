from __future__ import annotations

import textwrap
import pytest

from ghostwriter.music.core_melody import (
    CoreMelodyRow,
    EmotionWordRow,
    FeelingRow,
    TransitionRow,
    build_core_melody_csv,
    build_core_melody_rows,
    core_melody_rows_to_csv,
    parse_emotion_prompt_response,
    parse_emotion_chord_response,
)
from ghostwriter.context import UserActionRequired


def test_parse_emotion_prompt_response_success():
    sample = textwrap.dedent(
        """
        Some intro text...

        PRO.csv
        word, emotion, duration
        flame, awe, 1
        burns, fear, 0.5

        ANTI.csv
        word, emotion, duration
        shadow, dread, 1.5
        bites, anger, 0.75

        Closing line.
        """
    )

    pro, anti = parse_emotion_prompt_response(sample)

    assert pro == [
        EmotionWordRow(word="flame", emotion="awe", duration=1.0),
        EmotionWordRow(word="burns", emotion="fear", duration=0.5),
    ]
    assert anti == [
        EmotionWordRow(word="shadow", emotion="dread", duration=1.5),
        EmotionWordRow(word="bites", emotion="anger", duration=0.75),
    ]


def test_parse_emotion_prompt_response_missing_block():
    with pytest.raises(UserActionRequired):
        parse_emotion_prompt_response("No CSV here")


def test_parse_emotion_chord_response_success():
    sample = textwrap.dedent(
        """
        Emotions.csv
        emotion, semi-tones
        awe, (4, 3, 5)
        dread, (3, 3, 6)

        Transitions.csv
        emotion A, emotion B, semi-tone
        awe, dread, -2
        dread, awe, 5
        """
    )

    feelings, transitions = parse_emotion_chord_response(sample)

    assert feelings == [
        FeelingRow(emotion="awe", semitones=(4, 3, 5)),
        FeelingRow(emotion="dread", semitones=(3, 3, 6)),
    ]
    assert transitions == [
        TransitionRow(emotion_a="awe", emotion_b="dread", semitone=-2),
        TransitionRow(emotion_a="dread", emotion_b="awe", semitone=5),
    ]


def test_build_core_melody_rows_success():
    pro = [
        EmotionWordRow(word="flame", emotion="awe", duration=1.0),
        EmotionWordRow(word="burns", emotion="fear", duration=0.5),
    ]
    anti = [
        EmotionWordRow(word="shadow", emotion="dread", duration=1.5),
        EmotionWordRow(word="bites", emotion="anger", duration=0.75),
    ]
    feelings = [
        FeelingRow(emotion="awe", semitones=(4, 3, 5)),
        FeelingRow(emotion="fear", semitones=(3, 2)),
        FeelingRow(emotion="dread", semitones=(3, 3, 6)),
        FeelingRow(emotion="anger", semitones=(2, 2, 3)),
    ]
    transitions = [
        TransitionRow(emotion_a="awe", emotion_b="fear", semitone=-1),
        TransitionRow(emotion_a="fear", emotion_b="dread", semitone=-2),
        TransitionRow(emotion_a="dread", emotion_b="anger", semitone=5),
    ]

    rows = build_core_melody_rows(pro, anti, feelings, transitions, beats_per_measure=4.0)

    assert rows == [
        CoreMelodyRow(measure=1, emotion="awe", duration=1.0, semitones=(4, 3, 5), transition=(-1,)),
        CoreMelodyRow(measure=1, emotion="fear", duration=0.5, semitones=(3, 2), transition=(-2,)),
        CoreMelodyRow(measure=1, emotion="dread", duration=1.5, semitones=(3, 3, 6), transition=(5,)),
        CoreMelodyRow(measure=1, emotion="anger", duration=0.75, semitones=(2, 2, 3), transition=(0,)),
    ]


def test_build_core_melody_rows_missing_feeling():
    pro = [EmotionWordRow(word="flame", emotion="awe", duration=1.0)]
    anti: list[EmotionWordRow] = []
    feelings: list[FeelingRow] = []

    with pytest.raises(UserActionRequired):
        build_core_melody_rows(pro, anti, feelings, [])


def test_build_core_melody_csv_writes_file(tmp_path):
    pro_path = tmp_path / "PRO.csv"
    pro_path.write_text(
        "word, emotion, duration\nflame, awe, 1\nburns, fear, 0.5\n",
        encoding="utf-8",
    )

    anti_path = tmp_path / "ANTI.csv"
    anti_path.write_text(
        "word, emotion, duration\nshadow, dread, 1.5\nbites, anger, 0.75\n",
        encoding="utf-8",
    )

    feelings_path = tmp_path / "Feelings.csv"
    feelings_path.write_text(
        "emotion, semi-tones\nawe, (4, 3, 5)\nfear, (3, 2)\ndread, (3, 3, 6)\n"
        "anger, (2, 2, 3)\n",
        encoding="utf-8",
    )

    transitions_path = tmp_path / "Transitions.csv"
    transitions_path.write_text(
        "emotion A, emotion B, semi-tone\nawe, fear, -1\nfear, dread, -2\n"
        "dread, anger, 5\n",
        encoding="utf-8",
    )

    output_path = tmp_path / "CORE_MELODY.csv"
    rows = build_core_melody_csv(
        pro_csv_path=pro_path,
        anti_csv_path=anti_path,
        feelings_csv_path=feelings_path,
        transitions_csv_path=transitions_path,
        output_csv_path=output_path,
        overwrite=True,
    )

    expected_csv = (
        "measure,duration,semi_tones,transition\n"
        "1,1,(4, 3, 5),(0)\n"
        "1,0.5,(3, 2),(0)\n"
        "1,1.5,(3, 3, 6),(0)\n"
        "1,0.75,(2, 2, 3),(0)\n"
    )
    assert output_path.read_text(encoding="utf-8") == expected_csv
    assert core_melody_rows_to_csv(rows) == expected_csv

    # Second run without overwrite should leave file untouched but return rows.
    rows_again = build_core_melody_csv(
        pro_csv_path=pro_path,
        anti_csv_path=anti_path,
        feelings_csv_path=feelings_path,
        transitions_csv_path=transitions_path,
        output_csv_path=output_path,
        overwrite=False,
    )
    assert rows_again == rows
    assert output_path.read_text(encoding="utf-8") == expected_csv
