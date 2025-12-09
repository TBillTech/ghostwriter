from ghostwriter.music.pipeline import (
    _core_melody_note_sequence,
    _format_core_melody_example_line,
    _core_melody_csv_with_root,
)


def test_core_melody_example_helpers() -> None:
    csv_text = """measure,duration,semi_tones,transition\n1,1,(0),(2)\n1,1,(0),(-1)\n"""

    sequence = _core_melody_note_sequence(csv_text)
    assert sequence == ["C4", "D4"]

    line = _format_core_melody_example_line(sequence)
    assert line == "Example Notes (start=C4): C4, D4"

    rooted_csv = _core_melody_csv_with_root(csv_text, sequence)
    lines = rooted_csv.splitlines()
    assert lines[0] == "measure,duration,semi_tones,root"
    assert lines[1].endswith(",C4")
