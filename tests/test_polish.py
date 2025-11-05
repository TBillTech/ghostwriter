from ghostwriter.templates import build_polish_prompt


def test_build_polish_prompt_includes_rough_text():
    setting = {"title": "S"}
    chapter = {"Story-So-Far": "ssf", "Story-Relative-To": {"ch": 1}}
    rough = "He said, \"hello\".  She replied."
    out = build_polish_prompt(setting, chapter, "CHAPTER_999", 1, rough)
    assert "Please clean up the following text" in out
    assert rough in out
    # Should include story fields somewhere
    assert "ssf" in out or "Story-So-Far" in out
