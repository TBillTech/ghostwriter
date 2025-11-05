from pathlib import Path
import pytest

from ghostwriter.characters import build_character_call_prompt, substitute_character_calls
from ghostwriter.context import RunContext


def test_build_character_call_prompt_substitutions():
    system, user = build_character_call_prompt(
        character_id="red",
        call_prompt="Say hello politely.",
        dialog_lines=[
            "wolf: where are you going?",
            "red: to grandmother's house.",
            "wolf: what big basket you have!",
        ],
        agenda="care about grandmother",
        character_yaml="{\n  \"id\": \"red\", \n  \"name\": \"Little Red Riding Hood\"\n}",
        dialog_n_override=2,
    )
    assert "id 'red'" in system
    assert "<id>red</id>" in user or "id>red</id" in user
    assert "care about grandmother" in user
    assert "Say hello politely." in user
    # Only last 2 dialog lines should appear
    assert "where are you going?" not in user
    assert "to grandmother's house." in user
    assert "what big basket you have!" in user


def test_substitute_character_calls_with_template(monkeypatch: pytest.MonkeyPatch, use_lr_book_env, lr_book_dir: Path):
    # Prepare a small pre-draft with a template and one CHARACTER call
    pre = (
        "<CHARACTER TEMPLATE>\n"
        "<id>red</id>\n"
        "temperature_hint=0.25\n"
        "max_tokens_line=80\n"
        "</CHARACTER TEMPLATE>\n\n"
        "Before...\n"
        "<CHARACTER>\n"
        "<id>red</id>\n"
        "<agenda>be kind</agenda>\n"
        "<dialog>2</dialog>\n"
        "<prompt>Greet the wolf.</prompt>\n"
        "</CHARACTER>\n"
        "...After\n"
    )
    # Monkeypatch render_character_call to avoid LLM invocation and make deterministic
    import ghostwriter.characters as chars_mod
    monkeypatch.setattr(chars_mod, "render_character_call", lambda *a, **k: "DIALOG-LINE")

    ctx = RunContext.from_paths(chapter_path=str(lr_book_dir / "chapters/CHAPTER_001.yaml"), version=1)
    out, stats = substitute_character_calls(pre, ctx=ctx)
    assert "DIALOG-LINE" in out
    assert stats["templates"] == 1
    assert stats["calls"] == 1
    assert stats["missing_templates"] == 0
