from ghostwriter.touch_point import (
    _extract_bullet_contents,
    _rebuild_bullets,
    _apply_body_template,
    _inline_body_with_dialog,
)


def test_bullet_extract_and_rebuild():
    src = """
    # heading
    * one
    - two

    three
    *   four   
    """.strip()
    items = _extract_bullet_contents(src)
    assert items == ["one", "two", "four"]
    rebuilt = _rebuild_bullets(items)
    assert rebuilt.splitlines() == ["* one", "* two", "* four"]


def test_apply_body_template_quote_replacement():
    body = '* Henry says "<x>" as he turns.'
    dialog = 'I agree.'
    out = _apply_body_template(body, dialog)
    assert '"I agree."' in out


def test_inline_body_with_dialog_fallback():
    body = '* With a frown Henry turns'
    dialog = 'No.'
    out = _inline_body_with_dialog(body, dialog)
    assert out.endswith(', No.') or out.endswith(', No..')
