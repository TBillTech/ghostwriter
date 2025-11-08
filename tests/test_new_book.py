from pathlib import Path

from ghostwriter.cli import main


def test_new_book_creates_template(tmp_path: Path):
    dest = tmp_path / "my_first_book"
    code = main(["new-book", str(dest)])
    assert code == 0, "new-book should exit with code 0 on success"

    # Core files
    setting = dest / "SETTING.yaml"
    characters = dest / "CHARACTERS.yaml"
    chapters_dir = dest / "chapters"
    ct = chapters_dir / "CONTENT_TABLE.yaml"
    ch1 = chapters_dir / "CHAPTER_001.yaml"

    for p in [setting, characters, chapters_dir, ct, ch1]:
        assert p.exists(), f"Expected to find {p}"

    # Basic content sanity checks
    setting_text = setting.read_text(encoding="utf-8")
    ct_text = ct.read_text(encoding="utf-8")
    assert "MyFirstBook" in setting_text or "MyFirstBook" in ct_text, "Starter files should reference 'MyFirstBook'"


def test_new_book_requires_empty_directory(tmp_path: Path):
    dest = tmp_path / "non_empty"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "SOME_FILE.txt").write_text("not empty", encoding="utf-8")

    code = main(["new-book", str(dest)])
    # Expect a non-zero exit code due to non-empty directory (cli uses 2 on errors)
    assert code != 0, "new-book should fail when destination directory is not empty"
