import yaml
from pathlib import Path
import sys
import datetime

# ---------------------------
# Helpers
# ---------------------------

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_text(path, content):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def read_file(path):
    return Path(path).read_text(encoding="utf-8")

def build_prompt(setting, chapter, template_path):
    """Insert YAML into the master or check prompt template."""
    template = read_file(template_path)
    return (
        template
        .replace("[SETTING.yaml]", yaml.dump(setting, sort_keys=False))
        .replace("[CHAPTER_xx.yaml]", yaml.dump(chapter, sort_keys=False))
    )

# ---------------------------
# Main Functions
# ---------------------------

def generate_draft(chapter_path, version="v1"):
    """Generate a draft from YAML inputs using master_prompt.md."""
    setting = load_yaml("SETTING.yaml")
    chapter = load_yaml(chapter_path)
    chapter_id = Path(chapter_path).stem

    # Build draft prompt
    prompt = build_prompt(setting, chapter, "prompts/master_prompt.md")

    # --- PLACEHOLDER: Replace this with a real GPT API call ---
    prose = f"[DRAFT PLACEHOLDER]\nGenerated at {datetime.datetime.now()}\n\nPROMPT PREVIEW:\n{prompt[:500]}..."

    # Save draft
    out_path = Path("iterations") / chapter_id / f"draft_{version}.txt"
    save_text(out_path, prose)
    print(f"Draft saved to {out_path}")
    return prose, out_path, chapter_id

def verify_draft(draft_text, chapter_path, version="v1"):
    """Check draft against Touch-Points using check_prompt.md."""
    chapter = load_yaml(chapter_path)
    check_template = read_file("prompts/check_prompt.md")

    # Build check prompt
    prompt = check_template + "\n\nDRAFT:\n" + draft_text

    # --- PLACEHOLDER: Replace with real GPT API call ---
    check_results = f"[CHECK PLACEHOLDER]\nChecked at {datetime.datetime.now()}\n\nPROMPT PREVIEW:\n{prompt[:500]}..."

    # Save check
    chapter_id = Path(chapter_path).stem
    out_path = Path("iterations") / chapter_id / f"check_{version}.txt"
    save_text(out_path, check_results)
    print(f"Check results saved to {out_path}")
    return check_results

# ---------------------------
# CLI Entrypoint
# ---------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/driver.py chapters/CHAPTER_xx.yaml [version]")
        sys.exit(1)

    chapter_path = sys.argv[1]
    version = sys.argv[2] if len(sys.argv) > 2 else "v1"

    draft_text, draft_path, chapter_id = generate_draft(chapter_path, version)
    verify_draft(draft_text, chapter_path, version)

if __name__ == "__main__":
    main()