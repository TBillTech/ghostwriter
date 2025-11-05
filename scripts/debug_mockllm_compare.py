import os
import re
import json
import shutil
from pathlib import Path
import sys

# Ensure repo root is on sys.path before importing ghostwriter
_HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_HERE))

from ghostwriter.mock_support import regenerate_prompt_for_log
from ghostwriter.pipelines.common import build_pipeline_replacements
from ghostwriter.templates import apply_template
from ghostwriter.context import RunContext


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def build_runtime_user(step_log: Path, base: Path) -> str:
    step_dir = step_log.parent
    version = int(step_dir.parent.name.split("_v")[-1])
    chapter_id = step_dir.parent.parent.name
    chapter_path = base / 'chapters' / f'{chapter_id}.yaml'
    ctx = RunContext.from_paths(chapter_path=str(chapter_path), version=version, allow_missing_chapter=False)
    state_raw = json.loads((step_dir/'touch_point_state.json').read_text(encoding='utf-8'))
    class S:
        def __init__(self, d):
            self.active_actors = d.get('active_actors', [])
            self.current_scene = d.get('scene')
            self.foreshadowing = d.get('foreshadowing', [])
            self.setting_block = d.get('setting_block', '')
            self.characters_block = d.get('characters_block','')
            self.appended_dialog = d.get('appended_dialog', {})
        def recent_dialog(self, actor):
            return self.appended_dialog.get(actor, [])
    state = S(state_raw)
    tp = {'type': step_dir.name.split('_',1)[-1], 'content': state_raw.get('touchpoint','')}
    reps = build_pipeline_replacements(ctx.setting, ctx.chapter, chapter_id, version, tp, state, prior_paragraph='', ctx=ctx)
    # Inject subtle edit specifics from first gate
    fd = step_dir/'touch_point_first_draft.txt'
    fs = step_dir/'first_suggestions.txt'
    reps['[draft_text]'] = fd.read_text(encoding='utf-8') if fd.exists() else ''
    reps['[suggestions]'] = fs.read_text(encoding='utf-8') if fs.exists() else ''
    return apply_template(Path('prompts')/'subtle_edit_prompt.md', reps)


def main():
    here = Path(__file__).resolve().parents[1]
    src = here / 'testdata' / 'LittleRedRidingHood'
    dst = here / 'sandbox' / 'tmp_book'
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    os.environ['GW_BOOK_BASE_DIR'] = str(dst)

    log = dst / 'iterations/CHAPTER_001/pipeline_v1/05_narration/05_subtle_edit.txt'
    regen = regenerate_prompt_for_log(str(log))
    user = build_runtime_user(log, dst)

    if regen is None:
        print('regenerate_prompt_for_log returned None (no template mapping).')
        return

    eq = norm(regen) == norm(user)
    print('equal:', eq)
    if not eq:
        # find first difference
        a = user or ''
        b = regen or ''
        i = 0
        while i < min(len(a), len(b)) and a[i] == b[i]:
            i += 1
        print('first mismatch at index', i)
        print('user excerpt:\n', a[max(0, i-200):i+200])
        print('regen excerpt:\n', b[max(0, i-200):i+200])

if __name__ == '__main__':
    main()
