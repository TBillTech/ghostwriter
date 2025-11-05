import os, re, sys
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ghostwriter.mock_support import regenerate_prompt_for_log, _load_tp_state, _read_touchpoint_text
from ghostwriter.templates import apply_template
from ghostwriter.pipelines.common import build_pipeline_replacements
from ghostwriter.context import RunContext

base = Path('testdata/LittleRedRidingHood').resolve()
step_dir = base / 'iterations/CHAPTER_001/pipeline_v1/05_narration'
log = step_dir / '05_subtle_edit.txt'

os.environ['GW_BOOK_BASE_DIR'] = str(base)

regen_user = regenerate_prompt_for_log(str(log))

ctx = RunContext.from_paths(
    chapter_path=str(base / 'chapters/CHAPTER_001.yaml'),
    version=1,
    allow_missing_chapter=False,
)
state, tp_type = _load_tp_state(step_dir)
tp_dict = {'type': 'narration', 'content': _read_touchpoint_text(step_dir)}
reps = build_pipeline_replacements(ctx.setting, ctx.chapter, 'CHAPTER_001', 1, tp_dict, state, prior_paragraph='')
fd = (step_dir / 'touch_point_first_draft.txt').read_text(encoding='utf-8')
fs = (step_dir / 'first_suggestions.txt').read_text(encoding='utf-8')
reps['[draft_text]'] = fd
reps['[suggestions]'] = fs
user_runtime = apply_template(Path('prompts') / 'subtle_edit_prompt.md', reps)

norm = lambda s: re.sub(r"\s+", " ", s or '').strip()
print('Golden regen length:', len(regen_user or ''))
print('Runtime gen length:', len(user_runtime or ''))
print('Equal normalized?:', norm(regen_user)==norm(user_runtime))

if norm(regen_user)!=norm(user_runtime):
    from difflib import unified_diff
    a = (regen_user or '').splitlines()
    b = (user_runtime or '').splitlines()
    for ln in unified_diff(a,b, fromfile='golden_regen', tofile='runtime'):
        print(ln)
