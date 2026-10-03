"""Oct 3 package: Oct 2 package + revised documents + the AS02 PDF as submitted."""
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

W = Path(__file__).resolve().parent
P = W.parents[3]
OLD = W.parent / 'final_pass_20261002' / 'package'
K = W / 'package'
D = W / 'documents'
SUB = P / 'AS03_Final' / 'Submission'
SUBMITTED_AS02 = P.parent / 'AS02 submission.pdf'
REPO = P.parent / 'group-2-anlx'

if K.exists():
    shutil.rmtree(K)
shutil.copytree(OLD, K)
docs = K / 'documents'

# The AS02 report actually uploaded to Canvas replaces the Oct 2 draft as the current AS02 document.
hist = docs / 'history'
for name in ['AS02_Report.pdf', 'AS02_Report.md']:
    shutil.move(docs / name, hist / name.replace('AS02_Report', 'AS02_Report_2026_10_02_draft'))
shutil.copy2(SUBMITTED_AS02, docs / 'AS02_Report_as_submitted.pdf')

for name in ['Team_Research_Report.pdf', 'Team_Research_Report.docx', 'Team_Research_Report.md',
             'Scientific_Poster_36x24.pdf', 'Presentation_Guide.pdf', 'Presentation_Guide.md']:
    shutil.copy2(D / name, docs / name)
(docs / 'figures').mkdir(exist_ok=True)
for f in (W / 'figures').glob('*.png'):
    shutil.copy2(f, docs / 'figures' / f.name)
shutil.copytree(REPO / 'AS02_teammates' / 'tech_william_sun', K / 'sector_sources' / 'tech_as02',
                ignore=shutil.ignore_patterns('__pycache__', '.DS_Store', '.env*'))
(K / 'audits').mkdir(exist_ok=True)
shutil.copy2(REPO / 'verification' / 'verify_tech_as02.py', K / 'audits' / 'verify_tech_as02.py')
code = K / 'code' / 'final_pass_20261003'
code.mkdir(parents=True, exist_ok=True)
for name in ['revise_content.py', 'report_content.json', 'build_report.py', 'build_poster.py', 'build_guide.py',
             'package_final.py', 'presentation_script.md', 'note_c4_fences.py']:
    shutil.copy2(W / name, code / name)

s = (K / 'START_HERE.md').read_text()
s = s.replace('Chih-yu (Finance)', 'Chih-yu Li (Finance)')
s = s.replace(' Chih-yu’s name is retained as supplied; no surname was guessed.', '')
s = s.replace('- `documents/AS02_Report.pdf`: Shreya’s final ten-page AS02 report, using the policy-first series '
              'throughout and appending the exact initial and revised policies.',
              '- `documents/AS02_Report_as_submitted.pdf`: Shreya’s AS02 report exactly as uploaded to Canvas '
              '(nine pages). The Oct 2 draft is kept in `documents/history/`.')
s += ('\nOctober 3 revision: full author name Chih-yu Li; report expanded with a related-work section, an '
      'extended discussion and 32 cited references including every data platform; poster redesigned with '
      'charts generated directly from the saved run metrics (`documents/figures/`); the AS02 report in this '
      'package is now the version submitted to Canvas. Build scripts: `code/final_pass_20261003/`. No '
      'experimental number changed.\n'
      '\nTech AS02 (William Sun): `sector_sources/tech_as02/` holds his LLMBox runtime, posting-level split, '
      '70 safety probes and all Part B/C/D outputs; `audits/verify_tech_as02.py` recomputes every Tech number (run it from the team GitHub repository layout) '
      'without loading a model. His memo (`sector_sources/tech_as02/report/as02_memo.tex`) carries his own AI '
      'disclosure (Claude).\n')
(K / 'START_HERE.md').write_text(s)

u = (K / 'AI_USAGE.md').read_text()
u += ('\nOn October 3, Claude (Anthropic, via Claude Code) independently recomputed the AS01 and AS02 metrics '
      'from the raw prediction and response files, reran the 17 software tests and the AS01 corpus validator, '
      'checked package checksums, verified the added references, revised the team report (author name, '
      'related work, discussion, references), rebuilt the poster with charts read directly from the saved '
      'metrics, and rebuilt this package. No new model experiment, human judgment or timing was created. Later the same day '
      'it verified William Sun’s Tech AS02 results with his recomputation script, confirmed from the raw Pharma '
      'files that 31 of 50 C4 answers would pass every C4 check after fence removal (a post-hoc diagnostic), '
      'and rebuilt the report, posters, presentation guide and package. William’s own AI use is disclosed in his '
      'memo.\n')
(K / 'AI_USAGE.md').write_text(u)
(P / 'AI_USAGE.md').write_text(u)

sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
files = [f for f in sorted(K.rglob('*')) if f.is_file() and f.name != 'SHA256SUMS.json']
(K / 'SHA256SUMS.json').write_text(json.dumps({str(f.relative_to(K)): sha(f) for f in files}, indent=2) + '\n')

zpath = W / 'Team_Project_Package.zip'
with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
    for f in sorted(K.rglob('*')):
        if f.is_file():
            z.write(f, Path('Team_Project') / f.relative_to(K))
with zipfile.ZipFile(zpath) as z:
    assert z.testzip() is None

# Archive the Oct 2 Submission files, then install the Oct 3 set.
prev = W.parent / 'final_pass_20261002' / 'previous_submissions' / 'AS03_Final_as_of_2026_10_03_morning'
prev.mkdir(parents=True, exist_ok=True)
for f in SUB.iterdir():
    if f.is_file():
        shutil.copy2(f, prev / f.name)
        f.unlink()
for name in ['Team_Research_Report.pdf', 'Team_Research_Report.docx', 'Scientific_Poster_36x24.pdf',
             'Presentation_Guide.pdf', 'Presentation_Guide.md']:
    shutil.copy2(D / name, SUB / name)
shutil.copy2(zpath, SUB / zpath.name)
shutil.copy2(K / 'LLM_API_Code.zip', SUB / 'LLM_API_Code.zip')
shutil.copy2(SUBMITTED_AS02, P / 'AS02' / 'Submission' / 'AS02_Report_SUBMITTED_to_Canvas.pdf')
print('package files', len(files), '->', SUB)
