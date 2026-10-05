"""Build the final-project supporting package and install the submission files.

package_base/ (frozen evidence) + current documents + Tech AS02 + fence follow-up
-> package/ -> Team_Project_Package.zip -> ../Submission/
"""
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

B = Path(__file__).resolve().parent            # AS03_Final/build
P = B.parents[1]                                # Project/
REPO = P.parent / 'group-2-anlx'                # team GitHub clone
A = P / 'AS02' / 'as02_prospective'
SUB = P / 'AS03_Final' / 'Submission'
SUBMITTED_AS02 = P / 'AS02' / 'Submission' / 'AS02_Report_SUBMITTED_to_Canvas.pdf'
K = B / 'package'
skip = shutil.ignore_patterns('__pycache__', '.DS_Store', '.env*')

if K.exists():
    shutil.rmtree(K)
shutil.copytree(B / 'package_base', K)
docs = K / 'documents'
for name in ['Team_Research_Report.pdf', 'Team_Research_Report.docx', 'Team_Research_Report.md',
             'Scientific_Poster_36x24.pdf', 'Presentation_Guide.pdf', 'Presentation_Guide.md']:
    shutil.copy2(B / 'documents' / name, docs / name)
shutil.copy2(SUBMITTED_AS02, docs / 'AS02_Report_as_submitted.pdf')
shutil.copytree(B / 'figures', docs / 'figures', dirs_exist_ok=True)

# Tech AS02 (William Sun) and its recomputation script, from the team repository.
shutil.copytree(REPO / 'AS02_teammates' / 'tech_william_sun', K / 'sector_sources' / 'tech_as02', ignore=skip)
shutil.copy2(REPO / 'verification' / 'verify_tech_as02.py', K / 'audits' / 'verify_tech_as02.py')

# Fence-tolerant parser follow-up (Pharma), if present.
if (A / 'followup_fence' / 'run' / 'metrics.json').exists():
    shutil.copytree(A / 'followup_fence', K / 'prospective_rerun' / 'followup_fence', ignore=skip, dirs_exist_ok=True)
    for name in ['fence_parser.py', 'test_fence_parser.py', 'run_followup_fence.py']:
        shutil.copy2(A / 'project' / name, K / 'prospective_rerun' / 'project' / name)

for name in ['report_content.json', 'build_report.py', 'build_poster.py', 'build_guide.py', 'presentation_script.md',
             'package_final.py']:
    (K / 'code').mkdir(exist_ok=True)
    shutil.copy2(B / name, K / 'code' / name)

(K / 'START_HERE.md').write_text("""# Group 2 final project: supporting package

Team: Shreya Verma (Pharma), Chih-yu Liu (Finance), William Sun (Tech). CMU 95820, Fall 2026.

## Documents
- `documents/Team_Research_Report.pdf` (+ `.docx`, `.md`): the team research report.
- `documents/Scientific_Poster_36x24.pdf`: the poster (charts generated from the run files, `documents/figures/`).
- `documents/Presentation_Guide.pdf`: three-person script and Q&A owners.
- `documents/AS02_Report_as_submitted.pdf`: Shreya's AS02 report exactly as submitted.

## Appendix data and evidence
- `datasets/original_combined.jsonl`: all 650 records (sector, team_doc_id, unchanged original record).
- `datasets/pharma_as02/`: the Pharma development/evaluation split (100/100, seed 820) used by the API runs.
- `datasets/*_as01_labeled_records.jsonl`: the three sectors' reference-label subsets (25 / 28 / 25).
- `sector_sources/`: each sector's AS01 corpus, labels and memo; `sector_sources/tech_as02/` is William Sun's Tech
  AS02 API (runtime, posting-level split, 70 probes, all outputs and metrics).
- `prospective_rerun/`: Pharma AS02 code (`project/`), LLMBox snapshot (`llmbox/`), frozen and revised policies
  (`governance/`), all six 50-request runs and 30 safety pairs with per-run metrics (`runs/policy_first_gpu/`), and
  the pre-registered fence-tolerant parser follow-up (`followup_fence/`).
- `pharma_api/`: the earlier integrated series (outputs identical to the final runs); `historical_prototype/` and
  `prototype_cost/`: the prototype that produced the human safety labels and timings.
- `metrics/`: corpus audit and teammate-reported results with provenance.
- `LLM_API_Code.zip`: the API code; `code/`: scripts that build the report, poster, guide and this package.
- `AI_USAGE.md` and `assistance/`: AI-assistance disclosure and available records.

## Checks
- `python3 code/verify_package.py`: record counts, IDs and checksums (standard library only).
- `audits/verify_tech_as02.py`: recomputes every Tech AS02 number (run it from the team repository layout).
- Pharma tests: `cd prospective_rerun/project && python -m unittest test_career_api test_llmbox_integration test_fence_parser`.

No model was fine-tuned. Model weights, environments and credentials are excluded.
""")

u = (P / 'AI_USAGE.md').read_text()
(K / 'AI_USAGE.md').write_text(u)

sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
files = [f for f in sorted(K.rglob('*')) if f.is_file() and f.name != 'SHA256SUMS.json']
(K / 'SHA256SUMS.json').write_text(json.dumps({str(f.relative_to(K)): sha(f) for f in files}, indent=2) + '\n')
zpath = B / 'Team_Project_Package.zip'
with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
    for f in sorted(K.rglob('*')):
        if f.is_file():
            z.write(f, Path('Team_Project') / f.relative_to(K))
with zipfile.ZipFile(zpath) as z:
    assert z.testzip() is None

for f in SUB.iterdir():
    if f.is_file():
        f.unlink()
for name in ['Team_Research_Report.pdf', 'Team_Research_Report.docx', 'Scientific_Poster_36x24.pdf',
             'Presentation_Guide.pdf']:
    shutil.copy2(B / 'documents' / name, SUB / name)
shutil.move(str(zpath), SUB / zpath.name)
shutil.copy2(K / 'LLM_API_Code.zip', SUB / 'LLM_API_Code.zip')
shutil.rmtree(K)
print('package files', len(files), '->', SUB)
