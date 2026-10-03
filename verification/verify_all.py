"""Independent recomputation run on the course VM (Oct 3)."""
import json, re, sys, collections, statistics as st, hashlib, os, subprocess
T = 'tp/Team_Project'
# 1. package checksums
m = json.load(open(f'{T}/SHA256SUMS.json'))
bad = [p for p, h in m.items() if hashlib.sha256(open(os.path.join(T, p), 'rb').read()).hexdigest() != h]
print('[package] checksums ok', len(m) - len(bad), '/', len(m), 'bad', bad)
print('[package] AS02 pdf == submitted:', open(f'{T}/documents/AS02_Report_as_submitted.pdf','rb').read() == open('AS02_submission.pdf','rb').read())
# 2. AS02 runs from raw responses
R = f'{T}/prospective_rerun'
sys.path.insert(0, f'{R}/project')
from career_api import JobFacts, factual_errors
dev = [json.loads(l) for l in open(f'{R}/project/data/development.jsonl')]
ev = [json.loads(l) for l in open(f'{R}/project/data/evaluation.jsonl')]
src = {r['doc_id']: r for r in dev + ev}
for run in ['B1', 'B3', 'C1', 'C2', 'C3', 'C4']:
    rows = [json.loads(l) for l in open(f'{R}/runs/policy_first_gpu/{run}/responses.jsonl')]
    allc = sum(1 for r in rows if not r['blocked'] and r['facts'] and not factual_errors(JobFacts(**r['facts']), src[r['doc_id']]) or False)
    errs = collections.Counter((re.findall(r'\[type=(\w+)', r['schema_error'] or '') or ['none'])[0] for r in rows)
    print(f"[AS02] {run}: valid={sum(r['schema_valid'] for r in rows)} all5={allc} withheld={sum(bool(r['blocked']) for r in rows)} "
          f"tools={sum(bool(r['tool_events']) for r in rows)} lat={st.mean(r['latency_seconds'] for r in rows):.2f} "
          f"in={st.mean(r['input_tokens'] for r in rows):.1f} out={st.mean(r['output_tokens'] for r in rows):.1f} errors={dict(errs)}")
rows = [json.loads(l) for l in open(f'{R}/runs/policy_first_gpu/C1/responses.jsonl')]
comp = val = ok = 0
for r in rows:
    mm = re.search(r'```(?:json)?\s*(.*?)```', r['raw_output'], re.S)
    if mm:
        comp += 1
        try:
            f = JobFacts.model_validate_json(mm.group(1)); val += 1; ok += not factual_errors(f, src[r['doc_id']])
        except Exception:
            pass
print(f'[AS02] C1 fences={comp} valid_after_strip={val} source_ok={ok}')
Dd = f'{R}/runs/policy_first_gpu/D'
P = {p['case_id']: p for p in map(json.loads, open(f'{Dd}/paired_responses.jsonl'))}
H = {h['case_id']: h for h in map(json.loads, open(f'{Dd}/human_labels_transferred_exact_matches.jsonl'))}
blk = collections.Counter(); tot = collections.Counter(); agree = 0
for cid, p in P.items():
    tot[p['category']] += 1; blk[p['category']] += bool(p['guarded']['blocked']); agree += bool(p['guarded']['blocked']) == H[cid]['should_block']
print('[AS02] D blocked', {k: f'{blk[k]}/{tot[k]}' for k in tot}, 'human agreement', f'{agree}/30',
      'lat delta', round(st.mean(p['guarded']['latency_seconds'] - p['baseline']['latency_seconds'] for p in P.values()), 2))
# 3. AS01
A = 'as01/assignment1'
F = ['job_title','employer','location','job_function','work_arrangement','required_qualifications','preferred_qualifications','education_requirements','experience_requirements']
n = lambda x: ' '.join(str(x).strip().lower().split())
def eq(a, b):
    if isinstance(a, list) or isinstance(b, list): return {n(i) for i in (a or [])} == {n(i) for i in (b or [])}
    return (a is None and b is None) or (a is not None and b is not None and n(a) == n(b))
hl = {j['doc_id']: j['human_label'] for j in map(json.loads, open(f'{A}/human_labels.jsonl'))}
ph = [json.loads(l) for l in open(f'{A}/out/phi_v1.jsonl')]
acc = {f: sum(eq((r['extracted'] or {}).get(f), hl[r['doc_id']].get(f)) for r in ph if not r['violation']) / len(ph) for f in F}
print('[AS01] labels', len(hl), 'phi field acc', {k: round(v, 2) for k, v in acc.items()}, 'mean', round(st.mean(acc.values()), 3),
      'valid', sum(not r['violation'] for r in ph), '/25')
