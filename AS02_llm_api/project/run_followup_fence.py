"""Follow-up F1: C4 verification with a fence-tolerant parser on unused evaluation records 51-100.

The frozen AS02 modules are imported unchanged; only the parser used inside
LLMBoxCareerAPI.answer is swapped for this run. See followup_fence/PREREGISTRATION.md.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

import llmbox_integration
from career_api import parse_facts, factual_errors
from experiments import load_rows
from fence_parser import parse_facts_tolerant
from llmbox_integration import LLMBoxCareerAPI

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / 'followup_fence'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--limit', type=int, default=50)
    a = ap.parse_args()
    run = OUT / 'run'
    run.mkdir(parents=True, exist_ok=False)
    prereg = OUT / 'PREREGISTRATION.md'
    protocol = {'started_at': datetime.now(timezone.utc).isoformat(), 'model': a.model,
                'preregistration_sha256': sha(prereg),
                'code_sha256': {p.name: sha(p) for p in [ROOT / 'fence_parser.py', Path(__file__),
                                ROOT / 'llmbox_integration.py', ROOT / 'career_api.py']},
                'records': 'evaluation.jsonl[50:100]', 'mode': 'verified (C4) via Modes.run_structured_output'}
    (run / 'protocol.json').write_text(json.dumps(protocol, indent=2))

    evaluation = load_rows(ROOT / 'data' / 'evaluation.jsonl')
    selected = evaluation[50:50 + a.limit]
    llmbox_integration.parse_facts = lambda text: parse_facts_tolerant(text)[:2]
    api = LLMBoxCareerAPI(evaluation, a.model)
    rows = []
    with (run / 'responses.jsonl').open('x') as f:
        for k, record in enumerate(selected):
            i = 50 + k
            settings = dict(seed=820 + i, temperature=.2, top_p=.8, max_new_tokens=256, do_sample=True,
                            top_k=0, repetition_penalty=1.)
            r = api.answer(record['doc_id'], 'verified', settings)
            strict, _ = parse_facts(r['raw_output']) if r['raw_output'] else (None, None)
            r['strict_schema_valid'] = strict is not None
            r['fence_stripped'] = parse_facts_tolerant(r['raw_output'])[2] if r['raw_output'] else False
            r['delivered'] = not r['blocked']
            r['settings'] = settings
            f.write(json.dumps(r, ensure_ascii=False) + '\n'); f.flush(); rows.append(r)
            print(f'F1 {k + 1}/{len(selected)}: strict={r["strict_schema_valid"]} tolerant={r["schema_valid"]} '
                  f'delivered={r["delivered"]} {r["latency_seconds"]:.1f}s', flush=True)

    n = len(rows)
    tol = sum(r['schema_valid'] for r in rows)
    delivered = sum(r['delivered'] for r in rows)
    metrics = {'n': n, 'strict_schema_valid': sum(r['strict_schema_valid'] for r in rows),
               'tolerant_schema_valid': tol, 'fence_stripped': sum(r['fence_stripped'] for r in rows),
               'c4_delivered': delivered,
               'withhold_reasons': {k: sum(1 for r in rows if r['reasons'][:1] == [k])
                                    for k in sorted({x for r in rows for x in r['reasons'][:1]})},
               'mean_latency_seconds': statistics.mean(r['latency_seconds'] for r in rows),
               'mean_input_tokens': statistics.mean(r['input_tokens'] for r in rows),
               'mean_output_tokens': statistics.mean(r['output_tokens'] for r in rows),
               'decision': ('works' if tol >= 35 and delivered >= 20 else
                            'partly works' if tol >= 35 else 'does not work') if n == 50 else 'incomplete'}
    (run / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    protocol['ended_at'] = datetime.now(timezone.utc).isoformat()
    (run / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
