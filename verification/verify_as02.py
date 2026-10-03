import json, re, statistics as st, random
D='runs/policy_first_gpu'
dev=[json.loads(l) for l in open('project/data/development.jsonl')]
ev=[json.loads(l) for l in open('project/data/evaluation.jsonl')]
src={r['doc_id']:r for r in dev+ev}
def n(x): return ' '.join(str(x).strip().lower().split()) if x is not None else None
WA={'onsite','hybrid','remote','not_stated'}
for run in ['B1','B3','C1','C2','C3','C4']:
    rows=[json.loads(l) for l in open(f'{D}/{run}/responses.jsonl')]
    man=json.load(open(f'{D}/{run}/manifest.json'))
    ids_ok=[r['doc_id'] for r in rows]==man['input_ids']
    split='dev' if all(r['doc_id'] in {x['doc_id'] for x in dev} for r in rows) else 'eval' if all(r['doc_id'] in {x['doc_id'] for x in ev} for r in rows) else 'MIXED'
    first50 = [r['doc_id'] for r in rows]==[x['doc_id'] for x in (dev if split=='dev' else ev)][:50]
    sv=sum(r['schema_valid'] for r in rows); bl=sum(bool(r['blocked']) for r in rows)
    # independent all-5 check: delivered (not blocked) + parsed facts + fields match source
    allc=0
    for r in rows:
        if r['blocked'] or not r['facts']: continue
        f=r['facts']; s=src[r['doc_id']]; t=s['table_json']
        exp={'doc_id':s['doc_id'],'source_url':s['source_url'],'job_title':t.get('job_title'),'employer':t.get('employer'),'location':t.get('location')}
        if all(n(f.get(k))==n(v) for k,v in exp.items()): allc+=1
    fence=sum('```' in (r['raw_output'] or '') for r in rows)
    wa_err=sum('work_arrangement' in (r['schema_error'] or '') for r in rows)
    json_err=sum(bool(r['schema_error']) and 'validation error' not in r['schema_error'] for r in rows)
    tools=sum(any(e.get('ok',e.get('status')=='ok') if isinstance(e,dict) else False for e in (r['tool_events'] or [])) for r in rows)
    ntool=sum(bool(r['tool_events']) for r in rows)
    lat=st.mean(r['latency_seconds'] for r in rows); it=st.mean(r['input_tokens'] for r in rows); ot=st.mean(r['output_tokens'] for r in rows)
    s0=rows[0]['settings']
    print(f"{run} n={len(rows)} ids_match_manifest={ids_ok} split={split} first50={first50} valid={sv} all5={allc} blocked={bl} fences={fence} wa_err={wa_err} nonvalidation_err={json_err} tool_rows={ntool} lat={lat:.2f} in={it:.1f} out={ot:.1f}")
    print('   settings', {k:s0.get(k) for k in ['temperature','top_p','max_new_tokens','top_k','repetition_penalty','do_sample','seed']} if isinstance(s0,dict) else s0)
# split check
print('dev/eval overlap', len({x['doc_id'] for x in dev}&{x['doc_id'] for x in ev}))
