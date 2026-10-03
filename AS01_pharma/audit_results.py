"""Verify delivered metrics against saved predictions without rerunning inference."""
import json, hashlib, random
from pathlib import Path
from datetime import datetime, timezone
import extraction as e
from schema import validate_against_schema
ROOT=Path(__file__).resolve().parent

def read(name):return json.loads((ROOT/name).read_text())
def rows(name):return [json.loads(s) for s in (ROOT/name).read_text().splitlines() if s.strip()]
def main():
    corpus=rows('corpus.jsonl');labels=e.load_human_labels(str(ROOT/'human_labels.jsonl'))
    sample=read('evaluation_sample.json');taxonomy=read('taxonomy.json')
    assert sorted(r['doc_id'] for r in random.Random(sample['seed']).sample(corpus,25))==sample['doc_ids']
    baseline=read('out/evaluation.json');recovery=read('out/recovery.json')
    bm=read('out/baseline_manifest.json');rm=read('out/recovery_manifest.json')
    for manifest in [bm,rm]:
        assert manifest['corpus_sha256']==e.file_sha(ROOT/'corpus.jsonl')
        assert manifest['labels_sha256']==e.file_sha(ROOT/'human_labels.jsonl')
        assert manifest['fields']==taxonomy['fields']
        assert manifest['baseline_prompt']==taxonomy['prompts']['v1_original']
    for k in ['corpus_sha256','labels_sha256','fields','baseline_prompt','doc_ids','backend','model','max_new_tokens','reference_origin']:
        assert bm[k]==rm[k],k
    records=[r for r in corpus if r['doc_id'] in labels]
    raw=rows('out/phi_v1.jsonl')
    recalculated=e.run_phi_evaluation(records,labels,rows=raw)
    for k in ['field_accuracy','field_errors','list_field_scores','all_fields_correct_rate','schema_violation_rate','per_record']:
        assert baseline[k]==recalculated[k],k
    ids=taxonomy['recovery_doc_ids'];subset=[r for r in records if r['doc_id'] in ids]
    raw2=rows('out/phi_v2.jsonl');recalc2=e.run_phi_evaluation(subset,labels,rows=raw2)
    assert raw2==recovery['raw_predictions']
    assert recovery['per_record_after']==recalc2['per_record']
    assert recovery['after']['field_accuracy']==recalc2['field_accuracy']
    assert recovery['after']['field_errors']==recalc2['field_errors']
    before={r['doc_id']:r for r in baseline['per_record']}
    for name,t in recovery['field_transitions'].items():
        fixed=sum(not before[r['doc_id']]['fields'][name]['ok'] and r['fields'][name]['ok'] for r in recalc2['per_record'])
        regressed=sum(before[r['doc_id']]['fields'][name]['ok'] and not r['fields'][name]['ok'] for r in recalc2['per_record'])
        assert t=={'fixed':fixed,'regressed':regressed}
    for r in raw+raw2:
        if not r['violation']:assert validate_against_schema(r['extracted'],e.EXTRACTION_SCHEMA) is None
        assert r['raw_output'].strip()
    for r in rows('human_labels.jsonl'):
        assert r['annotation_origin']=='human' and r['annotation_status']=='complete' and r['human_reviewed'] is True
    result={'checked_at':datetime.now(timezone.utc).isoformat(),'status':'PASS','checks':['seed-820 sample reproduced','baseline and recovery input hashes unchanged','all baseline metrics recalculated from 25 saved predictions','all recovery metrics recalculated from 19 saved predictions','field fixes/regressions independently recounted','valid predictions meet schema','human annotation status verified'],'baseline_matching_fields':sum(v['ok'] for r in baseline['per_record'] for v in r['fields'].values()),'baseline_field_denominator':225,'baseline_valid_records':sum(not r['violation'] for r in raw),'recovery_work_arrangement_fixes':recovery['field_transitions']['work_arrangement']['fixed'],'recovery_schema_failures':sum(bool(r['violation']) for r in raw2),'recovery_other_field_regressions':sum(t['regressed'] for t in recovery['field_transitions'].values())}
    (ROOT/'out/final_accuracy_audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
