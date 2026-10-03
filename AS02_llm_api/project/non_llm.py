"""Explicit source-table comparator; leaves semantic fields unfilled."""
import argparse
import json
from pathlib import Path
import time
from career_api import reference
from experiments import load_rows, write_rows, score, sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();data=Path(a.data);out=Path(a.out)
    manifest=json.loads((data/'manifest.json').read_text())
    if sha(data/'evaluation.jsonl')!=manifest['split_hashes']['evaluation.jsonl']:
        raise ValueError('Evaluation split changed')
    records=load_rows(data/'evaluation.jsonl')[:50]
    if len(records)!=50:raise ValueError('Expected 50 evaluation records')
    out.mkdir(parents=True,exist_ok=False)
    rows=[]
    for record in records:
        start=time.perf_counter()
        facts={**reference(record),'work_arrangement':'not_stated','required_qualifications':[]}
        response=json.dumps(facts)
        rows.append({'doc_id':record['doc_id'],'mode':'source_table_lookup','facts':facts,
                     'response':response,'raw_output':response,'blocked':False,'reasons':[],
                     'schema_valid':True,'source_errors':[],'schema_error':None,'calls':[],
                     'tool_events':[],'input_tokens':0,'output_tokens':0,
                     'latency_seconds':time.perf_counter()-start})
    write_rows(out/'responses.jsonl',rows)
    metrics=score(rows,records)
    metrics['limitation']='Copies existing source fields only; does not interpret work arrangement or extract qualification sentences. Not a complete semantic-task substitute.'
    metrics['timing_scope']='In-memory field mapping and JSON serialization; excludes dataset loading.'
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
    (out/'manifest.json').write_text(json.dumps({'status':'complete','count':50,'backend':'deterministic_source_table_lookup','source_sha256':sha(__file__),'split_manifest_sha256':sha(data/'manifest.json')},indent=2))
    print('Saved 50 deterministic comparator outputs.')


if __name__=='__main__':main()
