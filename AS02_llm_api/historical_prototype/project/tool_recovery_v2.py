"""Exploratory development-only tool routing; never replaces frozen C2 results."""
import argparse
import json
from pathlib import Path
from career_api import CareerAPI, SYSTEM, REQUEST, JobFacts, parse_facts, factual_errors
from experiments import MLXBackend, load_rows, sha

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True);p.add_argument('--model',required=True)
    p.add_argument('--out',required=True)
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    records=load_rows(Path(a.data)/'development.jsonl')[:3]
    backend=MLXBackend(None,a.model);api=CareerAPI(backend,records)
    settings=dict(seed=2820,temperature=0.2,top_p=0.8,max_new_tokens=256,do_sample=True,top_k=0,repetition_penalty=1.0)
    results=[]
    for i,r in enumerate(records):
        settings={**settings,'seed':2820+i}
        messages=[{'role':'system','content':
            'You are a tool router. The lookup_selected_posting function is available and will be executed by the application. '
            'You must call it to read the requested posting before answering. '
            'Respond only with a JSON array containing one function call. '
            'Use exactly this shape: [{"name":"lookup_selected_posting","arguments":{}}]. '
            'Do not invent job facts or explain the call.'},
            {'role':'user','content':'Read posting ID: '+r['doc_id']}]
        first=backend(messages,settings);calls=[{'messages':messages,**first}]
        result={'doc_id':r['doc_id'],'settings':settings,'calls':calls,'tool_ok':False}
        try:
            raw=first['text'].strip()
            if raw.startswith('functools'):raw=raw[len('functools'):]
            planned=json.loads(raw)
            if not isinstance(planned,list) or len(planned)!=1:raise ValueError('Expected one call')
            call=planned[0]
            if set(call)!={'name','arguments'} or call['name']!='lookup_selected_posting':raise ValueError('Unknown tool')
            if call['arguments']!={}:raise ValueError('No arguments allowed')
            source=api.lookup_posting(r['doc_id'],r['doc_id'])
            result['tool_ok']=True;result['tool_call']=call
            second_messages=[{'role':'system','content':SYSTEM+'\nRequired JSON Schema:\n'+json.dumps(JobFacts.model_json_schema())},
                {'role':'user','content':REQUEST+'\nRead-only lookup_posting result:\n'+json.dumps(source,ensure_ascii=False)}]
            second=backend(second_messages,settings);calls.append({'messages':second_messages,**second})
            facts,error=parse_facts(second['text'])
            result.update(schema_valid=bool(facts),schema_error=error,source_errors=factual_errors(facts,r) if facts else None)
        except (ValueError,TypeError,KeyError) as exc:result['error']=str(exc)
        results.append(result)
        with (out/'responses.jsonl').open('a') as f:f.write(json.dumps(result,ensure_ascii=False)+'\n')
        print(f'Development tool check {i+1}/3: tool_ok={result["tool_ok"]}',flush=True)
    (out/'manifest.json').write_text(json.dumps({'status':'complete','purpose':'Exploratory v2 development-only routing; current record bound server-side; not C2 evaluation',
        'count':len(results),'tool_successes':sum(r['tool_ok'] for r in results),'input_ids':[r['doc_id'] for r in records],
        'backend':'mlx','model':a.model,'code_sha256':sha(__file__),'settings':settings},indent=2))

if __name__=='__main__':main()
