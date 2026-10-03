"""Select a recurring baseline failure and preserve the targeted revised prompt."""
import json
from pathlib import Path
import extraction as e
ROOT=Path(__file__).resolve().parent

def main():
    evaluation=json.loads((ROOT/'out/evaluation.json').read_text())
    ids=[r['doc_id'] for r in evaluation['per_record'] if not r.get('violation') and isinstance(r.get('phi'),dict) and not r['fields']['work_arrangement']['ok']]
    # Explicitly check saved prediction schema validity, not just successful parsing.
    raw={r['doc_id']:r for r in map(json.loads,(ROOT/'out/phi_v1.jsonl').read_text().splitlines())}
    ids=[i for i in ids if raw[i]['violation'] is None]
    if len(ids)<2:raise SystemExit('Fewer than two schema-valid work-arrangement errors; inspect baseline before choosing a different recovery target.')
    taxonomy=json.loads((ROOT/'taxonomy.json').read_text())
    original=e.build_prompts(taxonomy)['v1_original']
    addition='''

WORK-ARRANGEMENT EVIDENCE PROCEDURE (apply only to work_arrangement):
1. Locate an explicit work-mode statement for this specific job in its description, source location, or role-specific recruitment tags. Interpret #LI-Remote as remote and #LI-Hybrid as hybrid. Explicit weekly partial onsite attendance (for example 1-3 days per week) is hybrid; explicit full-time onsite or five in-office days is onsite.
2. A city, company headquarters address, laboratory/manufacturing responsibilities, travel requirement, or the word Field does NOT establish onsite, hybrid, or remote work. Do not infer a mode from what a job normally requires.
3. Generic compensation text about salaries for candidates working onsite is conditional salary boilerplate, NOT this role's work arrangement. It must not override #LI-Remote, #LI-Hybrid, or an explicit role-specific statement.
4. If there is no explicit role-specific work-mode evidence, output not_stated. If explicit alternatives remain unresolved, output not_stated.
Examples: location Cambridge, MA with no role-specific mode statement -> not_stated. Location Redwood City with #LI-Remote plus generic onsite salary boilerplate -> remote.
Keep every other field and its existing extraction rules unchanged. Do not add evidence keys to the JSON schema.'''
    taxonomy['prompts']={'v1_original':original,'v2_recovery':original+addition}
    taxonomy['recovery_doc_ids']=ids
    taxonomy['recovery_selection']='All schema-valid baseline records with an incorrect work_arrangement; selected before running revised prompt.'
    (ROOT/'taxonomy.json').write_text(json.dumps(taxonomy,ensure_ascii=False,indent=2))
    examples=[]
    for r in evaluation['per_record']:
        if r['doc_id'] in ids:
            examples.append(f"- {r['doc_id']}: baseline `{r['phi']['work_arrangement']}`; reference `{r['human']['work_arrangement']}`.")
    text='# Prompt history\n\nReference provenance: independently reviewed and finalized by Shreya Verma.\n\n## v1 original\n\nFull baseline prompt, used for all 25 records:\n\n```text\n'+original+'\n```\n\n## Observed failure and selection\n\n'+taxonomy['recovery_selection']+' The recurring failure is inference of work mode without sufficient role-specific evidence or confusion with generic employer/compensation text. Invalid JSON records are excluded from this targeted diagnostic because they do not provide a schema-valid work-mode prediction. No revised results were available during selection.\n\n'+'\n'.join(examples)+'\n\n## v2 recovery\n\nMotivation: force an evidence check and distinguish role tags from compensation boilerplate. Only work-arrangement instructions change; schema, reference labels, full input and decoding settings stay fixed. All original-prompt instructions are retained.\n\n```text\n'+original+addition+'\n```\n\n## Outcome\n\nActual before/after outcomes, fixes and regressions are saved in recovery.json; raw output is saved in phi_v2.jsonl. The rerun diagnoses known errors and is not a held-out performance estimate. No additional prompt versions were tried.\n'
    (ROOT/'out/prompt_history.md').write_text(text)
    print('Selected',len(ids),'schema-valid work-arrangement failures:',*ids,sep='\n')
if __name__=='__main__':main()
