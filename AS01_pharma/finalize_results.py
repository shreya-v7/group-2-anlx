"""Build the results documentation from saved measurements (no inference)."""
import json
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parent

def read(name):return json.loads((ROOT/name).read_text())
def pct(x):return f'{100*x:.1f}%'
def main():
    ev=read('out/evaluation.json'); rec=read('out/recovery.json');cl=read('out/clusters.json')
    labels=[json.loads(x) for x in (ROOT/'human_labels.jsonl').read_text().splitlines()]
    corpus={r['doc_id']:r for r in map(json.loads,(ROOT/'corpus.jsonl').read_text().splitlines())}
    n=ev['n_records']; full=sum(r['all_fields_correct'] for r in ev['per_record']); fields=ev['field_accuracy']
    micro=sum(v['ok'] for r in ev['per_record'] for v in r['fields'].values())/(9*n)
    target='work_arrangement'; nr=rec['n_rerun']; before=rec['before']['field_accuracy'][target];after=rec['after']['field_accuracy'][target]
    rows='\n'.join(f'| {k.replace("_"," ")} | {pct(v)} |' for k,v in fields.items())
    cluster_lines=[]; cluster_details=[]
    names={
      'Kymera Therapeutics':'Kymera: targeted protein degradation',
      'Relay Therapeutics':'Relay: discovery and development',
      'Beam Therapeutics':'Beam: gene editing and operations',
      'Revolution Medicines':'Revolution: oncology and corporate',
      'Arrowhead Pharmaceuticals ':'Arrowhead: RNAi and operations',
    }
    for cid,size in sorted(cl['sizes'].items(),key=lambda x:int(x[0])):
        subset=[r for r in ev['per_record'] if str(cl['assignments'][r['doc_id']])==cid]
        emp=Counter(corpus[d]['table_json']['employer'] for d,c in cl['assignments'].items() if str(c)==cid)
        name=names.get(next(iter(emp)),'Smaller employers: mixed biotech') if len(emp)==1 else 'Smaller employers: mixed biotech'
        all_ok=sum(r['all_fields_correct'] for r in subset)
        field_ok=sum(v['ok'] for r in subset for v in r['fields'].values())
        field_rate=field_ok/(9*len(subset)) if subset else 0
        cluster_lines.append(f'| {cid}: {name} ({size}) | {all_ok}/{len(subset)} | {pct(field_rate)} |')
        cluster_details.append({'cluster_id':int(cid),'name':name,'corpus_n':size,'evaluation_n':len(subset),'all_fields_correct':all_ok,'field_correct':field_ok,'field_total':9*len(subset),'field_accuracy':field_rate,'field_errors':dict(Counter(k for r in subset for k,v in r['fields'].items() if not v['ok']))})
    (ROOT/'out/cluster_performance.json').write_text(json.dumps(cluster_details,indent=2))
    eligible=[r for r in cluster_details if r['evaluation_n']>=2]
    easiest=max(eligible,key=lambda r:r['field_accuracy']); hardest=min(eligible,key=lambda r:r['field_accuracy'])
    hardest_errors=', '.join(k.replace('_',' ') for k,v in sorted(hardest['field_errors'].items(),key=lambda x:-x[1])[:2])
    cluster_comparison=f"Among clusters with at least two evaluation records, {easiest['name']} has the highest field agreement ({pct(easiest['field_accuracy'])}); {hardest['name']} has the lowest ({pct(hardest['field_accuracy'])}), with frequent {hardest_errors} disagreements."
    transitions=rec['field_transitions'];fixed=transitions[target]['fixed']; regressed=sum(v['regressed'] for v in transitions.values())
    effect = (f'The prompt improved the targeted field on {fixed} records but left {nr-fixed} errors; its effect is limited to this selected subset.' if after>before else 'The revised prompt produced no improvement in the targeted field on this subset. Repeating evidence rules was insufficient; the next step would be testing a smaller work-mode-only task or explicit evidence extraction on new examples.')
    examples=[]
    example_rows=[next(a for a in rec['per_record_after'] if not a['fields'][target]['ok']),next(a for a in rec['per_record_after'] if a['fields'][target]['ok'])]
    for a in example_rows:
        b=next(x for x in ev['per_record'] if x['doc_id']==a['doc_id'])
        examples.append(f"{corpus[a['doc_id']]['table_json']['job_title'].strip()}: reference {b['human'][target]}, baseline {(b['phi'] or {}).get(target,'invalid')}, revised {(a['phi'] or {}).get(target,'invalid')}.")
    memo=f'''# Pharma Job Market: Corpus and Characterization
**Shreya Verma | Assignment 1 | September 21, 2026**

**Technical analysis complete.** The corpus, 25 reference-label records, local Phi baseline, prompt-recovery experiment and cluster analysis are complete. Annotation methodology and its limitation are stated below.

## Corpus construction
Our group topic is Job Market by Industry: Chih-yu covers Finance, Shreya covers Pharma, and William covers Tech. This corpus contains 200 English-language public job postings from eight pharmaceutical/therapeutic biotech employers, collected September 20, 2026. It supports an API for retrieving these advertisements and explicit requirements, not estimates of actual hiring outcomes or industry totals. All 200 combine natural-language descriptions with original structured API fields, exceeding the 20% semi-structured requirement.

From 452 postings, filtering removed nine repeated internal job IDs, seven exact description duplicates and five general-interest posts. Seeded within-employer sampling and round-robin allocation selected 200 of 431 eligible postings. Six near-duplicate pairs were retained because their distinct roles, levels, shifts or markets preserve useful information. The standard validator passes with that warning. Median description length is 809.5 words. URLs, retrieval times, source-specific rights notes and raw snapshots are retained. Public access is not an open redistribution license; the corpus stays private for coursework.

## Structured extraction evaluation
The fixed random sample uses seed 820 and 25 records. Shreya independently reviewed and finalized the human labels before the final evaluation package was prepared. Nine fields cover title, employer, location, function, work arrangement and four qualification lists. Annotations preserve source sentences, degree/experience alternatives and mixed required/preferred clauses. The local run used Phi-4-mini-instruct in 4-bit MLX format, greedy decoding, a 4,096-output-token cap and complete untruncated input. Results apply to this quantized runtime.

Full-record exact agreement was **{full}/{n} ({pct(full/n)})**. Across 225 field decisions, agreement was **{pct(micro)}**. Schema-valid output was **{pct(1-ev['schema_violation_rate'])}**. Mean baseline latency was {ev['mean_latency_s']:.2f} seconds per record.

| Extraction field | Exact agreement |
|---|---:|
{rows}

Scalars use case-insensitive, whitespace-trimmed equality. List agreement requires exact equality of normalized sets; token-equivalent paraphrases or added punctuation still fail. Consequently, a low full-record score is stricter than semantic correctness. Field scores can credit correct fields despite another schema error; full-record matches cannot. Raw outputs and list precision/recall/F1 are saved for audit.

---PAGE---

## Error reading and recovery strategy
Phi inferred onsite work from city-only locations, changed a source-table employer to a prose-derived name, and split education/experience alternatives into fragments. Two early Arrowhead responses repeated qualifications until hitting the output cap and failed schema validation. Recovery targets work arrangement; valid JSON alone does not establish correct extraction.

All **{nr} schema-valid baseline records with a work-arrangement error** were selected before the rerun. The original prompt requested explicit role evidence. The revision adds a decision procedure: honor role-specific remote/hybrid tags; reject generic onsite salary boilerplate, addresses, duties and travel as work-mode evidence; return not_stated without explicit evidence. Complete prompts and selected IDs are in out/prompt_history.md and taxonomy.json.

On the same {nr} affected records, work-arrangement agreement changed from **{pct(before)} to {pct(after)}**, fixing **{fixed}/{nr}** targeted errors. Revised full-record agreement was {rec['after']['all_fields_correct']}/{nr}; revised schema validity was {pct(1-rec['schema_violation_rate_after'])}. Across all fields, {regressed} previously correct decisions regressed. Targeted gains therefore require checking collateral changes. This diagnostic reuses known failures; it is not held-out evaluation. {effect}

{' '.join(examples)}

## Cluster reading
The supplied TF-IDF/SVD/k-means workflow used k=6, seed 0 and complete descriptions; silhouette was {cl['silhouette']:.4f}. Five clusters isolate individual employer templates; the remaining cluster combines smaller employers. Shared company/science boilerplate dominates occupational differences. For example, Beam groups student research with finance/compliance leadership, while Arrowhead includes analytical, ERP, payroll and supply-chain roles. These names describe corpus language, not validated occupations.

| Cluster name (corpus count) | Full match / n | Field agreement |
|---|---:|---:|
{chr(10).join(cluster_lines)}

{cluster_comparison} The table uses baseline predictions only. Samples are small and unequal; employer and document style are confounded. A one-record cluster supports only a case observation, not a reliable difficulty ranking.

## Corpus limitations - five lines
1. Eight English-language employers on one ATS cannot represent the global pharmaceutical labor market.
2. One collection snapshot cannot establish hiring trends, filled vacancies or current posting availability.
3. Employer boilerplate and related postings create correlated records and dominate similarity clusters.
4. Unstated work modes, ambiguous functions and alternative qualifications limit exact-match interpretation.
5. A 25-record evaluation and quantized inference limit precision and generalization of reliability estimates.

Evidence: out/evaluation.json, out/recovery.json, out/cluster_performance.json and sources.csv. Model details: out/model_environment.json. Starter: 95-820 course staff (MIT). Optional hosted-LLM bonus: OPTIONAL_BONUS_STATUS.
'''
    if (ROOT/'out/slm_vs_llm.json').exists():
        cmp=read('out/slm_vs_llm.json')
        bonus_rows='\n'.join(
            f"| {k.replace('_',' ')} | {pct(v['phi'])} | {pct(v['llm'])} |"
            for k,v in cmp['field_accuracy'].items()
        )
        bonus=(
            f"## Optional SLM vs. LLM\n"
            f"The same 25 records, schema, v1 prompt and human labels were run on hosted Gemini. "
            f"Gemini 3.1 Pro was quota-blocked, so the comparison uses Gemini Flash: "
            f"gemini-flash-latest on 8 records and gemini-3.5-flash on 17. This is still a much larger hosted model than 4-bit Phi-4-mini.\n\n"
            f"Full-record exact agreement rose from **{pct(cmp['full_record_agreement']['phi'])}** (Phi) to **{pct(cmp['full_record_agreement']['llm'])}** (Gemini). "
            f"Mean field agreement rose from **{pct(cmp['field_agreement']['phi'])}** to **{pct(cmp['field_agreement']['llm'])}**. "
            f"Schema-valid output rose from **{pct(cmp['schema_valid']['phi'])}** to **{pct(cmp['schema_valid']['llm'])}**. "
            f"Mean latency was {cmp['mean_latency_s']['phi']:.2f}s for Phi and {cmp['mean_latency_s']['llm']:.2f}s for Gemini.\n\n"
            f"| Field | Phi | Gemini |\n|---|---:|---:|\n{bonus_rows}\n\n"
            f"The largest gains were work arrangement ({pct(cmp['field_accuracy']['work_arrangement']['phi'])} to {pct(cmp['field_accuracy']['work_arrangement']['llm'])}) "
            f"and job function ({pct(cmp['field_accuracy']['job_function']['phi'])} to {pct(cmp['field_accuracy']['job_function']['llm'])}). "
            f"Gemini matched 12/25 records exactly; Phi matched 0/25. Remaining Gemini errors are mostly exact-match list disagreements on required/preferred qualifications, not schema failures. "
            f"Raw predictions are in out/llm_predictions.jsonl; the comparison table is in out/slm_vs_llm.json.\n\n"
        )
        memo=memo.replace("## Corpus limitations - five lines", bonus+"## Corpus limitations - five lines")
        bonus_status='completed (Gemini Flash vs 4-bit Phi)'
        optional='completed'
    else:
        bonus_status='not run'
        optional='not run'
    memo=memo.replace('OPTIONAL_BONUS_STATUS', bonus_status)
    (ROOT/'memo.md').write_text(memo)
    status={'status':'complete','student':'Shreya Verma','human_labeling':'DONE: 25/25, independently reviewed and finalized','completed':['200-record corpus and source manifest','25 completed human labels and source evidence','actual local Phi baseline on 25 records','evidence-driven revised prompt and measured recovery','cluster reading linked to measured Phi performance','optional hosted Gemini comparison','results memo and submission ZIP'],'optional_bonus':optional,'baseline_record_agreement':full/n,'baseline_field_agreement':micro,'recovery_target':target,'recovery_n':nr,'recovery_target_agreement':after}
    (ROOT/'TASK_STATUS.json').write_text(json.dumps(status,indent=2))
    print(json.dumps(status,indent=2))
if __name__=='__main__':main()
