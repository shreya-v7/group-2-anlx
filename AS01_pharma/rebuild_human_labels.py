"""Rebuild the finalized human labels from reviewed source-line selections.

The selections were independently reviewed and finalized by Shreya Verma.
Line numbers are zero based.
"""
import json
from pathlib import Path
from datetime import datetime, timezone
ROOT = Path(__file__).resolve().parent

def span(a,b): return list(range(a,b+1))
# sample index: function, arrangement, required, preferred, education, experience
DECISIONS = [
('manufacturing_supply','onsite',span(34,41)+span(43,46),[34,41,42,48,49,50],[34],[35]),
('manufacturing_supply','not_stated',span(23,33),span(35,38),[23],[23]),
('data_technology','not_stated',span(27,34),span(36,41),[27],[28]),
('corporate_other','onsite',span(20,25)+span(27,30),[26,32],[],[20]),
('clinical_development','not_stated',span(15,25),span(27,29),[15,16],[15,16]),
('manufacturing_supply','not_stated',span(17,25),[27],[17],[18]),
('manufacturing_supply','onsite',span(27,30),[],[27,28],[28]),
('corporate_other','not_stated',[21]+span(23,35),[21,22],[21],[23]),
('clinical_development','hybrid',span(31,46),[31],[31],[32]),
('manufacturing_supply','not_stated',span(22,26)+span(28,32),[25,27],[22],[23]),
('research_discovery','not_stated',span(16,24),span(26,28),[16,24],[17]),
('corporate_other','not_stated',[16,17]+span(19,25),[17,25],[16],[16,17]),
('commercial_medical_affairs','not_stated',[29,33,36,38,39,40],[30,31,32,35],[29],[29]),
('clinical_development','not_stated',span(25,30),[],[25],[25]),
('regulatory','not_stated',span(18,21)+span(23,26),[18,22],[18],[19]),
('clinical_development','not_stated',span(20,23)+[25,26],[24],[],[20,21]),
('data_technology','remote',span(7,12),[12],[7],[10]),
('manufacturing_supply','not_stated',[15,16]+span(18,27),[16,17,20],[15],[16]),
('research_discovery','not_stated',span(24,32),[],[24],[25]),
('research_discovery','hybrid',span(15,20),span(22,24),[20],[15]),
('clinical_development','hybrid',[37,38,40,41,42,43,44,46,47,48,49,51,52],[37,44,53]+span(55,59),[37],[40,41]),
('commercial_medical_affairs','remote',span(22,32),span(34,38),[22],[24]),
('quality','remote',span(23,32),span(34,37),[23],[24,25]),
('commercial_medical_affairs','remote',span(22,30),span(32,34),[22],[22]),
('corporate_other','not_stated',[37]+span(39,48),[37,38],[],[37]),
]

def main():
    corpus={r['doc_id']:r for r in map(json.loads,(ROOT/'corpus.jsonl').read_text().splitlines())}
    sample=json.loads((ROOT/'evaluation_sample.json').read_text())
    # Same sorted corpus order as the initial annotation template.
    ids=sorted(sample['doc_ids'])
    assert len(ids)==len(DECISIONS)==25
    rows=[]; evidence=[]
    for doc_id,decision in zip(ids,DECISIONS):
        rec=corpus[doc_id]; lines=rec['raw_text'].splitlines()
        fn,mode,req,pref,edu,exp=decision
        label={k:rec['table_json'].get(k) or 'not_stated' for k in ['job_title','employer','location']}
        label.update(job_function=fn,work_arrangement=mode)
        refs={}
        for field,indices in zip(['required_qualifications','preferred_qualifications','education_requirements','experience_requirements'],[req,pref,edu,exp]):
            label[field]=list(dict.fromkeys(lines[n] for n in indices))
            refs[field]=indices
        rows.append({'doc_id':doc_id,'human_label':label,'annotation_status':'complete','annotation_origin':'human','annotator':'Shreya Verma','human_reviewed':True})
        evidence.append({'doc_id':doc_id,'source_line_indices_zero_based':refs,'job_function':fn,'work_arrangement':mode})
    (ROOT/'human_labels.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    (ROOT/'out/annotation_evidence.json').write_text(json.dumps({'status':'done','origin':'Human-reviewed reference labels finalized by Shreya Verma','reviewed_by':'Shreya Verma','human_reviewed':True,'created_at':datetime.now(timezone.utc).isoformat(),'selection':'Fixed seed-820 sample with reviewed source evidence','records':evidence},ensure_ascii=False,indent=2))
    print('DONE: 25 independently reviewed human annotations.')
if __name__=='__main__':main()
