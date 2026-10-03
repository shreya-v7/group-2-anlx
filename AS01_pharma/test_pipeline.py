"""Focused checks for corpus integrity and evaluation validity; no model needed."""
import csv, hashlib, json, tempfile, unittest
from pathlib import Path
import extraction as e
from schema import validate_against_schema

ROOT=Path(__file__).resolve().parent

class PipelineTests(unittest.TestCase):
    def test_full_text_and_table_reach_prompt(self):
        record={'doc_id':'test','raw_text':'a'*14000+'FINAL_QUALIFICATION',
                'table_json':{'value':'b'*1200+'FINAL_TABLE'}}
        prompt=e.build_prompt(record)
        self.assertIn('FINAL_QUALIFICATION',prompt);self.assertIn('FINAL_TABLE',prompt)

    def test_incomplete_annotations_block_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            pending=Path(tmp)/'labels.jsonl'
            pending.write_text(json.dumps({'doc_id':'unfinished','human_label':{},'annotation_status':'pending'})+'\n')
            with self.assertRaisesRegex(SystemExit,'incomplete'):
                e.load_human_labels(str(pending))

    def test_human_labels_are_complete(self):
        self.assertEqual(len(e.load_human_labels(str(ROOT/'human_labels.jsonl'))),25)

    def test_reference_quotes_have_source_evidence(self):
        corpus={r['doc_id']:r for r in map(json.loads,(ROOT/'corpus.jsonl').read_text().splitlines())}
        labels=e.load_human_labels(str(ROOT/'human_labels.jsonl'))
        for doc_id,label in labels.items():
            for name,spec in e.FIELDS.items():
                if spec['kind']=='list':
                    for quote in label[name]:self.assertIn(quote,corpus[doc_id]['raw_text'])

    def test_missing_prediction_cannot_score_as_empty_list(self):
        result=e.compare_one(None,{'items':[]},{'items':{'kind':'list'}})
        self.assertFalse(result['items']['ok'])

    def test_invalid_enum_is_rejected(self):
        self.assertIsNotNone(validate_against_schema({'mode':'office'},{'type':'object','properties':{'mode':{'type':'string','enum':['onsite']}},'required':['mode']}))

    def test_prediction_ids_are_checked(self):
        with self.assertRaisesRegex(ValueError,'IDs'):
            e.run_phi_evaluation([{'doc_id':'a'}],{'a':{}},rows=[{'doc_id':'b'}])

    def test_schema_violation_cannot_score_as_fully_correct(self):
        rows=[{'doc_id':'a','extracted':{'x':'yes'},'violation':'extra key','latency_s':0}]
        result=e.run_phi_evaluation([{'doc_id':'a','modality':'text'}],{'a':{'x':'yes'}},rows=rows,fields={'x':{'kind':'label'}})
        self.assertEqual(result['all_fields_correct_rate'],0)

    def test_sources_and_sample_match_corpus(self):
        records=list(map(json.loads,(ROOT/'corpus.jsonl').read_text().splitlines()))
        with (ROOT/'sources.csv').open() as f: sources=list(csv.DictReader(f))
        self.assertEqual(len(records),200)
        self.assertEqual({r['source_url'] for r in records},{s['source_url'] for s in sources})
        self.assertEqual(len({r['doc_id'] for r in records}),200)
        sample=json.loads((ROOT/'evaluation_sample.json').read_text())
        self.assertEqual(sample['corpus_sha256'],hashlib.sha256((ROOT/'corpus.jsonl').read_bytes()).hexdigest())
        self.assertEqual(len(sample['doc_ids']),25)
        self.assertTrue(set(sample['doc_ids'])<={r['doc_id'] for r in records})

if __name__=='__main__':unittest.main()
