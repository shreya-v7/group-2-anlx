import json
import unittest

from fence_parser import parse_facts_tolerant, strip_whole_fence

GOOD = {'doc_id': 'x', 'job_title': 'Scientist', 'employer': 'Acme', 'location': 'Boston',
        'source_url': 'https://example.com/1', 'work_arrangement': 'onsite', 'required_qualifications': ['PhD']}


class FenceParserTest(unittest.TestCase):
    def test_plain_json_unchanged(self):
        facts, error, stripped = parse_facts_tolerant(json.dumps(GOOD))
        self.assertIsNotNone(facts); self.assertFalse(stripped)

    def test_whole_json_fence_accepted(self):
        facts, error, stripped = parse_facts_tolerant('```json\n' + json.dumps(GOOD, indent=2) + '\n```')
        self.assertIsNotNone(facts); self.assertTrue(stripped)

    def test_unlabelled_fence_with_whitespace_accepted(self):
        facts, _, stripped = parse_facts_tolerant('\n```\n' + json.dumps(GOOD) + '\n```\n')
        self.assertIsNotNone(facts); self.assertTrue(stripped)

    def test_unclosed_fence_rejected(self):
        facts, error, stripped = parse_facts_tolerant('```json\n' + json.dumps(GOOD)[:-5])
        self.assertIsNone(facts); self.assertFalse(stripped)

    def test_prose_around_fence_rejected(self):
        facts, _, stripped = parse_facts_tolerant('Here you go:\n```json\n' + json.dumps(GOOD) + '\n```')
        self.assertIsNone(facts); self.assertFalse(stripped)

    def test_two_fences_rejected(self):
        text = '```json\n{}\n```\n```json\n' + json.dumps(GOOD) + '\n```'
        self.assertFalse(strip_whole_fence(text)[1])

    def test_schema_still_enforced_inside_fence(self):
        bad = dict(GOOD, work_arrangement='full-time onsite')
        facts, error, stripped = parse_facts_tolerant('```json\n' + json.dumps(bad) + '\n```')
        self.assertIsNone(facts); self.assertTrue(stripped); self.assertIn('work_arrangement', error)


if __name__ == '__main__':
    unittest.main()
