"""Rebuild a deterministic corpus from preserved public Greenhouse snapshots.

Default: offline rebuild. --refresh explicitly replaces snapshots with live data.
Only published GET endpoints are used; no applications or candidate data.
"""
from __future__ import annotations
import argparse, collections, csv, datetime, hashlib, html, json, random, re
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
BOARDS = {
    'arrowheadpharmacareers': 'Arrowhead Pharmaceuticals',
    'beamtherapeutics': 'Beam Therapeutics',
    'kymeratherapeutics': 'Kymera Therapeutics',
    'recursionpharmaceuticals': 'Recursion',
    'relaytherapeutics': 'Relay Therapeutics',
    'revolutionmedicines': 'Revolution Medicines',
    'tesseratherapeutics': 'Tessera Therapeutics',
    'ultragenyxpharmaceutical': 'Ultragenyx Pharmaceutical',
}
API_DOCS = 'https://docs.greenhouse.io/job-board.html'
LEGAL = 'https://www.greenhouse.com/legal'

class PlainText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.skip = 0
    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style'): self.skip += 1
        if tag in ('p', 'div', 'br', 'li', 'h1', 'h2', 'h3', 'h4', 'tr'): self.parts.append('\n')
        if tag in ('td', 'th'): self.parts.append(' | ')
    def handle_endtag(self, tag):
        if tag in ('script', 'style'): self.skip = max(0, self.skip - 1)
        if tag in ('p', 'div', 'li', 'h1', 'h2', 'h3', 'h4', 'tr'): self.parts.append('\n')
    def handle_data(self, data):
        if not self.skip: self.parts.append(data)

def plain_text(content):
    parser = PlainText()
    parser.feed(html.unescape(content))
    return '\n'.join(line for line in (re.sub(r'\s+', ' ', x).strip() for x in ''.join(parser.parts).splitlines()) if line)

def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def collect(target=200, seed=820, refresh=False):
    raw = ROOT / 'raw'; raw.mkdir(exist_ok=True)
    queues, rejected, source_info = {}, [], {}
    for board, company in BOARDS.items():
        url = f'https://boards-api.greenhouse.io/v1/boards/{board}/jobs?content=true'
        path = raw / f'{board}.json'
        provenance_path = raw / f'{board}.provenance.json'
        if refresh:
            request = Request(url, headers={'User-Agent': 'AcademicCorpus/1.0 (public job postings)'})
            with urlopen(request, timeout=40) as response:
                path.write_bytes(response.read())
                provenance_path.write_text(json.dumps({'url': url, 'retrieved_at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'http_status': response.status}, indent=2))
        payload = json.loads(path.read_text())
        provenance = json.loads(provenance_path.read_text())
        source_info[board] = {**provenance, 'company': company, 'raw_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'available_posts': len(payload['jobs'])}
        seen_internal, seen_text = set(), set()
        rows = []
        for job in sorted(payload['jobs'], key=lambda j: j['id']):
            description = plain_text(job.get('content', ''))
            internal = job.get('internal_job_id')
            content_hash = digest(re.sub(r'\s+', ' ', description.lower()))
            reason = None
            if not internal: reason = 'prospect/general-interest post (no internal job ID)'
            elif job.get('language') not in ('en', None): reason = 'non-English language marker'
            elif len(description) < 200: reason = 'insufficient description'
            elif internal in seen_internal: reason = 'duplicate internal job ID within employer'
            elif content_hash in seen_text: reason = 'exact normalized description duplicate within employer'
            if reason:
                rejected.append({'board': board, 'posting_id': job['id'], 'reason': reason})
                continue
            seen_internal.add(internal); seen_text.add(content_hash)
            rows.append((job, description, content_hash))
        random.Random(f'{seed}:{board}').shuffle(rows)
        queues[board] = rows
        source_info[board]['eligible_unique_posts'] = len(rows)
    # Equal employer allocation until smaller boards are exhausted. Not a market sample.
    chosen = []
    while len(chosen) < target and any(queues.values()):
        for board in sorted(queues):
            if queues[board] and len(chosen) < target:
                chosen.append((board, *queues[board].pop()))
    if len(chosen) < target: raise SystemExit(f'Only {len(chosen)} eligible postings; target={target}. No padding performed.')
    records, sources = [], []
    for board, job, description, content_hash in chosen:
        company = BOARDS[board]; info = source_info[board]
        # These values are copied from source API fields, not inferred from prose.
        table = {'job_title': job['title'], 'employer': job.get('company_name') or company,
                 'location': job.get('location', {}).get('name'), 'posting_id': job['id'],
                 'internal_job_id': job['internal_job_id'], 'requisition_id': job.get('requisition_id'),
                 'departments': [d['name'] for d in job.get('departments', [])],
                 'offices': [o['name'] for o in job.get('offices', [])],
                 'first_published': job.get('first_published'), 'updated_at': job.get('updated_at')}
        note = (f'{company} public job advertisement retrieved using the unauthenticated Greenhouse Job Board GET API. '
                f'Access documentation: {API_DOCS}; platform legal notices: {LEGAL}. '
                'No open redistribution license identified in the API response. Employer retains rights in posting text. '
                'Retained for this private coursework corpus; no public dataset publication or redistribution license claimed.')
        record = {'doc_id': f'pharma_{board}_{job["id"]}', 'source_url': job['absolute_url'],
                  'retrieved_at': info['retrieved_at'], 'modality': 'mixed',
                  'raw_text': description, 'table_json': table, 'license_note': note,
                  'metadata': {'group_topic': 'job_market', 'subtopic': 'pharma', 'student': 'Shreya',
                               'document_type': 'job_posting', 'employer': company, 'board_token': board,
                               'api_url': info['url'], 'source_posting_id': job['id'],
                               'source_internal_job_id': job['internal_job_id'], 'language': job.get('language'),
                               'content_sha256': content_hash, 'sampling_seed': seed,
                               'semi_structured_origin': 'Original Greenhouse API fields (not model-extracted)',
                               'collection_scope': 'English public postings, all source locations; drug-development employers including biotech'}}
        records.append(record)
        sources.append({'source_url': record['source_url'], 'source_name': company, 'retrieved_at': info['retrieved_at'],
                        'license_note': note, 'record_count': 1, 'api_url': info['url'], 'board_token': board})
    records.sort(key=lambda r: r['doc_id']); sources.sort(key=lambda r: r['source_url'])
    (ROOT / 'corpus.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in records), encoding='utf-8')
    with (ROOT / 'sources.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(sources[0])); writer.writeheader(); writer.writerows(sources)
    counts = collections.Counter(r['metadata']['board_token'] for r in records)
    for board, info in source_info.items(): info['selected_records'] = counts[board]
    report = {'target': target, 'n_records': len(records), 'seed': seed,
              'sampling': 'Deterministic randomized within-employer queues, round-robin employer allocation until target reached.',
              'sources': source_info, 'excluded': rejected,
              'not_selected_eligible': sum(len(q) for q in queues.values())}
    (ROOT / 'out' / 'collection_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'records': len(records), 'employer_counts': dict(counts), 'excluded': len(rejected)}, indent=2))

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target', type=int, default=200); p.add_argument('--seed', type=int, default=820)
    p.add_argument('--refresh', action='store_true')
    a=p.parse_args(); collect(a.target,a.seed,a.refresh)
