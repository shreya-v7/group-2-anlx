# Finance-Industry Job Postings — Assignment 1 Corpus

**Group topic:** job market, split by industry
**Individual subtopic:** finance industry
**Corpus size:** 200 records, drawn from 175 source documents across 27 employers and 3 data sources

---

## 1. What this corpus is

200 job postings published by **finance-industry employers**, in a common record
format that can be merged with the other industry subcorpora our group is
building (technology, healthcare, and so on).

The scope is set by **the employer, not the job function**, because the group
splits the job market by industry. A marketing role at a crypto exchange is in
scope; a payroll examiner at the city health department is not. Concretely, an
employer qualifies if it is either

* a private financial-services firm — proprietary trading, market making, hedge
  funds, asset and wealth management, brokerage, consumer and business banking,
  payments, lending, crypto finance, or financial infrastructure software; or
* a **public-sector financial institution** — a city agency whose mandate is
  financial: the Department of Finance, the Office of the Comptroller, the
  Office of Management & Budget, the Campaign Finance Board, the Tax Commission,
  and the municipal retirement and pension systems.

Every role at such an employer is included, so the corpus contains engineering,
legal, sales and communications postings alongside trading, accounting, risk and
budget roles. That is deliberate: a question like *"what does the finance
industry hire for, and how does that differ from technology?"* needs the whole
hiring footprint of the industry, not only the jobs with "analyst" in the title.

## 2. Sources

| Source | Access route | Records | Licence / terms |
|---|---|---:|---|
| 18 company job boards on **Greenhouse** | public, unauthenticated board API `boards-api.greenhouse.io/v1/boards/<token>/jobs?content=true` | 104 | Employer-published public postings. Text remains the employer's copyright; held in excerpt for non-commercial coursework, not redistributed. |
| 2 company job boards on **Lever** | public, unauthenticated postings API `api.lever.co/v0/postings/<company>?mode=json` | 28 | Same as above. |
| **NYC Open Data — "NYC Jobs"** (`kpav-sd4t`) | Socrata REST API | 68 | NYC Open Data Terms of Use: public data, free to use, share and adapt with attribution to the City of New York. |

No credentialed, paywalled or scraped-behind-login content is used. LinkedIn,
Indeed and Glassdoor were deliberately avoided: their terms of service prohibit
automated collection, so postings from those aggregators could not be documented
with a licence note we can stand behind. Per-source rows, with retrieval
timestamps and record counts, are in [`sources.csv`](sources.csv).

Employers represented: Anchorage Digital, Affirm, Alloy, AQR Capital Management,
Betterment, Brex, Campaign Finance Board, Carta, Chime, Coinbase, Department of
Finance (NYC), Gemini, IMC Trading, Jane Street, Jump Trading, Mercury, NYC
Employees' Retirement System, Office of Management & Budget, Office of the
Comptroller, Point72, Robinhood, SoFi, Stripe, Tax Commission, Upstart, Virtu
Financial, and others.

## 3. What is in a record

Every record carries the common top-level fields the assignment specifies
(`doc_id`, `source_url`, `retrieved_at`, `modality`, `raw_text`, `table_json`,
`license_note`, `metadata`). Subtopic-specific information lives in `metadata`:

```json
"metadata": {
  "group_topic": "job_market",
  "subtopic": "finance",
  "company": "Virtu Financial",
  "industry_segment": "market_making",
  "job_title": "ETF Trader",
  "location": "New York",
  "department": "Trading",
  "posting_date": "2026-08-14T09:12:00-04:00",
  "document_type": "job_posting_full",
  "section": "full_posting",
  "source_system": "greenhouse"
}
```

`group_topic`, `subtopic`, `company`, `job_title`, `location`, `posting_date`
and `document_type` are the keys agreed across the group, so the industry
subcorpora line up when they are merged.

### Record types

| `document_type` | Count | What it holds |
|---|---:|---|
| `job_posting_full` | 104 | A complete Greenhouse posting: responsibilities, requirements and, where published, the pay range. |
| `job_posting_description` | 68 | The prose description of an NYC or Lever posting. |
| `job_qualifications` | 15 | The minimum-qualification and preferred-skills columns of an NYC posting, which the dataset keeps separately from the description. |
| `job_requirements_list` | 13 | The structured bullet lists of a Lever posting, normalised into a table. |

### Modality and tables

148 of 200 records (74%) are `mixed`: prose plus a normalised table. The
remaining 52 are pure `text`. Table content is one of

* a **compensation row** — `{"component": "base_salary", "minimum": 125000,
  "maximum": 185000, "currency": "USD", "period": "year", "source": ...}`,
  parsed from the NYC salary-range columns or from the pay range that US pay
  transparency laws require in the posting text;
* **classification rows** from the NYC dataset — civil service title, title
  code, level, career level, full/part time, number of positions;
* **board metadata rows** from Greenhouse — for example time type; or
* **requirement rows** from Lever — `{"section": "Technical Skills", "item": ...}`.

Compensation rows and single-value classification rows do not share a key set,
so the table cells are about 40% sparse by the `corpus_stats.py` measure. That
is a property of the sources, not of the parsing: a salary band needs a minimum
and a maximum, a civil-service level needs neither.

## 4. Collection and preprocessing decisions

1. **One document can become several records.** An NYC posting yields a
   description record and, when the qualification columns are substantial, a
   separate qualifications record; a Lever posting yields its prose description
   and its bullet lists as a table. 200 records come from 175 source postings.
2. **HTML was flattened, not summarised.** Greenhouse double-escapes its posting
   HTML; it is unescaped twice, list items become `- ` bullets, tags are
   stripped, and whitespace is collapsed. No text is paraphrased or generated.
3. **Records are capped at 12,000 characters** so a single very long posting
   cannot dominate the corpus. Median length is 4,017 characters.
4. **Per-employer cap of 8 postings, sampled round-robin across departments.**
   Without the round robin, a board with 400 engineering openings would have
   contributed eight engineering roles and nothing else.
5. **Near-duplicates were removed at collection time.** The same requisition is
   routinely published once per office, and NYC civil-service qualification text
   is shared boilerplate across titles. Records at cosine similarity ≥ 0.80
   (TF-IDF over word 3–5-grams, the same measure `corpus_stats.py` uses) to an
   already-kept record were dropped — 49 of them in the final run — and the
   surviving pool was shuffled and trimmed to 200. The validator reports a 0%
   near-duplicate rate.
6. **Non-postings were excluded** — talent pools, "general application" entries,
   and the recruiting events (case competitions, interest forms) that some
   trading firms publish through the same board. They carry no requirements to
   extract.
7. **Sensitive information was masked with Phi-4-mini-instruct** before the data
   entered the final corpus. See below.

Collection is reproducible: `python build_corpus.py` rebuilds `corpus.jsonl` and
`sources.csv` with a fixed random seed (20260919). Live job boards change, so a
later run will not return exactly the same postings.

## 5. Sensitive information

Job postings are written for strangers, so they name roles rather than people.
A sweep of all 200 records for contact details found 21 email addresses, every
one an organisational mailbox (`accommodations@chime.com`, `security@carta.com`,
`agency-partnerships@janestreet.com`), no phone numbers, no government
identifiers and no personal names in a contact position. An earlier draft did
contain real PII — a named hiring contact's address and six direct phone lines
— but those came from non-finance city agencies and left the corpus when its
scope was narrowed to finance-industry employers.

Phi-4-mini was still run over the records carrying a contact cue, as the brief's
§2.1 asks. It proposed 81 spans, **none of them personally identifiable**: it
labelled job titles `<PHI>`, employer names and office locations `<PII>`, and
whole responsibility bullets `<CONF>` wherever a sentence happened to contain
the word "confidential". Ten of its 76 calls returned no parsable JSON at all.
That output was read and discarded, so the corpus ships **unmasked** — there was
nothing to mask, and applying Phi's spans would have deleted the data the
assignment is about. The numbers are reported in the memo as a second data point
on Phi's reliability.

## 6. Known limitations

* **Geographically narrow.** The private employers are US-headquartered and the
  entire public-sector half is one city government, so the corpus describes the
  US — largely New York — finance labour market. Questions about European or
  Asian finance hiring cannot be answered from it, even though a handful of
  postings name non-US offices.
* **A snapshot, not a time series.** Everything was retrieved on one day, so the
  corpus cannot support any question about trends, seasonality or how postings
  change as a role stays open.
* **Large employers are under-represented relative to their hiring.** The
  per-employer cap of 8 is a deliberate diversity choice, so record counts say
  nothing about how much each firm actually hires. Nor is the sample random:
  it is what these 20 company boards and 7 city agencies had open, and firms that use Workday or Taleo — most
  large banks and insurers — are absent entirely. There is no Goldman Sachs,
  JPMorgan, Citi or BlackRock posting here, and no insurance carrier at all, so
  `insurance` and `investment_banking` style roles are thin to non-existent.
* **Compensation coverage is uneven and legally driven.** Salary appears when a
  pay-transparency jurisdiction requires it or when the NYC dataset supplies the
  column; a posting without a salary band usually means "not legally required
  here", not "unpaid" — so any pay analysis over this corpus is biased toward
  New York, California, Colorado and Washington.
* **Postings describe what employers advertise, not what they hire.** Required
  years of experience, credentials and skill lists are negotiating positions
  written by recruiters. The corpus cannot say who was hired, at what pay, or
  whether the stated requirements were enforced.

## 7. Files

| File | What it is |
|---|---|
| `corpus.jsonl` | The 200-record corpus. |
| `sources.csv` | One row per source board, with licence note and record count. |
| `build_corpus.py` | Collection: three APIs → records, dedup, `sources.csv`. |
| `taxonomy.json` | The extraction task: fields, enums, prompt overrides. |
| `human_labels.jsonl` | The 28-record evaluation sample and its reference labels. Model-assisted: see below. |
| `annotation_guidelines.md` | The labelling standard the human labels follow. |
| `label_cli.py` | Prints a sampled record in full and prompts for each field, for hand labelling. |
| `extraction.py` | The graded run: Phi, human-vs-Phi, recovery prompt, clustering. |
| `out/` | Run artefacts: predictions, evaluation, one recovery file per prompt version, statistics, clusters, and the hosted-LLM comparison. |
| `memo.pdf` | The write-up. |

## 8. Reproducing

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch transformers accelerate safetensors scikit-learn matplotlib numpy requests

python build_corpus.py            # rebuild corpus.jsonl + sources.csv
python check_corpus.py corpus.jsonl --stats
python make_human_labels.py --n 25   # careful: --force overwrites existing labels
python label_cli.py               # label the sample, one field at a time
python extraction.py              # the graded run: Phi, every prompt version, stats, clusters
python extraction.py --llm groq   # optional bonus lane; needs GROQ_API_KEY in .env
```

Phi-4-mini-instruct weights are expected at `../models/phi-4-mini-instruct`, or
wherever `PHI_MODEL_PATH` points.
