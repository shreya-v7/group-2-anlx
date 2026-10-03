# Assignment 1 submission checklist

| Assignment requirement | Current status |
|---|---|
| Group topic and individual subtopic | Documented: job market / pharmaceutical and therapeutic biotech employers |
| Approximately 150-300 natural-language records | Complete: 200 records |
| At least 20% table/semi-structured records | Complete: 200 mixed records (100%) |
| Sensitive-information masking (2.1) | Complete: full Phi-4-mini-instruct pass run on all 200 records; reviewed and rejected as unreliable (mistagged required schema fields in nearly every record); corpus unchanged; evidence in `out/masking_report.json` and `corpus.masked.jsonl` |
| One document need not equal one record (2.2) | Addressed: 1 posting = 1 record is a deliberate choice, documented in README |
| Source URLs, retrieval timestamps and rights notes | Complete: corpus.jsonl and sources.csv |
| Random sample of 20-30 records | Complete: 25 records, seed 820 |
| Manually human-created reference labels | Complete: 25/25 independently reviewed and finalized by Shreya |
| Phi evaluation and error analysis | Complete: actual 25-record local run, evaluated against human references |
| Evidence-driven recovery experiment | Complete: revised prompt tested on 19 selected baseline failures |
| Cluster interpretation and comparison with model errors | Complete: six clusters, measured performance and sample-size limitations |
| Short memo with 4-5 limitation lines | Complete: three-page memo, five numbered limitations, includes masking-run summary |
| Reproducible extraction.py and README | Complete; supporting scripts and measured outputs included |
| Optional larger-LLM bonus | Complete: Gemini Flash vs 4-bit Phi on the same 25 records |
| Canvas upload | Not performed |

## What is verified

The standard corpus validator passes with a documented six-pair near-duplicate warning. Nine pipeline checks pass. The final accuracy audit recalculates all saved baseline/recovery metrics, reproduces the random sample, and verifies unchanged input hashes. The PDF was rendered and visually inspected. The archive contains all six required filenames and verified checksums.

Baseline: 108/225 matching fields (48%), 23/25 schema-valid responses (92%), and 0/25 full-record exact matches. Recovery fixes 6/19 targeted work-arrangement errors (31.6%), while producing two schema violations and three regressions in other field decisions. Low model accuracy is an experimental result, not a requirement to fabricate better results.

The Phi masking pass (full precision, MPS, ~6.4 hours over 200 records) accepted 2,140 spans as verbatim matches, but mistagged required schema fields (`job_title`, `employer`, `location`, `requisition_id`) in nearly every record, and none of the 361 accepted `<PII>` tags fell in `raw_text`. This was treated as an experimental result and documented, not smoothed over; `corpus.jsonl` was left unmasked because no genuine PII/PHI/FIN/CONF was found by this run, a regex screen, or a manual pattern check.

The run uses the documented 4-bit MLX conversion of Phi-4-mini-instruct for the baseline/recovery, and the full-precision Transformers backend for the masking pass; both are disclosed where used. Source-derived reference quotes were checked verbatim; interpretation of job function and ambiguous qualifications can still require human adjudication.

## Remaining required work

No required technical component remains. The optional larger-LLM comparison is complete (official saved run: Gemini Flash 12/25 vs 4-bit Phi 0/25). Upload the verified ZIP to Canvas before Tuesday 11:59pm. Do not include `.env` or API keys.

## What to submit

Upload one ZIP with the assignment1 folder to Canvas. The current package is `assignment1_completed.zip` and contains, at minimum:

- corpus.jsonl
- sources.csv
- extraction.py
- human_labels.jsonl
- memo.pdf
- README.md

Keep the included supporting modules, taxonomy, evidence, and measured outputs. Do not submit model weights, virtual environments, or `.env`. Do not publish the corpus to Hugging Face.
