# Assignment 1 - Pharma Job Market

**Student:** Shreya Verma  
**Group topic:** Job Market by Industry  
**Group split:** Chih-yu - Finance; Shreya - Pharma; William - Tech

**Human labeling: DONE (25/25).** Shreya independently reviewed and finalized all nine fields for every sampled posting. The evaluation reports Human vs. Phi agreement on this fixed sample.

The final measured technical status is in `TASK_STATUS.json`. `out/final_accuracy_audit.json` confirms independent recalculation of saved metrics, unchanged input hashes and the fixed random sample. Recheck with `python audit_results.py`. `memo.pdf` summarizes actual baseline, recovery and cluster results; complete predictions and metrics are in `out/`.

## Corpus and sources

The corpus contains 200 real English-language job advertisements from eight drug-development and therapeutic-biotechnology employers, retrieved September 20, 2026. Every posting is a `mixed` record with full natural-language description and original structured Greenhouse API fields. Thus 100% are semi-structured, exceeding the 20% minimum. General-interest posts, repeated employer-internal job IDs and exact normalized description duplicates were removed.

Each job posting is kept as exactly one record. Unlike the assignment's résumé/budget examples, a posting describes a single indivisible role; splitting it into sub-records (e.g. one row per requirement) would fragment rather than preserve information, so 1 posting = 1 record was the deliberate choice here, not an oversight of Section 2.2.

| Employer | Records |
|---|---:|
| Arrowhead Pharmaceuticals | 53 |
| Revolution Medicines | 52 |
| Beam Therapeutics | 36 |
| Kymera Therapeutics | 29 |
| Relay Therapeutics | 11 |
| Recursion | 10 |
| Tessera Therapeutics | 6 |
| Ultragenyx Pharmaceutical | 3 |
| Total | 200 |

The raw snapshots contain 452 postings. Filtering removes 21, leaving 431 eligible records. Deterministic within-employer shuffling and round-robin selection yield 200. This favors employer variety and is not a market-representative sample. All employer job functions and advertised locations are included, including corporate, IT, student and laboratory positions. Industry is defined by employer. Pharmacies, hospitals, staffing firms and medical-device-only employers are outside the source set.

HTML is decoded into readable lines without paraphrasing descriptions. Original API values are preserved in `table_json`; missing values remain null. Posting publication/update dates remain distinct from collection time. Each posting URL has an entry in `sources.csv`, with retrieval time, record count and a rights note. `raw/` preserves snapshots and `out/collection_report.json` their hashes. The corpus supports retrieval and extraction of advertised requirements for these jobs and can later combine with Finance and Tech through the shared record schema.

## Reference annotations and evaluation

The 25-record simple random sample uses seed 820; `evaluation_sample.json` binds it to the corpus and field-schema hashes. The nine fields are title, employer, location, job function, work arrangement, required qualifications, preferred qualifications, education requirements and numeric-experience requirements.

`ANNOTATION_GUIDE.md` documents decisions, and `out/annotation_evidence.json` provides the source-line audit trail. All list entries occur verbatim in the source. The labels were independently reviewed and finalized by Shreya. `label_review.html` remains available for reviewing the complete source alongside every label.

Scalar fields use case-insensitive, trimmed exact match. Lists use equality of sets after lowercase/outer-whitespace normalization; list precision, recall and F1 are also reported. This penalizes punctuation changes and paraphrases even when meaning is similar. Full-record agreement requires all nine fields and valid output schema. Missing/unparseable outputs fail every field, including empty-list references. Field scores can still credit correct fields in parsed outputs that fail another schema rule (for example, extra keys); such outputs never count as full-record matches.

Phi runs locally using the **4-bit MLX conversion of Phi-4-mini-instruct**, with greedy decoding, complete source text/table and a 4,096-token output cap. Quantization can affect predictions; these results must not be described as full-precision lab-model results. Exact model revision, file hashes, hardware and package versions are in `out/model_environment.json`. The Transformers starter backend is also retained.

Recovery is designed only after examining the baseline. It targets work-arrangement errors in schema-valid baseline responses. All such records are selected; original/revised prompts, the rule and IDs are in `out/prompt_history.md` and `taxonomy.json`. `out/recovery.json` reports before/after on the same subset, full-record matches, raw predictions and per-field fixes/regressions. This is diagnostic reuse of known errors, not a held-out benchmark. No reference labels or extraction fields change between runs.

## Reproduce the actual run (Apple Silicon)

From this folder, with Python 3.11:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-mlx.txt
hf download mlx-community/Phi-4-mini-instruct-4bit \
  --revision ac1c269cb4222a4e136a3d09edad301056c1f36a \
  --local-dir ../phi-4-mini-mlx
python check_corpus.py corpus.jsonl
python -m unittest -v test_pipeline
HF_HUB_OFFLINE=1 python extraction.py --stage baseline --backend mlx \
  --model-path ../phi-4-mini-mlx
HF_HUB_OFFLINE=1 python extraction.py --stage recovery --backend mlx \
  --model-path ../phi-4-mini-mlx
python finalize_results.py
python -m pip install reportlab
python render_memo.py
```

Optional hosted-LLM bonus (requires `GEMINI_API_KEY` in a local `.env`, never submit the key). Official saved results used Gemini Flash after Pro hit quota. Free-tier Flash is 5 RPM / 20 RPD on this project, so a 25-record rerun must use a 500-RPD Lite model or wait for midnight Pacific. Do not rerun against `out/` unless you intend to replace the saved comparison.

```bash
python extraction.py --stage llm --llm gemini --llm-model gemini-3.5-flash-lite --out out_gemini_rerun
```

The already-downloaded model is retained locally at `../.models/phi-4-mini-instruct-4bit`; use that as `--model-path` to avoid downloading again. The supplied `llmbox-main.zip` contained source and LoRA adapters, while the Phi browser download remained an incomplete `.crdownload` at the September 21 inspection; see `out/input_model_archive_check.json`.

GPU access is required by MLX. Model weights and virtual environments are not in the submission archive. The baseline and recovery use unchanged corpus, labels, model, decoding cap, schema and baseline prompt; mismatch checks prevent stale comparisons. Before rerunning, preserve `out/` if you need the original results. Different runtime versions or hardware may alter outputs.

To use the original full-precision Transformers backend, install `requirements-phi.txt` and run with `--backend transformers --model-path /path/to/lab/phi-4-mini-instruct`. Those would be new results, not the saved MLX run. To rebuild the unchanged corpus from saved snapshots, run `python collect_corpus.py`; `--refresh` downloads new data and invalidates existing annotations and manifests.

## Cluster interpretation and limitations

The supplied TF-IDF/SVD and k-means code clusters full descriptions with k=6 and seed 0. Employer introductions strongly shape the clusters. `CLUSTER_READING.md` names clusters and links representative postings; `out/cluster_performance.json` connects baseline agreement and field errors with each cluster. A cluster with only one sampled record supports no reliable general performance claim.

The standard corpus checker passes with a near-duplicate warning: six pairs (12 records) have distinct IDs and meaningful role/level/shift/market differences. `out/near_duplicate_review.json` documents retention. `--strict` intentionally fails because warnings become failures. The nine pipeline checks cover corpus/source consistency, schema validation, missing predictions, full-text preservation, annotation provenance and source-quote evidence.

Limits include eight employers on one ATS; English-only convenience sampling; a single time point; repeated boilerplate; unrepresented employers and roles; ambiguous or unstated qualifications; small cluster evaluation samples; exact-match sensitivity; and quantized inference. The corpus cannot establish industry hiring totals, salary distributions, actual hiring outcomes, applicant quality or longitudinal trends. The optional larger-LLM comparison was run on the same 25 records using hosted Gemini Flash (`gemini-flash-latest` on 8 records and `gemini-3.5-flash` on 17) after Gemini Pro hit quota. Results are in `out/slm_vs_llm.json` and `out/llm_evaluation.json`.

## Rights, privacy and attribution

Sources use the public [Greenhouse Job Board API](https://docs.greenhouse.io/job-board.html). Posting-level rights notes link source pages and [platform legal notices](https://www.greenhouse.com/legal). Public access is not an open content license; no open redistribution permission was identified. Keep this corpus private for coursework and do not publish it on Hugging Face or another dataset platform.

No applicant/patient records were collected. The existing regex screen found shared corporate contacts, not personal contact information; it is not comprehensive entity detection. Public disease/drug context in employer descriptions is not patient-specific health information. `out/privacy_screen.json` records the limited regex screen.

A full Phi-4-mini-instruct masking pass (`mask_corpus_content.py`, reusing `mask_sensitive.py`'s detection schema and verbatim-span acceptance) was also run over every `raw_text` and `table_json` field in all 200 records (full precision, MPS, ~6.4 hours; `out/masking_run.log`). It accepted 2,140 spans as verbatim matches and flagged them `<CONF>`, `<PHI>`, `<FIN>`, or `<PII>` (`out/masking_report.json`). On review, this output was rejected and NOT merged into `corpus.jsonl`: the model mistagged required schema fields in essentially every record -- `table_json.job_title`, `.employer`, `.location`, and even the sequential `.requisition_id` were routinely replaced with placeholder tags (e.g. "Scientist III, Antibody Engineering" -> `<PHI>`, "Arrowhead Pharmaceuticals" -> `<CONF>`, `"630"` -> `<FIN>`), and zero of the 361 accepted `<PII>` tags fell in `raw_text` -- all landed on non-personal `table_json` values such as office names and `first_published`/`updated_at` timestamps. This is consistent with the same 4-bit/full-precision Phi-4-mini structured-output unreliability already documented in the Human vs. Phi evaluation (Section: Structured extraction evaluation), now observed on a different task. Combined with the regex screen and a targeted manual check (personal-name, phone, SSN, and non-role-email patterns across all 200 `raw_text` records; zero hits), no evidence of genuine PII, PHI, FIN, or CONF content was found in this corpus, so `corpus.jsonl` is unchanged. `corpus.masked.jsonl` and `out/masking_report.json` are retained as the required run evidence.

Starter code: 95-820 course staff, MIT. The code license does not cover employer advertisements. See `ATTRIBUTION.md`. Model source: [Phi-4-mini-instruct MLX conversion](https://huggingface.co/mlx-community/Phi-4-mini-instruct-4bit); runtime: [MLX LM](https://github.com/ml-explore/mlx-lm).

## Submission contents

The archive contains one `assignment1/` folder, including all six required files: `corpus.jsonl`, `sources.csv`, `extraction.py`, `human_labels.jsonl`, `memo.pdf`, and `README.md`. Supporting code, taxonomy, raw snapshots, measured outputs and annotations are included for reproducibility. `SHA256SUMS.json` fingerprints the delivered files. No public upload or course submission was performed.
