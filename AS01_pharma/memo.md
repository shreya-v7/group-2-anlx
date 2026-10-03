# Pharma Job Market: Corpus and Characterization
**Shreya Verma | Assignment 1 | September 21, 2026**

**Technical analysis complete.** The corpus, 25 reference-label records, local Phi baseline, prompt-recovery experiment and cluster analysis are complete. Annotation methodology and its limitation are stated below.

## Corpus construction
Our group topic is Job Market by Industry: Chih-yu covers Finance, Shreya covers Pharma, and William covers Tech. This corpus contains 200 English-language public job postings from eight pharmaceutical/therapeutic biotech employers, collected September 20, 2026. It supports an API for retrieving these advertisements and explicit requirements, not estimates of actual hiring outcomes or industry totals. All 200 combine natural-language descriptions with original structured API fields, exceeding the 20% semi-structured requirement.

From 452 postings, filtering removed nine repeated internal job IDs, seven exact description duplicates and five general-interest posts. Seeded within-employer sampling and round-robin allocation selected 200 of 431 eligible postings. Six near-duplicate pairs were retained because their distinct roles, levels, shifts or markets preserve useful information. The standard validator passes with that warning. Median description length is 809.5 words. URLs, retrieval times, source-specific rights notes and raw snapshots are retained. Public access is not an open redistribution license; the corpus stays private for coursework.

## Structured extraction evaluation
The fixed random sample uses seed 820 and 25 records. Shreya independently reviewed and finalized the human labels before the final evaluation package was prepared. Nine fields cover title, employer, location, function, work arrangement and four qualification lists. Annotations preserve source sentences, degree/experience alternatives and mixed required/preferred clauses. The local run used Phi-4-mini-instruct in 4-bit MLX format, greedy decoding, a 4,096-output-token cap and complete untruncated input. Results apply to this quantized runtime.

Full-record exact agreement was **0/25 (0.0%)**. Across 225 field decisions, agreement was **48.0%**. Schema-valid output was **92.0%**. Mean baseline latency was 31.09 seconds per record.

| Extraction field | Exact agreement |
|---|---:|
| job title | 92.0% |
| employer | 72.0% |
| location | 92.0% |
| job function | 32.0% |
| work arrangement | 16.0% |
| required qualifications | 20.0% |
| preferred qualifications | 32.0% |
| education requirements | 32.0% |
| experience requirements | 44.0% |

Scalars use case-insensitive, whitespace-trimmed equality. List agreement requires exact equality of normalized sets; token-equivalent paraphrases or added punctuation still fail. Consequently, a low full-record score is stricter than semantic correctness. Field scores can credit correct fields despite another schema error; full-record matches cannot. Raw outputs and list precision/recall/F1 are saved for audit.

---PAGE---

## Error reading and recovery strategy
Phi inferred onsite work from city-only locations, changed a source-table employer to a prose-derived name, and split education/experience alternatives into fragments. Two early Arrowhead responses repeated qualifications until hitting the output cap and failed schema validation. Recovery targets work arrangement; valid JSON alone does not establish correct extraction.

All **19 schema-valid baseline records with a work-arrangement error** were selected before the rerun. The original prompt requested explicit role evidence. The revision adds a decision procedure: honor role-specific remote/hybrid tags; reject generic onsite salary boilerplate, addresses, duties and travel as work-mode evidence; return not_stated without explicit evidence. Complete prompts and selected IDs are in out/prompt_history.md and taxonomy.json.

On the same 19 affected records, work-arrangement agreement changed from **0.0% to 31.6%**, fixing **6/19** targeted errors. Revised full-record agreement was 0/19; revised schema validity was 89.5%. Across all fields, 3 previously correct decisions regressed. Targeted gains therefore require checking collateral changes. This diagnostic reuses known failures; it is not held-out evaluation. The prompt improved the targeted field on 6 records but left 13 errors; its effect is limited to this selected subset.

Senior Scientist II, Discovery Analytical Chemistry: reference not_stated, baseline onsite, revised onsite. Coordinator II, Translational Biomarkers: reference not_stated, baseline onsite, revised not_stated.

## Cluster reading
The supplied TF-IDF/SVD/k-means workflow used k=6, seed 0 and complete descriptions; silhouette was 0.1709. Five clusters isolate individual employer templates; the remaining cluster combines smaller employers. Shared company/science boilerplate dominates occupational differences. For example, Beam groups student research with finance/compliance leadership, while Arrowhead includes analytical, ERP, payroll and supply-chain roles. These names describe corpus language, not validated occupations.

| Cluster name (corpus count) | Full match / n | Field agreement |
|---|---:|---:|
| 0: Kymera: targeted protein degradation (29) | 0/3 | 44.4% |
| 1: Relay: discovery and development (11) | 0/2 | 50.0% |
| 2: Beam: gene editing and operations (36) | 0/6 | 48.1% |
| 3: Smaller employers: mixed biotech (19) | 0/1 | 66.7% |
| 4: Revolution: oncology and corporate (52) | 0/6 | 63.0% |
| 5: Arrowhead: RNAi and operations (53) | 0/7 | 33.3% |

Among clusters with at least two evaluation records, Revolution: oncology and corporate has the highest field agreement (63.0%); Arrowhead: RNAi and operations has the lowest (33.3%), with frequent employer, job function disagreements. The table uses baseline predictions only. Samples are small and unequal; employer and document style are confounded. A one-record cluster supports only a case observation, not a reliable difficulty ranking.

## Optional SLM vs. LLM
The same 25 records, schema, v1 prompt and human labels were run on hosted Gemini. Gemini 3.1 Pro was quota-blocked, so the comparison uses Gemini Flash: gemini-flash-latest on 8 records and gemini-3.5-flash on 17. This is still a much larger hosted model than 4-bit Phi-4-mini.

Full-record exact agreement rose from **0.0%** (Phi) to **48.0%** (Gemini). Mean field agreement rose from **48.0%** to **92.4%**. Schema-valid output rose from **92.0%** to **100.0%**. Mean latency was 31.09s for Phi and 25.50s for Gemini.

| Field | Phi | Gemini |
|---|---:|---:|
| job title | 92.0% | 100.0% |
| employer | 72.0% | 100.0% |
| location | 92.0% | 100.0% |
| job function | 32.0% | 92.0% |
| work arrangement | 16.0% | 100.0% |
| required qualifications | 20.0% | 72.0% |
| preferred qualifications | 32.0% | 72.0% |
| education requirements | 32.0% | 96.0% |
| experience requirements | 44.0% | 100.0% |

The largest gains were work arrangement (16.0% to 100.0%) and job function (32.0% to 92.0%). Gemini matched 12/25 records exactly; Phi matched 0/25. Remaining Gemini errors are mostly exact-match list disagreements on required/preferred qualifications, not schema failures. Raw predictions are in out/llm_predictions.jsonl; the comparison table is in out/slm_vs_llm.json.

## Sensitive-information screening

A full Phi-4-mini-instruct masking pass (`mask_corpus_content.py`, reusing `mask_sensitive.py`'s detection schema and verbatim-span acceptance) ran over every `raw_text` and `table_json` value in all 200 records (full precision, MPS, ~6.4 hours). It accepted 2,140 spans as verbatim matches and tagged them `<CONF>`, `<PHI>`, `<FIN>`, or `<PII>` (`out/masking_report.json`).

On review, this output was rejected and not merged into `corpus.jsonl`. The model mistagged required schema fields in nearly every record: `job_title`, `employer`, `location`, and even the sequential `requisition_id` were routinely replaced with placeholders (e.g. "Arrowhead Pharmaceuticals" -> `<CONF>`, `"630"` -> `<FIN>`), and none of the 361 accepted `<PII>` tags fell in `raw_text`; all landed on non-personal `table_json` values such as office names and timestamps. This mirrors the same structured-output unreliability already measured above. Combined with a regex screen and a targeted manual check (personal-name, phone, SSN, and non-role-email patterns across all 200 records; zero hits), no evidence of genuine sensitive content was found, so `corpus.jsonl` is unchanged. `corpus.masked.jsonl` and `out/masking_report.json` are retained as run evidence.

## Corpus limitations - five lines
1. Eight English-language employers on one ATS cannot represent the global pharmaceutical labor market.
2. One collection snapshot cannot establish hiring trends, filled vacancies or current posting availability.
3. Employer boilerplate and related postings create correlated records and dominate similarity clusters.
4. Unstated work modes, ambiguous functions and alternative qualifications limit exact-match interpretation.
5. A 25-record evaluation and quantized inference limit precision and generalization of reliability estimates.

Evidence: out/evaluation.json, out/recovery.json, out/cluster_performance.json and sources.csv. Model details: out/model_environment.json. Starter: 95-820 course staff (MIT). Optional hosted-LLM bonus: completed (Gemini Flash vs 4-bit Phi).
