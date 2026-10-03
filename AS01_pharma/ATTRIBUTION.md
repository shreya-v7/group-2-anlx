# Attribution

**Author:** 95-820 Course Staff
**Course:** 95-820, Assignment 1 — Corpus Construction and Characterization
**Date:** September 2026

This folder is code only — no corpus data is included.

## What is here

All of it is MIT licensed (see `LICENSE`) — use it for anything.

| File | What it does |
|---|---|
| `schema.py` | Corpus record contract, extraction schema from `taxonomy.json`, model-output validation. |
| `check_corpus.py` | Validates `corpus.jsonl` and `sources.csv` against the brief. |
| `corpus_stats.py` | Corpus statistics and k-means clustering. |
| `local_model.py` | Loads Phi-4-mini-instruct from local weights and runs it in-process, greedy, JSON validated. |
| `llm_utils.py` | Hosted OpenAI-compatible client for the optional SLM-vs-LLM bonus. |
| `example_extraction.py` | Example of turning raw text and CSV into corpus records. |
| `extraction.py` | The graded run: corpus → Phi → human vs. Phi → recovery prompt → statistics and clusters → optional hosted LLM. |
| `mask_sensitive.py` | Masks `<PII>` / `<PHI>` / `<FIN>` / `<CONF>` spans before data enters the corpus. |
| `make_human_labels.py` | Draws the evaluation sample and writes the `human_labels.jsonl` template. |
| `taxonomy.json` | The extraction task: group topic, subtopic and fields. |

## Third-party components

**Model.** Phi-4-mini-instruct, Microsoft Corporation, MIT License. Weights are
run locally from disk; nothing here redistributes them.

**Libraries.** PyTorch (BSD-3), Transformers (Apache-2.0), scikit-learn
(BSD-3), matplotlib (PSF-based), NumPy (BSD-3), Requests (Apache-2.0).

## Tool assistance

These scripts were drafted with Claude Code (Anthropic) and then reviewed,
tested and edited by me.


## Pharma corpus adaptation (September 20, 2026)

The attribution above describes the original starter code. This derivative package adds a third-party job corpus; the statement that the folder is code-only no longer describes this package. The MIT license covers code, not employer job-advertisement content. Original versions of changed starter files are preserved in starter_original/.

The pharma corpus adaptation adds collection tooling, provenance records, a field taxonomy, an annotation interface, local model evaluation, recovery analysis, clustering, tests, and the final memo. Shreya independently reviewed and finalized all 25 human reference annotations. Local Phi baseline and recovery outputs are retained in `out/`. The optional hosted-LLM bonus was completed (Gemini Flash vs. 4-bit Phi on the same 25 records); see `out/slm_vs_llm.json` and `out/llm_evaluation.json`. The optional MLX runtime uses a public 4-bit conversion of Microsoft Phi-4-mini-instruct; model provenance and versions are in `out/model_environment.json`.
