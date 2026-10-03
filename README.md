# Group 2: job posting extraction across Pharma, Finance and Tech

CMU 95820 Applications of NLX and LLM (Fall 2026). Team: Shreya Verma (Pharma), Chih-yu Li (Finance), William Sun (Tech).

**Question:** can a small language model (local 4-bit Phi-4-mini-instruct) produce reliable, source-linked job-posting summaries for university career advisers?

**Answer:** the lookup tool worked on 50/50 requests, but only 17/50 answers got all five source facts right. Strict JSON validation rejected 0/50 → 43/50 after stripping Markdown fences (post-hoc). A keyword guardrail blocked 15/20 attacks and 1/10 benign requests. Recommendation: copy known facts directly; use the model only under adviser review.

## Layout

| Folder | Contents |
|---|---|
| `final_project/documents/` | **Team research report** (PDF/DOCX/MD), **36×24 poster**, presentation guide, poster figures |
| `final_project/build/` | Scripts and content that generate the report, poster and guide |
| `final_project/metrics/` | Corpus audit, teammate-reported results, reference-label index |
| `AS01_pharma/` | Shreya's AS01: 200-posting Pharma corpus, 25 human labels, Phi/Gemini outputs (`out/`), extraction, clustering, recovery, memo |
| `AS01_teammates/` | Chih-yu Li's Finance and William Sun's Tech AS01 submissions (as supplied) |
| `AS02_llm_api/` | Shreya's AS02: LLMBox + MLX integration (`project/`, `llmbox/`), frozen policies (`governance/`), all six 50-request runs and 30 safety probes (`runs/policy_first_gpu/`), earlier prototype and timing evidence, submitted report (`report/`) |
| `verification/` | Independent recomputation scripts (run on the Mac and the course VM) |
| `AI_USAGE.md` | AI-assistance disclosure |

## Key numbers (all recomputed from raw outputs)

| Run | What it adds | Valid JSON | All 5 facts right |
|---|---|---|---|
| B1 | generate (T 0.2, top_p 0.8, 256 tokens) | 10/50 | 6/50 |
| B3 | generate (T 0.7, top_p 0.95, 384 tokens) | 5/50 | 2/50 |
| C1 | Pydantic structured output | 0/50 | 0/50 |
| C2 | read-only `lookup_selected_posting` tool | 22/50 | 17/50 |
| C3 | regex input/output guardrail | 0/50 | 0/50 |
| C4 | schema + source + quote verifier | 0/50 (50 withheld) | 0/50 |

AS01 (Pharma, 25 human labels): Phi 48% field agreement, 0/25 full records, 92% schema-valid; Gemini Flash 92.4%.

## Reproduce

```bash
cd AS02_llm_api/project
pip install -r requirements-local-lock.txt
python -m unittest test_career_api test_llmbox_integration   # 17 tests
```

Model: `mlx-community/Phi-4-mini-instruct-4bit` (Apple silicon, MLX). LLMBox upstream: https://github.com/sarakingsley/llmbox (commit `23f97e7`).
API keys are read from a local `.env` (not committed). Build scripts under `final_project/build/` assume the original local folder layout.

## Data note

Records are public job advertisements retrieved via the Greenhouse and Lever job-board APIs, NYC Open Data and employer career pages (Sept 18–20, 2026). Per-source terms are in each `sources.csv`; public access is not an open redistribution license.
