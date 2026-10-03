# Pharma Career Facts — technical package

Chosen scenario: a hypothetical university career-services office helps students
explore public pharma vacancies. Its advisers use the API to extract posting facts
and compare advertised requirements, checking the original source before advising
students. It does not rank applicants or decide hiring eligibility. No actual
university adoption or authorization is claimed.

## Implemented

- Pydantic `JobFacts`: title, employer, location, work arrangement, up to four
  verbatim qualification excerpts, posting ID and source URL.
- Read-only `lookup_posting` tool, restricted to the current posting. Unknown
  tools, malformed calls and requests for other records are rejected.
- C3 input/output regex guardrail with explicit refusal text and saved reasons.
- C4 source verification: withhold schema-invalid responses, mismatched source
  fields, or qualification excerpts absent from the original text. This does not
  prove that an excerpt is semantically a required qualification.
- Instrumented VM backend using the supplied LLMBox GenerationManager loader and
  generation-settings helper. Prompt assembly, dispatch and instrumentation are
  project extensions. Upstream files are not overwritten.
- Optional MLX backend for separately labelled 4-bit local experiments.
- Frozen 100/100 split; 50 development inputs reused in B1/B3 and 50 evaluation
  inputs reused in C1–C4. All 25 AS01 reference-labelled records stay in development.
  Per-file hashes prevent quietly changing the prepared data. Exact duplicate
  descriptions do not cross splits; similar roles/shared employers may.
- Logs include each model call, tool events, response, settings, total token counts
  and end-to-end request latency. Model load time is separate. Existing output
  directories are never overwritten.
- Source-field accuracy, schema validity, blocking, tool execution and token/latency
  metrics. Invalid/blocked responses receive no delivered-answer correctness credit.
- 20 authored adversarial probes (five per category) and 10 benign probes, run in
  paired baseline/guarded conditions. Counterbalanced run order reduces systematic
  warm-cache effects. Input/output rules are transparent but easy to evade.
- Blank human-review and safety-adjudication files; no synthetic human labels.

## Current validation status

All six 50-record main batches are complete using the local 4-bit MLX model.
Nine core code tests passed. All 300 saved responses were checked against frozen
input IDs, recomputed metrics, and per-call token totals. Core experiment/API code
hashes match. Safety instrumentation was extended separately after B1 began;
that file is not imported by the main experiment runner.

The VM loaded the full model in 10.6 seconds but took 191.8 seconds to generate
16 tokens for one posting. Main experiments therefore use the separately labelled
local quantized model. Do not describe them as VM/full-precision results.

The primary C2 run emitted zero executable tool calls out of 50. Its failed results
are retained. Any tool_recovery results are exploratory development-only checks,
not replacement evaluation results. Source/schema checks are not human accuracy.

See the adjacent results folder and START_HERE.md for current completion status.

## VM commands

After unpacking this folder alongside your existing `llmbox` directory:

```sh
cd /opt/95820/students/[COURSE_USERNAME]/as02_project
source ../.venv/bin/activate
python -m unittest -v test_career_api
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 timeout 300s python experiments.py run \
  --data data --regime C1 --limit 1 --out runs/smoke_C1
```

A one-record smoke test always uses development data and is marked as a smoke
in its manifest. Once it passes, use 50-record runs (no `--limit 1`):

```sh
python experiments.py run --data data --regime B1 --out runs/B1
python experiments.py run --data data --regime B3 --out runs/B3
python experiments.py run --data data --regime C1 --out runs/C1
python experiments.py run --data data --regime C2 --out runs/C2
python experiments.py run --data data --regime C3 --out runs/C3
python experiments.py run --data data --regime C4 --out runs/C4
python safety.py --data data --out runs/D
```

Run sequentially on the shared VM. Review the smoke timing before committing to
longer batches. Each process exits after its batch. A failed/partial batch remains
marked failed; use a new output directory when rerunning.

B1 uses temperature 0.2, top_p 0.8 and 256 new tokens; B3 uses 0.7, 0.95 and 384.
Sampling is enabled so temperature/top_p actually affect generation. C1–C4 share
B1 settings. C2 can use two model calls; its totals include both. Output caps may
truncate JSON and are recorded, not silently repaired.

## Remaining assignment work

- Consult the saved results; do not rerun or overwrite completed batches.
- Complete genuine human judgments of work arrangement, qualification meaning,
  usability and safety decisions. Timing/correction fields start blank.
- Human-adjudicated baseline attack success, guarded success and agreement are
  deliberately not inferred from whether the guardrail blocked a request.
  The only implemented automatic attack-success signal is an injection canary;
  other categories require human assessment. The paired safety runner records a transparent five-word toxicity lexicon;
  it can miss abuse and flag quotations. Human validation remains necessary.
- Write the governance policy, report and cost/benefit analysis in accordance with
  course requirements. No assignment report is included. Source-table extraction
  is an obvious non-LLM comparator, but its full workflow benefit is not measured.
- AS03 requires teammates' contributions and live presentation/peer activities.
- Preserve the full AI conversation (with credentials redacted) for the mandatory
  assistance appendix. This README is not a substitute for that transcript.

The source data is private coursework material. Do not publish it. No passwords,
VM credentials or API keys are included in this package.

## Local reproduction and human-review tools

Use a Mac with the same local 4-bit model and install requirements-local-lock.txt
in an isolated environment. Substitute your local model path; do not rename these
results as full-precision VM results. Main run example (choose a new output path):

```sh
python experiments.py run --data data --regime C1 --backend mlx --model /path/to/phi-4-mini-instruct-4bit --out runs/new_C1
python safety.py --data data --backend mlx --model /path/to/phi-4-mini-instruct-4bit --out runs/new_D
python build_review.py --run ../as02_results_4bit/D --out ../human_safety_review.html
python review_results.py --paired ../as02_results_4bit/D/paired_responses.jsonl --labels /path/to/human_safety_labels.jsonl --out ../human_agreement.json
python cost_calculator.py --inputs /path/to/completed_cost_inputs.json --metrics ../as02_results_4bit/C4/metrics.json --out ../cost_estimate.json
```

The JSON review and cost tools refuse missing judgments/assumptions. New output
files are created without overwriting an earlier adjudication or cost result.
