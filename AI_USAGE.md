# AI Usage

This document is the consolidated AI-use disclosure for the team research project, including the team report, scientific poster, presentation materials, supporting package, and Shreya’s AS02 work.

## Scope of AI assistance

AI coding assistants were used for **software development, debugging, experiment execution, and technical verification**. They were not used to author the substantive written content of the team report, scientific poster, or presentation materials.

The research questions, experimental design decisions, interpretation of results, human annotations, safety judgments, usability judgments, timing measurements, report writing, poster writing, poster organization, and presentation content were completed by the student authors.

Where AI-assisted code was used to generate charts, render files, recompute metrics, run tests, or package deliverables, the AI contribution was limited to the underlying software or automation. The authors selected the results to present, checked the outputs, and created the final written and visual materials.

## Tools used

- **OpenAI Codex** — used from September 27 to October 2, 2026 for coding, debugging, experiment runners, local execution support, and verification scripts. The exact public model identifier was not recorded in the saved user-facing logs.
- **Claude (Anthropic), via Claude Code** — used from October 3 to October 4, 2026 for coding, debugging, test execution, verification scripts, and follow-up experiment tooling.
- **Claude used by William Sun for Tech AS02** — disclosed separately in William’s memo.

## Coding and experiment assistance

AI assistance was used for the following software-related work:

| Component | Files | AI assistance |
|---|---|---|
| AS01 corpus pipeline | `collect_corpus.py`, `mask_corpus_content.py`, `extraction.py`, `corpus_stats.py`, `design_recovery.py`, `audit_results.py` | Adapting starter code, debugging, and supporting execution of the Phi and Gemini evaluations |
| LLMBox integration | `llmbox_integration.py`, `experiments.py` | Implementing and debugging the MLX backend extension (`MLXGenerationManager`, `MLXBackend`) while retaining LLMBox generation, structured-output, and tool-dispatch behavior |
| API logic | `career_api.py`, `safety.py`, `non_llm.py`, `cost_calculator.py` | Implementing and debugging `JobFacts` validation, regex guardrails, source checks, probe templates, and comparators |
| Experiment runners | `run_llmbox.py`, `run_prospective.py`, `retry_with_gpu.py`, `finalize_evidence.py` | Writing and debugging experiment runners, including hashing frozen policy and inputs before inference |
| Tests | `test_career_api.py`, `test_llmbox_integration.py`, `test_fence_parser.py` | Writing and running unit tests |
| Fence-tolerant parser follow-up | `fence_parser.py`, `run_followup_fence.py` | Writing the parser, runner, and supporting code for the pre-registered 50-record follow-up |
| Verification | `verification/verify_all.py`, `verify_as02.py`, `verify_tech_as02.py` | Writing and running scripts that recomputed reported metrics from saved outputs |
| Figures and build tooling | `build_poster.py`, `build_report.py`, `build_guide.py`, `package_final.py` | Writing code used to generate figures, render files, and assemble the final package from author-prepared content and saved experiment results |

## Human-authored work

The student authors were responsible for the substantive academic work and final communication of the project. In particular:

- The **team research report was written and edited by the authors**.
- The **scientific poster was designed, organized, and written by the authors**.
- The **presentation guide and presentation content were prepared by the authors**.
- The research questions, claims, discussion, interpretation, and conclusions were determined by the authors.
- Human annotation decisions, safety block/allow judgments, usability judgments, and recorded review/correction timings were supplied by the human researchers.
- AI-generated assessments were not treated as human judgments.

AI-assisted scripts may have been used to calculate metrics, generate plots from saved results, render documents, or package files, but those software operations are distinct from authorship of the report, poster, and presentation content.

## Experiment notes

The Pharma API experiments used local 4-bit Phi-4-mini-instruct with LLMBox mode and dispatcher methods plus an MLX generation extension. The earlier standalone prototype and the integrated rerun were separate experiments. Model prompts, responses, settings, versions, and validation outcomes were saved under `pharma_api/runs/llmbox_main/` and the historical prototype evidence.

Pharma AS01 also used a local 4-bit Phi baseline and recovery experiment and a hosted Gemini comparison. These were inference experiments, not model fine-tuning.

For the later prospective series, AI-assisted code was used to implement the frozen policy, run the unchanged-code experiment, audit chronological records and metrics, transfer only verified identical-input/output labels, and recompute the resulting measurements. No additional human review time or changed-output human judgment was invented.

The fence-parser follow-up used AI-assisted code for the parser, tests, experiment runner, and metric recomputation. The follow-up was run on 50 previously unused evaluation records with the same local 4-bit model. No human labels, reviews, or timings were created by AI.

## Human and AI judgments

Shreya supplied the recorded human annotation review, original safety judgments, ten-output usability judgments, and activity timings. No additional human review, timed activity, teammate contribution, in-person presentation, peer feedback, or nomination was invented.

Only identical response outputs inherited existing human output judgments in the integrated safety rerun.

Finance’s supplied reference-label records identify Claude Opus model pre-annotation and state that independent human review is not confirmed. Its memo reports local Phi and Groq-hosted `openai/gpt-oss-120b` experiments.

The Tech report separately discloses Claude assistance and reports Phi and Azure GPT-5-mini experiments. William Sun’s own AI usage is described in his memo.

## Saved evidence

Available prompts, model responses, experiment settings, predictions, test outputs, and verification records are retained in the project evidence, including:

- `assistance/conversation_redacted.jsonl`
- its readable Markdown companion
- `assistance/pasted_user_attachments_redacted.md`
- Pharma AS01 prompt history and saved predictions
- per-experiment response logs
- `AI_attack_assessment_NOT_HUMAN.jsonl`
- verification scripts and outputs
- prospective rerun outputs under `prospective_rerun/runs/policy_first_gpu/`

The conversation export is a timestamped snapshot of available visible user/assistant messages and tool calls. It excludes internal reasoning, system/developer instructions, and raw tool outputs. Credentials and the course pseudonym are redacted.

Original teammate documents and historical transcripts are retained as source evidence, including their existing disclosures.

## Summary

AI was used as a **coding and technical execution assistant**. The final report, poster, and presentation materials were **authored by the student team**. AI-assisted automation was used where applicable to run experiments, test software, recompute metrics, generate figures from saved data, render files, and package project artifacts, but it was not used as the author of the project’s substantive written or visual communication.
