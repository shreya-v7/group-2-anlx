# AI usage

This is the single consolidated disclosure for the team research report, scientific poster, presentation guide, supporting package and Shreya’s AS02 report. Shreya authorized AI assistance for these deliverables. The disclosure describes the actual assistance; it does not claim independent student authorship for generated text or convert automated judgments into human observations.

OpenAI Codex assisted with inspecting the supplied files, interpreting assignment requirements, adapting and debugging code, running local experiments, checking metrics, synthesizing the three sector contributions, locating primary research references, writing and formatting the reports and poster, preparing presentation notes, and packaging the evidence. Codex also assessed the delivered outputs of 20 adversarial response pairs in the native LLMBox rerun. Those assessments are stored separately as `AI_attack_assessment_NOT_HUMAN.jsonl`, with rationales and hashes; they are not human safety labels. The active assistant’s exact public model identifier is not asserted here because it is not established by the saved user-facing record.

The Pharma API experiments use local 4-bit Phi-4-mini-instruct with actual LLMBox mode and dispatcher methods and an MLX generation extension. The earlier standalone prototype and the integrated rerun are separate experiments. Model prompts, responses, settings, versions and validation outcomes are saved under `pharma_api/runs/llmbox_main/` and the historical prototype evidence. Pharma AS01 also used a local 4-bit Phi baseline and recovery experiment and a hosted Gemini comparison; its saved comparison records identify the mixed Gemini model identifiers. These inference runs are not model fine-tuning.

Shreya supplied the recorded human annotation review, original safety judgments, ten-output usability judgments and activity timings. No additional human review, timed activity, teammate contribution, in-person presentation, peer feedback or nomination was invented. Only identical response outputs inherited existing human output judgments in the integrated safety rerun.

Finance’s supplied reference-label records explicitly identify Claude Opus model pre-annotation and state that independent human review is not confirmed. Its memo reports local Phi and Groq-hosted `openai/gpt-oss-120b` experiments. The Tech report discloses Claude assistance and reports Phi and Azure GPT-5-mini experiments. These statements reflect the supplied materials; separate assistant conversations and raw model outputs for those teammates were not included. The reported numbers are not presented as independently reproduced results.

Available prompts and responses are in `assistance/conversation_redacted.jsonl`, its readable Markdown companion, `assistance/pasted_user_attachments_redacted.md`, Pharma AS01 prompt history and saved predictions, and the per-experiment response logs. The conversation export is a timestamped snapshot of available visible user/assistant messages and tool calls. It excludes internal reasoning, system/developer instructions and raw tool outputs; code and results are included as artifacts. Credentials and the course pseudonym are redacted. Separate earlier conversations, unavailable teammate assistance records and messages after the snapshot are not claimed to be included. Missing teammate records must be supplied for a complete assistance archive.

Original teammate documents and historical transcripts are retained as source evidence, including their existing disclosures. The newly prepared report, poster and presentation materials refer to this appendix rather than repeating disclosure sections.

For the subsequent prospective series, Codex drafted and froze the new initial policy before fresh inference, orchestrated the unchanged-code rerun, audited chronological records and metrics, transferred only verified identical-input/output labels, and prepared the measured policy revision and updated report. No human review time or changed-output human judgment was invented. The new inference logs are in prospective_rerun/runs/policy_first_gpu/.

On October 2, Codex performed a final requirements and evidence review, recomputed the saved AS01 baseline/recovery and AS02 metrics, reran the 17 existing software tests, reconciled report timings to the policy-first series, rendered and inspected the report/poster/presentation files, and rebuilt the handoff archives. Original AS01 ZIPs were preserved byte-for-byte. This pass introduced no new model experiment or human judgment.

On October 3, Claude (Anthropic, via Claude Code) independently recomputed the AS01 and AS02 metrics from the raw prediction and response files, reran the 17 software tests and the AS01 corpus validator, checked package checksums, verified the added references, revised the team report (author name, related work, discussion, references), rebuilt the poster with charts read directly from the saved metrics, and rebuilt this package. No new model experiment, human judgment or timing was created. Later the same day it verified William Sun’s Tech AS02 results with his recomputation script, confirmed from the raw Pharma files that 31 of 50 C4 answers would pass every C4 check after fence removal (a post-hoc diagnostic), and rebuilt the report, posters, presentation guide and package. William’s own AI use is disclosed in his memo.

Also on October 3, Claude wrote the fence-tolerant parser, its unit tests and the follow-up runner, wrote the pre-registration before the run, ran the follow-up on 50 previously unused evaluation records with the same local 4-bit model, recomputed its metrics independently, and added the result to the report, posters, presentation guide and project guide. No human labels, reviews or timings were created.

## Appendix: code assistance

AI coding assistants were used for software development and experiment execution. The research questions, design
decisions, human annotations, safety judgments, timing measurements and interpretation are the authors' own.

**Tools.** OpenAI Codex (Sept 27 to Oct 2, 2026; exact model identifier not recorded in the saved logs); Claude
(Anthropic) via Claude Code (Oct 3 to 4, 2026); Claude, used by William Sun for Tech AS02 (disclosed in his memo).

| Component | Files | Assistance |
|---|---|---|
| AS01 corpus pipeline | `collect_corpus.py`, `mask_corpus_content.py`, `extraction.py`, `corpus_stats.py`, `design_recovery.py`, `audit_results.py` | Adapting starter code, debugging, running the Phi and Gemini evaluations |
| LLMBox integration | `llmbox_integration.py`, `experiments.py` | MLX backend extension (`MLXGenerationManager`, `MLXBackend`) running LLMBox's unchanged modes |
| API logic | `career_api.py`, `safety.py`, `non_llm.py`, `cost_calculator.py` | `JobFacts` validation, regex guardrail, source checks, probe templates, comparators |
| Experiment runners | `run_llmbox.py`, `run_prospective.py`, `retry_with_gpu.py`, `finalize_evidence.py` | Runners that hash the frozen policy and inputs before inference; running the batches |
| Tests | `test_career_api.py`, `test_llmbox_integration.py`, `test_fence_parser.py` | 24 unit tests |
| Fence-tolerant parser follow-up | `fence_parser.py`, `run_followup_fence.py` | Parser, runner and the pre-registered 50-record run |
| Verification | `verification/verify_all.py`, `verify_as02.py`, `verify_tech_as02.py` | Recomputing every reported number from saved outputs (Mac and course VM) |
| Figures and builds | `build_poster.py`, `build_report.py`, `build_guide.py`, `package_final.py` | Charts from run files; document and package assembly |

Not AI-generated: the 25 human labels, safety block/allow judgments, usability judgments and review/correction
timings. AI assessments of attack outputs are stored separately and are not counted as human review.

### Selected code-assistance excerpts

Source A: OpenAI Codex session, `assistance/conversation_redacted.jsonl` (credentials and course username redacted).
Entry numbers and UTC timestamps are given; these are selected excerpts and the full log is the complete record.

1. [97] 2026-09-28 02:46, assistant: "The implementation now covers the six required experiment conditions, including
   the two custom features. I've also prepared separate development and evaluation sets, keeping the 25 previously
   labelled AS01 records in development."
2. [105] 02:50, assistant: "The code tests passed, including checks that an invented source fact is withheld, an
   unknown tool cannot run, and both steps of a tool call are counted."
3. [120] 04:45, user (terminal): "cd /opt/95820/students/[COURSE USERNAME]/as02_project ... python -m unittest -v
   test_career_api ..." / [121] 04:46, assistant: "All nine code tests passed. The model test produced no completed
   result. ... Let's use a shorter diagnostic to identify whether the delay is loading the model or generating an
   answer."
4. [123] 05:02, assistant: "model loading took 10.6 seconds, but processing the posting and generating just 16
   tokens took about 192 seconds. The CPU inference step is the bottleneck." / [132] 05:04: "The Mac GPU processed
   the same posting with a 16-token cap in 6.5 seconds, compared with 191.8 seconds on the VM CPU ... this is a
   runtime comparison, not proof of equal answer quality."
5. [269] 14:03, assistant: "The tool diagnostic found two issues: the model omitted the expected prefix and
   sometimes misspelled posting IDs. I'm testing a stricter interface that lets it request only the
   already-selected posting." / [276] 14:05: "The revised prompt now produces the correct tool request, but wraps it
   in a JSON code block. I'm checking a parser that accepts that complete block while still rejecting extra text,
   unknown tools and unexpected arguments."
6. [335] 17:23, assistant: "add an MLX model backend while retaining LLMBox's generation, structured-output and
   tool-dispatch code." / [341] 17:27: "The integration now calls LLMBox's actual mode methods and dispatcher. I've
   also restricted its tool registry to the selected posting." / [348] 17:30: "the model was trying to satisfy the
   tool-call format and the final answer schema at once. I've separated those stages."
7. [364] 18:53, assistant: "The structured-output failure is mostly formatting: 46 responses were wrapped in
   Markdown code fences. Removing only a complete fence in a separate diagnostic made 43 schema-valid, but just 31
   also passed source and quotation checks. ..."

Source B: Claude (Claude Code) session, October 3, 2026; user prompts quoted as typed.

8. User: "so please fix everything, whatever is wrong with my config, I want it to fully work, should i work on
   fixing schema then ?" / Claude: "the schema isn't the problem, the parser is. Your JobFacts schema is fine: 43 of
   50 answers fit it once the fences were removed." Resulting work: `fence_parser.py`, `test_fence_parser.py`,
   `run_followup_fence.py`; missing packages installed in the MLX environment (pydantic 2.13.5, hydra-core 1.3.7,
   omegaconf).
9. User: "connect to VM and verything and check again, all correct ?" Resulting work: `verification/verify_all.py`
   run on the course VM (17/17 tests; all metrics recomputed).
