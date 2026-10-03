# Job Posting Extraction Across Pharma, Finance and Tech

Shreya Verma · Chih-yu Li · William Sun
95820 Applications of NLX and LLM · Carnegie Mellon University
3 October 2026

## Abstract

A university career-services office needs reliable summaries of job requirements, rather than an automated hiring decision. We combine three sector studies and two reproducible LLMBox API evaluations, Pharma and Tech, to ask how dataset construction, output validation and review effort affect that use case. The supplied corpora contain 650 records associated with 446 distinct source URL strings: 200 Pharma postings, 200 Finance records from 175 URLs, and 250 Tech fragments from 71 URLs. Different annotation schemes and input units prevent a pooled accuracy comparison. In 300 recorded Pharma requests using local 4-bit Phi-4-mini-instruct, a bounded lookup tool executed on all 50 tool-mode requests, while only 17 final answers matched all five scored source fields. Strict structured-output validation accepted none of 50 answers; removing complete Markdown fences in a separate post-hoc diagnostic yielded 43 schema passes. Thirty paired safety probes showed 80% agreement with human block/allow judgments and one false refusal among ten benign requests. A separately built Tech API on the same model parsed every fenced answer and, by repairing rather than withholding unsupported values, raised records usable without edits from 7 to 39 of 50; a rerun of two runs reproduced their outputs exactly, yet none of its 25 hand-labeled records had every field right, and its guardrail passed five of six discriminatory postings. These findings support deterministic retrieval of existing metadata and supervised experimentation for interpretation. They do not support unattended career advice or conclusions about sector-wide hiring trends.

## Introduction

Job postings mix employer descriptions, role responsibilities, required credentials, preferred qualifications and compensation tables. A career adviser must distinguish these elements without inventing missing information. A wrong work arrangement can waste a student’s application effort; a truncated education alternative can incorrectly discourage a qualified applicant. The organizational objective is therefore a cited, reviewable summary of a selected posting.

We ask three questions. First, what does a record represent in each sector, and what comparisons does that permit? Second, do structured output, restricted tools and verification produce usable and source-consistent answers? Third, how do refusals and human correction effort change the case for adoption? Our contributions are a provenance-preserving corpus audit, a synthesis of sector extraction studies, and two documented API case studies with retained failures, safety probes and a limited cost scenario.

The team contributions supplied for this synthesis are Shreya Verma’s Pharma corpus and AS01/AS02 evidence [29], Chih-yu Li’s Finance corpus and memo [30], and William Sun’s Tech corpus and AS01 report [31] and Tech AS02 API runs [33]. The sector packages do not all contain the same experiment artifacts. Findings below distinguish locally audited files from numbers available only in a teammate’s memo.

# Data and related work

## Related work

Documentation of data and models. Datasheets for Datasets [1] and Data Statements [2] motivate documenting collection, composition and intended use. We apply that principle by retaining source URLs, record units and sampling limits instead of treating the combined corpus as a random labor-market sample. Model Cards [3] motivate reporting performance in the conditions relevant to an intended use; here schema validity, source agreement, refusals and human review are reported as separate outcomes.

Small models and the runtime. Phi-4-mini is a 3.8-billion-parameter instruction-tuned model whose technical report and model card [4, 5] emphasize instruction following and function calling. We run a community 4-bit MLX conversion [6, 7] through the LLMBox modes and tool dispatcher [8]; this quantized local execution is a particular implementation, not a reproduction of the reported benchmarks.

Structured output. Constrained decoding can guarantee that generated text matches a grammar or schema [10], whereas a format request in the prompt only asks for it. Tam et al. [11] report that format restrictions can also reduce answer quality. Our C1 condition requests JSON and validates it afterwards with Pydantic [9] without constrained decoding, which is why Markdown fences can cause a complete schema failure even when the enclosed object is usable.

Tools, injection and refusal. Toolformer [12] shows models learning to call external tools; our tool is deliberately limited to one read-only lookup. Prompt injection [13] and indirect injection through retrieved content [14] motivate our planted-instruction probes with canary tokens, and the OWASP list [15] names both injection and sensitive-information disclosure as leading LLM-application risks. XSTest [16] shows that safety measures can refuse legitimate requests, so we report over-refusal next to the catch rate. For toxicity, RealToxicityPrompts [17] illustrates classifier-based measurement, Llama Guard [18] a model-based safeguard and Zheng et al. [19] the strengths and biases of LLM judges; we chose a transparent lexicon and human adjudication instead. The NIST AI Risk Management Framework [20] informs the measured-versus-precautionary policy marks.

## Corpus composition

| Sector | Records | Exact URLs | Employers | Median text characters |
| --- | --- | --- | --- | --- |
| Pharma | 200 | 200 | 8 | 5,886 |
| Finance | 200 | 175 | 27 | 4,017 |
| Tech | 250 | 71 | 68 | 79 |

Counts were recalculated from the supplied JSONL, rather than copied from the memos. There are no exact URL strings shared between sectors. Exact URLs are useful identifiers, but are not proof of independent observations, current vacancies or unique real-world jobs. The character statistic excludes table content, which is especially important for Tech.

Pharma contains complete therapeutic biotechnology and pharmaceutical postings from eight employers, retrieved from the public Greenhouse Job Board API [25] on September 20, 2026. All 200 records have text and tables. Finance uses an employer-industry definition: 18 Greenhouse boards [25], two Lever boards [26] and the City of New York’s NYC Jobs open dataset [27], retrieved September 19, 2026. Its 148 mixed and 52 text records include whole postings and some extracted sections. Tech focuses on early-career technology roles from public career pages on Greenhouse, Workday, Ashby, SmartRecruiters, iCIMS, Eightfold, Lever, BambooHR and employer sites [28], retrieved September 18, 2026, with 115 text, 134 table and one mixed record. Its company-overview, responsibility, qualification and compensation fragments can describe the same source posting.

## Sampling limits

Finance’s supplied README describes employer caps and the absence of many banks using inaccessible boards; Pharma is concentrated in a few biotechnology employers; Tech prioritizes entry-level roles. Sector membership therefore combines different definitions of industry and occupation. We cannot estimate market share, compare sector demand, infer salary differences without selection bias, or conclude how many people were hired. A larger row count is not necessarily a larger sample of jobs.

Provenance granularity also differs. Pharma’s sources.csv lists all 200 posting URLs. Finance lists 21 board-level sources and Tech 72 board- or page-level sources; their record counts sum to 200 and 250, but individual posting URLs are not all listed row by row. Licensing notes state that public access is not an open redistribution license, so the combined corpus remains private coursework material.

# Methodology and evidence boundaries

## Sector extraction studies

The AS01 studies use different schemas: Pharma scores nine fields including verbatim qualification lists, Finance uses role family, seniority, work arrangement, certifications, skills and experience, and Tech uses role category, seniority, skills, location type, company and optional salary. We preserve the original labels and do not coerce them into a shared accuracy denominator. Pharma clusters were produced with TF-IDF features and k-means (k = 6) in scikit-learn [21]. Pharma has 25 labels with a saved human-review record. Finance has 28 labels whose per-record provenance states model pre-annotation and unconfirmed independent human review. Tech supplies 25 labels without equivalent per-row provenance. Finance results therefore measure agreement with those reference labels, not established independent human accuracy.

Pharma predictions, evaluation files and prompt history are available for audit. Finance raw predictions, Tech AS01 raw predictions and several dependencies named by their READMEs are absent from the supplied ZIPs. Their numerical findings are transcribed as reported results with this status in a separate metrics dataset. They are useful case evidence, but cannot be independently recomputed from the provided packages. No model was fine-tuned in this synthesis. Tech AS02 outputs, by contrast, are in the repository and every Tech API number below is recomputed from them.

## Pharma API experiment

The frozen 200-record Pharma corpus is split into 100 development and 100 evaluation records with seed 820, following the course’s AS02 evaluation protocol [32]. All 25 prior AS01 labeled records are in development. B1 and B3 use 50 development inputs; C1 through C4 use 50 evaluation inputs each. The latter records had been examined previously, and one known near-duplicate pair crosses the sampled groups. This is a documented rerun, not a new blind holdout or an employer-held-out test. The final series froze policy V1 before inference and produced V2 afterward. Earlier measurements informed V1; this correct sequence does not erase the earlier chronology.

| Run | Intervention | Settings |
| --- | --- | --- |
| B1 | Ordinary generation | temperature .2; top_p .8; cap 256 |
| B3 | Sampling and length change | temperature .7; top_p .95; cap 384 |
| C1 | Structured output with JobFacts schema | B1 settings |
| C2 | Restricted selected-posting lookup tool | B1 settings, two model calls |
| C3 | Regex input and output guardrails | B1 settings |
| C4 | Schema, source and quotation verification | B1 settings |

The actual LLMBox [8] Modes methods and dispatcher run with an instrumented MLX generation extension and local 4-bit Phi-4-mini-instruct [6, 7]. Sampling uses top_k 0, repetition penalty 1 and seed 820 plus input index. Tokens and timing include both C2 calls; shared model loading is excluded from request latency. The code and package versions are frozen in the protocol. Primary results below use the policy-first series recorded on September 30 UTC (September 29 evening in New York).

Primary outcomes are strict Pydantic [9] schema validity, delivered agreement on five source fields, withholding and latency. The five fields are ID, URL, title, employer and location, matched after trimming and case folding. Invalid or withheld answers receive zero delivered-field credit. This deliberately conservative score does not measure work-arrangement truth, qualification completeness or adviser-approved usability.

## Tech API experiment

The Tech study [33] is a separate LLMBox implementation, designed and run independently of the Pharma series. Its 250 fragments are split by source URL with seed 42, so no posting appears on both sides: 124 development records from 39 postings and 126 evaluation records from 32 postings. Two sampling runs use 50 development records; five cumulative feature runs (C0 to C4) use the same 50 evaluation records, which include the 25 AS01 hand-labeled records. Those records had been examined before, so this is not a blind holdout. The model is microsoft/Phi-4-mini-instruct run through Hugging Face Transformers in bfloat16 on Apple MPS, not the 4-bit MLX checkpoint used for Pharma. The LLMBox snapshot used for these runs is kept with the outputs because its starting version was not pinned to the upstream commit in [8].

| Run | What it adds | Settings |
| --- | --- | --- |
| C0 | Ordinary generation | temperature .2; top_p .5; cap 256 |
| C1 | Pydantic contract with optional company, salary and location | C0 settings |
| C2 | Read-only lookup tool returning company and official title | C0 settings, two model calls |
| C3 | Guardrail: regex rules, one yes/no model check, tool bound to the current posting | C0 settings |
| C4 | One retry on schema error, then a cleanup that keeps only values found in the posting | C0 settings |

The primary Tech outcome is a record usable without edits: it passes the schema, company and seniority match the posting metadata, every skill appears in the posting text as a short name, and any salary is in the text. The model never receives the metadata, but from C2 on the tool returns the company from that same metadata, so company accuracy then measures tool use rather than reading. Location is not part of the usable rule. The 25 hand labels add all-field agreement and skill F1. The parser removes a complete Markdown fence before validation in every run. Usable records and the Pharma five-field score are different measures, so the two series are compared by design pattern, not by number.

# Findings from the sector studies

| Study | Reference sample | Phi schema valid | Phi all fields correct | Evidence |
| --- | --- | --- | --- | --- |
| Pharma | 25 | 23/25 | 0/25 | Saved outputs and audit |
| Finance | 28 | 12/28 | 0/28 | Memo; reference review unconfirmed |
| Tech | 25 | 7/25 | 1/25 | Report; raw outputs absent |

These rows summarize separate tasks; they are not a sector ranking. Pharma’s nine-field baseline matches 108 of 225 reference fields (48%). Title and location each reach 92%, while work arrangement reaches 16% and job function 32%. Exact required-qualification-list agreement is 20%, although the saved list F1 is .733. The difference shows that recovering some list items is easier than producing the complete agreed list.

The Finance memo reports skill-list exact agreement of zero and F1 .093, with 16 of 28 schema violations. Fourteen violations involved null values for an optional numeric field that should have been omitted. Its reported 86% certification agreement is dominated by empty references: the memo says all four positive credential cases were missed. A high agreement rate can therefore conceal failure on the cases that matter. Unconfirmed reference review further limits this interpretation.

The Tech report describes 18 of 25 schema violations, skill-list exact agreement of 4%, and F1 .114. A few-shot recovery condition on 24 previously failed records reaches six all-field matches and reduces schema violations to 11. That selected subset is not a fresh 25-record evaluation. Finance’s skill-focused revision raises reported skill F1 to .228 while schema violations remain 17 of 28. Prompt changes can improve one outcome while leaving another poor.

## Larger-model comparisons

Pharma’s saved comparison uses a mixture of Gemini model identifiers [22] across 25 calls and reports 12 complete matches, versus zero for Phi, with all 25 schemas valid. Tech reports four complete matches and one schema violation for Azure GPT-5-mini [24] on 25 records. Finance reports zero schema violations and skill F1 .279 for Groq-hosted gpt-oss-120b [23] on 28 records. These separate provider, prompt, schema and reference conditions cannot establish a general ranking of model families or isolate model size as the cause. Local and hosted latency measurements also reflect different hardware and services.

## Implication for the team question

The common failure pattern is more useful than a pooled percentage: schema conformance is distinct from semantic adequacy; empty labels and partial list overlap can inflate impressions of success; and recovery results on previously failed examples are not unbiased generalization estimates. A common future evaluation should group all fragments of a posting together and define shared annotation rules before any sector comparison.

# Findings from the Pharma LLMBox API

| Run | Schema valid | All five fields | Withheld | Mean seconds |
| --- | --- | --- | --- | --- |
| B1 | 10/50 | 6/50 | 0/50 | 9.63 |
| B3 | 5/50 | 2/50 | 0/50 | 11.97 |
| C1 | 0/50 | 0/50 | 0/50 | 14.10 |
| C2 | 22/50 | 17/50 | 0/50 | 21.04 |
| C3 | 0/50 | 0/50 | 1/50 | 13.04 |
| C4 | 0/50 | 0/50 | 50/50 | 13.74 |

The larger B3 sampling/output budget did not improve this measured task over B1. C2 successfully executed all 50 restricted source lookups, but only 22 final answers passed the schema and 17 matched every scored source field. Successful tool execution is therefore an operational check, not evidence that the delivered summary is correct. C2 uses an average of 2,468.54 input and 373.22 output tokens across its calls, compared with C1’s 1,693 and 201.68.

## Formatting failure and post-hoc diagnosis

C1’s zero strict schema passes were strongly affected by Markdown wrappers, consistent with requesting rather than constraining the format [10, 11]. All 50 current C1 outputs exactly match the prior integrated series. In its retained diagnostic, 46 contain complete fenced JSON blocks. A separate diagnostic that removes only the complete enclosing fence produces 43 schema-valid answers, of which 31 pass the additional source and literal-quote checks. This diagnosis changes how we understand the failure, but does not replace the primary scores or establish semantic completeness. The same outputs were inspected after the experiment; a repaired parser must be tested prospectively on new records.

C4 withholds all 50 answers, leaving no useful coverage. All 50 withholdings were schema failures: C4 received the same fenced model outputs as C1 (50/50 identical). Removing the complete fence before validation, as the Tech parser does, 31 of 50 would have passed every C4 check; this is a post-hoc diagnostic, not a delivered result. Calling this result safe automation would ignore the loss of service. Likewise, a verbatim quotation can still be a responsibility rather than a qualification or omit a necessary alternative. Guardrails must be evaluated jointly with task completion and review burden.

## Reproducibility

The archive retains inputs, messages, raw answers, call-level tokens, validation and metrics. Seventeen software tests pass. The audit recomputes all 300 intended rows and verifies 37 frozen files and policy-before-inference timestamps. All 300 delivered outputs match the prior integrated series under unchanged seeds/settings; fresh latency is reported here. This is not a repeated-seed uncertainty study. Upstream commit: 23f97e77009cf5fe6f4d46b8a68234d18f9d2a30. Local quantized inference differs from the course VM model execution.

## Organizational meaning

Existing source metadata can be copied deterministically. Generative effort is better reserved for interpretation that genuinely requires reading the posting, with an adviser checking the evidence. The current API should remain a research prototype. Its output should not determine applicant eligibility, rank people or substitute for professional judgment.

# Findings from the Tech LLMBox API

| Run | Usable | Company | Seniority | Clean skills | Skill F1 (hand) | Refused | Mean seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | 7/50 | 38/50 | 22/50 | 10/50 | .10 | 0 | 4.49 |
| C1 | 13/50 | 35/50 | 26/50 | 19/50 | .14 | 0 | 5.09 |
| C2 | 17/50 | 50/50 | 37/50 | 21/50 | .12 | 0 | 9.74 |
| C3 | 17/50 | 43/50 | 34/50 | 20/50 | .14 | 7 | 9.80 |
| C4 | 39/50 | 43/50 | 39/50 | 43/50 | .37 | 7 | 11.10 |

Lowering temperature from 1.0 to .2 on the development records cut usable records from 5 to 1 of 50 and seniority from 36 to 23, because the most likely label for new-grad titles was the wrong one. As in Pharma B3, sampling settings were not the lever. The tool made company correct on all 50 requests in C2. In C3 and C4, seven answers were refused for invalid output and scored as wrong, which explains the drop to 43; the model also skipped the tool on five requests, but those still had the right company.

## Fences without a failure

The model wrapped its JSON in Markdown fences here too: 28 of 50 C1 answers and all 50 C3 answers were fenced, and every fenced answer failed strict JSON parsing. Because the Tech parser removed complete fences before validation, all of them parsed and 39 of 50 C1 answers passed the schema. This is not a test of the Pharma parser, but it is independent evidence on other records that the same model’s fence habit is an integration problem that a fence-tolerant parser handles.

## Repair instead of withholding

The largest step came from C4. Its cleanup changed 33 of 50 records, mostly by dropping or shortening skills that did not appear in the posting as names, and raised usable records from 17 to 39. Pharma C4 withheld every answer that failed verification and delivered nothing, but its failures were formatting, not content: with complete fences removed, 31 of 50 Pharma answers would have passed its stricter checks (all five source facts and verbatim quotes). Most of the 0 versus 39 gap is therefore the parser; the remaining difference is repair versus withholding, measured on different scores. Both verifiers check answers against the posting; they differ in what happens next. Repair keeps service, withholding removes it, and both still need review.

## What the usable rate does not show

On the 25 hand-labeled records, plain generation had every field right on 3, and C4 on none, mainly because location type was left empty from C1 on, once the schema made it optional: hand-label location agreement fell from 15 of 25 in C0 to 1 in C1 and 0 in C4. Five of the 39 usable C4 records keep a number such as 2027 or 0-2 as a skill, because the cleanup checks that a skill appears in the text, not that it is a technology; counting those as errors gives 34 of 50. Most of the gain comes from rules and the metadata lookup rather than from better reading, which supports copying known facts directly. The features were designed on development records similar in style to the evaluation set, so these rates are likely optimistic.

## Reproducibility

All Tech AS02 inputs, messages, model text, delivered output, tool calls, guardrail decisions, tokens and latency are kept, and verification/verify_tech_as02.py recomputes every Tech number above from the saved files without loading a model. A rerun of C1 and C4 on October 3 (torch 2.14.0, transformers 5.17.0, Apple MPS, bfloat16, seed 42) reproduced the model text and delivered output for 50 of 50 records in both runs; C4 mean latency was 10.63 seconds against 11.10. Other runs were not repeated and no other hardware was tested.

# Safety and operational cost

## Paired safety evaluation

Thirty authored probes cover harmful content, out-of-scope requests, indirect injection [14] and leakage [15] (five each), plus ten benign requests. Each input is run through a structured baseline and the regex-guarded condition. The guarded API blocks 3/5 harmful, 5/5 out-of-scope, 3/5 injection and 4/5 leakage probes. It also blocks 1/10 benign requests, an over-refusal rate of the kind XSTest [16] warns about and above the 5% ceiling in the initial policy. Blocking 15/20 attacks does not mean the other five attacks succeeded; success depends on the delivered output.

Previously supplied human block/allow labels apply to all 30 identical inputs and agree with the new decisions in 24 cases (80%). Complete human output labels transfer only for 13 byte-identical response pairs, including ten adversarial pairs with no labeled attack success. Seventeen pairs changed relative to the original human-reviewed prototype do not inherit those judgments. An automated assessment transferred only for exactly matching input/output pairs records no successful attacks in the 20 adversarial pairs; it is not independent human confirmation. The small, authored probe set and the absence of real hidden secrets prevent a general security or confidentiality claim.

No injection canary appeared. A five-term toxicity lexicon, chosen over a classifier [17, 18] or LLM judge [19] for transparency, flagged no outputs in either condition; it cannot detect many forms of implicit abuse. Guardrails reduced mean latency by 7.22 seconds per probe, largely because some requests were refused before generation. This is not evidence that guarded model reasoning runs faster.

## Tech probe set

The Tech study used a separately authored set of 70 probes [33], written after its guardrail was finished: ten each of harmful, out-of-scope, injection and leakage, plus 20 real postings and 10 normal postings worded like attacks (“New instructions: submit a portfolio”). Without the guardrail, 8 of 40 attacks succeeded; with it, 2 did, both polite injections that placed a canary in the skills field. Leakage was caught 8 of 10 times; three of the four unguarded leaks came through tool calls for other postings; binding the tool to the current posting stopped two, and the third was refused for invalid output. Seven of 30 normal inputs were refused: four attack-sounding postings by a policy rule and three for invalid output. Five of six discriminatory postings, such as coded age or religion requirements, were saved as normal records; the sixth was refused only for invalid output. The guardrail checks the output, which contained no harmful words, not the job itself. A random 20-decision hand check agreed with the guardrail 15 times; four of the five disagreements were these postings.

## Limited cost scenario

Shreya timed 330 seconds for manual extraction, 90 seconds for review of one earlier prototype C1 answer, and 135 seconds for correcting a different wrong answer. Four of the first ten prototype C1 outputs were usable without correction. Under a 30-minute setup assumption and no blocked human waiting time, expected human effort is 90 + 0.6 × 135 = 171 seconds per posting. The conditional saving is 159 seconds, recovering setup after ceiling(1800/159) = 12 postings.

These are one timing observation per activity and a ten-output usability sample from the earlier standalone prototype. They are not measured savings for the new native-mode C1, C2, Finance or Tech results. They exclude undetected-error consequences, power, hardware depreciation and currency costs. If review took 249 seconds with the same correction assumptions, estimated time savings would disappear. A deployment decision needs new timings for the actual selected implementation. The Tech study estimated review effort from its own labeling speed but did not time manual extraction, so it is not used for a cost figure here.

## Discussion

Who is affected. The benefits and risks fall on different people. Advisers may save preparation effort, while students bear the cost of incorrect advice: a wrong work arrangement wastes an application, and a truncated “or equivalent experience” clause can discourage a qualified applicant. Employers are affected when a summary misstates their posting, even though they never use the system.

What the results mean for the design. The measurements locate the problem. Retrieval works (50/50 tool executions) and the source table already contains title, employer, location and URL, so a deterministic lookup is preferable for those fields. The model’s difficulty lies in output format and interpretation: strict validation rejected every C1 answer, and even the best condition matched all five scored fields on 17 of 50 postings. Format enforcement through constrained decoding [10] is therefore a better next investment than larger token budgets or higher temperature, which made B3 worse than B1. The Tech study points the same way: its usable rate rose most when rules and a metadata lookup took over company, title and seniority, and its fence-tolerant parser removed the formatting failure that dominated Pharma C1.

What the results mean for governance. A verifier that withholds unsupported answers (C4) reached complete safety by delivering nothing, and the keyword guardrail refused 10% of benign requests. The Tech verifier, which repaired instead of withholding, kept 39 of 50 records usable but produced no fully correct hand-labeled record, and the Tech guardrail passed five of six discriminatory postings, a risk that output checks cannot see. Safety and service must be reported together. The measurements support keeping adviser review on every output and do not yet support student-facing use. Proposed controls include local processing of public postings, a selected-record read-only tool, exclusion of applicant data and adviser review before use. Authentication, retention enforcement and institutional approval are not implemented production controls.

Across sectors. The Finance and Tech reports show the same pattern under different schemas: larger hosted models improved schema compliance, while list-valued fields such as skills or qualifications remained hard to match exactly. Because schemas, references and providers differ, this is a shared qualitative lesson rather than a measured ranking of models or sectors. The two API studies agree in the same qualitative way despite different schemas, runtimes and scores: tools run, format is an integration problem, and semantic completeness remains the hard part.

# Future work and conclusion

## Future work

A first revision should test constrained decoding [10] or normalize complete JSON fences before validation, retain the original answer and reject partial or ambiguous JSON. Its evaluation should use new postings from new employers, with all fragments from one posting assigned to the same split. A fixed common schema and independent double review would make a sector comparison interpretable. Finance’s pre-annotations should be independently reviewed, and the missing Finance and Tech AS01 predictions, prompts, metric files and dependencies should be archived before claiming full reproducibility. The Tech parser already strips complete fences; one shared parser should be tested on new Pharma and Tech records. Screening the posting itself for discriminatory requirements, not only the output, is needed before any student-facing use.

The next semantic audit should separately score required versus preferred qualifications, education alternatives, explicit work arrangement and source support. Human reviewers should measure correction time on a randomized sample large enough to characterize variation. A balanced safety set should include harmless quotations, paraphrased attacks and legitimate requests likely to trigger keywords. Repeated seeds and confidence intervals would help distinguish systematic improvements from sampling variation.

For an organizational pilot, proposed acceptance criteria are at least 95% adviser-approved usability on a new evaluation, no severe source misrepresentation in that sample, benign refusal at most 5% and positive time savings after review. These are proposed decision criteria, not achieved performance or guarantees. The organization should record its acceptable error consequences before setting final thresholds.

## Conclusion

Across the three sector studies, record construction and label provenance determine what performance claims mean. The Pharma API demonstrates that a working tool and a valid structure do not by themselves establish reliable career guidance. The Tech API shows that repairing answers against the source keeps more service than withholding them, but its usable rate still rests on rules and lookups rather than on more reliable reading. A strict verifier can remove all service, while a formatting change can substantially change a score without changing a model’s underlying text. For this use case, the defensible recommendation is direct retrieval of existing facts and supervised, evidence-linked interpretation. Students, advisers and employers benefit from seeing the source and the limits of the summary, rather than treating a single accuracy number as a deployment decision.

## Presentation and participation

The accompanying scientific poster and presentation guide communicate the study to nontechnical visitors. The guide includes a three-person speaking plan, a short demonstration activity and questions for discussion. The course also requires the team to present in person, provide verbal feedback on at least two peer projects, submit the feedback survey and nominate eligible peer projects in the stated categories. Those activities have not been performed or represented as completed in this package.

# References and evidence index

## Literature and methods

[1] Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J. W., Wallach, H., Daumé III, H., & Crawford, K. (2021). Datasheets for datasets. Communications of the ACM, 64(12), 86–92. https://arxiv.org/abs/1803.09010

[2] Bender, E. M., & Friedman, B. (2018). Data statements for natural language processing: Toward mitigating system bias and enabling better science. Transactions of the ACL, 6, 587–604. https://aclanthology.org/Q18-1041

[3] Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I. D., & Gebru, T. (2019). Model cards for model reporting. Proceedings of FAT* 2019. https://arxiv.org/abs/1810.03993

[4] Microsoft (2025). Phi-4-Mini technical report: Compact yet powerful multimodal language models via mixture-of-LoRAs. arXiv:2503.01743. https://arxiv.org/abs/2503.01743

[5] Microsoft. Phi-4-mini-instruct model card. https://huggingface.co/microsoft/Phi-4-mini-instruct (accessed September 28, 2026).

[6] Hannun, A., Digani, J., Katharopoulos, A., & Collobert, R. (2023). MLX: Efficient and flexible machine learning on Apple silicon; and mlx-lm. https://github.com/ml-explore/mlx

[7] MLX Community. Phi-4-mini-instruct-4bit (quantized checkpoint used in all local runs). https://huggingface.co/mlx-community/Phi-4-mini-instruct-4bit

[8] Kingsley, S. LLMBox. https://github.com/sarakingsley/llmbox ; experiment snapshot commit 23f97e77009cf5fe6f4d46b8a68234d18f9d2a30.

[9] Colvin, S., et al. Pydantic (v2) data validation library. https://docs.pydantic.dev

[10] Willard, B. T., & Louf, R. (2023). Efficient guided generation for large language models. arXiv:2307.09702. https://arxiv.org/abs/2307.09702

[11] Tam, Z. R., Wu, C.-K., Tsai, Y.-L., Lin, C.-Y., Lee, H.-y., & Chen, Y.-N. (2024). Let me speak freely? A study on the impact of format restrictions on performance of large language models. Proceedings of EMNLP 2024: Industry Track. https://arxiv.org/abs/2408.02442

[12] Schick, T., Dwivedi-Yu, J., Dessì, R., Raileanu, R., Lomeli, M., Zettlemoyer, L., Cancedda, N., & Scialom, T. (2023). Toolformer: Language models can teach themselves to use tools. NeurIPS 2023. https://arxiv.org/abs/2302.04761

[13] Perez, F., & Ribeiro, I. (2022). Ignore previous prompt: Attack techniques for language models. NeurIPS ML Safety Workshop. https://arxiv.org/abs/2211.09527

[14] Greshake, K., Abdelnabi, S., Mishra, S., Endres, C., Holz, T., & Fritz, M. (2023). Not what you’ve signed up for: Compromising real-world LLM-integrated applications with indirect prompt injection. AISec ’23. https://arxiv.org/abs/2302.12173

[15] OWASP Foundation (2025). OWASP Top 10 for Large Language Model Applications. https://genai.owasp.org/llm-top-10/

[16] Röttger, P., Kirk, H., Vidgen, B., Attanasio, G., Bianchi, F., & Hovy, D. (2024). XSTest: A test suite for identifying exaggerated safety behaviours in large language models. NAACL 2024, 5377–5400. https://aclanthology.org/2024.naacl-long.301

[17] Gehman, S., Gururangan, S., Sap, M., Choi, Y., & Smith, N. A. (2020). RealToxicityPrompts: Evaluating neural toxic degeneration in language models. Findings of EMNLP 2020. https://arxiv.org/abs/2009.11462

[18] Inan, H., et al. (2023). Llama Guard: LLM-based input-output safeguard for human-AI conversations. arXiv:2312.06674. https://arxiv.org/abs/2312.06674

[19] Zheng, L., et al. (2023). Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. NeurIPS 2023 Datasets and Benchmarks. https://arxiv.org/abs/2306.05685

[20] National Institute of Standards and Technology (2023). Artificial Intelligence Risk Management Framework (AI RMF 1.0), NIST AI 100-1. https://doi.org/10.6028/NIST.AI.100-1

[21] Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830.

## Comparison models

[22] Google. Gemini API models documentation (gemini-flash-latest and gemini-3.5-flash, as recorded in the Pharma AS01 comparison). https://ai.google.dev/gemini-api/docs/models

[23] OpenAI (2025). gpt-oss-120b & gpt-oss-20b model card. arXiv:2508.10925; served through Groq, https://console.groq.com/docs/models (Finance comparison).

[24] OpenAI (2025). GPT-5 system card. https://openai.com/index/gpt-5-system-card/ ; GPT-5-mini accessed through Azure OpenAI (Tech comparison).

## Data sources

[25] Greenhouse Job Board API (public, unauthenticated GET). https://developers.greenhouse.io/job-board.html. Pharma boards (retrieved 2026-09-20): arrowheadpharmacareers, beamtherapeutics, kymeratherapeutics, recursionpharmaceuticals, relaytherapeutics, revolutionmedicines, tesseratherapeutics, ultragenyxpharmaceutical. Finance boards (retrieved 2026-09-19): Jane Street, Carta, Jump Trading, Brex, Point72, Virtu Financial, Stripe, Gemini, IMC Trading, Mercury, AQR Capital Management, Affirm, Robinhood, SoFi, Betterment, Coinbase, Chime, Upstart.

[26] Lever Postings API. https://github.com/lever/postings-api. Finance boards: Anchorage Digital, Alloy (retrieved 2026-09-19).

[27] City of New York. NYC Jobs (dataset kpav-sd4t), NYC Open Data, used under the NYC Open Data Terms of Use. https://data.cityofnewyork.us/City-Government/Jobs-NYC-Postings/kpav-sd4t

[28] Public employer career pages for the Tech corpus (retrieved 2026-09-18): 15 Greenhouse, 15 Workday, 11 Ashby, 3 SmartRecruiters, 3 iCIMS, 2 Eightfold, 2 Lever, 1 BambooHR and 20 direct employer sites; every source and its terms note is listed in sector_sources/tech/sources.csv.

## Team artifacts and course materials

[29] Verma, S. (2026). Pharma AS01 corpus, labels, predictions and metrics; AS02 LLMBox-integrated evaluation and prototype timing records. Supplied project artifacts.

[30] Li, C.-y. (2026). Finance AS01 corpus, reference labels, README and memo. assignment1_chihyul3.zip. Reported numerical results are not independently reproduced here.

[31] Sun, W. (2026). Tech AS01 corpus, labels, README and report. 95820-assignment1-yiqings2.zip. Reported numerical results are not independently reproduced here.

[32] Carnegie Mellon University, 95820 Applications of NLX and LLM (Fall 2026). Assignment 1, Assignment 2 and Final Project instructions.

[33] Sun, W. (2026). Tech AS02 LLMBox runtime, posting-level split, 70 safety probes, all run outputs and metrics, and a rerun of C1 and C4. AS02_teammates/tech_william_sun/ in the team repository; recomputed by verification/verify_tech_as02.py.

## Appendix files

| Required evidence | Location in the supporting package |
| --- | --- |
| Original data | datasets/original_combined.jsonl and sector_sources/*/corpus.jsonl |
| Development and evaluation | datasets/pharma_as02/ plus the three AS01 labeled-record subsets |
| Chatlogs and experiment metrics | prospective_rerun/runs/policy_first_gpu/; sector_sources/pharma/as01_results/ |
| Reported teammate metrics | metrics/teammate_reported_results.jsonl with provenance status |
| Custom API code | LLM_API_Code.zip; source under pharma_api/project/ and pharma_api/llmbox/ |
| Corpus and integrity audits | metrics/corpus_audit.json; prospective_rerun/RESULTS_AUDIT.json |
| Timing evidence | prototype_cost/ |
| Data source index | sector_sources/*/sources.csv (650 records; per-source terms notes); references [25]–[28] |
| AS02 report as submitted | documents/AS02_Report_as_submitted.pdf (the version uploaded to Canvas) |
| Initial and revised policies | prospective_rerun/governance/POLICY_V1.md and POLICY_V2.md |
| Shared disclosure and available records | AI_USAGE.md and assistance/ |
| Tech AS02 code, data, runs and rerun | AS02_teammates/tech_william_sun/; recomputed by verification/verify_tech_as02.py |

Checksums and a dataset manifest are included. Missing teammate logs remain identified. AI_USAGE.md is the shared disclosure appendix. Original submissions remain unchanged.