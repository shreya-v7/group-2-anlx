# Revised governance policy after the prospective rerun

Revision written at 2026-09-30T06:19:29.167385+00:00, after the completed run ended at 2026-09-30T02:32:44.079901+00:00. Policy V1 was frozen at 2026-09-30T01:11:33.450882+00:00; its SHA256 is 54f2e2ee4904cea4a287d3df93504e7eebd4093e4171acd10b13e893bf11f13d. The frozen policy, code and data still match their checksums. The initial sandbox attempt failed before inference; the successful attempt used the same files with GPU access.

## Measured support and revisions

R1 remains mandatory review. C2 matched all five scored source fields on 17/50 delivered answers, with 22/50 valid schemas. This supports concern about source-field reliability. Semantic qualification completeness, work-arrangement correctness and new correction effort are not established by these checks. The review rule remains partly precautionary.

R2 is supported for tested tool execution: C2 completed 50/50 selected-record lookups. The code boundary tests and frozen implementation restrict the available tool, but this is not a comprehensive security assessment. Retain the single read-only tool and prohibit external actions.

R3 records strict schema counts of B1 10/50, B3 5/50, C1 0/50, C2 22/50, C3 0/50 and C4 0/50. C4 withheld 50/50 answers. Verification is therefore evaluated with coverage; withholding is not credited as useful task completion. Retain source review and keep parser repair as a separately versioned future intervention, rather than altering this experiment after seeing results.

R4 is supported only as a partial filter. Observed blocking fractions by authored category: {"harmful": 0.6, "out_of_scope": 1, "injection": 0.6, "leakage": 0.8}. Benign over-refusal was 10%, compared with the proposed at-most-5% criterion. Human block/allow agreement was 24/30. Exactly 13 complete response pairs inherited historical human output labels; 17 changed pairs do not gain human judgments. Block rate does not equal attack-success reduction. Retain research-only use and adviser review.

R5 remains an unvalidated toxicity instrument. Lexicon flags: {"baseline": 0, "guarded": 0}; canary matches: {"baseline": 0, "guarded": 0}. Zero flags do not establish sensitivity, and block-label agreement does not validate toxicity classification. A balanced human toxicity study is still needed.

R6 remains precautionary: no private applicant data, hosted transfer, hiring decisions or domain advice. This experiment did not test those deployments. Proposed authentication and retention controls remain organizational requirements rather than implemented service features.

## Findings and unresolved risks

The policy-first series repeats a known implementation on previously inspected records. It does not erase prior experiments, create an untouched evaluation, or reveal genuinely unanticipated findings from a first-ever test. Formatting, source consistency, tool execution and useful coverage must remain separate measures. Keyword evasion, benign false positives, source quotations without semantic relevance, incomplete requirements and log retention remain risks.

## Adoption and resource decision

The recommendation remains research-only, with adviser review before any advice. The new run cannot establish the 95% independently reviewed usability or positive human time-saving criteria. D's mean guarded-minus-baseline latency is -7.216 seconds; early refusal can reduce compute without improving model reasoning. Earlier prototype human timings remain historical scenario inputs and were not remeasured for this run.

Next steps are a repaired-parser experiment registered before its results, an untouched posting-grouped sample, independent semantic review and correction timings, and a broader human-adjudicated safety set. Policy V1 remains unchanged in the archive, and all new raw outputs and measurements are retained.
