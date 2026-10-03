# Follow-up F1: fence-tolerant parser (pre-registered October 3, 2026, before any model run)

## Why
In the frozen policy-first series, C1 and C4 delivered 0/50 because every answer failed strict JSON
parsing: 46 outputs were one complete Markdown fence and 4 were unclosed (token cap). A post-hoc check on
those same outputs found 43/50 schema-valid and 31/50 passing every C4 check once the fence was removed.
A post-hoc result on inspected outputs is not evidence, so this follow-up tests the fix on new records.

## What changes
Only the parser (`project/fence_parser.py`): if the whole output is exactly one complete fence, validate
the inner text; otherwise validate as-is (unclosed, prose-wrapped or multiple fences still fail).
Model, prompt, schema, LLMBox `run_structured_output`, C4 checks (schema + five source fields + verbatim
quotes) and B1 settings (T 0.2, top_p 0.8, 256 tokens, top_k 0, repetition penalty 1) are unchanged.
The frozen AS02 code, data and runs are not modified.

## Records
Evaluation records 51–100 (`evaluation.jsonl[50:100]`, seeds 870–919). They appear in no previous model run
output (checked against every saved response file). They were part of the corpus and may have been read
during collection, so this is new to the API but not a blind third-party holdout. Single run, n = 50.

## Outcomes (computed from the same generations)
1. Strict schema validity (old parser), for comparison.
2. Fence-tolerant schema validity.
3. C4 delivered: tolerant-valid AND all five source fields exact AND every quote verbatim.

## Decision rule, fixed now
- The fix "works" if tolerant validity >= 35/50 and C4 delivered >= 20/50.
- It "partly works" if tolerant validity >= 35/50 but C4 delivered < 20/50 (format fixed, content still fails).
- It "does not work" if tolerant validity < 35/50.
Whatever the result, every output is saved and reported. Delivered answers still require adviser review;
this follow-up does not change the AS02 recommendation unless an adviser-approval study is also run.
