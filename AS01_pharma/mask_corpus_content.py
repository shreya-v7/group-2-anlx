#!/usr/bin/env python3
"""Run the required Phi-4-mini-instruct PII/PHI/FIN/CONF masking pass over
corpus.jsonl's actual content fields (raw_text, table_json), per Assignment 1
section 2.1.

This reuses the exact detection schema, system prompt, and span-acceptance
logic from mask_sensitive.py (Phi proposes spans; a span is only accepted if
it appears character-for-character in the source, so the model cannot rewrite
data) but is scoped to raw_text and table_json only -- not source_url,
retrieved_at, license_note, doc_id, or modality, which are corpus plumbing,
not source content, and were already manually reviewed as non-sensitive.

Usage (from the assignment1/ folder, with the venv that has torch+transformers
and local model weights, matching README's "Reproduce the actual run" steps):

    python mask_corpus_content.py \
        --input corpus.jsonl \
        --output corpus.masked.jsonl \
        --report out/masking_report.json \
        --model-path ../.models/phi4-mini-instruct \
        --device mps

Review corpus.masked.jsonl and out/masking_report.json before replacing
corpus.jsonl. If out/masking_report.json shows zero accepted spans, no
corpus change is needed -- the run itself, plus this report, is the
required evidence that the Phi-based screen was performed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from local_model import PhiClient
from mask_sensitive import CATEGORIES, DETECTION_SCHEMA, SYSTEM, mask_json, mask_value


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="corpus.jsonl")
    ap.add_argument("--output", default="corpus.masked.jsonl")
    ap.add_argument("--report", default="out/masking_report.json")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    ap.add_argument("--limit", type=int, default=None,
                     help="only process the first N records (for a quick test run)")
    args = ap.parse_args()

    records = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        records = records[: args.limit]

    print(f"[info] loaded {len(records)} records from {args.input}", file=sys.stderr)
    phi = PhiClient(args.model_path, device=args.device)

    report: list = []
    cache: dict = {}
    t0 = time.time()
    out_path = Path(args.output)
    with out_path.open("w", encoding="utf-8") as out_f:
        for i, rec in enumerate(records, 1):
            before_len = len(report)
            rec["raw_text"] = mask_value(rec.get("raw_text") or "", phi, report, cache)
            if rec.get("table_json") is not None:
                rec["table_json"] = mask_json(rec["table_json"], phi, report, cache, None)
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
            new_hits = len(report) - before_len
            elapsed = time.time() - t0
            print(f"[{i}/{len(records)}] {rec.get('doc_id')} "
                  f"(+{new_hits} candidate span(s), {elapsed:.0f}s elapsed)", file=sys.stderr)

    accepted = [r for r in report if r["accepted"]]
    by_cat: dict = {}
    for r in accepted:
        by_cat[r["category"]] = by_cat.get(r["category"], 0) + 1

    print(f"\n[done] {len(records)} records processed in {time.time() - t0:.0f}s", file=sys.stderr)
    print(f"[done] {len(accepted)} span(s) accepted and masked "
          f"({len(report) - len(accepted)} model-proposed span(s) rejected as not verbatim).",
          file=sys.stderr)
    print(f"[done] by category: {by_cat or '(none)'}", file=sys.stderr)
    print(f"[done] wrote {out_path}", file=sys.stderr)

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps({
        "scope": ["raw_text", "table_json"],
        "records_processed": len(records),
        "elapsed_s": round(time.time() - t0, 1),
        "spans_accepted": len(accepted),
        "spans_rejected_not_verbatim": len(report) - len(accepted),
        "accepted_by_category": by_cat,
        "detail": report,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] audit report: {args.report}", file=sys.stderr)
    print("[done] review corpus.masked.jsonl and the report before replacing corpus.jsonl.",
          file=sys.stderr)


if __name__ == "__main__":
    main()
