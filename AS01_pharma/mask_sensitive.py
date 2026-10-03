#!/usr/bin/env python3
"""Mask sensitive spans with Phi-4-mini-instruct before they enter the corpus.

    python mask_sensitive.py --input raw/notes.txt   --output raw/notes.masked.txt
    python mask_sensitive.py --input raw/table.csv   --output raw/table.masked.csv --report audit.json
    python mask_sensitive.py --input corpus.jsonl --output corpus.masked.jsonl

Phi proposes the spans; this script only accepts a span if it appears in the
input character-for-character, so the model cannot rewrite your data. Accepted
spans are replaced with the assignment's placeholders:

    <PII>   names, personal emails, phone numbers, addresses, government ids
    <PHI>   diagnoses, treatments, medications, medical record information
    <FIN>   account numbers, card numbers, private financial identifiers
    <CONF>  non-public organizational, contractual or proprietary information

Add --hash to write <PII>a1b2c3... instead, so the same value is traceable
across records without being readable. Read the masked output before you keep
it: on a 3.8B model, both misses and false positives are normal.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import secrets
import sys
from pathlib import Path

from local_model import PhiClient

CATEGORIES = ["PII", "PHI", "FIN", "CONF"]

DETECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "enum": CATEGORIES},
                    "text": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["category", "text"],
            },
        }
    },
    "required": ["entities"],
}

SYSTEM = (
    "You are a data-privacy classifier. Find every substring of the input that is:\n"
    "- PII: personal names, personal emails, phone numbers, home addresses, "
    "government identifiers, dates of birth, IP addresses\n"
    "- PHI: diagnoses, treatments, medications, medical record or insurance identifiers\n"
    "- FIN: account, routing or card numbers, private financial identifiers\n"
    "- CONF: non-public organizational, contractual or proprietary information\n\n"
    "Copy each span EXACTLY as it appears in the input -- never paraphrase, "
    "summarize or invent text. If nothing is sensitive, return an empty list."
)


def digest(value: str, salt: str) -> str:
    return hashlib.sha256((salt + value).encode("utf-8")).hexdigest()[:16]


def mask_value(value: str, phi: PhiClient, report: list, cache: dict,
               salt: str | None = None) -> str:
    """Mask one string. Values are cached so repeats cost nothing."""
    if not isinstance(value, str) or len(value.strip()) < 3:
        return value
    if value not in cache:
        response = phi.structured(value, DETECTION_SCHEMA, system=SYSTEM)
        cache[value] = (response.parsed or {}).get("entities", []) if response.ok else []
        if response.violation:
            print(f"[warn] unusable model output, value left untouched: "
                  f"{response.violation}", file=sys.stderr)

    masked = value
    # Longest first, so a short span nested in a longer one cannot corrupt a
    # placeholder that has already been written.
    for entity in sorted(cache[value], key=lambda e: len(e.get("text", "")), reverse=True):
        span, category = entity.get("text", ""), entity.get("category")
        if not span or category not in CATEGORIES:
            continue
        found = span in masked
        report.append({
            "category": category,
            "accepted": found,
            "preview": span[:2] + "***",
            "reason": entity.get("reason", ""),
        })
        if found:
            tag = f"<{category}>" + (digest(span, salt) if salt else "")
            masked = masked.replace(span, tag)
    return masked


def mask_json(data, phi, report, cache, salt):
    if isinstance(data, dict):
        return {k: mask_json(v, phi, report, cache, salt) for k, v in data.items()}
    if isinstance(data, list):
        return [mask_json(v, phi, report, cache, salt) for v in data]
    if isinstance(data, str):
        return mask_value(data, phi, report, cache, salt)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--format", default="auto", choices=["auto", "txt", "csv", "json", "jsonl"])
    parser.add_argument("--report", default=None, help="write an audit report here")
    parser.add_argument("--hash", action="store_true",
                        help="append a salted hash to each placeholder")
    parser.add_argument("--salt", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    args = parser.parse_args()

    source, target = Path(args.input), Path(args.output)
    if not source.exists():
        raise SystemExit(f"No such file: {source}")

    salt = None
    if args.hash:
        salt = args.salt or secrets.token_hex(8)
        print(f"[info] salt for this run: {salt} (save it to reproduce these hashes)",
              file=sys.stderr)

    fmt = args.format
    if fmt == "auto":
        fmt = {".csv": "csv", ".json": "json", ".jsonl": "jsonl"}.get(source.suffix.lower(), "txt")

    phi = PhiClient(args.model_path, device=args.device)
    report: list = []
    cache: dict = {}

    if fmt == "csv":
        with open(source, newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        masked = [[mask_value(cell, phi, report, cache, salt) for cell in row] for row in rows]
        with open(target, "w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerows(masked)

    elif fmt == "json":
        data = json.loads(source.read_text(encoding="utf-8"))
        target.write_text(json.dumps(mask_json(data, phi, report, cache, salt), indent=2,
                                     ensure_ascii=False), encoding="utf-8")

    elif fmt == "jsonl":
        lines = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
        with open(target, "w", encoding="utf-8") as handle:
            for record in lines:
                handle.write(json.dumps(mask_json(record, phi, report, cache, salt),
                                        ensure_ascii=False) + "\n")

    else:
        masked_lines = [mask_value(line, phi, report, cache, salt)
                        for line in source.read_text(encoding="utf-8").splitlines()]
        target.write_text("\n".join(masked_lines), encoding="utf-8")

    accepted = sum(1 for r in report if r["accepted"])
    print(f"[done] wrote {target}: {accepted} span(s) masked, "
          f"{len(report) - accepted} model-proposed span(s) rejected as not verbatim.",
          file=sys.stderr)
    print("[done] review the output before adding it to corpus.jsonl.", file=sys.stderr)

    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                     encoding="utf-8")
        print(f"[done] audit report: {args.report}", file=sys.stderr)


if __name__ == "__main__":
    main()
