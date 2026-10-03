#!/usr/bin/env python3
"""Example: turn raw source files into corpus.jsonl records.

    # one text document -> one text record
    python example_extraction.py --text article.txt \
        --source-url https://example.org/article/1 \
        --license-note "Public webpage; see site terms of use." \
        --id-prefix news --out corpus.jsonl

    # one text document -> several records, split on Markdown-style headings
    python example_extraction.py --text report.md --split-on "^#+ " ...

    # a CSV table -> one table record (the whole table as table_json)
    python example_extraction.py --csv budget.csv ...

    # text plus a table from the same page -> one mixed record
    python example_extraction.py --text page.txt --csv page_table.csv ...

Records are appended to --out, so you can build the corpus source by source.
This is an example of the *shape*; your own collection code will differ, and
if a source contains sensitive information run mask_sensitive.py on the raw
files before this step.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from schema import validate_record

# ---------------------------------------------------------------------------
# Building one record
# ---------------------------------------------------------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def make_record(doc_id: str, source_url: str, license_note: str, *,
                raw_text: str | None = None, table_json: list | dict | None = None,
                metadata: dict | None = None, retrieved_at: str | None = None) -> dict:
    """Fill in the common fields. Modality follows from what you pass in."""
    if table_json and raw_text:
        modality = "mixed"
    elif table_json:
        modality = "table"
    else:
        modality = "text"
    return {
        "doc_id": doc_id,
        "source_url": source_url,
        "retrieved_at": retrieved_at or now_iso(),
        "modality": modality,
        "raw_text": raw_text,
        "table_json": table_json or None,
        "license_note": license_note,
        "metadata": metadata or {},
    }


# ---------------------------------------------------------------------------
# Reading raw inputs
# ---------------------------------------------------------------------------

def read_csv_table(path: Path) -> list[dict]:
    """A CSV as a list of row dicts -- the simplest normalized table shape.

    Numbers stay strings here on purpose: decide per column what to coerce,
    and record that decision in your README.
    """
    with open(path, newline="", encoding="utf-8") as handle:
        return [{k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
                for row in csv.DictReader(handle)]


def split_sections(text: str, heading_pattern: str) -> list[tuple[str, str]]:
    """Split a document into (heading, body) pairs on lines matching the pattern.

    Text before the first heading becomes a section with an empty heading.
    """
    pattern = re.compile(heading_pattern, re.MULTILINE)
    sections, last_heading, last_end = [], "", 0
    for match in pattern.finditer(text):
        body = text[last_end:match.start()].strip()
        if body:
            sections.append((last_heading, body))
        line_end = text.find("\n", match.end())
        line_end = len(text) if line_end == -1 else line_end
        last_heading = text[match.end():line_end].strip()
        last_end = line_end
    body = text[last_end:].strip()
    if body:
        sections.append((last_heading, body))
    return sections


# ---------------------------------------------------------------------------
# The example run
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--text", type=Path, help="a plain-text or markdown source file")
    parser.add_argument("--csv", type=Path, help="a CSV table from the same source")
    parser.add_argument("--split-on", default=None,
                        help="regex for heading lines; each section becomes its own record")
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--license-note", required=True)
    parser.add_argument("--id-prefix", default="doc")
    parser.add_argument("--group-topic", default="")
    parser.add_argument("--subtopic", default="")
    parser.add_argument("--document-type", default="")
    parser.add_argument("--out", type=Path, default=Path("corpus.jsonl"))
    args = parser.parse_args()

    if not args.text and not args.csv:
        raise SystemExit("Give --text, --csv, or both.")

    existing = sum(1 for _ in open(args.out, encoding="utf-8")) if args.out.exists() else 0
    metadata = {"group_topic": args.group_topic, "subtopic": args.subtopic,
                "document_type": args.document_type, "source_file": (args.text or args.csv).name}

    text = args.text.read_text(encoding="utf-8") if args.text else None
    table = read_csv_table(args.csv) if args.csv else None
    records = []

    if text and args.split_on:
        for heading, body in split_sections(text, args.split_on):
            n = existing + len(records) + 1
            records.append(make_record(f"{args.id_prefix}_{n:03d}", args.source_url,
                                       args.license_note, raw_text=body,
                                       metadata={**metadata, "section": heading}))
        if table:   # the table goes with the document as a whole
            n = existing + len(records) + 1
            records.append(make_record(f"{args.id_prefix}_{n:03d}", args.source_url,
                                       args.license_note, table_json=table,
                                       metadata={**metadata, "section": "table"}))
    else:
        n = existing + 1
        records.append(make_record(f"{args.id_prefix}_{n:03d}", args.source_url,
                                   args.license_note, raw_text=text, table_json=table,
                                   metadata=metadata))

    problems = [p for i, r in enumerate(records) for p in validate_record(r, existing + i + 1)]
    for problem in problems:
        print(f"  ! {problem}")

    with open(args.out, "a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    kinds = {r["modality"] for r in records}
    print(f"Appended {len(records)} record(s) ({', '.join(sorted(kinds))}) to {args.out}"
          + (f" -- {len(problems)} problem(s) above to fix" if problems else ""))


if __name__ == "__main__":
    main()
