#!/usr/bin/env python3
"""Draw the evaluation sample and write a human_labels.jsonl template to fill in.

    python make_human_labels.py --n 25

The template's fields come from taxonomy.json, so it always matches whatever
your group decided to extract. Each line gets an empty `human_label` and a
`_preview` of the document; fill in `human_label` by hand -- that is the point
of the exercise, so do not let the model do it. `_preview` is ignored by
extraction.py; delete it or leave it.

Nothing is overwritten: if --out already exists, pass --force.
"""

from __future__ import annotations

import argparse
import json
import hashlib
import random
from pathlib import Path

from schema import load_corpus
from extraction import DEFAULT_TAXONOMY, apply_taxonomy

BLANKS = {"list": [], "number": None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", default="corpus.jsonl")
    parser.add_argument("--out", default="human_labels.jsonl")
    parser.add_argument("--taxonomy", default=str(DEFAULT_TAXONOMY))
    parser.add_argument("--n", type=int, default=25, help="20-30 per the brief")
    parser.add_argument("--seed", type=int, default=0,
                        help="record it, so the sample is reproducible")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not 20 <= args.n <= 30: raise SystemExit("Choose 20-30 records.")
    target = Path(args.out)
    if target.exists() and not args.force:
        raise SystemExit(f"{target} already exists. Pass --force to replace it.")

    fields = apply_taxonomy(args.taxonomy)["fields"]
    records = load_corpus(args.corpus, validate=False).records
    if len(records) < args.n:
        raise SystemExit(f"Only {len(records)} records in {args.corpus}.")

    blank = {name: BLANKS.get(spec.get("kind", "label"), "") for name, spec in fields.items()}
    sample = random.Random(args.seed).sample(records, args.n)
    with open(target, "w", encoding="utf-8") as handle:
        for record in sorted(sample, key=lambda r: r["doc_id"]):
            handle.write(json.dumps({
                "doc_id": record["doc_id"],
                "human_label": dict(blank),
                "annotation_status": "pending",
                "annotation_origin": "human",
                "annotator": "",
                "_preview": (record.get("raw_text") or "")[:400],
            }, ensure_ascii=False) + "\n")

    target.with_name('evaluation_sample.json').write_text(json.dumps({
        'seed': args.seed, 'n': args.n, 'doc_ids': sorted(r['doc_id'] for r in sample),
        'corpus_sha256': hashlib.sha256(Path(args.corpus).read_bytes()).hexdigest(),
        'fields_sha256': hashlib.sha256(json.dumps(fields,sort_keys=True).encode()).hexdigest()
    },indent=2))
    print(f"Wrote {args.n} records to {target} (seed {args.seed}). Fill in:")
    for name, spec in fields.items():
        allowed = (", ".join(spec.get("values") or []) or spec.get("description")
                   or f"open {spec.get('kind', 'label')} field")
        print(f"  {name:<14} {allowed}")


if __name__ == "__main__":
    main()
