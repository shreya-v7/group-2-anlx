"""Corpus access for SimplifyJobs: prompts, dev/eval split, reference labels.

Only `raw_text` and `table_json` are sent to the model (Part A data boundary).
`metadata` is scraped posting-level information and is used ONLY as the
reference answer, never as model input.
"""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "simplifyjobs"
CORPUS_PATH = DATA_DIR / "corpus.jsonl"
TAXONOMY_PATH = DATA_DIR / "taxonomy.json"
HUMAN_LABELS_PATH = DATA_DIR / "human_labels.jsonl"
SPLIT_DIR = DATA_DIR / "splits"

# Same truncation as AS01 build_prompt, so inputs match the AS01 evaluation.
MAX_TEXT_CHARS = 3000
MAX_TABLE_CHARS = 800


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def load_human_labels(path: str | Path = HUMAN_LABELS_PATH) -> dict[str, dict]:
    if not Path(path).exists():
        return {}
    return {row["doc_id"]: row["human_label"] for row in read_jsonl(path)}


def user_prompt(record: dict) -> str:
    """The model input: document text, then the table if present (AS01 format)."""
    prompt = f"Document {record['doc_id']}:\n{(record.get('raw_text') or '')[:MAX_TEXT_CHARS]}"
    table = record.get("table_json")
    if table:
        prompt += (
            "\n\nTable:\n" + json.dumps(table, ensure_ascii=False)[:MAX_TABLE_CHARS]
        )
    return prompt


def source_text(record: dict) -> str:
    """Untruncated text used for grounding checks (superset of the prompt)."""
    table = record.get("table_json")
    table_text = json.dumps(table, ensure_ascii=False) if table else ""
    return f"{record.get('raw_text') or ''}\n{table_text}"


# ---------------------------------------------------------------------------
# Reference labels from scraped metadata
# ---------------------------------------------------------------------------

_INTERN = re.compile(r"\bintern(ship)?\b", re.IGNORECASE)


def reference_labels(record: dict) -> dict:
    """Reference answers derived from metadata.

    seniority: every posting in the corpus is new-grad/entry-level by
    construction (AS01 corpus scope), so the reference is "entry" unless the
    role title says intern. location_type is only present for 99/250 records.
    """
    meta = record.get("metadata", {})
    title = meta.get("role_title", "")
    return {
        "company": meta.get("company"),
        "seniority": "intern" if _INTERN.search(title) else "entry",
        "location_type": meta.get("location_type"),
        "role_title": title,
        "document_type": meta.get("document_type"),
    }


# ---------------------------------------------------------------------------
# Development / evaluation split
# ---------------------------------------------------------------------------


def split_by_posting(
    records: list[dict],
    force_eval_ids: set[str],
    eval_fraction: float = 0.5,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Split at the posting level (source_url), not the record level.

    One posting is cut into several records (overview, qualifications,
    compensation). Splitting by record would put fragments of the same posting
    on both sides and leak company names and skills from dev into eval.
    Postings containing an AS01 human-labeled record are forced into eval so
    Part C can also be scored against human labels.
    """
    postings: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        postings[record["source_url"]].append(record)

    forced = [
        url
        for url, recs in postings.items()
        if any(r["doc_id"] in force_eval_ids for r in recs)
    ]
    rest = sorted(url for url in postings if url not in set(forced))
    random.Random(seed).shuffle(rest)

    target = eval_fraction * len(records)
    eval_urls = list(forced)
    n_eval = sum(len(postings[u]) for u in eval_urls)
    for url in rest:
        if n_eval >= target:
            break
        eval_urls.append(url)
        n_eval += len(postings[url])

    eval_set = set(eval_urls)
    dev = [r for r in records if r["source_url"] not in eval_set]
    ev = [r for r in records if r["source_url"] in eval_set]
    return dev, ev


def sample_records(
    records: list[dict], n: int, seed: int = 42, must_include: set[str] | None = None
) -> list[dict]:
    """Sample n records, always keeping `must_include` doc_ids. Order is stable."""
    must_include = must_include or set()
    kept = [r for r in records if r["doc_id"] in must_include]
    pool = [r for r in records if r["doc_id"] not in must_include]
    if len(kept) > n:
        raise ValueError(f"{len(kept)} required records exceed sample size {n}")
    chosen = kept + random.Random(seed).sample(pool, min(n - len(kept), len(pool)))
    return sorted(chosen, key=lambda r: r["doc_id"])
