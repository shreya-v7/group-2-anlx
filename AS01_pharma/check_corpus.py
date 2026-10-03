#!/usr/bin/env python3
"""Validate an Assignment 1 corpus before you hand it in.

    python check_corpus.py corpus.jsonl
    python check_corpus.py corpus.jsonl --sources sources.csv --strict
    python check_corpus.py corpus.jsonl --stats

Exit codes:
    0   passes
    1   fails a hard requirement
    2   could not read the file at all
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

from schema import MAX_DOCS, MIN_DOCS, MIN_TABULAR_SHARE, is_tabular, load_corpus

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"
SOURCES_COLUMNS = ("source_url", "source_name", "retrieved_at", "license_note", "record_count")
_DOMAIN = re.compile(r"https?://([^/]+)")


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def add(self, status: str, check: str, detail: str = "") -> None:
        self.rows.append((status, check, detail))

    @property
    def failed(self) -> bool:
        return any(status == FAIL for status, _, _ in self.rows)

    @property
    def warned(self) -> bool:
        return any(status == WARN for status, _, _ in self.rows)

    def render(self) -> None:
        width = max((len(check) for _, check, _ in self.rows), default=10)
        print("\n" + "=" * 72)
        print("CORPUS VALIDATION")
        print("=" * 72)
        for status, check, detail in self.rows:
            print(f"  [{status}]  {check.ljust(width)}   {detail}")
        print("=" * 72)


def domain_of(url: str) -> str | None:
    match = _DOMAIN.match(url or "")
    return match.group(1).lower() if match else None


def check_sources_csv(path: Path, report: Report, corpus_domains: set[str], n_records: int) -> None:
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        report.add(FAIL, "sources.csv", f"could not read: {exc}")
        return
    if not rows:
        report.add(FAIL, "sources.csv", "is empty")
        return

    missing = set(SOURCES_COLUMNS) - set(rows[0])
    if missing:
        report.add(FAIL, "sources.csv columns", f"missing: {', '.join(sorted(missing))}")
        return
    report.add(PASS, "sources.csv columns", f"{len(rows)} source(s) listed")

    thin = [r["source_url"] for r in rows if len((r.get("license_note") or "").strip()) < 10]
    if thin:
        report.add(WARN, "sources.csv licensing",
                   f"{len(thin)} source(s) have a license note under 10 chars: {thin[0][:50]}")

    listed = {d for d in (domain_of(r["source_url"]) for r in rows) if d}
    undocumented = corpus_domains - listed
    if undocumented:
        report.add(FAIL, "source coverage",
                   f"{len(undocumented)} domain(s) in corpus but not in sources.csv: "
                   + ", ".join(sorted(undocumented)[:4]))
    else:
        report.add(PASS, "source coverage", "every corpus domain is documented")

    try:
        total = sum(int(r["record_count"]) for r in rows)
    except (TypeError, ValueError):
        report.add(WARN, "sources.csv record_count", "not all values are integers")
    else:
        status = PASS if total == n_records else WARN
        report.add(status, "sources.csv record_count",
                   f"sums to {total}, corpus has {n_records}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("corpus", nargs="?", default="corpus.jsonl")
    parser.add_argument("--sources", default=None, help="path to sources.csv")
    parser.add_argument("--strict", action="store_true", help="treat warnings as failures")
    parser.add_argument("--stats", action="store_true", help="also print the full characterization")
    parser.add_argument("--max-shown", type=int, default=25)
    args = parser.parse_args()

    path = Path(args.corpus)
    if not path.exists():
        print(f"No such file: {path}", file=sys.stderr)
        return 2

    result = load_corpus(path)
    records = result.records
    report = Report()

    # -- record-level ---------------------------------------------------
    if result.problems:
        report.add(FAIL, "record schema", f"{len(result.problems)} problem(s), listed below")
    else:
        report.add(PASS, "record schema", f"all {len(records)} records conform")

    # -- corpus size ----------------------------------------------------
    n = len(records)
    if n < MIN_DOCS:
        report.add(FAIL, "corpus size", f"{n} records, minimum is {MIN_DOCS}")
    elif n > MAX_DOCS:
        report.add(WARN, "corpus size", f"{n} records, guideline maximum is {MAX_DOCS}")
    else:
        report.add(PASS, "corpus size", f"{n} records")

    # -- tabular share --------------------------------------------------
    domains: set[str] = set()
    if n:
        tabular = sum(1 for r in records if is_tabular(r))
        share = tabular / n
        report.add(PASS if share >= MIN_TABULAR_SHARE else FAIL, "tabular share",
                   f"{share:.1%} ({tabular}/{n}), minimum is {MIN_TABULAR_SHARE:.0%}")

        # -- everything that needs the stats module ---------------------
        from corpus_stats import describe, print_report

        stats = describe(records)
        domains = {d for d in (domain_of(r.get("source_url")) for r in records) if d}

        if stats["n_sources"] < 2:
            report.add(WARN, "source diversity", "only one source domain")
        else:
            report.add(PASS, "source diversity",
                       f"{stats['n_sources']} domains, largest "
                       f"{stats['source_concentration']:.0%} of corpus")
            if stats["source_concentration"] > 0.8:
                report.add(WARN, "source concentration",
                           f"one domain is {stats['source_concentration']:.0%} of the corpus")

        rate = stats["near_duplicate_rate"]
        if rate > 0.20:
            report.add(FAIL, "near-duplicates", f"{rate:.1%} of records, threshold is 20%")
        elif rate > 0.05:
            report.add(WARN, "near-duplicates", f"{rate:.1%} of records")
        else:
            report.add(PASS, "near-duplicates", f"{rate:.1%} of records")

        if stats["distinct_license_notes"] < 2 and stats["n_sources"] > 1:
            report.add(WARN, "license notes",
                       "every record shares one license note across multiple domains")
        else:
            report.add(PASS, "license notes", f"{stats['distinct_license_notes']} distinct note(s)")

        median_chars = stats["char_length"].get("median", 0)
        if median_chars < 200:
            report.add(WARN, "document length",
                       f"median is {median_chars:.0f} characters; these may be too short to retrieve on")

    # -- sources.csv ----------------------------------------------------
    sources_path = Path(args.sources) if args.sources else path.parent / "sources.csv"
    if sources_path.exists():
        check_sources_csv(sources_path, report, domains, n)
    else:
        report.add(FAIL, "sources.csv", f"not found at {sources_path}")

    # -- output ---------------------------------------------------------
    report.render()

    if result.problems:
        print(f"\nRECORD PROBLEMS ({len(result.problems)} total, showing up to {args.max_shown}):\n")
        for problem in result.problems[: args.max_shown]:
            print(f"  {problem}")
        if len(result.problems) > args.max_shown:
            print(f"  ... and {len(result.problems) - args.max_shown} more")

    if args.stats and records:
        print()
        print_report(stats)

    failed = report.failed or (args.strict and report.warned)
    print("\n" + ("SUBMISSION WOULD FAIL" if failed else "SUBMISSION LOOKS VALID"))
    if report.warned and not failed:
        print("Warnings are not blocking, but each one is something the memo should address.")
    print()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
