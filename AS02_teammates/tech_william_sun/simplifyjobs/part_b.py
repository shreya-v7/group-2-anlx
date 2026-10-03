"""Part B: dataset split and baseline LLMBOX `mode=generate` evaluations.

    python -m simplifyjobs.part_b split
    python -m simplifyjobs.part_b run --run b1
    python -m simplifyjobs.part_b run --run b3
    python -m simplifyjobs.part_b evaluate --run b1
    python -m simplifyjobs.part_b evaluate --run b3
    python -m simplifyjobs.part_b compare
    python -m simplifyjobs.part_b spotcheck --run b3

Both runs use the same 50 development records, the same system prompt
(AS01 v1 + JSON instruction), the same seed, and differ only in
temperature / top_p / max_new_tokens.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys

from simplifyjobs.corpus import (
    ROOT,
    SPLIT_DIR,
    CORPUS_PATH,
    TAXONOMY_PATH,
    load_human_labels,
    read_jsonl,
    sample_records,
    source_text,
    split_by_posting,
    user_prompt,
    write_jsonl,
)
from simplifyjobs.extraction_schema import baseline_system_prompt, build_schema, load_taxonomy
from simplifyjobs.metrics import aggregate, evaluate_record

logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")
log = logging.getLogger("part_b")

RESULTS = ROOT / "results" / "part_b"
SEED = 42
N_SAMPLE = 50
DEFAULT_MODEL = "phi4_instruct"
DEFAULT_MODEL_PATH = "./models/llms/microsoft/Phi-4-mini-instruct"

RUNS = {
    # LLMBOX defaults (conf/schema GenerationConfig).
    "b1": {"temperature": 1.0, "top_p": 0.95, "max_new_tokens": 512},
    # Extraction has one correct answer: sharpen sampling, cap runaway output.
    "b3": {"temperature": 0.2, "top_p": 0.5, "max_new_tokens": 256},
}

TAXONOMY = load_taxonomy(TAXONOMY_PATH)
SCHEMA = build_schema(TAXONOMY["fields"])
SYSTEM_PROMPT = baseline_system_prompt(TAXONOMY)


def generate_messages(record: dict) -> list[dict]:
    """Exactly what Modes.run_generate builds: optional system turn + user turn."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt(record)},
    ]


def respond_generate(engine, record: dict) -> dict:
    """Plain mode=generate: one generation per record."""
    messages = generate_messages(record)
    return {"messages": messages, **engine.generate(messages)}


# ---------------------------------------------------------------------------
# split
# ---------------------------------------------------------------------------


def cmd_split(_args) -> None:
    records = read_jsonl(CORPUS_PATH)
    human_ids = set(load_human_labels())
    dev, ev = split_by_posting(records, human_ids, eval_fraction=0.5, seed=SEED)

    dev_sample = sample_records(dev, N_SAMPLE, seed=SEED)
    eval_sample = sample_records(ev, N_SAMPLE, seed=SEED, must_include=human_ids)

    write_jsonl(SPLIT_DIR / "dev.jsonl", dev)
    write_jsonl(SPLIT_DIR / "eval.jsonl", ev)
    write_jsonl(SPLIT_DIR / "dev_sample50.jsonl", dev_sample)
    write_jsonl(SPLIT_DIR / "eval_sample50.jsonl", eval_sample)

    dev_urls = {r["source_url"] for r in dev}
    eval_urls = {r["source_url"] for r in ev}
    summary = {
        "seed": SEED,
        "unit": "posting (source_url)",
        "records": {"dev": len(dev), "eval": len(ev)},
        "postings": {"dev": len(dev_urls), "eval": len(eval_urls)},
        "posting_overlap": len(dev_urls & eval_urls),
        "samples": {"dev_sample50": len(dev_sample), "eval_sample50": len(eval_sample)},
        "human_labeled_in_eval_sample": sum(r["doc_id"] in human_ids for r in eval_sample),
        "modality": {
            name: {m: sum(r["modality"] == m for r in rows) for m in ("text", "table", "mixed")}
            for name, rows in (("dev_sample50", dev_sample), ("eval_sample50", eval_sample))
        },
    }
    (SPLIT_DIR / "split_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def generation_overrides(run: str, model: str, model_path: str) -> list[str]:
    params = RUNS[run]
    return [
        "mode=generate",
        f"model={model}",
        "model.source=local",
        f"seed={SEED}",
        f"generation.temperature={params['temperature']}",
        f"generation.top_p={params['top_p']}",
        f"generation.max_new_tokens={params['max_new_tokens']}",
    ]


def cmd_run(args) -> None:
    from simplifyjobs.harness import Engine, compose_cfg, run_batch

    sample_path = SPLIT_DIR / "dev_sample50.jsonl"
    if not sample_path.exists():
        sys.exit("Run `python -m simplifyjobs.part_b split` first.")
    records = read_jsonl(sample_path)[: args.limit]

    cfg = compose_cfg(generation_overrides(args.run, args.model, args.model_path),
                      values={"model.local_path": args.model_path})
    engine = Engine(cfg)
    run_batch(engine, records, respond_generate, RESULTS / f"{args.run}.jsonl", args.run, SEED)
    log.info("Done. Next: python -m simplifyjobs.part_b evaluate --run %s", args.run)


# ---------------------------------------------------------------------------
# evaluate / compare
# ---------------------------------------------------------------------------


def score_run(run_path, records_path) -> tuple[list[dict], dict]:
    records = {r["doc_id"]: r for r in read_jsonl(records_path)}
    runs = read_jsonl(run_path)
    scored = [evaluate_record(records[row["doc_id"]], row["raw_output"], SCHEMA) for row in runs]
    return scored, aggregate(scored, runs)


def cmd_evaluate(args) -> None:
    run_path = RESULTS / f"{args.run}.jsonl"
    if not run_path.exists():
        sys.exit(f"{run_path} not found. Run `python -m simplifyjobs.part_b run --run {args.run}` first.")
    scored, report = score_run(run_path, SPLIT_DIR / "dev_sample50.jsonl")
    report["hyperparameters"] = RUNS[args.run]
    write_jsonl(RESULTS / f"{args.run}_scored.jsonl", scored)
    (RESULTS / f"{args.run}_metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


COMPARE_ROWS = [
    ("Usable without correction", "usable_rate"),
    ("Parse rate", "parse_rate"),
    ("Schema valid", "schema_valid_rate"),
    ("Company accuracy", "company_acc"),
    ("  when company in input", "company_acc_when_in_source"),
    ("  when company not in input", "company_acc_when_not_in_source"),
    ("Seniority accuracy", "seniority_acc"),
    ("Location accuracy", "location_acc"),
    ("Skill grounding precision", "skill_grounding_precision"),
    ("Skill atomic rate", "skill_atomic_rate"),
    ("Records with all skills OK", "skills_ok_rate"),
    ("Salary hallucination rate", "salary_hallucination_rate"),
    ("Latency mean (s)", "latency_mean_s"),
    ("Latency p95 (s)", "latency_p95_s"),
    ("Input tokens (mean)", "input_tokens_mean"),
    ("Output tokens (mean)", "output_tokens_mean"),
    ("Output tokens (p95)", "output_tokens_p95"),
    ("Hit max_new_tokens", "n_hit_max_new_tokens"),
]


def cmd_compare(_args) -> None:
    reports = {}
    for run in RUNS:
        path = RESULTS / f"{run}_metrics.json"
        if not path.exists():
            sys.exit(f"{path} not found. Evaluate {run} first.")
        reports[run] = json.loads(path.read_text())

    print(f"{'Metric':<32}" + "".join(f"{r.upper():>12}" for r in RUNS))
    print("-" * (32 + 12 * len(RUNS)))
    for label, key in COMPARE_ROWS:
        cells = []
        for run in RUNS:
            v = reports[run].get(key)
            cells.append(f"{'n/a' if v is None else v:>12}")
        print(f"{label:<32}" + "".join(cells))
    (RESULTS / "comparison.json").write_text(
        json.dumps({k: {r: reports[r].get(k) for r in RUNS} for _, k in COMPARE_ROWS}, indent=2)
    )


# ---------------------------------------------------------------------------
# spotcheck (Part A policy: 10% human spot-check on generate outputs)
# ---------------------------------------------------------------------------


def cmd_spotcheck(args) -> None:
    scored_path = RESULTS / f"{args.run}_scored.jsonl"
    if not scored_path.exists():
        sys.exit(f"Evaluate {args.run} first.")
    scored = read_jsonl(scored_path)
    records = {r["doc_id"]: r for r in read_jsonl(SPLIT_DIR / "dev_sample50.jsonl")}
    picks = random.Random(SEED).sample(scored, min(args.n, len(scored)))

    labels = []
    for i, row in enumerate(picks, 1):
        print("\n" + "=" * 72)
        print(f"[{i}/{len(picks)}] {row['doc_id']}\n--- POSTING ---")
        print(source_text(records[row["doc_id"]])[:1500])
        print("--- MODEL OUTPUT ---")
        print(json.dumps(row["parsed"], indent=2, ensure_ascii=False))
        answer = ""
        while answer not in {"y", "n"}:
            answer = input("Accept into the job database without edits? [y/n] ").strip().lower()
        note = input("Note (optional): ").strip()
        labels.append({"doc_id": row["doc_id"], "human_usable": answer == "y",
                       "metric_usable": row["usable"], "note": note})

    agree = sum(l["human_usable"] == l["metric_usable"] for l in labels)
    out = {"run": args.run, "n": len(labels), "agreement": round(agree / len(labels), 3), "labels": labels}
    (RESULTS / f"{args.run}_spotcheck.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nHuman vs. metric agreement: {agree}/{len(labels)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("split").set_defaults(func=cmd_split)

    p = sub.add_parser("run")
    p.add_argument("--run", choices=RUNS, required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--limit", type=int, default=None, help="only the first N records (smoke test)")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("evaluate")
    p.add_argument("--run", choices=RUNS, required=True)
    p.set_defaults(func=cmd_evaluate)

    sub.add_parser("compare").set_defaults(func=cmd_compare)

    p = sub.add_parser("spotcheck")
    p.add_argument("--run", choices=RUNS, required=True)
    p.add_argument("--n", type=int, default=5, help="10%% of 50 = 5")
    p.set_defaults(func=cmd_spotcheck)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
