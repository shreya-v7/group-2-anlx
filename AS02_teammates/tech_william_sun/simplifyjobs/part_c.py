"""Part C: custom SimplifyJobs API features, evaluated on 50 evaluation records.

    python -m simplifyjobs.part_c prepare          # Pydantic schema + posting index
    python -m simplifyjobs.part_c run --run c0     # repeat for c1 .. c4
    python -m simplifyjobs.part_c evaluate --run c0
    python -m simplifyjobs.part_c compare

Runs are cumulative, so each row isolates one feature:

  c0  mode=generate (Eval 3 settings) on the evaluation set -- reference point
  c1  + Pydantic contract via mode=structured_output             (Custom Eval 1)
  c2  + lookup_posting tool via mode=tool_calling                (Custom Eval 2)
  c3  + guardrail (input/tool/output checks, local LLM check)    (Custom Eval 3)
  c4  + validation-repair loop and grounding verifier            (Custom Eval 4)

All runs share the Part B Eval 3 generation settings, seed, system prompt,
and the same 50 records, and are scored under the same Pydantic contract.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from simplifyjobs.corpus import (
    CORPUS_PATH,
    DATA_DIR,
    ROOT,
    SPLIT_DIR,
    TAXONOMY_PATH,
    load_human_labels,
    read_jsonl,
)
from simplifyjobs.extraction_schema import baseline_system_prompt, load_taxonomy
from simplifyjobs.job_schema import SCHEMA_PATH, export_schema, validate_text
from simplifyjobs.metrics import aggregate, evaluate_record, human_label_report
from simplifyjobs.part_b import DEFAULT_MODEL, DEFAULT_MODEL_PATH, RUNS as B_RUNS, SEED

logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")
log = logging.getLogger("part_c")

RESULTS = ROOT / "results" / "part_c"
POSTING_INDEX = DATA_DIR / "posting_index.json"
SAMPLE = SPLIT_DIR / "eval_sample50.jsonl"
GEN = B_RUNS["b3"]
SYSTEM_PROMPT = baseline_system_prompt(load_taxonomy(TAXONOMY_PATH))
CONTRACT = "simplifyjobs.job_schema:contract_error"


def _features(run: str):
    from simplifyjobs.pipeline import Features

    return {
        "c0": Features(),
        "c1": Features(structured=True),
        "c2": Features(structured=True, tools=True),
        "c3": Features(structured=True, tools=True, guardrail=True),
        "c4": Features(structured=True, tools=True, guardrail=True, repair_attempts=1, verifier=True),
    }[run]


RUN_NAMES = ("c0", "c1", "c2", "c3", "c4")
# First-draft guardrail runs (v1 classifier + baseline LLMBOX tool parser), kept
# as evidence for the redesign. Evaluate/compare only; reproduce from git history.
REPORT_NAMES = ("c0", "c1", "c2", "c3_v1", "c3", "c4_v1", "c4")


def overrides(run: str, model: str) -> list[str]:
    return overrides_for(_features(run), model)


def overrides_for(f, model: str) -> list[str]:
    """Scalar Hydra overrides only (paths and hook strings go through `values`)."""
    mode = "tool_calling" if f.tools else "structured_output" if f.structured else "generate"
    out = [
        f"mode={mode}",
        f"model={model}",
        "model.source=local",
        f"seed={SEED}",
        f"generation.temperature={GEN['temperature']}",
        f"generation.top_p={GEN['top_p']}",
        f"generation.max_new_tokens={GEN['max_new_tokens']}",
    ]
    if f.structured:
        out.append("structured_output.enabled=true")
    if f.tools:
        out.append("tool_calling.enabled=true")
    if f.guardrail:
        out += ["guardrail.enabled=true", "guardrail.llm_check=true"]
    if f.repair_attempts:
        out.append(f"structured_output.repair_attempts={f.repair_attempts}")
    return out


def values(run: str, model_path: str, results_dir=RESULTS) -> dict:
    return values_for(_features(run), model_path, results_dir / f"{run}_guardrail_log.jsonl")


def values_for(f, model_path: str, log_path) -> dict:
    out = {"model.local_path": model_path}
    if f.structured:
        out["structured_output.schema_path"] = str(SCHEMA_PATH)
    if f.guardrail:
        out["guardrail.output_validator"] = CONTRACT
        out["guardrail.log_path"] = str(log_path)
    if f.repair_attempts:
        out["structured_output.validator"] = CONTRACT
    if f.verifier:
        out["structured_output.postprocess"] = "simplifyjobs.verifier:ground"
    return out


def build_pipeline(run: str, model: str, model_path: str, results_dir=RESULTS):
    """Compose the LLMBOX config for a run and return (engine, pipeline)."""
    return build_pipeline_for(_features(run), model, model_path, results_dir / f"{run}_guardrail_log.jsonl")


def build_pipeline_for(features, model: str, model_path: str, log_path):
    from src.schema import ToolDef
    from src.tools import LOOKUP_POSTING_TOOL

    from simplifyjobs.harness import Engine, compose_cfg
    from simplifyjobs.pipeline import Pipeline

    cfg = compose_cfg(overrides_for(features, model), values_for(features, model_path, log_path))
    if features.tools:
        cfg.tool_calling.tools = [ToolDef(**LOOKUP_POSTING_TOOL)]
    engine = Engine(cfg)
    return engine, Pipeline(engine, features, SYSTEM_PROMPT)


# ---------------------------------------------------------------------------


def cmd_prepare(_args) -> None:
    path = export_schema()
    index = {
        r["doc_id"]: {"company": r["metadata"].get("company"), "role_title": r["metadata"].get("role_title")}
        for r in read_jsonl(CORPUS_PATH)
    }
    POSTING_INDEX.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {path} and {POSTING_INDEX} ({len(index)} postings)")


def cmd_run(args) -> None:
    from simplifyjobs.harness import run_batch

    if not SAMPLE.exists():
        sys.exit("Run `python -m simplifyjobs.part_b split` first.")
    if not SCHEMA_PATH.exists() or not POSTING_INDEX.exists():
        sys.exit("Run `python -m simplifyjobs.part_c prepare` first.")
    records = read_jsonl(SAMPLE)[: args.limit]
    engine, pipeline = build_pipeline(args.run, args.model, args.model_path)
    run_batch(engine, records, pipeline, RESULTS / f"{args.run}.jsonl", args.run, SEED)
    log.info("Done. Next: python -m simplifyjobs.part_c evaluate --run %s", args.run)


def score(run_path, sample_path=SAMPLE) -> tuple[list[dict], list[dict]]:
    records = {r["doc_id"]: r for r in read_jsonl(sample_path)}
    runs = read_jsonl(run_path)
    scored = [evaluate_record(records[r["doc_id"]], r["raw_output"], validate_text) for r in runs]
    return scored, runs


def cmd_evaluate(args) -> None:
    run_path = RESULTS / f"{args.run}.jsonl"
    if not run_path.exists():
        sys.exit(f"{run_path} not found. Run it first.")
    scored, runs = score(run_path)
    report = aggregate(scored, runs)
    report["human_labels"] = human_label_report(scored, load_human_labels())
    from simplifyjobs.corpus import write_jsonl

    write_jsonl(RESULTS / f"{args.run}_scored.jsonl", scored)
    (RESULTS / f"{args.run}_metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


COMPARE_ROWS = [
    ("Usable without correction", "usable_rate"),
    ("Schema valid (Pydantic)", "schema_valid_rate"),
    ("Company accuracy", "company_acc"),
    ("  company fabricated", "company_fabricated_rate"),
    ("  null when not in input", "company_null_when_not_in_source"),
    ("Seniority accuracy", "seniority_acc"),
    ("Location unsupported (count)", "n_location_unsupported"),
    ("Skill grounding precision", "skill_grounding_precision"),
    ("Skill atomic rate", "skill_atomic_rate"),
    ("Records with all skills OK", "skills_ok_rate"),
    ("Salary hallucination rate", "salary_hallucination_rate"),
    ("Tool call rate", "tool_call_rate"),
    ("Tool args correct", "tool_args_ok_rate"),
    ("Blocked by guardrail", "blocked"),
    ("Repair rate", "repair_rate"),
    ("Records corrected by verifier", "records_corrected"),
    ("Generations per request", "generations_per_request"),
    ("Latency mean (s)", "latency_mean_s"),
    ("Latency p95 (s)", "latency_p95_s"),
    ("Input tokens (mean)", "input_tokens_mean"),
    ("Output tokens (mean)", "output_tokens_mean"),
    ("Guardrail latency (s)", "guardrail_latency_mean_s"),
    ("Guardrail tokens (mean)", "guardrail_tokens_mean"),
]
HUMAN_ROWS = [
    ("All fields correct", "all_fields_correct"),
    ("role_category", "role_category_acc"),
    ("seniority", "seniority_acc"),
    ("required_skills (exact set)", "required_skills_acc"),
    ("skills precision", "skills_precision"),
    ("skills recall", "skills_recall"),
    ("skills F1", "skills_f1"),
    ("location_type", "location_type_acc"),
    ("company", "company_acc"),
    ("salary_min", "salary_min_acc"),
]


def _table(title: str, rows, reports: dict, section: str | None = None) -> dict:
    runs = list(reports)
    print(f"\n{title}\n{'Metric':<32}" + "".join(f"{r.upper():>10}" for r in runs))
    print("-" * (32 + 10 * len(runs)))
    table = {}
    for label, key in rows:
        vals = [(reports[r].get(section) or {}).get(key) if section else reports[r].get(key) for r in runs]
        table[key] = dict(zip(runs, vals))
        print(f"{label:<32}" + "".join(f"{'-' if v is None else v:>10}" for v in vals))
    return table


def cmd_compare(_args) -> None:
    reports = {}
    for run in REPORT_NAMES:
        path = RESULTS / f"{run}_metrics.json"
        if path.exists():
            reports[run] = json.loads(path.read_text())
    if not reports:
        sys.exit("No evaluated runs found.")
    out = {
        "metric": _table("50 evaluation records (automatic reference)", COMPARE_ROWS, reports),
        "human": _table("25 AS01 human-labeled records", HUMAN_ROWS, reports, section="human_labels"),
        "corrections": {r: reports[r].get("corrections") for r in reports},
        "block_rules": {r: reports[r].get("block_rules") for r in reports},
    }
    print("\nVerifier corrections:", json.dumps(out["corrections"]))
    print("Guardrail block rules:", json.dumps(out["block_rules"]))
    (RESULTS / "comparison.json").write_text(json.dumps(out, indent=2))


def cmd_guardcheck(args) -> None:
    """Input-guardrail false-positive rate on the development set, v1 vs v2.

    All 124 development records are benign job postings, so every block is a
    false positive. v2 was designed after seeing v1 over-refuse on the
    evaluation set; measuring both on the untouched development set checks
    that the difference is real and not tuned to the evaluation records.
    """
    from src.guardrail import Guardrail

    from simplifyjobs.corpus import user_prompt
    from simplifyjobs.harness import Engine, compose_cfg
    from simplifyjobs.pipeline import classifier_cfg

    records = read_jsonl(SPLIT_DIR / "dev.jsonl")[: args.limit]
    base = ["mode=generate", f"model={args.model}", "model.source=local", "guardrail.enabled=true",
            "guardrail.llm_check=true"]
    cfg = compose_cfg(base, {"model.local_path": args.model_path,
                             "guardrail.log_path": str(RESULTS / "guardcheck_log.jsonl")})
    engine = Engine(cfg)
    report = {}
    for version in (1, 2):
        cfg.guardrail.llm_check_version = version
        check_cfg = classifier_cfg(cfg)
        guard = Guardrail(cfg.guardrail, classifier=lambda msgs: engine.generate(msgs, check_cfg))
        blocked, latency = {}, []
        for record in records:
            decision = guard.check_input(user_prompt(record))
            latency.append(decision.cost["latency_s"])
            if not decision.allowed:
                blocked[record["doc_id"]] = decision.rules
        report[f"v{version}"] = {
            "n": len(records),
            "false_positive_rate": round(len(blocked) / len(records), 4),
            "by_rule": {r: sum(r in v for v in blocked.values()) for r in {x for v in blocked.values() for x in v}},
            "latency_mean_s": round(sum(latency) / len(latency), 3),
            "blocked_doc_ids": blocked,
        }
        log.info("v%d: %d/%d benign dev records blocked", version, len(blocked), len(records))
    (RESULTS / "guardcheck.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: {x: v[x] for x in ("n", "false_positive_rate", "by_rule", "latency_mean_s")}
                      for k, v in report.items()}, indent=2))


def cmd_inspect(args) -> None:
    """Print the assistant turns of a few requests (e.g. to see tool-call formats)."""
    rows = [r for r in read_jsonl(RESULTS / f"{args.run}.jsonl") if r.get("block_stage") != "input"]
    for row in rows[: args.n]:
        print("=" * 72, f"\n{row['doc_id']}  tool_calls={row.get('tool_calls')}  blocked={row.get('blocked')}")
        for m in row.get("messages", []):
            if m["role"] != "system" and m["role"] != "user":
                print(f"--- {m['role']} ---\n{m['content'][:400]}")
        print(f"--- final ---\n{row['raw_output'][:400]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare").set_defaults(func=cmd_prepare)
    p = sub.add_parser("run")
    p.add_argument("--run", choices=RUN_NAMES, required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(func=cmd_run)
    p = sub.add_parser("evaluate")
    p.add_argument("--run", choices=REPORT_NAMES, required=True)
    p.set_defaults(func=cmd_evaluate)
    sub.add_parser("compare").set_defaults(func=cmd_compare)
    p = sub.add_parser("guardcheck")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(func=cmd_guardcheck)
    p = sub.add_parser("inspect")
    p.add_argument("--run", choices=REPORT_NAMES, required=True)
    p.add_argument("--n", type=int, default=3)
    p.set_defaults(func=cmd_inspect)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
