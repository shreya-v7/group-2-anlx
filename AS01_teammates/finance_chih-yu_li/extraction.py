#!/usr/bin/env python3
"""Assignment 1: from the finished corpus through evaluation and analysis.

    python extraction.py
    python extraction.py --corpus corpus.jsonl --labels human_labels.jsonl --llm groq

The steps are the ones in the brief, in order:

    1. load and validate the corpus          schema.py
    2. define the schema and the prompt       taxonomy.json, schema.py
    3. run Phi on the evaluation records      local_model.py
    4. validate the structured outputs        schema.py
    5. human vs. Phi                          run_phi_evaluation()
    6. recovery experiment                    run_recovery_experiment()
    7. corpus statistics and clustering       cluster_corpus()
    8. optional bonus: a larger hosted LLM    run_llm()

The extraction task itself is not hard-coded here. Your group's topic, subtopic
and fields live in taxonomy.json, and this script builds the JSON schema, the
prompts and the evaluation from whatever is in that file. Point at a different
one with --taxonomy.

Everything printed is also written to --out (default ./out) so the memo can
quote numbers instead of re-running the model.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from corpus_stats import cluster, describe, print_report
from local_model import PhiClient
from schema import build_schema, load_corpus, load_taxonomy

DEFAULT_TAXONOMY = Path(__file__).resolve().parent / "taxonomy.json"

# ---------------------------------------------------------------------------
# 1. The extraction task, built from taxonomy.json (see schema.build_schema
#    for the field kinds: label, list, number)
# ---------------------------------------------------------------------------

def describe_fields(fields: dict, strict: bool = False) -> str:
    """The field list as prompt text. `strict` is the recovery-prompt wording."""
    lines = []
    for name, spec in fields.items():
        kind, note = spec.get("kind", "label"), spec.get("description", "")
        if spec.get("values"):
            allowed = ", ".join(spec["values"])
            rule = (f"MUST be exactly one of: {allowed}. Never invent a value outside "
                    f"that list." if strict else f"one of {allowed}.")
        elif kind == "list":
            rule = "a list of strings"
            rule += (". Copy each one exactly as it appears in the document, and return "
                     "an empty list if there are none." if strict else ".")
        elif kind == "number":
            rule = "a number taken from the document"
            rule += ". Do not guess; omit the field if it is not stated." if strict else "."
        else:
            rule = "a short value taken from the document."
        lines.append(f"- {name}: " + (f"{note} -- {rule}" if note else rule))
    return "\n".join(lines)


def build_prompts(taxonomy: dict) -> dict[str, str]:
    """Baseline and recovery prompts, generated from the taxonomy.

    Keep every version you try -- the brief asks for the history, and the
    recovery experiment compares two of them side by side. Override either one,
    or add v3, v4..., in the "prompts" block of taxonomy.json.
    """
    subject = taxonomy.get("subtopic") or taxonomy.get("group_topic") or "our corpus"
    fields = taxonomy["fields"]
    prompts = {
        "v1_original": (
            f"You label documents from a corpus about {subject}. "
            f"Fill in every field:\n{describe_fields(fields)}"
        ),
        # The strict version. After reading your v1 errors, rewrite this so it
        # targets the failure Phi actually repeats on your corpus, and say in
        # the memo why you expected that change to help.
        "v2_recovery": (
            f"You label documents from a corpus about {subject}. "
            f"Fill in every field, following these rules exactly:\n"
            f"{describe_fields(fields, strict=True)}\n"
            "Use only what the document says. Do not explain your answer."
        ),
    }
    prompts.update(taxonomy.get("prompts") or {})
    return prompts


def apply_taxonomy(path: str | Path = DEFAULT_TAXONOMY) -> dict:
    """Load the taxonomy and rebuild everything derived from it."""
    global TAXONOMY, FIELDS, EXTRACTION_SCHEMA, PROMPTS
    TAXONOMY = load_taxonomy(path)
    FIELDS = TAXONOMY["fields"]
    EXTRACTION_SCHEMA = build_schema(FIELDS)
    PROMPTS = build_prompts(TAXONOMY)
    return TAXONOMY


apply_taxonomy()

MAX_TEXT_CHARS = 3000
MAX_TABLE_CHARS = 800


def build_prompt(record: dict) -> str:
    """One document as the user turn: text, then the table if there is one."""
    prompt = f"Document {record['doc_id']}:\n{(record.get('raw_text') or '')[:MAX_TEXT_CHARS]}"
    table = record.get("table_json")
    if table:
        prompt += "\n\nTable:\n" + json.dumps(table, ensure_ascii=False)[:MAX_TABLE_CHARS]
    return prompt


# ---------------------------------------------------------------------------
# 2. Running Phi
# ---------------------------------------------------------------------------

def run_phi(records: list[dict], system_prompt: str, phi: PhiClient) -> list[dict]:
    """Extract against EXTRACTION_SCHEMA. Returns one row per record."""
    first_field = next(iter(FIELDS))
    rows = []
    for i, record in enumerate(records, 1):
        response = phi.structured(build_prompt(record), EXTRACTION_SCHEMA, system=system_prompt)
        rows.append({
            "doc_id": record["doc_id"],
            "extracted": response.parsed,
            "violation": response.violation,
            "latency_s": round(response.latency_s, 2),
            "raw_output": response.text.strip()[:400],
        })
        flag = " " if response.ok else "!"
        print(f"  {flag} {i:>3}/{len(records)}  {record['doc_id']:<20} "
              f"{(response.parsed or {}).get(first_field, '--')}")
    return rows


# ---------------------------------------------------------------------------
# 3. Human vs. Phi
# ---------------------------------------------------------------------------

def _as_set(values) -> set[str]:
    if isinstance(values, str):
        values = [values]
    return {str(v).strip().lower() for v in (values or []) if str(v).strip()}


def compare_one(predicted: dict | None, human: dict, fields: dict | None = None) -> dict:
    """Field-level comparison of one prediction against one human label."""
    fields = fields or FIELDS
    predicted = predicted or {}
    result: dict[str, dict] = {}
    for name, spec in fields.items():
        if spec.get("kind", "label") == "list":
            gold, pred = _as_set(human.get(name)), _as_set(predicted.get(name))
            result[name] = {
                "ok": gold == pred,
                "hits": len(gold & pred),
                "n_predicted": len(pred),
                "n_gold": len(gold),
                "phi_only": sorted(pred - gold)[:5],
                "human_only": sorted(gold - pred)[:5],
            }
        else:
            gold, pred = human.get(name), predicted.get(name)
            same = (str(gold).strip().lower() == str(pred).strip().lower()
                    if isinstance(gold, str) and isinstance(pred, str) else gold == pred)
            result[name] = {"ok": bool(same), "human": gold, "phi": pred}
    return result


def run_phi_evaluation(records: list[dict], human_labels: dict[str, dict],
                       rows: list[dict] | None = None, phi: PhiClient | None = None,
                       system_prompt: str | None = None, fields: dict | None = None) -> dict:
    """Run Phi on the evaluation sample and compare outputs with human labels."""
    fields = fields or FIELDS
    system_prompt = system_prompt or PROMPTS["v1_original"]
    if rows is None:
        rows = run_phi(records, system_prompt, phi)

    per_record, errors = [], Counter()
    correct = Counter()
    totals = {name: Counter() for name, spec in fields.items()
              if spec.get("kind", "label") == "list"}
    by_modality: dict[str, list[bool]] = {}

    for row, record in zip(rows, records):
        human = human_labels[row["doc_id"]]
        result = compare_one(row["extracted"], human, fields)

        for name in fields:
            if result[name]["ok"]:
                correct[name] += 1
            else:
                errors[name] += 1
            if name in totals:
                for key in ("hits", "n_predicted", "n_gold"):
                    totals[name][key] += result[name][key]
        if row["violation"]:
            errors["schema_violation"] += 1

        all_ok = all(result[name]["ok"] for name in fields)
        by_modality.setdefault(record.get("modality", "?"), []).append(all_ok)
        per_record.append({
            "doc_id": row["doc_id"],
            "modality": record.get("modality"),
            "human": human,
            "phi": row["extracted"],
            "violation": row["violation"],
            "all_fields_correct": all_ok,
            "fields": result,
        })

    n = len(rows) or 1
    list_scores = {}
    for name, counts in totals.items():
        precision = counts["hits"] / counts["n_predicted"] if counts["n_predicted"] else 0.0
        recall = counts["hits"] / counts["n_gold"] if counts["n_gold"] else 0.0
        list_scores[name] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(2 * precision * recall / (precision + recall), 3) if counts["hits"] else 0.0,
        }

    return {
        "n_records": len(rows),
        "prompt": system_prompt,
        "schema_violation_rate": round(errors["schema_violation"] / n, 3),
        "field_accuracy": {name: round(correct[name] / n, 3) for name in fields},
        "list_field_scores": list_scores,
        "all_fields_correct_rate": round(sum(r["all_fields_correct"] for r in per_record) / n, 3),
        "field_errors": {k: v for k, v in errors.items()},
        "accuracy_by_modality": {m: round(sum(v) / len(v), 3)
                                 for m, v in sorted(by_modality.items())},
        "mean_latency_s": round(sum(r["latency_s"] for r in rows) / n, 2),
        "disagreements": [r for r in per_record if not r["all_fields_correct"]][:8],
        "per_record": per_record,
    }


def print_evaluation(title: str, ev: dict) -> None:
    print(f"\n{'=' * 64}\n{title}\n{'=' * 64}")
    for key in ("n_records", "all_fields_correct_rate", "schema_violation_rate",
                "mean_latency_s"):
        print(f"  {key:<26} {ev[key]}")
    for key in ("field_accuracy", "list_field_scores", "field_errors",
                "accuracy_by_modality"):
        print(f"  {key:<26} {ev[key]}")
    if ev["disagreements"]:
        print("\n  representative disagreements:")
        for row in ev["disagreements"][:4]:
            print(f"    {row['doc_id']}")
            print(f"      human: {json.dumps(row['human'], ensure_ascii=False)[:150]}")
            print(f"      phi:   {json.dumps(row['phi'], ensure_ascii=False)[:150]}")


# ---------------------------------------------------------------------------
# 4. Recovery experiment
# ---------------------------------------------------------------------------

def run_recovery_experiment(records: list[dict], original_prompt: str, revised_prompt: str,
                            human_labels: dict[str, dict], baseline: dict,
                            phi: PhiClient) -> dict:
    """Rerun the records the original prompt got wrong, using the revised prompt."""
    failed_ids = {r["doc_id"] for r in baseline["per_record"] if not r["all_fields_correct"]}
    subset = [r for r in records if r["doc_id"] in failed_ids]
    if not subset:
        print("\nNothing failed under the original prompt -- no recovery run needed.")
        return {}

    print(f"\nRerunning {len(subset)} failed record(s) with the revised prompt ...")
    rows = run_phi(subset, revised_prompt, phi)
    after = run_phi_evaluation(subset, human_labels, rows=rows, system_prompt=revised_prompt)

    before = [r for r in baseline["per_record"] if r["doc_id"] in failed_ids]
    fixed = [r["doc_id"] for r in after["per_record"] if r["all_fields_correct"]]
    return {
        "original_prompt": original_prompt,
        "revised_prompt": revised_prompt,
        "n_rerun": len(subset),
        "before": {"all_fields_correct": 0,
                   "field_accuracy": {name: _accuracy(before, name) for name in FIELDS},
                   "field_errors": _count_errors(before)
                   | {"schema_violation": sum(1 for r in before if r.get("violation"))}},
        "after": {"all_fields_correct": len(fixed),
                  "field_accuracy": after["field_accuracy"],
                  "field_errors": after["field_errors"]},
        "fixed_doc_ids": fixed,
        "still_wrong_doc_ids": [r["doc_id"] for r in after["per_record"]
                                if not r["all_fields_correct"]],
        "per_record_after": after["per_record"],
    }


def _count_errors(per_record: list[dict]) -> dict:
    errors = Counter()
    for row in per_record:
        for name, result in row["fields"].items():
            if not result["ok"]:
                errors[name] += 1
    return dict(errors)


def _accuracy(per_record: list[dict], name: str) -> float:
    if not per_record:
        return 0.0
    return round(sum(r["fields"][name]["ok"] for r in per_record) / len(per_record), 3)


# ---------------------------------------------------------------------------
# 5. Clustering
# ---------------------------------------------------------------------------

def cluster_corpus(records: list[dict], k: int = 6, plot_path: str | None = None):
    """Create clusters from corpus records for analysis."""
    result = cluster(records, k=k, plot_path=plot_path)
    result.print_summary()
    for cluster_id in sorted(result.sizes()):
        print(f"\n  cluster {cluster_id} -- read these, then name it:")
        for doc in result.sample(cluster_id, n=3):
            print(f"    {doc['doc_id']:<20} {(doc.get('raw_text') or '')[:110]!r}")
    return result


def accuracy_by_cluster(result, evaluation: dict) -> dict:
    """Connects the cluster reading to the Human vs. Phi results.

    Two measures per cluster, because one of them is degenerate on this task:
    `accuracy` is the share of records with every field correct, which is 0
    everywhere as soon as one field (here `required_skills`) is scored by exact
    set equality; `field_accuracy` is the share of individual fields correct,
    which is the one worth reading.
    """
    cluster_of = {r["doc_id"]: int(label)
                  for r, label in zip(result.records, result.labels.tolist())}
    buckets: dict[int, dict] = {}
    for row in evaluation["per_record"]:
        cluster_id = cluster_of.get(row["doc_id"])
        if cluster_id is None:
            continue
        bucket = buckets.setdefault(cluster_id, {"n": 0, "all_ok": 0, "ok": 0, "total": 0})
        bucket["n"] += 1
        bucket["all_ok"] += int(row["all_fields_correct"])
        bucket["ok"] += sum(1 for f in row["fields"].values() if f["ok"])
        bucket["total"] += len(row["fields"])
    return {cid: {"n": b["n"],
                  "accuracy": round(b["all_ok"] / b["n"], 3),
                  "field_accuracy": round(b["ok"] / b["total"], 3)}
            for cid, b in sorted(buckets.items())}


# ---------------------------------------------------------------------------
# 6. Optional bonus: the same task on a larger hosted LLM
# ---------------------------------------------------------------------------

def run_llm(prompt: str, provider: str = "groq", system: str | None = None):
    """Run the structured extraction prompt using the configured hosted LLM."""
    from llm_utils import PROVIDERS, get_client

    if provider not in PROVIDERS:
        raise SystemExit(f"--llm is the hosted comparison lane: {', '.join(sorted(PROVIDERS))}. "
                         "The local lane is already Phi-4-mini-instruct (local_model.py).")
    client = get_client(provider)
    return client.structured(prompt, EXTRACTION_SCHEMA,
                             system=system or PROMPTS["v1_original"], tag="bonus")


def run_llm_comparison(records: list[dict], human_labels: dict[str, dict],
                       provider: str) -> dict:
    first_field = next(iter(FIELDS))
    rows = []
    for i, record in enumerate(records, 1):
        response = run_llm(build_prompt(record), provider=provider)
        rows.append({
            "doc_id": record["doc_id"],
            "extracted": response.parsed,
            "violation": response.schema_violation,
            "latency_s": round(response.latency_s, 2),
            "raw_output": response.text.strip()[:400],
        })
        print(f"  {i:>3}/{len(records)}  {record['doc_id']:<20} "
              f"{(response.parsed or {}).get(first_field, '--')}")
    return run_phi_evaluation(records, human_labels, rows=rows,
                              system_prompt=f"hosted:{provider}")


# ---------------------------------------------------------------------------
# 7. The run
# ---------------------------------------------------------------------------

def validate_corpus(path: str):
    """Validate corpus.jsonl against the assignment requirements."""
    result = load_corpus(path)
    print(f"Loaded {len(result.records)} records from {path}")
    for problem in result.problems[:10]:
        print(f"  {problem}")
    if len(result.problems) > 10:
        print(f"  ... and {len(result.problems) - 10} more")
    if result.problems:
        print(f"  run `python check_corpus.py {path}` for the full report")
    return result


def load_human_labels(path: str) -> dict[str, dict]:
    labels = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                labels[row["doc_id"]] = row["human_label"]
    return labels


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  wrote {path}")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", default="corpus.jsonl")
    parser.add_argument("--labels", default="human_labels.jsonl")
    parser.add_argument("--taxonomy", default=str(DEFAULT_TAXONOMY))
    parser.add_argument("--out", default="out", help="directory for results")
    parser.add_argument("--k", type=int, default=6, help="number of clusters")
    parser.add_argument("--model-path", default=None, help="local Phi-4-mini weights")
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--llm", default=None,
                        help="optional bonus lane: a provider from llm_utils.py, e.g. groq")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    taxonomy = apply_taxonomy(args.taxonomy)
    print(f"Task: {taxonomy.get('group_topic') or '(group topic unset)'} / "
          f"{taxonomy.get('subtopic') or '(subtopic unset)'}")
    print(f"Fields: {', '.join(FIELDS)}")

    # -- 1. load ---------------------------------------------------------
    corpus = validate_corpus(args.corpus)
    human_labels = load_human_labels(args.labels)
    evaluation_set = [r for r in corpus.records if r["doc_id"] in human_labels]
    print(f"Evaluation sample: {len(evaluation_set)} labelled records")
    if not evaluation_set:
        raise SystemExit("No corpus record matches a doc_id in the human labels.")

    # -- 2-4. run Phi and compare with the human labels ------------------
    phi = PhiClient(args.model_path, device=args.device)
    print(f"\nRunning Phi on {len(evaluation_set)} records (prompt v1_original) ...")
    rows = run_phi(evaluation_set, PROMPTS["v1_original"], phi)
    write_jsonl(out / "phi_v1.jsonl", rows)

    evaluation = run_phi_evaluation(evaluation_set, human_labels, rows=rows)
    print_evaluation("HUMAN vs. PHI (v1_original)", evaluation)
    write_json(out / "evaluation.json", evaluation)

    # -- 5. recovery -----------------------------------------------------
    # Every prompt version in taxonomy.json other than the baseline is rerun and
    # written out, so the whole prompt history is in out/ and not only the one
    # the brief asks about. v2_recovery keeps the plain filename.
    versions = [name for name in PROMPTS if name != "v1_original"]
    for version in versions:
        recovery = run_recovery_experiment(evaluation_set, PROMPTS["v1_original"],
                                           PROMPTS[version], human_labels, evaluation, phi)
        if not recovery:
            continue
        print(f"\n{'=' * 64}\nRECOVERY (v1_original -> {version})\n{'=' * 64}")
        print(f"  rerun          {recovery['n_rerun']} previously-failed records")
        print(f"  before         {recovery['before']['field_accuracy']}")
        print(f"  after          {recovery['after']['field_accuracy']}")
        print(f"  violations     {recovery['before']['field_errors'].get('schema_violation', 0)}"
              f" -> {recovery['after']['field_errors'].get('schema_violation', 0)}")
        print(f"  fixed          {recovery['fixed_doc_ids']}")
        filename = "recovery.json" if version == "v2_recovery" else f"recovery_{version}.json"
        write_json(out / filename, recovery)

    # -- 6. statistics and clustering ------------------------------------
    stats = describe(corpus.records)
    print()
    print_report(stats)
    write_json(out / "corpus_stats.json", stats)

    print(f"\n{'=' * 64}\nCLUSTERS\n{'=' * 64}")
    result = cluster_corpus(corpus.records, k=args.k, plot_path=str(out / "clusters.png"))
    per_cluster = accuracy_by_cluster(result, evaluation)
    print("\n  Phi accuracy on the labelled records, by cluster:")
    for cluster_id, info in per_cluster.items():
        print(f"    cluster {cluster_id:>2}  n={info['n']:>3}  "
              f"field accuracy {info['field_accuracy']:.0%}  "
              f"(every field correct: {info['accuracy']:.0%})")
    write_json(out / "clusters.json", {
        "k": args.k,
        "silhouette": result.silhouette,
        "sizes": result.sizes(),
        "top_terms": result.top_terms,
        "phi_accuracy_by_cluster": per_cluster,
        "assignments": {r["doc_id"]: int(label)
                        for r, label in zip(result.records, result.labels.tolist())},
    })

    # -- 7. optional bonus lane ------------------------------------------
    if args.llm:
        print(f"\n{'=' * 64}\nBONUS: {args.llm}\n{'=' * 64}")
        llm_evaluation = run_llm_comparison(evaluation_set, human_labels, args.llm)
        print_evaluation(f"HUMAN vs. {args.llm.upper()}", llm_evaluation)
        write_json(out / "llm_evaluation.json", llm_evaluation)

    print(f"\nDone. Results in {out}/ -- the memo is written from these numbers.\n")


if __name__ == "__main__":
    main()
