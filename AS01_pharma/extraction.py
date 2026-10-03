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
import hashlib
import platform
import sys
import time
from datetime import datetime, timezone
from collections import Counter
from pathlib import Path

from local_model import PhiClient
from schema import build_schema, load_corpus, load_taxonomy, validate_against_schema

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

# Full text and table are deliberately retained. local_model.py checks token capacity.
MAX_TEXT_CHARS = None
MAX_TABLE_CHARS = None


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

def run_phi(records: list[dict], system_prompt: str, phi: PhiClient, checkpoint: Path | None = None) -> list[dict]:
    """Extract against EXTRACTION_SCHEMA. Returns one row per record."""
    first_field = next(iter(FIELDS))
    rows = []
    done = set()
    if checkpoint and checkpoint.exists() and checkpoint.stat().st_size:
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(row)
            done.add(row["doc_id"])
        print(f"  resuming Phi from {len(done)} saved predictions", flush=True)
    elif checkpoint:
        checkpoint.write_text("")
    for i, record in enumerate(records, 1):
        if record["doc_id"] in done:
            print(f"    {i:>3}/{len(records)}  {record['doc_id']:<20} (cached)", flush=True)
            continue
        response = phi.structured(build_prompt(record), EXTRACTION_SCHEMA, system=system_prompt)
        rows.append({
            "doc_id": record["doc_id"],
            "extracted": response.parsed,
            "violation": response.violation,
            "latency_s": round(response.latency_s, 2),
            "raw_output": response.text.strip() ,
        })
        if checkpoint:
            with checkpoint.open("a") as handle: handle.write(json.dumps(rows[-1], ensure_ascii=False)+"\n")
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
    if not isinstance(values, list): return set()
    return {str(v).strip().lower() for v in values if isinstance(v, str) and str(v).strip()}


def compare_one(predicted: dict | None, human: dict, fields: dict | None = None) -> dict:
    """Field-level comparison of one prediction against one human label."""
    fields = fields or FIELDS
    predicted = predicted if isinstance(predicted, dict) else {}
    result: dict[str, dict] = {}
    for name, spec in fields.items():
        if spec.get("kind", "label") == "list":
            gold, pred = _as_set(human.get(name)), _as_set(predicted.get(name))
            result[name] = {
                "ok": name in predicted and isinstance(predicted[name], list) and all(isinstance(v, str) for v in predicted[name]) and gold == pred,
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
            result[name] = {"ok": name in predicted and bool(same), "human": gold, "phi": pred}
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

    record_by_id = {r['doc_id']: r for r in records}
    if len(rows) != len(records) or len({r['doc_id'] for r in rows}) != len(rows) or {r['doc_id'] for r in rows} != set(record_by_id):
        raise ValueError('Prediction IDs must match the evaluation records exactly, without duplicates.')
    for row in rows:
        record = record_by_id[row['doc_id']]
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

        all_ok = not row["violation"] and all(result[name]["ok"] for name in fields)
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
                            phi: PhiClient, selected_ids: list[str] | None = None, checkpoint: Path | None = None) -> dict:
    """Rerun the records the original prompt got wrong, using the revised prompt."""
    failed_ids = {r["doc_id"] for r in baseline["per_record"] if not r["all_fields_correct"]}
    if selected_ids is not None:
        if len(set(selected_ids)) != len(selected_ids) or not set(selected_ids) <= failed_ids:
            raise ValueError('Recovery IDs must be unique baseline failures.')
        failed_ids = set(selected_ids)
    subset = [r for r in records if r["doc_id"] in failed_ids]
    if not subset:
        print("\nNothing failed under the original prompt -- no recovery run needed.")
        return {}

    print(f"\nRerunning {len(subset)} failed record(s) with the revised prompt ...")
    rows = run_phi(subset, revised_prompt, phi, checkpoint=checkpoint)
    after = run_phi_evaluation(subset, human_labels, rows=rows, system_prompt=revised_prompt)

    before = [r for r in baseline["per_record"] if r["doc_id"] in failed_ids]
    fixed = [r["doc_id"] for r in after["per_record"] if r["all_fields_correct"]]
    return {
        "original_prompt": original_prompt,
        "revised_prompt": revised_prompt,
        "n_rerun": len(subset),
        "selection_doc_ids": [r["doc_id"] for r in subset],
        "schema_violation_rate_after": after["schema_violation_rate"],
        "raw_predictions": rows,
        "field_transitions": {name: {
            "fixed": sum(not b["fields"][name]["ok"] and a["fields"][name]["ok"] for b,a in zip(before,after["per_record"])),
            "regressed": sum(b["fields"][name]["ok"] and not a["fields"][name]["ok"] for b,a in zip(before,after["per_record"]))
        } for name in FIELDS},
        "before": {"all_fields_correct": 0,
                   "field_accuracy": {name: _accuracy(before, name) for name in FIELDS},
                   "field_errors": _count_errors(before)},
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
    from corpus_stats import cluster
    result = cluster(records, k=k, plot_path=plot_path)
    result.print_summary()
    for cluster_id in sorted(result.sizes()):
        print(f"\n  cluster {cluster_id} -- read these, then name it:")
        for doc in result.sample(cluster_id, n=3):
            print(f"    {doc['doc_id']:<20} {(doc.get('raw_text') or '')[:110]!r}")
    return result


def accuracy_by_cluster(result, evaluation: dict) -> dict:
    """Connects the cluster reading to the Human vs. Phi results."""
    cluster_of = {r["doc_id"]: int(label)
                  for r, label in zip(result.records, result.labels.tolist())}
    buckets: dict[int, list[bool]] = {}
    for row in evaluation["per_record"]:
        cluster_id = cluster_of.get(row["doc_id"])
        if cluster_id is not None:
            buckets.setdefault(cluster_id, []).append(row["all_fields_correct"])
    return {cid: {"n": len(v), "accuracy": round(sum(v) / len(v), 3)}
            for cid, v in sorted(buckets.items())}


# ---------------------------------------------------------------------------
# 6. Optional bonus: the same task on a larger hosted LLM
# ---------------------------------------------------------------------------

def run_llm(prompt: str, provider: str = "groq", system: str | None = None,
            model: str | None = None):
    """Run the structured extraction prompt using the configured hosted LLM."""
    from llm_utils import PROVIDERS, get_client

    if provider not in PROVIDERS:
        raise SystemExit(f"--llm is the hosted comparison lane: {', '.join(sorted(PROVIDERS))}. "
                         "The local lane is already Phi-4-mini-instruct (local_model.py).")
    # Long qualification lists overflow the starter 1,024-token default.
    # One attempt per model: 429/503/404 immediately fall back instead of
    # burning remaining daily quota retrying an exhausted model.
    client = get_client(provider, model=model, max_tokens=4096, timeout=180, max_attempts=1)
    return client.structured(prompt, EXTRACTION_SCHEMA,
                             system=system or PROMPTS["v1_original"])


def run_llm_comparison(records: list[dict], human_labels: dict[str, dict],
                       provider: str, model: str | None = None,
                       checkpoint: Path | None = None) -> dict:
    first_field = next(iter(FIELDS))
    rows = []
    done = set()
    if checkpoint and checkpoint.exists():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(row)
            done.add(row["doc_id"])
        print(f"  resuming hosted LLM from {len(done)} saved predictions")
    elif checkpoint:
        checkpoint.write_text("")
    for i, record in enumerate(records, 1):
        if record["doc_id"] in done:
            print(f"  {i:>3}/{len(records)}  {record['doc_id']:<20} (cached)")
            continue
        last_error = None
        response = None
        candidates = [model] if model else [None]
        if provider == "gemini":
            # Prefer models whose project RPD is still unused. Exhausted today:
            # gemini-3.5/3.6/3.8-flash (20/20). Free-tier Flash is 5 RPM / 20 RPD;
            # Lite is 15 RPM / 500 RPD. Stay sequential and sleep between calls.
            for fallback in ("gemini-3.5-flash-lite", "gemini-3.1-flash-lite",
                             "gemini-2.5-flash-lite"):
                if fallback not in candidates:
                    candidates.append(fallback)
        for candidate in candidates:
            try:
                response = run_llm(build_prompt(record), provider=provider, model=candidate)
                last_error = None
                if candidate and candidate != model:
                    print(f"  {i:>3}/{len(records)}  fallback model {candidate}", flush=True)
                break
            except RuntimeError as exc:
                last_error = exc
                text = str(exc)
                if "HTTP 429" not in text and "HTTP 503" not in text and "HTTP 404" not in text:
                    raise
                print(f"  {i:>3}/{len(records)}  {candidate or 'default'} unavailable ({text[14:40].strip()})", flush=True)
        if last_error:
            raise last_error
        row = {
            "doc_id": record["doc_id"],
            "extracted": response.parsed,
            "violation": response.schema_violation,
            "latency_s": round(response.latency_s, 2),
            "raw_output": response.text.strip(),
            "model": response.model,
            "provider": response.provider,
        }
        rows.append(row)
        if checkpoint:
            with checkpoint.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  {i:>3}/{len(records)}  {record['doc_id']:<20} "
              f"{(response.parsed or {}).get(first_field, '--')}", flush=True)
        if i < len(records):
            # Lite is 15 RPM / 500 RPD on this project; 8s stays under 8 RPM.
            time.sleep(8)
    evaluation = run_phi_evaluation(records, human_labels, rows=rows,
                                    system_prompt=f"hosted:{provider}:{rows[0]['model'] if rows else provider}")
    evaluation["provider"] = provider
    evaluation["model"] = rows[0]["model"] if rows else model
    return evaluation


def compare_slm_and_llm(phi_eval: dict, llm_eval: dict) -> dict:
    """Side-by-side Human vs Phi vs hosted LLM on the same 25 records."""
    fields = list(phi_eval["field_accuracy"])
    field_delta = {
        name: round(llm_eval["field_accuracy"][name] - phi_eval["field_accuracy"][name], 3)
        for name in fields
    }
    phi_by_id = {r["doc_id"]: r for r in phi_eval["per_record"]}
    llm_by_id = {r["doc_id"]: r for r in llm_eval["per_record"]}
    both_correct = [
        doc_id for doc_id, row in llm_by_id.items()
        if row["all_fields_correct"] and phi_by_id[doc_id]["all_fields_correct"]
    ]
    llm_only = [
        doc_id for doc_id, row in llm_by_id.items()
        if row["all_fields_correct"] and not phi_by_id[doc_id]["all_fields_correct"]
    ]
    phi_only = [
        doc_id for doc_id, row in phi_by_id.items()
        if row["all_fields_correct"] and not llm_by_id[doc_id]["all_fields_correct"]
    ]
    return {
        "n_records": phi_eval["n_records"],
        "phi_model": "mlx-community/Phi-4-mini-instruct-4bit",
        "llm_provider": llm_eval.get("provider"),
        "llm_model": llm_eval.get("model"),
        "same_prompt": True,
        "same_schema": True,
        "same_human_labels": True,
        "full_record_agreement": {
            "phi": phi_eval["all_fields_correct_rate"],
            "llm": llm_eval["all_fields_correct_rate"],
        },
        "field_agreement": {
            "phi": round(sum(phi_eval["field_accuracy"].values()) / len(fields), 3),
            "llm": round(sum(llm_eval["field_accuracy"].values()) / len(fields), 3),
        },
        "schema_valid": {
            "phi": round(1 - phi_eval["schema_violation_rate"], 3),
            "llm": round(1 - llm_eval["schema_violation_rate"], 3),
        },
        "mean_latency_s": {
            "phi": phi_eval["mean_latency_s"],
            "llm": llm_eval["mean_latency_s"],
        },
        "field_accuracy": {
            name: {"phi": phi_eval["field_accuracy"][name],
                   "llm": llm_eval["field_accuracy"][name],
                   "delta": field_delta[name]}
            for name in fields
        },
        "list_field_scores": {
            name: {"phi": phi_eval["list_field_scores"][name],
                   "llm": llm_eval["list_field_scores"][name]}
            for name in phi_eval.get("list_field_scores", {})
        },
        "full_record_ids": {
            "both_correct": both_correct,
            "llm_only_correct": llm_only,
            "phi_only_correct": phi_only,
        },
    }


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
        for line_no, line in enumerate(handle, 1):
            if not line.strip(): continue
            row = json.loads(line)
            doc_id = row['doc_id']
            if doc_id in labels: raise SystemExit(f'Duplicate human label ID: {doc_id}')
            if row.get('annotation_status') != 'complete':
                raise SystemExit(f'{doc_id}: annotation incomplete.')
            if row.get('annotation_origin') != 'human' or row.get('human_reviewed') is not True:
                raise SystemExit(f'{doc_id}: completed, human-reviewed annotation required.')
            if not row.get('annotator', '').strip(): raise SystemExit(f'{doc_id}: missing annotator name.')
            problem = validate_against_schema(row.get('human_label'), EXTRACTION_SCHEMA)
            if problem: raise SystemExit(f'{doc_id}: {problem}')
            labels[doc_id] = row['human_label']
    if not 20 <= len(labels) <= 30: raise SystemExit('Human evaluation requires 20-30 completed records.')
    return labels


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  wrote {path}")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  wrote {path}")


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze(records, out, k, evaluation=None):
    from corpus_stats import describe, print_report
    stats = describe(records)
    print_report(stats)
    write_json(out / 'corpus_stats.json', stats)
    result = cluster_corpus(records, k=k, plot_path=str(out / 'clusters.png'))
    payload = {'k': k, 'silhouette': result.silhouette, 'sizes': result.sizes(),
               'top_terms': result.top_terms, 'employer_crosstab': result.crosstab('metadata.employer'),
               'phi_accuracy_by_cluster': accuracy_by_cluster(result, evaluation) if evaluation else None,
               'assignments': {r['doc_id']: int(label) for r, label in zip(result.records, result.labels.tolist())},
               'representatives': {str(i): [r['doc_id'] for r in result.sample(i, n=3)] for i in result.sizes()}}
    write_json(out / 'clusters.json', payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', default='corpus.jsonl')
    parser.add_argument('--labels', default='human_labels.jsonl')
    parser.add_argument('--taxonomy', default=str(DEFAULT_TAXONOMY))
    parser.add_argument('--out', default='out')
    parser.add_argument('--k', type=int, default=6)
    parser.add_argument('--model-path', default=None)
    parser.add_argument('--backend', choices=['transformers','mlx'], default='transformers')
    parser.add_argument('--max-new-tokens', type=int, default=4096)
    parser.add_argument('--device', default='auto', choices=['auto','mps','cuda','cpu'])
    parser.add_argument('--llm', default=None)
    parser.add_argument('--llm-model', default=None)
    parser.add_argument('--stage', choices=['analyze','baseline','recovery','llm'], default='baseline')
    args = parser.parse_args()
    out=Path(args.out); out.mkdir(exist_ok=True, parents=True)
    taxonomy=apply_taxonomy(args.taxonomy)
    corpus=validate_corpus(args.corpus)
    if corpus.problems: raise SystemExit('Fix corpus validation errors first.')
    if not 150 <= len(corpus.records) <= 300: raise SystemExit('Expected 150-300 corpus records.')
    if args.stage == 'analyze':
        analyze(corpus.records,out,args.k); return
    human_labels=load_human_labels(args.labels)
    corpus_ids={r['doc_id'] for r in corpus.records}
    if set(human_labels)-corpus_ids: raise SystemExit('Human labels reference missing corpus records.')
    records=[r for r in corpus.records if r['doc_id'] in human_labels]
    sample_path=Path(args.labels).with_name('evaluation_sample.json')
    if not sample_path.exists(): raise SystemExit('evaluation_sample.json is required; use make_human_labels.py.')
    sample=json.loads(sample_path.read_text())
    if set(sample['doc_ids']) != set(human_labels): raise SystemExit('Human labels differ from the fixed random sample.')
    if sample['corpus_sha256'] != file_sha(args.corpus): raise SystemExit('Corpus changed since sampling. Regenerate and re-annotate the sample.')
    if sample['fields_sha256'] != hashlib.sha256(json.dumps(FIELDS,sort_keys=True).encode()).hexdigest():
        raise SystemExit('Extraction fields changed since sampling. Re-annotate the sample.')
    if args.stage == 'llm':
        provider = args.llm or 'gemini'
        llm_eval = run_llm_comparison(records, human_labels, provider, model=args.llm_model,
                                      checkpoint=out/'llm_predictions.jsonl')
        write_json(out/'llm_evaluation.json', llm_eval)
        print_evaluation(f'HUMAN vs HOSTED LLM ({llm_eval.get("model")})', llm_eval)
        phi_path = out/'evaluation.json'
        if phi_path.exists():
            comparison = compare_slm_and_llm(json.loads(phi_path.read_text()), llm_eval)
            write_json(out/'slm_vs_llm.json', comparison)
            print_evaluation('SLM vs LLM SUMMARY', {
                **llm_eval,
                "all_fields_correct_rate": comparison["full_record_agreement"]["llm"],
            })
            print(json.dumps({
                "full_record": comparison["full_record_agreement"],
                "mean_field": comparison["field_agreement"],
                "schema_valid": comparison["schema_valid"],
                "latency_s": comparison["mean_latency_s"],
                "field_delta": {k: v["delta"] for k, v in comparison["field_accuracy"].items()},
            }, indent=2))
        return
    manifest={'timestamp':datetime.now(timezone.utc).isoformat(), 'corpus_sha256':file_sha(args.corpus),
              'labels_sha256':file_sha(args.labels), 'fields':FIELDS, 'baseline_prompt':PROMPTS['v1_original'],
              'doc_ids':[r['doc_id'] for r in records], 'python':platform.python_version(),
              'max_new_tokens':args.max_new_tokens, 'truncation':False,
              'model': args.model_path or ('mlx-community/Phi-4-mini-instruct-4bit' if args.backend=='mlx' else 'microsoft/Phi-4-mini-instruct'),
              'backend':args.backend,'reference_origin':'human'}
    client_class = PhiClient
    if args.backend == 'mlx':
        from mlx_model import MLXPhiClient
        client_class = MLXPhiClient
    # Recovery must be designed after inspecting baseline errors, not silently preselected.
    if args.stage == 'recovery':
        if not (taxonomy.get('prompts') or {}).get('v2_recovery'):
            raise SystemExit('Inspect out/evaluation.json, then add an evidence-based prompts.v2_recovery in taxonomy.json.')
        prior=json.loads((out/'baseline_manifest.json').read_text())
        for key in ['corpus_sha256','labels_sha256','fields','baseline_prompt','doc_ids','backend','model','max_new_tokens','reference_origin']:
            if manifest[key] != prior[key]: raise SystemExit(f'Baseline mismatch for {key}; use unchanged baseline data/schema/prompt.')
        baseline=json.loads((out/'evaluation.json').read_text())
        phi=client_class(args.model_path, device=args.device,max_new_tokens=args.max_new_tokens)
        recovery=run_recovery_experiment(records,PROMPTS['v1_original'],PROMPTS['v2_recovery'],human_labels,baseline,phi,selected_ids=taxonomy.get('recovery_doc_ids'),checkpoint=out/'phi_v2.jsonl')
        recovery['reference_origin']=manifest['reference_origin']
        write_json(out/'recovery.json',recovery)
        write_json(out/'recovery_manifest.json',{**manifest,'revised_prompt':PROMPTS['v2_recovery']})
        return
    phi=client_class(args.model_path,device=args.device,max_new_tokens=args.max_new_tokens)
    write_json(out/'baseline_manifest.json',manifest)
    rows=run_phi(records,PROMPTS['v1_original'],phi,checkpoint=out/'phi_v1.jsonl')
    write_jsonl(out/'phi_v1.jsonl',rows)
    evaluation=run_phi_evaluation(records,human_labels,rows=rows)
    evaluation['reference_origin']=manifest['reference_origin']
    print_evaluation('HUMAN vs PHI',evaluation)
    write_json(out/'evaluation.json',evaluation)
    analyze(corpus.records,out,args.k,evaluation)
    if args.llm:
        write_json(out/'llm_evaluation.json',run_llm_comparison(records,human_labels,args.llm))
    print('Inspect baseline errors before writing prompts.v2_recovery and running --stage recovery.')


if __name__ == '__main__':
    main()
