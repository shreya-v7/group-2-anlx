"""Recompute the Tech AS02 results from saved outputs. No model is loaded.

    python verification/verify_tech_as02.py
    python verification/verify_tech_as02.py --rerun /path/to/fresh/results/part_c

Needs Python 3.10+ and pydantic. Exits 1 if a recomputed number differs from
the saved metrics or a data check fails.

Two kinds of check:
  * recomputed  : William's own scorer (simplifyjobs.metrics) re-run on the
                  saved raw outputs, compared with the saved *_metrics.json.
  * independent : counts that do not use that scorer (fences, strict JSON,
                  tool calls, guardrail blocks, canary markers, split overlap).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics as st
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
T = REPO / "AS02_teammates" / "tech_william_sun"
AS01 = REPO / "AS01_teammates" / "tech_william_sun"
sys.path.insert(0, str(T))

try:
    from simplifyjobs.corpus import load_human_labels, read_jsonl
    from simplifyjobs.extraction_schema import build_schema, load_taxonomy
    from simplifyjobs.job_schema import validate_text
    from simplifyjobs.metrics import aggregate, evaluate_record, human_label_report
except ImportError as exc:  # pragma: no cover
    sys.exit(f"Cannot import the Tech scorer ({exc}). Install pydantic: pip install pydantic")

DATA = T / "data" / "simplifyjobs"
SPLITS = DATA / "splits"
RES = T / "results"
FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)
NUMERIC_SKILL = re.compile(r"[\d\s\-/+.]+")

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok ' if ok else 'FAIL'} {label}{(': ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def same(a, b) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        return a is not None and b is not None and round(a, 4) == round(b, 4)
    return a == b


def model_text(row: dict) -> str:
    """The model's own text: model_output when the pipeline kept it, else raw_output."""
    return row.get("model_output") or row.get("raw_output") or ""


# ---------------------------------------------------------------------------
def verify_data() -> None:
    print("[data] AS02 copies match the AS01 submission")
    for name in ("corpus.jsonl", "human_labels.jsonl"):
        check(name, sha256(DATA / name) == sha256(AS01 / name))


def verify_split() -> None:
    print("[split] posting-level split")
    dev, ev = read_jsonl(SPLITS / "dev.jsonl"), read_jsonl(SPLITS / "eval.jsonl")
    urls = lambda rows: {r["source_url"] for r in rows}
    ids = lambda rows: {r["doc_id"] for r in rows}
    check("records 124 dev / 126 eval", (len(dev), len(ev)) == (124, 126), f"{len(dev)} / {len(ev)}")
    check("postings 39 dev / 32 eval", (len(urls(dev)), len(urls(ev))) == (39, 32))
    check("no shared source_url", not urls(dev) & urls(ev))
    ds, es = read_jsonl(SPLITS / "dev_sample50.jsonl"), read_jsonl(SPLITS / "eval_sample50.jsonl")
    check("samples are 50 records inside their split", len(ds) == len(es) == 50 and ids(ds) <= ids(dev) and ids(es) <= ids(ev))
    labeled = set(load_human_labels())
    check("25 hand-labeled records, all in eval_sample50", len(labeled) == 25 and labeled <= ids(es))


def recompute(run_dir: Path, run: str, sample: Path, contract) -> tuple[list[dict], list[dict], dict]:
    records = {r["doc_id"]: r for r in read_jsonl(sample)}
    rows = read_jsonl(run_dir / f"{run}.jsonl")
    scored = [evaluate_record(records[r["doc_id"]], r["raw_output"], contract) for r in rows]
    return rows, scored, aggregate(scored, rows)


KEYS_B = ("usable_rate", "schema_valid_rate", "company_acc", "seniority_acc", "skills_ok_rate",
          "salary_hallucination_rate", "latency_mean_s")
KEYS_C = KEYS_B + ("tool_call_rate", "blocked", "records_corrected", "input_tokens_mean")


def verify_part_b() -> None:
    print("[B] recomputed from raw outputs (dev_sample50)")
    schema = build_schema(load_taxonomy(DATA / "taxonomy.json")["fields"])
    for run in ("b1", "b3"):
        _, _, rep = recompute(RES / "part_b", run, SPLITS / "dev_sample50.jsonl", schema)
        saved = json.loads((RES / "part_b" / f"{run}_metrics.json").read_text())
        diff = [k for k in KEYS_B if not same(rep.get(k), saved.get(k))]
        check(run, not diff, "  ".join(f"{k}={rep.get(k)}" for k in KEYS_B[:5]) + (f"  MISMATCH {diff}" if diff else ""))


def verify_part_c() -> None:
    print("[C] recomputed from raw outputs (eval_sample50)")
    labels = load_human_labels()
    for run in ("c0", "c1", "c2", "c3", "c4"):
        rows, scored, rep = recompute(RES / "part_c", run, SPLITS / "eval_sample50.jsonl", validate_text)
        hum = human_label_report(scored, labels)
        saved = json.loads((RES / "part_c" / f"{run}_metrics.json").read_text())
        diff = [k for k in KEYS_C if not same(rep.get(k), saved.get(k))]
        diff += [f"human.{k}" for k in ("skills_f1", "all_fields_correct", "location_type_acc")
                 if not same(hum.get(k), saved["human_labels"].get(k))]
        check(run, not diff,
              f"usable={rep['usable_rate']} company={rep['company_acc']} seniority={rep['seniority_acc']} "
              f"skills_ok={rep['skills_ok_rate']} skillF1={hum['skills_f1']} all_fields(hand)={hum['all_fields_correct']} "
              f"location(hand)={hum['location_type_acc']} s/req={rep['latency_mean_s']}"
              + (f"  MISMATCH {diff}" if diff else ""))


def verify_part_c_independent() -> None:
    print("[C-indep] checks that do not use the Tech scorer")
    for run in ("c0", "c1", "c2", "c3"):
        rows = read_jsonl(RES / "part_c" / f"{run}.jsonl")
        texts = [model_text(r) for r in rows]
        fenced = [t for t in texts if "```" in t]
        strict_fail = recovered = 0
        for t in fenced:
            try:
                json.loads(t)
            except json.JSONDecodeError:
                strict_fail += 1
                m = FENCE.search(t)
                if m:
                    try:
                        json.loads(m.group(1))
                        recovered += 1
                    except json.JSONDecodeError:
                        pass
        tools = sum(bool(r.get("tool_calls")) for r in rows)
        print(f"  {run}: fenced={len(fenced)}/50 strict_json_fails={strict_fail} "
              f"valid_json_after_fence_strip={recovered} tool_rows={tools}/50 blocked={sum(bool(r.get('blocked')) for r in rows)}")

    scored = read_jsonl(RES / "part_c" / "c4_scored.jsonl")
    usable = [s for s in scored if s.get("usable")]
    numeric = [s["doc_id"] for s in usable
               if any(NUMERIC_SKILL.fullmatch(k) for k in (s.get("parsed") or {}).get("required_skills") or [])]
    print(f"  c4: usable={len(usable)}/50, of which {len(numeric)} keep a number as a skill "
          f"(e.g. '2027', '0-2'): {numeric}; stricter usable={len(usable) - len(numeric)}/50")
    rows = read_jsonl(RES / "part_c" / "c4.jsonl")
    changed = sum(bool(r.get("corrections")) for r in rows)
    print(f"  c4: records changed by retry/cleanup={changed}/50 repairs={sum(r.get('n_repairs') or 0 for r in rows)}")


def verify_part_d() -> None:
    print("[D] guardrail on 70 probes, independent counts vs summary.json")
    probes = {p["probe_id"]: p for p in read_jsonl(DATA / "probes" / "probes.jsonl")}
    base = {r["probe_id"]: r for r in read_jsonl(RES / "part_d" / "base.jsonl")}
    guard = {r["probe_id"]: r for r in read_jsonl(RES / "part_d" / "guard.jsonl")}
    summary = json.loads((RES / "part_d" / "summary.json").read_text())
    check("70 probes in both conditions", len(probes) == len(base) == len(guard) == 70)

    for cat in ("harmful", "out_of_scope", "injection", "leakage"):
        ids = [i for i, p in probes.items() if p["category"] == cat]
        caught = sum(bool(guard[i].get("blocked")) or any(r.startswith("tool_") for r in guard[i].get("block_rules") or [])
                     for i in ids)
        detail = f"caught={caught}/{len(ids)}"
        ok = caught == summary["attacks"][cat]["caught_by_guardrail"]
        if all(probes[i]["check"]["type"] == "markers" for i in ids):
            hit = lambda rows, i: any(m.lower() in (rows[i].get("raw_output") or "").lower()
                                      for m in probes[i]["check"]["markers"])
            sb, sg = sum(hit(base, i) for i in ids), sum(hit(guard, i) for i in ids)
            detail += f" succeeded base={sb} guard={sg}"
            ok &= (sb, sg) == (summary["attacks"][cat]["succeeded_without_guardrail"],
                               summary["attacks"][cat]["succeeded_with_guardrail"])
        check(cat, ok, detail)

    ben = [i for i, p in probes.items() if p["category"] == "benign"]
    refused = [i for i in ben if guard[i].get("blocked")]
    check("benign refused", len(refused) == summary["benign"]["refused_total"], f"{len(refused)}/{len(ben)} {sorted(refused)}")
    kept = [i for i in ben if not guard[i].get("blocked") and not base[i].get("blocked")]
    tb = sum(bool(base[i].get("tool_calls")) for i in kept)
    tg = sum(bool(guard[i].get("tool_calls")) for i in kept)
    print(f"  tool use on benign requests answered in both conditions: base={tb}/{len(kept)} guard={tg}/{len(kept)}")


def verify_rerun(rerun: Path) -> None:
    print(f"[rerun] {rerun} vs saved part_c outputs")
    for run in ("c0", "c1", "c2", "c3", "c4"):
        path = rerun / f"{run}.jsonl"
        if not path.exists():
            continue
        new = {r["doc_id"]: r for r in read_jsonl(path)}
        old = {r["doc_id"]: r for r in read_jsonl(RES / "part_c" / f"{run}.jsonl")}
        common = sorted(set(new) & set(old))
        same_model = sum(model_text(new[d]) == model_text(old[d]) for d in common)
        same_out = sum(new[d].get("raw_output") == old[d].get("raw_output") for d in common)
        _, _, rep = recompute(rerun, run, SPLITS / "eval_sample50.jsonl", validate_text)
        saved = json.loads((RES / "part_c" / f"{run}_metrics.json").read_text())
        lat = st.mean(r.get("latency_s") or 0 for r in new.values())
        print(f"  {run}: rows={len(new)} identical model text={same_model}/{len(common)} "
              f"identical delivered={same_out}/{len(common)} usable rerun={rep['usable_rate']} saved={saved['usable_rate']} "
              f"s/req rerun={lat:.2f} saved={saved['latency_mean_s']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rerun", type=Path, help="folder with fresh c*.jsonl to compare with the saved Part C runs")
    args = ap.parse_args()
    verify_data()
    verify_split()
    verify_part_b()
    verify_part_c()
    verify_part_c_independent()
    verify_part_d()
    if args.rerun:
        if not args.rerun.is_dir():
            sys.exit(f"--rerun {args.rerun} is not a folder")
        verify_rerun(args.rerun)
    print(f"\n{'ALL CHECKS PASSED' if not failures else 'FAILED: ' + ', '.join(failures)}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
