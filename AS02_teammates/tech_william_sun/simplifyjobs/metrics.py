"""Evaluation metrics for SimplifyJobs extraction.

Evaluation criterion (Part B): a record is USABLE (can enter the job database
without human correction) when all of the following hold:

  1. schema_valid      output parses and matches the AS01 extraction schema
  2. company_correct   matches the scraped company name
  3. seniority_correct matches the reference ("entry" for this corpus)
  4. skills_ok         every extracted skill appears verbatim in the posting
                       AND is a short skill name (<= 4 words), not a sentence
  5. salary_ok         salary_min is omitted, or equals a number in the posting

role_category and location_type are reported as field accuracy but are not
part of USABLE: role has no reliable automatic reference, and most fragments
do not state location, so the reference is often unanswerable from the input.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from collections.abc import Callable

from simplifyjobs.corpus import reference_labels, source_text
from simplifyjobs.extraction_schema import extract_json, validate_against_schema

Validator = Callable[[str], tuple]  # text -> (parsed dict | None, error | None)

MAX_SKILL_WORDS = 4
_COMPANY_SUFFIX = re.compile(r"\b(inc|llc|ltd|corp|corporation|co|company|plc)\b\.?", re.IGNORECASE)
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_401K = re.compile(r"401\s*\(?k\)?", re.IGNORECASE)
_MONEY_K = re.compile(r"(?<![\w.])\$?\s?(\d{2,3}(?:\.\d+)?)\s?[kK](?![a-zA-Z])")
_MONEY_FULL = re.compile(r"(?<![\w.])\$?\s?(\d{1,3}(?:,\d{3})+|\d{5,7})(?:\.\d+)?(?![\d])")
_LOCATION_WORDS = re.compile(r"\b(remote|hybrid|on-?site|in[- ]office|in person)\b", re.IGNORECASE)


def normalize_company(name: str | None) -> str:
    if not name:
        return ""
    return _NON_ALNUM.sub(" ", _COMPANY_SUFFIX.sub("", name.lower())).strip()


def company_match(pred: str | None, ref: str | None) -> bool:
    p, r = normalize_company(pred), normalize_company(ref)
    if not p or not r:
        return False
    return p == r or (len(p) >= 3 and p in r) or (len(r) >= 3 and r in p)


def salary_numbers(text: str) -> set[int]:
    """Every annual-salary-like number in the text (>= 10,000), 401k excluded."""
    text = _401K.sub(" ", text)
    values = {int(float(m) * 1000) for m in _MONEY_K.findall(text)}
    values |= {int(m.replace(",", "")) for m in _MONEY_FULL.findall(text)}
    return {v for v in values if v >= 10_000}


def skill_grounded(skill: str, text_lower: str) -> bool:
    s = skill.strip().lower()
    if not s:
        return False
    return re.search(r"(?<!\w)" + re.escape(s) + r"(?!\w)", text_lower) is not None


def as01_validator(schema: dict) -> Validator:
    """Part B contract: the AS01 JSON schema (salary must be a number or absent)."""

    def validate(text: str):
        try:
            parsed = extract_json(text)
        except ValueError as exc:
            return None, str(exc)
        if not isinstance(parsed, dict):
            return None, "output is not a JSON object"
        return parsed, validate_against_schema(parsed, schema)

    return validate


def evaluate_record(record: dict, raw_output: str, contract: dict | Validator) -> dict:
    """Score one output against the posting and reference labels.

    `contract` is the AS01 JSON schema (Part B) or a validator callable such as
    the Pydantic contract (Part C)."""
    validate = as01_validator(contract) if isinstance(contract, dict) else contract
    ref = reference_labels(record)
    text = source_text(record)
    text_lower = text.lower()
    source_salaries = salary_numbers(text)

    result = {
        "doc_id": record["doc_id"],
        "parse_ok": False,
        "schema_valid": False,
        "schema_error": None,
        "company_in_source": normalize_company(ref["company"]) in _NON_ALNUM.sub(" ", text_lower),
        "location_in_source": bool(_LOCATION_WORDS.search(text)),
        "salary_in_source": bool(source_salaries),
    }

    parsed, result["schema_error"] = validate(raw_output)
    result["parse_ok"] = isinstance(parsed, dict)
    result["schema_valid"] = result["parse_ok"] and result["schema_error"] is None
    if not isinstance(parsed, dict):
        parsed = {}

    skills = parsed.get("required_skills")
    skills = [s for s in skills if isinstance(s, str)] if isinstance(skills, list) else []
    grounded = [skill_grounded(s, text_lower) for s in skills]
    atomic = [len(s.split()) <= MAX_SKILL_WORDS for s in skills]

    salary = parsed.get("salary_min")
    salary_returned = isinstance(salary, (int, float)) and not isinstance(salary, bool)
    salary_grounded = salary_returned and int(salary) in source_salaries
    salary_hallucinated = salary_returned and not salary_grounded

    result.update(
        {
            "parsed": parsed,
            "company_pred": parsed.get("company"),
            "company_correct": company_match(parsed.get("company"), ref["company"]),
            "company_null": parsed.get("company") in (None, ""),
            "location_unsupported": bool(parsed.get("location_type")) and not result["location_in_source"],
            "seniority_pred": parsed.get("seniority"),
            "seniority_correct": parsed.get("seniority") == ref["seniority"],
            "location_pred": parsed.get("location_type"),
            "location_ref": ref["location_type"],
            "location_correct": (
                parsed.get("location_type") == ref["location_type"] if ref["location_type"] else None
            ),
            "n_skills": len(skills),
            "n_skills_grounded": sum(grounded),
            "n_skills_atomic": sum(atomic),
            "ungrounded_skills": [s for s, g in zip(skills, grounded) if not g],
            "skills_ok": result["parse_ok"] and all(grounded) and all(atomic),
            "salary_returned": salary_returned,
            "salary_hallucinated": salary_hallucinated,
            "salary_ok": result["parse_ok"] and not salary_hallucinated,
        }
    )
    result["usable"] = all(
        result[k]
        for k in ("schema_valid", "company_correct", "seniority_correct", "skills_ok", "salary_ok")
    )
    return result


def _rate(values: list[bool]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(q * len(ordered)))], 3)


def aggregate(scored: list[dict], runs: list[dict]) -> dict:
    """Aggregate per-record scores and per-request cost into one report."""
    total_skills = sum(r["n_skills"] for r in scored)
    no_salary = [r for r in scored if not r["salary_in_source"]]
    has_loc = [r for r in scored if r["location_correct"] is not None]
    latency = [r["latency_s"] for r in runs]
    tok_in = [r["input_tokens"] for r in runs]
    tok_out = [r["output_tokens"] for r in runs]

    return {
        "n": len(scored),
        "usable_rate": _rate([r["usable"] for r in scored]),
        "parse_rate": _rate([r["parse_ok"] for r in scored]),
        "schema_valid_rate": _rate([r["schema_valid"] for r in scored]),
        "schema_errors": Counter(
            re.sub(r"'[^']*'", "'…'", r["schema_error"]) for r in scored if r["schema_error"]
        ).most_common(5),
        "company_acc": _rate([r["company_correct"] for r in scored]),
        "company_acc_when_in_source": _rate([r["company_correct"] for r in scored if r["company_in_source"]]),
        "company_acc_when_not_in_source": _rate(
            [r["company_correct"] for r in scored if not r["company_in_source"]]
        ),
        "n_company_not_in_source": sum(not r["company_in_source"] for r in scored),
        "company_fabricated_rate": _rate([not r["company_correct"] and not r["company_null"] for r in scored]),
        "company_null_when_not_in_source": _rate(
            [r["company_null"] for r in scored if not r["company_in_source"]]
        ),
        "n_location_unsupported": sum(r["location_unsupported"] for r in scored),
        "seniority_acc": _rate([r["seniority_correct"] for r in scored]),
        "seniority_errors": Counter(str(r["seniority_pred"]) for r in scored if not r["seniority_correct"]).most_common(),
        "location_acc": _rate([r["location_correct"] for r in has_loc]),
        "n_location_scored": len(has_loc),
        "skills_total": total_skills,
        "skill_grounding_precision": round(sum(r["n_skills_grounded"] for r in scored) / total_skills, 4)
        if total_skills
        else None,
        "skill_atomic_rate": round(sum(r["n_skills_atomic"] for r in scored) / total_skills, 4)
        if total_skills
        else None,
        "skills_ok_rate": _rate([r["skills_ok"] for r in scored]),
        "salary_hallucination_rate": _rate([r["salary_hallucinated"] for r in no_salary]),
        "n_salary_hallucinated": sum(r["salary_hallucinated"] for r in scored),
        "n_no_salary_in_source": len(no_salary),
        "latency_mean_s": round(statistics.mean(latency), 3) if latency else None,
        "latency_p50_s": _pct(latency, 0.5),
        "latency_p95_s": _pct(latency, 0.95),
        "input_tokens_mean": round(statistics.mean(tok_in), 1) if tok_in else None,
        "output_tokens_mean": round(statistics.mean(tok_out), 1) if tok_out else None,
        "output_tokens_p95": _pct([float(t) for t in tok_out], 0.95),
        "output_tokens_max": max(tok_out) if tok_out else None,
        "n_hit_max_new_tokens": sum(r["hit_token_cap"] for r in runs),
        **pipeline_stats(runs),
    }


def pipeline_stats(runs: list[dict]) -> dict:
    """Tool use, repair, verifier, and guardrail statistics (Part C/D runs only)."""
    if not any("n_generations" in r for r in runs):
        return {}
    out = {"generations_per_request": round(statistics.mean(r["n_generations"] for r in runs), 2)}
    if any(r.get("tools_enabled") for r in runs):
        # Denominator: requests that reached the model (not refused at input).
        reached = [r for r in runs if r.get("block_stage") != "input"]
        calls = [r.get("tool_calls") or [] for r in reached]
        out["tool_call_rate"] = _rate([bool(c) for c in calls])
        out["tool_args_ok_rate"] = _rate(
            [all(str(c.get("arguments", {}).get("doc_id")) == r["doc_id"] for c in cs)
             for r, cs in zip(reached, calls) if cs]
        )
    if any("n_repairs" in r for r in runs):
        out["repair_rate"] = _rate([r.get("n_repairs", 0) > 0 for r in runs])
    if any(r.get("corrections") for r in runs):
        kinds = Counter(c.split(":")[0] for r in runs for c in r.get("corrections") or [])
        out["records_corrected"] = _rate([bool(r.get("corrections")) for r in runs])
        out["corrections"] = dict(kinds.most_common())
    if any("blocked" in r for r in runs):
        out["blocked"] = sum(r.get("blocked", False) for r in runs)
        out["block_rules"] = dict(Counter(rule for r in runs for rule in r.get("block_rules") or []))
        guard = [r["guardrail_latency_s"] for r in runs if r.get("guardrail_latency_s") is not None]
        out["guardrail_latency_mean_s"] = round(statistics.mean(guard), 3) if guard else None
        out["guardrail_tokens_mean"] = round(
            statistics.mean(r.get("guardrail_tokens", 0) for r in runs), 1
        )
    return out


HUMAN_FIELDS = ("role_category", "seniority", "required_skills", "location_type", "company", "salary_min")


def human_label_report(scored: list[dict], human_labels: dict[str, dict]) -> dict:
    """Field accuracy against AS01 human labels, using AS01 compare_one semantics."""
    rows = [r for r in scored if r["doc_id"] in human_labels]
    if not rows:
        return {}
    hits = Counter()
    tp = n_pred = n_gold = all_ok = 0
    for r in rows:
        gold, pred = human_labels[r["doc_id"]], r.get("parsed") or {}
        ok_all = True
        for name in HUMAN_FIELDS:
            if name == "required_skills":
                g = {str(v).strip().lower() for v in gold.get(name) or []}
                p = {str(v).strip().lower() for v in pred.get(name) or [] if isinstance(v, str)}
                tp, n_pred, n_gold = tp + len(g & p), n_pred + len(p), n_gold + len(g)
                ok = g == p
            else:
                g, p = gold.get(name), pred.get(name)
                ok = (g.strip().lower() == p.strip().lower()) if isinstance(g, str) and isinstance(p, str) else g == p
            hits[name] += ok
            ok_all &= ok
        all_ok += ok_all
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gold if n_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n": len(rows),
        "all_fields_correct": round(all_ok / len(rows), 4),
        **{f"{k}_acc": round(hits[k] / len(rows), 4) for k in HUMAN_FIELDS},
        "skills_precision": round(precision, 4),
        "skills_recall": round(recall, 4),
        "skills_f1": round(f1, 4),
    }
