"""Grounding verifier for the SimplifyJobs API (Part C, Eval 4).

Plugged into LLMBOX through `structured_output.postprocess`. It never adds
information the model or the posting index did not provide; it only removes
or corrects values that fail a check, and reports every correction so the
review queue can see what changed.

Checks (each maps to a Part B / spot-check failure):
  skills     keep a skill only if it appears verbatim in the posting as a short
             technical name; recover names from copied requirement sentences
             ("Experience with SQL, Python, or SAS" -> SQL, Python, SAS);
             drop soft skills and generic phrases ("problem-solving").
  salary     keep salary_min only if the number appears in the posting.
  company    prefer the posting-index company returned by lookup_posting;
             otherwise null it when the posting does not name it.
  location   null it when the posting never states a work arrangement.
  seniority  titles with explicit new-grad markers are 'entry' (intern if the
             title says intern), overriding mid/senior guesses.
"""

from __future__ import annotations

import json
import re

from simplifyjobs.job_schema import MAX_SKILLS, validate_text
from simplifyjobs.metrics import normalize_company, salary_numbers

MAX_WORDS = 3
_SPLIT = re.compile(r",|;|/|\(|\)|\bor\b|\band\b|\bsuch as\b|\bincluding\b|\be\.g\.,?|\blike\b", re.IGNORECASE)
_FILLER = re.compile(
    r"^(strong|solid|sound|basic|good|deep|hands-on|working|some|prior|demonstrated|proven|"
    r"experience|experienced|knowledge|familiarity|familiar|proficiency|proficient|exposure|"
    r"understanding|expertise|skills?|ability|with|in|of|to|using|the|a|an|similar|other|modern)\b\s*",
    re.IGNORECASE,
)
_SOFT = {
    "communication", "communication skills", "problem-solving", "problem solving", "teamwork",
    "collaboration", "leadership", "analytical skills", "analytical", "critical thinking",
    "time management", "attention to detail", "programming languages", "programming",
    "software development", "coding", "presentation skills", "written", "verbal",
    "organizational skills", "interpersonal skills", "self-starter", "curiosity",
}
_LOCATION_WORDS = re.compile(r"\b(remote|hybrid|on-?site|in[- ]office|in person)\b", re.IGNORECASE)
_ENTRY_TITLE = re.compile(
    r"(?i:new (college )?grad|graduate|early career|entry[- ]level|\bjunior\b|\bassociate\b|"
    r"recent grad|campus)|\b(I|1)\b(?![\w.])"
)
_INTERN_TITLE = re.compile(r"\bintern(ship)?\b", re.IGNORECASE)


def _grounded(term: str, source_lower: str) -> bool:
    return re.search(r"(?<!\w)" + re.escape(term.lower()) + r"(?!\w)", source_lower) is not None


# Capitalized words that start requirement sentences or name degrees, not skills.
_NOT_SKILL = {
    "interest", "strong", "solid", "sound", "experience", "knowledge", "familiarity", "proficiency",
    "exposure", "understanding", "ability", "comfort", "comfortable", "built", "recently", "required",
    "preferred", "excellent", "must", "bachelor", "bachelor's", "master", "master's", "bs", "ms", "ba",
    "phd", "mba", "bachelors", "masters", "associate's", "doctorate", "gpa", "degree", "statistics", "mathematics", "math", "economics", "finance",
    "engineering", "physics", "science", "computer", "cs", "us", "usa", "usd", "the", "a", "an", "we",
    "you", "our", "this", "i", "graduating", "expected", "may", "december", "spring", "fall", "summer",
}


def _tech_word(word: str) -> bool:
    core = word.strip(".,:;!?\"'()[]")
    if not core or core.lower() in _NOT_SKILL:
        return False
    return bool(
        re.search(r"[0-9+#]", core)
        or re.search(r"\w\.\w|^\.\w", core)
        or re.search(r"[A-Z]", core[1:])
        or core[0].isupper()
    )


def _looks_technical(term: str) -> bool:
    return any(_tech_word(w) for w in term.split())


def _clean_chunk(chunk: str) -> str:
    chunk = chunk.strip(" .:-\"'")
    previous = None
    while previous != chunk:
        previous, chunk = chunk, _FILLER.sub("", chunk).strip(" .:-\"'")
    return chunk


def _tech_runs(chunk: str) -> list[str]:
    """Maximal runs of technical-looking words, e.g. 'systems programming on Linux' -> ['Linux']."""
    runs, current = [], []
    for word in chunk.split():
        if _tech_word(word):
            current.append(word.strip(".,:;!?\"'"))
        elif current:
            runs.append(" ".join(current))
            current = []
    if current:
        runs.append(" ".join(current))
    return [r for r in runs if len(r.split()) <= MAX_WORDS]


def _acceptable(term: str, source_lower: str) -> bool:
    return (
        bool(term)
        and len(term.split()) <= MAX_WORDS
        and term.lower() not in _SOFT
        and term.lower() not in _NOT_SKILL
        and _looks_technical(term)
        and _grounded(term, source_lower)
    )


def verify_skills(skills: list[str], source: str) -> tuple[list[str], list[str]]:
    source_lower = source.lower()
    kept, corrections, seen = [], [], set()

    def keep(term: str):
        if term.lower() not in seen:
            seen.add(term.lower())
            kept.append(term)

    for skill in skills:
        s = skill.strip()
        if _acceptable(s, source_lower):
            keep(s)
            continue
        recovered = []
        for chunk in _SPLIT.split(s):
            # From a sentence, keep only the technical words: "Sound C++ fundamentals" -> "C++".
            recovered += [r for r in _tech_runs(_clean_chunk(chunk)) if _acceptable(r, source_lower)]
        for term in recovered:
            keep(term)
        corrections.append(
            f"skill_split:{s[:40]}->{'|'.join(recovered)}" if recovered else f"skill_dropped:{s[:40]}"
        )
    # Splitting sentences can yield more names than the contract allows; the
    # verifier must never turn a valid answer into an invalid one.
    if len(kept) > MAX_SKILLS:
        corrections.append(f"skills_truncated:{len(kept)}->{MAX_SKILLS}")
        kept = kept[:MAX_SKILLS]
    return kept, corrections


def _title_from_context(context: dict) -> str | None:
    for call in context.get("tool_calls") or []:
        result = call.get("result") or {}
        if isinstance(result, dict) and result.get("role_title"):
            return result["role_title"]
    return None


def _company_from_context(context: dict) -> str | None:
    for call in context.get("tool_calls") or []:
        result = call.get("result") or {}
        if isinstance(result, dict) and result.get("company"):
            return result["company"]
    return None


def ground(answer: str, source: str, context: dict) -> tuple[str, list[str]]:
    """LLMBOX postprocess hook: (answer_text, source_text, context) -> (answer_text, corrections)."""
    parsed, error = validate_text(answer)
    if error or parsed is None:
        return answer, ["verifier_skipped:invalid_contract"]

    corrections: list[str] = []
    parsed["required_skills"], skill_fixes = verify_skills(parsed.get("required_skills") or [], source)
    corrections += skill_fixes

    salary = parsed.get("salary_min")
    if salary is not None and int(salary) not in salary_numbers(source):
        parsed["salary_min"] = None
        corrections.append("salary_ungrounded")

    tool_company = _company_from_context(context)
    if tool_company:
        if normalize_company(parsed.get("company")) != normalize_company(tool_company):
            parsed["company"] = tool_company
            corrections.append("company_from_index")
    elif parsed.get("company") and not _grounded(parsed["company"], source.lower()):
        parsed["company"] = None
        corrections.append("company_ungrounded")

    if parsed.get("location_type") and not _LOCATION_WORDS.search(source):
        parsed["location_type"] = None
        corrections.append("location_unstated")

    title = _title_from_context(context)
    if title:
        if _INTERN_TITLE.search(title) and parsed.get("seniority") != "intern":
            parsed["seniority"] = "intern"
            corrections.append("seniority_title_rule")
        elif _ENTRY_TITLE.search(title) and parsed.get("seniority") not in ("entry", "intern"):
            parsed["seniority"] = "entry"
            corrections.append("seniority_title_rule")

    return json.dumps(parsed, ensure_ascii=False), corrections
