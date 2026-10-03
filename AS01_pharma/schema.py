"""Define and validate structured data for Assignment 1.

Two contracts live here, because both are "what shape must this JSON have":

1. The corpus record -- what every line of corpus.jsonl must look like.
   `check_corpus.py` validates against it and `corpus_stats.py` reads it.

2. The extraction output -- the JSON schema Phi (or a hosted LLM) is held to.
   It is built from taxonomy.json by `build_schema`, and model output is
   checked against it with `extract_json` + `validate_against_schema`.

If you change a field here, everything downstream sees the change. Do not
redefine either contract anywhere else.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

# --------------------------------------------------------------------------
# 1. The corpus record contract
# --------------------------------------------------------------------------

MODALITIES = ("text", "table", "mixed")

#: field name -> (required, accepted python types, human description)
FIELDS: dict[str, tuple[bool, tuple[type, ...], str]] = {
    "doc_id": (True, (str,), "stable unique identifier"),
    "source_url": (True, (str,), "where it came from, resolvable"),
    "retrieved_at": (True, (str,), "ISO 8601 timestamp"),
    "modality": (True, (str,), f"one of {', '.join(MODALITIES)}"),
    "raw_text": (True, (str, type(None)), "extracted text, null for a pure table"),
    "table_json": (False, (dict, list, type(None)), "normalized table, null for pure text"),
    "license_note": (True, (str,), "terms under which you hold this document"),
    "metadata": (True, (dict,), "free-form object, document-type dependent"),
}

# Corpus-level requirements from the brief.
MIN_DOCS = 150
MAX_DOCS = 300
MIN_TABULAR_SHARE = 0.20
MIN_LICENSE_NOTE_CHARS = 10
MIN_RAW_TEXT_CHARS = 50


@dataclass
class Problem:
    """One thing wrong with one record, in terms a student can act on."""

    line: int | None
    doc_id: str | None
    field: str | None
    message: str

    def __str__(self) -> str:
        where = f"line {self.line}" if self.line is not None else "corpus"
        if self.doc_id:
            where += f" (doc_id={self.doc_id!r})"
        what = f"{self.field}: " if self.field else ""
        return f"{where} -- {what}{self.message}"


def _is_iso8601(value: str) -> bool:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        datetime.fromisoformat(text)
        return True
    except ValueError:
        return False


def is_tabular(record: dict) -> bool:
    """Does this record count toward the 20% tabular requirement?"""
    return record.get("modality") in ("table", "mixed") and record.get("table_json") not in (
        None, {}, [],
    )


def validate_record(record: Any, line: int | None = None) -> list[Problem]:
    """Check one record against the contract. Returns an empty list if it is fine."""
    problems: list[Problem] = []

    if not isinstance(record, dict):
        return [Problem(line, None, None, f"expected a JSON object, got {type(record).__name__}")]

    doc_id = record.get("doc_id") if isinstance(record.get("doc_id"), str) else None

    for name, (required, types, description) in FIELDS.items():
        if name not in record:
            if required:
                problems.append(Problem(line, doc_id, name, f"missing required field ({description})"))
            continue
        value = record[name]
        if not isinstance(value, types):
            allowed = " or ".join("null" if t is type(None) else t.__name__ for t in types)
            problems.append(Problem(line, doc_id, name,
                                    f"expected {allowed}, got {type(value).__name__}"))
            continue
        if required and isinstance(value, str) and not value.strip():
            problems.append(Problem(line, doc_id, name, "is empty"))

    unknown = set(record) - set(FIELDS)
    if unknown:
        problems.append(Problem(line, doc_id, None,
                                "unexpected field(s): " + ", ".join(sorted(unknown))
                                + " -- put document-specific fields inside metadata"))

    modality = record.get("modality")
    if isinstance(modality, str) and modality not in MODALITIES:
        problems.append(Problem(line, doc_id, "modality",
                                f"{modality!r} is not one of {', '.join(MODALITIES)}"))

    retrieved_at = record.get("retrieved_at")
    if isinstance(retrieved_at, str) and retrieved_at.strip() and not _is_iso8601(retrieved_at):
        problems.append(Problem(line, doc_id, "retrieved_at",
                                f"{retrieved_at!r} is not ISO 8601 "
                                "(try datetime.now(timezone.utc).isoformat())"))

    url = record.get("source_url")
    if isinstance(url, str) and url.strip() and not url.startswith(("http://", "https://")):
        problems.append(Problem(line, doc_id, "source_url", "should be an http(s) URL"))

    note = record.get("license_note")
    if isinstance(note, str) and 0 < len(note.strip()) < MIN_LICENSE_NOTE_CHARS:
        problems.append(Problem(line, doc_id, "license_note",
                                f"is {len(note.strip())} characters -- name the terms, "
                                "not just the word 'public'"))

    text = record.get("raw_text")
    if isinstance(text, str) and 0 < len(text.strip()) < MIN_RAW_TEXT_CHARS:
        problems.append(Problem(line, doc_id, "raw_text",
                                f"is only {len(text.strip())} characters"))
    if text is None and modality != "table":
        problems.append(Problem(line, doc_id, "raw_text",
                                f"is null but modality is {modality!r}"))

    if modality in ("table", "mixed") and record.get("table_json") in (None, {}, []):
        problems.append(Problem(line, doc_id, "table_json",
                                f"is empty but modality is {modality!r}"))
    if modality == "text" and record.get("table_json") not in (None, {}, []):
        problems.append(Problem(line, doc_id, "table_json",
                                "is populated but modality is 'text'"))

    return problems


@dataclass
class LoadResult:
    records: list[dict] = field(default_factory=list)
    problems: list[Problem] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def iter_jsonl(path: str | Path) -> Iterator[tuple[int, Any, str | None]]:
    """Yield (line_number, parsed_object_or_None, error_or_None) for each line."""
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                yield line_no, json.loads(raw), None
            except json.JSONDecodeError as exc:
                yield line_no, None, f"invalid JSON ({exc.msg} at column {exc.colno})"


def load_corpus(path: str | Path, validate: bool = True) -> LoadResult:
    """Load corpus.jsonl. Collects every problem rather than stopping at the first."""
    result = LoadResult()
    seen: dict[str, int] = {}

    for line_no, obj, error in iter_jsonl(path):
        if error:
            result.problems.append(Problem(line_no, None, None, error))
            continue
        if validate:
            result.problems.extend(validate_record(obj, line_no))
        if isinstance(obj, dict):
            doc_id = obj.get("doc_id")
            if isinstance(doc_id, str):
                if doc_id in seen:
                    result.problems.append(Problem(line_no, doc_id, "doc_id",
                                                   f"duplicate of line {seen[doc_id]}"))
                else:
                    seen[doc_id] = line_no
            result.records.append(obj)
    return result


def write_corpus(path: str | Path, records: list[dict]) -> None:
    """Write records as JSONL. Round-trips with load_corpus."""
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------
# 2. The extraction schema, built from taxonomy.json
#
#    field kinds:
#      label   one string. "values" constrains it to a list; leave "values"
#              empty and the field is open -- the model answers freely.
#      list    a list of strings (entities, topics mentioned, column names...).
#      number  a number (fiscal year, headcount, amount...).
#    every field is required unless it says "required": false.
# --------------------------------------------------------------------------

def load_taxonomy(path: str | Path) -> dict:
    taxonomy = json.loads(Path(path).read_text(encoding="utf-8"))
    if not taxonomy.get("fields"):
        raise SystemExit(f"{path} defines no fields -- nothing to extract.")
    return taxonomy


def build_schema(fields: dict) -> dict:
    """The JSON schema the model is held to, and the shape human labels must match."""
    properties, required = {}, []
    for name, spec in fields.items():
        kind = spec.get("kind", "label")
        if kind == "list":
            prop: dict = {"type": "array", "items": {"type": "string"}}
        elif kind == "number":
            prop = {"type": "number"}
        else:
            prop = {"type": "string"}
            if spec.get("values"):
                prop["enum"] = list(spec["values"])
        properties[name] = prop
        if spec.get("required", True):
            required.append(name)
    return {"type": "object", "properties": properties, "required": required,
            "additionalProperties": False}


# --------------------------------------------------------------------------
# 3. Validating what a model returned
# --------------------------------------------------------------------------

_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,), "number": (int, float), "integer": (int,), "boolean": (bool,),
    "object": (dict,), "array": (list,), "null": (type(None),),
}


def validate_against_schema(value: Any, schema: dict, path: str = "$") -> str | None:
    """Validate against the JSON Schema subset used in this course.

    Supports type, properties, required, items, enum, additionalProperties.
    Returns None if valid, else a one-line human-readable reason.
    """
    expected = schema.get("type")
    if expected:
        types = _TYPES.get(expected, ())
        if expected in ("number", "integer") and isinstance(value, bool):
            return f"{path}: expected {expected}, got boolean"
        if types and not isinstance(value, types):
            return f"{path}: expected {expected}, got {type(value).__name__}"

    if "enum" in schema and value not in schema["enum"]:
        allowed = ", ".join(repr(v) for v in schema["enum"])
        return f"{path}: {value!r} not in [{allowed}]"

    if isinstance(value, dict) and (expected == "object" or "properties" in schema):
        for key in schema.get("required", []):
            if key not in value:
                return f"{path}.{key}: missing required key"
        props = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            extra = set(value) - set(props)
            if extra:
                return f"{path}: unexpected key(s) {', '.join(sorted(extra))}"
        for key, sub in props.items():
            if key in value:
                problem = validate_against_schema(value[key], sub, f"{path}.{key}")
                if problem:
                    return problem

    if isinstance(value, list) and "items" in schema:
        for i, item in enumerate(value):
            problem = validate_against_schema(item, schema["items"], f"{path}[{i}]")
            if problem:
                return problem

    return None


def extract_json(text: str) -> Any:
    """Pull a JSON value out of model output that may be wrapped in prose or fences.

    Small models ignore "respond with JSON only" at a meaningful rate. Strip
    markdown fences, then take the first balanced {...} or [...] block.
    """
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.split("```")[1] if "```" in candidate[3:] else candidate[3:]
        if candidate.lstrip().lower().startswith("json"):
            candidate = candidate.lstrip()[4:]
        candidate = candidate.strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    for opener, closer in (("{", "}"), ("[", "]")):
        start = candidate.find(opener)
        if start == -1:
            continue
        depth, in_string, escape = 0, False, False
        for i in range(start, len(candidate)):
            char = candidate[i]
            if escape:
                escape = False
                continue
            if char == "\\":
                escape = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == opener:
                depth += 1
            elif char == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(candidate[start:i + 1])
                    except json.JSONDecodeError:
                        break
    raise ValueError("no JSON value found in model output")
