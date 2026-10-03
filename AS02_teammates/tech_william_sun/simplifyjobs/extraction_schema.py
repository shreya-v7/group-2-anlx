"""Extraction contract for SimplifyJobs, carried over from Assignment 1.

The schema is built from taxonomy.json exactly as in AS01 (course/schema.py),
so Part B/C/D results are comparable with the AS01 Phi evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
    "object": (dict,),
    "array": (list,),
    "null": (type(None),),
}


def load_taxonomy(path: str | Path) -> dict:
    taxonomy = json.loads(Path(path).read_text(encoding="utf-8"))
    if not taxonomy.get("fields"):
        raise SystemExit(f"{path} defines no fields -- nothing to extract.")
    return taxonomy


def build_schema(fields: dict) -> dict:
    """JSON schema the model output is held to (identical to AS01)."""
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
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def describe_fields(fields: dict) -> str:
    """Field list as prompt text (AS01 v1_original wording)."""
    lines = []
    for name, spec in fields.items():
        kind, note = spec.get("kind", "label"), spec.get("description", "")
        if spec.get("values"):
            rule = f"one of {', '.join(spec['values'])}."
        elif kind == "list":
            rule = "a list of strings."
        elif kind == "number":
            rule = "a number taken from the document."
        else:
            rule = "a short value taken from the document."
        lines.append(f"- {name}: " + (f"{note} -- {rule}" if note else rule))
    return "\n".join(lines)


def baseline_system_prompt(taxonomy: dict) -> str:
    """AS01 v1_original prompt plus a one-line JSON instruction.

    Held constant across Part B and Part C so that differences between runs
    come from hyperparameters or API features, not prompt wording.
    """
    subject = taxonomy.get("subtopic") or taxonomy.get("group_topic") or "our corpus"
    return (
        f"You label documents from a corpus about {subject}. "
        f"Fill in every field:\n{describe_fields(taxonomy['fields'])}\n"
        "Respond with a JSON object containing these fields."
    )


def validate_against_schema(value: Any, schema: dict, path: str = "$") -> str | None:
    """Validate against the course JSON Schema subset. None if valid."""
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
    """Pull a JSON value out of model output wrapped in prose or fences."""
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = (
            candidate.split("```")[1] if "```" in candidate[3:] else candidate[3:]
        )
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
                        return json.loads(candidate[start : i + 1])
                    except json.JSONDecodeError:
                        break
    raise ValueError("no JSON value found in model output")
