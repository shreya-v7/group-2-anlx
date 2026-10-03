"""Pydantic contract for the SimplifyJobs extraction API (Part C, Eval 1).

Differences from the AS01 JSON schema, each driven by a Part B failure:

* salary_min, company, location_type are Optional. Part B showed a required
  field forces the model to write *something* (null, "Not specified", a guess)
  when the posting has no answer; the contract now has a legal way to say
  "not stated".
* required_skills defaults to []. Compensation-only fragments have no skills,
  and the model invented generic ones when the list felt mandatory.
* Placeholder strings ("Not specified", "N/A", "") normalize to None, and
  money strings ("$140,000", "140k") normalize to int, at the API boundary.
* extra="forbid": unexpected keys (e.g. salary_max) are rejected.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from simplifyjobs.corpus import ROOT
from simplifyjobs.extraction_schema import extract_json

SCHEMA_PATH = ROOT / "data" / "pydantic_models" / "job_posting.json"

RoleCategory = Literal[
    "software_engineering", "data_science_ml", "devops_infra",
    "product_management", "design_ux", "security", "other",
]
Seniority = Literal[
    "intern", "entry", "mid", "senior", "staff", "principal", "manager", "director", "vp",
]
LocationType = Literal["remote", "hybrid", "onsite"]

MAX_SKILLS = 20
_PLACEHOLDERS = {"", "none", "null", "n/a", "na", "not specified", "not stated", "unknown", "not mentioned", "-"}
_MONEY = re.compile(r"^\$?\s*(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*([kK])?$")


class JobPosting(BaseModel):
    """One extracted job posting record, as written to the SimplifyJobs database."""

    model_config = ConfigDict(extra="forbid")

    role_category: RoleCategory = Field(description="Primary functional category of the role.")
    seniority: Seniority = Field(
        description="Seniority level. New grad, new college grad, early career, entry level, "
        "junior, associate, and level I or 1 roles are 'entry'."
    )
    required_skills: list[str] = Field(
        default_factory=list,
        max_length=MAX_SKILLS,
        description="Technology, language, framework, or tool names copied exactly as written "
        "in the posting, e.g. 'Python', 'SQL', 'AWS'. Empty list if the posting names none. "
        "No soft skills, no degrees, no full sentences.",
    )
    location_type: Optional[LocationType] = Field(
        default=None, description="Work arrangement; null if the posting does not state it."
    )
    company: Optional[str] = Field(
        default=None, description="Hiring company name as written; null if the text does not name it."
    )
    salary_min: Optional[int] = Field(
        default=None,
        ge=10_000,
        le=1_000_000,
        description="Minimum annual base salary in USD; null if no salary is stated.",
    )

    @field_validator("company", "location_type", "salary_min", mode="before")
    @classmethod
    def _placeholder_to_none(cls, value):
        if isinstance(value, str) and value.strip().lower() in _PLACEHOLDERS:
            return None
        return value

    @field_validator("salary_min", mode="before")
    @classmethod
    def _money_string_to_int(cls, value):
        if isinstance(value, str):
            match = _MONEY.match(value.strip())
            if match:
                number = float(match.group(1).replace(",", ""))
                return int(number * 1000) if match.group(2) else int(number)
        return value

    @field_validator("required_skills")
    @classmethod
    def _strip_skills(cls, skills: list[str]) -> list[str]:
        return [s.strip() for s in skills if s and s.strip()]


def export_schema(path: Path = SCHEMA_PATH) -> Path:
    """Write the JSON Schema LLMBOX's structured_output mode appends to the prompt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(JobPosting.model_json_schema(), separators=(",", ":")), encoding="utf-8")
    return path


def validate_text(text: str) -> tuple[dict | None, str | None]:
    """Parse model output into a normalized JobPosting dict, or return the error."""
    try:
        raw = extract_json(text)
    except ValueError as exc:
        return None, str(exc)
    if not isinstance(raw, dict):
        return None, "output is not a JSON object"
    try:
        return JobPosting.model_validate(raw).model_dump(), None
    except ValidationError as exc:
        first = exc.errors()[0]
        loc = ".".join(str(p) for p in first["loc"]) or "$"
        return raw, f"$.{loc}: {first['msg']}"


def contract_error(text: str) -> str | None:
    """LLMBOX validator hook (`module:function` returning an error or None)."""
    return validate_text(text)[1]
