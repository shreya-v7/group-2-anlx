"""Fence-tolerant JobFacts parsing (Oct 3 follow-up; the frozen AS02 code is unchanged).

Rule: if the entire model output is exactly one complete Markdown code fence
(optionally labelled json), validate the text inside it. Anything else is
validated as-is, so partial, unclosed or ambiguous output still fails. The raw
output is always kept by the caller.
"""
import re

from career_api import parse_facts

WHOLE_FENCE = re.compile(r'\s*```(?:json)?[ \t]*\n(.*?)\n?```\s*', re.S)


def strip_whole_fence(text):
    """Return (inner_text, True) for one complete enclosing fence, else (text, False)."""
    match = WHOLE_FENCE.fullmatch(text or '')
    if match and '```' not in match.group(1):
        return match.group(1).strip(), True
    return text, False


def parse_facts_tolerant(text):
    inner, stripped = strip_whole_fence(text)
    facts, error = parse_facts(inner)
    return facts, error, stripped
