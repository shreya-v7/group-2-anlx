#!/usr/bin/env python3
"""Phi-4-mini-instruct, loaded once and run in-process. No server, no network.

    from local_model import PhiClient

    phi = PhiClient()                                  # loads the weights once
    r = phi.structured(prompt, SCHEMA, system=SYSTEM)
    r.parsed, r.violation, r.latency_s

Weights are read straight from a local directory -- the same folder lab01's
chat.py uses -- and nothing is ever downloaded. Point at them with --model-path
or PHI_MODEL_PATH; otherwise the search list below finds them. Decoding is
greedy (do_sample=False) so the JSON is repeatable between runs, which is what
the evaluation needs.

    python local_model.py --check
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from schema import extract_json, validate_against_schema

DEFAULT_MAX_NEW_TOKENS = 512

#: Checked in order. First directory containing a config.json wins.
SEARCH_PATHS = [
    "models/phi-4-mini-instruct",
    "../models/phi-4-mini-instruct",
    "~/lab01/models/phi-4-mini-instruct",
    "~/Documents/lab01/models/phi-4-mini-instruct",
    "~/Downloads/course-setup/models/phi-4-mini-instruct",
]


def find_model_path(explicit: str | None = None) -> str:
    for candidate in [explicit, os.environ.get("PHI_MODEL_PATH"), *SEARCH_PATHS]:
        if candidate and (Path(candidate).expanduser() / "config.json").exists():
            return str(Path(candidate).expanduser())
    # Not a directory on disk: the Hub id, resolved from the local cache only.
    return "microsoft/Phi-4-mini-instruct"


def pick_device(requested: str = "auto") -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class PhiResponse:
    """One generation, already parsed and checked against the schema."""

    text: str
    parsed: dict | None
    violation: str | None
    latency_s: float

    @property
    def ok(self) -> bool:
        return self.parsed is not None and self.violation is None


class PhiClient:
    """Same surface as HostedClient in llm_utils.py: .structured()."""

    provider = "local:phi-4-mini-instruct"

    def __init__(
        self,
        model_path: str | None = None,
        *,
        device: str = "auto",
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.model_path = find_model_path(model_path)
        self.device = pick_device(device)
        self.max_new_tokens = max_new_tokens
        self.model = Path(self.model_path).name

        print(f"Loading {self.model_path} onto {self.device} (once) ...", flush=True)
        t0 = time.time()
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
            self.llm = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                dtype=torch.bfloat16 if self.device != "cpu" else torch.float32,
                local_files_only=True,
            ).to(self.device)
        except Exception as exc:
            raise SystemExit(
                f"Could not load Phi-4-mini-instruct from {self.model_path!r}: {exc}\n"
                "  Pass --model-path /path/to/phi-4-mini-instruct, or set PHI_MODEL_PATH."
            ) from exc
        self.llm.eval()
        print(f"  ready in {time.time() - t0:.1f}s", flush=True)

    def chat(self, messages: list[dict], max_new_tokens: int | None = None) -> tuple[str, float]:
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.device)

        t0 = time.time()
        with self._torch.no_grad():
            output = self.llm.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = output[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True), time.time() - t0

    def structured(
        self, prompt: str, schema: dict, *, system: str | None = None, **kwargs
    ) -> PhiResponse:
        """Ask for JSON matching `schema`. Violations are reported, not raised."""
        instruction = (
            "Respond with a single JSON object matching this schema. "
            "No preamble, no explanation, no markdown fences.\n"
            + json.dumps(schema, indent=2)
        )
        messages = [
            {"role": "system", "content": f"{system}\n\n{instruction}" if system else instruction},
            {"role": "user", "content": prompt},
        ]
        text, latency = self.chat(messages, **kwargs)

        try:
            parsed = extract_json(text)
        except ValueError as exc:
            return PhiResponse(text, None, f"unparseable output: {exc}", latency)
        if not isinstance(parsed, dict):
            return PhiResponse(text, None, f"expected an object, got {type(parsed).__name__}", latency)
        return PhiResponse(text, parsed, validate_against_schema(parsed, schema), latency)


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Check the local Phi lane.")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    phi = PhiClient(args.model_path, device=args.device)
    schema = {
        "type": "object",
        "properties": {"topic": {"type": "string"}, "entities": {"type": "array",
                                                                 "items": {"type": "string"}}},
        "required": ["topic", "entities"],
    }
    response = phi.structured(
        "FY2024 general fund expenditures by department, City of Pittsburgh.", schema
    )
    print(f"\n  latency:   {response.latency_s:.1f}s")
    print(f"  parsed:    {json.dumps(response.parsed)}")
    print(f"  violation: {response.violation or 'none'}")


if __name__ == "__main__":
    _cli()
