"""Hosted LLM client for the optional SLM-vs-LLM comparison.

Every provider here speaks the OpenAI-compatible /chat/completions API, so
switching is one string:

    from llm_utils import get_client

    client = get_client("groq")                       # reads GROQ_API_KEY from .env
    response = client.structured(prompt, schema, system=SYSTEM)
    response.parsed, response.schema_violation, response.latency_s

Keys go in a `.env` file next to this script, one per line: GROQ_API_KEY=...
Free tiers rate-limit hard, so calls are sequential and 429s are retried with
backoff. The local lane (Phi) does not live here -- see local_model.py.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

from schema import extract_json, validate_against_schema

DEFAULT_TIMEOUT = 90
DEFAULT_MAX_ATTEMPTS = 6


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    env_key: str
    default_model: str
    json_mode: bool = True   # supports response_format={"type": "json_object"}


PROVIDERS: dict[str, Provider] = {
    "groq": Provider("groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY",
                     "llama-3.3-70b-versatile"),
    "gemini": Provider("gemini", "https://generativelanguage.googleapis.com/v1beta/openai",
                       "GEMINI_API_KEY", "gemini-3.5-flash-lite"),
    "cerebras": Provider("cerebras", "https://api.cerebras.ai/v1", "CEREBRAS_API_KEY",
                         "llama-3.3-70b"),
    "openrouter": Provider("openrouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY",
                           "meta-llama/llama-3.3-70b-instruct:free", json_mode=False),
    "openai": Provider("openai", "https://api.openai.com/v1", "OPENAI_API_KEY", "gpt-4o-mini"),
}


def load_dotenv(path: str | Path = Path(__file__).with_name(".env")) -> None:
    """Minimal .env loader so nobody has to install python-dotenv."""
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


@dataclass
class Response:
    """One completion, parsed and checked if a schema was given."""

    text: str
    provider: str
    model: str
    latency_s: float
    parsed: Any = None
    schema_violation: str | None = None
    repaired: bool = False          # JSON had to be dug out of prose/fences
    attempts: int = 1
    usage: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.parsed is not None and self.schema_violation is None


class HostedClient:
    """Any OpenAI-compatible chat endpoint. Same .structured() surface as PhiClient."""

    def __init__(self, provider: str = "groq", model: str | None = None, *,
                 temperature: float = 0.0, max_tokens: int = 1024,
                 timeout: int = DEFAULT_TIMEOUT, max_attempts: int = DEFAULT_MAX_ATTEMPTS):
        load_dotenv()
        if provider not in PROVIDERS:
            raise ValueError(f"Unknown provider {provider!r}. Known: {', '.join(sorted(PROVIDERS))}")
        self.spec = PROVIDERS[provider]
        self.provider = provider
        self.model = model or self.spec.default_model
        self.temperature, self.max_tokens = temperature, max_tokens
        self.timeout, self.max_attempts = timeout, max_attempts
        self.api_key = os.environ.get(self.spec.env_key)
        if not self.api_key:
            raise RuntimeError(f"{self.spec.env_key} is not set. Put it in a .env file next to "
                               f"llm_utils.py:\n    {self.spec.env_key}=your-key-here")

    def _send(self, payload: dict) -> tuple[dict, int]:
        """POST with retry on rate limits, server errors and timeouts."""
        url = f"{self.spec.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        delay = 2.0
        for attempt in range(1, self.max_attempts + 1):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            except requests.RequestException as exc:
                if attempt == self.max_attempts:
                    raise RuntimeError(f"{self.provider}: {exc}") from exc
            else:
                if resp.status_code < 400:
                    return resp.json(), attempt
                if resp.status_code not in (408, 429, 500, 502, 503, 504) or attempt == self.max_attempts:
                    raise RuntimeError(f"{self.provider} HTTP {resp.status_code}: {resp.text[:300]}")
                retry_after = resp.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    delay = max(delay, float(retry_after))
            time.sleep(delay)
            delay = min(delay * 2, 60)
        raise RuntimeError("unreachable")

    def chat(self, messages: list[dict], schema: dict | None = None) -> Response:
        payload: dict[str, Any] = {
            "model": self.model, "messages": messages,
            "temperature": self.temperature, "max_tokens": self.max_tokens,
        }
        if schema is not None and self.spec.json_mode:
            payload["response_format"] = {"type": "json_object"}

        t0 = time.time()
        body, attempts = self._send(payload)
        latency = time.time() - t0
        try:
            text = body["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected response shape from {self.provider}: {exc}") from exc

        response = Response(text=text, provider=self.provider, model=self.model,
                            latency_s=latency, attempts=attempts, usage=body.get("usage") or {})
        if schema is not None:
            stripped = text.strip()
            try:
                response.parsed = extract_json(text)
                response.repaired = not (stripped.startswith(("{", "[")) and stripped.endswith(("}", "]")))
            except ValueError as exc:
                response.schema_violation = f"unparseable output: {exc}"
                return response
            response.schema_violation = validate_against_schema(response.parsed, schema)
        return response

    def complete(self, prompt: str, *, system: str | None = None) -> Response:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}]
        return self.chat(messages)

    def structured(self, prompt: str, schema: dict, *, system: str | None = None, **_) -> Response:
        """Ask for JSON conforming to `schema`. Violations are reported, not raised."""
        instruction = ("Respond with a single JSON object matching this schema. "
                       "No preamble, no explanation, no markdown fences.\n"
                       + json.dumps(schema, indent=2))
        messages = [
            {"role": "system", "content": f"{system}\n\n{instruction}" if system else instruction},
            {"role": "user", "content": prompt},
        ]
        return self.chat(messages, schema=schema)


def get_client(provider: str | None = None, model: str | None = None, **kwargs) -> HostedClient:
    """Client by provider name. With no argument, reads COURSE_PROVIDER, default 'groq'."""
    return HostedClient(provider or os.environ.get("COURSE_PROVIDER", "groq"), model, **kwargs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Check a hosted lane.")
    parser.add_argument("--provider", default="groq", choices=sorted(PROVIDERS))
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    client = get_client(args.provider, args.model)
    schema = {"type": "object", "properties": {"topic": {"type": "string"}},
              "required": ["topic"]}
    r = client.structured("Classify: 'FY2024 general fund expenditures by department.'", schema)
    print(f"model:     {r.model}\nlatency:   {r.latency_s:.2f}s\nparsed:    {r.parsed}\n"
          f"violation: {r.schema_violation or 'none'}")
