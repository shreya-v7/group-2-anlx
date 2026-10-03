"""SimplifyJobs request pipeline on top of LLMBOX (Parts C and D).

Each feature maps to an LLMBOX mode or config switch, and each step calls the
same LLMBOX function the CLI uses (see src/modes.py):

  structured  mode=structured_output   GenerationManager.structured_system_prompt
  tools       mode=tool_calling        build_functools_system_prompt / run_tool_turn
  guardrail   guardrail.enabled=true   src.guardrail.Guardrail
  repair      structured_output.repair_attempts   GenerationManager.repair_structured
  verifier    structured_output.postprocess       simplifyjobs.verifier.ground

The pipeline adds only measurement: tokens, latency, tool calls, repairs,
corrections, and guardrail decisions per request.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

from omegaconf import OmegaConf

from simplifyjobs.corpus import user_prompt
from simplifyjobs.harness import Capture, Engine

TOOL_INSTRUCTION = (
    "\nBefore answering, call lookup_posting with this document's id to get the employer and "
    "official job title. Use the returned company for the company field and the job title to "
    "decide seniority. Then answer with the JSON object."
)
_DOC_HEADER = re.compile(r"\s*Document\s+([\w-]+)\s*:")


@dataclass(frozen=True)
class Features:
    structured: bool = False
    tools: bool = False
    guardrail: bool = False
    repair_attempts: int = 0
    verifier: bool = False


def classifier_cfg(cfg):
    """Greedy, very short decoding for the guardrail's one-word classification."""
    return OmegaConf.merge(
        cfg,
        {"generation": {"max_new_tokens": cfg.guardrail.llm_check_max_new_tokens, "do_sample": False,
                        "temperature": 1.0, "top_p": 1.0, "top_k": 50}},
    )


class Pipeline:
    def __init__(self, engine: Engine, features: Features, base_prompt: str):
        from src.guardrail import Guardrail, load_callable

        from simplifyjobs.job_schema import contract_error

        self.engine, self.f, self.base = engine, features, base_prompt
        cfg = engine.cfg
        self.guard = None
        if features.guardrail:
            check_cfg = classifier_cfg(cfg)
            self.guard = Guardrail(cfg.guardrail, classifier=lambda msgs: engine.generate(msgs, check_cfg))
        self.validator = load_callable(cfg.structured_output.validator) or contract_error
        self.postprocess = load_callable(cfg.structured_output.postprocess) if features.verifier else None
        self.tools = OmegaConf.to_container(cfg.tool_calling.tools, resolve=True) if features.tools else []

    # ------------------------------------------------------------------
    def system_prompt(self) -> str:
        gm, cfg = self.engine.gm, self.engine.cfg
        base = self.base + (TOOL_INSTRUCTION if self.f.tools else "")
        system = gm.structured_system_prompt(cfg, base) if self.f.structured else base
        return self.guard.protect_system_prompt(system) if self.guard else system

    def __call__(self, engine: Engine, record: dict) -> dict:
        return self.respond(user_prompt(record), request_id=record["doc_id"])

    def respond(self, prompt: str, request_id: str | None = None) -> dict:
        """Run one request end to end. `prompt` is the API's user turn."""
        engine, gm, cfg = self.engine, self.engine.gm, self.engine.cfg
        start = time.perf_counter()
        decisions, guard_in, guard_out, guard_latency = [], 0, 0, 0.0
        result = {"tools_enabled": self.f.tools, "tool_calls": [], "n_repairs": 0, "corrections": []}

        if self.guard:
            d_in = self.guard.check_input(prompt)
            decisions.append(d_in)
            guard_in, guard_out = d_in.cost["input_tokens"], d_in.cost["output_tokens"]
            guard_latency += d_in.cost["latency_s"]
            if not d_in.allowed:
                self.guard.log(prompt, decisions, request_id)
                return {
                    **result,
                    "text": cfg.guardrail.refusal_message,
                    "model_output": None,
                    "messages": [{"role": "user", "content": prompt}],
                    "input_tokens": d_in.cost["input_tokens"],
                    "output_tokens": d_in.cost["output_tokens"],
                    "model_latency_s": 0.0,
                    "hit_token_cap": False,
                    "n_generations": 0,
                    "blocked": True,
                    "block_stage": "input",
                    "block_rules": d_in.rules,
                    "guardrail_latency_s": round(guard_latency, 4),
                    "guardrail_tokens": guard_in + guard_out,
                    "latency_s": round(time.perf_counter() - start, 3),
                }
            prompt = d_in.text

        system = self.system_prompt()
        with Capture(engine) as cap:
            if self.f.tools:
                messages = [
                    {"role": "system", "content": gm.build_functools_system_prompt(system, self.tools)},
                    {"role": "user", "content": prompt},
                ]
                answer, result["tool_calls"] = gm.run_tool_turn(
                    engine.model, engine.tokenizer, engine.device, messages, cfg,
                    tool_filter=self._tool_filter(prompt, decisions) if self.guard else None,
                )
            else:
                messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
                answer = cap.generate(messages)
            if self.f.repair_attempts:
                answer, result["n_repairs"] = gm.repair_structured(
                    answer, messages, self.validator, self.f.repair_attempts, cap.generate
                )
        result["model_output"] = answer

        if self.postprocess:
            answer, result["corrections"] = self.postprocess(answer, prompt, {"tool_calls": result["tool_calls"]})

        result["blocked"], result["block_rules"] = False, []
        if self.guard:
            d_out = self.guard.check_output(answer)
            decisions.append(d_out)
            guard_latency += d_out.cost["latency_s"]
            self.guard.log(prompt, decisions, request_id)
            tool_blocks = [r for d in decisions if d.stage == "tool" and not d.allowed for r in d.rules]
            result["block_rules"] = tool_blocks
            if not d_out.allowed:
                result.update(blocked=True, block_stage="output", block_rules=tool_blocks + d_out.rules)
                answer = cfg.guardrail.refusal_message
            result["guardrail_latency_s"] = round(guard_latency, 4)
            result["guardrail_tokens"] = guard_in + guard_out

        totals = cap.totals()
        totals["input_tokens"] += guard_in
        totals["output_tokens"] += guard_out
        return {
            **result,
            **totals,
            "text": answer,
            "messages": messages,
            "latency_s": round(time.perf_counter() - start, 3),
        }

    def _tool_filter(self, prompt: str, decisions: list):
        match = _DOC_HEADER.match(prompt)
        bound = match.group(1) if match else None

        def tool_filter(name, arguments):
            decision = self.guard.check_tool_call(name, arguments, bound)
            decisions.append(decision)
            return None if decision.allowed else ",".join(decision.rules)

        return tool_filter
