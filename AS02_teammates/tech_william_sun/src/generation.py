'''
    LLMBox -- A Software Application for Building Customized and Affordable AI Solutions.
    Copyright (C) 2026  Sara Kingsley

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

import json
import logging
import random
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from omegaconf import OmegaConf

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Shared model / generation helpers
# --------------------------------------------------------------------------

class GenerationManager:

    def __init__(self) -> None:
         self.log = logging.getLogger(__name__)

    def _resolve_device_and_dtype(self, cfg):
        import torch
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        dtype = dtype_map[cfg.model.dtype]
        if cfg.model.device != "auto":
            device = cfg.model.device
        elif torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        return device, dtype


    def _resolve_model_path(self, cfg):
        """Turn model.source/model_id/local_path into a (path_or_repo_id,
        local_files_only) pair for from_pretrained().

        source=local uses local_files_only=True so a bad path fails fast and
        clearly with a filesystem-style error, instead of transformers quietly
        trying to interpret it as a hub repo id. source=huggingface leaves
        local_files_only off, so from_pretrained uses its normal cache-or-
        download behavior."""
        if cfg.model.source == "local":
            if not cfg.model.local_path:
                raise ValueError(
                    f"model.source=local requires model.local_path to be set for model '{cfg.model.name}'."
                )
            return cfg.model.local_path, True
        if cfg.model.source == "huggingface":
            return cfg.model.model_id, False
        raise ValueError(f"Unknown model.source '{cfg.model.source}'. Use 'huggingface' or 'local'.")

    def _ensure_remote_code_compat(self):
        """Some trust_remote_code model repos (e.g. Phi-4-mini-instruct's
        modeling_phi3.py) still import `LossKwargs` from `transformers.utils`,
        a name that newer transformers releases renamed to `TransformersKwargs`.
        That break isn't specific to this one model -- it hits a lot of
        trust_remote_code repos whose custom code hasn't been updated for the
        renamed symbol. Rather than pin an older transformers version (and
        lose whatever else changed since), alias the old name back in right
        before loading."""
        import transformers.utils as _tu
        if not hasattr(_tu, "LossKwargs") and hasattr(_tu, "TransformersKwargs"):
            _tu.LossKwargs = _tu.TransformersKwargs

    def _load_model_and_tokenizer(self, cfg):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        device, dtype = self._resolve_device_and_dtype(cfg)
        model_path, local_files_only = self._resolve_model_path(cfg)
        log.info(
            "Loading '%s' (source=%s, %s) onto %s as %s...",
            cfg.model.name, cfg.model.source, model_path, device, cfg.model.dtype,
        )
        if cfg.model.trust_remote_code:
            self._ensure_remote_code_compat()
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=cfg.model.trust_remote_code, local_files_only=local_files_only
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            trust_remote_code=cfg.model.trust_remote_code,  #fixme
            local_files_only=local_files_only,
            low_cpu_mem_usage=True,
        )
        model.to(device)
        model.eval()
        return model, tokenizer, device

    def _generation_kwargs(self, cfg, tokenizer):
        gen = cfg.generation
        return dict(
            max_new_tokens=gen.max_new_tokens,
            do_sample=gen.do_sample,
            temperature=max(gen.temperature, 1e-5),
            top_p=gen.top_p,
            top_k=gen.top_k,
            repetition_penalty=gen.repetition_penalty,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    def _generate_once(self, model, tokenizer, device, messages, cfg, **template_kwargs):
        import torch

        template_kwargs = {**OmegaConf.to_container(cfg.model.chat_template_kwargs, resolve=True), **template_kwargs}
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            **template_kwargs,
        ).to(device)
        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            output_ids = model.generate(**inputs, **self._generation_kwargs(cfg, tokenizer))

        return tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()

    def _resolve_prompt(self, cfg) -> str:
        if cfg.prompt_file:
            text = Path(cfg.prompt_file).read_text(encoding="utf-8").strip()
        elif cfg.prompt:
            text = cfg.prompt.strip()
        else:
            raise ValueError('This mode needs a prompt. Pass prompt="..." or prompt_file=path/to/file.txt')
        if not text:
            raise ValueError("Resolved prompt is empty.")
        return text

    # --------------------------------------------------------------------------
    # Output contract helpers (added for SimplifyJobs, Part C)
    # --------------------------------------------------------------------------

    REPAIR_TEMPLATE = (
        "Your previous answer failed validation: {error}\n"
        "Return only the corrected JSON object, with no other text."
    )

    def structured_system_prompt(self, cfg, base=None) -> str:
        """System prompt used by structured_output: base prompt + JSON Schema."""
        base = cfg.system_prompt if base is None else base
        schema_instruction = ""
        if cfg.structured_output.schema_path:
            schema_text = Path(cfg.structured_output.schema_path).read_text(encoding="utf-8")
            schema_instruction = (
                "\n\nRespond with ONLY a single JSON object that strictly matches "
                f"this JSON Schema, with no other text:\n{schema_text}"
            )
        return ((base or "") + schema_instruction).strip()

    def repair_structured(self, answer, messages, validator, attempts, generate):
        """Re-prompt with the validation error until the output passes or the
        attempts run out. `generate(messages) -> text`. Returns (answer, n_repairs)."""
        n_repairs = 0
        for _ in range(max(0, attempts)):
            error = validator(answer)
            if not error:
                break
            if not (messages and messages[-1]["role"] == "assistant" and messages[-1]["content"] == answer):
                messages.append({"role": "assistant", "content": answer})
            messages.append({"role": "user", "content": self.REPAIR_TEMPLATE.format(error=error)})
            answer = generate(messages)
            n_repairs += 1
        return answer, n_repairs

    # --------------------------------------------------------------------------
    # "functools_prompt" tool calling -- for models (e.g. Phi-4-mini-instruct)
    # whose chat template doesn't do structured tool calling via a `tools=`
    # kwarg, and instead expects tool definitions embedded in the system
    # prompt as <|tool|>[...]<|/tool|>, with the model replying with a
    # "functools[...]"-prefixed JSON list of calls when it wants to invoke
    # one. Mirrors the convention chat_phi4mini.py established.
    # --------------------------------------------------------------------------

    def build_functools_system_prompt(self, base_system_prompt: str, tools: list) -> str:
        parts = [base_system_prompt] if base_system_prompt else []
        tools_json = json.dumps(tools)
        parts.append(f"<|tool|>{tools_json}<|/tool|>")
        parts.append(
            "In addition to plain text responses, you can choose to call one "
            "or more of the provided functions.\n"
            "Use the following rule to decide when to call a function:\n"
            "* if the response can be generated from your internal knowledge, "
            "do so\n"
            "* if you need external information that can be obtained by "
            "calling one or more of the provided functions, generate "
            "function calls\n"
            "If you decide to call functions:\n"
            "* prefix function calls with the functools marker (no closing "
            "marker required)\n"
            "* all function calls should be generated in a single JSON list "
            'formatted as functools[{"name": [function name], "arguments": '
            "[function arguments as JSON]}, ...]\n"
            "* follow the provided JSON schema. Do not hallucinate arguments "
            "or values.\n"
            "* respect the argument type formatting."
        )
        return "\n".join(parts)

    def parse_tool_calls(self, text: str):
        """Parse tool calls from a model reply. Returns a list of
        {"name": ..., "arguments": {...}} dicts, or None if there is no call.

        Accepts the formats models actually emit, not only the LLMBOX
        "functools[...]" convention: Phi-4-mini's <|tool_call|>...<|/tool_call|>,
        a bare JSON list or single object, OpenAI-style
        {"type": "function", "function": {...}} entries, "parameters" instead of
        "arguments", string-encoded arguments, and pythonic name(key="value").
        Only registered tool names count, so an ordinary JSON answer is never
        mistaken for a call. Non-dict entries are ignored instead of crashing
        (the baseline iterated over a dict's keys when a model emitted
        functools{...} without the list brackets)."""
        from src.tools import TOOL_REGISTRY

        chunks = []
        for marker in ("functools", "<|tool_call|>"):
            idx = text.find(marker)
            if idx != -1:
                chunks.append(text[idx + len(marker):])
        chunks.append(text)
        for chunk in chunks:
            calls = self._normalize_tool_calls(self._first_json(chunk), TOOL_REGISTRY)
            if calls:
                return calls
        for name in TOOL_REGISTRY:
            match = re.search(rf"\b{re.escape(name)}\s*\((.*?)\)", text, re.S)
            if match:
                args = dict(re.findall(r"(\w+)\s*=\s*[\"']([^\"']*)[\"']", match.group(1)))
                if args:
                    return [{"name": name, "arguments": args}]
        return None

    @staticmethod
    def _first_json(text: str):
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
        starts = [i for i in (text.find("["), text.find("{")) if i != -1]
        if not starts:
            return None
        try:
            value, _ = json.JSONDecoder().raw_decode(text[min(starts):])
            return value
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _normalize_tool_calls(value, registry):
        if isinstance(value, dict):
            value = [value]
        if not isinstance(value, list):
            return None
        calls = []
        for entry in value:
            if not isinstance(entry, dict):
                continue
            if isinstance(entry.get("function"), dict):
                entry = entry["function"]
            name = entry.get("name")
            args = entry.get("arguments", entry.get("parameters", {}))
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    continue
            if name in registry and isinstance(args, dict):
                calls.append({"name": name, "arguments": args})
        return calls or None

    def run_tool_turn(self, model, tokenizer, device, messages, cfg, tool_filter=None):
        """Generate one assistant turn under the functools_prompt convention:
        if the model requests tool call(s), execute them locally against
        src.tools.TOOL_REGISTRY, feed the results back in, and generate the
        final answer. `messages` is mutated in place with the raw response
        and any tool result, same as run_turn did in chat_phi4mini.py.
        Returns (final_text, tool_call_records)."""
        from src.tools import TOOL_REGISTRY

        raw_response = self._generate_once(model, tokenizer, device, messages, cfg)
        tool_call_records = []

        tool_calls = self.parse_tool_calls(raw_response)
        if tool_calls:
            results = []
            for call in tool_calls:
                name = call.get("name")
                arguments = call.get("arguments", {})
                func = TOOL_REGISTRY.get(name)
                blocked = tool_filter(name, arguments) if tool_filter else None
                if blocked:
                    result = {"error": f"Tool call blocked by policy ({blocked})"}
                elif func is None:
                    result = {"error": f"Unknown tool '{name}'"}
                else:
                    try:
                        result = func(**arguments)
                    except Exception as exc:  # noqa: BLE001 - surface bad args to the model
                        result = {"error": str(exc)}
                results.append({"name": name, "result": result})
                tool_call_records.append({"name": name, "arguments": arguments, "result": result})

            messages.append({"role": "assistant", "content": raw_response})
            messages.append({"role": cfg.model.get("tool_response_role", "tool"), "content": json.dumps(results)})

            raw_response = self._generate_once(model, tokenizer, device, messages, cfg)

        messages.append({"role": "assistant", "content": raw_response})
        return raw_response, tool_call_records
