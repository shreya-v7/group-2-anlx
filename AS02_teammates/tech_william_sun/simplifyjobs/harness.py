"""Batch harness around LLMBOX.

`python -m startllm mode=generate prompt=...` reloads the model on every call.
For 50-input evaluations this harness composes the same Hydra config the CLI
would, loads the model once through LLMBOX's GenerationManager, and runs the
same message construction as Modes.run_generate for each input.

The only addition is measurement: input/output token counts, wall-clock
latency, and whether the output hit max_new_tokens.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from simplifyjobs.corpus import ROOT, read_jsonl

log = logging.getLogger(__name__)


def compose_cfg(overrides: list[str], values: dict | None = None) -> DictConfig:
    """Compose conf/config.yaml with CLI-style overrides, like startllm.py does.

    `values` ({"dotted.key": value}) are set after composition. Use it for
    filesystem paths: Hydra's override grammar rejects spaces, brackets, and
    parentheses, which appear in paths such as iCloud Drive folders.
    """
    from src.schema import register_configs

    register_configs()
    with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base=None):
        cfg = compose(config_name="config", overrides=overrides)
    for key, value in (values or {}).items():
        OmegaConf.update(cfg, key, value, merge=False)
    if str(cfg.model.source).lower() != "local":
        raise SystemExit("Policy: model.source must be 'local' (Part A data boundary).")
    return cfg


class Engine:
    """One loaded model, reused across all inputs of a run."""

    def __init__(self, cfg: DictConfig):
        from src.generation import GenerationManager

        self.cfg = cfg
        self.gm = GenerationManager()
        self.model, self.tokenizer, self.device = self.gm._load_model_and_tokenizer(cfg)

    def generate(self, messages: list[dict], cfg: DictConfig | None = None, **template_kwargs) -> dict:
        """Mirror of GenerationManager._generate_once that also returns cost stats."""
        import torch

        cfg = cfg or self.cfg
        template_kwargs = {
            **OmegaConf.to_container(cfg.model.chat_template_kwargs, resolve=True),
            **template_kwargs,
        }
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            **template_kwargs,
        ).to(self.device)
        n_in = int(inputs["input_ids"].shape[-1])

        start = time.perf_counter()
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **self.gm._generation_kwargs(cfg, self.tokenizer))
        latency = time.perf_counter() - start

        new_ids = output_ids[0][n_in:]
        n_out = int(new_ids.shape[-1])
        return {
            "text": self.tokenizer.decode(new_ids, skip_special_tokens=True).strip(),
            "input_tokens": n_in,
            "output_tokens": n_out,
            "latency_s": round(latency, 3),
            "hit_token_cap": n_out >= cfg.generation.max_new_tokens,
        }


class Capture:
    """Routes GenerationManager._generate_once through Engine.generate so that
    LLMBOX-internal loops (run_tool_turn, repair_structured) are measured."""

    def __init__(self, engine: "Engine"):
        self.engine = engine
        self.calls: list[dict] = []

    def generate(self, messages, cfg=None, **template_kwargs) -> str:
        result = self.engine.generate(messages, cfg, **template_kwargs)
        self.calls.append(result)
        return result["text"]

    def __enter__(self):
        def patched(model, tokenizer, device, messages, cfg, **template_kwargs):
            return self.generate(messages, cfg, **template_kwargs)

        self.engine.gm._generate_once = patched  # instance attribute shadows the method
        return self

    def __exit__(self, *exc):
        del self.engine.gm._generate_once

    def totals(self) -> dict:
        return {
            "input_tokens": sum(c["input_tokens"] for c in self.calls),
            "output_tokens": sum(c["output_tokens"] for c in self.calls),
            "model_latency_s": round(sum(c["latency_s"] for c in self.calls), 3),
            "hit_token_cap": any(c["hit_token_cap"] for c in self.calls),
            "n_generations": len(self.calls),
        }


def seed_everything(seed: int) -> None:
    from transformers import set_seed

    set_seed(seed)


def run_batch(
    engine: Engine,
    records: list[dict],
    respond: Callable[[Engine, dict], dict],
    out_path: Path,
    run_name: str,
    seed: int,
) -> list[dict]:
    """Run every record through `respond`, appending one JSON line per record.

    `respond(engine, record)` returns at least {"text", "messages",
    "input_tokens", "output_tokens", "latency_s", "hit_token_cap"}.
    Resumable: records already in out_path are skipped. Each record is seeded
    with the same seed, so a single record can be reproduced in isolation.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {row["doc_id"] for row in read_jsonl(out_path)} if out_path.exists() else set()
    gen = OmegaConf.to_container(engine.cfg.generation, resolve=True)

    with open(out_path, "a", encoding="utf-8") as handle:
        for i, record in enumerate(records, start=1):
            if record["doc_id"] in done:
                continue
            seed_everything(seed)
            result = respond(engine, record)
            row = {
                "run": run_name,
                "doc_id": record["doc_id"],
                "seed": seed,
                "generation": gen,
                "raw_output": result.pop("text"),
                **result,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            log.info("[%s] %d/%d %s  %.1fs  out=%d tok%s", run_name, i, len(records), record["doc_id"],
                     row["latency_s"], row["output_tokens"], "  BLOCKED" if row.get("blocked") else "")
    return read_jsonl(out_path)
