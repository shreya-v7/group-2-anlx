"""
layer_selection.py

Choose which parameters train, driven by TrainingConfig.

Patterns are regular expressions searched against parameter names from
model.named_parameters(), e.g. "model.layers.12.self_attn.q_proj.weight".
Plain substrings such as "embed" or "norm" therefore work as-is.

Rules, applied in this order:
  1. If trainable_modules is non-empty, everything is frozen except matches.
  2. Anything matching freeze_modules is then frozen (freeze wins on conflict).

Each pattern must match at least one parameter, otherwise a ValueError lists
the unmatched patterns (a silent typo would train the wrong layers).
"""
import re
from typing import Any, Dict, Iterable, List, Optional


def layers_pattern(indices: Iterable[int], container: str = "layers") -> str:
    """Regex matching transformer blocks by index.

    layers_pattern(range(20, 24))          -> Llama/Mistral/Qwen style  ...layers.20.
    layers_pattern([0, 1], container="h")  -> GPT-2 style               transformer.h.0.
    """
    idx = "|".join(str(i) for i in indices)
    return rf"(?:^|\.){re.escape(container)}\.(?:{idx})\."


def summarize_trainable(model: Any) -> Dict[str, Any]:
    total = trainable = 0
    for _, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return {"trainable": trainable, "total": total,
            "percent": 100.0 * trainable / total if total else 0.0}


def apply_layer_selection(model: Any,
                          freeze_modules: Optional[List[str]] = None,
                          trainable_modules: Optional[List[str]] = None,
                          strict: bool = True) -> Dict[str, Any]:
    freeze = [re.compile(p) for p in (freeze_modules or [])]
    train = [re.compile(p) for p in (trainable_modules or [])]
    hit = {rx.pattern: False for rx in freeze + train}

    for name, param in model.named_parameters():
        if train:
            matched = [rx for rx in train if rx.search(name)]
            param.requires_grad = bool(matched)
            for rx in matched:
                hit[rx.pattern] = True
        for rx in freeze:
            if rx.search(name):
                param.requires_grad = False
                hit[rx.pattern] = True

    unmatched = [p for p, ok in hit.items() if not ok]
    if strict and unmatched:
        raise ValueError(f"These patterns matched no parameter names: {unmatched}")

    summary = summarize_trainable(model)
    if summary["trainable"] == 0:
        raise ValueError("No trainable parameters remain after applying the selection.")
    print(f"Trainable parameters: {summary['trainable']:,} of {summary['total']:,} "
          f"({summary['percent']:.2f}%)")
    return summary


def build_lora_config(cfg: Any):
    """LoraConfig from TrainingConfig. Needs `pip install peft`."""
    from peft import LoraConfig
    return LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_target_modules) or None,
        layers_to_transform=list(cfg.lora_layers) if cfg.lora_layers else None,
        modules_to_save=list(cfg.lora_modules_to_save) or None,
        task_type="CAUSAL_LM",
    )


def configure_trainable(model: Any, cfg: Any) -> Any:
    """Prepare `model` for cfg.method and return it (LoRA returns a wrapped model)."""
    method = cfg.method
    if method in ("full", "freeze"):
        if method == "freeze" and not (cfg.freeze_modules or cfg.trainable_modules):
            raise ValueError("method='freeze' needs freeze_modules or trainable_modules.")
        apply_layer_selection(model, cfg.freeze_modules, cfg.trainable_modules)
    elif method == "bitfit":
        apply_layer_selection(model, trainable_modules=cfg.bitfit_bias_params or [r"\.bias$"])
    elif method in ("lora", "qlora"):
        from peft import get_peft_model
        model = get_peft_model(model, build_lora_config(cfg))
        model.print_trainable_parameters()
    else:
        raise NotImplementedError(f"method '{method}' is not handled here yet.")
    return model
