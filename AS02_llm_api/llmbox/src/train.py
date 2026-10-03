import os
import sys
import json
import logging
from torch.utils.data import DataLoader
import hydra
from omegaconf import OmegaConf

from src.data_services import TrainingDataLoader
from src.schema import Config, register_configs
from src.modes import Modes

modes = Modes()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

log = logging.getLogger(__name__)

register_configs()

_DISPATCH = {
    "chat": modes.run_chat,
    "generate": modes.run_generate,
    "tool_calling": modes.run_tool_calling,
    "structured_output": modes.run_structured_output,
    "train": modes.run_train,
    "finetune": modes.run_finetune,
}

training_data = TrainingDataLoader(
    assistant_mask_strategy="template",
)

conversations = training_data._load_conversations(cfg)

dataset = training_data._build_dataset(
    conversations=conversations,
    tokenizer=tokenizer,
    max_length=2048,
)

pad_token_id = tokenizer.pad_token_id

if pad_token_id is None:
    # Reusing EOS for padding is common for decoder-only training.
    # Padding labels are ignored and padding attention positions are zero.
    pad_token_id = tokenizer.eos_token_id

if pad_token_id is None:
    raise ValueError(
        "Provide a padding token ID; this tokenizer has neither PAD nor EOS."
    )

data_loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=training_data._make_collator(pad_token_id),
)

batch = next(iter(data_loader))

print("Input shape:", batch["input_ids"].shape)
print("Label shape:", batch["labels"].shape)
print("Trainable targets:", (batch["labels"] != -100).sum().item())

# Hugging Face causal-LM models generally shift labels internally:
# outputs = model(**batch)
# loss = outputs.loss

# If the tokenizer’s template does not support assistant masks, explicitly try:
training_data = TrainingDataLoader(
    assistant_mask_strategy="verified_prefix",
)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def llmbox(cfg: Config) -> None:
    detect_and_warn_llm_api_usage(cfg)
    log.info("Resolved configuration:\n%s", OmegaConf.to_yaml(cfg))
    mode_name = cfg.mode.name
    if mode_name not in _DISPATCH:
        raise ValueError(f"Unknown mode '{mode_name}'. Choose one of: {sorted(_DISPATCH)}")
    _DISPATCH[mode_name](cfg)


if __name__ == "__main__":
    llmbox()
