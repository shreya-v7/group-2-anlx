"""
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
"""

"""
startllm.py

Single entry point for running google/gemma-3-270m-it or microsoft/phi-4
(or any local checkout of either) in one of six modes, fully driven by a
Hydra + OmegaConf config tree under `conf/`:

  chat               interactive multi-turn chat session, logged to
                      <output_dir>/chat_log/ (same shape as gemma_chat.py)
  generate           single prompt in, single response out, no history
  tool_calling       single-turn generation with tool/function definitions
                      passed through the model's chat template
  structured_output  single-turn generation constrained to a JSON schema
  train              continued pretraining / full fine-tune from scratch
                      on a data source (full weights only)
  finetune           supervised fine-tune (LoRA or full) on a data source

Which model, which mode, every generation/optimizer/training hyperparameter,
and where the data comes from are all config values, not hardcoded flags --
any of it can be swapped from the command line:

    # Chat with Gemma 3 270M-IT interactively (pulled from the HF hub, using
    # whatever's in your local HF cache or downloading it if not)
    python llmstart.py model=gemma3_270m mode=chat

    # Same, but loaded strictly from a local checkout -- never touches the
    # network, fails clearly if the path is wrong
    python3 llmstart.py model=gemma3_270m mode=chat model.source=local model.local_path=/Users/skingsle/Documents/llmbox/models/llms/google/gemma-3-270m-it

    # Phi-4-mini-instruct defaults to loading from ./models/llms/microsoft/
    # Phi-4-mini-instruct locally -- override the path for your machine, or
    # switch back to the hub with model.source=huggingface
    python3 llmstart.py model=phi4_instruct mode=chat \
        model.local_path=/actual/path/to/Phi-4-mini-instruct

    # One-shot generation with Phi-4, higher temperature
    python3 llmstart.py model=phi4_instruct mode=generate \
        generation.temperature=1.2 prompt="Explain the tides."

    # Tool calling with Phi-4
    python3 llmstart.py model=phi4_instruct mode=tool_calling model.source=local model.local_path=/Users/skingsle/Documents/llmbox/models/llms/microsoft/phi-4-mini-instruct \
        tool_calling.enabled=true prompt="What's the weather in Boston?" \
        'tool_calling.tools=[{name: get_weather, description: "Get current weather", parameters: {type: object, properties: {location: {type: string}}}}]'

    # Structured output (JSON-schema constrained)
    python configurator.py mode=structured_output structured_output.enabled=true \
        structured_output.schema_path=./schemas/person.json prompt="Describe a fictional person."

    # LoRA fine-tune Gemma 3 270M-IT on chat_log sessions with a custom LR
    python configurator.py model=gemma3_270m mode=finetune \
        training.enabled=true training.method=lora optimizer=adamw \
        optimizer.learning_rate=5e-5 data=chat_log

    # Full fine-tune Phi-4 on your own JSONL dataset with SGD
    python configurator.py model=phi4_instruct mode=finetune \
        training.enabled=true training.method=full data=jsonl data.path=my_data.jsonl \
        optimizer=sgd optimizer.learning_rate=1e-4

Run `python3 startllm.py --cfg job` to print the fully composed config
for any combination of overrides without loading a model -- handy for
sanity-checking a command before it downloads/loads anything.
"""

import logging
import os
import smtplib
import sys
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path

import hydra

# import LLMBOX application software:
from omegaconf import OmegaConf
from src.modes import Modes
from src.schema import Config, register_configs

modes = Modes()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
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
    "prepare_data": modes.run_prepare_data,
    "evaluate": modes.run_evaluation,
}


def detect_and_warn_llm_api_usage(cfg):
    # If source is not 'local', treat as LLM API (e.g., huggingface hub)
    is_remote = str(cfg.model.source).lower().strip() != "local"
    model_info = f"model source: {cfg.model.source}, "
    if hasattr(cfg.model, "model_id") and cfg.model.model_id:
        model_info += f"model_id: {cfg.model.model_id}"
    elif hasattr(cfg.model, "local_path") and cfg.model.local_path:
        model_info += f"local_path: {cfg.model.local_path}"
    else:
        model_info += "(model identifier info unavailable)"

    date_str = datetime.now().strftime("%Y-%m-%d")
    output_dir = Path(cfg.output_dir).expanduser().resolve()
    log_dir = output_dir / f"{date_str}-of-config-log-directory"
    log_dir.mkdir(parents=True, exist_ok=True)
    redflag_file = log_dir / "redflagalert.txt"

    if is_remote:
        warn_msg = (
            f"WARNING: You are running with a remote model/API ({model_info}).\n"
            "You must use models that are hosted locally and not interact with external LLM APIs "
            "for coursework.\n"
            "A notification will be sent to course staff. If this is unintended, please check your configuration."
        )
        # Terminal warning
        log.warning(warn_msg)
        print(warn_msg, file=sys.stderr)
        # Add to the primary configurator log
        configurator_log_file = log_dir / "configurator.log"
        with open(configurator_log_file, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().isoformat()}] [WARNING] {warn_msg}\n")
        # Add a special redflag alert file
        try:
            with open(redflag_file, "a", encoding="utf-8") as rf:
                rf.write(f"[RED FLAG] [{datetime.now().isoformat()}] {warn_msg}\n")
                rf.write("\n===== CONFIG SNAPSHOT =====\n")
                rf.write(OmegaConf.to_yaml(cfg))
                rf.write("\n==========================\n\n")
        except Exception as rf_exc:
            log.error(f"Could not write to redflag alert file: {rf_exc}")
        # Optionally send notification
        send_llm_api_notice(cfg, warn_msg)
    else:
        log.info(f"Using a local model only. ({model_info})")


def send_llm_api_notice(cfg, warn_msg):
    # Attempts to email course staff
    email_to = "skingsle@cs.cmu.edu"
    email_from = (
        f"llmbox@{os.uname().nodename}" if hasattr(os, "uname") else "llmbox@localhost"
    )
    subject = f"LLMBox API Usage Detected: {getattr(cfg.username, 'username', None) or os.environ.get('USER', 'unknown user')} ({cfg.model.source})"
    body = (
        f"User: {getattr(cfg, 'username', 'unknown user')}\n"
        f"Host: {os.uname().nodename if hasattr(os, 'uname') else 'unknown'}\n"
        f"Model source: {cfg.model.source}\n"
        f"{'model_id: ' + getattr(cfg.model, 'model_id', str(cfg.model)) if hasattr(cfg.model, 'model_id') else ''}\n"
        f"{'local_path: ' + getattr(cfg.model, 'local_path', '') if hasattr(cfg.model, 'local_path') else ''}\n"
        f"Config: {OmegaConf.to_yaml(cfg)}\n\n"
        f"Warning: {warn_msg}"
    )
    msg = EmailMessage()
    msg["From"] = email_from
    msg["To"] = email_to
    msg["Subject"] = subject
    msg.set_content(body)
    try:
        # Will try to send using localhost. Actual campus configs may vary.
        with smtplib.SMTP("localhost") as server:
            server.send_message(msg)
        log.info(f"Sent LLM API usage warning email to {email_to}.")
    except Exception as e:
        log.error(f"Failed to send LLM API usage warning email: {e}")
        print(f"[warning] Could not send notification email: {e}", file=sys.stderr)


def _guess_file_extension(path):
    """
    Helper to detect the file extension in a more robust way for prepare_data mode.
    Returns file extension or raises ValueError if cannot determine.
    """
    if isinstance(path, Path):
        ext = path.suffix
    elif isinstance(path, str):
        ext = Path(path).suffix
    else:
        ext = ""
    if not ext:
        return None
    return ext.lower()


@hydra.main(version_base=None, config_path="conf", config_name="config")
def llmbox(cfg: Config) -> None:
    log.info("Resolved configuration:\n%s", OmegaConf.to_yaml(cfg))
    mode_name = cfg.mode.name
    if mode_name not in _DISPATCH:
        raise ValueError(
            f"Unknown mode '{mode_name}'. Choose one of: {sorted(_DISPATCH)}"
        )

    # Special case for prepare_data: robustly check for file extension and warn/handle if missing
    if mode_name == "prepare_data":
        # Attempt to resolve input path (accepts data.raw_path, data.original_path, or data.path)
        input_path = (
            getattr(cfg.data, "raw_path", None)
            or getattr(cfg.data, "original_path", None)
            or getattr(cfg.data, "path", None)
        )
        if not input_path:
            raise ValueError(
                "No data.raw_path or data.original_path or data.path set in configuration."
            )
        ext = _guess_file_extension(input_path)
        if not ext:
            # If the extension is missing, attempt to detect by file content or raise a clearer error
            path_to_check = Path(input_path).expanduser()
            if not path_to_check.exists():
                raise ValueError(
                    f"prepare_data: Input file does not exist at path: {input_path}"
                )
            # Peek at the file for content-type guessing
            try:
                with open(path_to_check, "r", encoding="utf-8") as f:
                    first_bytes = f.read(2048)
                    if first_bytes.lstrip().startswith(
                        "{"
                    ) or first_bytes.lstrip().startswith("["):
                        ext = ".json"
                    elif first_bytes.count("\n") and "," in first_bytes.split("\n")[0]:
                        ext = ".csv"
                    else:
                        # Could be JSONL if many lines start with '{'
                        lines = first_bytes.strip().split("\n")
                        if all(
                            line.lstrip().startswith("{")
                            for line in lines
                            if line.strip()
                        ):
                            ext = ".jsonl"
                        else:
                            raise ValueError(
                                f"prepare_data: Unknown input file format and missing file extension for: {input_path}"
                            )
            except Exception as e:
                raise ValueError(
                    f"prepare_data: Could not infer or read file '{input_path}': {e}"
                )
            # Patch the file path to appear to have extension for underlying routines
            input_path_str = str(path_to_check)
            if not input_path_str.endswith(ext):
                # Symlink or copy file to temp with a correct extension if really needed,
                # or patch the attribute for downstream users -- we patch the config here.
                patched_path = input_path_str + ext
                # Just patch the config (do NOT actually copy file)
                cfg.data.raw_path = patched_path
                input_path = patched_path
        # Now input_path has a usable extension
    _DISPATCH[mode_name](cfg)
    match mode_name:
        case (
            "chat"
            | "generate"
            | "structured_output"
            | "train"
            | "finetune"
            | "evaluate"
        ):
            detect_and_warn_llm_api_usage(cfg)
        case "prepare_data":
            pass


if __name__ == "__main__":
    llmbox()
