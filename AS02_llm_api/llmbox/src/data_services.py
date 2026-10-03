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

'''
ABOUT DATA_SERVICES MODULE:
    This module provides a few Python classes. These Python classes have functions that perform the following tasks:
        * Transform datasets: the DataTransformer CLASS has functions that can transform a dataset into a format accepted by the train and finetune modes of the LLMBox application.
        * Load training data: the TrainingDataLoader CLASS has functions that will load data into training or finetuning jobs. Note: the data must already be transformed into a format that LLMBOX train and finetune modes accept.

'''

import json
import logging
import random
import sys
import uuid
import torch

from datetime import datetime, timezone
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from omegaconf import OmegaConf

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

# PyTorch cross-entropy ignores target positions with this value.
# This affects the loss, not whether a token is visible through attention.
IGNORE_INDEX = -100

class DataTransformer:
    """
    Prepare row-oriented datasets for causal language-model training.
    Every standardized example contains:
        prompt:
            The input context. Empty for text-only language modeling.
        completion:
            The desired continuation or response.
        text:
            prompt + completion, suitable for full-sequence causal-LM training.
        messages:
            A chat representation suitable for chat-template-based fine-tuning.
    This class does not tokenize data or create loss masks. Those operations
    belong in the training pipeline.
    """

    @staticmethod
    def _json_text(value: Any) -> str:
        """
        Serialize a Python value as strict JSON.
        ensure_ascii=False preserves characters such as accented letters.
        allow_nan=False rejects NaN and Infinity, which are not valid JSON.
        Unsupported values, such as sets or tensors, must be converted by
        the caller before using this utility.
        """
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
        )

    @staticmethod
    def _render_value(value: Any) -> str:
        """
        Convert a dataset value into model-visible text.
        Strings are returned unchanged. In particular, a string that already
        contains JSON must not be JSON-encoded a second time.
        Other values use JSON formatting:
            True -> "true"
            42 -> "42"
            ["a", "b"] -> '["a", "b"]'
        """
        if isinstance(value, str):
            return value
        return DataTransformer._json_text(value)

    @staticmethod
    def _normalize_columns(
        columns: str | Sequence[str],
        argument_name: str,
    ) -> list[str]:
        """
        Normalize one column name or a sequence of names into a list.
        Reject empty names and duplicates instead of silently producing
        incomplete or ambiguous examples.
        """
        if isinstance(columns, str):
            names = [columns]
        elif isinstance(columns, Sequence) and not isinstance(
            columns, (bytes, bytearray)
        ):
            names = list(columns)
        else:
            raise TypeError(
                f"{argument_name} must be a column name "
                "or a sequence of column names."
            )
        if not names:
            raise ValueError(f"{argument_name} must not be empty.")
        for name in names:
            if not isinstance(name, str) or not name.strip():
                raise ValueError(
                    f"{argument_name} must contain non-empty strings."
                )
        if len(names) != len(set(names)):
            raise ValueError(
                f"{argument_name} contains duplicate column names."
            )
        return names

    @staticmethod
    def _read_value(row: Mapping[str, Any], column: str) -> Any:
        """
        Read a required, non-null field.
        An absent field and a field containing None are both rejected.
        Empty strings are checked separately where the field is used.
        """
        if column not in row:
            raise ValueError(f"missing column {column!r}")
        value = row[column]
        if value is None:
            raise ValueError(f"column {column!r} is null")
        return value

    @staticmethod
    def _validate_prompt_completion(
        example: Mapping[str, Any],
    ) -> tuple[str, str]:
        """
        Validate an already string-valued prompt/completion pair.
        An empty prompt is valid for text-only language modeling.
        A completion must contain at least one non-whitespace character.
        Return the original strings without stripping meaningful whitespace.
        """
        for field in ("prompt", "completion"):
            if field not in example:
                raise ValueError(f"missing field {field!r}")
            if not isinstance(example[field], str):
                raise TypeError(f"{field!r} must be a string")
        prompt = example["prompt"]
        completion = example["completion"]
        if not completion.strip():
            raise ValueError("'completion' must not be empty")
        return prompt, completion

    @staticmethod
    def normalize_prompt_completion(
        example: Mapping[str, Any],
    ) -> dict[str, Any]:
        """
        Add training representations to a prompt/completion example.
        Both fields are required. Unlike a schema-detection function, this
        function rejects unrelated records rather than silently passing
        them through.
        A completion may be:
            - a string, preserved exactly;
            - a dictionary or list, serialized once as JSON.
        Other fields are preserved. Existing text/messages fields are
        regenerated so they agree with prompt/completion.
        No separator is inserted between prompt and completion. The prompt
        is responsible for its own formatting, such as a final "Output:\\n".
        """
        if not isinstance(example, Mapping):
            raise TypeError("Each example must be a mapping.")
        result = dict(example)
        if "completion" in result:
            completion = result["completion"]
            if isinstance(completion, (dict, list)):
                result["completion"] = DataTransformer._json_text(completion)
        prompt, completion = DataTransformer._validate_prompt_completion(
            result
        )
        result["text"] = prompt + completion
        # Do not fabricate an empty user turn for text-only documents.
        # Such records are intended primarily for the train/text path.
        messages: list[dict[str, str]] = []
        if prompt:
            messages.append({
                "role": "user",
                "content": prompt,
            })
        messages.append({
            "role": "assistant",
            "content": completion,
        })
        result["messages"] = messages
        return result

    def standardize_llm_dataset(
        self,
        dataset: Iterable[Mapping[str, Any]],
        *,
        text_columns: str | Sequence[str] | None = None,   # text_columns: these should be the data representing the user question, prompt or input context used for the AI Assistant prediction.
        target_columns: str | Sequence[str] | None = None, # target_columns: these should be the data representing the AI assistant `response` or `completion`.
        instruction: str = "",                             # instructions (optional): this describes the task an LLM will either trained or ask to perform on the dataset or a similar dataset.
        label_maps: Mapping[str, Mapping[Any, Any]] | None = None,  # label_maps (optional): this parameter allows you to apply a text label to a feature in your dataset that has a numeric value. For example, if the column_values are either 1 or 0, you could specify that 1 = "acceptable" and 0 = "unacceptable".
    ) -> Iterator[dict[str, Any]]:
        """
        Convert rows into prompt/completion/text/messages examples.

        Mode 1: Existing prompt/completion pairs
            Leave text_columns and target_columns unset.

            The existing prompt is preserved exactly. This avoids wrapping
            an already formatted prompt in another "Input:/Output:" block.

        Mode 2: Text-only language modeling
            Set text_columns; leave target_columns unset.

            Selected text becomes the completion, and the prompt is empty.
            One text column is preserved as-is. Multiple text columns are
            combined using field names and blank-line separators.

        Mode 3: Supervised generation
            Set both text_columns and target_columns.

            Input fields become a formatted prompt. One target becomes a
            text/scalar completion. Multiple targets become a JSON object.

        label_maps:
            Optional mappings for selected target columns. For example:
                {"label": {0: "negative", 1: "positive"}}
            Every encountered label must appear in its mapping.

        Important behavior:
            - Accepts lists, generators, and individual HF Dataset splits.
            - Rejects a single row dict or a dict of dataset splits.
            - Returns only prompt/completion/text/messages.
            - Does not modify the input rows.
            - Row errors include a zero-based row index.
            - Validation is deferred until this iterator is consumed.
        """
        # A mapping iterates over keys, not over row dictionaries.
        if isinstance(dataset, Mapping):
            raise TypeError(
                "dataset must be an iterable of row mappings. "
                "For one row, use [row]. "
                "For a dataset dictionary, select a split such as "
                "dataset['train']."
            )
        if isinstance(dataset, (str, bytes, bytearray)):
            raise TypeError(
                "dataset must contain rows, not a path or string. "
                "Load the file before calling this method."
            )
        if not isinstance(instruction, str):
            raise TypeError("instruction must be a string.")
        # Omitting text_columns explicitly selects the existing-pair path.
        use_existing_pairs = text_columns is None
        if use_existing_pairs:
            if target_columns is not None:
                raise ValueError(
                    "target_columns requires text_columns."
                )
            if instruction.strip():
                raise ValueError(
                    "instruction is not supported in existing-pair mode. "
                    "Edit the existing prompts or select text_columns "
                    "and target_columns explicitly."
                )
            inputs: list[str] = []
            targets: list[str] = []
        else:
            inputs = self._normalize_columns(
                text_columns,
                "text_columns",
            )
            targets = (
                self._normalize_columns(
                    target_columns,
                    "target_columns",
                )
                if target_columns is not None
                else []
            )
            overlapping_columns = set(inputs) & set(targets)
            if overlapping_columns:
                raise ValueError(
                    "Input and target columns overlap: "
                    f"{sorted(overlapping_columns)}"
                )
            if instruction.strip() and not targets:
                raise ValueError(
                    "instruction requires target_columns. "
                    "Text-only language modeling uses an empty prompt."
                )
        # Validate mapping configuration once, rather than once per row.
        if label_maps is None:
            maps: Mapping[str, Mapping[Any, Any]] = {}
        elif isinstance(label_maps, Mapping):
            maps = label_maps
        else:
            raise TypeError("label_maps must be a mapping.")
        for column, replacements in maps.items():
            if column not in targets:
                raise ValueError(
                    f"label_maps refers to non-target column {column!r}."
                )
            if not isinstance(replacements, Mapping):
                raise TypeError(
                    f"label_maps[{column!r}] must be a mapping."
                )
        try:
            rows = iter(dataset)
        except TypeError as exc:
            raise TypeError("dataset must be iterable.") from exc
        for row_index, row in enumerate(rows):
            try:
                if not isinstance(row, Mapping):
                    raise TypeError("each row must be a mapping")
                if use_existing_pairs:
                    # Preserve prompt formatting and JSON completion strings.
                    # Ignore unrelated source columns.
                    pair = {
                        "prompt": self._read_value(row, "prompt"),
                        "completion": self._read_value(row, "completion"),
                    }
                else:
                    # Render inputs in the order requested by the caller.
                    input_values: dict[str, str] = {}
                    for column in inputs:
                        value = self._read_value(row, column)
                        rendered_value = self._render_value(value)
                        if not rendered_value.strip():
                            raise ValueError(
                                f"input column {column!r} is empty"
                            )
                        input_values[column] = rendered_value
                    if not targets:
                        # Text-only LM: learn to predict the whole document.
                        if len(inputs) == 1:
                            completion = input_values[inputs[0]]
                        else:
                            completion = "\n\n".join(
                                f"{column}:\n{value}"
                                for column, value in input_values.items()
                            )
                        pair = {
                            "prompt": "",
                            "completion": completion,
                        }
                    else:
                        # Supervised generation: preserve target types until
                        # deciding whether to serialize one value or an object.
                        output_values: dict[str, Any] = {}
                        for column in targets:
                            value = self._read_value(row, column)
                            if column in maps:
                                try:
                                    value = maps[column][value]
                                except (KeyError, TypeError) as exc:
                                    raise ValueError(
                                        f"no label mapping for {column!r} "
                                        f"value {value!r}"
                                    ) from exc
                            if value is None:
                                raise ValueError(
                                    f"target column {column!r} resolves to null"
                                )
                            if isinstance(value, str) and not value.strip():
                                raise ValueError(
                                    f"target column {column!r} is empty"
                                )
                            output_values[column] = value
                        if len(targets) == 1:
                            completion = self._render_value(
                                output_values[targets[0]]
                            )
                        else:
                            completion = self._json_text(output_values)
                        input_block = "\n\n".join(
                            f"{column}:\n{value}"
                            for column, value in input_values.items()
                        )
                        instruction_prefix = ""
                        if instruction.strip():
                            instruction_prefix = instruction.strip() + "\n\n"
                        prompt = (
                            instruction_prefix
                            + f"Input:\n{input_block}\n\nOutput:\n"
                        )
                        pair = {
                            "prompt": prompt,
                            "completion": completion,
                        }
                # Normalize AFTER formatting. Otherwise the derived fields
                # would be lost when constructing the final example.
                standardized_example = self.normalize_prompt_completion(
                    pair
                )
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"Row {row_index}: {exc}"
                ) from exc
            yield standardized_example

    @staticmethod
    def _serialize_examples(
        examples: Iterable[Mapping[str, Any]],
        *,
        require_prompt_completion: bool = False,
    ) -> list[str]:
        """
        Validate and serialize all rows before opening destination files.
        This prevents a malformed example from truncating an existing file.
        It deliberately uses memory proportional to the serialized dataset.
        Returned strings already contain their JSONL newline.
        """
        if isinstance(examples, (Mapping, str, bytes, bytearray)):
            raise TypeError(
                "examples must be an iterable of row mappings."
            )
        lines: list[str] = []
        for index, example in enumerate(examples):
            try:
                if not isinstance(example, Mapping):
                    raise TypeError("each example must be a mapping")
                if require_prompt_completion:
                    DataTransformer._validate_prompt_completion(example)
                serialized_example = DataTransformer._json_text(dict(example))
                lines.append(serialized_example + "\n")
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"Example {index}: {exc}"
                ) from exc
        return lines

    @staticmethod
    def write_jsonl(
        examples: Iterable[Mapping[str, Any]],
        path: str | Path,
        *,
        overwrite: bool = True,
    ) -> int:
        """
        Write JSON-serializable mappings, one per line.
        Nested fields such as messages are supported.
        Parent directories are created if necessary.
        All examples are serialized before opening the output file.
        This is not an atomic write: an I/O failure can leave a partial file.
        Return the number of examples written.
        """
        destination_path = Path(path).expanduser()
        if not overwrite and destination_path.exists():
            raise FileExistsError(
                f"{destination_path} already exists. "
                "Set overwrite=True to replace it."
            )
        lines = DataTransformer._serialize_examples(examples)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        # "x" also prevents overwriting a file created after the earlier check.
        file_mode = "w" if overwrite else "x"
        with destination_path.open(
            file_mode,
            encoding="utf-8",
            newline="\n",
        ) as destination:
            destination.writelines(lines)
        return len(lines)

    @staticmethod
    def split_and_save_jsonl(
        examples: Iterable[Mapping[str, Any]],  # these should be from your transformed dataset, e.g. the dataset produced by the function standardize_llm_dataset
        output_directory: str | Path,           # the path where you want your train and test sets saved
        *,
        train_fraction: float = 0.8,            # use this parameter to specify what percentage of your whole dataset should go into the train vs. test set
        seed: int = 42,                         # this is a random number used to make the process reproducible. Note: the number isn't truly random. What does it do? It makes sure the same sequence of numbers is produced by the iterator. The seed establishes a number the number generator uses at the start of the sequence, e.g. the number is used to generate the next number in the sequence
        overwrite: bool = True,
    ) -> dict[str, Any]:
        """
        Shuffle examples and write train.jsonl and test.jsonl.
        Requirements:
            - At least two examples.
            - String-valued prompt and non-empty completion fields.
            - JSON-serializable values in all retained fields.
        Behavior:
            - Preserves all example fields, including text and messages.
            - Uses a local RNG, leaving global random state unchanged.
            - Keeps at least one example in each split.
            - Loads serialized examples into memory.
            - Validates all examples before opening output files.
        Limitations:
            - This is a row-level random split, not a grouped or stratified split.
            - Writes are not atomic, either individually or as a pair.
              An I/O failure can leave incomplete output.
        """
        if (
            isinstance(train_fraction, bool)
            or not isinstance(train_fraction, (int, float))
            or not 0 < train_fraction < 1
        ):
            raise ValueError(
                "train_fraction must be a number strictly between 0 and 1."
            )
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer.")
        output_directory = (
            Path(output_directory).expanduser().resolve()
        )
        train_path = output_directory / "train.jsonl"
        test_path = output_directory / "test.jsonl"
        if not overwrite:
            for path in (train_path, test_path):
                if path.exists():
                    raise FileExistsError(
                        f"{path} already exists. "
                        "Set overwrite=True to replace it."
                    )
        lines = DataTransformer._serialize_examples(
            examples,
            require_prompt_completion=True,
        )
        total_count = len(lines)
        if total_count < 2:
            raise ValueError("At least two examples are required.")
        local_random_generator = random.Random(seed)
        local_random_generator.shuffle(lines)
        # Round down, then clamp so neither split is empty.
        requested_train_count = int(total_count * train_fraction)
        train_count = max(
            1,
            min(total_count - 1, requested_train_count),
        )
        test_count = total_count - train_count
        output_directory.mkdir(parents=True, exist_ok=True)
        file_mode = "w" if overwrite else "x"
        with train_path.open(
            file_mode,
            encoding="utf-8",
            newline="\n",
        ) as train_file:
            for index in range(train_count):
                train_file.write(lines[index])
        with test_path.open(
            file_mode,
            encoding="utf-8",
            newline="\n",
        ) as test_file:
            for index in range(train_count, total_count):
                test_file.write(lines[index])
        return {
            "train_path": train_path,
            "test_path": test_path,
            "train_count": train_count,
            "test_count": test_count,
            "total_count": total_count,
            "actual_train_fraction": train_count / total_count,
        }

class TokenizedChatDataset(Dataset):
    """
    Store conversations that have already been tokenized.

    Each example contains equally sized lists:
        input_ids: tokens visible to the model
        labels: target token IDs, or IGNORE_INDEX for unsupervised positions

    Padding is postponed until examples are assembled into a batch.
    """

    def __init__(
        self,
        examples: list[dict[str, list[int]]],
    ) -> None:
        self.examples = examples

    def __len__(self) -> int:
        """Return the number of usable training conversations."""
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        """Return one unpadded example."""
        return self.examples[index]


class CausalLMCollator:
    """
    Right-pad examples to the longest sequence in the current batch.

    Padding rules:
        input_ids      -> pad_token_id
        labels         -> IGNORE_INDEX
        attention_mask -> 0

    Real tokens receive attention_mask=1, including prompt tokens whose
    labels are ignored. Ignoring a label must not hide its input context.

    A module-level callable class is used instead of a nested function so
    the collator can be pickled by multiprocessing DataLoader workers.
    """

    def __init__(
        self,
        pad_token_id: int,
        ignore_index: int = IGNORE_INDEX,
    ) -> None:
        if (
            isinstance(pad_token_id, bool)
            or not isinstance(pad_token_id, int)
            or pad_token_id < 0
        ):
            raise ValueError("pad_token_id must be a non-negative integer.")

        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index

    def __call__(
        self,
        batch: list[dict[str, list[int]]],
    ) -> dict[str, torch.Tensor]:
        """Validate, pad, and convert one batch into PyTorch tensors."""
        if not batch:
            raise ValueError("Cannot collate an empty batch.")

        for example_index, example in enumerate(batch):
            token_ids = example["input_ids"]
            target_ids = example["labels"]

            if not token_ids:
                raise ValueError(
                    f"Batch example {example_index} has no tokens."
                )

            if len(token_ids) != len(target_ids):
                raise ValueError(
                    f"Batch example {example_index} has different "
                    "input_ids and labels lengths."
                )

        batch_max_length = max(
            len(example["input_ids"])
            for example in batch
        )
        padded_inputs: list[list[int]] = []
        padded_labels: list[list[int]] = []
        attention_masks: list[list[int]] = []
        for example in batch:
            # Make new lists rather than modifying the stored dataset.
            token_ids = list(example["input_ids"])
            target_ids = list(example["labels"])
            sequence_length = len(token_ids)
            padding_length = batch_max_length - sequence_length
            padded_inputs.append(
                token_ids + [self.pad_token_id] * padding_length
            )
            padded_labels.append(
                target_ids + [self.ignore_index] * padding_length
            )
            attention_masks.append(
                [1] * sequence_length + [0] * padding_length
            )
        return {
            "input_ids": torch.tensor(
                padded_inputs,
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                padded_labels,
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                attention_masks,
                dtype=torch.long,
            ),
        }


class TrainingDataLoader:
    """
    Load chat records and prepare assistant-only causal-LM training data.
    assistant_mask_strategy:
        "template":
            Use assistant token masks returned by the tokenizer's chat
            template. The template must mark assistant output with
            {% generation %} ... {% endgeneration %} blocks.
        "verified_prefix":
            Locate assistant responses using tokenized conversation
            prefixes. Every prefix must exactly match the corresponding
            beginning of the full token sequence.
            This works only for compatible chat templates. Incompatible
            templates raise an error instead of receiving guessed masks.
    This class expects text-only system/user/assistant messages.
    Tool calls and multimodal messages require additional handling.
    """

    def __init__(
        self,
        *,
        assistant_mask_strategy: str = "template",
    ) -> None:
        allowed_strategies = {"template", "verified_prefix"}
        if assistant_mask_strategy not in allowed_strategies:
            raise ValueError(
                "assistant_mask_strategy must be "
                "'template' or 'verified_prefix'."
            )
        self.log = logging.getLogger(__name__)
        self.IGNORE_INDEX = IGNORE_INDEX
        self.assistant_mask_strategy = assistant_mask_strategy

    @staticmethod
    def _validate_messages(
        messages: Any,
    ) -> list[dict[str, str]]:
        """
        Validate and copy a text-only conversation.
        Require at least one non-empty assistant response because this
        loader constructs assistant-only training targets.
        Role alternation is not enforced here: individual chat templates
        may impose additional conversation rules during tokenization.
        """
        if not isinstance(messages, list) or not messages:
            raise ValueError("'messages' must be a non-empty list.")
        allowed_roles = {"system", "user", "assistant"}
        validated_messages: list[dict[str, str]] = []
        has_assistant_response = False
        for message_index, message in enumerate(messages):
            if not isinstance(message, Mapping):
                raise TypeError(
                    f"Message {message_index} must be a mapping."
                )
            role = message.get("role")
            content = message.get("content")
            if not isinstance(role, str) or role not in allowed_roles:
                raise ValueError(
                    f"Message {message_index} has unsupported role {role!r}."
                )
            if not isinstance(content, str):
                raise TypeError(
                    f"Message {message_index}: content must be a string."
                )
            if not content.strip():
                raise ValueError(
                    f"Message {message_index}: content must not be empty."
                )
            # Do not silently discard structured tool-call information.
            if "tool_calls" in message or "function_call" in message:
                raise ValueError(
                    "Tool/function-call messages are not supported."
                )
            validated_messages.append({
                "role": role,
                "content": content,
            })
            if role == "assistant":
                has_assistant_response = True
        if not has_assistant_response:
            raise ValueError(
                "Conversation must contain an assistant response."
            )
        return validated_messages

    def _record_to_conversation(
        self,
        record: Any,
    ) -> dict[str, list[dict[str, str]]]:
        """
        Convert a JSONL record into the common conversation format.

        Prefer messages when explicitly present. Otherwise, normalize
        prompt/completion using DataTransformer from this same module.

        An invalid messages field is not silently replaced by other fields.
        Text-only records are not implicitly treated as instruction data.
        """
        if not isinstance(record, Mapping):
            raise TypeError("Each JSONL record must be a JSON object.")
        if "messages" in record:
            messages = record["messages"]
        elif "prompt" in record and "completion" in record:
            normalized_record = (
                DataTransformer.normalize_prompt_completion(record)
            )
            messages = normalized_record["messages"]
        else:
            raise ValueError(
                "Expected 'messages' or both 'prompt' and 'completion'. "
                "Text-only data should use the program's language-model "
                "training path."
            )
        return {
            "messages": self._validate_messages(messages),
        }

    def _load_conversations(
        self,
        cfg: Any,
    ) -> list[dict[str, list[dict[str, str]]]]:
        """
        Dispatch using the program's attribute-based configuration.
        Expected fields:
            cfg.data.type: "chat_log" or "jsonl"
            cfg.data.path: file or directory path
            cfg.data.min_turns: optional, defaults to 1 for chat logs
        """
        data_config = cfg.data
        dataset_path = Path(data_config.path).expanduser()
        if data_config.type == "chat_log":
            return self._load_conversations_from_chat_log(
                dataset_path,
                min_turns=getattr(data_config, "min_turns", 1),
            )
        if data_config.type == "jsonl":
            return self._load_conversations_from_jsonl(dataset_path)
        raise ValueError(
            f"Unknown data.type {data_config.type!r}. "
            "Use 'chat_log' or 'jsonl'."
        )

    def _load_conversations_from_chat_log(
        self,
        chat_log_dir: str | Path,
        min_turns: int,
    ) -> list[dict[str, list[dict[str, str]]]]:
        """
        Load one conversation per JSON chat-log file.
        Expected log structure:
            {
                "session": {
                    "settings": {"system_prompt": "..."}
                },
                "turns": [
                    {
                        "user": {"content": "..."},
                        "assistant": {"content": "..."}
                    }
                ]
            }
        Incomplete turns are skipped. min_turns counts complete retained
        user/assistant pairs, not the number of raw turn records.
        Missing paths raise errors. Malformed individual files are logged
        and skipped.
        """
        if (
            isinstance(min_turns, bool)
            or not isinstance(min_turns, int)
            or min_turns < 1
        ):
            raise ValueError("min_turns must be a positive integer.")
        chat_log_dir = Path(chat_log_dir).expanduser()
        if not chat_log_dir.exists():
            raise FileNotFoundError(chat_log_dir)
        if not chat_log_dir.is_dir():
            raise NotADirectoryError(chat_log_dir)
        conversations: list[dict[str, list[dict[str, str]]]] = []
        # Sorting makes file traversal reproducible.
        for log_file in sorted(chat_log_dir.glob("*.json")):
            try:
                record = json.loads(
                    log_file.read_text(encoding="utf-8")
                )
                if not isinstance(record, Mapping):
                    raise ValueError("Chat log must contain a JSON object.")
                turns = record.get("turns", [])
                if not isinstance(turns, list):
                    raise ValueError("'turns' must be a list.")
                session = record.get("session") or {}
                if not isinstance(session, Mapping):
                    raise ValueError("'session' must be an object.")
                settings = session.get("settings") or {}
                if not isinstance(settings, Mapping):
                    raise ValueError("'session.settings' must be an object.")
                system_prompt = (
                    settings.get("system_prompt")
                    or session.get("system_prompt")
                )
                messages: list[dict[str, str]] = []
                if system_prompt is not None:
                    if not isinstance(system_prompt, str):
                        raise TypeError("system_prompt must be a string.")
                    if system_prompt.strip():
                        messages.append({
                            "role": "system",
                            "content": system_prompt,
                        })
                complete_turn_count = 0
                for turn_index, turn in enumerate(turns):
                    if not isinstance(turn, Mapping):
                        self.log.warning(
                            "Skipping %s turn %d: expected an object.",
                            log_file,
                            turn_index,
                        )
                        continue
                    user_message = turn.get("user")
                    assistant_message = turn.get("assistant")
                    if (
                        not isinstance(user_message, Mapping)
                        or not isinstance(assistant_message, Mapping)
                    ):
                        self.log.warning(
                            "Skipping %s turn %d: incomplete turn.",
                            log_file,
                            turn_index,
                        )
                        continue
                    user_content = user_message.get("content")
                    assistant_content = assistant_message.get("content")
                    if (
                        not isinstance(user_content, str)
                        or not user_content.strip()
                        or not isinstance(assistant_content, str)
                        or not assistant_content.strip()
                    ):
                        self.log.warning(
                            "Skipping %s turn %d: missing or empty text.",
                            log_file,
                            turn_index,
                        )
                        continue
                    messages.extend([
                        {"role": "user", "content": user_content},
                        {
                            "role": "assistant",
                            "content": assistant_content,
                        },
                    ])
                    complete_turn_count += 1
                if complete_turn_count < min_turns:
                    self.log.debug(
                        "Skipping %s: %d complete turns; need %d.",
                        log_file,
                        complete_turn_count,
                        min_turns,
                    )
                    continue
                conversations.append({
                    "messages": self._validate_messages(messages),
                })
            except (OSError, UnicodeError, TypeError, ValueError) as exc:
                self.log.warning(
                    "Skipping chat log %s: %s",
                    log_file,
                    exc,
                )
        self.log.info(
            "Loaded %d conversations from %s.",
            len(conversations),
            chat_log_dir,
        )
        return conversations

    def _load_conversations_from_jsonl(
        self,
        dataset_path: str | Path,
    ) -> list[dict[str, list[dict[str, str]]]]:
        """
        Read messages or prompt/completion records from a JSONL file.
        Blank lines are ignored. Malformed records are logged and skipped.
        Errors opening or reading the file propagate to the caller.
        Report physical, one-based line numbers for easy file inspection.
        """
        dataset_path = Path(dataset_path).expanduser()
        conversations: list[dict[str, list[dict[str, str]]]] = []
        with dataset_path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    conversation = self._record_to_conversation(record)
                except (TypeError, ValueError, OverflowError) as exc:
                    self.log.warning(
                        "Skipping %s line %d: %s",
                        dataset_path,
                        line_number,
                        exc,
                    )
                    continue
                conversations.append(conversation)
        self.log.info(
            "Loaded %d conversations from %s.",
            len(conversations),
            dataset_path,
        )
        return conversations

    @staticmethod
    def _validate_token_ids(token_ids: Any) -> list[int]:
        """
        Require a flat, unbatched list of token IDs.
        These methods deliberately do not request tensors. Padding and
        conversion to tensors happen later in the collator.
        """
        if not isinstance(token_ids, list):
            raise TypeError(
                "Chat template must return a flat list of token IDs."
            )
        if any(
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or token_id < 0
            for token_id in token_ids
        ):
            raise ValueError(
                "Token IDs must be non-negative integers."
            )
        return token_ids

    def _tokenize_with_template_mask(
        self,
        tokenizer: Any,
        messages: list[dict[str, str]],
    ) -> tuple[list[int], list[int]]:
        """
        Use assistant regions explicitly marked by the chat template.
        This is preferable to inferring token boundaries from string
        lengths or separately tokenized message content.
        Requires a tokenizer/template supporting:
            return_dict=True
            return_assistant_tokens_mask=True
        """
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_assistant_tokens_mask=True,
            truncation=False,
            padding=False,
        )
        if not isinstance(encoded, Mapping):
            raise TypeError(
                "Tokenizer did not return the requested dictionary."
            )
        token_ids = self._validate_token_ids(encoded.get("input_ids"))
        assistant_mask = encoded.get("assistant_masks")
        if not isinstance(assistant_mask, list):
            raise ValueError(
                "Tokenizer did not return 'assistant_masks'. "
                "Use a supported tokenizer and chat template."
            )
        if len(assistant_mask) != len(token_ids):
            raise ValueError(
                "Assistant mask length differs from token sequence length."
            )
        if any(
            not isinstance(value, (int, bool)) or value not in (0, 1)
            for value in assistant_mask
        ):
            raise ValueError("Assistant mask must contain only 0 and 1.")
        if not any(assistant_mask):
            raise ValueError(
                "The template returned no assistant tokens. "
                "Check its {% generation %} blocks and tokenizer support, "
                "or explicitly select assistant_mask_strategy="
                "'verified_prefix' for a compatible template."
            )
        labels = [
            token_id if is_assistant else self.IGNORE_INDEX
            for token_id, is_assistant in zip(
                token_ids,
                assistant_mask,
            )
        ]
        return token_ids, labels

    def _tokenize_with_verified_prefixes(
        self,
        tokenizer: Any,
        messages: list[dict[str, str]],
    ) -> tuple[list[int], list[int]]:
        """
        Infer assistant spans only when token prefixes are exact matches.
        For each assistant turn:
            start = length of prior context + assistant generation header
            end   = length of conversation through that assistant turn
        The assistant header is excluded from loss. Template suffixes
        following the answer, such as end-of-turn tokens, are included.
        This strategy is intentionally conservative. Templates that change
        earlier text depending on later messages are rejected.
        """
        def tokenize_prefix(
            prefix_messages: list[dict[str, str]],
            *,
            add_generation_prompt: bool,
        ) -> list[int]:
            token_ids = tokenizer.apply_chat_template(
                prefix_messages,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
                return_dict=False,      # explicit: we want a flat list of IDs
                truncation=False,
                padding=False,
            )
            # Defensive: some tokenizer versions return a BatchEncoding anyway.
            if isinstance(token_ids, Mapping):
                token_ids = token_ids["input_ids"]
            return self._validate_token_ids(token_ids)

        ''' SK EDITED: september 19 2026 around 7:22 PM ET
        def tokenize_prefix(
            prefix_messages: list[dict[str, str]],
            *,
            add_generation_prompt: bool,
        ) -> list[int]:
            token_ids = tokenizer.apply_chat_template(
                prefix_messages,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
                truncation=False,
                padding=False,
            )
            return self._validate_token_ids(token_ids)
        '''

        full_ids = tokenize_prefix(
            messages,
            add_generation_prompt=False,
        )
        labels = [self.IGNORE_INDEX] * len(full_ids)
        for message_index, message in enumerate(messages):
            if message["role"] != "assistant":
                continue
            previous_messages = messages[:message_index]
            # Rendering an empty conversation is unsupported by many
            # templates. Do not invent an empty user message to work around it.
            if not previous_messages:
                raise ValueError(
                    "verified_prefix requires context before an assistant "
                    "message. Use a suitable template mask or the text-only "
                    "training path for assistant-only documents."
                )
            context_ids = tokenize_prefix(
                previous_messages,
                add_generation_prompt=False,
            )
            generation_prefix_ids = tokenize_prefix(
                previous_messages,
                add_generation_prompt=True,
            )
            completed_prefix_ids = tokenize_prefix(
                messages[:message_index + 1],
                add_generation_prompt=False,
            )
            # Some templates ignore add_generation_prompt entirely.
            # In that case we cannot establish the assistant-header boundary.
            if generation_prefix_ids == context_ids:
                raise ValueError(
                    "Chat template did not add an assistant generation "
                    "header. Use template-provided assistant masks instead."
                )
            start = len(generation_prefix_ids)
            end = len(completed_prefix_ids)
            if (
                not 0 <= start < end <= len(full_ids)
                or full_ids[:start] != generation_prefix_ids
                or full_ids[:end] != completed_prefix_ids
            ):
                raise ValueError(
                    f"Assistant turn {message_index}: token prefixes do not "
                    "match the full conversation. Assistant boundaries "
                    "cannot be inferred safely for this template."
                )
            labels[start:end] = full_ids[start:end]
        return full_ids, labels

    def _build_labeled_example(
        self,
        tokenizer: Any,
        messages: Any,
        max_length: int,
    ) -> tuple[list[int], list[int]]:
        """
        Tokenize one conversation and keep loss only on assistant positions.

        Right truncation keeps the first max_length tokens. It may remove
        all assistant targets; _build_dataset detects and skips that case.

        Labels remain aligned with input_ids. Hugging Face causal-LM models
        normally perform the next-token shift internally.
        """
        if (
            isinstance(max_length, bool)
            or not isinstance(max_length, int)
            or max_length < 2
        ):
            raise ValueError("max_length must be an integer of at least 2.")

        validated_messages = self._validate_messages(messages)

        if self.assistant_mask_strategy == "template":
            input_ids, labels = self._tokenize_with_template_mask(
                tokenizer,
                validated_messages,
            )
        else:
            input_ids, labels = self._tokenize_with_verified_prefixes(
                tokenizer,
                validated_messages,
            )

        input_ids = input_ids[:max_length]
        labels = labels[:max_length]

        # In an ordinary next-token loss, position 0 has no preceding
        # prediction. Mark it ignored explicitly so our usability check
        # counts only targets that actually participate in the shifted loss.
        if labels:
            labels[0] = self.IGNORE_INDEX

        return input_ids, labels

    def _build_dataset(
        self,
        conversations: Iterable[Mapping[str, Any]],
        tokenizer: Any,
        max_length: int,
    ) -> TokenizedChatDataset:
        """
        Eagerly tokenize conversations and return an unpadded Dataset.
        Conversations without surviving trainable tokens are skipped.
        Tokenization/template errors are fatal: silently skipping them
        could hide a systematic masking bug.
        Memory use is proportional to the complete tokenized dataset.
        """
        examples: list[dict[str, list[int]]] = []
        skipped_count = 0
        for conversation_index, conversation in enumerate(conversations):
            if (
                not isinstance(conversation, Mapping)
                or "messages" not in conversation
            ):
                raise ValueError(
                    f"Conversation {conversation_index} must contain "
                    "'messages'."
                )
            try:
                input_ids, labels = self._build_labeled_example(
                    tokenizer,
                    conversation["messages"],
                    max_length,
                )
            except Exception as exc:
                # Re-raise with context; do not silently swallow failures.
                raise ValueError(
                    f"Could not tokenize conversation "
                    f"{conversation_index}: {exc}"
                ) from exc

            has_trainable_tokens = any(
                label != self.IGNORE_INDEX
                for label in labels
            )
            if not has_trainable_tokens:
                skipped_count += 1
                continue
            examples.append({
                "input_ids": input_ids,
                "labels": labels,
            })
        if skipped_count:
            self.log.info(
                "Skipped %d conversations with no trainable tokens "
                "after truncation and next-token alignment.",
                skipped_count,
            )
        if not examples:
            raise ValueError(
                "No usable training examples remain. Check the source data, "
                "chat template, assistant masks, and max_length."
            )
        self.log.info(
            "Prepared %d tokenized training examples.",
            len(examples),
        )
        return TokenizedChatDataset(examples)

    def _make_collator(
        self,
        pad_token_id: int,
    ) -> CausalLMCollator:
        """Create a reusable right-padding collator."""
        return CausalLMCollator(
            pad_token_id=pad_token_id,
            ignore_index=self.IGNORE_INDEX,
        )
