"""
ThinkRL Datasets
================

Dataset classes for Reinforcement Learning from Human Feedback (RLHF).
Includes support for:
- SFT Datasets (Supervised Fine-Tuning)
- Preference Datasets (Reward Modeling/DPO)
- Prompt-only Datasets (RL/PPO)

Author: Archit Sood @ EllanorAI
"""

from collections.abc import Callable
import logging
from typing import Any

from torch.utils.data import Dataset


# --- Fix: Make datasets optional ---
try:
    from datasets import load_dataset

    _DATASETS_AVAILABLE = True
except ImportError:
    load_dataset = None
    _DATASETS_AVAILABLE = False
# -----------------------------------

logger = logging.getLogger(__name__)


class BaseRLHFDataset(Dataset):
    """Base class for RLHF datasets."""

    def __init__(
        self,
        tokenizer: Any,
        dataset_name_or_path: str | None,
        max_length: int = 512,
        split: str = "train",
        preprocess_fn: Callable | None = None,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocess_fn = preprocess_fn

        if dataset_name_or_path:
            # Check availability before usage
            if not _DATASETS_AVAILABLE:
                raise ImportError(
                    "The 'datasets' library is required to load datasets from path/name. "
                    "Please install it via `pip install datasets` or `pip install thinkrl[sota]`."
                )

            # Load dataset
            source = kwargs.pop("source", None)

            if isinstance(dataset_name_or_path, str):
                dataset_config = kwargs.pop("dataset_config", None)
                if source and source != "hf":
                    # Load from local files with specific format (json, csv, text)
                    self.dataset = load_dataset(source, data_files=dataset_name_or_path, split=split)
                elif dataset_name_or_path.endswith((".json", ".jsonl")):
                    # Auto-detect JSON
                    self.dataset = load_dataset("json", data_files=dataset_name_or_path, split=split)
                else:
                    # Default HF hub loading (with optional config)
                    if dataset_config:
                        self.dataset = load_dataset(dataset_name_or_path, dataset_config, split=split)
                    else:
                        self.dataset = load_dataset(dataset_name_or_path, split=split)
            else:
                self.dataset = dataset_name_or_path
        else:
            self.dataset = []

    def __len__(self) -> int:
        if hasattr(self, "data"):
            return len(self.data)
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        raise NotImplementedError


class RLHFDataset(BaseRLHFDataset):
    """
    Dataset for Supervised Fine-Tuning (SFT) or PPO Prompt generation.
    Expects data with a prompt column. Automatically filters invalid rows.
    """

    def __init__(
        self,
        dataset_name_or_path: str,
        tokenizer: Any,
        prompt_column: str = "prompt",
        response_column: str | None = None,
        max_length: int = 512,
        max_samples: int | None = None,
        split: str = "train",
        preprocess_fn: Callable | None = None,
        apply_chat_template: bool = True,
        **kwargs,
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_name_or_path=dataset_name_or_path,
            max_length=max_length,
            split=split,
            preprocess_fn=preprocess_fn,
            **kwargs,
        )
        self.prompt_column = prompt_column
        self.response_column = response_column
        self.system_prompt = kwargs.get("system_prompt", None)
        self.target_column = kwargs.get("target_column", "answer")
        self.apply_chat_template = apply_chat_template
        self._chat_template_failed = False

        # Filter and load data into memory to handle invalid rows efficiently
        self.data = []
        for item in self.dataset:
            if max_samples and len(self.data) >= max_samples:
                break
            # Custom preprocessing first
            if self.preprocess_fn:
                try:
                    item = self.preprocess_fn(item)
                except Exception:
                    continue  # Skip if preprocessing fails

            prompt = item.get(self.prompt_column)
            if not prompt or not isinstance(prompt, str) or not prompt.strip():
                continue

            # Basic cleaning
            item[self.prompt_column] = prompt.strip()

            # Store target if available
            if self.target_column in item:
                item[self.target_column] = str(item[self.target_column]).strip()

            self.data.append(item)

        if not self.data:
            logger.warning(
                f"No valid samples found in {dataset_name_or_path} with prompt_column='{prompt_column}'. "
                f"Dataset length: {len(self.dataset)}"
            )
        else:
            logger.info(f"Loaded {len(self.data)} valid samples from {dataset_name_or_path}")

    def _use_chat_template(self) -> bool:
        """Chat template applies only when asked for and the tokenizer defines one."""
        if not self.apply_chat_template or self._chat_template_failed:
            return False
        if not hasattr(self.tokenizer, "apply_chat_template"):
            return False
        return getattr(self.tokenizer, "chat_template", None) is not None

    def _render(self, prompt: str, response: str | None) -> tuple[str, str, bool]:
        """Return (prompt_text, text, templated).

        prompt_text is what the policy is asked to continue, so it carries the
        generation prompt. text is what gets tokenized, and in SFT mode also
        carries the response. templated says whether the chat template ran, which
        the caller needs because a rendered template already contains the special
        tokens the tokenizer would otherwise add a second time.

        Falls back to plain concatenation for base models, for tokenizers without
        a template, and if rendering raises.
        """
        if self._use_chat_template():
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": prompt})
            try:
                prompt_text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                if response is None:
                    return prompt_text, prompt_text, True
                text = self.tokenizer.apply_chat_template(
                    [*messages, {"role": "assistant", "content": response}],
                    tokenize=False,
                    add_generation_prompt=False,
                )
                return prompt_text, text, True
            except Exception as exc:  # noqa: BLE001 - any jinja template can raise anything
                self._chat_template_failed = True
                logger.warning(
                    f"Chat template failed ({exc}); falling back to raw prompts for this dataset. "
                    "Pass apply_chat_template=False to silence this."
                )

        prompt_text = prompt
        if self.system_prompt:
            prompt_text = f"{self.system_prompt}\n\n{prompt}"
        text = prompt_text if response is None else f"{prompt_text} {response}"
        return prompt_text, text, False

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.data[idx]

        prompt = sample.get(self.prompt_column)

        response = None
        if self.response_column and self.response_column in sample:
            response = str(sample[self.response_column]).strip()

        prompt, text, templated = self._render(prompt, response)

        # Tokenize. A rendered chat template already carries BOS and the role
        # markers, so letting the tokenizer add specials again duplicates them.
        encodings = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=False,  # Padding handled by collator
            truncation=True,
            return_tensors="pt",
            add_special_tokens=not templated,
        )

        return {
            "input_ids": encodings["input_ids"].squeeze(0),
            "attention_mask": encodings["attention_mask"].squeeze(0),
            "prompt_text": prompt,
            "target": sample.get(self.target_column, ""),
        }


class PreferenceDataset(BaseRLHFDataset):
    """
    Dataset for Reward Modeling or DPO.
    Expects 'chosen' and 'rejected' columns. Automatically filters invalid rows.
    """

    def __init__(
        self,
        dataset_name_or_path: str,
        tokenizer: Any,
        prompt_column: str = "prompt",
        chosen_column: str = "chosen",
        rejected_column: str = "rejected",
        max_length: int = 512,
        max_samples: int | None = None,
        split: str = "train",
        **kwargs,
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_name_or_path=dataset_name_or_path,
            max_length=max_length,
            split=split,
            **kwargs,
        )
        self.prompt_column = prompt_column
        self.chosen_column = chosen_column
        self.rejected_column = rejected_column

        # Filter data
        self.data = []
        for item in self.dataset:
            if max_samples and len(self.data) >= max_samples:
                break
            prompt = item.get(self.prompt_column)
            chosen = item.get(self.chosen_column)
            rejected = item.get(self.rejected_column)

            if not prompt or not isinstance(prompt, str) or not prompt.strip():
                continue
            if not chosen or not isinstance(chosen, str) or not chosen.strip():
                continue
            if not rejected or not isinstance(rejected, str) or not rejected.strip():
                continue

            item[self.prompt_column] = prompt.strip()
            item[self.chosen_column] = chosen.strip()
            item[self.rejected_column] = rejected.strip()
            self.data.append(item)

        logger.info(f"Loaded {len(self.data)} valid preference pairs from {dataset_name_or_path}")

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.data[idx]

        prompt = sample.get(self.prompt_column)
        chosen = sample.get(self.chosen_column)
        rejected = sample.get(self.rejected_column)

        # Format: Prompt + Response + EOS
        def tokenize_pair(text_a, text_b):
            # Note: Simple concatenation. In real scenarios, use chat templates.
            full_text = f"{text_a}{text_b}{self.tokenizer.eos_token}"
            return self.tokenizer(
                full_text,
                max_length=self.max_length,
                padding=False,
                truncation=True,
                return_tensors="pt",
            )

        chosen_enc = tokenize_pair(prompt, chosen)
        rejected_enc = tokenize_pair(prompt, rejected)

        return {
            "chosen_input_ids": chosen_enc["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_enc["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected_enc["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_enc["attention_mask"].squeeze(0),
            "prompt": prompt,
        }
