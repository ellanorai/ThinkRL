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

import logging
from typing import Any, Callable, Dict, List, Optional, Union
import torch
from torch.utils.data import Dataset
from datasets import load_dataset

logger = logging.getLogger(__name__)

class BaseRLHFDataset(Dataset):
    """Base class for RLHF datasets."""

    def __init__(
        self,
        tokenizer: Any,
        dataset_name_or_path: Optional[str],
        max_length: int = 512,
        split: str = "train",
        preprocess_fn: Optional[Callable] = None,
        **kwargs
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocess_fn = preprocess_fn
        
        if dataset_name_or_path:
            # Load dataset
            if isinstance(dataset_name_or_path, str):
                if dataset_name_or_path.endswith((".json", ".jsonl")):
                    self.dataset = load_dataset("json", data_files=dataset_name_or_path, split=split)
                else:
                    self.dataset = load_dataset(dataset_name_or_path, split=split)
            else:
                self.dataset = dataset_name_or_path
        else:
            self.dataset = []

    def __len__(self) -> int:
        # Assuming self.data holds the processed list if populated in subclasses
        if hasattr(self, 'data') and self.data:
             return len(self.data)
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
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
        response_column: Optional[str] = None,
        max_length: int = 512,
        split: str = "train",
        preprocess_fn: Optional[Callable] = None,
        **kwargs
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_name_or_path=dataset_name_or_path,
            max_length=max_length,
            split=split,
            preprocess_fn=preprocess_fn,
            **kwargs
        )
        self.prompt_column = prompt_column
        self.response_column = response_column
        
        # Filter and load data into memory to handle invalid rows efficiently
        self.data = []
        for item in self.dataset:
            # Custom preprocessing first
            if self.preprocess_fn:
                try:
                    item = self.preprocess_fn(item)
                except Exception:
                    continue # Skip if preprocessing fails
            
            prompt = item.get(self.prompt_column)
            if not prompt or not isinstance(prompt, str) or not prompt.strip():
                continue
            
            # Basic cleaning
            item[self.prompt_column] = prompt.strip()
            self.data.append(item)
            
        logger.info(f"Loaded {len(self.data)} valid samples from {dataset_name_or_path}")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.data[idx]

        # Extract text
        prompt = sample.get(self.prompt_column)
        
        # Add response if available (SFT mode)
        text = prompt
        if self.response_column and self.response_column in sample:
            response = str(sample[self.response_column]).strip()
            text = f"{prompt} {response}"

        # Tokenize
        encodings = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=False, # Padding handled by collator
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encodings["input_ids"].squeeze(0),
            "attention_mask": encodings["attention_mask"].squeeze(0),
            "prompt_text": prompt,
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
        split: str = "train",
        **kwargs
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_name_or_path=dataset_name_or_path,
            max_length=max_length,
            split=split,
            **kwargs
        )
        self.prompt_column = prompt_column
        self.chosen_column = chosen_column
        self.rejected_column = rejected_column
        
        # Filter data
        self.data = []
        for item in self.dataset:
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

    def __getitem__(self, idx: int) -> Dict[str, Any]:
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
                return_tensors="pt"
            )

        chosen_enc = tokenize_pair(prompt, chosen)
        rejected_enc = tokenize_pair(prompt, rejected)

        return {
            "chosen_input_ids": chosen_enc["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_enc["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected_enc["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_enc["attention_mask"].squeeze(0),
            "prompt": prompt
        }