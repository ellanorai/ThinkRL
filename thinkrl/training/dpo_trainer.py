"""
DPO Trainer
===========

Offline preference training, the step DPO/IPO take instead of PPO-style rollouts.

The batch shape (chosen/rejected input_ids + labels) and the training loop are
identical across DPO and IPO -- they differ only in the loss math inside each
algorithm's own compute_loss. This trainer drives any BaseRLHFAlgorithm exposing
that shape (#124) rather than duplicating the loop per algorithm.

Author: EllanorAI
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import os
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from thinkrl.utils.logging import get_logger


logger = get_logger(__name__)


@dataclass
class DPOTrainerConfig:
    """Configuration for DPOTrainer's loop. Loss hyperparameters (beta, loss_type,
    ...) live on the algorithm's own config (DPOConfig/IPOConfig), not here."""

    num_train_epochs: int = 1
    per_device_train_batch_size: int = 4
    logging_steps: int = 10
    output_dir: str = "./dpo_output"


class DPOTrainer:
    """
    Trains a policy against chosen/rejected preference pairs via any algorithm
    implementing the DPO-shaped training_step (DPOAlgorithm, IPOAlgorithm).

    Example:
        ```python
        from thinkrl.algorithms.dpo import DPOAlgorithm
        from thinkrl.data.datasets import PreferenceDataset

        dataset = PreferenceDataset("Anthropic/hh-rlhf", tokenizer=tokenizer)
        trainer = DPOTrainer(
            algorithm=DPOAlgorithm(policy_model, ref_model),
            tokenizer=tokenizer,
            train_dataset=dataset,
        )
        trainer.train()
        trainer.save_model("./dpo_checkpoint")
        ```
    """

    def __init__(
        self,
        algorithm: Any,
        tokenizer: Any,
        train_dataset: Any,
        args: DPOTrainerConfig | None = None,
        data_collator: Callable | None = None,
        device: torch.device | str | None = None,
    ):
        if algorithm is None:
            raise ValueError("DPOTrainer requires an algorithm (DPOAlgorithm, IPOAlgorithm, ...).")
        if tokenizer is None:
            raise ValueError("DPOTrainer requires a tokenizer: pairs are padded with its pad token.")
        if not hasattr(algorithm, "training_step"):
            raise TypeError(
                f"{type(algorithm).__name__} has no training_step; DPOTrainer needs an algorithm whose "
                "compute_loss reads chosen_input_ids/chosen_attention_mask/chosen_labels and the "
                "rejected_* equivalents (DPOAlgorithm, IPOAlgorithm)."
            )

        self.algorithm = algorithm
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.args = args or DPOTrainerConfig()
        self.data_collator = data_collator or self._default_collator

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)

        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(tokenizer, "eos_token_id", None)
        if pad_token_id is None:
            raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id, so pairs cannot be padded.")
        self.pad_token_id = pad_token_id

        self.global_step = 0

    def _default_collator(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Pad chosen and rejected to one shared length, each with its own labels
        masking the prompt (and padding) out at -100, using PreferenceDataset's
        prompt_length.

        Both sides must share one length, not be padded separately to their own
        max: DPOAlgorithm/IPOAlgorithm's compute_loss torch.cat's chosen and
        rejected along the batch dim for a single forward pass, which requires
        matching sequence length.
        """
        required = ("chosen_input_ids", "rejected_input_ids")
        for key in required:
            if key not in batch[0]:
                raise KeyError(f"Preference batch is missing {key!r}; DPOTrainer expects a PreferenceDataset.")

        max_length = max(
            max(torch.as_tensor(x["chosen_input_ids"]).size(0) for x in batch),
            max(torch.as_tensor(x["rejected_input_ids"]).size(0) for x in batch),
        )

        def pad_side(ids_key: str, mask_key: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            sequences = [torch.as_tensor(x[ids_key]) for x in batch]
            masks = [torch.as_tensor(x[mask_key]) for x in batch]
            input_ids = torch.full((len(sequences), max_length), self.pad_token_id, dtype=torch.long)
            attention_mask = torch.zeros((len(sequences), max_length), dtype=torch.long)
            labels = torch.full((len(sequences), max_length), -100, dtype=torch.long)
            for i, (seq, mask) in enumerate(zip(sequences, masks, strict=True)):
                input_ids[i, : seq.size(0)] = seq
                attention_mask[i, : mask.size(0)] = mask
                prompt_length = min(batch[i].get("prompt_length", 0), seq.size(0))
                labels[i, prompt_length : seq.size(0)] = seq[prompt_length:]
            return input_ids, attention_mask, labels

        chosen_input_ids, chosen_attention_mask, chosen_labels = pad_side("chosen_input_ids", "chosen_attention_mask")
        rejected_input_ids, rejected_attention_mask, rejected_labels = pad_side(
            "rejected_input_ids", "rejected_attention_mask"
        )

        return {
            "chosen_input_ids": chosen_input_ids,
            "chosen_attention_mask": chosen_attention_mask,
            "chosen_labels": chosen_labels,
            "rejected_input_ids": rejected_input_ids,
            "rejected_attention_mask": rejected_attention_mask,
            "rejected_labels": rejected_labels,
        }

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("No training dataset provided.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

    def train(self) -> dict[str, float]:
        """Run offline preference training and return the last step's metrics."""
        dataloader = self.get_train_dataloader()
        if len(dataloader) == 0:
            raise ValueError("Training dataloader is empty: no preference pairs to train on.")

        self.algorithm.to(self.device)

        metrics: dict[str, float] = {}
        for epoch in range(self.args.num_train_epochs):
            progress = tqdm(dataloader, desc=f"DPO epoch {epoch + 1}/{self.args.num_train_epochs}")
            for batch in progress:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                metrics = self.algorithm.training_step(batch)
                self.global_step += 1
                if self.global_step % self.args.logging_steps == 0:
                    logger.info(f"step {self.global_step}: {metrics}")

        if self.global_step == 0:
            raise RuntimeError(
                "DPO training finished without a single optimizer step. "
                "Check batch size and gradient_accumulation_steps against the dataset size."
            )

        return metrics

    def save_model(self, output_dir: str | None = None) -> str:
        """Save the policy model and tokenizer."""
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        model = self.algorithm.policy_model
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(output_dir)
        else:
            torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
        if hasattr(self.tokenizer, "save_pretrained"):
            self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Policy model saved to {output_dir}")
        return output_dir


__all__ = ["DPOTrainerConfig", "DPOTrainer"]
