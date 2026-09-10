"""
Reward Model Trainer
====================

Pairwise preference training for reward models, the step between SFT and RL
in the standard RLHF pipeline.

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

from thinkrl.models.loss import PairWiseLoss
from thinkrl.utils.logging import get_logger


logger = get_logger(__name__)


@dataclass
class RMConfig:
    """Configuration for reward model training."""

    # Optimization
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.0

    # Training
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 1

    # Loss
    margin: float = 0.0

    # Logging and output
    logging_steps: int = 10
    output_dir: str = "./rm_output"


class RMTrainer:
    """
    Trains a reward model on (chosen, rejected) preference pairs.

    The objective is the standard Bradley-Terry pairwise loss,
    ``-logsigmoid(r_chosen - r_rejected)``, implemented in
    :class:`thinkrl.models.loss.PairWiseLoss`.

    Example:
        ```python
        from thinkrl.data.datasets import PreferenceDataset
        from thinkrl.models.reward_model import RewardModel

        dataset = PreferenceDataset("Anthropic/hh-rlhf", tokenizer=tokenizer)
        trainer = RMTrainer(
            model=RewardModel("meta-llama/Llama-3.1-8B"),
            tokenizer=tokenizer,
            train_dataset=dataset,
        )
        trainer.train()
        trainer.save_model("./rm_checkpoint")
        ```
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        train_dataset: Any,
        args: RMConfig | None = None,
        data_collator: Callable | None = None,
        device: torch.device | str | None = None,
    ):
        if model is None:
            raise ValueError("RMTrainer requires a reward model.")
        if tokenizer is None:
            raise ValueError("RMTrainer requires a tokenizer: pairs are padded with its pad token.")

        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.args = args or RMConfig()
        self.data_collator = data_collator or self._default_collator
        self.loss_fn = PairWiseLoss(margin=self.args.margin)

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
        """Pad chosen and rejected to one common length.

        Both halves are padded together rather than separately, because they go
        through the model as a single batch and a ragged split would misalign
        the two reward vectors.
        """
        required = ("chosen_input_ids", "rejected_input_ids")
        for key in required:
            if key not in batch[0]:
                raise KeyError(f"Preference batch is missing {key!r}; RMTrainer expects a PreferenceDataset.")

        sequences = [torch.as_tensor(x["chosen_input_ids"]) for x in batch]
        sequences += [torch.as_tensor(x["rejected_input_ids"]) for x in batch]
        masks = [torch.as_tensor(x["chosen_attention_mask"]) for x in batch]
        masks += [torch.as_tensor(x["rejected_attention_mask"]) for x in batch]

        max_length = max(seq.size(0) for seq in sequences)
        input_ids = torch.full((len(sequences), max_length), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(sequences), max_length), dtype=torch.long)
        for i, (seq, mask) in enumerate(zip(sequences, masks, strict=True)):
            input_ids[i, : seq.size(0)] = seq
            attention_mask[i, : mask.size(0)] = mask

        return {"input_ids": input_ids, "attention_mask": attention_mask}

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

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Score both halves in one forward pass and rank them."""
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        rewards = self.model(input_ids=input_ids, attention_mask=attention_mask)
        if isinstance(rewards, tuple):
            rewards = rewards[0]
        rewards = rewards.squeeze(-1) if rewards.dim() > 1 else rewards

        # The collator stacks every chosen sequence before every rejected one.
        half = rewards.size(0) // 2
        chosen_rewards, rejected_rewards = rewards[:half], rewards[half:]
        return self.loss_fn(chosen_rewards, rejected_rewards)

    def train(self) -> dict[str, float]:
        """Run pairwise training and return the last metrics."""
        dataloader = self.get_train_dataloader()
        if len(dataloader) == 0:
            raise ValueError("Training dataloader is empty: no preference pairs to train on.")

        self.model.to(self.device)
        self.model.train()

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )

        total_steps = (len(dataloader) // self.args.gradient_accumulation_steps) * self.args.num_train_epochs
        scheduler = None
        if self.args.warmup_ratio > 0 and total_steps > 0:
            warmup_steps = int(total_steps * self.args.warmup_ratio)
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, total_iters=max(warmup_steps, 1)
            )

        metrics: dict[str, float] = {}
        for epoch in range(self.args.num_train_epochs):
            progress = tqdm(dataloader, desc=f"RM epoch {epoch + 1}/{self.args.num_train_epochs}")
            for step, batch in enumerate(progress):
                loss, step_metrics = self.compute_loss(batch)
                (loss / self.args.gradient_accumulation_steps).backward()

                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad()
                    self.global_step += 1

                metrics = {k: float(v) for k, v in step_metrics.items()}
                if self.global_step and self.global_step % self.args.logging_steps == 0:
                    logger.info(f"step {self.global_step}: {metrics}")

        if self.global_step == 0:
            raise RuntimeError(
                "Reward model training finished without a single optimizer step. "
                "Check batch size and gradient_accumulation_steps against the dataset size."
            )

        return metrics

    def save_model(self, output_dir: str | None = None) -> str:
        """Save the reward model and tokenizer."""
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(output_dir)
        else:
            torch.save(self.model.state_dict(), os.path.join(output_dir, "model.pt"))
        if hasattr(self.tokenizer, "save_pretrained"):
            self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Reward model saved to {output_dir}")
        return output_dir


def create_rm_trainer(model: Any, tokenizer: Any, train_dataset: Any, **kwargs: Any) -> RMTrainer:
    """Build an RMTrainer, passing unknown keyword arguments to RMConfig."""
    config_fields = RMConfig.__dataclass_fields__
    config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
    trainer_kwargs = {k: v for k, v in kwargs.items() if k not in config_fields}
    return RMTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=RMConfig(**config_kwargs),
        **trainer_kwargs,
    )


__all__ = ["RMConfig", "RMTrainer", "create_rm_trainer"]
