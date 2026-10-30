"""
ThinkRL COPO Trainer
====================

Dedicated trainer for COPO (Count-based Online Preference Optimization).

Author: EllanorAI
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer
from tqdm import tqdm

from thinkrl.algorithms.copo import COPOAlgorithm, COPOConfig
from thinkrl.data.loaders import RLHFDataLoader
from thinkrl.utils.logging import get_logger


logger = get_logger(__name__)


@dataclass
class COPOTrainerConfig:
    """Configuration for COPOTrainer's loop. COPO's own hyperparameters (alpha,
    beta, the CFN settings, ...) live on COPOConfig, not here."""

    num_iterations: int = 10
    prompts_per_iteration: int = 8
    output_dir: str = "./copo_output"


class COPOTrainer:
    """
    Trainer for COPO.

    COPO is online but is not a member of the rollout family GRPOTrainer drives:
    one iteration generates N completions per prompt, scores them with a reward
    model, pairs best against worst, updates the coin-flipping network on every
    visited state and only then takes DPO-shaped optimizer steps over the pairs
    it just built. `COPOAlgorithm.run_iteration` already owns that sequence, so
    this trainer is the prompt-feeding loop around it and nothing more: it reads
    prompt batches off an RLHFDataset and hands each batch to `run_iteration`.

    Kept as its own class and its own CLI command (`cli/copo.py`) per #124,
    which listed COPO among the algorithms that shipped a loss, a config, a
    factory and a registry entry with no code path that stepped an optimizer
    with them.

    Example:
        ```python
        trainer = COPOTrainer(
            model=policy_model,
            reward_model=reward_model,
            tokenizer=tokenizer,
            dataset=RLHFDataset("gsm8k", tokenizer=tokenizer),
            config=COPOConfig(alpha=0.1),
        )
        trainer.train()
        trainer.save_model("./copo_checkpoint")
        ```
    """

    def __init__(
        self,
        model: nn.Module,
        reference_model: nn.Module | None = None,
        reward_model: nn.Module | None = None,
        tokenizer: Any = None,
        dataset: Any = None,
        config: COPOConfig | None = None,
        optimizer: Optimizer | None = None,
        args: COPOTrainerConfig | None = None,
    ):
        if tokenizer is None:
            raise ValueError(
                "COPOTrainer requires a tokenizer: COPO encodes its own generations for the reward model."
            )
        if dataset is None:
            raise ValueError(
                "COPOTrainer requires a prompt dataset: COPO generates the preference pairs it trains on."
            )

        config = config or COPOConfig()
        config.hidden_size = self._resolve_hidden_size(model, config.hidden_size)

        self.algorithm = COPOAlgorithm(
            policy_model=model,
            reference_model=reference_model,
            reward_model=reward_model,
            config=config,
            optimizer=optimizer,
            tokenizer=tokenizer,
        )
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.config = config
        self.args = args or COPOTrainerConfig()

    @staticmethod
    def _resolve_hidden_size(model: nn.Module, configured: int) -> int:
        """Take the policy's real hidden size over COPOConfig's default.

        The coin-flipping network is a Linear stack over the policy's last hidden
        state, so `cfn_hidden_size` has to be the policy's width. COPOConfig
        defaults it to 4096 (Llama-3-8B), which is a shape error waiting at the
        first exploration bonus for every other model, raised from inside the CFN
        rather than from the configuration that was wrong.
        """
        actual = getattr(getattr(model, "config", None), "hidden_size", None)
        if actual is None or actual == configured:
            return configured
        if configured != COPOConfig.hidden_size:
            logger.warning(
                f"COPOConfig.hidden_size={configured} does not match the policy model's "
                f"hidden size ({actual}); using {actual}, or the coin-flipping network "
                "cannot consume the policy's hidden states."
            )
        return int(actual)

    def train(self) -> dict[str, Any]:
        """Run `num_iterations` COPO iterations and return the last one's metrics."""
        # Checked before the loader is built, not after: RLHFDataLoader shuffles, and
        # RandomSampler raises "num_samples should be a positive integer value" on an
        # empty dataset, which says nothing about COPO needing prompts.
        if len(self.dataset) == 0:
            raise ValueError("Prompt dataset is empty: COPO has nothing to generate from.")

        dataloader = RLHFDataLoader(
            dataset=self.dataset,
            tokenizer=self.tokenizer,
            batch_size=self.args.prompts_per_iteration,
            shuffle=True,
            drop_last=False,
            # COPO's generations go straight into model.generate, and a decoder-only
            # model continues from the last position, so right padding would have short
            # prompts continue from pad tokens (same reason as GRPOTrainer).
            padding_side="left",
        )
        metrics: dict[str, Any] = {}
        completed = 0
        progress = tqdm(total=self.args.num_iterations, desc="COPO iterations")

        while completed < self.args.num_iterations:
            for batch in dataloader:
                if completed >= self.args.num_iterations:
                    break
                prompts = batch["prompt_text"]
                metrics = self.algorithm.run_iteration(prompts)
                completed += 1
                progress.update(1)
                logger.info(f"iteration {completed}/{self.args.num_iterations}: {metrics}")

        progress.close()
        return metrics

    def save_model(self, output_dir: str | None = None) -> str:
        """Save the policy model, the tokenizer and the coin-flipping network.

        The CFN is part of the trained state, not a scratch buffer: it carries the
        visitation counts the exploration bonus is derived from, so a checkpoint
        without it restarts exploration from zero.
        """
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Loose type on purpose: `save_pretrained` is a transformers method, not an
        # nn.Module one, so nn.Module.__getattr__ types it as Tensor | Module.
        model: Any = self.algorithm.policy_model
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(output_dir)
        else:
            torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
        if hasattr(self.tokenizer, "save_pretrained"):
            self.tokenizer.save_pretrained(output_dir)
        torch.save(self.algorithm.cfn.state_dict(), os.path.join(output_dir, "cfn.pt"))

        logger.info(f"Policy model and coin-flipping network saved to {output_dir}")
        return output_dir


__all__ = ["COPOTrainerConfig", "COPOTrainer"]
