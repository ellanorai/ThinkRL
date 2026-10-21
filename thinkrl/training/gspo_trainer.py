"""
ThinkRL GSPO Trainer
=====================

Dedicated trainer for GSPO (sequence-level GRPO).

Author: EllanorAI
"""

from __future__ import annotations

import torch.nn as nn
from torch.optim import Optimizer

from thinkrl.algorithms.gspo import GSPOAlgorithm, GSPOConfig
from thinkrl.training.grpo_trainer import GRPOTrainer


class GSPOTrainer(GRPOTrainer):
    """
    Trainer for GSPO.

    Drives GSPOAlgorithm through the same prompt -> rollout -> reward ->
    train_on_rollout loop GRPOTrainer already implements (GSPO's rollout
    mechanics are identical to GRPO's; only the surrogate loss inside
    `train_on_rollout` differs -- a length-normalized, sequence-level ratio
    instead of a per-token one -- which GSPOAlgorithm itself already owns).
    Kept as its own class and its own CLI command (`cli/gspo.py`) rather than
    `GRPOTrainer(algorithm=...)`, per #124/#195.
    """

    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module | None = None,
        config: GSPOConfig | None = None,
        optimizer: Optimizer | None = None,
        **kwargs,
    ):
        algorithm = GSPOAlgorithm(
            policy_model=model,
            ref_model=ref_model,
            optimizer=optimizer,
            config=config or GSPOConfig(),
        )
        super().__init__(algorithm=algorithm, **kwargs)
