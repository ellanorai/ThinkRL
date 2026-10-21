"""
ThinkRL DAPO Trainer
=====================

Dedicated trainer for Decoupled Clip and Dynamic Sampling Policy Optimization (DAPO).

Author: EllanorAI
"""

from __future__ import annotations

import torch.nn as nn
from torch.optim import Optimizer

from thinkrl.algorithms.dapo import DAPOAlgorithm, DAPOConfig
from thinkrl.training.grpo_trainer import GRPOTrainer


class DAPOTrainer(GRPOTrainer):
    """
    Trainer for Decoupled Clip and Dynamic Sampling Policy Optimization (DAPO).

    Drives DAPOAlgorithm through the same prompt -> rollout -> reward ->
    train_on_rollout loop GRPOTrainer already implements (DAPO's rollout mechanics
    are identical to GRPO's; only the update rule inside `train_on_rollout` differs,
    which DAPOAlgorithm itself already owns). Kept as its own class and its own CLI
    command (`cli/dapo.py`) rather than `GRPOTrainer(algorithm=...)`, per #124.
    """

    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module | None = None,
        config: DAPOConfig | None = None,
        optimizer: Optimizer | None = None,
        **kwargs,
    ):
        algorithm = DAPOAlgorithm(
            policy_model=model,
            ref_model=ref_model,
            optimizer=optimizer,
            config=config or DAPOConfig(),
        )
        super().__init__(algorithm=algorithm, **kwargs)
