"""
ThinkRL Dr.GRPO Trainer
=========================

Dedicated trainer for Dr.GRPO (GRPO without group-normalization bias).

Author: EllanorAI
"""

from __future__ import annotations

import torch.nn as nn
from torch.optim import Optimizer

from thinkrl.algorithms.dr_grpo import DrGRPOAlgorithm, DrGRPOConfig
from thinkrl.training.grpo_trainer import GRPOTrainer


class DrGRPOTrainer(GRPOTrainer):
    """
    Trainer for Dr.GRPO.

    Drives DrGRPOAlgorithm through the same prompt -> rollout -> reward ->
    train_on_rollout loop GRPOTrainer already implements (Dr.GRPO's rollout
    mechanics are identical to GRPO's; only the advantage estimator inside
    `train_on_rollout` differs -- group mean-centering with no std normalization --
    which DrGRPOAlgorithm itself already owns). Kept as its own class and its own
    CLI command (`cli/dr_grpo.py`) rather than `GRPOTrainer(algorithm=...)`, per #124.
    """

    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module | None = None,
        config: DrGRPOConfig | None = None,
        optimizer: Optimizer | None = None,
        **kwargs,
    ):
        algorithm = DrGRPOAlgorithm(
            policy_model=model,
            ref_model=ref_model,
            optimizer=optimizer,
            config=config or DrGRPOConfig(),
        )
        super().__init__(algorithm=algorithm, **kwargs)
