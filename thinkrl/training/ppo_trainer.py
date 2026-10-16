"""
ThinkRL PPO Trainer
====================

Dedicated trainer for Proximal Policy Optimization (PPO).

Author: EllanorAI
"""

from __future__ import annotations

import torch.nn as nn
from torch.optim import Optimizer

from thinkrl.algorithms.ppo import PPOAlgorithm, PPOConfig
from thinkrl.training.grpo_trainer import GRPOTrainer


class PPOTrainer(GRPOTrainer):
    """
    Trainer for Proximal Policy Optimization (PPO).

    Drives PPOAlgorithm through the same prompt -> rollout -> reward ->
    train_on_rollout loop GRPOTrainer already implements (PPO's rollout mechanics
    are identical; only the update rule inside `train_on_rollout` differs, which
    PPOAlgorithm itself already owns). Kept as its own class and its own CLI command
    (`cli/ppo.py`) rather than `GRPOTrainer(algorithm=...)`, per #124.
    """

    def __init__(
        self,
        model: nn.Module,
        value_model: nn.Module | None = None,
        ref_model: nn.Module | None = None,
        config: PPOConfig | None = None,
        optimizer: Optimizer | None = None,
        value_optimizer: Optimizer | None = None,
        **kwargs,
    ):
        algorithm = PPOAlgorithm(
            policy_model=model,
            value_model=value_model,
            ref_model=ref_model,
            optimizer=optimizer,
            value_optimizer=value_optimizer,
            config=config or PPOConfig(),
        )
        super().__init__(algorithm=algorithm, **kwargs)
