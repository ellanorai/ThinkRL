"""
ThinkRL VAPO Trainer
=====================

Dedicated trainer for Value-model-based Augmented Proximal Policy Optimization (VAPO).

Author: EllanorAI
"""

from __future__ import annotations

import torch.nn as nn
from torch.optim import Optimizer

from thinkrl.algorithms.vapo import VAPOAlgorithm, VAPOConfig
from thinkrl.training.grpo_trainer import GRPOTrainer


class VAPOTrainer(GRPOTrainer):
    """
    Trainer for Value-model-based Augmented Proximal Policy Optimization (VAPO).

    Drives VAPOAlgorithm through the same prompt -> rollout -> reward ->
    train_on_rollout loop GRPOTrainer already implements (VAPO's rollout mechanics
    are identical to PPO's; only the update rule inside `train_on_rollout` differs,
    which VAPOAlgorithm itself already owns -- decoupled GAE, length-adaptive lambda,
    clip-higher). Kept as its own class and its own CLI command (`cli/vapo.py`)
    rather than `GRPOTrainer(algorithm=...)`, per #124.
    """

    def __init__(
        self,
        model: nn.Module,
        value_model: nn.Module | None = None,
        ref_model: nn.Module | None = None,
        config: VAPOConfig | None = None,
        optimizer: Optimizer | None = None,
        value_optimizer: Optimizer | None = None,
        **kwargs,
    ):
        algorithm = VAPOAlgorithm(
            policy_model=model,
            value_model=value_model,
            ref_model=ref_model,
            optimizer=optimizer,
            value_optimizer=value_optimizer,
            config=config or VAPOConfig(),
        )
        super().__init__(algorithm=algorithm, **kwargs)
