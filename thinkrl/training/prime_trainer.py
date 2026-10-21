"""
ThinkRL PRIME Trainer
=======================

Dedicated trainer for Process Reinforcement through Implicit Rewards (PRIME).

Author: EllanorAI
"""

from __future__ import annotations

import torch.nn as nn
from torch.optim import Optimizer

from thinkrl.algorithms.prime import PRIMEAlgorithm, PRIMEConfig
from thinkrl.training.grpo_trainer import GRPOTrainer


class PRIMETrainer(GRPOTrainer):
    """
    Trainer for Process Reinforcement through Implicit Rewards (PRIME).

    Drives PRIMEAlgorithm through the same prompt -> rollout -> reward ->
    train_on_rollout loop GRPOTrainer already implements (PRIME's rollout mechanics
    are identical to GRPO's -- K completions per prompt; only the update rule inside
    `train_on_rollout` differs, which PRIMEAlgorithm itself already owns: an online
    Implicit PRM update plus an RLOO-baselined PPO policy step). Kept as its own
    class and its own CLI command (`cli/prime.py`) rather than
    `GRPOTrainer(algorithm=...)`, per #124.
    """

    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module | None = None,
        prm_model: nn.Module | None = None,
        config: PRIMEConfig | None = None,
        optimizer: Optimizer | None = None,
        learning_rate: float | None = None,
        **kwargs,
    ):
        # PRIMEConfig has no learning_rate field of its own (only prm_learning_rate,
        # for the Implicit PRM); the policy's own learning_rate is BaseRLHFAlgorithm's,
        # so it has to be forwarded as a kwarg rather than read off config.
        algo_kwargs = {} if learning_rate is None else {"learning_rate": learning_rate}
        algorithm = PRIMEAlgorithm(
            policy_model=model,
            ref_model=ref_model,
            prm_model=prm_model,
            optimizer=optimizer,
            config=config or PRIMEConfig(),
            **algo_kwargs,
        )
        super().__init__(algorithm=algorithm, **kwargs)
