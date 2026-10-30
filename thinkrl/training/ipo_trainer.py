"""
ThinkRL IPO Trainer
===================

Dedicated trainer for IPO (Identity Preference Optimization).

Author: EllanorAI
"""

from __future__ import annotations

import torch.nn as nn
from torch.optim import Optimizer

from thinkrl.algorithms.ipo import IPOAlgorithm, IPOConfig
from thinkrl.training.dpo_trainer import DPOTrainer


class IPOTrainer(DPOTrainer):
    """
    Trainer for IPO.

    Drives IPOAlgorithm through the same chosen/rejected -> single concatenated
    forward -> training_step loop DPOTrainer already implements (IPO's batch
    shape is identical to DPO's; only the loss differs -- a squared deviation
    from 1/(2*tau) instead of a log-sigmoid -- which IPOAlgorithm itself already
    owns). Kept as its own class and its own CLI command (`cli/ipo.py`) rather
    than `DPOTrainer(algorithm=...)`, per #124.

    A reference model is not optional here. IPO's loss is a function of the
    policy/reference log-ratio, so IPOAlgorithm refuses to construct without one.
    """

    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module | None = None,
        config: IPOConfig | None = None,
        optimizer: Optimizer | None = None,
        **kwargs,
    ):
        algorithm = IPOAlgorithm(
            policy_model=model,
            ref_model=ref_model,  # type: ignore[arg-type]  # IPOAlgorithm raises on None itself
            optimizer=optimizer,
            config=config or IPOConfig(),
        )
        super().__init__(algorithm=algorithm, **kwargs)


__all__ = ["IPOTrainer"]
