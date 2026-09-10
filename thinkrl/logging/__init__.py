"""
ThinkRL Logging System
======================

Unified logging with W&B and TensorBoard support.

Provides:
- Abstract logger interface
- Console, W&B, TensorBoard backends
- Composite logger for multiple backends
- Distributed-safe logging
- Periodic rollout sampling, so a run shows what the policy is generating

Author: EllanorAI
"""

from thinkrl.logging.loggers import (
    CompositeLogger,
    ConsoleLogger,
    Logger,
    NullLogger,
    create_logger,
    log_only_main_process,
)
from thinkrl.logging.rollout import RolloutInspector
from thinkrl.logging.tensorboard import TensorBoardLogger
from thinkrl.logging.wandb import WandBLogger


__all__ = [
    # Base
    "Logger",
    "RolloutInspector",
    "NullLogger",
    "ConsoleLogger",
    "CompositeLogger",
    # Backends
    "WandBLogger",
    "TensorBoardLogger",
    # Factory
    "create_logger",
    "log_only_main_process",
]
