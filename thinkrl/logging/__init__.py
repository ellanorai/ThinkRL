"""
ThinkRL Logging System
======================

Unified logging with W&B and TensorBoard support.

Provides:
- Abstract logger interface
- Console, W&B, TensorBoard backends
- Composite logger for multiple backends
- Distributed-safe logging

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
from thinkrl.logging.tensorboard import TensorBoardLogger
from thinkrl.logging.wandb import WandBLogger


__all__ = [
    # Base
    "Logger",
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
