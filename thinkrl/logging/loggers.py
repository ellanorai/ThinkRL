"""
Logger Implementations
======================

Abstract logger interface and implementations.

Author: EllanorAI
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
import sys
from typing import Any

import torch.distributed as dist


logger = logging.getLogger(__name__)


class Logger(ABC):
    """Abstract base class for loggers."""

    @abstractmethod
    def log(self, metrics: dict[str, float | int], step: int) -> None:
        """
        Log metrics at given step.

        Args:
            metrics: Dictionary of metric name -> value
            step: Training step
        """
        pass

    @abstractmethod
    def log_hyperparams(self, params: dict[str, Any]) -> None:
        """
        Log hyperparameters.

        Args:
            params: Dictionary of hyperparameter name -> value
        """
        pass

    @abstractmethod
    def log_text(self, key: str, text: str, step: int | None = None) -> None:
        """Log text/string data."""
        pass

    @abstractmethod
    def log_table(
        self,
        key: str,
        columns: list[str],
        data: list[list[Any]],
        step: int | None = None,
    ) -> None:
        """Log tabular data."""
        pass

    @abstractmethod
    def finish(self) -> None:
        """Cleanup and finalize logging."""
        pass

    def __enter__(self) -> Logger:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.finish()


class NullLogger(Logger):
    """Logger that does nothing (for non-main processes)."""

    def log(self, metrics: dict[str, float | int], step: int) -> None:
        pass

    def log_hyperparams(self, params: dict[str, Any]) -> None:
        pass

    def finish(self) -> None:
        pass


class ConsoleLogger(Logger):
    """Logger that prints to console."""

    def __init__(
        self,
        prefix: str = "",
        log_every_n_steps: int = 1,
        stream=None,
    ):
        """
        Initialize console logger.

        Args:
            prefix: Prefix for log messages
            log_every_n_steps: Only log every N steps
            stream: Output stream (default: sys.stdout)
        """
        self.prefix = prefix
        self.log_every_n_steps = log_every_n_steps
        self.stream = stream or sys.stdout
        self._last_step = -1

    def log(self, metrics: dict[str, float | int], step: int) -> None:
        if step % self.log_every_n_steps != 0:
            return

        # Skip if same step
        if step == self._last_step:
            return
        self._last_step = step

        # Format metrics
        parts = [f"{self.prefix}Step {step}:"] if self.prefix else [f"Step {step}:"]

        for key, value in sorted(metrics.items()):
            if isinstance(value, float):
                parts.append(f"{key}={value:.4f}")
            else:
                parts.append(f"{key}={value}")

        message = " | ".join(parts)
        print(message, file=self.stream)
        self.stream.flush()

    def log_hyperparams(self, params: dict[str, Any]) -> None:
        print(f"{self.prefix}Hyperparameters:", file=self.stream)
        for key, value in sorted(params.items()):
            print(f"  {key}: {value}", file=self.stream)
        self.stream.flush()

    def finish(self) -> None:
        pass


class CompositeLogger(Logger):
    """Logger that combines multiple loggers."""

    def __init__(self, loggers: list[Logger]):
        """
        Initialize composite logger.

        Args:
            loggers: List of loggers to combine
        """
        self.loggers = loggers

    def log(self, metrics: dict[str, float | int], step: int) -> None:
        for logger_instance in self.loggers:
            logger_instance.log(metrics, step)

    def log_hyperparams(self, params: dict[str, Any]) -> None:
        for logger_instance in self.loggers:
            logger_instance.log_hyperparams(params)

    def log_text(self, key: str, text: str, step: int | None = None) -> None:
        for logger_instance in self.loggers:
            logger_instance.log_text(key, text, step)

    def log_table(
        self,
        key: str,
        columns: list[str],
        data: list[list[Any]],
        step: int | None = None,
    ) -> None:
        for logger_instance in self.loggers:
            logger_instance.log_table(key, columns, data, step)

    def finish(self) -> None:
        for logger_instance in self.loggers:
            logger_instance.finish()

    def add_logger(self, logger_instance: Logger) -> None:
        """Add a logger to the composite."""
        self.loggers.append(logger_instance)


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def log_only_main_process(
    logger_instance: Logger,
    metrics: dict[str, float | int],
    step: int,
) -> None:
    """
    Log only on main process (rank 0).

    Args:
        logger_instance: Logger to use
        metrics: Metrics to log
        step: Training step
    """
    if is_main_process():
        logger_instance.log(metrics, step)


def create_logger(
    backends: list[str],
    project: str = "thinkrl",
    name: str | None = None,
    config: dict[str, Any] | None = None,
    log_dir: str = "./logs",
    log_every_n_steps: int = 10,
    rank: int = 0,
) -> Logger:
    """
    Create logger with specified backends.

    Args:
        backends: List of backend names ("console", "wandb", "tensorboard")
        project: Project name (for W&B)
        name: Run name
        config: Configuration to log
        log_dir: Directory for logs (TensorBoard)
        log_every_n_steps: Logging frequency
        rank: Process rank (non-zero ranks get NullLogger)

    Returns:
        Logger instance

    Example:
        >>> logger = create_logger(["console", "wandb"], project="my-project")
        >>> logger.log({"loss": 0.5}, step=100)
    """
    # Non-main processes get null logger
    if rank != 0:
        return NullLogger()

    loggers: list[Logger] = []

    for backend in backends:
        backend = backend.lower()

        if backend == "console":
            loggers.append(ConsoleLogger(log_every_n_steps=log_every_n_steps))

        elif backend == "wandb":
            from thinkrl.logging.wandb import WandBLogger

            loggers.append(
                WandBLogger(
                    project=project,
                    name=name,
                    config=config,
                )
            )

        elif backend == "tensorboard":
            from thinkrl.logging.tensorboard import TensorBoardLogger

            loggers.append(
                TensorBoardLogger(
                    log_dir=log_dir,
                )
            )

        else:
            logger.warning(f"Unknown logging backend: {backend}")

    if not loggers:
        return NullLogger()

    if len(loggers) == 1:
        return loggers[0]

    return CompositeLogger(loggers)


__all__ = [
    "Logger",
    "NullLogger",
    "ConsoleLogger",
    "CompositeLogger",
    "is_main_process",
    "log_only_main_process",
    "create_logger",
]
