"""
TensorBoard Logger
==================

TensorBoard integration for experiment tracking.

Author: EllanorAI
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from thinkrl.logging.loggers import Logger


# Check TensorBoard availability
try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except ImportError:
    SummaryWriter = None  # type: ignore
    TENSORBOARD_AVAILABLE = False


logger = logging.getLogger(__name__)


class TensorBoardLogger(Logger):
    """
    TensorBoard logger for experiment tracking.

    Features:
    - Scalar metrics logging
    - Hyperparameter logging
    - Text logging
    """

    def __init__(
        self,
        log_dir: str = "./logs/tensorboard",
        comment: str = "",
        flush_secs: int = 120,
    ):
        """
        Initialize TensorBoard logger.

        Args:
            log_dir: Directory for TensorBoard logs
            comment: Comment to append to log directory
            flush_secs: How often to flush to disk
        """
        if not TENSORBOARD_AVAILABLE:
            raise ImportError(
                "TensorBoard is not installed. Install with: pip install tensorboard"
            )

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._writer = SummaryWriter(
            log_dir=str(self.log_dir),
            comment=comment,
            flush_secs=flush_secs,
        )

        logger.info(f"Initialized TensorBoard logger: {self.log_dir}")

    @property
    def writer(self) -> "SummaryWriter":
        """Get the SummaryWriter instance."""
        return self._writer

    def log(self, metrics: dict[str, float | int], step: int) -> None:
        for key, value in metrics.items():
            self._writer.add_scalar(key, value, step)

    def log_hyperparams(self, params: dict[str, Any]) -> None:
        # TensorBoard uses hparams for hyperparameter logging
        # Filter out non-scalar values
        scalar_params = {}
        for key, value in params.items():
            if isinstance(value, (int, float, bool, str)):
                scalar_params[key] = value
            elif value is None:
                scalar_params[key] = "None"
            else:
                scalar_params[key] = str(value)

        self._writer.add_hparams(scalar_params, {})

    def log_text(self, key: str, text: str, step: int | None = None) -> None:
        self._writer.add_text(key, text, global_step=step or 0)

    def log_histogram(
        self,
        key: str,
        values,
        step: int,
        bins: str = "tensorflow",
    ) -> None:
        """Log a histogram of values."""
        self._writer.add_histogram(key, values, step, bins=bins)

    def log_image(
        self,
        key: str,
        image,
        step: int,
        dataformats: str = "CHW",
    ) -> None:
        """Log an image."""
        self._writer.add_image(key, image, step, dataformats=dataformats)

    def finish(self) -> None:
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()

    def __del__(self):
        self.finish()


__all__ = ["TensorBoardLogger", "TENSORBOARD_AVAILABLE"]
