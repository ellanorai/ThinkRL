"""
Weights & Biases Logger
=======================

W&B integration for experiment tracking.

Author: EllanorAI
"""

from __future__ import annotations

import logging
from typing import Any

from thinkrl.logging.loggers import Logger


# Check W&B availability
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    wandb = None  # type: ignore
    WANDB_AVAILABLE = False


logger = logging.getLogger(__name__)


class WandBLogger(Logger):
    """
    Weights & Biases logger for experiment tracking.

    Features:
    - Automatic hyperparameter logging
    - Real-time metrics visualization
    - Artifact tracking
    - Table logging
    """

    def __init__(
        self,
        project: str = "thinkrl",
        name: str | None = None,
        config: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        entity: str | None = None,
        group: str | None = None,
        notes: str | None = None,
        mode: str = "online",  # "online", "offline", "disabled"
        resume: str | None = None,
        **kwargs,
    ):
        """
        Initialize W&B logger.

        Args:
            project: W&B project name
            name: Run name
            config: Configuration dictionary
            tags: Tags for the run
            entity: W&B entity (team/user)
            group: Run group
            notes: Run notes
            mode: Logging mode
            resume: Resume from run ID
            **kwargs: Additional wandb.init kwargs
        """
        # Initialize _run before potential ImportError so __del__ works
        self._run = None

        if not WANDB_AVAILABLE:
            raise ImportError("wandb is not installed. Install with: pip install wandb")

        self.project = project

        try:
            self._run = wandb.init(
                project=project,
                name=name,
                config=config,
                tags=tags,
                entity=entity,
                group=group,
                notes=notes,
                mode=mode,
                resume=resume,
                **kwargs,
            )
            logger.info(f"Initialized W&B run: {self._run.name}")
        except Exception as e:
            logger.error(f"Failed to initialize W&B: {e}")
            self._run = None

    @property
    def run(self):
        """Get the W&B run object."""
        return self._run

    def log(self, metrics: dict[str, float | int], step: int) -> None:
        if self._run is None:
            return

        wandb.log(metrics, step=step)

    def log_hyperparams(self, params: dict[str, Any]) -> None:
        if self._run is None:
            return

        wandb.config.update(params, allow_val_change=True)

    def log_text(self, key: str, text: str, step: int | None = None) -> None:
        if self._run is None:
            return

        table = wandb.Table(columns=["text"], data=[[text]])
        wandb.log({key: table}, step=step)

    def log_table(
        self,
        key: str,
        columns: list[str],
        data: list[list[Any]],
        step: int | None = None,
    ) -> None:
        if self._run is None:
            return

        table = wandb.Table(columns=columns, data=data)
        wandb.log({key: table}, step=step)

    def log_artifact(
        self,
        name: str,
        artifact_type: str,
        path: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log an artifact (model, dataset, etc.)."""
        if self._run is None:
            return

        artifact = wandb.Artifact(name=name, type=artifact_type, metadata=metadata)
        artifact.add_file(path)
        self._run.log_artifact(artifact)

    def log_model(
        self,
        path: str,
        name: str = "model",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log a model checkpoint as artifact."""
        self.log_artifact(name, "model", path, metadata)

    def finish(self) -> None:
        if self._run is not None:
            wandb.finish()
            self._run = None

    def __del__(self):
        self.finish()


__all__ = ["WandBLogger", "WANDB_AVAILABLE"]
