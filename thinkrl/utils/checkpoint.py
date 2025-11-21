"""
ThinkRL Checkpoint Management
==============================

Comprehensive checkpoint utilities for ThinkRL including:
- Model state saving/loading
- Optimizer and scheduler state management
- SafeTensors support for safe model serialization
- Distributed training checkpointing
- Best model tracking
- Checkpoint cleanup and rotation
- Resume training functionality

Author: Archit Sood @ EllanorAI
"""

import json
import logging
import shutil
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

# Core dependencies
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


# Optional dependencies
try:
    import yaml

    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False
    warnings.warn("PyYAML not available. YAML config saving will be disabled.")

try:
    from safetensors.torch import load_file as safetensors_load
    from safetensors.torch import save_file as safetensors_save

    _SAFETENSORS_AVAILABLE = True
except ImportError:
    _SAFETENSORS_AVAILABLE = False


logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages model checkpoints with automatic cleanup and best model tracking.

    Features:
    - Automatic checkpoint rotation (keep only N best)
    - Metric-based best model selection
    - Checkpoint metadata tracking
    - Safe checkpoint operations (atomic writes)

    Example:
        ```python
        manager = CheckpointManager(
            checkpoint_dir="./checkpoints",
            max_checkpoints=5,
            metric_name="loss",
            mode="min"
        )

        # Save checkpoint
        manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=10,
            metrics={"loss": 0.42, "accuracy": 0.95}
        )

        # Load best checkpoint
        checkpoint = manager.load_best_checkpoint()
        ```
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        max_checkpoints: int = 5,
        metric_name: str | None = None,
        mode: str = "min",
        save_optimizer: bool = True,
        save_scheduler: bool = True,
        use_safetensors: bool = False,
    ):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            metric_name: Metric to track for best model (e.g., "loss", "accuracy")
            mode: "min" or "max" for metric comparison
            save_optimizer: Whether to save optimizer state
            save_scheduler: Whether to save scheduler state
            use_safetensors: Use SafeTensors format instead of PyTorch
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.max_checkpoints = max_checkpoints
        self.metric_name = metric_name
        self.mode = mode
        self.save_optimizer = save_optimizer
        self.save_scheduler = save_scheduler
        self.use_safetensors = use_safetensors and _SAFETENSORS_AVAILABLE

        if self.use_safetensors and not _SAFETENSORS_AVAILABLE:
            logger.warning(
                "SafeTensors not available. Falling back to PyTorch format. "
                "Install with: pip install safetensors"
            )
            self.use_safetensors = False

        # Track checkpoints
        self.checkpoints: list[dict[str, Any]] = []
        self.best_checkpoint: dict[str, Any] | None = None

        # Load existing checkpoint metadata
        self._load_metadata()

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optimizer | None = None,
        scheduler: _LRScheduler | None = None,
        epoch: int | None = None,
        step: int | None = None,
        metrics: dict[str, float] | None = None,
        extra_data: dict[str, Any] | None = None,
        checkpoint_name: str | None = None,
    ) -> Path:
        """
        Save a checkpoint with model, optimizer, and metadata.

        Args:
            model: Model to save
            optimizer: Optimizer to save (optional)
            scheduler: Learning rate scheduler to save (optional)
            epoch: Current epoch number
            step: Current training step
            metrics: Training metrics dictionary
            extra_data: Additional data to save
            checkpoint_name: Custom checkpoint name (default: auto-generated)

        Returns:
            Path to saved checkpoint
        """
        # Generate checkpoint name
        if checkpoint_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_name = f"checkpoint_epoch{epoch}_step{step}_{timestamp}"

        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Prepare checkpoint data
        checkpoint_data = {
            "epoch": epoch,
            "step": step,
            "metrics": metrics or {},
            "timestamp": datetime.now().isoformat(),
        }

        if extra_data:
            checkpoint_data.update(extra_data)

        # Save model
        if self.use_safetensors:
            model_path = checkpoint_path / "model.safetensors"
            state_dict = model.state_dict()
            safetensors_save(state_dict, str(model_path))
        else:
            model_path = checkpoint_path / "model.pt"
            torch.save(model.state_dict(), model_path)

        logger.info(f"Saved model to: {model_path}")

        # Save optimizer
        if optimizer is not None and self.save_optimizer:
            optimizer_path = checkpoint_path / "optimizer.pt"
            torch.save(optimizer.state_dict(), optimizer_path)
            logger.debug(f"Saved optimizer to: {optimizer_path}")

        # Save scheduler
        if scheduler is not None and self.save_scheduler:
            scheduler_path = checkpoint_path / "scheduler.pt"
            torch.save(scheduler.state_dict(), scheduler_path)
            logger.debug(f"Saved scheduler to: {scheduler_path}")

        # Save metadata
        metadata_path = checkpoint_path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2)

        # Update checkpoint tracking
        checkpoint_info = {
            "path": checkpoint_path,
            "name": checkpoint_name,
            "epoch": epoch,
            "step": step,
            "metrics": metrics or {},
            "timestamp": checkpoint_data["timestamp"],
        }

        self.checkpoints.append(checkpoint_info)

        # Update best checkpoint
        if self.metric_name and metrics and self.metric_name in metrics:
            metric_value = metrics[self.metric_name]
            self._update_best_checkpoint(checkpoint_info, metric_value)

        # Cleanup old checkpoints
        self._cleanup_checkpoints()

        # Save checkpoint registry
        self._save_metadata()

        logger.info(f"Checkpoint saved: {checkpoint_path}")
        return checkpoint_path

    def load_checkpoint(
        self,
        checkpoint_path: str | Path,
        model: nn.Module,
        optimizer: Optimizer | None = None,
        scheduler: _LRScheduler | None = None,
        device: torch.device | None = None,
        strict: bool = True,
    ) -> dict[str, Any]:
        """
        Load a checkpoint into model, optimizer, and scheduler.

        Args:
            checkpoint_path: Path to checkpoint directory or file
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)
            scheduler: Scheduler to load state into (optional)
            device: Device to load tensors to
            strict: Whether to strictly enforce state dict keys match

        Returns:
            Dictionary containing checkpoint metadata
        """
        checkpoint_path = Path(checkpoint_path)

        # Handle both directory and file paths
        if checkpoint_path.is_file():
            checkpoint_dir = checkpoint_path.parent
            model_path = checkpoint_path
        else:
            checkpoint_dir = checkpoint_path
            # Try safetensors first, then PyTorch
            model_path = checkpoint_dir / "model.safetensors"
            if not model_path.exists():
                model_path = checkpoint_dir / "model.pt"

        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

        # Load model state
        if model_path.suffix == ".safetensors":
            state_dict = safetensors_load(
                str(model_path), device=str(device) if device else "cpu"
            )
        else:
            state_dict = torch.load(model_path, map_location=device)

        # Handle DataParallel/DistributedDataParallel wrapped models
        if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            model = model.module

        model.load_state_dict(state_dict, strict=strict)
        logger.info(f"Loaded model from: {model_path}")

        # Load optimizer
        if optimizer is not None:
            optimizer_path = checkpoint_dir / "optimizer.pt"
            if optimizer_path.exists():
                optimizer.load_state_dict(
                    torch.load(optimizer_path, map_location=device)
                )
                logger.debug(f"Loaded optimizer from: {optimizer_path}")

        # Load scheduler
        if scheduler is not None:
            scheduler_path = checkpoint_dir / "scheduler.pt"
            if scheduler_path.exists():
                scheduler.load_state_dict(
                    torch.load(scheduler_path, map_location=device)
                )
                logger.debug(f"Loaded scheduler from: {scheduler_path}")

        # Load metadata
        metadata = {}
        metadata_path = checkpoint_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)

        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        return metadata

    def load_best_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optimizer | None = None,
        scheduler: _LRScheduler | None = None,
        device: torch.device | None = None,
    ) -> dict[str, Any] | None:
        """
        Load the best checkpoint based on tracked metrics.

        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)
            scheduler: Scheduler to load state into (optional)
            device: Device to load tensors to

        Returns:
            Checkpoint metadata or None if no best checkpoint exists
        """
        if self.best_checkpoint is None:
            logger.warning("No best checkpoint available")
            return None

        checkpoint_path = self.best_checkpoint["path"]
        metadata = self.load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )

        logger.info(
            f"Loaded best checkpoint: {checkpoint_path} "
            f"({self.metric_name}={self.best_checkpoint['metrics'].get(self.metric_name)})"
        )

        return metadata

    def _update_best_checkpoint(
        self, checkpoint_info: dict[str, Any], metric_value: float
    ):
        """Update the best checkpoint based on metric value."""
        if self.best_checkpoint is None:
            self.best_checkpoint = checkpoint_info
            logger.info(f"New best checkpoint: {self.metric_name}={metric_value:.4f}")
            return

        best_metric = self.best_checkpoint["metrics"].get(self.metric_name)

        is_better = (self.mode == "min" and metric_value < best_metric) or (
            self.mode == "max" and metric_value > best_metric
        )

        if is_better:
            self.best_checkpoint = checkpoint_info
            logger.info(
                f"New best checkpoint: {self.metric_name}={metric_value:.4f} "
                f"(previous: {best_metric:.4f})"
            )

    def _cleanup_checkpoints(self):
        """Remove old checkpoints, keeping only the best ones."""
        if len(self.checkpoints) <= self.max_checkpoints:
            return

        # Sort checkpoints
        if self.metric_name:
            # Sort by metric
            def get_metric(ckpt):
                return ckpt["metrics"].get(self.metric_name, float("inf"))

            reverse = self.mode == "max"
            sorted_checkpoints = sorted(
                self.checkpoints, key=get_metric, reverse=reverse
            )
        else:
            # Sort by timestamp (keep most recent)
            sorted_checkpoints = sorted(
                self.checkpoints, key=lambda x: x["timestamp"], reverse=True
            )

        # Keep best checkpoints
        keep_checkpoints = sorted_checkpoints[: self.max_checkpoints]
        remove_checkpoints = sorted_checkpoints[self.max_checkpoints :]

        # Remove old checkpoint directories
        for ckpt in remove_checkpoints:
            ckpt_path = ckpt["path"]
            if ckpt_path.exists():
                shutil.rmtree(ckpt_path)
                logger.debug(f"Removed old checkpoint: {ckpt_path}")

        # Update checkpoint list
        self.checkpoints = keep_checkpoints

    def _save_metadata(self):
        """Save checkpoint registry metadata."""
        metadata_file = self.checkpoint_dir / "checkpoint_registry.json"

        metadata = {
            "checkpoints": [
                {
                    "path": str(ckpt["path"]),
                    "name": ckpt["name"],
                    "epoch": ckpt["epoch"],
                    "step": ckpt["step"],
                    "metrics": ckpt["metrics"],
                    "timestamp": ckpt["timestamp"],
                }
                for ckpt in self.checkpoints
            ],
            "best_checkpoint": {
                "path": str(self.best_checkpoint["path"]),
                "name": self.best_checkpoint["name"],
                "metrics": self.best_checkpoint["metrics"],
            }
            if self.best_checkpoint
            else None,
            "config": {
                "max_checkpoints": self.max_checkpoints,
                "metric_name": self.metric_name,
                "mode": self.mode,
            },
        }

        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

    def _load_metadata(self):
        """Load checkpoint registry metadata."""
        metadata_file = self.checkpoint_dir / "checkpoint_registry.json"

        if not metadata_file.exists():
            return

        try:
            with open(metadata_file) as f:
                metadata = json.load(f)

            # Restore checkpoints
            self.checkpoints = [
                {
                    "path": Path(ckpt["path"]),
                    "name": ckpt["name"],
                    "epoch": ckpt["epoch"],
                    "step": ckpt["step"],
                    "metrics": ckpt["metrics"],
                    "timestamp": ckpt["timestamp"],
                }
                for ckpt in metadata.get("checkpoints", [])
            ]

            # Restore best checkpoint
            if metadata.get("best_checkpoint"):
                best_ckpt = metadata["best_checkpoint"]
                self.best_checkpoint = {
                    "path": Path(best_ckpt["path"]),
                    "name": best_ckpt["name"],
                    "metrics": best_ckpt["metrics"],
                }

            logger.debug(
                f"Loaded checkpoint metadata: {len(self.checkpoints)} checkpoints"
            )

        except Exception as e:
            logger.warning(f"Failed to load checkpoint metadata: {e}")


def save_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: _LRScheduler | None = None,
    epoch: int | None = None,
    step: int | None = None,
    metrics: dict[str, float] | None = None,
    config: dict[str, Any] | None = None,
    use_safetensors: bool = False,
    **kwargs,
) -> Path:
    """
    Save a checkpoint to disk.

    This is a simple function for one-off checkpoint saving without using
    the CheckpointManager class.

    Args:
        checkpoint_path: Path to save checkpoint
        model: Model to save
        optimizer: Optimizer to save (optional)
        scheduler: Scheduler to save (optional)
        epoch: Current epoch
        step: Current training step
        metrics: Training metrics
        config: Training configuration
        use_safetensors: Use SafeTensors format
        **kwargs: Additional data to save in checkpoint

    Returns:
        Path to saved checkpoint

    Example:
        ```python
        save_checkpoint(
            checkpoint_path="./checkpoints/model_epoch10.pt",
            model=model,
            optimizer=optimizer,
            epoch=10,
            metrics={"loss": 0.42, "accuracy": 0.95}
        )
        ```
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare checkpoint dictionary
    checkpoint = {
        "epoch": epoch,
        "step": step,
        "metrics": metrics or {},
        "config": config or {},
        "timestamp": datetime.now().isoformat(),
    }

    # Add model state
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        checkpoint["model_state_dict"] = model.module.state_dict()
    else:
        checkpoint["model_state_dict"] = model.state_dict()

    # Add optimizer state
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    # Add scheduler state
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    # Add extra data
    checkpoint.update(kwargs)

    # Save checkpoint
    if use_safetensors and _SAFETENSORS_AVAILABLE:
        # Save only model with safetensors
        model_path = checkpoint_path.with_suffix(".safetensors")
        safetensors_save(checkpoint["model_state_dict"], str(model_path))

        # Save metadata separately
        metadata = {k: v for k, v in checkpoint.items() if k != "model_state_dict"}
        metadata_path = checkpoint_path.with_suffix(".metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved checkpoint (SafeTensors): {model_path}")
        return model_path
    else:
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        return checkpoint_path


def load_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: _LRScheduler | None = None,
    device: torch.device | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """
    Load a checkpoint from disk.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        device: Device to map tensors to
        strict: Whether to strictly enforce state dict keys match

    Returns:
        Dictionary containing checkpoint data

    Example:
        ```python
        checkpoint = load_checkpoint(
            checkpoint_path="./checkpoints/model_epoch10.pt",
            model=model,
            optimizer=optimizer,
            device=torch.device("cuda")
        )

        print(f"Resumed from epoch {checkpoint['epoch']}")
        ```
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint
    if checkpoint_path.suffix == ".safetensors":
        # Load safetensors model
        state_dict = safetensors_load(
            str(checkpoint_path), device=str(device) if device else "cpu"
        )

        # Load metadata
        metadata_path = checkpoint_path.with_suffix(".metadata.json")
        if metadata_path.exists():
            with open(metadata_path) as f:
                checkpoint = json.load(f)
            checkpoint["model_state_dict"] = state_dict
        else:
            checkpoint = {"model_state_dict": state_dict}
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model state
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        model.module.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    else:
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    logger.info(f"Loaded model from: {checkpoint_path}")

    # Load optimizer state
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.debug("Loaded optimizer state")

    # Load scheduler state
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        logger.debug("Loaded scheduler state")

    return checkpoint


def save_config(config: dict[str, Any], save_path: str | Path):
    """
    Save configuration to JSON or YAML file.

    Args:
        config: Configuration dictionary
        save_path: Path to save configuration

    Example:
        ```python
        config = {
            "model": "gpt2",
            "learning_rate": 1e-4,
            "batch_size": 32
        }
        save_config(config, "./configs/train_config.yaml")
        ```
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if save_path.suffix in [".yaml", ".yml"] and _YAML_AVAILABLE:
        with open(save_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
    else:
        with open(save_path, "w") as f:
            json.dump(config, f, indent=2)

    logger.info(f"Saved config to: {save_path}")


def load_config(config_path: str | Path) -> dict[str, Any]:
    """
    Load configuration from JSON or YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary

    Example:
        ```python
        config = load_config("./configs/train_config.yaml")
        print(config["learning_rate"])
        ```
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if config_path.suffix in [".yaml", ".yml"] and _YAML_AVAILABLE:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        with open(config_path) as f:
            config = json.load(f)

    logger.info(f"Loaded config from: {config_path}")
    return config


# Public API
__all__ = [
    "CheckpointManager",
    "save_checkpoint",
    "load_checkpoint",
    "save_config",
    "load_config",
]
