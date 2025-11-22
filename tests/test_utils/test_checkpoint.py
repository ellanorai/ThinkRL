"""
Test Suite for ThinkRL Checkpoint Utilities
===========================================
"""

from pathlib import Path
import shutil
import tempfile
from unittest.mock import patch

import pytest
import torch.nn as nn
import torch.optim as optim

from thinkrl.utils.checkpoint import (
    CheckpointManager,
    load_checkpoint,
    load_config,
    save_checkpoint,
    save_config,
)


# Fixtures
@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    d = tempfile.mkdtemp()
    yield Path(d)
    try:
        shutil.rmtree(d)
    except Exception:
        pass


@pytest.fixture
def simple_model():
    """Create a simple model."""
    return nn.Linear(10, 1)


@pytest.fixture
def simple_optimizer(simple_model):
    """Create a simple optimizer."""
    return optim.SGD(simple_model.parameters(), lr=0.01)


# Tests


@pytest.mark.slow
class TestCheckpointManager:
    """Tests for CheckpointManager."""

    def test_init(self, temp_dir):
        """Test initialization."""
        manager = CheckpointManager(checkpoint_dir=temp_dir, max_checkpoints=2)
        assert manager.checkpoint_dir == temp_dir
        assert manager.max_checkpoints == 2

    def test_save_and_cleanup(self, temp_dir, simple_model, simple_optimizer):
        """Test saving checkpoints and automatic cleanup."""
        manager = CheckpointManager(
            checkpoint_dir=temp_dir, max_checkpoints=2, metric_name="loss", mode="min"
        )

        # Save 3 checkpoints
        for i, loss in enumerate([0.5, 0.3, 0.4]):
            manager.save_checkpoint(
                model=simple_model,
                optimizer=simple_optimizer,
                epoch=i,
                step=i * 10,
                metrics={"loss": loss},
                checkpoint_name=f"ckpt_{i}",
            )

        checkpoints = list(temp_dir.glob("ckpt_*"))
        assert len(checkpoints) == 2

        ckpt_names = [p.name for p in checkpoints]
        assert "ckpt_1" in ckpt_names
        assert "ckpt_2" in ckpt_names
        assert "ckpt_0" not in ckpt_names

        assert manager.best_checkpoint["name"] == "ckpt_1"
        assert manager.best_checkpoint["metrics"]["loss"] == 0.3

    def test_load_best(self, temp_dir, simple_model, simple_optimizer):
        """Test loading the best checkpoint."""
        manager = CheckpointManager(
            checkpoint_dir=temp_dir, metric_name="acc", mode="max"
        )

        manager.save_checkpoint(
            simple_model, epoch=1, metrics={"acc": 0.8}, checkpoint_name="ckpt_1"
        )
        manager.save_checkpoint(
            simple_model, epoch=2, metrics={"acc": 0.9}, checkpoint_name="ckpt_2"
        )  # Best
        manager.save_checkpoint(
            simple_model, epoch=3, metrics={"acc": 0.85}, checkpoint_name="ckpt_3"
        )

        nn.init.constant_(simple_model.weight, 0.0)

        metadata = manager.load_best_checkpoint(simple_model)
        assert metadata["metrics"]["acc"] == 0.9
        assert metadata["epoch"] == 2

    def test_save_options(self, temp_dir, simple_model, simple_optimizer):
        """Test saving with options disabled."""
        manager = CheckpointManager(
            checkpoint_dir=temp_dir, save_optimizer=False, save_scheduler=False
        )

        manager.save_checkpoint(
            simple_model, optimizer=simple_optimizer, checkpoint_name="ckpt_no_opt"
        )

        ckpt_path = temp_dir / "ckpt_no_opt"
        assert (ckpt_path / "model.pt").exists()
        assert not (ckpt_path / "optimizer.pt").exists()

    def test_safetensors_fallback(self, temp_dir, simple_model):
        """Test fallback if safetensors is not available or used."""
        # Mock unavailability if installed
        with patch("thinkrl.utils.checkpoint._SAFETENSORS_AVAILABLE", False):
            manager = CheckpointManager(temp_dir, use_safetensors=True)
            manager.save_checkpoint(simple_model, checkpoint_name="ckpt_fallback")

            # Should have saved as .pt
            assert (temp_dir / "ckpt_fallback" / "model.pt").exists()
            assert not (temp_dir / "ckpt_fallback" / "model.safetensors").exists()


@pytest.mark.slow
class TestStandaloneFunctions:
    """Tests for standalone save/load functions."""

    def test_save_load_checkpoint(self, temp_dir, simple_model):
        """Test basic save and load."""
        path = temp_dir / "test.pt"

        save_checkpoint(path, simple_model, epoch=10, metrics={"loss": 0.1})

        assert path.exists()

        loaded_metadata = load_checkpoint(path, simple_model)
        assert loaded_metadata["epoch"] == 10
        assert loaded_metadata["metrics"]["loss"] == 0.1

    def test_save_load_config_json(self, temp_dir):
        """Test saving and loading config as JSON."""
        config = {"lr": 0.01, "batch_size": 32}
        path = temp_dir / "config.json"

        save_config(config, path)
        loaded = load_config(path)

        assert loaded == config

    def test_save_load_config_yaml(self, temp_dir):
        """Test saving and loading config as YAML."""
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML not installed")

        config = {"lr": 0.01, "model": "gpt2"}
        path = temp_dir / "config.yaml"

        save_config(config, path)
        loaded = load_config(path)

        assert loaded == config
