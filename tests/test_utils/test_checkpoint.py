"""
Test Suite for ThinkRL Checkpoint Utilities
===========================================

Tests for:
- CheckpointManager
- save_checkpoint
- load_checkpoint
- save_config
- load_config

Author: Archit Sood
"""

import pytest
import torch.nn as nn
import torch.optim as optim
import tempfile
import shutil
from pathlib import Path

from thinkrl.utils.checkpoint import (
    CheckpointManager,
    save_checkpoint,
    load_checkpoint,
    save_config,
    load_config,
)

# Fixtures
@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d)

@pytest.fixture
def simple_model():
    """Create a simple model."""
    return nn.Linear(10, 1)

@pytest.fixture
def simple_optimizer(simple_model):
    """Create a simple optimizer."""
    return optim.SGD(simple_model.parameters(), lr=0.01)

# Tests

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
            checkpoint_dir=temp_dir, 
            max_checkpoints=2, 
            metric_name="loss", 
            mode="min"
        )

        # Save 3 checkpoints
        for i, loss in enumerate([0.5, 0.3, 0.4]):
            manager.save_checkpoint(
                model=simple_model,
                optimizer=simple_optimizer,
                epoch=i,
                step=i*10,
                metrics={"loss": loss},
                checkpoint_name=f"ckpt_{i}"
            )

        # Check cleanup: Should keep 2 best (lowest loss: 0.3 and 0.4)
        # ckpt_1 (loss 0.3) and ckpt_2 (loss 0.4) should exist
        # ckpt_0 (loss 0.5) should be deleted
        
        checkpoints = list(temp_dir.glob("ckpt_*"))
        assert len(checkpoints) == 2
        
        ckpt_names = [p.name for p in checkpoints]
        assert "ckpt_1" in ckpt_names
        assert "ckpt_2" in ckpt_names
        assert "ckpt_0" not in ckpt_names

        # Check metadata
        assert manager.best_checkpoint["name"] == "ckpt_1"
        assert manager.best_checkpoint["metrics"]["loss"] == 0.3

    def test_load_best(self, temp_dir, simple_model, simple_optimizer):
        """Test loading the best checkpoint."""
        manager = CheckpointManager(checkpoint_dir=temp_dir, metric_name="acc", mode="max")

        # Save checkpoints
        manager.save_checkpoint(simple_model, epoch=1, metrics={"acc": 0.8}, checkpoint_name="ckpt_1")
        manager.save_checkpoint(simple_model, epoch=2, metrics={"acc": 0.9}, checkpoint_name="ckpt_2") # Best
        manager.save_checkpoint(simple_model, epoch=3, metrics={"acc": 0.85}, checkpoint_name="ckpt_3")

        # Reset model weights
        nn.init.constant_(simple_model.weight, 0.0)
        
        # Load best
        metadata = manager.load_best_checkpoint(simple_model)
        assert metadata["metrics"]["acc"] == 0.9
        assert metadata["epoch"] == 2

class TestStandaloneFunctions:
    """Tests for standalone save/load functions."""

    def test_save_load_checkpoint(self, temp_dir, simple_model):
        """Test basic save and load."""
        path = temp_dir / "test.pt"
        
        # Save
        save_checkpoint(
            path,
            simple_model,
            epoch=10,
            metrics={"loss": 0.1}
        )
        
        assert path.exists()
        
        # Load
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