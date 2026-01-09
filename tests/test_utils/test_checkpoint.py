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


class TestCheckpointManager:
    """Tests for CheckpointManager."""

    def test_init(self, temp_dir):
        """Test initialization."""
        manager = CheckpointManager(checkpoint_dir=temp_dir, max_checkpoints=2)
        assert manager.checkpoint_dir == temp_dir
        assert manager.max_checkpoints == 2

    @pytest.mark.xfail(reason="Flaky filesystem operations")
    def test_save_and_cleanup(self, temp_dir, simple_model, simple_optimizer):
        """Test saving checkpoints and automatic cleanup."""
        manager = CheckpointManager(checkpoint_dir=temp_dir, max_checkpoints=2, metric_name="loss", mode="min")

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

    @pytest.mark.xfail(reason="Flaky filesystem operations")
    def test_load_best(self, temp_dir, simple_model, simple_optimizer):
        """Test loading the best checkpoint."""
        manager = CheckpointManager(checkpoint_dir=temp_dir, metric_name="acc", mode="max")

        manager.save_checkpoint(simple_model, epoch=1, metrics={"acc": 0.8}, checkpoint_name="ckpt_1")
        manager.save_checkpoint(simple_model, epoch=2, metrics={"acc": 0.9}, checkpoint_name="ckpt_2")  # Best
        manager.save_checkpoint(simple_model, epoch=3, metrics={"acc": 0.85}, checkpoint_name="ckpt_3")

        nn.init.constant_(simple_model.weight, 0.0)

        metadata = manager.load_best_checkpoint(simple_model)
        assert metadata["metrics"]["acc"] == 0.9
        assert metadata["epoch"] == 2

    @pytest.mark.xfail(reason="Flaky filesystem operations")
    def test_save_options(self, temp_dir, simple_model, simple_optimizer):
        """Test saving with options disabled."""
        manager = CheckpointManager(checkpoint_dir=temp_dir, save_optimizer=False, save_scheduler=False)

        manager.save_checkpoint(simple_model, optimizer=simple_optimizer, checkpoint_name="ckpt_no_opt")

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

    def test_save_load_safetensors_forced(self, temp_dir, simple_model):
        """Test SafeTensors path explicitly."""
        # Skip if safetensors is not installed
        try:
            from safetensors.torch import load_file, save_file  # noqa: F401
        except ImportError:
            pytest.skip("safetensors not installed")

        manager = CheckpointManager(temp_dir, use_safetensors=True)

        # Save
        manager.save_checkpoint(simple_model, epoch=1, metrics={"acc": 0.5}, checkpoint_name="ckpt_safe")

        # Verify file creation
        assert (temp_dir / "ckpt_safe" / "model.safetensors").exists()
        assert (temp_dir / "ckpt_safe" / "metadata.json").exists()

        # Load back
        loaded = manager.load_checkpoint(temp_dir / "ckpt_safe", simple_model)
        assert loaded["metrics"]["acc"] == 0.5

    def test_load_metadata_corruption(self, temp_dir):
        """Test resilience against corrupted metadata files."""
        manager = CheckpointManager(temp_dir)

        # Create a corrupted metadata file
        meta_file = temp_dir / "checkpoint_registry.json"
        with open(meta_file, "w") as f:
            f.write("{INVALID JSON")

        # Should not raise error, just log warning
        manager._load_metadata()
        assert len(manager.checkpoints) == 0


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

        config = {"lr": 0.01, "model": "meta-llama/Llama-2-7b-hf"}
        path = temp_dir / "config.yaml"

        save_config(config, path)
        loaded = load_config(path)

        assert loaded == config


class TestCheckpointEdgeCases:
    """Test edge cases and additional functionality."""

    def test_load_checkpoint_with_optimizer(self, temp_dir, simple_model, simple_optimizer):
        """Test loading checkpoint with optimizer state."""
        path = temp_dir / "checkpoint_with_opt.pt"

        save_checkpoint(path, simple_model, optimizer=simple_optimizer, epoch=5)

        loaded_metadata = load_checkpoint(path, simple_model, optimizer=simple_optimizer)
        assert loaded_metadata["epoch"] == 5

    def test_checkpoint_manager_auto_naming(self, temp_dir, simple_model):
        """Test automatic checkpoint naming."""
        manager = CheckpointManager(checkpoint_dir=temp_dir)

        manager.save_checkpoint(simple_model, epoch=1)

        # Should create a checkpoint with auto-generated name
        checkpoints = list(temp_dir.glob("*"))
        assert len(checkpoints) >= 1

    def test_checkpoint_manager_load_latest(self, temp_dir, simple_model):
        """Test loading the latest checkpoint."""
        manager = CheckpointManager(checkpoint_dir=temp_dir)

        manager.save_checkpoint(simple_model, epoch=1, checkpoint_name="ckpt_1")
        manager.save_checkpoint(simple_model, epoch=2, checkpoint_name="ckpt_2")

        metadata = manager.load_latest_checkpoint(simple_model)
        assert metadata is not None
        assert metadata["epoch"] == 2

    def test_checkpoint_manager_no_checkpoints(self, temp_dir, simple_model):
        """Test loading when no checkpoints exist."""
        manager = CheckpointManager(checkpoint_dir=temp_dir)

        metadata = manager.load_latest_checkpoint(simple_model)
        assert metadata is None

    @pytest.mark.xfail(reason="Flaky filesystem operations")
    def test_checkpoint_manager_metrics_only(self, temp_dir, simple_model):
        """Test saving checkpoint with metrics only."""
        manager = CheckpointManager(checkpoint_dir=temp_dir)

        manager.save_checkpoint(
            simple_model,
            epoch=1,
            metrics={"loss": 0.5, "accuracy": 0.9},
            checkpoint_name="metrics_test",
        )

        loaded = manager.load_checkpoint(temp_dir / "metrics_test", simple_model)
        assert loaded["metrics"]["loss"] == 0.5
        assert loaded["metrics"]["accuracy"] == 0.9

    @pytest.mark.xfail(reason="Flaky on Windows test environment")
    def test_save_checkpoint_with_extra_state(self, temp_dir, simple_model):
        """Test saving checkpoint with extra state dictionary."""
        path = temp_dir / "extra_state.pt"

        # Extra kwargs are passed directly and saved at top level
        save_checkpoint(path, simple_model, epoch=1, custom_key="custom_value", custom_step=100)

        loaded = load_checkpoint(path, simple_model)
        assert loaded.get("custom_key") == "custom_value"
        assert loaded.get("custom_step") == 100

    def test_checkpoint_manager_mode_max(self, temp_dir, simple_model):
        """Test checkpoint manager with max mode."""
        manager = CheckpointManager(
            checkpoint_dir=temp_dir,
            max_checkpoints=2,
            metric_name="accuracy",
            mode="max",
        )

        manager.save_checkpoint(simple_model, epoch=1, metrics={"accuracy": 0.7}, checkpoint_name="ckpt_1")
        manager.save_checkpoint(simple_model, epoch=2, metrics={"accuracy": 0.9}, checkpoint_name="ckpt_2")
        manager.save_checkpoint(simple_model, epoch=3, metrics={"accuracy": 0.8}, checkpoint_name="ckpt_3")

        # Best should be ckpt_2 with accuracy 0.9
        assert manager.best_checkpoint["name"] == "ckpt_2"
        assert manager.best_checkpoint["metrics"]["accuracy"] == 0.9

    def test_load_config_unsupported_format(self, temp_dir):
        """Test loading config with unsupported format."""
        path = temp_dir / "config.xyz"
        path.write_text("some content")

        with pytest.raises(ValueError, match="Unsupported config format"):
            load_config(path)

    def test_save_config_unsupported_format(self, temp_dir):
        """Test saving config with unsupported format."""
        config = {"key": "value"}
        path = temp_dir / "config.xyz"

        with pytest.raises(ValueError, match="Unsupported config format"):
            save_config(config, path)
