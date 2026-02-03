"""
Tests for W&B Logger
====================

Tests for Weights & Biases integration.
"""

from unittest.mock import MagicMock, patch

import pytest

from thinkrl.logging.wandb import WANDB_AVAILABLE, WandBLogger


class TestWandBAvailability:
    """Tests for W&B availability."""

    def test_availability_flag(self):
        """Test WANDB_AVAILABLE is a boolean."""
        assert isinstance(WANDB_AVAILABLE, bool)


class TestWandBLoggerWithoutWandB:
    """Tests when W&B is not available."""

    def test_raises_import_error(self):
        """Test raises ImportError when wandb not available."""
        if WANDB_AVAILABLE:
            pytest.skip("wandb is available")

        with pytest.raises(ImportError):
            WandBLogger()


@pytest.mark.skipif(not WANDB_AVAILABLE, reason="wandb not installed")
class TestWandBLoggerMocked:
    """Tests for WandBLogger with mocked wandb."""

    @patch("thinkrl.logging.wandb.wandb")
    def test_initialization(self, mock_wandb):
        """Test WandBLogger initialization."""
        mock_run = MagicMock()
        mock_run.name = "test-run"
        mock_wandb.init.return_value = mock_run

        logger = WandBLogger(project="test-project")

        mock_wandb.init.assert_called_once()
        assert logger.run is not None

    @patch("thinkrl.logging.wandb.wandb")
    def test_log_calls_wandb(self, mock_wandb):
        """Test log method calls wandb.log."""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run

        logger = WandBLogger(project="test")
        logger.log({"loss": 0.5}, step=1)

        mock_wandb.log.assert_called_once_with({"loss": 0.5}, step=1)

    @patch("thinkrl.logging.wandb.wandb")
    def test_log_hyperparams(self, mock_wandb):
        """Test log_hyperparams updates wandb config."""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run

        logger = WandBLogger(project="test")
        logger.log_hyperparams({"lr": 1e-4})

        mock_wandb.config.update.assert_called_once()

    @patch("thinkrl.logging.wandb.wandb")
    def test_finish_calls_wandb_finish(self, mock_wandb):
        """Test finish calls wandb.finish."""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run

        logger = WandBLogger(project="test")
        logger.finish()

        mock_wandb.finish.assert_called_once()

    @patch("thinkrl.logging.wandb.wandb")
    def test_log_text(self, mock_wandb):
        """Test log_text creates table."""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        mock_wandb.Table = MagicMock()

        logger = WandBLogger(project="test")
        logger.log_text("sample", "test text", step=1)

        mock_wandb.Table.assert_called()
        mock_wandb.log.assert_called()

    @patch("thinkrl.logging.wandb.wandb")
    def test_log_table(self, mock_wandb):
        """Test log_table creates wandb table."""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        mock_wandb.Table = MagicMock()

        logger = WandBLogger(project="test")
        logger.log_table(
            "my_table",
            columns=["a", "b"],
            data=[[1, 2], [3, 4]],
            step=1,
        )

        mock_wandb.Table.assert_called_with(
            columns=["a", "b"],
            data=[[1, 2], [3, 4]],
        )

    @patch("thinkrl.logging.wandb.wandb")
    def test_init_with_tags(self, mock_wandb):
        """Test initialization with tags."""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run

        WandBLogger(
            project="test",
            tags=["experiment", "v1"],
        )

        call_kwargs = mock_wandb.init.call_args[1]
        assert call_kwargs["tags"] == ["experiment", "v1"]

    @patch("thinkrl.logging.wandb.wandb")
    def test_init_with_config(self, mock_wandb):
        """Test initialization with config."""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run

        config = {"lr": 1e-4, "epochs": 10}
        WandBLogger(project="test", config=config)

        call_kwargs = mock_wandb.init.call_args[1]
        assert call_kwargs["config"] == config

    @patch("thinkrl.logging.wandb.wandb")
    def test_log_without_run(self, mock_wandb):
        """Test log does nothing when run initialization failed."""
        mock_wandb.init.side_effect = Exception("Failed to init")

        logger = WandBLogger(project="test")
        # Should not raise
        logger.log({"loss": 0.5}, step=1)

    @patch("thinkrl.logging.wandb.wandb")
    def test_offline_mode(self, mock_wandb):
        """Test offline mode initialization."""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run

        WandBLogger(project="test", mode="offline")

        call_kwargs = mock_wandb.init.call_args[1]
        assert call_kwargs["mode"] == "offline"

    @patch("thinkrl.logging.wandb.wandb")
    def test_log_artifact(self, mock_wandb):
        """Test log_artifact method."""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        mock_artifact = MagicMock()
        mock_wandb.Artifact.return_value = mock_artifact

        logger = WandBLogger(project="test")
        logger.log_artifact(
            name="my_model",
            artifact_type="model",
            path="/path/to/model.pt",
            metadata={"accuracy": 0.95},
        )

        mock_wandb.Artifact.assert_called_once_with(
            name="my_model",
            type="model",
            metadata={"accuracy": 0.95},
        )
        mock_artifact.add_file.assert_called_once_with("/path/to/model.pt")
        mock_run.log_artifact.assert_called_once_with(mock_artifact)

    @patch("thinkrl.logging.wandb.wandb")
    def test_log_model(self, mock_wandb):
        """Test log_model method (wraps log_artifact)."""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        mock_artifact = MagicMock()
        mock_wandb.Artifact.return_value = mock_artifact

        logger = WandBLogger(project="test")
        logger.log_model(
            path="/path/to/model.pt",
            name="checkpoint",
            metadata={"epoch": 10},
        )

        mock_wandb.Artifact.assert_called_once_with(
            name="checkpoint",
            type="model",
            metadata={"epoch": 10},
        )

    @patch("thinkrl.logging.wandb.wandb")
    def test_run_property(self, mock_wandb):
        """Test run property returns the run object."""
        mock_run = MagicMock()
        mock_run.name = "test-run"
        mock_wandb.init.return_value = mock_run

        logger = WandBLogger(project="test")
        assert logger.run is mock_run

    @patch("thinkrl.logging.wandb.wandb")
    def test_log_hyperparams_without_run(self, mock_wandb):
        """Test log_hyperparams does nothing when run failed."""
        mock_wandb.init.side_effect = Exception("Failed")

        logger = WandBLogger(project="test")
        # Should not raise
        logger.log_hyperparams({"lr": 1e-4})

    @patch("thinkrl.logging.wandb.wandb")
    def test_log_text_without_run(self, mock_wandb):
        """Test log_text does nothing when run failed."""
        mock_wandb.init.side_effect = Exception("Failed")

        logger = WandBLogger(project="test")
        # Should not raise
        logger.log_text("key", "value", step=1)

    @patch("thinkrl.logging.wandb.wandb")
    def test_log_table_without_run(self, mock_wandb):
        """Test log_table does nothing when run failed."""
        mock_wandb.init.side_effect = Exception("Failed")

        logger = WandBLogger(project="test")
        # Should not raise
        logger.log_table("key", ["a", "b"], [[1, 2]], step=1)

    @patch("thinkrl.logging.wandb.wandb")
    def test_log_artifact_without_run(self, mock_wandb):
        """Test log_artifact does nothing when run failed."""
        mock_wandb.init.side_effect = Exception("Failed")

        logger = WandBLogger(project="test")
        # Should not raise
        logger.log_artifact("name", "type", "/path")

    @patch("thinkrl.logging.wandb.wandb")
    def test_finish_without_run(self, mock_wandb):
        """Test finish does nothing when run failed."""
        mock_wandb.init.side_effect = Exception("Failed")

        logger = WandBLogger(project="test")
        # Should not raise
        logger.finish()

    @patch("thinkrl.logging.wandb.wandb")
    def test_init_with_all_params(self, mock_wandb):
        """Test initialization with all parameters."""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run

        WandBLogger(
            project="test-project",
            name="test-run",
            config={"lr": 0.01},
            tags=["tag1", "tag2"],
            entity="my-team",
            group="experiment-group",
            notes="Test notes",
            mode="online",
            resume="allow",
        )

        call_kwargs = mock_wandb.init.call_args[1]
        assert call_kwargs["project"] == "test-project"
        assert call_kwargs["name"] == "test-run"
        assert call_kwargs["entity"] == "my-team"
        assert call_kwargs["group"] == "experiment-group"
        assert call_kwargs["notes"] == "Test notes"
        assert call_kwargs["resume"] == "allow"

    @patch("thinkrl.logging.wandb.wandb")
    def test_finish_sets_run_to_none(self, mock_wandb):
        """Test finish sets _run to None."""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run

        logger = WandBLogger(project="test")
        assert logger._run is not None
        logger.finish()
        assert logger._run is None
