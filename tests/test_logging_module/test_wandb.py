"""
Tests for W&B Logger
====================

Tests for Weights & Biases integration.
"""

from unittest.mock import patch, MagicMock

import pytest

from thinkrl.logging.wandb import WandBLogger, WANDB_AVAILABLE


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
