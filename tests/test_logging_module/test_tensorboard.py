"""
Tests for TensorBoard Logger
============================

Tests for TensorBoard integration.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from thinkrl.logging.tensorboard import TensorBoardLogger, TENSORBOARD_AVAILABLE


class TestTensorBoardAvailability:
    """Tests for TensorBoard availability."""

    def test_availability_flag(self):
        """Test TENSORBOARD_AVAILABLE is a boolean."""
        assert isinstance(TENSORBOARD_AVAILABLE, bool)


@pytest.mark.skipif(not TENSORBOARD_AVAILABLE, reason="TensorBoard not installed")
class TestTensorBoardLogger:
    """Tests for TensorBoardLogger when TensorBoard is available."""

    def test_initialization(self):
        """Test TensorBoardLogger initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TensorBoardLogger(log_dir=tmpdir)

            assert logger.log_dir == Path(tmpdir)
            assert logger.writer is not None

            logger.finish()

    def test_log_creates_events(self):
        """Test log method creates TensorBoard events."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TensorBoardLogger(log_dir=tmpdir)

            logger.log({"loss": 0.5, "accuracy": 0.9}, step=1)
            logger.finish()

            # Check that event files were created
            event_files = list(Path(tmpdir).glob("events.out.*"))
            assert len(event_files) > 0

    def test_log_hyperparams(self):
        """Test log_hyperparams method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TensorBoardLogger(log_dir=tmpdir)

            # Should not raise
            logger.log_hyperparams({"lr": 1e-4, "batch_size": 32})
            logger.finish()

    def test_log_text(self):
        """Test log_text method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TensorBoardLogger(log_dir=tmpdir)

            # Should not raise
            logger.log_text("sample", "This is a test", step=1)
            logger.finish()

    def test_log_multiple_steps(self):
        """Test logging multiple steps."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TensorBoardLogger(log_dir=tmpdir)

            for step in range(10):
                logger.log({"loss": 1.0 / (step + 1)}, step=step)

            logger.finish()

    def test_finish_can_be_called_multiple_times(self):
        """Test finish can be called multiple times safely."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TensorBoardLogger(log_dir=tmpdir)

            logger.finish()
            # Should not raise
            logger.finish()

    def test_creates_log_directory(self):
        """Test creates log directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "nested" / "logs"
            logger = TensorBoardLogger(log_dir=str(log_dir))

            assert log_dir.exists()
            logger.finish()


class TestTensorBoardLoggerWithoutTensorBoard:
    """Tests when TensorBoard is not available."""

    def test_raises_import_error(self):
        """Test raises ImportError when TensorBoard not available."""
        if TENSORBOARD_AVAILABLE:
            pytest.skip("TensorBoard is available")

        with pytest.raises(ImportError):
            TensorBoardLogger()
