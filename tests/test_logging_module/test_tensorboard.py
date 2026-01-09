"""
Tests for TensorBoard Logger
============================

Tests for TensorBoard integration.
"""

from pathlib import Path
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from thinkrl.logging.tensorboard import TENSORBOARD_AVAILABLE, TensorBoardLogger


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


class TestTensorBoardLoggerMocked:
    """Tests for TensorBoardLogger with mocked SummaryWriter."""

    @patch("thinkrl.logging.tensorboard.SummaryWriter")
    @patch("thinkrl.logging.tensorboard.TENSORBOARD_AVAILABLE", True)
    def test_log_histogram(self, mock_writer_class):
        """Test log_histogram method."""
        mock_writer = MagicMock()
        mock_writer_class.return_value = mock_writer

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TensorBoardLogger(log_dir=tmpdir)
            logger.log_histogram("weights", [0.1, 0.2, 0.3, 0.4], step=1)

            mock_writer.add_histogram.assert_called_once_with("weights", [0.1, 0.2, 0.3, 0.4], 1, bins="tensorflow")
            logger.finish()

    @patch("thinkrl.logging.tensorboard.SummaryWriter")
    @patch("thinkrl.logging.tensorboard.TENSORBOARD_AVAILABLE", True)
    def test_log_image(self, mock_writer_class):
        """Test log_image method."""
        mock_writer = MagicMock()
        mock_writer_class.return_value = mock_writer

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TensorBoardLogger(log_dir=tmpdir)
            # Simulate a simple image tensor (3, 32, 32)
            fake_image = [[1, 2], [3, 4]]
            logger.log_image("sample_image", fake_image, step=1)

            mock_writer.add_image.assert_called_once_with("sample_image", fake_image, 1, dataformats="CHW")
            logger.finish()

    @patch("thinkrl.logging.tensorboard.SummaryWriter")
    @patch("thinkrl.logging.tensorboard.TENSORBOARD_AVAILABLE", True)
    def test_log_metrics(self, mock_writer_class):
        """Test log method calls add_scalar for each metric."""
        mock_writer = MagicMock()
        mock_writer_class.return_value = mock_writer

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TensorBoardLogger(log_dir=tmpdir)
            logger.log({"loss": 0.5, "accuracy": 0.9}, step=10)

            assert mock_writer.add_scalar.call_count == 2
            logger.finish()

    @patch("thinkrl.logging.tensorboard.SummaryWriter")
    @patch("thinkrl.logging.tensorboard.TENSORBOARD_AVAILABLE", True)
    def test_log_hyperparams_with_none_values(self, mock_writer_class):
        """Test log_hyperparams handles None values."""
        mock_writer = MagicMock()
        mock_writer_class.return_value = mock_writer

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TensorBoardLogger(log_dir=tmpdir)
            logger.log_hyperparams({"lr": 1e-4, "scheduler": None, "model": "meta-llama/Llama-3.2-1B"})

            mock_writer.add_hparams.assert_called_once()
            call_args = mock_writer.add_hparams.call_args[0][0]
            assert call_args["scheduler"] == "None"
            logger.finish()

    @patch("thinkrl.logging.tensorboard.SummaryWriter")
    @patch("thinkrl.logging.tensorboard.TENSORBOARD_AVAILABLE", True)
    def test_log_hyperparams_with_complex_values(self, mock_writer_class):
        """Test log_hyperparams converts complex values to strings."""
        mock_writer = MagicMock()
        mock_writer_class.return_value = mock_writer

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TensorBoardLogger(log_dir=tmpdir)
            logger.log_hyperparams({"config": {"nested": "dict"}, "lr": 0.01})

            mock_writer.add_hparams.assert_called_once()
            call_args = mock_writer.add_hparams.call_args[0][0]
            # Complex values are converted to strings
            assert isinstance(call_args["config"], str)
            logger.finish()

    @patch("thinkrl.logging.tensorboard.SummaryWriter")
    @patch("thinkrl.logging.tensorboard.TENSORBOARD_AVAILABLE", True)
    def test_writer_property(self, mock_writer_class):
        """Test writer property returns the SummaryWriter instance."""
        mock_writer = MagicMock()
        mock_writer_class.return_value = mock_writer

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TensorBoardLogger(log_dir=tmpdir)
            assert logger.writer is mock_writer
            logger.finish()

    @patch("thinkrl.logging.tensorboard.SummaryWriter")
    @patch("thinkrl.logging.tensorboard.TENSORBOARD_AVAILABLE", True)
    def test_log_text_with_default_step(self, mock_writer_class):
        """Test log_text uses default step when None."""
        mock_writer = MagicMock()
        mock_writer_class.return_value = mock_writer

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TensorBoardLogger(log_dir=tmpdir)
            logger.log_text("tag", "some text")

            mock_writer.add_text.assert_called_once_with("tag", "some text", global_step=0)
            logger.finish()

    @patch("thinkrl.logging.tensorboard.SummaryWriter")
    @patch("thinkrl.logging.tensorboard.TENSORBOARD_AVAILABLE", True)
    def test_finish_flushes_and_closes(self, mock_writer_class):
        """Test finish method flushes and closes writer."""
        mock_writer = MagicMock()
        mock_writer_class.return_value = mock_writer

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TensorBoardLogger(log_dir=tmpdir)
            logger.finish()

            mock_writer.flush.assert_called_once()
            mock_writer.close.assert_called_once()
