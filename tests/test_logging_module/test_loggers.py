"""
Tests for Logger Implementations
================================

Comprehensive tests for logging backends.
"""

import io
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from thinkrl.logging.loggers import (
    Logger,
    NullLogger,
    ConsoleLogger,
    CompositeLogger,
    is_main_process,
    log_only_main_process,
    create_logger,
)


class TestNullLogger:
    """Tests for NullLogger."""

    def test_initialization(self):
        """Test NullLogger initialization."""
        logger = NullLogger()
        assert logger is not None

    def test_log_does_nothing(self):
        """Test log method does nothing."""
        logger = NullLogger()
        # Should not raise
        logger.log({"loss": 0.5}, step=1)

    def test_log_hyperparams_does_nothing(self):
        """Test log_hyperparams does nothing."""
        logger = NullLogger()
        # Should not raise
        logger.log_hyperparams({"lr": 1e-4})

    def test_finish_does_nothing(self):
        """Test finish does nothing."""
        logger = NullLogger()
        # Should not raise
        logger.finish()


class TestConsoleLogger:
    """Tests for ConsoleLogger."""

    def test_initialization(self):
        """Test ConsoleLogger initialization."""
        logger = ConsoleLogger()

        assert logger.prefix == ""
        assert logger.log_every_n_steps == 1

    def test_custom_initialization(self):
        """Test ConsoleLogger with custom parameters."""
        logger = ConsoleLogger(
            prefix="[TRAIN]",
            log_every_n_steps=10,
        )

        assert logger.prefix == "[TRAIN]"
        assert logger.log_every_n_steps == 10

    def test_log_prints_to_stream(self):
        """Test log method prints to stream."""
        stream = io.StringIO()
        logger = ConsoleLogger(stream=stream)

        logger.log({"loss": 0.5, "accuracy": 0.9}, step=1)

        output = stream.getvalue()
        assert "Step 1" in output
        assert "loss" in output
        assert "0.5" in output

    def test_log_respects_frequency(self):
        """Test log respects log_every_n_steps."""
        stream = io.StringIO()
        logger = ConsoleLogger(log_every_n_steps=5, stream=stream)

        logger.log({"loss": 0.5}, step=1)
        logger.log({"loss": 0.4}, step=2)
        logger.log({"loss": 0.3}, step=5)

        output = stream.getvalue()
        # Step 1 and 2 should not be logged (1 % 5 != 0)
        assert "Step 1" not in output
        assert "Step 2" not in output
        # Step 5 should be logged
        assert "Step 5" in output

    def test_log_with_prefix(self):
        """Test log with prefix."""
        stream = io.StringIO()
        logger = ConsoleLogger(prefix="[TEST]", stream=stream)

        logger.log({"loss": 0.5}, step=1)

        output = stream.getvalue()
        assert "[TEST]" in output

    def test_log_hyperparams(self):
        """Test log_hyperparams method."""
        stream = io.StringIO()
        logger = ConsoleLogger(stream=stream)

        logger.log_hyperparams({"lr": 1e-4, "batch_size": 32})

        output = stream.getvalue()
        assert "Hyperparameters" in output
        assert "lr" in output

    def test_log_formats_floats(self):
        """Test that floats are formatted correctly."""
        stream = io.StringIO()
        logger = ConsoleLogger(stream=stream)

        logger.log({"loss": 0.123456789}, step=1)

        output = stream.getvalue()
        # Should be formatted to 4 decimal places
        assert "0.1235" in output

    def test_log_handles_integers(self):
        """Test that integers are logged correctly."""
        stream = io.StringIO()
        logger = ConsoleLogger(stream=stream)

        logger.log({"step": 100, "epoch": 5}, step=1)

        output = stream.getvalue()
        assert "100" in output
        assert "5" in output

    def test_skip_duplicate_step(self):
        """Test that duplicate steps are skipped."""
        stream = io.StringIO()
        logger = ConsoleLogger(stream=stream)

        logger.log({"loss": 0.5}, step=1)
        logger.log({"loss": 0.4}, step=1)

        output = stream.getvalue()
        # Should only log once
        lines = [l for l in output.split("\n") if l.strip()]
        assert len(lines) == 1


class TestCompositeLogger:
    """Tests for CompositeLogger."""

    def test_initialization(self):
        """Test CompositeLogger initialization."""
        logger = CompositeLogger([])
        assert logger.loggers == []

    def test_log_calls_all_loggers(self):
        """Test log method calls all child loggers."""
        mock1 = MagicMock(spec=Logger)
        mock2 = MagicMock(spec=Logger)
        logger = CompositeLogger([mock1, mock2])

        logger.log({"loss": 0.5}, step=1)

        mock1.log.assert_called_once_with({"loss": 0.5}, 1)
        mock2.log.assert_called_once_with({"loss": 0.5}, 1)

    def test_log_hyperparams_calls_all_loggers(self):
        """Test log_hyperparams calls all child loggers."""
        mock1 = MagicMock(spec=Logger)
        mock2 = MagicMock(spec=Logger)
        logger = CompositeLogger([mock1, mock2])

        logger.log_hyperparams({"lr": 1e-4})

        mock1.log_hyperparams.assert_called_once()
        mock2.log_hyperparams.assert_called_once()

    def test_finish_calls_all_loggers(self):
        """Test finish calls all child loggers."""
        mock1 = MagicMock(spec=Logger)
        mock2 = MagicMock(spec=Logger)
        logger = CompositeLogger([mock1, mock2])

        logger.finish()

        mock1.finish.assert_called_once()
        mock2.finish.assert_called_once()

    def test_add_logger(self):
        """Test add_logger method."""
        logger = CompositeLogger([])
        mock = MagicMock(spec=Logger)

        logger.add_logger(mock)

        assert mock in logger.loggers

    def test_log_text_calls_all_loggers(self):
        """Test log_text calls all child loggers."""
        mock1 = MagicMock(spec=Logger)
        mock2 = MagicMock(spec=Logger)
        logger = CompositeLogger([mock1, mock2])

        logger.log_text("key", "value", step=1)

        mock1.log_text.assert_called_once()
        mock2.log_text.assert_called_once()


class TestIsMainProcess:
    """Tests for is_main_process function."""

    def test_returns_true_without_distributed(self):
        """Test returns True when distributed is not initialized."""
        result = is_main_process()
        assert result is True


class TestLogOnlyMainProcess:
    """Tests for log_only_main_process function."""

    def test_logs_when_main_process(self):
        """Test logging when on main process."""
        mock_logger = MagicMock(spec=Logger)

        with patch("thinkrl.logging.loggers.is_main_process", return_value=True):
            log_only_main_process(mock_logger, {"loss": 0.5}, step=1)

        mock_logger.log.assert_called_once_with({"loss": 0.5}, 1)

    def test_skips_when_not_main_process(self):
        """Test skipping log when not on main process."""
        mock_logger = MagicMock(spec=Logger)

        with patch("thinkrl.logging.loggers.is_main_process", return_value=False):
            log_only_main_process(mock_logger, {"loss": 0.5}, step=1)

        mock_logger.log.assert_not_called()


class TestCreateLogger:
    """Tests for create_logger factory function."""

    def test_returns_null_logger_for_non_main_rank(self):
        """Test returns NullLogger for non-main rank."""
        logger = create_logger(["console"], rank=1)

        assert isinstance(logger, NullLogger)

    def test_returns_console_logger(self):
        """Test returns ConsoleLogger for console backend."""
        logger = create_logger(["console"])

        assert isinstance(logger, ConsoleLogger)

    def test_returns_composite_for_multiple_backends(self):
        """Test returns CompositeLogger for multiple backends."""
        logger = create_logger(["console", "console"])

        assert isinstance(logger, CompositeLogger)

    def test_returns_null_logger_for_empty_backends(self):
        """Test returns NullLogger for empty backends."""
        logger = create_logger([])

        assert isinstance(logger, NullLogger)

    def test_ignores_unknown_backend(self):
        """Test ignores unknown backend names."""
        logger = create_logger(["unknown_backend"])

        assert isinstance(logger, NullLogger)

    def test_console_with_log_frequency(self):
        """Test console logger respects log frequency."""
        logger = create_logger(["console"], log_every_n_steps=10)

        # Single logger returned directly
        assert isinstance(logger, ConsoleLogger)
        assert logger.log_every_n_steps == 10


class TestLoggerContextManager:
    """Tests for logger context manager protocol."""

    def test_console_logger_context_manager(self):
        """Test ConsoleLogger as context manager."""
        with ConsoleLogger() as logger:
            logger.log({"test": 1}, step=1)
        # finish() should have been called

    def test_null_logger_context_manager(self):
        """Test NullLogger as context manager."""
        with NullLogger() as logger:
            logger.log({"test": 1}, step=1)

    def test_composite_logger_context_manager(self):
        """Test CompositeLogger as context manager."""
        mock = MagicMock(spec=Logger)
        with CompositeLogger([mock]) as logger:
            logger.log({"test": 1}, step=1)

        mock.finish.assert_called_once()
