"""
Test Suite for ThinkRL Logging Utilities
========================================

Tests for:
- setup_logger
- get_logger
- configure_logging_for_distributed
- ColoredFormatter
- ThinkRLLogger (via setup_logger)

Author: Archit Sood
"""

import logging
import logging.handlers  # Import handlers for isinstance check
from pathlib import Path
import shutil
import tempfile

import pytest

# Modules under test
from thinkrl.utils.logging import (
    ColoredFormatter,
    ThinkRLLogger,
    configure_logging_for_distributed,
    get_logger,
    setup_logger,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests and ensure loggers are closed."""
    temp_dir_path = tempfile.mkdtemp()
    yield Path(temp_dir_path)
    # --- Close all logging handlers before removing the directory ---
    logging.shutdown()
    # Add error handling for Windows file locking issues during teardown
    try:
        shutil.rmtree(temp_dir_path)
    except PermissionError:
        print(f"Warning: Could not remove temp directory {temp_dir_path} due to PermissionError.")
    except OSError as e:
        print(f"Warning: Error removing temp directory {temp_dir_path}: {e}")


# ============================================================================
# Logging Tests
# ============================================================================


class TestLogging:
    """Test logging utilities."""

    def test_setup_logger_basic(self, temp_dir):
        """Test basic logger setup."""
        logger_name = "test_logger_basic"
        logger = setup_logger(name=logger_name, level=logging.INFO, log_dir=temp_dir)

        assert logger is not None
        assert logger.name == logger_name
        assert logger.level == logging.INFO
        assert isinstance(logger, ThinkRLLogger)  # Check for custom class

        # Test logging
        logger.info("Test message")
        logger.warning("Test warning")

        # Check log file was created (using the name provided)
        log_files = list(temp_dir.glob(f"{logger_name}_*.log"))
        assert len(log_files) == 1, f"Expected 1 log file matching {logger_name}_*.log"

        # --- Explicitly close handlers for this logger ---
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

    def test_get_logger(self):
        """Test get_logger function."""
        # This will create a basic logger since one isn't set up
        logger = get_logger("thinkrl.test_get_new")
        assert logger is not None
        assert isinstance(logger, logging.Logger)  # get_logger returns base logger if not setup
        assert logger.name == "thinkrl.test_get_new"

        # Test getting an already configured logger
        setup_logger("thinkrl.test_get_setup")
        logger2 = get_logger("thinkrl.test_get_setup")
        assert isinstance(logger2, ThinkRLLogger)

    def test_colored_formatter(self):
        """Test colored formatter."""
        formatter = ColoredFormatter(fmt="%(levelname)s - %(message)s", use_colors=True)

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)

        # Check if color codes are present (if terminal supports it)
        if formatter.use_colors:
            assert "\033[" in formatted
            assert "\033[0m" in formatted
        assert "Test message" in formatted

    def test_distributed_logging(self, temp_dir):
        """Test distributed logging configuration."""
        # --- Logger base name for this test ---
        base_name_for_test = "thinkrl.test_dist"
        logger_name_rank0 = f"{base_name_for_test}.rank0"
        logger_name_rank1 = f"{base_name_for_test}.rank1"

        # --- Rank 0 ---
        # Rank 0 should log to console and file
        logger_rank0 = configure_logging_for_distributed(
            rank=0,
            world_size=2,
            log_dir=temp_dir,
            level=logging.INFO,
            base_name=base_name_for_test,
        )
        assert logger_rank0 is not None
        assert logger_rank0.name == logger_name_rank0  # Check name
        assert any(
            isinstance(h, logging.StreamHandler) for h in logger_rank0.handlers
        ), "Rank 0 should have a StreamHandler"

        # UP038 Fix: Combined check using | operator
        assert any(
            isinstance(h, logging.FileHandler | logging.handlers.RotatingFileHandler) for h in logger_rank0.handlers
        ), "Rank 0 should have a FileHandler"

        logger_rank0.info("Rank 0 message")
        # Check rank 0 log file exists
        rank0_log_files = list(temp_dir.glob(f"{logger_name_rank0}_*.log"))
        assert len(rank0_log_files) == 1, "Log file for rank 0 should exist"

        # --- Rank 1 with log_dir ---
        # Rank 1 should log ONLY to file if log_dir provided
        logger_rank1_file = configure_logging_for_distributed(
            rank=1,
            world_size=2,
            log_dir=temp_dir,
            level=logging.INFO,
            base_name=base_name_for_test,
        )
        assert logger_rank1_file is not None
        assert logger_rank1_file.name == logger_name_rank1  # Check name
        # Assert ONLY FileHandler (or RotatingFileHandler) exists
        assert len(logger_rank1_file.handlers) == 1, "Rank 1 with log_dir should have exactly one handler"

        # UP038 Fix: Using | instead of tuple (,)
        assert isinstance(
            logger_rank1_file.handlers[0],
            logging.FileHandler | logging.handlers.RotatingFileHandler,
        )

        logger_rank1_file.info("Rank 1 message to file")
        # Check that the file was actually created using the correct name pattern
        rank1_log_files = list(temp_dir.glob(f"{logger_name_rank1}_*.log"))
        assert len(rank1_log_files) == 1, f"Log file matching {logger_name_rank1}_*.log for rank 1 should exist"

        # --- FIX: Clear handlers for rank 1 logger BEFORE reconfiguring ---
        # Get the logger instance directly
        logger_rank1_instance = logging.getLogger(logger_name_rank1)
        for handler in logger_rank1_instance.handlers[:]:
            handler.close()
            logger_rank1_instance.removeHandler(handler)
        # -----------------------------------------------------------------

        # --- Rank 1 without log_dir ---
        # Rank 1 without log_dir should now correctly get only a NullHandler
        logger_rank1_null = configure_logging_for_distributed(
            rank=1,
            world_size=2,
            log_dir=None,
            base_name=base_name_for_test,  # log_dir is None
        )
        assert logger_rank1_null is not None
        assert logger_rank1_null.name == logger_name_rank1  # Should be the same logger instance

        # Assert NullHandler is present and ONLY handler
        assert len(logger_rank1_null.handlers) == 1, "Rank 1 without log_dir should have exactly one handler"
        assert isinstance(
            logger_rank1_null.handlers[0], logging.NullHandler
        ), "Rank 1 without log_dir should have NullHandler"

        logger_rank1_null.info("This rank 1 message should NOT appear anywhere")  # Test it doesn't log

    def test_thinkrl_logger_methods(self, temp_dir):
        """Test the custom methods of ThinkRLLogger."""
        logger = setup_logger("thinkrl.custom", log_dir=temp_dir)

        assert isinstance(logger, ThinkRLLogger)

        # Test metric logging
        logger.metric("loss", 0.5, step=10)
        logger.metric("accuracy", 0.9, step=10)

        buffer = logger.get_metrics_buffer()
        assert "loss" in buffer
        assert buffer["loss"]["value"] == 0.5
        assert buffer["loss"]["step"] == 10
        assert buffer["accuracy"]["value"] == 0.9

        logger.clear_metrics_buffer()
        assert not logger.get_metrics_buffer()

        # Test checkpoint logging
        logger.checkpoint("/path/to/model.pt", metrics={"loss": 0.5})

        # Test progress logging
        logger.progress(50, 100, prefix="Epoch 1")

        # Check if logs are in the file
        log_file = list(temp_dir.glob("thinkrl.custom_*.log"))[0]
        with open(log_file) as f:
            content = f.read()

        assert "Metric: loss=0.5 (step=10)" in content
        assert "Checkpoint saved: /path/to/model.pt (loss=0.5000)" in content
        assert "Epoch 1: 50/100 (50.0%)" in content

    def test_setup_logger_rank_nonzero(self, temp_dir):
        """Test that non-zero ranks get a NullHandler."""
        logger = setup_logger("thinkrl.rank5", log_dir=temp_dir, rank=5)

        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], logging.NullHandler)
        assert logger.propagate is False

        # Try logging, should produce no output
        logger.info("This should not be logged")

        # No log file should be created
        log_files = list(temp_dir.glob("thinkrl.rank5_*.log"))
        assert len(log_files) == 0

    def test_colored_formatter_no_colors(self):
        """Test colored formatter with colors disabled."""
        formatter = ColoredFormatter(fmt="%(levelname)s - %(message)s", use_colors=False)

        record = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname="test.py",
            lineno=1,
            msg="Warning message",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)

        # Should not contain color codes
        assert "\033[" not in formatted
        assert "Warning message" in formatted

    def test_colored_formatter_different_levels(self):
        """Test colored formatter with different log levels."""
        formatter = ColoredFormatter(fmt="%(levelname)s - %(message)s", use_colors=True)

        levels = [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]

        for level in levels:
            record = logging.LogRecord(
                name="test",
                level=level,
                pathname="test.py",
                lineno=1,
                msg="Test message",
                args=(),
                exc_info=None,
            )
            formatted = formatter.format(record)
            assert "Test message" in formatted

    def test_setup_logger_with_rotation(self, temp_dir):
        """Test logger setup with log rotation."""
        logger = setup_logger(
            "thinkrl.rotation",
            log_dir=temp_dir,
            max_bytes=1024,
            backup_count=3,
        )

        assert logger is not None

        # Log some messages
        for i in range(10):
            logger.info(f"Test message {i}")

        # Close handlers for cleanup
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

    def test_thinkrl_logger_log_dict(self, temp_dir):
        """Test logging dictionary data."""
        logger = setup_logger("thinkrl.dictlog", log_dir=temp_dir)

        data = {"loss": 0.5, "accuracy": 0.9, "epoch": 10}
        logger.info(f"Training stats: {data}")

        log_file = list(temp_dir.glob("thinkrl.dictlog_*.log"))[0]
        with open(log_file) as f:
            content = f.read()

        assert "Training stats:" in content
        assert "loss" in content

    def test_setup_logger_console_only(self):
        """Test setting up logger without file logging."""
        logger = setup_logger("thinkrl.console_only", log_dir=None)

        assert logger is not None
        # Should only have console handler
        stream_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)]
        assert len(stream_handlers) >= 1

    def test_thinkrl_logger_epoch_logging(self, temp_dir):
        """Test epoch progress logging."""
        logger = setup_logger("thinkrl.epoch", log_dir=temp_dir)

        # Test progress method with different scenarios
        logger.progress(0, 100, prefix="Epoch 1")
        logger.progress(50, 100, prefix="Epoch 1")
        logger.progress(100, 100, prefix="Epoch 1")

        log_file = list(temp_dir.glob("thinkrl.epoch_*.log"))[0]
        with open(log_file) as f:
            content = f.read()

        assert "Epoch 1" in content
        assert "100.0%" in content

    def test_get_logger_returns_existing(self):
        """Test that get_logger returns existing configured logger."""
        # First set up a logger
        original = setup_logger("thinkrl.test_existing")

        # Get the same logger
        retrieved = get_logger("thinkrl.test_existing")

        # Should be the same logger instance
        assert retrieved.name == original.name
