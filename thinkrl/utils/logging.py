"""
ThinkRL Logging Utilities
==========================

Comprehensive logging system for ThinkRL with support for:
- Colored console output
- File logging with rotation
- Integration with ML frameworks (wandb, tensorboard)
- Structured logging
- Performance tracking
- Distributed training logging

Author: Archit Sood @ EllanorAI
"""

import logging
import sys
import os
from pathlib import Path
from typing import Optional, Union, Dict, Any
from datetime import datetime
import warnings
import logging.handlers  # Import handlers


# ANSI color codes for terminal output
class Colors:
    """ANSI color codes for colored terminal output."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Foreground colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright foreground colors
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

    # Background colors
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"


class ColoredFormatter(logging.Formatter):
    """
    Custom formatter that adds colors to log messages based on log level.

    Colors are only applied when outputting to a terminal that supports them.
    """

    # Color mapping for different log levels
    LEVEL_COLORS = {
        logging.DEBUG: Colors.BRIGHT_BLACK,
        logging.INFO: Colors.BRIGHT_BLUE,
        logging.WARNING: Colors.BRIGHT_YELLOW,
        logging.ERROR: Colors.BRIGHT_RED,
        logging.CRITICAL: Colors.BG_RED + Colors.WHITE + Colors.BOLD,
    }

    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        use_colors: bool = True,
    ):
        """
        Initialize the colored formatter.

        Args:
            fmt: Log message format string
            datefmt: Date format string
            use_colors: Whether to use colors (auto-detected if None)
        """
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors and self._supports_color()

    @staticmethod
    def _supports_color() -> bool:
        """
        Check if the terminal supports color output.

        Returns:
            True if colors are supported, False otherwise
        """
        # Check if running in a terminal
        if not hasattr(sys.stdout, "isatty"):
            return False
        if not sys.stdout.isatty():
            return False

        # Check environment variables
        if os.environ.get("NO_COLOR"):
            return False
        if os.environ.get("FORCE_COLOR"):
            return True

        # Windows terminal color support
        if sys.platform == "win32":
            try:
                import colorama

                colorama.init()
                return True
            except ImportError:
                return False

        return True

    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record with colors.

        Args:
            record: Log record to format

        Returns:
            Formatted log message with colors
        """
        if self.use_colors:
            # Save original values
            orig_levelname = record.levelname
            orig_msg = record.msg

            # Apply colors
            color = self.LEVEL_COLORS.get(record.levelno, "")
            record.levelname = f"{color}{record.levelname}{Colors.RESET}"

            # Add color to the message if it's an error or warning
            if record.levelno >= logging.WARNING:
                record.msg = f"{color}{record.msg}{Colors.RESET}"

        # Format the message
        result = super().format(record)

        if self.use_colors:
            # Restore original values
            record.levelname = orig_levelname
            record.msg = orig_msg

        return result


class ThinkRLLogger(logging.Logger):
    """
    Custom logger class with additional methods for ML training.

    Extends the standard Logger with methods for:
    - Metrics logging
    - Progress tracking
    - Model checkpoints
    - Distributed training coordination
    """

    def __init__(self, name: str, level: int = logging.NOTSET):
        """
        Initialize the ThinkRL logger.

        Args:
            name: Logger name
            level: Logging level
        """
        super().__init__(name, level)
        self._metrics_buffer: Dict[str, Any] = {}

    def metric(self, name: str, value: Union[int, float], step: Optional[int] = None):
        """
        Log a metric value.

        Args:
            name: Metric name
            value: Metric value
            step: Training step (optional)
        """
        msg = f"Metric: {name}={value}"
        if step is not None:
            msg += f" (step={step})"
        self.info(msg)

        # Store in buffer for batch logging
        self._metrics_buffer[name] = {"value": value, "step": step}

    def checkpoint(self, path: str, metrics: Optional[Dict[str, Any]] = None):
        """
        Log a model checkpoint.

        Args:
            path: Checkpoint file path
            metrics: Associated metrics (optional)
        """
        msg = f"Checkpoint saved: {path}"
        if metrics:
            metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            msg += f" ({metrics_str})"
        self.info(msg)

    def progress(self, current: int, total: int, prefix: str = "Progress"):
        """
        Log progress information.

        Args:
            current: Current step
            total: Total steps
            prefix: Progress message prefix
        """
        percentage = 100 * current / total
        self.info(f"{prefix}: {current}/{total} ({percentage:.1f}%)")

    def get_metrics_buffer(self) -> Dict[str, Any]:
        """
        Get the buffered metrics.

        Returns:
            Dictionary of buffered metrics
        """
        return self._metrics_buffer.copy()

    def clear_metrics_buffer(self):
        """Clear the metrics buffer."""
        self._metrics_buffer.clear()


def setup_logger(
    name: str = "thinkrl",
    level: Union[int, str] = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    log_dir: Optional[Union[str, Path]] = None,
    format_string: Optional[str] = None,
    date_format: Optional[str] = None,
    use_colors: bool = True,
    file_mode: str = "a",
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
    rank: Optional[int] = None,
) -> logging.Logger:
    """
    Set up a logger with console and optional file output.

    This is the main function for configuring logging in ThinkRL.
    It supports:
    - Colored console output
    - File logging with rotation
    - Distributed training (rank-based logging)
    - Custom formatting

    Args:
        name: Logger name (default: "thinkrl")
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        log_dir: Directory for log files (optional, creates timestamped file using logger name)
        format_string: Custom log format string
        date_format: Custom date format string
        use_colors: Whether to use colored output for console
        file_mode: File opening mode ('a' for append, 'w' for write)
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup files to keep
        rank: Process rank for distributed training (if rank != 0, only NullHandler is added)

    Returns:
        Configured logger instance

    Example:
        ```python
        # Basic setup
        logger = setup_logger("thinkrl")
        logger.info("Training started")

        # With file logging in a directory
        logger = setup_logger(
            name="thinkrl_train",
            level=logging.DEBUG,
            log_dir="./logs",
            use_colors=True
        )

        # Distributed training (only rank 0 logs)
        logger = setup_logger(
            name="thinkrl_dist",
            rank=int(os.environ.get("RANK", 0))
        )
        ```
    """
    # Convert string level to logging constant
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    # Register custom logger class
    logging.setLoggerClass(ThinkRLLogger)

    # Get or create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # --- Important: Clear existing handlers to prevent duplicates ---
    # This is crucial if setup_logger might be called multiple times on the same logger name
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    # ----------------------------------------------------------------

    # Only log from rank 0 in distributed training, others get NullHandler
    # This simplifies logic: non-zero ranks *only* get NullHandler from this function.
    if rank is not None and rank != 0:
        logger.addHandler(logging.NullHandler())
        logger.propagate = False  # Prevent propagation for non-zero ranks too
        return logger

    # --- Setup for rank 0 (or non-distributed) ---

    # Default format strings
    if format_string is None:
        format_string = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "%(filename)s:%(lineno)d - %(message)s"
        )

    if date_format is None:
        date_format = "%Y-%m-%d %H:%M:%S"

    # Console handler with colors (always add for rank 0 or non-distributed)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = ColoredFormatter(
        fmt=format_string, datefmt=date_format, use_colors=use_colors
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler (optional, only for rank 0 or non-distributed)
    actual_log_path = None
    if log_file or log_dir:
        # Determine log file path
        if log_file:
            log_path = Path(log_file)
        else:  # log_dir is specified
            log_dir_path = Path(log_dir)
            log_dir_path.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Use logger name in the filename
            log_path = log_dir_path / f"{name}_{timestamp}.log"

        actual_log_path = log_path
        # Create parent directory if it doesn't exist
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Use rotating file handler to prevent huge log files
        try:
            file_handler = logging.handlers.RotatingFileHandler(
                filename=str(log_path),
                mode=file_mode,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
        except ImportError:
            # Fallback to regular file handler
            warnings.warn(
                "RotatingFileHandler not available. Using standard FileHandler."
            )
            file_handler = logging.FileHandler(
                filename=str(log_path), mode=file_mode, encoding="utf-8"
            )

        file_handler.setLevel(level)

        # File handler without colors
        file_formatter = logging.Formatter(fmt=format_string, datefmt=date_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Log the file path *only* if a file handler was successfully added
        logger.info(f"Logging to file: {log_path}")

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_logger(
    name: str = "thinkrl", level: Optional[Union[int, str]] = None
) -> logging.Logger:
    """
    Get an existing logger or create a new one with basic configuration if needed.

    This function ensures that if a logger is requested and hasn't been configured
    by `setup_logger`, it gets a default console handler. It avoids adding handlers
    if the logger already has some.

    Args:
        name: Logger name (default: "thinkrl")
        level: Logging level (optional, used only if creating default handlers)

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)

    # If logger has no handlers AND is not disabled (level > CRITICAL), set it up.
    # Check for level NOTSET as well, as getLogger returns a logger with NOTSET by default.
    if (
        not logger.handlers
        and logger.level <= logging.CRITICAL
        and logger.level != logging.NOTSET
    ):
        # Use provided level or default to INFO if creating handlers
        effective_level = level
        if effective_level is None:
            effective_level = logging.INFO
        elif isinstance(effective_level, str):
            effective_level = getattr(logging, effective_level.upper(), logging.INFO)

        # Only set level if it's currently NOTSET (don't override existing level)
        if logger.level == logging.NOTSET:
            logger.setLevel(effective_level)

        # Add console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(effective_level)  # Use the determined level

        formatter = ColoredFormatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False  # Prevent double logging to root

    return logger


def configure_logging_for_distributed(
    rank: int,
    world_size: int,
    log_dir: Optional[Path] = None,
    level: int = logging.INFO,
    base_name: str = "thinkrl",
):
    logger_name = f"{base_name}.rank{rank}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False

    # Always start clean: remove/close any existing handlers
    for h in logger.handlers[:]:
        try:
            h.close()
        except Exception:
            pass
        logger.removeHandler(h)

    fmt = (
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    )
    formatter = logging.Formatter(fmt)

    if rank == 0:
        # console
        sh = logging.StreamHandler()
        sh.setLevel(level)
        sh.setFormatter(formatter)
        logger.addHandler(sh)
        # file (if requested)
        if log_dir is not None:
            log_path = (
                Path(log_dir) / f"{logger_name}_{datetime.now():%Y%m%d_%H%M%S}.log"
            )
            fh = logging.FileHandler(log_path, encoding="utf-8")
            fh.setLevel(level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
    else:
        if log_dir is not None:
            # file-only for nonzero ranks
            log_path = (
                Path(log_dir) / f"{logger_name}_{datetime.now():%Y%m%d_%H%M%S}.log"
            )
            fh = logging.FileHandler(log_path, encoding="utf-8")
            fh.setLevel(level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        else:
            # no outputs for nonzero ranks when no log_dir
            logger.addHandler(logging.NullHandler())

    return logger


def disable_external_loggers(level: int = logging.WARNING):
    """
    Disable or reduce verbosity of external library loggers.

    This is useful to reduce noise from third-party libraries like
    transformers, accelerate, etc.

    Args:
        level: Logging level to set for external loggers

    Example:
        ```python
        # Silence noisy libraries
        disable_external_loggers(logging.ERROR)
        ```
    """
    external_loggers = [
        "transformers",
        "accelerate",
        "datasets",
        "torch",
        "tensorboard",
        "wandb",
        "deepspeed",
        "peft",
        "bitsandbytes",
    ]

    for logger_name in external_loggers:
        logging.getLogger(logger_name).setLevel(level)


# Module-level logger
_module_logger: Optional[logging.Logger] = None


def get_module_logger() -> logging.Logger:
    """
    Get the module-level logger for thinkrl.utils.logging.

    Returns:
        Module logger instance
    """
    global _module_logger
    if _module_logger is None:
        _module_logger = get_logger(__name__)
    return _module_logger


# Public API
__all__ = [
    "setup_logger",
    "get_logger",
    "configure_logging_for_distributed",
    "disable_external_loggers",
    "ColoredFormatter",
    "ThinkRLLogger",
    "Colors",
]
