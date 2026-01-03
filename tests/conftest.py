"""
ThinkRL Test Configuration
==========================

pytest configuration and fixtures for the ThinkRL test suite.
"""

import logging

import pytest


@pytest.fixture(autouse=True)
def enable_log_propagation():
    """
    Enable log propagation for pytest's caplog fixture.

    The ThinkRL logging module sets `propagate = False` on loggers to avoid
    double-logging in production. However, pytest's caplog fixture captures
    logs by attaching a handler to the root logger. When propagation is
    disabled, logs from child loggers (like `thinkrl.utils.agent`) don't
    reach the root logger, causing caplog to see nothing.

    This fixture temporarily enables propagation on the thinkrl logger
    hierarchy during tests to allow caplog to capture all log messages.
    """
    # Get the thinkrl root logger
    thinkrl_logger = logging.getLogger("thinkrl")

    # Store original propagation states
    original_propagate = thinkrl_logger.propagate

    # Enable propagation for the thinkrl logger hierarchy
    thinkrl_logger.propagate = True

    # Also enable for any existing child loggers
    child_loggers = []
    for name in list(logging.Logger.manager.loggerDict.keys()):
        if name.startswith("thinkrl."):
            child_logger = logging.getLogger(name)
            child_loggers.append((child_logger, child_logger.propagate))
            child_logger.propagate = True

    yield

    # Restore original propagation states
    thinkrl_logger.propagate = original_propagate
    for child_logger, orig_propagate in child_loggers:
        child_logger.propagate = orig_propagate
