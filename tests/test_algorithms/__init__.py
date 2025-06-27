"""
Test module for ThinkRL algorithms.

This module contains tests for all algorithm implementations in ThinkRL,
including DAPO, VAPO, GRPO, PPO, and REINFORCE. It provides base classes
and utilities for testing algorithm functionality.

Test Modules:
    test_dapo: Tests for DAPO algorithm implementation
    test_vapo: Tests for VAPO algorithm implementation (when available)
    test_grpo: Tests for GRPO algorithm implementation (when available)
    test_ppo: Tests for PPO algorithm implementation (when available)
    test_reinforce: Tests for REINFORCE algorithm implementation (when available)

Base Classes and Utilities:
    AlgorithmConfig: Base configuration for testing
    AlgorithmOutput: Standardized output format
    BaseAlgorithm: Abstract base class
    MockModel: Mock model for testing
    TestDataGenerator: Utilities for generating test data

Example:
    >>> from tests.test_algorithms.base import MockModel, create_dummy_batch
    >>> model = MockModel(vocab_size=1000, hidden_size=512)
    >>> batch = create_dummy_batch(batch_size=4, seq_len=32)
    >>> outputs = model(**batch)
"""

# Import base classes and utilities for easy access
from .base import (  # Base algorithm interfaces; Mock implementations; Test utilities; Constants
    TEST_CONFIG,
    TEST_DEVICES,
    AlgorithmConfig,
    AlgorithmOutput,
    AlgorithmRegistry,
    BaseAlgorithm,
    MockModel,
    MockValueModel,
    ModelProtocol,
    TestDataGenerator,
    assert_model_output,
    create_dummy_batch,
    create_mock_tokenizer,
)

# Version information
__version__ = "0.1.0"

# Public API exports
__all__ = [
    # Base classes
    "AlgorithmConfig",
    "AlgorithmOutput",
    "BaseAlgorithm",
    "AlgorithmRegistry",
    "ModelProtocol",
    # Mock implementations
    "MockModel",
    "MockValueModel",
    # Test utilities
    "TestDataGenerator",
    "assert_model_output",
    "create_dummy_batch",
    "create_mock_tokenizer",
    # Constants
    "TEST_DEVICES",
    "TEST_CONFIG",
]

# Module metadata
__author__ = "ThinkRL Team"
__description__ = "Test utilities and base classes for ThinkRL algorithms"
