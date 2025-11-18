# thinkrl/__init__.py
"""
ThinkRL: State-of-the-art RLHF Training Library
"""

__version__ = "0.1.0"

# Core imports for user convenience
from thinkrl.utils.logging import setup_logger, get_logger
from thinkrl.utils.metrics import MetricsTracker, compute_metrics
from thinkrl.utils.checkpoint import CheckpointManager

# Data components (when implemented)
from thinkrl.data.datasets import RLHFDataset, PreferenceDataset
from thinkrl.data.loaders import RLHFDataLoader

# Will add more as we implement:


__all__ = [
    "__version__",
    "setup_logger",
    "get_logger",
    "MetricsTracker",
    "compute_metrics",
    "CheckpointManager",
    "RLHFDataset",
    "PreferenceDataset",
    "RLHFDataLoader",
]