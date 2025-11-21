# thinkrl/__init__.py
"""
ThinkRL: State-of-the-art RLHF Training Library
"""

__version__ = "0.1.0"

# Core imports for user convenience
# Data components (when implemented)
from thinkrl.data.datasets import PreferenceDataset, RLHFDataset
from thinkrl.data.loaders import RLHFDataLoader
from thinkrl.utils.checkpoint import CheckpointManager
from thinkrl.utils.logging import get_logger, setup_logger
from thinkrl.utils.metrics import MetricsTracker, compute_metrics


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
