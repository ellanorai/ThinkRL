import pytest
from unittest.mock import patch, MagicMock
import numpy as np
import torch
from thinkrl.utils import metrics

class TestMetricsMocked:
    def test_compute_statistical_metrics_cpu_fallback(self):
        """Force CPU fallback even if CuPy is installed."""
        # Simulate CuPy missing or import error
        with patch("thinkrl.utils.metrics._CUPY_AVAILABLE", False):
            data = np.array([1.0, 2.0, 3.0])
            stats = metrics.compute_statistical_metrics(data)
            assert stats["mean"] == 2.0
            assert stats["count"] == 3

    def test_compute_higher_moments_manual(self):
        """Force manual calculation of skew/kurtosis (simulating no scipy/cupyx)."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # Mock _CUPY_AVAILABLE=False and _SCIPY_AVAILABLE=False
        with patch("thinkrl.utils.metrics._CUPY_AVAILABLE", False), \
             patch("thinkrl.utils.metrics._SCIPY_AVAILABLE", False):
            
            # accessing the private helper directly to ensure it runs
            moments = metrics._compute_moments_manual(data, np)
            assert "skewness" in moments
            assert "kurtosis" in moments