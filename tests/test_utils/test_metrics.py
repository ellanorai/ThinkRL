"""
Test Suite for ThinkRL Metrics Utilities
========================================
"""

import numpy as np
import pytest
import torch


# Try importing cupy to check availability for tests
try:
    import cupy as cp

    _CUPY_AVAILABLE = True
except (ImportError, OSError):
    cp = None
    _CUPY_AVAILABLE = False

from thinkrl.utils.metrics import (
    MetricsTracker,
    _compute_moments_manual,  # For direct testing
    aggregate_metrics,
    compute_accuracy,
    compute_advantages,
    compute_clip_fraction,
    compute_explained_variance,
    compute_group_metrics,
    compute_kl_divergence,
    compute_metrics,
    compute_perplexity,
    compute_policy_entropy,
    compute_ranking_metrics,
    compute_returns,
    compute_reward,
    compute_statistical_metrics,
    compute_statistical_metrics_batch,  # New import
)


# Use numpy as fallback for data generation if cupy missing
xp = cp if _CUPY_AVAILABLE else np


class TestMetrics:
    """Test metrics utilities."""

    def test_metrics_tracker(self):
        tracker = MetricsTracker()

        tracker.update("loss", 0.5)
        tracker.update("loss", 0.4)
        tracker.update("accuracy", 0.95)

        current = tracker.get_current()
        assert current["loss"] == 0.4
        assert current["accuracy"] == 0.95

        avg = tracker.get_average()
        assert avg["loss"] == pytest.approx(0.45)
        assert tracker.get_average("loss") == pytest.approx(0.45)
        assert tracker.get_average("accuracy") == pytest.approx(0.95)

        tracker.update_dict({"loss": 0.3, "new_metric": 1.0})
        assert tracker.get_current("loss") == 0.3
        assert tracker.get_current("new_metric") == 1.0

        history = tracker.get_history("loss")
        assert history == [0.5, 0.4, 0.3]

        summary = tracker.get_summary("loss")
        assert summary["mean"] == pytest.approx(0.4)
        assert summary["min"] == 0.3
        assert summary["max"] == 0.5

        tracker.reset("loss")
        assert "loss" not in tracker.get_current()
        assert "new_metric" in tracker.get_current()

        tracker.reset()
        assert "new_metric" not in tracker.get_current()
        assert not tracker.get_average()

    def test_compute_reward(self):
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = compute_reward(rewards, normalize=False)
        assert torch.allclose(result, rewards)

        normalized = compute_reward(rewards, normalize=True)
        assert torch.abs(normalized.mean()) < 1e-6
        assert torch.abs(normalized.std() - 1.0) < 1e-6

    def test_compute_kl_divergence(self):
        log_probs_policy = torch.log(torch.tensor([0.2, 0.3, 0.5]))
        log_probs_ref = torch.log(torch.tensor([0.1, 0.4, 0.5]))

        kl_div_mean = compute_kl_divergence(log_probs_policy, log_probs_ref, reduction="mean")
        assert isinstance(kl_div_mean, torch.Tensor)
        assert kl_div_mean.dim() == 0

        kl_div_sum = compute_kl_divergence(log_probs_policy, log_probs_ref, reduction="sum")
        assert kl_div_sum.dim() == 0
        assert kl_div_sum == pytest.approx(kl_div_mean * 3)

        kl_div_none = compute_kl_divergence(log_probs_policy, log_probs_ref, reduction="none")
        assert kl_div_none.dim() == 1
        assert kl_div_none.shape == (3,)

    def test_compute_advantages(self):
        rewards = torch.randn(4, 10)
        values = torch.randn(4, 10)

        advantages = compute_advantages(rewards=rewards, values=values, gamma=0.99, lambda_=0.95, normalize=True)
        assert advantages.shape == rewards.shape
        assert torch.abs(advantages.mean()) < 1e-6
        assert torch.abs(advantages.std() - 1.0) < 1e-5

        advantages_unnorm = compute_advantages(
            rewards=rewards, values=values, gamma=0.99, lambda_=0.95, normalize=False
        )
        assert advantages_unnorm.shape == rewards.shape
        assert torch.abs(advantages_unnorm.mean()) > 1e-5

    def test_compute_returns(self):
        rewards = torch.tensor([[1.0, 2.0, 3.0]])
        returns = compute_returns(rewards, gamma=0.9)
        assert returns.shape == rewards.shape
        assert torch.allclose(returns[0, 0], torch.tensor(5.23))
        assert torch.allclose(returns[0, 1], torch.tensor(4.7))
        assert torch.allclose(returns[0, 2], torch.tensor(3.0))

        returns_norm = compute_returns(rewards, gamma=0.9, normalize=True)
        assert torch.abs(returns_norm.mean()) < 1e-6
        assert torch.abs(returns_norm.std() - 1.0) < 1e-5

    def test_compute_policy_entropy(self):
        logits = torch.randn(4, 10, 100)
        entropy = compute_policy_entropy(logits, reduction="mean")
        assert isinstance(entropy, torch.Tensor)
        assert entropy >= 0

        entropy_none = compute_policy_entropy(logits, reduction="none")
        assert entropy_none.shape == (4, 10)

    def test_compute_accuracy(self):
        predictions_labels = torch.tensor([[1, 2, 3], [1, 1, 1]])
        targets = torch.tensor([[1, 2, 0], [1, 1, 1]])
        accuracy = compute_accuracy(predictions_labels, targets)
        assert 0.0 <= accuracy <= 1.0
        assert accuracy == pytest.approx(5 / 6)

        predictions_logits = torch.rand(2, 3, 4)
        predictions_logits[0, 0, 1] = 2.0
        predictions_logits[0, 1, 2] = 2.0
        predictions_logits[0, 2, 3] = 2.0
        predictions_logits[1, 0, 1] = 2.0
        predictions_logits[1, 1, 1] = 2.0
        predictions_logits[1, 2, 1] = 2.0

        accuracy_logits = compute_accuracy(predictions_logits, targets)
        assert accuracy_logits == pytest.approx(5 / 6)

        targets_with_ignore = torch.tensor([[1, 2, 0], [1, -100, 1]])
        predictions_ignore = torch.tensor([[1, 2, 3], [1, 1, 1]])
        accuracy_ignore = compute_accuracy(predictions_ignore, targets_with_ignore, ignore_index=-100)
        assert accuracy_ignore == pytest.approx(4 / 5)

    def test_compute_perplexity(self):
        loss = 2.5
        ppl = compute_perplexity(loss)
        assert ppl > 0
        # Use xp (numpy/cupy) for assertion
        assert float(xp.abs(ppl - xp.exp(2.5))) < 1e-6

        loss_tensor = torch.tensor(1.5)
        ppl_tensor = compute_perplexity(loss_tensor)
        assert float(xp.abs(ppl_tensor - xp.exp(1.5))) < 1e-6

    def test_compute_clip_fraction(self):
        ratio = torch.tensor([0.5, 1.0, 1.5, 2.0])
        clip_frac = compute_clip_fraction(ratio, epsilon=0.2)
        assert 0.0 <= clip_frac <= 1.0
        assert clip_frac == 0.75

    def test_compute_explained_variance(self):
        predictions = torch.randn(100)
        targets = torch.randn(100)
        ev = compute_explained_variance(predictions, targets)
        assert ev <= 1.0

        ev_perfect = compute_explained_variance(targets, targets)
        assert ev_perfect == pytest.approx(1.0)

        ev_zero_var = compute_explained_variance(predictions, torch.ones(100))
        assert ev_zero_var == 0.0

    def test_aggregate_metrics(self):
        metrics_list = [
            {"loss": 0.5, "accuracy": 0.9},
            {"loss": 0.6, "accuracy": 0.85},
            {"loss": 0.4, "accuracy": 0.95},
        ]
        avg_metrics = aggregate_metrics(metrics_list)
        assert avg_metrics["loss"] == pytest.approx(0.5)
        assert avg_metrics["accuracy"] == pytest.approx(0.9)

        weights = [1.0, 1.0, 2.0]
        avg_weighted = aggregate_metrics(metrics_list, weights)
        assert avg_weighted["loss"] == pytest.approx(0.475)

    def test_compute_statistical_metrics_basic(self):
        """Test basic statistical metrics calculation with NumPy."""
        values = np.arange(1, 101, dtype=float)
        stats = compute_statistical_metrics(values)

        assert "mean" in stats
        assert stats["mean"] == pytest.approx(50.5)
        assert stats["min"] == 1.0
        assert stats["max"] == 100.0
        assert stats["median"] == pytest.approx(50.5)
        assert stats["p25"] == pytest.approx(25.75)
        assert stats["p75"] == pytest.approx(75.25)
        assert stats["count"] == 100
        assert stats["nan_count"] == 0

    def test_compute_statistical_metrics_empty(self):
        """Test handling of empty or None inputs."""
        # None input
        stats_none = compute_statistical_metrics(None)
        assert stats_none["count"] == 0
        assert stats_none["mean"] == 0.0

        # Empty array
        stats_empty = compute_statistical_metrics([])
        assert stats_empty["count"] == 0
        assert stats_empty["mean"] == 0.0

        # Empty numpy array
        stats_np_empty = compute_statistical_metrics(np.array([]))
        assert stats_np_empty["count"] == 0

    def test_compute_statistical_metrics_single_value(self):
        """Test handling of scalar or single-element inputs."""
        # Scalar float
        stats_scalar = compute_statistical_metrics(5.0)
        assert stats_scalar["mean"] == 5.0
        assert stats_scalar["std"] == 0.0

        # Single element array
        stats_single = compute_statistical_metrics([10.0])
        assert stats_single["mean"] == 10.0
        assert stats_single["std"] == 0.0
        assert stats_single["p99"] == 10.0

    def test_compute_statistical_metrics_nan_inf(self):
        """Test robustness against NaN and Inf values."""
        data = np.array([1.0, 2.0, np.nan, 4.0, np.inf])
        stats = compute_statistical_metrics(data)

        assert stats["count"] == 5
        assert stats["nan_count"] == 1
        assert stats["inf_count"] == 1
        # Statistics should be computed on valid data: [1.0, 2.0, 4.0]
        assert stats["mean"] == pytest.approx(7.0 / 3.0)
        assert stats["max"] == 4.0

    def test_compute_statistical_metrics_tensor(self):
        """Test handling of PyTorch tensors (CPU)."""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        stats = compute_statistical_metrics(tensor)

        assert stats["mean"] == 2.0
        assert stats["std"] == 1.0  # Sample std of [1,2,3] is 1.0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_compute_statistical_metrics_gpu_tensor(self):
        """Test handling of PyTorch tensors on GPU (integrates with CuPy if available)."""
        tensor = torch.tensor([1.0, 2.0, 3.0], device="cuda")

        # This will internally try to use CuPy via DLPack
        stats = compute_statistical_metrics(tensor)

        assert stats["mean"] == 2.0
        assert stats["count"] == 3

    def test_compute_statistical_metrics_batch(self):
        """Test batch processing of multiple arrays."""
        batch = [
            np.array([1.0, 2.0, 3.0]),
            np.array([4.0, 5.0, 6.0]),
            [],  # Empty one
        ]

        results = compute_statistical_metrics_batch(batch)

        assert len(results) == 3
        assert results[0]["mean"] == 2.0
        assert results[1]["mean"] == 5.0
        assert results[2]["count"] == 0

    def test_manual_moments_computation(self):
        """Test the manual skewness/kurtosis fallback."""
        # Normal distribution (should be close to 0 skew, 0 excess kurtosis)
        # We use a deterministic array for testing logic
        data = np.array([-2, -1, 0, 1, 2], dtype=float)

        # Use helper directly
        moments = _compute_moments_manual(data, np)

        assert moments["skewness"] == pytest.approx(0.0)
        # Kurtosis for normal is 3, excess is 0. Manual calculation:
        # std=sqrt(2.5). z=[-1.26, -0.63, 0, 0.63, 1.26]. mean(z^4) approx 2.12
        # The formula calculates Fisher excess kurtosis.
        # For this symmetric, light-tailed distribution, it should be negative
        assert moments["kurtosis"] < 0

    def test_compute_group_metrics(self):
        rewards = torch.tensor([1.0, 2.0, 3.0, 10.0, 11.0, 20.0])
        group_ids = torch.tensor([0, 0, 0, 1, 1, 2])

        group_metrics = compute_group_metrics(rewards, group_ids)

        assert torch.allclose(group_metrics["group_means"], torch.tensor([2.0, 10.5, 20.0]))
        assert torch.allclose(group_metrics["group_maxs"], torch.tensor([3.0, 11.0, 20.0]))
        assert torch.allclose(group_metrics["group_mins"], torch.tensor([1.0, 10.0, 20.0]))

    def test_compute_ranking_metrics(self):
        scores = torch.tensor([0.9, 0.7, 0.5, 0.3, 0.1])
        labels = torch.tensor([1, 0, 1, 0, 0])

        metrics = compute_ranking_metrics(scores, labels, k=3)

        assert metrics["precision@3"] == pytest.approx(2 / 3)
        assert metrics["recall@3"] == pytest.approx(2 / 2)
        assert metrics["mrr"] == pytest.approx(1.0 / 1.0)
        assert metrics["average_precision"] == pytest.approx((1.0 + 2 / 3) / 2)

    def test_compute_metrics_convenience_fn(self):
        logits = torch.randn(2, 4, 10)
        targets = torch.randint(0, 10, (2, 4))

        targets[0, 0] = 5
        logits[0, 0, 5] = 10.0

        outputs = {
            "logits": logits,
            "values": torch.randn(2, 4),
            "rewards": torch.randn(2, 4),
            "log_probs": torch.log_softmax(logits, dim=-1),
            "ref_log_probs": torch.log_softmax(torch.randn(2, 4, 10), dim=-1),
            "ratio": torch.rand(2, 4) * 2,
            "returns": torch.randn(2, 4),
            "loss": torch.tensor(1.2),
        }

        metrics = compute_metrics(outputs, targets, metric_names=["all"])

        assert "accuracy" in metrics
        assert "perplexity" in metrics
        assert "entropy" in metrics
        assert "reward_mean" in metrics

        # Use xp (numpy/cupy) for assertion
        assert metrics["perplexity"] == pytest.approx(float(xp.exp(1.2)))

        metrics_subset = compute_metrics(outputs, targets, metric_names=["accuracy", "loss"])

        assert "accuracy" in metrics_subset
        assert "perplexity" not in metrics_subset
        assert "entropy" not in metrics_subset

    def test_compute_metrics_integration(self):
        """Test the high-level compute_metrics wrapper integrates statistical metrics."""
        outputs = {"rewards": torch.randn(10), "values": torch.randn(10)}

        metrics = compute_metrics(outputs, metric_names=["reward_stats", "value_stats"])

        assert "reward_mean" in metrics
        assert "reward_std" in metrics
        assert "value_p99" in metrics
        assert "value_nan_count" in metrics
