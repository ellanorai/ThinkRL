"""
Test Suite for ThinkRL Metrics Utilities
========================================

Tests for:
- MetricsTracker
- compute_reward
- compute_kl_divergence
- compute_advantages
- compute_returns
- compute_policy_entropy
- compute_accuracy
- compute_perplexity
- compute_clip_fraction
- compute_explained_variance
- aggregate_metrics
- compute_group_metrics
- compute_ranking_metrics
- compute_statistical_metrics
- compute_metrics (convenience function)

Author: Archit Sood
"""

import pytest
import torch
# REPLACED: replaced numpy with cupy for GPU acceleration.
import cupy as cp

# Modules under test
from thinkrl.utils.metrics import (
    MetricsTracker,
    compute_reward,
    compute_kl_divergence,
    compute_advantages,
    compute_returns,
    compute_policy_entropy,
    compute_accuracy,
    compute_perplexity,
    compute_clip_fraction,
    compute_explained_variance,
    aggregate_metrics,
    compute_group_metrics,
    compute_ranking_metrics,
    compute_statistical_metrics,
    compute_metrics,
)

# ============================================================================
# Metrics Tests
# ============================================================================

class TestMetrics:
    """Test metrics utilities."""

    def test_metrics_tracker(self):
        """Test MetricsTracker class."""
        tracker = MetricsTracker()

        # Update metrics
        tracker.update("loss", 0.5)
        tracker.update("loss", 0.4)
        tracker.update("accuracy", 0.95)

        # Get current values
        current = tracker.get_current()
        assert current["loss"] == 0.4
        assert current["accuracy"] == 0.95

        # Get averages
        avg = tracker.get_average()
        assert avg["loss"] == pytest.approx(0.45)  # (0.5 + 0.4) / 2
        assert "loss" in tracker.get_average()
        assert tracker.get_average("loss") == pytest.approx(0.45)
        assert tracker.get_average("accuracy") == pytest.approx(0.95)


        # Test update_dict
        tracker.update_dict({"loss": 0.3, "new_metric": 1.0})
        assert tracker.get_current("loss") == 0.3
        assert tracker.get_current("new_metric") == 1.0

        # Test get_history
        history = tracker.get_history("loss")
        assert history == [0.5, 0.4, 0.3]

        # Test get_summary
        summary = tracker.get_summary("loss")
        assert summary["mean"] == pytest.approx(0.4)
        assert summary["min"] == 0.3
        assert summary["max"] == 0.5

        # Reset single
        tracker.reset("loss")
        assert "loss" not in tracker.get_current()
        assert "new_metric" in tracker.get_current() # Should still be there

        # Reset all
        tracker.reset()
        assert "new_metric" not in tracker.get_current()
        assert not tracker.get_average()

    def test_compute_reward(self):
        """Test reward computation."""
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])

        # Without normalization
        result = compute_reward(rewards, normalize=False)
        assert torch.allclose(result, rewards)

        # With normalization
        normalized = compute_reward(rewards, normalize=True)
        assert torch.abs(normalized.mean()) < 1e-6
        assert torch.abs(normalized.std() - 1.0) < 1e-6 # std of sample, not population

    def test_compute_kl_divergence(self):
        """Test KL divergence computation."""
        log_probs_policy = torch.log(torch.tensor([0.2, 0.3, 0.5]))
        log_probs_ref = torch.log(torch.tensor([0.1, 0.4, 0.5]))

        # Test mean (default)
        kl_div_mean = compute_kl_divergence(log_probs_policy, log_probs_ref, reduction="mean")
        assert isinstance(kl_div_mean, torch.Tensor)
        assert kl_div_mean.dim() == 0  # Scalar

        # Test sum
        kl_div_sum = compute_kl_divergence(log_probs_policy, log_probs_ref, reduction="sum")
        assert kl_div_sum.dim() == 0
        assert kl_div_sum == pytest.approx(kl_div_mean * 3)

        # Test none
        kl_div_none = compute_kl_divergence(log_probs_policy, log_probs_ref, reduction="none")
        assert kl_div_none.dim() == 1
        assert kl_div_none.shape == (3,)

    def test_compute_advantages(self):
        """Test GAE advantage computation."""
        rewards = torch.randn(4, 10)
        values = torch.randn(4, 10)

        advantages = compute_advantages(
            rewards=rewards, values=values, gamma=0.99, lambda_=0.95, normalize=True
        )

        assert advantages.shape == rewards.shape
        # Check normalization
        assert (
            torch.abs(advantages.mean()) < 1e-6
        )
        assert (
            torch.abs(advantages.std() - 1.0) < 1e-5
        )

        # Test without normalization
        advantages_unnorm = compute_advantages(
            rewards=rewards, values=values, gamma=0.99, lambda_=0.95, normalize=False
        )
        assert advantages_unnorm.shape == rewards.shape
        assert torch.abs(advantages_unnorm.mean()) > 1e-5 # Should not be zero-mean

    def test_compute_returns(self):
        """Test returns computation."""
        rewards = torch.tensor([[1.0, 2.0, 3.0]])
        returns = compute_returns(rewards, gamma=0.9)

        assert returns.shape == rewards.shape
        # First return should be highest (includes all future rewards)
        # 1.0 + 0.9*2.0 + 0.9*0.9*3.0 = 1 + 1.8 + 2.43 = 5.23
        assert torch.allclose(returns[0, 0], torch.tensor(5.23))
        assert torch.allclose(returns[0, 1], torch.tensor(2.0 + 0.9*3.0)) # 4.7
        assert torch.allclose(returns[0, 2], torch.tensor(3.0)) # 3.0

        # Test normalization
        returns_norm = compute_returns(rewards, gamma=0.9, normalize=True)
        assert torch.abs(returns_norm.mean()) < 1e-6
        assert torch.abs(returns_norm.std() - 1.0) < 1e-5 # Bessesl's correction for std

    def test_compute_policy_entropy(self):
        """Test policy entropy computation."""
        logits = torch.randn(4, 10, 100)  # batch, seq, vocab
        entropy = compute_policy_entropy(logits, reduction="mean")

        assert isinstance(entropy, torch.Tensor)
        assert entropy >= 0  # Entropy is non-negative
        
        entropy_none = compute_policy_entropy(logits, reduction="none")
        assert entropy_none.shape == (4, 10)

    def test_compute_accuracy(self):
        """Test accuracy computation."""
        # Test with label predictions
        predictions_labels = torch.tensor([[1, 2, 3], [1, 1, 1]])
        targets = torch.tensor([[1, 2, 0], [1, 1, 1]])

        accuracy = compute_accuracy(predictions_labels, targets)
        assert 0.0 <= accuracy <= 1.0
        assert accuracy == pytest.approx(5 / 6)  # 5 correct out of 6

        # Test with logits
        predictions_logits = torch.rand(2, 3, 4) # B, L, C
        predictions_logits[0, 0, 1] = 2.0 # correct
        predictions_logits[0, 1, 2] = 2.0 # correct
        predictions_logits[0, 2, 3] = 2.0 # correct (target 0) -> this is wrong
        predictions_logits[1, 0, 1] = 2.0 # correct
        predictions_logits[1, 1, 1] = 2.0 # correct
        predictions_logits[1, 2, 1] = 2.0 # correct
        
        # Logits should predict:
        # [1, 2, 3]
        # [1, 1, 1]
        
        accuracy_logits = compute_accuracy(predictions_logits, targets)
        assert accuracy_logits == pytest.approx(5 / 6) # 5 correct out of 6

        # Test with ignore_index
        targets_with_ignore = torch.tensor([[1, 2, 0], [1, -100, 1]])
        predictions_ignore = torch.tensor([[1, 2, 3], [1, 1, 1]])
        accuracy_ignore = compute_accuracy(predictions_ignore, targets_with_ignore, ignore_index=-100)
        # Should ignore [1,0] (val 1) and [0,2] (val 3)
        # Correct: [0,0], [0,1], [1,0], [1,2] -> 4 correct
        # Total: 5 valid
        assert accuracy_ignore == pytest.approx(4 / 5)

    def test_compute_perplexity(self):
        """Test perplexity computation."""
        loss = 2.5
        ppl = compute_perplexity(loss)

        assert ppl > 0
        # UPDATED: Using cupy for verification
        assert float(cp.abs(ppl - cp.exp(2.5))) < 1e-6

        loss_tensor = torch.tensor(1.5)
        ppl_tensor = compute_perplexity(loss_tensor)
        # UPDATED: Using cupy for verification
        assert float(cp.abs(ppl_tensor - cp.exp(1.5))) < 1e-6

    def test_compute_clip_fraction(self):
        """Test clip fraction computation."""
        ratio = torch.tensor([0.5, 1.0, 1.5, 2.0])
        clip_frac = compute_clip_fraction(ratio, epsilon=0.2)

        assert 0.0 <= clip_frac <= 1.0
        # Ratios 0.5 (< 0.8), 1.5 (> 1.2), and 2.0 (> 1.2) are clipped
        assert clip_frac == 0.75  # 3/4

    def test_compute_explained_variance(self):
        """Test explained variance computation."""
        predictions = torch.randn(100)
        targets = torch.randn(100)

        ev = compute_explained_variance(predictions, targets)
        assert ev <= 1.0 # Can be negative if predictions are bad

        # Test perfect prediction
        ev_perfect = compute_explained_variance(targets, targets)
        assert ev_perfect == pytest.approx(1.0)
        
        # Test zero variance target
        ev_zero_var = compute_explained_variance(predictions, torch.ones(100))
        assert ev_zero_var == 0.0

    def test_aggregate_metrics(self):
        """Test metrics aggregation."""
        metrics_list = [
            {"loss": 0.5, "accuracy": 0.9},
            {"loss": 0.6, "accuracy": 0.85},
            {"loss": 0.4, "accuracy": 0.95},
        ]

        # Test uniform weights
        avg_metrics = aggregate_metrics(metrics_list)
        assert "loss" in avg_metrics
        assert "accuracy" in avg_metrics
        assert avg_metrics["loss"] == pytest.approx(0.5)  # (0.5 + 0.6 + 0.4) / 3
        assert avg_metrics["accuracy"] == pytest.approx(0.9)

        # Test weighted aggregation
        weights = [1.0, 1.0, 2.0] # Total weight 4
        # loss: (0.5*1 + 0.6*1 + 0.4*2) / 4 = (0.5 + 0.6 + 0.8) / 4 = 1.9 / 4 = 0.475
        avg_weighted = aggregate_metrics(metrics_list, weights)
        assert avg_weighted["loss"] == pytest.approx(0.475)

    def test_compute_statistical_metrics(self):
        """Test statistical metrics computation."""
        # UPDATED: Using cupy array for test data
        values = cp.arange(1, 101, dtype=cp.float32) # 1 to 100
        stats = compute_statistical_metrics(values)

        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "median" in stats
        assert "p25" in stats
        assert "p75" in stats
        assert "p99" in stats
        
        assert stats["mean"] == pytest.approx(50.5)
        assert stats["min"] == 1.0
        assert stats["max"] == 100.0
        assert stats["median"] == pytest.approx(50.5)
        assert stats["p25"] == pytest.approx(25.75)
        assert stats["p75"] == pytest.approx(75.25)
        
        # Test scipy metrics if available
        if "skewness" in stats:
            assert stats["skewness"] == pytest.approx(0.0)

    # --- New tests for functions not in test_all_utils.py ---

    def test_compute_group_metrics(self):
        """Test group-wise metrics computation."""
        rewards = torch.tensor([1.0, 2.0, 3.0, 10.0, 11.0, 20.0])
        group_ids = torch.tensor([0, 0, 0, 1, 1, 2])
        
        group_metrics = compute_group_metrics(rewards, group_ids)
        
        assert "group_means" in group_metrics
        assert "group_stds" in group_metrics
        assert "group_maxs" in group_metrics
        assert "group_mins" in group_metrics
        
        # Group 0: [1, 2, 3] -> mean=2, max=3, min=1
        # Group 1: [10, 11] -> mean=10.5, max=11, min=10
        # Group 2: [20] -> mean=20, max=20, min=20
        
        assert torch.allclose(group_metrics["group_means"], torch.tensor([2.0, 10.5, 20.0]))
        assert torch.allclose(group_metrics["group_maxs"], torch.tensor([3.0, 11.0, 20.0]))
        assert torch.allclose(group_metrics["group_mins"], torch.tensor([1.0, 10.0, 20.0]))

    def test_compute_ranking_metrics(self):
        """Test ranking metrics computation."""
        scores = torch.tensor([0.9, 0.7, 0.5, 0.3, 0.1])
        labels = torch.tensor([1,   0,   1,   0,   0]) # Relevant items at index 0, 2
        
        metrics = compute_ranking_metrics(scores, labels, k=3)
        
        # Sorted labels: [1, 0, 1] (at k=3)
        # Total relevant = 2
        
        assert "precision@3" in metrics
        assert "recall@3" in metrics
        assert "mrr" in metrics
        assert "average_precision" in metrics
        
        assert metrics["precision@3"] == pytest.approx(2 / 3)
        assert metrics["recall@3"] == pytest.approx(2 / 2) # Found 2 out of 2 relevant
        assert metrics["mrr"] == pytest.approx(1.0 / 1.0) # First relevant is at pos 1
        
        # AP: (1/1 * 1) + (1/2 * 0) + (2/3 * 1) + (2/4 * 0) + (2/5 * 0) = 1 + 2/3 = 1.666...
        # AP = 1.666... / 2 = 0.8333...
        assert metrics["average_precision"] == pytest.approx( (1.0 + 2/3) / 2 )

    def test_compute_metrics_convenience_fn(self):
        """Test the main compute_metrics convenience function."""
        logits = torch.randn(2, 4, 10) # B, L, C
        targets = torch.randint(0, 10, (2, 4))
        
        # Make one prediction correct
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
            "loss": torch.tensor(1.2)
        }
        
        # Test with all metrics
        metrics = compute_metrics(outputs, targets, metric_names=["all"])
        
        assert "accuracy" in metrics
        assert "perplexity" in metrics
        assert "entropy" in metrics
        assert "reward_mean" in metrics
        assert "value_std" in metrics
        assert "kl_div" in metrics
        assert "clip_fraction" in metrics
        assert "explained_variance" in metrics
        
        # UPDATED: Using cupy for verification
        assert metrics["perplexity"] == pytest.approx(float(cp.exp(1.2)))
        
        # Test with specific metrics
        metrics_subset = compute_metrics(
            outputs, targets, metric_names=["accuracy", "loss"] # Note: "loss" is not a computed metric
        )
        
        assert "accuracy" in metrics_subset
        assert "perplexity" not in metrics_subset # "loss" was in outputs, but "perplexity" wasn't requested
        assert "entropy" not in metrics_subset