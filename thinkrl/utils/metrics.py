"""
ThinkRL Metrics Utilities
==========================

Comprehensive metrics computation for RLHF training including:
- Reward computation and normalization
- KL divergence calculation
- Policy metrics (entropy, advantage, etc.)
- Accuracy and evaluation metrics
- Metric aggregation and tracking
- Statistical utilities

Author: Archit Sood @ EllanorAI
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict
import warnings

import torch
import torch.nn.functional as F
import numpy as np

# Optional dependencies
try:
    from scipy import stats
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False
    warnings.warn("SciPy not available. Some statistical functions will be limited.")

logger = logging.getLogger(__name__)


class MetricsTracker:
    """
    Track and aggregate metrics over training.
    
    Features:
    - Running average computation
    - Metric history tracking
    - Statistical summaries
    - Batch and epoch aggregation
    
    Example:
        ```python
        tracker = MetricsTracker()
        
        # Add metrics
        tracker.update("loss", 0.5)
        tracker.update("accuracy", 0.95)
        
        # Get current values
        print(tracker.get_current())
        
        # Get averages
        print(tracker.get_average())
        
        # Reset for new epoch
        tracker.reset()
        ```
    """
    
    def __init__(self):
        """Initialize the metrics tracker."""
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.current_values: Dict[str, float] = {}
    
    def update(self, name: str, value: Union[float, torch.Tensor], count: int = 1):
        """
        Update a metric with a new value.
        
        Args:
            name: Metric name
            value: Metric value (scalar or tensor)
            count: Number of samples (for weighted averaging)
        """
        # Convert tensor to float
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().item()
        
        # Store value
        self.metrics[name].append((value, count))
        self.current_values[name] = value
    
    def update_dict(self, metrics: Dict[str, Union[float, torch.Tensor]]):
        """
        Update multiple metrics at once.
        
        Args:
            metrics: Dictionary of metric names to values
        """
        for name, value in metrics.items():
            self.update(name, value)
    
    def get_current(self, name: Optional[str] = None) -> Union[float, Dict[str, float]]:
        """
        Get current (most recent) metric value(s).
        
        Args:
            name: Metric name (if None, returns all metrics)
            
        Returns:
            Current metric value or dictionary of all current values
        """
        if name is not None:
            return self.current_values.get(name, 0.0)
        return self.current_values.copy()
    
    def get_average(self, name: Optional[str] = None) -> Union[float, Dict[str, float]]:
        """
        Get average metric value(s) over all updates.
        
        Args:
            name: Metric name (if None, returns all metrics)
            
        Returns:
            Average metric value or dictionary of all averages
        """
        if name is not None:
            if name not in self.metrics or not self.metrics[name]:
                return 0.0
            
            values, counts = zip(*self.metrics[name])
            total_count = sum(counts)
            weighted_sum = sum(v * c for v, c in zip(values, counts))
            return weighted_sum / total_count if total_count > 0 else 0.0
        
        # Return all averages
        return {
            metric_name: self.get_average(metric_name)
            for metric_name in self.metrics.keys()
        }
    
    def get_history(self, name: str) -> List[float]:
        """
        Get full history of a metric.
        
        Args:
            name: Metric name
            
        Returns:
            List of metric values
        """
        if name not in self.metrics:
            return []
        return [value for value, _ in self.metrics[name]]
    
    def get_summary(self, name: str) -> Dict[str, float]:
        """
        Get statistical summary of a metric.
        
        Args:
            name: Metric name
            
        Returns:
            Dictionary with mean, std, min, max
        """
        history = self.get_history(name)
        if not history:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        
        return {
            "mean": float(np.mean(history)),
            "std": float(np.std(history)),
            "min": float(np.min(history)),
            "max": float(np.max(history)),
        }
    
    def reset(self, name: Optional[str] = None):
        """
        Reset tracked metrics.
        
        Args:
            name: Metric name to reset (if None, resets all)
        """
        if name is not None:
            if name in self.metrics:
                self.metrics[name].clear()
            if name in self.current_values:
                del self.current_values[name]
        else:
            self.metrics.clear()
            self.current_values.clear()
    
    def __repr__(self) -> str:
        """String representation of tracker."""
        avg_metrics = self.get_average()
        return f"MetricsTracker({avg_metrics})"


def compute_reward(
    rewards: torch.Tensor,
    normalize: bool = True,
    epsilon: float = 1e-8
) -> torch.Tensor:
    """
    Compute and optionally normalize rewards.
    
    Args:
        rewards: Reward tensor of shape (batch_size,) or (batch_size, seq_len)
        normalize: Whether to normalize rewards (zero mean, unit variance)
        epsilon: Small constant for numerical stability
        
    Returns:
        Processed reward tensor
        
    Example:
        ```python
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
        normalized_rewards = compute_reward(rewards, normalize=True)
        # Output: tensor([-1.3416, -0.4472,  0.4472,  1.3416])
        ```
    """
    if normalize:
        mean = rewards.mean()
        std = rewards.std() + epsilon
        rewards = (rewards - mean) / std
    
    return rewards


def compute_kl_divergence(
    log_probs_policy: torch.Tensor,
    log_probs_ref: torch.Tensor,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    Compute KL divergence between policy and reference distributions.
    
    KL(policy || ref) = E[log(policy) - log(ref)]
    
    Args:
        log_probs_policy: Log probabilities from policy model
        log_probs_ref: Log probabilities from reference model
        reduction: How to reduce the KL divergence ("mean", "sum", "none")
        
    Returns:
        KL divergence tensor
        
    Example:
        ```python
        log_probs_policy = torch.log(torch.tensor([0.2, 0.3, 0.5]))
        log_probs_ref = torch.log(torch.tensor([0.1, 0.4, 0.5]))
        kl_div = compute_kl_divergence(log_probs_policy, log_probs_ref)
        ```
    """
    kl_div = log_probs_policy - log_probs_ref
    
    if reduction == "mean":
        return kl_div.mean()
    elif reduction == "sum":
        return kl_div.sum()
    elif reduction == "none":
        return kl_div
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def compute_advantages(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float = 0.99,
    lambda_: float = 0.95,
    normalize: bool = True
) -> torch.Tensor:
    """
    Compute Generalized Advantage Estimation (GAE).
    
    GAE(λ) = Σ(γλ)^t δ_t
    where δ_t = r_t + γV(s_{t+1}) - V(s_t)
    
    Args:
        rewards: Rewards tensor of shape (batch_size, seq_len)
        values: Value estimates of shape (batch_size, seq_len)
        gamma: Discount factor
        lambda_: GAE lambda parameter
        normalize: Whether to normalize advantages
        
    Returns:
        Advantage tensor of shape (batch_size, seq_len)
        
    Example:
        ```python
        rewards = torch.randn(4, 10)  # batch_size=4, seq_len=10
        values = torch.randn(4, 10)
        advantages = compute_advantages(rewards, values)
        ```
    """
    batch_size, seq_len = rewards.shape
    advantages = torch.zeros_like(rewards)
    
    # Compute TD residuals
    next_values = torch.cat([values[:, 1:], torch.zeros(batch_size, 1, device=values.device)], dim=1)
    deltas = rewards + gamma * next_values - values
    
    # Compute GAE
    gae = 0
    for t in reversed(range(seq_len)):
        gae = deltas[:, t] + gamma * lambda_ * gae
        advantages[:, t] = gae
    
    # Normalize advantages
    if normalize:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    return advantages


def compute_returns(
    rewards: torch.Tensor,
    gamma: float = 0.99,
    normalize: bool = False
) -> torch.Tensor:
    """
    Compute discounted returns.
    
    G_t = Σ(γ^k * r_{t+k}) for k = 0 to T-t
    
    Args:
        rewards: Rewards tensor of shape (batch_size, seq_len)
        gamma: Discount factor
        normalize: Whether to normalize returns
        
    Returns:
        Returns tensor of shape (batch_size, seq_len)
        
    Example:
        ```python
        rewards = torch.tensor([[1.0, 2.0, 3.0]])
        returns = compute_returns(rewards, gamma=0.9)
        # returns[0, 0] = 1.0 + 0.9*2.0 + 0.81*3.0 = 5.23
        ```
    """
    batch_size, seq_len = rewards.shape
    returns = torch.zeros_like(rewards)
    
    # Compute returns backward
    G = 0
    for t in reversed(range(seq_len)):
        G = rewards[:, t] + gamma * G
        returns[:, t] = G
    
    # Normalize returns
    if normalize:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    
    return returns


def compute_policy_entropy(
    logits: torch.Tensor,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    Compute entropy of policy distribution.
    
    H(π) = -Σ π(a|s) log π(a|s)
    
    Args:
        logits: Logits tensor of shape (batch_size, seq_len, vocab_size)
        reduction: How to reduce entropy ("mean", "sum", "none")
        
    Returns:
        Entropy tensor
        
    Example:
        ```python
        logits = torch.randn(4, 10, 50000)  # batch, seq_len, vocab
        entropy = compute_policy_entropy(logits)
        ```
    """
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    
    if reduction == "mean":
        return entropy.mean()
    elif reduction == "sum":
        return entropy.sum()
    elif reduction == "none":
        return entropy
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def compute_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100
) -> float:
    """
    Compute accuracy of predictions.
    
    Args:
        predictions: Predicted labels of shape (batch_size, seq_len) or logits
        targets: Target labels of shape (batch_size, seq_len)
        ignore_index: Index to ignore in accuracy computation
        
    Returns:
        Accuracy as a float between 0 and 1
        
    Example:
        ```python
        predictions = torch.tensor([[1, 2, 3], [1, 1, 1]])
        targets = torch.tensor([[1, 2, 0], [1, 1, 1]])
        acc = compute_accuracy(predictions, targets)
        # acc = 5/6 = 0.833
        ```
    """
    # If predictions are logits, get argmax
    if predictions.dim() > targets.dim():
        predictions = predictions.argmax(dim=-1)
    
    # Create mask for valid positions
    mask = targets != ignore_index
    
    # Compute accuracy
    correct = (predictions == targets) & mask
    accuracy = correct.sum().float() / mask.sum().float()
    
    return accuracy.item()


def compute_perplexity(
    loss: Union[float, torch.Tensor]
) -> float:
    """
    Compute perplexity from cross-entropy loss.
    
    Perplexity = exp(loss)
    
    Args:
        loss: Cross-entropy loss value
        
    Returns:
        Perplexity value
        
    Example:
        ```python
        loss = 2.5
        ppl = compute_perplexity(loss)
        # ppl = exp(2.5) = 12.18
        ```
    """
    if isinstance(loss, torch.Tensor):
        loss = loss.item()
    
    return np.exp(loss)


def compute_clip_fraction(
    ratio: torch.Tensor,
    epsilon: float = 0.2
) -> float:
    """
    Compute fraction of ratios that were clipped in PPO.
    
    This is useful for monitoring if clipping is too aggressive.
    
    Args:
        ratio: Probability ratio r_t(θ) = π_θ(a|s) / π_θ_old(a|s)
        epsilon: Clipping threshold
        
    Returns:
        Fraction of ratios that were clipped
        
    Example:
        ```python
        ratio = torch.tensor([0.5, 1.0, 1.5, 2.0])
        clip_frac = compute_clip_fraction(ratio, epsilon=0.2)
        # Ratios outside [0.8, 1.2] are clipped: 3/4 = 0.75
        ```
    """
    clipped = ((ratio < 1 - epsilon) | (ratio > 1 + epsilon)).float()
    return clipped.mean().item()


def compute_explained_variance(
    predictions: torch.Tensor,
    targets: torch.Tensor
) -> float:
    """
    Compute explained variance for value function.
    
    EV = 1 - Var(y - ŷ) / Var(y)
    
    A value close to 1 indicates the value function is doing a good job.
    
    Args:
        predictions: Predicted values
        targets: Target values (e.g., returns)
        
    Returns:
        Explained variance between -inf and 1
        
    Example:
        ```python
        predictions = torch.randn(100)
        targets = torch.randn(100)
        ev = compute_explained_variance(predictions, targets)
        ```
    """
    var_y = targets.var()
    if var_y == 0:
        return 0.0
    
    return 1.0 - (targets - predictions).var() / var_y


def aggregate_metrics(
    metrics_list: List[Dict[str, float]],
    weights: Optional[List[float]] = None
) -> Dict[str, float]:
    """
    Aggregate metrics from multiple batches or processes.
    
    Args:
        metrics_list: List of metric dictionaries
        weights: Optional weights for weighted averaging
        
    Returns:
        Dictionary of aggregated metrics
        
    Example:
        ```python
        batch_metrics = [
            {"loss": 0.5, "accuracy": 0.9},
            {"loss": 0.6, "accuracy": 0.85},
            {"loss": 0.4, "accuracy": 0.95}
        ]
        avg_metrics = aggregate_metrics(batch_metrics)
        # {"loss": 0.5, "accuracy": 0.9}
        ```
    """
    if not metrics_list:
        return {}
    
    # Uniform weights if not provided
    if weights is None:
        weights = [1.0] * len(metrics_list)
    
    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    
    # Aggregate each metric
    aggregated = {}
    all_keys = set()
    for metrics in metrics_list:
        all_keys.update(metrics.keys())
    
    for key in all_keys:
        values = [metrics.get(key, 0.0) for metrics in metrics_list]
        aggregated[key] = sum(v * w for v, w in zip(values, weights))
    
    return aggregated


def compute_group_metrics(
    rewards: torch.Tensor,
    group_ids: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    Compute metrics per group (useful for GRPO - Group Relative Policy Optimization).
    
    Args:
        rewards: Reward tensor of shape (batch_size,)
        group_ids: Group IDs tensor of shape (batch_size,)
        
    Returns:
        Dictionary with group-wise statistics
        
    Example:
        ```python
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        group_ids = torch.tensor([0, 0, 1, 1, 2, 2])
        group_metrics = compute_group_metrics(rewards, group_ids)
        ```
    """
    unique_groups = torch.unique(group_ids)
    
    group_means = []
    group_stds = []
    group_maxs = []
    group_mins = []
    
    for group_id in unique_groups:
        mask = group_ids == group_id
        group_rewards = rewards[mask]
        
        group_means.append(group_rewards.mean())
        group_stds.append(group_rewards.std())
        group_maxs.append(group_rewards.max())
        group_mins.append(group_rewards.min())
    
    return {
        "group_means": torch.stack(group_means),
        "group_stds": torch.stack(group_stds),
        "group_maxs": torch.stack(group_maxs),
        "group_mins": torch.stack(group_mins),
    }


def compute_ranking_metrics(
    scores: torch.Tensor,
    labels: torch.Tensor,
    k: int = 10
) -> Dict[str, float]:
    """
    Compute ranking metrics (useful for preference learning).
    
    Args:
        scores: Predicted scores of shape (n_samples,)
        labels: True preference labels (1 for preferred, 0 otherwise)
        k: Top-k for precision@k and recall@k
        
    Returns:
        Dictionary with ranking metrics
        
    Example:
        ```python
        scores = torch.tensor([0.9, 0.7, 0.5, 0.3, 0.1])
        labels = torch.tensor([1, 0, 1, 0, 0])
        metrics = compute_ranking_metrics(scores, labels, k=3)
        ```
    """
    # Sort by scores
    sorted_indices = torch.argsort(scores, descending=True)
    sorted_labels = labels[sorted_indices]
    
    # Precision@k
    precision_at_k = sorted_labels[:k].float().mean().item()
    
    # Recall@k
    total_relevant = labels.sum().item()
    if total_relevant > 0:
        recall_at_k = sorted_labels[:k].sum().item() / total_relevant
    else:
        recall_at_k = 0.0
    
    # Mean Reciprocal Rank (MRR)
    relevant_positions = (sorted_labels == 1).nonzero(as_tuple=True)[0]
    if len(relevant_positions) > 0:
        mrr = 1.0 / (relevant_positions[0].item() + 1)
    else:
        mrr = 0.0
    
    # Average Precision (AP)
    cumsum = sorted_labels.cumsum(dim=0)
    positions = torch.arange(1, len(sorted_labels) + 1, device=sorted_labels.device)
    precision_at_i = cumsum / positions
    ap = (precision_at_i * sorted_labels).sum().item() / max(total_relevant, 1)
    
    return {
        f"precision@{k}": precision_at_k,
        f"recall@{k}": recall_at_k,
        "mrr": mrr,
        "average_precision": ap,
    }


def compute_statistical_metrics(
    values: Union[torch.Tensor, np.ndarray, List[float]]
) -> Dict[str, float]:
    """
    Compute comprehensive statistical metrics.
    
    Args:
        values: Values to compute statistics for
        
    Returns:
        Dictionary with statistical metrics
        
    Example:
        ```python
        values = torch.randn(1000)
        stats = compute_statistical_metrics(values)
        print(stats)
        ```
    """
    # Convert to numpy
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    elif isinstance(values, list):
        values = np.array(values)
    
    metrics = {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "median": float(np.median(values)),
    }
    
    # Add percentiles
    for percentile in [25, 75, 90, 95, 99]:
        metrics[f"p{percentile}"] = float(np.percentile(values, percentile))
    
    # Add scipy-based metrics if available
    if _SCIPY_AVAILABLE:
        metrics["skewness"] = float(stats.skew(values))
        metrics["kurtosis"] = float(stats.kurtosis(values))
    
    return metrics


def compute_metrics(
    outputs: Dict[str, torch.Tensor],
    targets: Optional[torch.Tensor] = None,
    metric_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Compute multiple metrics from model outputs.
    
    This is a convenience function that computes various metrics based on
    what's available in the outputs dictionary.
    
    Args:
        outputs: Dictionary of model outputs (logits, values, rewards, etc.)
        targets: Optional target labels
        metric_names: List of metric names to compute (if None, computes all)
        
    Returns:
        Dictionary of computed metrics
        
    Example:
        ```python
        outputs = {
            "logits": logits,
            "values": values,
            "rewards": rewards,
            "log_probs": log_probs,
        }
        metrics = compute_metrics(outputs, targets=labels)
        ```
    """
    metrics = {}
    
    # Determine which metrics to compute
    if metric_names is None:
        metric_names = ["all"]
    
    compute_all = "all" in metric_names
    
    # Accuracy
    if (compute_all or "accuracy" in metric_names) and "logits" in outputs and targets is not None:
        metrics["accuracy"] = compute_accuracy(outputs["logits"], targets)
    
    # Perplexity
    if (compute_all or "perplexity" in metric_names) and "loss" in outputs:
        metrics["perplexity"] = compute_perplexity(outputs["loss"])
    
    # Entropy
    if (compute_all or "entropy" in metric_names) and "logits" in outputs:
        metrics["entropy"] = compute_policy_entropy(outputs["logits"]).item()
    
    # Reward statistics
    if (compute_all or "reward_stats" in metric_names) and "rewards" in outputs:
        reward_stats = compute_statistical_metrics(outputs["rewards"])
        metrics.update({f"reward_{k}": v for k, v in reward_stats.items()})
    
    # Value statistics
    if (compute_all or "value_stats" in metric_names) and "values" in outputs:
        value_stats = compute_statistical_metrics(outputs["values"])
        metrics.update({f"value_{k}": v for k, v in value_stats.items()})
    
    # KL divergence
    if (compute_all or "kl_div" in metric_names) and "log_probs" in outputs and "ref_log_probs" in outputs:
        metrics["kl_div"] = compute_kl_divergence(
            outputs["log_probs"],
            outputs["ref_log_probs"]
        ).item()
    
    # Clip fraction
    if (compute_all or "clip_fraction" in metric_names) and "ratio" in outputs:
        metrics["clip_fraction"] = compute_clip_fraction(outputs["ratio"])
    
    # Explained variance
    if (compute_all or "explained_variance" in metric_names) and "values" in outputs and "returns" in outputs:
        metrics["explained_variance"] = compute_explained_variance(
            outputs["values"],
            outputs["returns"]
        )
    
    return metrics


# Public API
__all__ = [
    "MetricsTracker",
    "compute_reward",
    "compute_kl_divergence",
    "compute_advantages",
    "compute_returns",
    "compute_policy_entropy",
    "compute_accuracy",
    "compute_perplexity",
    "compute_clip_fraction",
    "compute_explained_variance",
    "aggregate_metrics",
    "compute_group_metrics",
    "compute_ranking_metrics",
    "compute_statistical_metrics",
    "compute_metrics",
]