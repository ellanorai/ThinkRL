"""
ThinkRL Metrics Utilities
==========================

Comprehensive metrics computation for RLHF training.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict
import warnings

import torch
import torch.nn.functional as F
import numpy as np
import torch.utils.dlpack

# Handle CuPy import failure gracefully (e.g., no GPU/CUDA)
try:
    import cupy as cp
    try:
        from cupyx.scipy import stats as cupy_stats
        _CUPY_SCIPY_AVAILABLE = True
    except ImportError:
        _CUPY_SCIPY_AVAILABLE = False
    _CUPY_AVAILABLE = True
except (ImportError, OSError):
    # ImportError: Package not installed
    # OSError: Shared library (libcuda.so) not found
    cp = None
    _CUPY_AVAILABLE = False
    _CUPY_SCIPY_AVAILABLE = False

try:
    from scipy import stats as scipy_stats
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)

class MetricsTracker:
    """
    Track and aggregate metrics over training.
    """

    def __init__(self):
        """Initialize the metrics tracker."""
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.current_values: Dict[str, float] = {}

    def update(self, name: str, value: Union[float, torch.Tensor], count: int = 1):
        """
        Update a metric with a new value.
        """
        # Convert tensor to float
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().item()

        # Store value
        self.metrics[name].append((value, count))
        self.current_values[name] = value

    def update_dict(self, metrics: Dict[str, Union[float, torch.Tensor]]):
        """Update multiple metrics at once."""
        for name, value in metrics.items():
            self.update(name, value)

    def get_current(self, name: Optional[str] = None) -> Union[float, Dict[str, float]]:
        """Get current (most recent) metric value(s)."""
        if name is not None:
            return self.current_values.get(name, 0.0)
        return self.current_values.copy()

    def get_average(self, name: Optional[str] = None) -> Union[float, Dict[str, float]]:
        """Get average metric value(s) over all updates."""
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
        """Get full history of a metric."""
        if name not in self.metrics:
            return []
        return [value for value, _ in self.metrics[name]]

    def get_summary(self, name: str) -> Dict[str, float]:
        """Get statistical summary of a metric."""
        history = self.get_history(name)
        if not history:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

        # OPTIMIZATION: Use NumPy here.
        return {
            "mean": float(np.mean(history)),
            "std": float(np.std(history)),
            "min": float(np.min(history)),
            "max": float(np.max(history)),
        }

    def reset(self, name: Optional[str] = None):
        """Reset tracked metrics."""
        if name is not None:
            if name in self.metrics:
                self.metrics[name].clear()
            if name in self.current_values:
                del self.current_values[name]
        else:
            self.metrics.clear()
            self.current_values.clear()

    def __repr__(self) -> str:
        avg_metrics = self.get_average()
        return f"MetricsTracker({avg_metrics})"


def compute_reward(
    rewards: torch.Tensor, normalize: bool = True, epsilon: float = 1e-8
) -> torch.Tensor:
    """Compute and optionally normalize rewards."""
    if normalize:
        mean = rewards.mean()
        std = rewards.std() + epsilon
        rewards = (rewards - mean) / std

    return rewards


def compute_kl_divergence(
    log_probs_policy: torch.Tensor, log_probs_ref: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    """Compute KL divergence between policy and reference distributions."""
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
    normalize: bool = True,
) -> torch.Tensor:
    """Compute Generalized Advantage Estimation (GAE)."""
    batch_size, seq_len = rewards.shape
    advantages = torch.zeros_like(rewards)

    # Compute TD residuals
    next_values = torch.cat(
        [values[:, 1:], torch.zeros(batch_size, 1, device=values.device)], dim=1
    )
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
    rewards: torch.Tensor, gamma: float = 0.99, normalize: bool = False
) -> torch.Tensor:
    """Compute discounted returns."""
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
    logits: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    """Compute entropy of policy distribution."""
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
    predictions: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100
) -> float:
    """Compute accuracy of predictions."""
    # If predictions are logits, get argmax
    if predictions.dim() > targets.dim():
        predictions = predictions.argmax(dim=-1)

    # Create mask for valid positions
    mask = targets != ignore_index

    # Compute accuracy
    correct = (predictions == targets) & mask
    if mask.sum() == 0:
        return 0.0
    accuracy = correct.sum().float() / mask.sum().float()

    return accuracy.item()


def compute_perplexity(loss: Union[float, torch.Tensor]) -> float:
    """Compute perplexity from cross-entropy loss."""
    if isinstance(loss, torch.Tensor):
        loss = loss.item()
    return float(np.exp(loss))


def compute_clip_fraction(ratio: torch.Tensor, epsilon: float = 0.2) -> float:
    """Compute fraction of ratios that were clipped in PPO."""
    clipped = ((ratio < 1 - epsilon) | (ratio > 1 + epsilon)).float()
    return clipped.mean().item()


def compute_explained_variance(
    predictions: torch.Tensor, targets: torch.Tensor
) -> float:
    """Compute explained variance for value function."""
    var_y = targets.var()
    if var_y == 0:
        return 0.0

    return 1.0 - (targets - predictions).var() / var_y


def aggregate_metrics(
    metrics_list: List[Dict[str, float]], weights: Optional[List[float]] = None
) -> Dict[str, float]:
    """Aggregate metrics from multiple batches or processes."""
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
    rewards: torch.Tensor, group_ids: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """Compute metrics per group."""
    unique_groups = torch.unique(group_ids)

    group_means = []
    group_stds = []
    group_maxs = []
    group_mins = []

    for group_id in unique_groups:
        mask = group_ids == group_id
        group_rewards = rewards[mask]

        group_means.append(group_rewards.mean())
        
        # Handle groups with 1 element
        if group_rewards.numel() > 1:
            group_stds.append(group_rewards.std())
        else:
            group_stds.append(torch.tensor(0.0, device=rewards.device, dtype=rewards.dtype))
            
        group_maxs.append(group_rewards.max())
        group_mins.append(group_rewards.min())

    return {
        "group_means": torch.stack(group_means),
        "group_stds": torch.stack(group_stds),
        "group_maxs": torch.stack(group_maxs),
        "group_mins": torch.stack(group_mins),
    }


def compute_ranking_metrics(
    scores: torch.Tensor, labels: torch.Tensor, k: int = 10
) -> Dict[str, float]:
    """Compute ranking metrics (useful for preference learning)."""
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


# -----------------------------------------------------------------------------
# Optimized Statistical Metrics (GPU-accelerated)
# -----------------------------------------------------------------------------

def compute_statistical_metrics(
    values: Union[torch.Tensor, np.ndarray, List[float], float, None]
) -> Dict[str, float]:
    """
    Compute comprehensive statistical metrics with GPU acceleration when available.
    
    Intelligently uses CuPy for GPU tensors and NumPy for CPU data to optimize performance.
    Handles edge cases gracefully including empty, NaN, and infinite values.
    
    Args:
        values: Input data as tensor, array, list, scalar, or None
        
    Returns:
        Dictionary containing statistical metrics (mean, std, min, max, median, percentiles, etc.)
        Returns zeros for invalid/empty inputs.
    """
    # Define default metrics to ensure consistent output
    default_metrics = {
        "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0,
        "p25": 0.0, "p75": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0,
        "skewness": 0.0, "kurtosis": 0.0, "count": 0, "nan_count": 0
    }
    
    # Early return for invalid inputs
    if values is None:
        return default_metrics
        
    # Convert to numpy/cupy array with proper error handling
    data, xp = _prepare_array(values)
    
    if data is None:
        return default_metrics
    
    # Ensure at least 1D
    if data.ndim == 0:
        data = xp.reshape(data, (1,))
    
    # Flatten for statistics (most stats don't care about shape)
    data = xp.ravel(data)
    
    # Check for empty array
    if data.size == 0:
        return default_metrics
    
    # Handle NaN and Inf values
    metrics = {"count": int(data.size)}
    
    # Count and filter NaN/Inf
    if xp == cp and _CUPY_AVAILABLE:
        nan_mask = cp.isnan(data)
        inf_mask = cp.isinf(data)
        valid_mask = ~(nan_mask | inf_mask)
        metrics["nan_count"] = int(cp.sum(nan_mask))
        metrics["inf_count"] = int(cp.sum(inf_mask))
        valid_data = data[valid_mask]
    else:
        nan_mask = np.isnan(data)
        inf_mask = np.isinf(data)
        valid_mask = ~(nan_mask | inf_mask)
        metrics["nan_count"] = int(np.sum(nan_mask))
        metrics["inf_count"] = int(np.sum(inf_mask))
        valid_data = data[valid_mask]
    
    # If no valid data, return defaults with counts
    if valid_data.size == 0:
        return {**default_metrics, **metrics}
    
    # Compute basic statistics efficiently
    try:
        metrics.update(_compute_basic_stats(valid_data, xp))
        metrics.update(_compute_percentiles(valid_data, xp))
        metrics.update(_compute_higher_moments(valid_data, xp))
    except Exception as e:
        logger.debug(f"Error computing statistics: {e}")
        # Return what we have so far
        return {**default_metrics, **metrics}
    
    return metrics


def _prepare_array(
    values: Union[torch.Tensor, np.ndarray, List[float], float]
) -> Tuple[Optional[Union[np.ndarray, 'cp.ndarray']], Any]:
    """
    Convert input to appropriate array type (CuPy for GPU, NumPy for CPU).
    
    Returns:
        Tuple of (array, module) where module is either np or cp
    """
    try:
        # Handle torch tensors
        if isinstance(values, torch.Tensor):
            if values.is_cuda and _CUPY_AVAILABLE:
                # GPU tensor -> CuPy (zero-copy via DLPack)
                try:
                    data = cp.from_dlpack(torch.utils.dlpack.to_dlpack(values))
                    return data, cp
                except Exception as e:
                    logger.debug(f"DLPack conversion failed: {e}, falling back to CPU")
                    data = values.detach().cpu().numpy()
                    return data, np
            else:
                # CPU tensor -> NumPy
                data = values.detach().cpu().numpy()
                return data, np
        
        # Handle CuPy arrays
        if _CUPY_AVAILABLE and isinstance(values, cp.ndarray):
            return values, cp
        
        # Handle NumPy arrays
        if isinstance(values, np.ndarray):
            return values, np
        
        # Convert lists and scalars to NumPy
        try:
            data = np.asarray(values, dtype=np.float64)
            return data, np
        except Exception as e:
            logger.warning(f"Failed to convert input to array: {e}")
            return None, np
            
    except Exception as e:
        logger.warning(f"Array preparation failed: {e}")
        return None, np


def _compute_basic_stats(
    data: Union[np.ndarray, 'cp.ndarray'], 
    xp: Any
) -> Dict[str, float]:
    """Compute basic statistical measures."""
    stats = {}
    
    # Use appropriate functions based on array library
    if data.size == 1:
        # Special case for single element
        val = float(data.item() if hasattr(data, 'item') else data[0])
        stats.update({
            "mean": val,
            "std": 0.0,
            "min": val,
            "max": val,
            "median": val,
            "variance": 0.0
        })
    else:
        # Compute statistics in one pass where possible
        stats["mean"] = float(xp.mean(data))
        stats["std"] = float(xp.std(data, ddof=1))  # Use sample std
        stats["variance"] = float(xp.var(data, ddof=1))
        stats["min"] = float(xp.min(data))
        stats["max"] = float(xp.max(data))
        stats["median"] = float(xp.median(data))
        
        # Additional useful stats
        stats["range"] = stats["max"] - stats["min"]
        stats["cv"] = stats["std"] / abs(stats["mean"]) if stats["mean"] != 0 else 0.0
    
    return stats


def _compute_percentiles(
    data: Union[np.ndarray, 'cp.ndarray'], 
    xp: Any
) -> Dict[str, float]:
    """Compute percentile statistics efficiently."""
    percentiles = [25, 75, 90, 95, 99]
    
    # Compute all percentiles in one call for efficiency
    if data.size == 1:
        val = float(data.item() if hasattr(data, 'item') else data[0])
        return {f"p{p}": val for p in percentiles}
    
    try:
        if xp == cp and _CUPY_AVAILABLE:
            # CuPy percentile computation
            results = cp.percentile(data, percentiles)
            return {f"p{p}": float(results[i]) for i, p in enumerate(percentiles)}
        else:
            # NumPy percentile computation  
            results = np.percentile(data, percentiles)
            return {f"p{p}": float(results[i]) for i, p in enumerate(percentiles)}
    except Exception as e:
        logger.debug(f"Percentile computation failed: {e}")
        return {f"p{p}": 0.0 for p in percentiles}


def _compute_higher_moments(
    data: Union[np.ndarray, 'cp.ndarray'], 
    xp: Any
) -> Dict[str, float]:
    """Compute skewness and kurtosis with appropriate libraries."""
    moments = {"skewness": 0.0, "kurtosis": 0.0}
    
    # Need at least 3 elements for skewness, 4 for kurtosis
    if data.size < 3:
        return moments
    
    try:
        if xp == cp and _CUPY_AVAILABLE and _CUPY_SCIPY_AVAILABLE:
            # Use CuPy's scipy stats
            from cupyx.scipy import stats as cupy_stats
            
            skew = cupy_stats.skew(data, axis=None, nan_policy='omit')
            moments["skewness"] = float(skew.item() if hasattr(skew, 'item') else skew)
            
            if data.size >= 4:
                kurt = cupy_stats.kurtosis(data, axis=None, nan_policy='omit')
                moments["kurtosis"] = float(kurt.item() if hasattr(kurt, 'item') else kurt)
                
        elif xp == np and _SCIPY_AVAILABLE:
            # Use SciPy stats
            from scipy import stats as scipy_stats
            
            skew = scipy_stats.skew(data, axis=None, nan_policy='omit')
            moments["skewness"] = float(skew)
            
            if data.size >= 4:
                kurt = scipy_stats.kurtosis(data, axis=None, nan_policy='omit')
                moments["kurtosis"] = float(kurt)
        else:
            # Manual computation as fallback
            moments.update(_compute_moments_manual(data, xp))
            
    except Exception as e:
        logger.debug(f"Higher moments computation failed: {e}")
    
    return moments


def _compute_moments_manual(
    data: Union[np.ndarray, 'cp.ndarray'], 
    xp: Any
) -> Dict[str, float]:
    """Manually compute skewness and kurtosis without scipy."""
    try:
        mean = xp.mean(data)
        std = xp.std(data, ddof=1)
        
        if std == 0:
            return {"skewness": 0.0, "kurtosis": 0.0}
        
        n = data.size
        
        # Standardized moments
        z = (data - mean) / std
        
        # Skewness (third moment)
        if n >= 3:
            m3 = xp.mean(z**3)
            # Fisher-Pearson coefficient with bias correction
            skewness = float(m3 * xp.sqrt(n * (n - 1)) / (n - 2))
        else:
            skewness = 0.0
        
        # Kurtosis (fourth moment)
        if n >= 4:
            m4 = xp.mean(z**4)
            # Excess kurtosis with bias correction
            kurtosis = float((m4 - 3) * (n + 1) * n / ((n - 1) * (n - 2) * (n - 3)))
        else:
            kurtosis = 0.0
        
        return {"skewness": skewness, "kurtosis": kurtosis}
        
    except Exception as e:
        logger.debug(f"Manual moments computation failed: {e}")
        return {"skewness": 0.0, "kurtosis": 0.0}


def compute_statistical_metrics_batch(
    values_list: List[Union[torch.Tensor, np.ndarray, List[float]]]
) -> List[Dict[str, float]]:
    """
    Compute statistics for multiple arrays efficiently.
    
    Useful for computing metrics across different groups or batches.
    """
    if not values_list:
        return []
    
    # Process in parallel if using CuPy (GPU operations are async)
    results = []
    for values in values_list:
        results.append(compute_statistical_metrics(values))
    
    return results


def compute_metrics(
    outputs: Dict[str, torch.Tensor],
    targets: Optional[torch.Tensor] = None,
    metric_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Compute multiple metrics from model outputs."""
    metrics = {}

    if metric_names is None:
        metric_names = ["all"]

    compute_all = "all" in metric_names

    if (compute_all or "accuracy" in metric_names) and "logits" in outputs and targets is not None:
        metrics["accuracy"] = compute_accuracy(outputs["logits"], targets)

    if (compute_all or "perplexity" in metric_names) and "loss" in outputs:
        metrics["perplexity"] = compute_perplexity(outputs["loss"])

    if (compute_all or "entropy" in metric_names) and "logits" in outputs:
        metrics["entropy"] = compute_policy_entropy(outputs["logits"]).item()

    if (compute_all or "reward_stats" in metric_names) and "rewards" in outputs:
        reward_stats = compute_statistical_metrics(outputs["rewards"])
        metrics.update({f"reward_{k}": v for k, v in reward_stats.items()})

    if (compute_all or "value_stats" in metric_names) and "values" in outputs:
        value_stats = compute_statistical_metrics(outputs["values"])
        metrics.update({f"value_{k}": v for k, v in value_stats.items()})

    if (compute_all or "kl_div" in metric_names) and "log_probs" in outputs and "ref_log_probs" in outputs:
        metrics["kl_div"] = compute_kl_divergence(outputs["log_probs"], outputs["ref_log_probs"]).item()

    if (compute_all or "clip_fraction" in metric_names) and "ratio" in outputs:
        metrics["clip_fraction"] = compute_clip_fraction(outputs["ratio"])

    if (compute_all or "explained_variance" in metric_names) and "values" in outputs and "returns" in outputs:
        metrics["explained_variance"] = compute_explained_variance(outputs["values"], outputs["returns"])

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
    "compute_statistical_metrics_batch",
    "compute_metrics",
]