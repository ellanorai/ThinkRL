"""
Distributed Training Utilities
==============================

Helper functions for distributed training with DeepSpeed.

Author: EllanorAI
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn


def is_deepspeed_available() -> bool:
    """Check if DeepSpeed is available."""
    try:
        import deepspeed  # noqa: F401

        return True
    except ImportError:
        return False


def get_deepspeed_config(path: str | None = None) -> dict[str, Any] | None:
    """
    Get DeepSpeed config from environment or path.

    Args:
        path: Optional path to config JSON

    Returns:
        Config dictionary or None
    """
    import json

    if path is not None:
        with open(path) as f:
            return json.load(f)

    # Check environment variable
    config_path = os.environ.get("DEEPSPEED_CONFIG")
    if config_path is not None:
        with open(config_path) as f:
            return json.load(f)

    return None


def reduce_tensor(
    tensor: torch.Tensor,
    world_size: int | None = None,
    reduce_op: str = "mean",
) -> torch.Tensor:
    """
    Reduce tensor across all processes.

    Args:
        tensor: Tensor to reduce
        world_size: World size (auto-detected if None)
        reduce_op: Reduction operation ("mean", "sum", "max", "min")

    Returns:
        Reduced tensor

    Raises:
        ValueError: If reduce_op is not one of "mean", "sum", "max", "min"
    """
    # Map string op to torch op - validate before checking dist
    op_map = {
        "mean": dist.ReduceOp.SUM,  # Will divide after
        "sum": dist.ReduceOp.SUM,
        "max": dist.ReduceOp.MAX,
        "min": dist.ReduceOp.MIN,
    }

    if reduce_op not in op_map:
        raise ValueError(f"Unknown reduce_op: {reduce_op}. Must be one of: {list(op_map.keys())}")

    if not dist.is_initialized():
        return tensor

    if world_size is None:
        world_size = dist.get_world_size()

    # Clone to avoid modifying original
    tensor = tensor.clone()

    dist.all_reduce(tensor, op=op_map[reduce_op])

    if reduce_op == "mean":
        tensor = tensor / world_size

    return tensor


def broadcast_tensor(
    tensor: torch.Tensor,
    src: int = 0,
) -> torch.Tensor:
    """
    Broadcast tensor from source rank to all ranks.

    Args:
        tensor: Tensor to broadcast
        src: Source rank

    Returns:
        Broadcasted tensor
    """
    if not dist.is_initialized():
        return tensor

    dist.broadcast(tensor, src=src)
    return tensor


def all_gather_tensors(
    tensor: torch.Tensor,
    world_size: int | None = None,
) -> list[torch.Tensor]:
    """
    Gather tensors from all processes.

    Args:
        tensor: Tensor to gather
        world_size: World size (auto-detected if None)

    Returns:
        List of tensors from all ranks
    """
    if not dist.is_initialized():
        return [tensor]

    if world_size is None:
        world_size = dist.get_world_size()

    # Create output list
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor)

    return tensor_list


def sync_model_params(model: nn.Module, src: int = 0) -> None:
    """
    Synchronize model parameters from source rank.

    Args:
        model: Model to synchronize
        src: Source rank
    """
    if not dist.is_initialized():
        return

    for param in model.parameters():
        dist.broadcast(param.data, src=src)


def get_world_info() -> dict[str, int]:
    """
    Get distributed training info.

    Returns:
        Dictionary with rank, local_rank, world_size
    """
    if not dist.is_initialized():
        return {
            "rank": 0,
            "local_rank": 0,
            "world_size": 1,
        }

    return {
        "rank": dist.get_rank(),
        "local_rank": int(os.environ.get("LOCAL_RANK", 0)),
        "world_size": dist.get_world_size(),
    }


def is_main_process() -> bool:
    """Check if current process is main process (rank 0)."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def barrier() -> None:
    """Synchronize all processes."""
    if dist.is_initialized():
        dist.barrier()


def setup_distributed_logging() -> None:
    """Configure logging for distributed training (only main process logs)."""
    import logging

    if not is_main_process():
        logging.disable(logging.CRITICAL)


def print_rank_0(*args, **kwargs) -> None:
    """Print only on rank 0."""
    if is_main_process():
        pass


def gather_scalar(value: float | int, world_size: int | None = None) -> list[float]:
    """
    Gather scalar values from all processes.

    Args:
        value: Scalar value
        world_size: World size

    Returns:
        List of values from all ranks
    """
    if not dist.is_initialized():
        return [float(value)]

    if world_size is None:
        world_size = dist.get_world_size()

    tensor = torch.tensor([value], dtype=torch.float32)
    if torch.cuda.is_available():
        tensor = tensor.cuda()

    gathered = all_gather_tensors(tensor, world_size)
    return [t.item() for t in gathered]


def compute_global_metrics(
    local_metrics: dict[str, float],
    world_size: int | None = None,
) -> dict[str, float]:
    """
    Compute global metrics by averaging across processes.

    Args:
        local_metrics: Local metric dictionary
        world_size: World size

    Returns:
        Averaged metrics
    """
    if not dist.is_initialized():
        return local_metrics

    if world_size is None:
        world_size = dist.get_world_size()

    global_metrics = {}
    for key, value in local_metrics.items():
        if isinstance(value, (int, float)):
            tensor = torch.tensor([value], dtype=torch.float32)
            if torch.cuda.is_available():
                tensor = tensor.cuda()
            tensor = reduce_tensor(tensor, world_size, "mean")
            global_metrics[key] = tensor.item()
        else:
            global_metrics[key] = value

    return global_metrics


__all__ = [
    "is_deepspeed_available",
    "get_deepspeed_config",
    "reduce_tensor",
    "broadcast_tensor",
    "all_gather_tensors",
    "sync_model_params",
    "get_world_info",
    "is_main_process",
    "barrier",
    "setup_distributed_logging",
    "print_rank_0",
    "gather_scalar",
    "compute_global_metrics",
]
