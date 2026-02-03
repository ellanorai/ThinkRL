"""
ThinkRL Distributed Training Utilities
=======================================

Distributed training utilities aligned with OpenRLHF patterns.
Provides synchronization primitives and process group management.

Author: Archit Sood @ EllanorAI
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
import torch.distributed as dist


logger = logging.getLogger(__name__)


def torch_dist_barrier_and_cuda_sync() -> None:
    """
    Synchronize all distributed processes and CUDA operations.

    This function ensures all distributed training processes reach the same point
    and all CUDA operations are completed before proceeding. Useful for
    coordinating checkpointing, logging, and other collective operations.

    Example:
        ```python
        # Before checkpointing
        torch_dist_barrier_and_cuda_sync()
        if is_main_process():
            save_checkpoint(model, path)
        torch_dist_barrier_and_cuda_sync()
        ```
    """
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def init_distributed(
    backend: str = "nccl",
    init_method: str | None = None,
    world_size: int | None = None,
    rank: int | None = None,
) -> bool:
    """
    Initialize distributed training environment.

    Args:
        backend: Distributed backend ("nccl", "gloo", "mpi")
        init_method: URL for process group initialization
        world_size: Total number of processes
        rank: Rank of current process

    Returns:
        True if distributed training is initialized, False otherwise

    Example:
        ```python
        if init_distributed():
            model = DistributedDataParallel(model)
        ```
    """
    if dist.is_initialized():
        return True

    # Try to get distributed info from environment
    if world_size is None:
        world_size = int(os.environ.get("WORLD_SIZE", 1))

    if rank is None:
        rank = int(os.environ.get("RANK", 0))

    if world_size <= 1:
        logger.info("Single process mode, skipping distributed initialization")
        return False

    try:
        if init_method is None:
            # Use default environment variable initialization
            dist.init_process_group(backend=backend)
        else:
            dist.init_process_group(
                backend=backend,
                init_method=init_method,
                world_size=world_size,
                rank=rank,
            )

        # Set CUDA device for this process
        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(local_rank)

        logger.info(
            f"Distributed training initialized: "
            f"rank={get_rank()}, world_size={get_world_size()}, backend={backend}"
        )
        return True

    except Exception as e:
        logger.warning(f"Failed to initialize distributed training: {e}")
        return False


def get_rank() -> int:
    """
    Get the rank of current process in distributed training.

    Returns:
        Process rank (0 for single-process mode)
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """
    Get the total number of processes in distributed training.

    Returns:
        World size (1 for single-process mode)
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_local_rank() -> int:
    """
    Get the local rank of current process on this node.

    Returns:
        Local rank (0 for single-process mode)
    """
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main_process() -> bool:
    """
    Check if current process is the main process (rank 0).

    Returns:
        True if main process, False otherwise
    """
    return get_rank() == 0


def is_distributed() -> bool:
    """
    Check if distributed training is active.

    Returns:
        True if distributed training is initialized
    """
    return dist.is_available() and dist.is_initialized()


def barrier() -> None:
    """
    Synchronize all processes in distributed training.

    No-op if not in distributed mode.
    """
    if is_distributed():
        dist.barrier()


def all_reduce(
    tensor: torch.Tensor,
    op: dist.ReduceOp = dist.ReduceOp.SUM,
    async_op: bool = False,
) -> torch.Tensor | Any:
    """
    Reduce tensor across all processes.

    Args:
        tensor: Tensor to reduce
        op: Reduction operation (SUM, AVG, MAX, MIN)
        async_op: Whether to perform asynchronously

    Returns:
        Reduced tensor (or async work handle if async_op=True)
    """
    if not is_distributed():
        return tensor

    work = dist.all_reduce(tensor, op=op, async_op=async_op)
    return work if async_op else tensor


def all_gather(tensor: torch.Tensor) -> list[torch.Tensor]:
    """
    Gather tensors from all processes.

    Args:
        tensor: Tensor to gather

    Returns:
        List of tensors from all processes
    """
    if not is_distributed():
        return [tensor]

    world_size = get_world_size()
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor)
    return tensor_list


def broadcast(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """
    Broadcast tensor from source rank to all processes.

    Args:
        tensor: Tensor to broadcast
        src: Source rank

    Returns:
        Broadcasted tensor
    """
    if not is_distributed():
        return tensor

    dist.broadcast(tensor, src=src)
    return tensor


def reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute mean of tensor across all processes.

    Args:
        tensor: Tensor to reduce

    Returns:
        Mean tensor
    """
    if not is_distributed():
        return tensor

    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor = tensor / get_world_size()
    return tensor


def gather_object(obj: Any, dst: int = 0) -> list[Any] | None:
    """
    Gather Python objects from all processes to destination rank.

    Args:
        obj: Object to gather
        dst: Destination rank

    Returns:
        List of objects on destination rank, None on other ranks
    """
    if not is_distributed():
        return [obj]

    world_size = get_world_size()
    rank = get_rank()

    if rank == dst:
        object_list = [None] * world_size
        dist.gather_object(obj, object_list, dst=dst)
        return object_list
    else:
        dist.gather_object(obj, dst=dst)
        return None


def broadcast_object(obj: Any, src: int = 0) -> Any:
    """
    Broadcast Python object from source rank to all processes.

    Args:
        obj: Object to broadcast (only used on source rank)
        src: Source rank

    Returns:
        Broadcasted object
    """
    if not is_distributed():
        return obj

    object_list = [obj] if get_rank() == src else [None]
    dist.broadcast_object_list(object_list, src=src)
    return object_list[0]


# vLLM-specific utilities (optional)
try:
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup

    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False
    PyNcclCommunicator = None
    StatelessProcessGroup = None


def stateless_init_process_group(
    master_address: str,
    master_port: int,
    rank: int,
    world_size: int,
    device: str | torch.device,
) -> Any | None:
    """
    Create a process group for vLLM without relying on PyTorch's global distributed state.

    This is useful for creating separate process groups for vLLM inference
    that don't interfere with the main training process group.

    Args:
        master_address: Host address for rendezvous
        master_port: Port for rendezvous
        rank: Process rank in this group
        world_size: Total processes in this group
        device: Device specification for NCCL communicator

    Returns:
        PyNcclCommunicator instance for inter-process communication, or None if vLLM unavailable

    Example:
        ```python
        comm = stateless_init_process_group(
            master_address="localhost",
            master_port=29500,
            rank=0,
            world_size=2,
            device="cuda:0"
        )
        if comm is not None:
            # Use communicator for vLLM operations
            pass
        ```
    """
    if not _VLLM_AVAILABLE:
        logger.warning("vLLM not available, cannot create stateless process group. " "Install with: pip install vllm")
        return None

    try:
        pg = StatelessProcessGroup.create(
            host=master_address,
            port=master_port,
            rank=rank,
            world_size=world_size,
        )
        pynccl_comm = PyNcclCommunicator(
            group=pg,
            device=device,
        )
        logger.info(f"Created stateless process group: rank={rank}, world_size={world_size}")
        return pynccl_comm

    except Exception as e:
        logger.error(f"Failed to create stateless process group: {e}")
        return None


def cleanup_distributed() -> None:
    """
    Cleanup distributed training resources.

    Should be called at the end of training to properly release resources.
    """
    if is_distributed():
        dist.destroy_process_group()
        logger.info("Distributed process group destroyed")


# Public API
__all__ = [
    # Synchronization
    "torch_dist_barrier_and_cuda_sync",
    "barrier",
    # Initialization
    "init_distributed",
    "cleanup_distributed",
    # Process info
    "get_rank",
    "get_world_size",
    "get_local_rank",
    "is_main_process",
    "is_distributed",
    # Collective operations
    "all_reduce",
    "all_gather",
    "broadcast",
    "reduce_mean",
    "gather_object",
    "broadcast_object",
    # vLLM integration
    "stateless_init_process_group",
]
