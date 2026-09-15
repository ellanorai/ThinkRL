"""Data-parallel helpers for the RL trainers.

`grep -rn "deepspeed\\|DistributedDataParallel" thinkrl/training/` used to match
`sft_trainer.py` only: GRPOTrainer, ReinforcePPTrainer and STaRTrainer built a model, moved
it to one device and stepped a plain optimizer. The README recommends GRPO for reasoning
work, and GRPO was the algorithm with no distributed path at all. See #128.

This file was empty and is the intended home for that.

The awkward part of data-parallel RL, and the reason this is two functions rather than one
wrap call: `DistributedDataParallel` gives gradient synchronisation through ``forward``,
but does not proxy ``generate``. A rollout needs both, so the trainers hold the wrapper for
the loss path and unwrap for generation and for anything that reads the weights, such as
checkpointing and the vLLM weight push.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from thinkrl.utils.logging import get_logger


logger = get_logger(__name__)


def is_distributed() -> bool:
    """True when a process group is initialized and holds more than one rank."""
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def unwrap_model(model: nn.Module) -> nn.Module:
    """The underlying module, whether or not it is wrapped.

    Safe to call unconditionally, which is the point: call sites should not have to know
    whether the run is distributed.
    """
    return model.module if isinstance(model, DistributedDataParallel) else model


def wrap_policy(model: nn.Module, device: torch.device | str | None = None) -> nn.Module:
    """Wrap a policy in DDP when running distributed, otherwise return it unchanged.

    ``find_unused_parameters`` is left off. It costs an extra graph traversal every step,
    and an RL policy uses all of its parameters on every forward; turning it on to silence
    an error would hide a real bug in the loss rather than fix it.
    """
    if not is_distributed():
        return model

    if isinstance(model, DistributedDataParallel):
        return model

    device_ids = None
    if device is not None and torch.device(device).type == "cuda":
        device_ids = [torch.device(device).index or 0]

    wrapped = DistributedDataParallel(model, device_ids=device_ids)
    logger.info(
        "Policy wrapped in DistributedDataParallel (rank %d of %d)",
        dist.get_rank(),
        dist.get_world_size(),
    )
    return wrapped
