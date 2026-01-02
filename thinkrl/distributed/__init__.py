"""
ThinkRL Distributed Training
============================

DeepSpeed integration for distributed RLHF training.

Provides:
- DeepSpeed engine wrapper
- ZeRO-2 and ZeRO-3 strategies
- Distributed utilities

Author: EllanorAI
"""

from thinkrl.distributed.deepspeed import (
    DeepSpeedEngine,
    create_deepspeed_config,
    init_deepspeed,
)
from thinkrl.distributed.strategies import (
    DeepSpeedStrategy,
    ZeRO1Strategy,
    ZeRO2Strategy,
    ZeRO3Strategy,
    get_strategy,
)
from thinkrl.distributed.utils import (
    all_gather_tensors,
    broadcast_tensor,
    get_deepspeed_config,
    is_deepspeed_available,
    reduce_tensor,
    sync_model_params,
)


__all__ = [
    # Engine
    "DeepSpeedEngine",
    "init_deepspeed",
    "create_deepspeed_config",
    # Strategies
    "DeepSpeedStrategy",
    "ZeRO1Strategy",
    "ZeRO2Strategy",
    "ZeRO3Strategy",
    "get_strategy",
    # Utils
    "is_deepspeed_available",
    "get_deepspeed_config",
    "reduce_tensor",
    "broadcast_tensor",
    "all_gather_tensors",
    "sync_model_params",
]
