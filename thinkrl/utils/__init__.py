"""
ThinkRL Utilities Module
=========================

Core utility functions and helpers for ThinkRL.
Aligned with OpenRLHF patterns for RLHF training.

Available modules:
- logging: Logging setup and utilities
- metrics: Metrics computation and tracking
- checkpoint: Model checkpoint management
- datasets: Data loading and processing utilities
- tokenizer: Tokenization helpers
- distributed_util: Distributed training utilities
- distributed_sampler: Distributed data sampling
- seqlen_balancing: Sequence length balancing algorithms
- processor: Data processing pipelines
- agent: Agent utilities for agentic RLHF
- remote_rm_utils: Remote reward model utilities

Author: Archit Sood @ EllanorAI
"""

# Checkpoint utilities
# Agent utilities
from .agent import (
    AgentExecutorBase,
    AgentInstanceBase,
    AgentState,
    create_agent_remote,
    load_agent_class,
)
from .checkpoint import (
    CheckpointManager,
    load_checkpoint,
    load_config,
    save_checkpoint,
    save_config,
)

# Dataset utilities
from .datasets import (
    BatchEncoding,
    apply_chat_template,
    collate_batch,
    compute_sequence_lengths,
    convert_token_to_id,
    create_attention_mask,
    create_causal_mask,
    create_dataloader,
    create_labels_for_clm,
    create_position_ids,
    get_strategy,
    mask_padding_in_loss,
    pad_sequences,
    prepare_batch_for_training,
    preprocess_text,
    remove_pad_token,
    shuffle_batch,
    split_batch,
    to_device,
    truncate_sequence,
    zero_pad_sequences,
)

# Distributed sampler
from .distributed_sampler import (
    DistributedBatchSampler,
    DistributedSampler,
    SequentialDistributedSampler,
)

# Distributed utilities
from .distributed_util import (
    all_gather,
    all_reduce,
    barrier,
    broadcast,
    broadcast_object,
    cleanup_distributed,
    gather_object,
    get_local_rank,
    get_rank,
    get_world_size,
    init_distributed,
    is_distributed,
    is_main_process,
    reduce_mean,
    stateless_init_process_group,
    torch_dist_barrier_and_cuda_sync,
)

# Logging utilities
from .logging import (
    ColoredFormatter,
    NewLineFormatter,
    ThinkRLLogger,
    configure_logging_for_distributed,
    disable_external_loggers,
    get_logger,
    init_logger,
    setup_logger,
)

# Metrics utilities
from .metrics import (
    MetricsTracker,
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
)

# Processor utilities
from .processor import (
    PROCESSORS,
    best_of_n_processor,
    conditional_sft_processor,
    create_pairwise_data,
    filter_by_reward_threshold,
    get_processor,
    iterative_dpo_processor,
    register_processor,
    rejection_sampling_processor,
    reward_normalization,
)

# Remote reward model utilities
from .remote_rm_utils import (
    RemoteRewardModel,
    create_reward_server_handler,
    request_api_wrapper,
)

# Sequence length balancing
from .seqlen_balancing import (
    ceildiv,
    get_minimum_num_micro_batch_size,
    get_reverse_idx,
    get_seqlen_balanced_partitions,
    greedy_partition,
    karmarkar_karp,
    log_seqlen_unbalance,
    reorder_by_seqlen,
)

# Tokenizer utilities
from .tokenizer import (
    TokenizerConfig,
    add_special_tokens,
    count_tokens,
    decode_tokens,
    get_special_tokens,
    get_tokenizer,
    get_tokenizer_info,
    load_tokenizer,
    prepare_input_for_generation,
    save_tokenizer,
    tokenize_batch,
    tokenize_conversation,
    tokenize_text,
    truncate_to_token_limit,
)


__all__ = [
    # Logging
    "setup_logger",
    "get_logger",
    "init_logger",
    "configure_logging_for_distributed",
    "disable_external_loggers",
    "ColoredFormatter",
    "NewLineFormatter",
    "ThinkRLLogger",
    # Checkpointing
    "CheckpointManager",
    "save_checkpoint",
    "load_checkpoint",
    "save_config",
    "load_config",
    # Metrics
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
    # Data utilities
    "BatchEncoding",
    "pad_sequences",
    "create_attention_mask",
    "create_position_ids",
    "create_causal_mask",
    "collate_batch",
    "create_dataloader",
    "preprocess_text",
    "truncate_sequence",
    "create_labels_for_clm",
    "mask_padding_in_loss",
    "split_batch",
    "compute_sequence_lengths",
    "shuffle_batch",
    "to_device",
    "prepare_batch_for_training",
    "zero_pad_sequences",
    "remove_pad_token",
    "convert_token_to_id",
    "get_strategy",
    "apply_chat_template",
    # Tokenizers
    "TokenizerConfig",
    "get_tokenizer",
    "tokenize_text",
    "tokenize_batch",
    "decode_tokens",
    "get_special_tokens",
    "add_special_tokens",
    "tokenize_conversation",
    "prepare_input_for_generation",
    "count_tokens",
    "truncate_to_token_limit",
    "get_tokenizer_info",
    "save_tokenizer",
    "load_tokenizer",
    # Distributed utilities
    "torch_dist_barrier_and_cuda_sync",
    "barrier",
    "init_distributed",
    "cleanup_distributed",
    "get_rank",
    "get_world_size",
    "get_local_rank",
    "is_main_process",
    "is_distributed",
    "all_reduce",
    "all_gather",
    "broadcast",
    "reduce_mean",
    "gather_object",
    "broadcast_object",
    "stateless_init_process_group",
    # Distributed samplers
    "DistributedSampler",
    "DistributedBatchSampler",
    "SequentialDistributedSampler",
    # Sequence length balancing
    "karmarkar_karp",
    "greedy_partition",
    "get_seqlen_balanced_partitions",
    "log_seqlen_unbalance",
    "get_minimum_num_micro_batch_size",
    "reorder_by_seqlen",
    "ceildiv",
    "get_reverse_idx",
    # Processors
    "reward_normalization",
    "conditional_sft_processor",
    "rejection_sampling_processor",
    "iterative_dpo_processor",
    "best_of_n_processor",
    "filter_by_reward_threshold",
    "create_pairwise_data",
    "PROCESSORS",
    "get_processor",
    "register_processor",
    # Agent utilities
    "AgentState",
    "AgentInstanceBase",
    "AgentExecutorBase",
    "load_agent_class",
    "create_agent_remote",
    # Remote reward model
    "request_api_wrapper",
    "RemoteRewardModel",
    "create_reward_server_handler",
]
