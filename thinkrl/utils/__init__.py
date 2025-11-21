"""
ThinkRL Utilities Module
=========================

Core utility functions and helpers for ThinkRL.

Available modules:
- logging: Logging setup and utilities
- metrics: Metrics computation and tracking
- checkpoint: Model checkpoint management
- data: Data loading and processing utilities
- tokenizers: Tokenization helpers

Author: Archit Sood @ EllanorAI
"""

from .checkpoint import (
    CheckpointManager,
    load_checkpoint,
    load_config,
    save_checkpoint,
    save_config,
)
from .datasets import (
    BatchEncoding,
    collate_batch,
    compute_sequence_lengths,
    create_attention_mask,
    create_causal_mask,
    create_dataloader,
    create_labels_for_clm,
    create_position_ids,
    mask_padding_in_loss,
    pad_sequences,
    prepare_batch_for_training,
    preprocess_text,
    shuffle_batch,
    split_batch,
    to_device,
    truncate_sequence,
)
from .logging import (
    ColoredFormatter,
    ThinkRLLogger,
    configure_logging_for_distributed,
    disable_external_loggers,
    get_logger,
    setup_logger,
)
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
    "configure_logging_for_distributed",
    "disable_external_loggers",
    "ColoredFormatter",
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
]
