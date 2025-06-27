import dataclasses
import gc
import logging
import math
from collections import defaultdict
from typing import Callable, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from data_types import Episode, MiniBatch
from qwen2_model import Transformer
from tokenizer import Tokenizer

# Constants
EPSILON = 1e-4  # Small value to avoid division by zero
ENTROPY_WEIGHT = 0.01  # Weight for entropy regularization
REWARD_CLIP_MIN = -5.0  # Minimum value for reward clipping
REWARD_CLIP_MAX = 5.0  # Maximum value for reward clipping

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@torch.no_grad()
def _generate_tokens(
    model: Transformer,
    tokens: torch.Tensor,
    input_text_mask: torch.Tensor,
    min_prompt_len: int,
    total_len: int,
    end_token_id: int,
    pad_token_id: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates tokens for the rollout process.

    Args:
        model: The transformer model.
        tokens: Tensor of shape (bsz, total_len) containing initial tokens.
        input_text_mask: Mask indicating input text positions.
        min_prompt_len: Minimum length of prompt tokens.
        total_len: Total sequence length (prompt + generated).
        end_token_id: ID of the end-of-sequence token.
        pad_token_id: ID of the padding token.
        device: Device to run the model on.
        dtype: Data type for model computations.

    Returns:
        Tuple of (updated tokens, is_finished flags).
    """
    is_finished = torch.zeros(tokens.shape[0], dtype=torch.bool, device=device)
    prev_pos = 0
    for cur_pos in tqdm(
        range(min_prompt_len, total_len), desc="Generating trajectories"
    ):
        with torch.autocast(device_type=device.type, dtype=dtype):
            logits = model.inference(tokens[:, prev_pos:cur_pos], prev_pos)
        probs = torch.softmax(logits[:, -1], dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).reshape(-1)
        next_token = torch.where(
            input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
        )
        next_token = torch.where(is_finished, pad_token_id, next_token)
        tokens[:, cur_pos] = next_token
        if end_token_id is not None:
            is_end_token = next_token == end_token_id
            is_generated_token = ~input_text_mask[:, cur_pos]
            is_finished = is_finished | (is_end_token & is_generated_token)
        prev_pos = cur_pos
        if is_finished.all():
            break
    return tokens, is_finished


@torch.no_grad()
def rollout(
    model: Transformer,
    batch: MiniBatch,
    tokenizer: Tokenizer,
    max_gen_len: int,
    num_answer_per_question: int,
    reward_function: Callable,
    device: torch.device,
    dtype: torch.dtype,
    debug: bool = False,
) -> List[Episode]:
    """
    Generates rollout trajectories for a given batch using the model.

    Args:
        model: The transformer model to use for generation.
        batch: The input batch containing prefixes and metadata.
        tokenizer: The tokenizer for encoding/decoding tokens.
        max_gen_len: Maximum length of generated sequences.
        num_answer_per_question: Number of answers to generate per question.
        reward_function: Function to compute rewards for generated text.
        device: The device (CPU/GPU) to run the model on.
        dtype: The data type for model computations.
        debug: If True, logs additional debugging information.

    Returns:
        A list of Episode objects containing generated text, rewards, and metadata.

    Raises:
        ValueError: If input batch is empty or max_gen_len is non-positive.
        RuntimeError: If GPU memory is insufficient.
    """
    if not batch.prefix_token_ids:
        raise ValueError("batch.prefix_token_ids cannot be empty")
    if max_gen_len <= 0:
        raise ValueError("max_gen_len must be positive")

    end_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    bsz = len(batch.prefix) * num_answer_per_question
    max_prompt_len = max(len(t) for t in batch.prefix_token_ids)
    min_prompt_len = min(len(t) for t in batch.prefix_token_ids)
    total_len = max_gen_len + max_prompt_len

    try:
        model.init_kv_cache(
            max_batch_size=bsz, max_seq_len=total_len, device=device, dtype=dtype
        )
        tokens = torch.full(
            (bsz, total_len), pad_token_id, dtype=torch.long, device=device
        )
        prefix_tensor = torch.tensor(
            batch.prefix_token_ids, dtype=torch.long, device=device
        )
        tokens[:bsz, :max_prompt_len] = prefix_tensor.repeat_interleave(
            num_answer_per_question, dim=0
        )
    except torch.cuda.OutOfMemoryError:
        raise RuntimeError("Insufficient GPU memory for batch size")

    input_text_mask = tokens != pad_token_id
    if min_prompt_len >= total_len:
        raise ValueError(
            f"min_prompt_len ({min_prompt_len}) must be less than total_len ({total_len})"
        )

    tokens, is_finished = _generate_tokens(
        model,
        tokens,
        input_text_mask,
        min_prompt_len,
        total_len,
        end_token_id,
        pad_token_id,
        device,
        dtype,
    )

    if debug:
        logger.debug(
            f"Tokens shape: {tokens.shape}, Finished: {is_finished.sum().item()}/{bsz}"
        )

    model.del_kv_cache()
    torch.cuda.empty_cache()
    gc.collect()

    episodes = []
    for i in range(bsz // num_answer_per_question):
        for j in range(num_answer_per_question):
            idx = i * num_answer_per_question + j
            generated_token_ids = tokens[idx, len(batch.prefix_token_ids[i]) :].tolist()
            pad_mask = torch.tensor(generated_token_ids, device=device) == pad_token_id
            if pad_mask.any():
                first_pad_idx = pad_mask.nonzero(as_tuple=True)[0][0].item()
                generated_token_ids = generated_token_ids[:first_pad_idx]
            generated_text = (
                tokenizer.detokenize(generated_token_ids) if generated_token_ids else ""
            )
            rewards = reward_function(
                response=generated_text,
                numbers=batch.numbers[i],
                target=batch.target[i],
                end_token=tokenizer.eos_token,
            )
            episodes.append(
                Episode(
                    prefix=batch.prefix[i],
                    text=batch.prefix[i] + generated_text,
                    prefix_token_ids=batch.prefix_token_ids[i],
                    prefix_tokens=batch.prefix_tokens[i],
                    generated_token_ids=generated_token_ids,
                    is_finished=is_finished[idx].item(),
                    reward=rewards["reward"],
                    reward_info=rewards["reward_info"],
                )
            )
    return episodes


def normalize_rewards_per_group(episodes: List[Episode]) -> List[Episode]:
    """
    Normalizes rewards per group, where a group is defined by the prefix.

    Args:
        episodes: List of Episode objects to normalize.

    Returns:
        List of Episode objects with normalized rewards.

    Raises:
        ValueError: If rewards contain NaN or infinite values.
    """
    groups = defaultdict(list)
    for episode in episodes:
        groups[tuple(episode.prefix)].append(episode)
    output = []
    for group in groups.values():
        group_rewards = [item.reward for item in group]
        if any(not np.isfinite(r) for r in group_rewards):
            raise ValueError("Invalid reward values detected (NaN or infinite)")
        mean_reward = np.mean(group_rewards)
        std_reward = np.std(group_rewards)
        for episode in group:
            normalized_reward = np.clip(
                (episode.reward - mean_reward) / (std_reward + EPSILON),
                REWARD_CLIP_MIN,
                REWARD_CLIP_MAX,
            )
            episode = dataclasses.replace(episode, reward=normalized_reward)
            output.append(episode)
    return output


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Computes the entropy of the logits.

    Args:
        logits: Tensor of shape (batch_size, seq_len, vocab_size).

    Returns:
        Entropy tensor of shape (batch_size, seq_len).
    """
    probs = torch.nn.functional.softmax(logits, dim=-1)
    log_probs = torch.log(probs + EPSILON)  # Avoid log(0)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy


def update_policy(
    model: Transformer,
    optimizer,
    episodes: List[Episode],
    micro_batch_size: int,
    pad_token_id: int,
    max_grad_norm: float,
    device: torch.device,
    dtype: torch.dtype,
    debug: bool = False,
) -> dict:
    """
    Updates the policy using the GRPO algorithm with normalized advantages and entropy regularization.

    Args:
        model: The transformer model to update.
        optimizer: The optimizer for model parameters.
        episodes: List of Episode objects with rewards.
        micro_batch_size: Size of micro-batches for gradient computation.
        pad_token_id: ID of the padding token.
        max_grad_norm: Maximum gradient norm for clipping.
        device: Device to run the model on.
        dtype: Data type for model computations.
        debug: If True, logs additional debugging information.

    Returns:
        Dictionary with loss, gradient norm, and entropy metrics.

    Raises:
        ValueError: If episodes list is empty or micro_batch_size is non-positive.
    """
    if not episodes:
        raise ValueError("episodes list cannot be empty")
    if micro_batch_size <= 0:
        raise ValueError("micro_batch_size must be positive")

    episodes = normalize_rewards_per_group(episodes)
    episodes.sort(key=lambda x: len(x.prefix_token_ids) + len(x.generated_token_ids))
    num_micro_batches = math.ceil(len(episodes) / micro_batch_size)
    num_target_tokens = sum(len(episode.generated_token_ids) for episode in episodes)
    entropy = 0.0

    for i in tqdm(
        range(0, len(episodes), micro_batch_size), desc="Computing policy gradient"
    ):
        j = min(i + micro_batch_size, len(episodes))
        batch_episodes = episodes[i:j]
        batch_lengths = [
            len(episode.prefix_token_ids) + len(episode.generated_token_ids)
            for episode in batch_episodes
        ]
        batch_max_length = max(batch_lengths)
        batch_token_ids = torch.full(
            (len(batch_episodes), batch_max_length),
            pad_token_id,
            device=device,
            dtype=torch.long,
        )
        batch_masks = torch.zeros(
            (len(batch_episodes), batch_max_length), device=device, dtype=torch.bool
        )
        for k, episode in enumerate(batch_episodes):
            length = len(episode.prefix_token_ids) + len(episode.generated_token_ids)
            batch_token_ids[k, :length] = torch.tensor(
                episode.prefix_token_ids + episode.generated_token_ids, device=device
            )
            batch_masks[k, len(episode.prefix_token_ids) : length] = 1

        batch_advantages = torch.tensor(
            [episode.reward for episode in batch_episodes],
            device=device,
            dtype=torch.float32,
        )
        batch_advantages = (batch_advantages - batch_advantages.mean()) / (
            batch_advantages.std() + EPSILON
        )

        try:
            with torch.autocast(device_type=device.type, dtype=dtype):
                input_token_ids = batch_token_ids[:, :-1]
                target_token_ids = batch_token_ids[:, 1:]
                target_masks = batch_masks[:, 1:]
                logits = model.forward(input_token_ids).float()
        except torch.cuda.OutOfMemoryError:
            raise RuntimeError("Insufficient GPU memory during policy update")

        log_probs = -torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_token_ids.reshape(-1),
            ignore_index=pad_token_id,
            reduction="none",
        ).reshape(input_token_ids.shape[0], -1)

        with torch.no_grad():
            token_entropy = compute_entropy(logits)
            entropy += (token_entropy * target_masks).sum() / num_target_tokens

        obj = log_probs * batch_advantages[:, None]
        obj = (obj * target_masks).sum() / num_target_tokens
        loss = -obj - ENTROPY_WEIGHT * token_entropy.mean()
        loss.backward()

        if debug:
            logger.debug(
                f"Batch {i//micro_batch_size+1}/{num_micro_batches}: Loss={loss.item():.4f}"
            )

    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=max_grad_norm
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {
        "loss": loss.item(),
        "grad_norm": grad_norm.item(),
        "entropy": entropy.item(),
    }
