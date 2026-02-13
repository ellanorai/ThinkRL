from collections.abc import Callable
from typing import Any, Union

import torch
import torch.nn as nn
from transformers import GenerationConfig, PreTrainedTokenizer

from thinkrl.algorithms.reinforce_pp import REINFORCEPPAlgorithm, REINFORCEPPConfig, create_reinforce_pp
from thinkrl.data.datasets import RLHFDataset
from thinkrl.data.loaders import RLHFDataLoader
from thinkrl.integration.vllm_client import VLLMClient
from thinkrl.utils.logging import get_logger


logger = get_logger(__name__)


class ReinforcePPTrainer:
    """
    Trainer for REINFORCE++.

    Orchestrates the training process:
    1. Sampling prompts from dataset
    2. Generating completions (rollouts)
    3. Computing rewards (using provided reward_fn)
    4. Updating policy using REINFORCEPPAlgorithm
    """

    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module,
        tokenizer: PreTrainedTokenizer,
        dataset: RLHFDataset,
        reward_fn: Callable[[list[str], list[str]], torch.Tensor],
        config: REINFORCEPPConfig | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        generation_config: GenerationConfig | None = None,
        device: Union[str, torch.device] | None = None,
        use_vllm: bool = False,
        vllm_group_port: int = 51216,
        **algo_kwargs,
    ):
        """
        Args:
            model: The policy model.
            ref_model: The reference model.
            tokenizer: Tokenizer for encoding/decoding.
            dataset: Dataset containing prompts.
            reward_fn: Callable taking (prompts, completions) and returning rewards tensor [B].
            config: Algorithm configuration.
            optimizer: Optimizer.
            generation_config: Configuration for generation (sampling).
            device: Device to train on.
            use_vllm: Whether to use VLLM for generation.
            **algo_kwargs: Args passed to create_reinforce_pp if config is None.
        """
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.reward_fn = reward_fn
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_vllm = use_vllm
        self.vllm_client = None

        # Setup generation config
        num_return_sequences = 1
        if config and config.mode == "baseline":
            num_return_sequences = config.group_size

        self.generation_config = generation_config or GenerationConfig(
            max_new_tokens=128,
            do_sample=True,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_return_sequences=num_return_sequences,
        )

        # Create Algorithm
        if config is None:
            # Create from simple args
            self.algorithm = create_reinforce_pp(
                policy_model=model, ref_model=ref_model, optimizer=optimizer, **algo_kwargs
            )
        else:
            self.algorithm = REINFORCEPPAlgorithm(
                policy_model=model,
                ref_model=ref_model,
                optimizer=optimizer,
                config=config,
            )

        # Ensure models are on device
        self.algorithm.to(self.device)

        # Initialize VLLM Client if needed
        if self.use_vllm:
            self.vllm_client = VLLMClient(group_port=vllm_group_port)
            # Initialize weight sync on proper device
            self.vllm_client.init_weight_sync(self.device)

    def train(self, steps: int = 1000, batch_size: int = 4, log_interval: int = 10):
        """
        Main training loop.
        """
        try:
            from tqdm import tqdm
        except ImportError:

            def tqdm(x, **kwargs):
                return x

        # Check for wandb
        import sys

        is_wandb_active = False
        if "wandb" in sys.modules:
            import wandb

            if wandb.run is not None:
                is_wandb_active = True

        logger.info(f"Starting REINFORCE++ training for {steps} steps...")
        if self.use_vllm:
            logger.info("Using VLLM for generation.")

        # Create DataLoader using RLHFDataLoader from loaders.py
        dataloader = RLHFDataLoader(
            dataset=self.dataset,
            tokenizer=self.tokenizer,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )

        step = 0
        epoch = 0

        progress_bar = tqdm(total=steps, desc="Training")

        while step < steps:
            for batch_prompts in dataloader:
                if step >= steps:
                    break

                # Sync weights if using VLLM (every step or every N steps?)
                # For basic PPO/Reinforce++, we sync every step because policy changes.
                if self.use_vllm:
                    self.vllm_client.update_model_weights(self.algorithm.policy_model)

                # 1. Generate Rollouts
                rollout_data = self.make_experience(batch_prompts)

                # 2. Compute Rewards
                prompts_text = batch_prompts["prompt_text"]
                # Decode if we used local generation (VLLM gives text directly)
                if "completions_text" in rollout_data:
                    completions_text = rollout_data["completions_text"]
                else:
                    # Decoding from generated_ids
                    completions_text = self.tokenizer.batch_decode(
                        rollout_data["generated_ids"], skip_special_tokens=True
                    )

                # Expand prompts if needed for reward fn
                num_return_sequences = self.generation_config.num_return_sequences
                if num_return_sequences > 1:
                    expanded_prompts = []
                    for p in prompts_text:
                        expanded_prompts.extend([p] * num_return_sequences)
                    prompts_text = expanded_prompts

                # Extract and expand targets if available
                targets = batch_prompts.get("target", None)
                kwargs = {}
                if targets is not None:
                    if num_return_sequences > 1:
                        expanded_targets = []
                        for t in targets:
                            expanded_targets.extend([t] * num_return_sequences)
                        targets = expanded_targets
                    kwargs["targets"] = targets

                rewards = self.reward_fn(prompts_text, completions_text, **kwargs).to(self.device)

                # Handle batch size mismatch (if last batch is smaller)
                curr_bs = len(prompts_text)
                if rewards.shape[0] != curr_bs:
                    rewards = rewards[:curr_bs]

                rollout_data["rewards"] = rewards

                # 3. Train Step (with gradient accumulation)
                grad_accum_steps = self.algorithm.config.gradient_accumulation_steps
                is_accumulating = (step + 1) % grad_accum_steps != 0

                # Train step handles its own backward, but we control optimizer step
                metrics = self.algorithm.train_on_rollout(rollout_data, accumulate_grad=is_accumulating)
                # Take the last epoch metrics
                step_metrics = metrics[-1] if metrics else {}

                # Log to WandB
                if is_wandb_active:
                    wandb_metrics = {
                        f"train/{k}": v.item() if isinstance(v, torch.Tensor) else v for k, v in step_metrics.items()
                    }
                    wandb.log(wandb_metrics, step=step)

                if step % log_interval == 0:
                    loss_val = step_metrics.get("loss", 0.0)
                    if isinstance(loss_val, torch.Tensor):
                        loss_val = loss_val.item()
                    reward_val = rewards.mean().item()

                    logger.info(f"Step {step}: Loss={loss_val:.4f}, " f"Reward={reward_val:.4f}")
                    progress_bar.set_postfix({"loss": f"{loss_val:.3f}", "reward": f"{reward_val:.3f}"})

                progress_bar.update(1)
                step += 1

            epoch += 1

        progress_bar.close()

    def make_experience(self, batch_prompts: dict[str, Any]) -> dict[str, torch.Tensor]:
        """
        Generate rollouts and compute log probs.
        """
        prompts_text = batch_prompts["prompt_text"]
        input_ids = batch_prompts["input_ids"].to(self.device)
        attention_mask = batch_prompts["attention_mask"].to(self.device)

        # Expand prompts if generating multiple sequences per prompt
        num_return_sequences = self.generation_config.num_return_sequences
        if num_return_sequences > 1:
            expanded_prompts = []
            for p in prompts_text:
                expanded_prompts.extend([p] * num_return_sequences)
            prompts_text = expanded_prompts

            # Also expand input_ids and attention_mask to match
            input_ids = input_ids.repeat_interleave(num_return_sequences, dim=0)
            attention_mask = attention_mask.repeat_interleave(num_return_sequences, dim=0)

        if self.use_vllm:
            # --- VLLM Generation ---
            params = {
                "max_tokens": self.generation_config.max_new_tokens,
                "temperature": self.generation_config.temperature,
                "top_p": self.generation_config.top_p if self.generation_config.top_p else 1.0,
            }

            output = self.vllm_client.generate(prompts_text, params, return_logprobs=True)

            completions_text = output["text"]
            token_ids_list = output["token_ids"]
            log_probs_list = output["log_probs"]

            # Convert token ID lists to tensors and pad
            generated_ids = [torch.tensor(ids, dtype=torch.long, device=self.device) for ids in token_ids_list]
            generated_ids_padded = torch.nn.utils.rnn.pad_sequence(
                generated_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )

            # Build full sequences, labels, and aligned old_log_probs
            full_sequences = []
            labels = []
            old_log_probs_list = []
            has_valid_logprobs = bool(log_probs_list) and all(len(lp) > 0 for lp in log_probs_list)

            for i in range(len(prompts_text)):
                # Get unpadded prompt tokens
                curr_input_ids = input_ids[i][attention_mask[i] == 1]
                curr_gen_ids = generated_ids[i]

                curr_full = torch.cat([curr_input_ids, curr_gen_ids])
                full_sequences.append(curr_full)

                # Labels: -100 for prompt tokens
                curr_labels = curr_full.clone()
                curr_labels[: len(curr_input_ids)] = -100
                labels.append(curr_labels)

                # Construct aligned old_log_probs for the full sequence.
                # The algorithm's get_log_probs() applies a causal shift: logits[t]
                # predicts labels[t+1], so log_probs[t] = log P(token[t+1] | context[0:t]).
                # After the shift, generated token logprobs start at position
                # (num_prompt_tokens - 1) in the output tensor. vLLM's logprobs[k]
                # is the log prob of generating the k-th token, matching this alignment.
                if has_valid_logprobs:
                    curr_log_probs = torch.tensor(
                        log_probs_list[i], dtype=torch.float, device=self.device
                    )
                    full_log_probs = torch.zeros(len(curr_full), device=self.device)
                    num_prompt_tokens = len(curr_input_ids)
                    start_pos = num_prompt_tokens - 1
                    end_pos = min(start_pos + len(curr_log_probs), len(full_log_probs))
                    use_len = end_pos - start_pos
                    full_log_probs[start_pos:end_pos] = curr_log_probs[:use_len]
                    old_log_probs_list.append(full_log_probs)

            # Pad everything
            full_sequences_padded = torch.nn.utils.rnn.pad_sequence(
                full_sequences, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
            labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

            result = {
                "input_ids": full_sequences_padded,
                "attention_mask": (full_sequences_padded != self.tokenizer.pad_token_id).long(),
                "labels": labels_padded,
                "generated_ids": generated_ids_padded,
                "completions_text": completions_text,
            }

            # Include pre-computed old_log_probs if vLLM returned valid logprobs.
            # This avoids a redundant forward pass in train_on_rollout().
            if has_valid_logprobs and old_log_probs_list:
                result["old_log_probs"] = torch.nn.utils.rnn.pad_sequence(
                    old_log_probs_list, batch_first=True, padding_value=0.0
                )

            return result

        else:
            # --- Local Generation ---
            with torch.no_grad():
                self.algorithm.policy_model.eval()
                outputs = self.algorithm.policy_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.generation_config.max_new_tokens,
                    do_sample=self.generation_config.do_sample,
                    temperature=self.generation_config.temperature,
                    top_p=self.generation_config.top_p,
                    top_k=self.generation_config.top_k,
                    num_return_sequences=self.generation_config.num_return_sequences,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

                # Set model back to train mode after generation
                self.algorithm.policy_model.train()

                full_sequences = outputs.sequences

                input_len = input_ids.shape[1]
                generated_ids = full_sequences[:, input_len:]

                labels = full_sequences.clone()
                labels[:, :input_len] = -100

            return {
                "input_ids": full_sequences,
                "attention_mask": (full_sequences != self.tokenizer.pad_token_id).long(),
                "labels": labels,
                "generated_ids": generated_ids,
            }
