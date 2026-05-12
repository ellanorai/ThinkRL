from collections.abc import Callable
from typing import Any, Union

import torch
import torch.nn as nn
from transformers import GenerationConfig, PreTrainedTokenizer

from thinkrl.algorithms.kto import KTOAlgorithm, KTOConfig
from thinkrl.data.datasets import RLHFDataset
from thinkrl.data.loaders import RLHFDataLoader
from thinkrl.integration.vllm_client import VLLMClient
from thinkrl.utils.logging import get_logger
from thinkrl.utils.metrics import MetricsTracker


logger = get_logger(__name__)


class KTOTrainer:
    """
    Trainer for Kahneman-Tversky Optimization (KTO).

    Orchestrates the online training process:
    1. Sampling prompts from dataset
    2. Generating completions (rollouts)
    3. Computing rewards (using provided reward_fn)
    4. Binarizing rewards into desirable/undesirable
    5. Updating policy using KTOAlgorithm
    """

    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module,
        tokenizer: PreTrainedTokenizer,
        dataset: RLHFDataset,
        reward_fn: Callable[[list[str], list[str]], torch.Tensor],
        config: KTOConfig | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        generation_config: GenerationConfig | None = None,
        device: Union[str, torch.device] | None = None,
        use_vllm: bool = False,
        vllm_group_port: int = 51216,
        reward_threshold: float = 0.5,
        **algo_kwargs,
    ):
        """
        Args:
            model: The policy model.
            ref_model: The reference model.
            tokenizer: Tokenizer for encoding/decoding.
            dataset: Dataset containing prompts.
            reward_fn: Callable taking (prompts, completions) and returning rewards tensor [B].
            config: KTO configuration.
            optimizer: Optimizer.
            generation_config: Configuration for generation (sampling).
            device: Device to train on.
            use_vllm: Whether to use VLLM for generation.
            reward_threshold: Threshold to consider a generation desirable (1) vs undesirable (0).
            **algo_kwargs: Additional kwargs for Algorithm.
        """
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.reward_fn = reward_fn
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_vllm = use_vllm
        self.vllm_client = None
        self.reward_threshold = reward_threshold

        self.config = config

        self.generation_config = generation_config or GenerationConfig(
            max_new_tokens=256,
            do_sample=True,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
        )

        # Create Algorithm
        if config is None:
            from thinkrl.algorithms.kto import create_kto
            self.algorithm = create_kto(policy_model=model, reference_model=ref_model, optimizer=optimizer, **algo_kwargs)
        else:
            self.algorithm = KTOAlgorithm(
                policy_model=model, reference_model=ref_model, optimizer=optimizer, config=config, **algo_kwargs
            )

        self.config = self.algorithm.config

        # Ensure models are on device
        self.algorithm.to(self.device)

        # Initialize VLLM Client if needed
        if self.use_vllm:
            self.vllm_client = VLLMClient(group_port=vllm_group_port)
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

        import sys

        is_wandb_active = False
        if "wandb" in sys.modules:
            import wandb

            if wandb.run is not None:
                is_wandb_active = True

        logger.info(f"Starting KTO training for {steps} steps...")
        if self.use_vllm:
            logger.info("Using VLLM for generation.")

        # Create DataLoader
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
        metrics_tracker = MetricsTracker()

        while step < steps:
            for batch_prompts in dataloader:
                if step >= steps:
                    break

                if self.use_vllm:
                    self.vllm_client.update_model_weights(self.algorithm.policy_model)

                # 1. Generate Rollouts
                rollout_data = self.make_experience(batch_prompts)

                # 2. Compute Rewards & Binarize
                prompts_text = batch_prompts["prompt_text"]
                if "completions_text" in rollout_data:
                    completions_text = rollout_data["completions_text"]
                else:
                    completions_text = self.tokenizer.batch_decode(
                        rollout_data["generated_ids"], skip_special_tokens=True
                    )

                num_return_sequences = self.generation_config.num_return_sequences
                expanded_prompts = []
                for p in prompts_text:
                    expanded_prompts.extend([p] * num_return_sequences)
                prompts_text = expanded_prompts

                targets = batch_prompts.get("target", None)
                kwargs = {}
                if targets is not None:
                    expanded_targets = []
                    for t in targets:
                        expanded_targets.extend([t] * num_return_sequences)
                    kwargs["targets"] = expanded_targets

                rewards = self.reward_fn(prompts_text, completions_text, **kwargs).to(self.device)

                curr_bs = len(prompts_text)
                if rewards.shape[0] != curr_bs:
                    rewards = rewards[:curr_bs]

                # Binarize rewards based on threshold
                binarized_labels = (rewards > self.reward_threshold).float()
                
                # If we want batch-relative KTO, we could threshold on the batch mean
                # binarized_labels = (rewards > rewards.mean()).float()

                rollout_data["rewards"] = rewards
                rollout_data["binarized_labels"] = binarized_labels

                # 3. Train Step
                metrics = self.algorithm.train_on_rollout(rollout_data)
                step_metrics = metrics[-1] if metrics else {}

                # Track metrics
                metrics_tracker.update_dict(step_metrics)
                metrics_tracker.update("reward", rewards.mean().item())

                # 4. Clean up Memory
                del rollout_data
                torch.cuda.empty_cache()

                # Log to WandB
                if is_wandb_active:
                    wandb_metrics = {
                        f"train/{k}": v.item() if isinstance(v, torch.Tensor) else v for k, v in step_metrics.items()
                    }
                    wandb_metrics["train/reward"] = rewards.mean().item()
                    wandb.log(wandb_metrics, step=step)

                if step % log_interval == 0:
                    loss_val = step_metrics.get("loss", 0.0)
                    if isinstance(loss_val, torch.Tensor):
                        loss_val = loss_val.item()
                    reward_val = rewards.mean().item()

                    logger.info(f"Step {step}: Loss={loss_val:.4f}, Reward={reward_val:.4f}")
                    progress_bar.set_postfix({"loss": f"{loss_val:.3f}", "reward": f"{reward_val:.3f}"})

                progress_bar.update(1)
                step += 1

            epoch += 1

        progress_bar.close()

    def make_experience(self, batch_prompts: dict[str, Any]) -> dict[str, torch.Tensor]:
        """
        Generate rollouts.
        """
        prompts_text = batch_prompts["prompt_text"]
        input_ids = batch_prompts["input_ids"].to(self.device)
        attention_mask = batch_prompts["attention_mask"].to(self.device)

        num_return_sequences = self.generation_config.num_return_sequences

        if self.use_vllm:
            expanded_prompts = []
            for p in prompts_text:
                expanded_prompts.extend([p] * num_return_sequences)
            prompts_text = expanded_prompts

            input_ids = input_ids.repeat_interleave(num_return_sequences, dim=0)
            attention_mask = attention_mask.repeat_interleave(num_return_sequences, dim=0)
            params = {
                "max_tokens": self.generation_config.max_new_tokens,
                "temperature": self.generation_config.temperature,
                "top_p": getattr(self.generation_config, "top_p", 1.0),
            }

            output = self.vllm_client.generate(prompts_text, params, return_logprobs=False)

            completions_text = output["text"]
            token_ids_list = output["token_ids"]

            generated_ids = [torch.tensor(ids, dtype=torch.long, device=self.device) for ids in token_ids_list]
            generated_ids_padded = torch.nn.utils.rnn.pad_sequence(
                generated_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )

            full_sequences = []
            labels = []

            for i in range(len(prompts_text)):
                curr_input_ids = input_ids[i][attention_mask[i] == 1]
                curr_gen_ids = generated_ids[i]

                curr_full = torch.cat([curr_input_ids, curr_gen_ids])
                full_sequences.append(curr_full)

                curr_labels = curr_full.clone()
                curr_labels[: len(curr_input_ids)] = -100
                labels.append(curr_labels)

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

            return result

        else:
            with torch.no_grad():
                self.algorithm.policy_model.eval()
                outputs = self.algorithm.policy_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.generation_config.max_new_tokens,
                    do_sample=self.generation_config.do_sample,
                    temperature=self.generation_config.temperature,
                    top_p=getattr(self.generation_config, "top_p", 1.0),
                    top_k=getattr(self.generation_config, "top_k", 50),
                    num_return_sequences=self.generation_config.num_return_sequences,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

                self.algorithm.policy_model.train()

                full_sequences = outputs

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
