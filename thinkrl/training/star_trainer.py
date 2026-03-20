from collections.abc import Callable
import copy
from typing import Union

import torch
import torch.nn as nn
from transformers import GenerationConfig, PreTrainedTokenizer

from thinkrl.algorithms.star import STaRAlgorithm, STaRConfig
from thinkrl.data.datasets import RLHFDataset
from thinkrl.data.loaders import RLHFDataLoader
from thinkrl.utils.logging import get_logger


logger = get_logger(__name__)


class STaRTrainer:
    """
    Trainer for STaR (Self-Taught Reasoner).

    Orchestrates the iterative loop:
    1. Sample prompts & generate rationales (Bootstrapping).
    2. Identify correct answers and buffer successful rationales.
    3. For failed samples, provide "hints" and rationalise (Rationalization).
    4. Reset model to base weights.
    5. Fine-tune on the collected dataset.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizer,
        dataset: RLHFDataset,
        reward_fn: Callable[[list[str], list[str]], torch.Tensor],
        config: STaRConfig | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        generation_config: GenerationConfig | None = None,
        device: Union[str, torch.device] | None = None,
        **algo_kwargs,
    ):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.reward_fn = reward_fn
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Save base model weights for the "Reset-to-Base" protocol
        self.base_model_state = copy.deepcopy(model.state_dict())
        self.model = model

        self.config = config or STaRConfig()

        self.generation_config = generation_config or GenerationConfig(
            max_new_tokens=256,
            do_sample=True,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        self.algorithm = STaRAlgorithm(policy_model=model, optimizer=optimizer, config=self.config, **algo_kwargs)
        self.algorithm.to(self.device)

    def train(self, iterations: int | None = None):
        """
        Main STaR loop.
        """
        iterations = iterations or self.config.max_iterations

        logger.info(f"Starting STaR training for {iterations} iterations...")

        for iter_idx in range(iterations):
            logger.info(f"Iteration {iter_idx + 1}/{iterations}")

            # 1 & 2. Bootstrapping & Filtering
            collected_data = self.collect_successful_rationales(iter_idx)

            if not collected_data:
                logger.warning("No successful rationales found in this iteration. Skipping training step.")
                continue

            # 4. Reset Model to Base weights
            logger.info("Resetting model to base pre-trained weights...")
            self.model.load_state_dict(self.base_model_state)

            # 5. Fine-tune on collected data
            self.fine_tune(collected_data, iter_idx)

    def collect_successful_rationales(self, iter_idx: int):
        """
        Generates rationales and filters them based on correctness.
        Also performs the Rationalization step for failed samples.
        """
        dataloader = RLHFDataLoader(
            dataset=self.dataset,
            tokenizer=self.tokenizer,
            batch_size=self.config.generation_batch_size,
            shuffle=True,
        )

        successful_samples = []

        for batch in dataloader:
            prompts = batch["prompt_text"]
            targets = batch.get("target", None)

            # Step 1: Bootstrapping (Generation)
            rollout = self.generate_and_filter(prompts, targets)
            successful_samples.extend(rollout["success_data"])

            # Step 3: Rationalization (Hint-based)
            if rollout["failed_prompts"]:
                rationalized = self.rationalize(rollout["failed_prompts"], rollout["failed_targets"])
                successful_samples.extend(rationalized)

        logger.info(f"Iteration {iter_idx}: Collected {len(successful_samples)} successful reasoning chains.")
        return successful_samples

    def generate_and_filter(self, prompts: list[str], targets: list[str] = None):
        """
        Standard generation and correctness filtering.
        """
        self.model.eval()
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, generation_config=self.generation_config)

        completions = self.tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True)

        # Identity correct ones
        rewards = self.reward_fn(prompts, completions, targets=targets)

        success_data = []
        failed_prompts = []
        failed_targets = []

        for i, r in enumerate(rewards):
            if r > 0.5:  # Correct
                # Store (input_ids, labels) where labels mask the prompt
                prompt_ids = inputs.input_ids[i][inputs.attention_mask[i] == 1]
                gen_ids = outputs[i][inputs.input_ids.shape[1] :]
                full_seq = torch.cat([prompt_ids, gen_ids])
                labels = full_seq.clone()
                labels[: len(prompt_ids)] = -100
                success_data.append({"input_ids": full_seq, "labels": labels})
            else:
                failed_prompts.append(prompts[i])
                if targets:
                    failed_targets.append(targets[i])

        return {"success_data": success_data, "failed_prompts": failed_prompts, "failed_targets": failed_targets}

    def rationalize(self, failed_prompts: list[str], failed_targets: list[str]):
        """
        Backward reasoning: provide the correct answer as a hint.
        """
        if not failed_targets:
            return []

        hint_prompts = [
            p + "\n" + self.config.rationalization_hint_format.format(answer=t)
            for p, t in zip(failed_prompts, failed_targets)
        ]

        self.model.eval()
        inputs = self.tokenizer(hint_prompts, return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, generation_config=self.generation_config)

        completions = self.tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True)

        # Verify if the rationalized chain still leads to the correct answer
        # (Though we provided the answer, we want a valid logical chain)
        rewards = self.reward_fn(hint_prompts, completions, targets=failed_targets)

        rationalized_data = []
        for i, r in enumerate(rewards):
            if r > 0.5:
                prompt_ids = inputs.input_ids[i][inputs.attention_mask[i] == 1]
                gen_ids = outputs[i][inputs.input_ids.shape[1] :]
                full_seq = torch.cat([prompt_ids, gen_ids])
                labels = full_seq.clone()
                labels[: len(prompt_ids)] = -100
                rationalized_data.append({"input_ids": full_seq, "labels": labels})

        return rationalized_data

    def fine_tune(self, collected_data: list[dict], iter_idx: int):
        """
        Standard SFT on the collected dataset.
        Applies scaling: 40 * 1.2^iter
        """
        # Calculate steps
        steps = int(self.config.base_training_steps * (self.config.step_scaling_factor**iter_idx))
        logger.info(f"Iteration {iter_idx}: Fine-tuning for {steps} steps.")

        # Create a simple loader for the collected tokens
        # For simplicity in this implementation, we manually loop over the collected data
        # or use a Small Batch DataLoader helper.

        self.model.train()

        bs = self.config.train_batch_size
        for step in range(steps):
            # Permutation-based sampling: cycle through data, reshuffle each epoch
            if step % len(collected_data) == 0:
                perm = torch.randperm(len(collected_data))
            offset = (step * bs) % len(collected_data)
            indices = [perm[i % len(perm)].item() for i in range(offset, offset + bs)]
            batch = []
            max_len = 0
            for idx in indices:
                item = collected_data[idx]
                batch.append(item)
                max_len = max(max_len, len(item["input_ids"]))

            # Pad batch
            input_ids = torch.full((len(batch), max_len), self.tokenizer.pad_token_id, dtype=torch.long)
            labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
            attn_mask = torch.zeros((len(batch), max_len), dtype=torch.long)

            for i, item in enumerate(batch):
                seq_len = len(item["input_ids"])
                input_ids[i, :seq_len] = item["input_ids"]
                labels[i, :seq_len] = item["labels"]
                attn_mask[i, :seq_len] = 1

            batch_tensor = {
                "input_ids": input_ids.to(self.device),
                "labels": labels.to(self.device),
                "attention_mask": attn_mask.to(self.device),
            }

            metrics = self.algorithm.training_step(batch_tensor)

            if step % 10 == 0:
                logger.debug(f"Step {step}/{steps}: Loss={metrics['loss_val']:.4f}")
