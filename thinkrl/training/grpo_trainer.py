from collections.abc import Callable
from typing import Any, Union

import torch
import torch.nn as nn
from transformers import GenerationConfig, PreTrainedTokenizer

from thinkrl.algorithms.base import BaseRLHFAlgorithm
from thinkrl.algorithms.grpo import GRPOAlgorithm, GRPOConfig
from thinkrl.data.datasets import RLHFDataset
from thinkrl.data.loaders import RLHFDataLoader
from thinkrl.evaluation.periodic import build_periodic_evaluator
from thinkrl.integration.vllm_client import VLLMClient
from thinkrl.logging.rollout import RolloutInspector
from thinkrl.utils.checkpoint import CheckpointManager, save_training_checkpoint
from thinkrl.utils.logging import get_logger


logger = get_logger(__name__)


class GRPOTrainer:
    """
    Trainer for Group Relative Policy Optimization (GRPO).

    Orchestrates the training process:
    1. Sampling prompts from dataset
    2. Generating a group of completions per prompt (rollouts)
    3. Computing rewards (using provided reward_fn)
    4. Updating policy using GRPOAlgorithm
    """

    def __init__(
        self,
        model: nn.Module | None = None,
        ref_model: nn.Module | None = None,
        tokenizer: PreTrainedTokenizer = None,
        dataset: RLHFDataset = None,
        reward_fn: Callable[[list[str], list[str]], torch.Tensor] = None,
        config: GRPOConfig | None = None,
        algorithm: BaseRLHFAlgorithm | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        generation_config: GenerationConfig | None = None,
        device: Union[str, torch.device] | None = None,
        use_vllm: bool = False,
        vllm_group_port: int = 51216,
        vllm_url: str = "http://localhost:8000",
        vllm_sync_world_size: int = 2,
        **algo_kwargs,
    ):
        """
        Args:
            model: The policy model.
            ref_model: The reference model.
            tokenizer: Tokenizer for encoding/decoding.
            dataset: Dataset containing prompts.
            reward_fn: Callable taking (prompts, completions) and returning rewards tensor [B].
            config: GRPO configuration.
            optimizer: Optimizer.
            generation_config: Configuration for generation (sampling).
            device: Device to train on.
            use_vllm: Whether to use VLLM for generation.
            vllm_group_port: Port for the NCCL weight-sync bridge.
            vllm_url: Address of the vLLM worker. VLLMClient accepted this and the trainer
                never forwarded it, so a remote worker was unreachable (#85).
            vllm_sync_world_size: Processes participating in the weight sync, likewise.
            algorithm: An already-constructed algorithm to drive instead of building GRPO.
                Any BaseRLHFAlgorithm implementing train_on_rollout works, which covers
                PPO, DAPO, VAPO, PRIME and Dr.GRPO. They each shipped a loss, a config, a
                factory and a registry entry with no code path that stepped an optimizer
                with them (#124); generalising this loop was cheaper and less duplicated
                than writing four more trainers.
            **algo_kwargs: Additional kwargs for Algorithm.
        """
        # Checked before anything else is read: passing both would leave the caller
        # training a model they did not hand over, which only shows up in the loss curve.
        if algorithm is not None and model is not None:
            raise ValueError("pass either `algorithm` or `model`, not both")

        self.tokenizer = tokenizer
        self.dataset = dataset
        self.reward_fn = reward_fn
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_vllm = use_vllm
        self.vllm_client = None

        self.config = config

        # A group of outputs per prompt (G samples). Read off whichever config is in play,
        # since a non-GRPO algorithm may not define group_size at all.
        source_config = getattr(algorithm, "config", None) if algorithm is not None else config
        num_return_sequences = getattr(source_config, "group_size", None) or GRPOConfig().group_size

        self.generation_config = generation_config or GenerationConfig(
            max_new_tokens=256,
            do_sample=True,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_return_sequences=num_return_sequences,
        )

        # Create Algorithm
        if algorithm is not None:
            self.algorithm = algorithm
        elif config is None:
            # Create from simple kwargs using the factory method
            from thinkrl.algorithms.grpo import create_grpo

            self.algorithm = create_grpo(policy_model=model, ref_model=ref_model, optimizer=optimizer, **algo_kwargs)
        else:
            # Initialize explicitly with the provided config object
            self.algorithm = GRPOAlgorithm(
                policy_model=model, ref_model=ref_model, optimizer=optimizer, config=config, **algo_kwargs
            )

        # Ensure we bind to the instantiated config in the algorithm
        self.config = self.algorithm.config

        # Ensure models are on device
        self.algorithm.to(self.device)

        # Initialize VLLM Client if needed
        if self.use_vllm:
            self.vllm_client = VLLMClient(
                url=vllm_url,
                group_port=vllm_group_port,
                sync_world_size=vllm_sync_world_size,
            )
            self.vllm_client.init_weight_sync(self.device)

    def train(
        self,
        steps: int = 1000,
        batch_size: int = 4,
        log_interval: int = 10,
        inspect_every: int = 0,
        inspect_samples: int = 3,
        checkpoint_dir: str | None = None,
        save_every: int = 0,
        max_checkpoints: int = 5,
        eval_dataset: Any = None,
        eval_every: int = 0,
        eval_batch_size: int = 8,
        eval_max_new_tokens: int = 128,
    ):
        """
        Main training loop.

        Args:
            steps: Number of optimisation steps to run.
            batch_size: Prompts per rollout.
            log_interval: Steps between log lines.
            inspect_every: Print a sample of prompts, completions and rewards every N
                steps. 0 disables it. A scalar reward cannot distinguish a bad policy from
                a broken reward function or empty completions; this can.
            inspect_samples: How many rollouts to show each time.
            checkpoint_dir: Where to write checkpoints. Nothing is written without it.
            save_every: Save every N steps; 0 disables periodic saves. A final checkpoint is
                still written whenever checkpoint_dir is set.
            max_checkpoints: How many checkpoints to keep before the oldest rotates out.
            eval_dataset: Optional held-out set, evaluated with the same reward function.
                Training reward is the quantity being optimized, so it rises whether or not
                the policy improves; a held-out number that diverges from it is what
                catches a policy that has learned the verifier instead of the task.
            eval_every: Evaluate every N steps. 0 disables evaluation.
            eval_batch_size: Prompts per generation batch during evaluation.
            eval_max_new_tokens: Generation budget per prompt during evaluation.
        """
        inspector = RolloutInspector(every=inspect_every, num_samples=inspect_samples)
        evaluator = build_periodic_evaluator(
            model=self.algorithm.policy_model,
            tokenizer=self.tokenizer,
            reward_fn=self.reward_fn,
            dataset=eval_dataset,
            every=eval_every,
            batch_size=eval_batch_size,
            max_new_tokens=eval_max_new_tokens,
        )
        checkpointer = (
            CheckpointManager(
                checkpoint_dir,
                max_checkpoints=max_checkpoints,
                # With a held-out signal available, "best" means best on it rather than
                # last written. Without one there is nothing honest to rank by.
                metric_name="eval/reward_mean" if evaluator.enabled else None,
                mode="max",
            )
            if checkpoint_dir
            else None
        )
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

        logger.info(f"Starting GRPO training for {steps} steps...")
        if self.use_vllm:
            logger.info("Using VLLM for generation.")

        # Create DataLoader
        # Rollout batches are padded on the left. These prompts go straight into
        # model.generate, and a decoder-only model continues from the last position, so
        # right padding would make every prompt shorter than the batch maximum continue
        # from pad tokens. The completions scored by the reward function would then be
        # ones the policy never produces at inference.
        dataloader = RLHFDataLoader(
            dataset=self.dataset,
            tokenizer=self.tokenizer,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            padding_side="left",
        )

        step = 0
        epoch = 0
        step_metrics: dict[str, Any] = {}

        progress_bar = tqdm(total=steps, desc="Training")

        def checkpoint():
            return save_training_checkpoint(
                checkpointer,
                model=self.algorithm.policy_model,
                optimizer=getattr(self.algorithm, "optimizer", None),
                epoch=epoch,
                step=step,
                metrics=step_metrics,
            )

        while step < steps:
            for batch_prompts in dataloader:
                if step >= steps:
                    break

                if self.use_vllm:
                    self.vllm_client.update_model_weights(self.algorithm.policy_model)

                # 1. Generate Rollouts
                # `make_experience` handles either local Hugging Face `generate` or vLLM server requests.
                # Output: `rollout_data` containing expanded `input_ids`, `generated_ids`, and `completions_text`.
                rollout_data = self.make_experience(batch_prompts)

                # 2. Compute Rewards
                # For GRPO, the reward function must evaluate each prompt against G=group_size generated completions.
                prompts_text = batch_prompts["prompt_text"]
                if "completions_text" in rollout_data:
                    completions_text = rollout_data["completions_text"]
                else:
                    completions_text = self.tokenizer.batch_decode(
                        rollout_data["generated_ids"], skip_special_tokens=True
                    )

                # Expand prompts for reward fn
                num_return_sequences = self.generation_config.num_return_sequences
                expanded_prompts = []
                for p in prompts_text:
                    expanded_prompts.extend([p] * num_return_sequences)
                prompts_text = expanded_prompts

                # Extract and expand targets if available
                targets = batch_prompts.get("target", None)
                kwargs = {}
                if targets is not None:
                    expanded_targets = []
                    for t in targets:
                        expanded_targets.extend([t] * num_return_sequences)
                    kwargs["targets"] = expanded_targets

                rewards = self.reward_fn(prompts_text, completions_text, **kwargs).to(self.device)

                curr_bs = len(completions_text)
                if rewards.shape[0] != curr_bs:
                    raise ValueError(
                        f"reward_fn returned {rewards.shape[0]} rewards for {curr_bs} "
                        f"completions. It must return exactly one reward per completion, "
                        f"in prompt-major order ({len(batch_prompts['prompt_text'])} prompts "
                        f"x {num_return_sequences} samples)."
                    )

                rollout_data["rewards"] = rewards

                grouped = rewards.view(-1, num_return_sequences)
                dead = int((grouped.std(dim=1, unbiased=False) == 0).sum())
                if dead:
                    logger.warning(
                        f"Step {step}: {dead}/{grouped.shape[0]} groups have zero reward "
                        f"variance, so they contribute no gradient. The reward function may "
                        f"not be discriminating between completions."
                    )

                # 3. Train Step
                # Executes the GRPO Inner Loop via `train_on_rollout`, computing group-relative
                # advantages, clipping losses, and maintaining KL-divergence constraints.
                metrics = self.algorithm.train_on_rollout(rollout_data)
                step_metrics = metrics[-1] if metrics else {}

                # 4. Clean up Memory
                del rollout_data
                torch.cuda.empty_cache()

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

                    logger.info(f"Step {step}: Loss={loss_val:.4f}, Reward={reward_val:.4f}")
                    progress_bar.set_postfix({"loss": f"{loss_val:.3f}", "reward": f"{reward_val:.3f}"})

                inspector.maybe_show(step, prompts_text, completions_text, rewards)

                progress_bar.update(1)
                step += 1

                # Evaluate before saving, so a checkpoint written on this step carries the
                # held-out metric the manager ranks "best" by.
                eval_metrics = evaluator.maybe_evaluate(step)
                if eval_metrics:
                    step_metrics = {**step_metrics, **eval_metrics}
                    if is_wandb_active:
                        wandb.log(eval_metrics, step=step)

                if save_every and step % save_every == 0:
                    checkpoint()

            epoch += 1

        progress_bar.close()

        # A final pass, so the last checkpoint is ranked on the same footing as the rest
        # rather than being the only one without a held-out number.
        final_eval = evaluator.evaluate()
        if final_eval:
            step_metrics = {**step_metrics, **final_eval}
        checkpoint()

    def make_experience(self, batch_prompts: dict[str, Any]) -> dict[str, torch.Tensor]:
        """
        Generate rollouts and compute log probs.

        Workflow:
        1. Expand each prompt G times (where G = `group_size`).
        2. Leverage `use_vllm` to dispatch high-throughput generation requests to the vLLM server,
           or perform local `policy_model.generate`.
        3. Splice generation chunks, append `labels`, and extract `old_log_probs` natively.
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

            output = self.vllm_client.generate(prompts_text, params, return_logprobs=True)

            completions_text = output["text"]
            token_ids_list = output["token_ids"]
            log_probs_list = output["log_probs"]

            generated_ids = [torch.tensor(ids, dtype=torch.long, device=self.device) for ids in token_ids_list]
            generated_ids_padded = torch.nn.utils.rnn.pad_sequence(
                generated_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )

            full_sequences = []
            labels = []
            old_log_probs_list = []
            has_valid_logprobs = bool(log_probs_list) and all(len(lp) > 0 for lp in log_probs_list)

            for i in range(len(prompts_text)):
                curr_input_ids = input_ids[i][attention_mask[i] == 1]
                curr_gen_ids = generated_ids[i]

                curr_full = torch.cat([curr_input_ids, curr_gen_ids])
                full_sequences.append(curr_full)

                curr_labels = curr_full.clone()
                curr_labels[: len(curr_input_ids)] = -100
                labels.append(curr_labels)

                if has_valid_logprobs:
                    curr_log_probs = torch.tensor(log_probs_list[i], dtype=torch.float, device=self.device)
                    full_log_probs = torch.zeros(len(curr_full), device=self.device)
                    num_prompt_tokens = len(curr_input_ids)
                    start_pos = num_prompt_tokens - 1
                    end_pos = min(start_pos + len(curr_log_probs), len(full_log_probs))
                    use_len = end_pos - start_pos
                    full_log_probs[start_pos:end_pos] = curr_log_probs[:use_len]
                    old_log_probs_list.append(full_log_probs)

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

            if has_valid_logprobs and old_log_probs_list:
                result["old_log_probs"] = torch.nn.utils.rnn.pad_sequence(
                    old_log_probs_list, batch_first=True, padding_value=0.0
                )

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
