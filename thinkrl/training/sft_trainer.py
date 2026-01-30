"""
ThinkRL SFT Trainer
===================

Supervised Fine-Tuning (SFT) Trainer for instruction-following models.

Similar to TRL's SFTTrainer, provides a simple interface for fine-tuning
language models on instruction-response pairs.

Author: EllanorAI
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import os
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from thinkrl.utils.logging import get_logger


# Optional DeepSpeed import
try:
    import deepspeed

    _DEEPSPEED_AVAILABLE = True
except ImportError:
    _DEEPSPEED_AVAILABLE = False
    deepspeed = None

logger = get_logger(__name__)


@dataclass
class SFTConfig:
    """Configuration for SFT training."""

    # Model
    model_name_or_path: str = "meta-llama/Llama-3.1-8B"

    # Training
    learning_rate: float = 2e-5
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 1

    # Optimizer
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    warmup_steps: int = 0

    # Sequence
    max_seq_length: int = 2048
    packing: bool = False
    dataset_text_field: str = "text"

    # Logging
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    logging_backend: str = "tensorboard"  # 'tensorboard', 'wandb', or 'none'
    wandb_project: str = "thinkrl-sft"
    wandb_run_name: Optional[str] = None

    # Output
    output_dir: str = "./sft_output"
    save_total_limit: int = 3

    # Mixed precision
    bf16: bool = True
    fp16: bool = False

    # Gradient
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = False

    # LoRA (optional)
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: list[str] | None = None

    # Distributed
    local_rank: int = -1
    deepspeed: str | None = None


class SFTTrainer:
    """
    Supervised Fine-Tuning Trainer.

    Provides a simple interface for fine-tuning language models on
    instruction-response datasets, similar to TRL's SFTTrainer.

    Features:
    - Standard LM loss on response tokens
    - Optional sequence packing
    - LoRA/QLoRA support
    - DeepSpeed integration
    - Gradient checkpointing
    - Mixed precision training
    - TensorBoard/WandB logging
    """

    def __init__(
        self,
        model: Any = None,
        args: SFTConfig | None = None,
        train_dataset: Any = None,
        eval_dataset: Any = None,
        tokenizer: Any = None,
        data_collator: Callable | None = None,
        formatting_func: Callable | None = None,
        packing: bool = False,
        max_seq_length: int = 2048,
        callbacks: list | None = None,
        **kwargs,
    ):
        """
        Initialize the SFT Trainer.

        Args:
            model: The model to fine-tune
            args: Training configuration
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            tokenizer: Tokenizer for the model
            data_collator: Custom data collator
            formatting_func: Function to format dataset examples
            packing: Whether to use sequence packing
            max_seq_length: Maximum sequence length
            callbacks: Optional training callbacks
        """
        self.model = model
        self.args = args or SFTConfig()
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.data_collator = data_collator
        self.formatting_func = formatting_func
        self.packing = packing or self.args.packing
        self.max_seq_length = max_seq_length or self.args.max_seq_length
        self.callbacks = callbacks or []

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Setup dtype
        if self.args.bf16:
            self.dtype = torch.bfloat16
        elif self.args.fp16:
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        # Ensure pad token
        if self.tokenizer and self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Setup gradient checkpointing
        if self.args.gradient_checkpointing and self.model is not None:
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled.")

        # Initialize logging backend
        self.writer = None
        self.wandb_run = None
        self._setup_logging()

        # Initialize optimizer and scheduler (created in train())
        self.optimizer = None
        self.scheduler = None

        # State tracking
        self.global_step = 0
        self.epoch = 0
        self.model_engine = None  # DeepSpeed engine

    def _setup_logging(self):
        """Initialize logging backend."""
        if self.args.logging_backend == "tensorboard":
            try:
                from torch.utils.tensorboard import SummaryWriter

                log_dir = os.path.join(self.args.output_dir, "tensorboard")
                os.makedirs(log_dir, exist_ok=True)
                self.writer = SummaryWriter(log_dir=log_dir)
                logger.info(f"TensorBoard logging to: {log_dir}")
            except ImportError:
                logger.warning("TensorBoard not available. Install with: pip install tensorboard")
        elif self.args.logging_backend == "wandb":
            try:
                import wandb

                self.wandb_run = wandb.init(
                    project=self.args.wandb_project,
                    name=self.args.wandb_run_name,
                    config=self.args.__dict__,
                )
                logger.info(f"WandB logging to project: {self.args.wandb_project}")
            except ImportError:
                logger.warning("WandB not available. Install with: pip install wandb")

    def _log_metrics(self, metrics: dict[str, float], step: int):
        """Log metrics to the configured backend."""
        if self.writer:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, step)

        if self.wandb_run:
            import wandb

            wandb.log(metrics, step=step)

    def _default_collator(self, batch):
        """Default data collator for SFT."""
        if isinstance(batch[0], dict):
            # Assume tokenized inputs
            input_ids = torch.stack([torch.tensor(x["input_ids"]) for x in batch])
            attention_mask = torch.stack([torch.tensor(x["attention_mask"]) for x in batch])
            labels = input_ids.clone()
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
        else:
            raise ValueError(f"Unsupported batch type: {type(batch[0])}")

    def create_dataloader(
        self,
        dataset: Any,
        batch_size: int,
        shuffle: bool = True,
    ) -> DataLoader:
        """
        Create a DataLoader for training or evaluation.

        Args:
            dataset: Dataset to load
            batch_size: Batch size
            shuffle: Whether to shuffle

        Returns:
            DataLoader instance
        """
        collator = self.data_collator or self._default_collator
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collator,
            num_workers=0,
            pin_memory=True,
        )

    def get_train_dataloader(self) -> DataLoader:
        """Get training DataLoader."""
        return self.create_dataloader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
        )

    def get_eval_dataloader(self, eval_dataset: Any = None) -> DataLoader:
        """Get evaluation DataLoader."""
        dataset = eval_dataset or self.eval_dataset
        if dataset is None:
            raise ValueError("No evaluation dataset provided.")
        return self.create_dataloader(
            dataset,
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False,
        )

    def compute_loss(
        self,
        model: Any,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        """
        Compute training loss.

        Args:
            model: The model
            inputs: Input batch
            return_outputs: Whether to return model outputs

        Returns:
            Loss tensor, optionally with outputs
        """
        # Handle DeepSpeed Engine wrapper
        if _DEEPSPEED_AVAILABLE and hasattr(model, "module"):
            # Unwrap DeepSpeed engine to access underlying Actor or Model
            real_model = model.module
        else:
            real_model = model

        # Handle Actor wrapper (get_model) which returns tuple in forward
        # For SFT, we need standard CausalLM outputs (loss), so use underlying model
        # We check class name to avoid importing Actor (circular dep risk) and to avoid matching HF models (which have .model)
        if type(model).__name__ == "Actor" and hasattr(model, "model"):
            outputs = model.model(**inputs)
        else:
            outputs = real_model(**inputs)

        loss = outputs.loss

        if return_outputs:
            return loss, outputs
        return loss

    def train(self, resume_from_checkpoint: str | None = None) -> dict[str, Any]:
        """
        Run training.

        Args:
            resume_from_checkpoint: Path to checkpoint to resume from

        Returns:
            Training metrics
        """
        if self.model is None:
            raise ValueError("No model provided for training.")
        if self.train_dataset is None:
            raise ValueError("No training dataset provided.")

        # Move model to device
        self.model.to(self.device)
        self.model.train()

        # Create dataloader
        train_dataloader = self.get_train_dataloader()

        # Calculate total steps
        num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
        # Total steps calculated but maybe not used immediately if not passed to scheduler directly here
        total_steps = num_update_steps_per_epoch * self.args.num_train_epochs

        # Setup DeepSpeed if enabled
        if self.args.deepspeed and _DEEPSPEED_AVAILABLE:
            if not os.path.exists(self.args.deepspeed):
                logger.warning(f"DeepSpeed config not found at: {self.args.deepspeed}, assuming it's a dict or None")
            # DeepSpeed accepts path or dict directly

            self.model_engine, self.optimizer, _, self.scheduler = deepspeed.initialize(
                model=self.model,
                model_parameters=self.model.parameters(),
                config=self.args.deepspeed,  # initialize accepts path or dict
                optimizer=self.optimizer,  # Optional: Pass optimizer if we created it manually
                lr_scheduler=self.scheduler,
            )
            # If DS creates optimizer (offload etc), self.optimizer is updated.
            # If we passed optimizer, it wraps it.
            # Usually better to let DS create optimizer if config specifies it.
            # If we created optimizer above (AdamW), and DS config has "optimizer", DS might error or override.
            # For simplicity: If DS is enabled, we use what DS gives us.
            logger.info("DeepSpeed Engine Initialized")
        else:
            self.model_engine = self.model

            # Restore manual optimizer/scheduler setup for non-DeepSpeed
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
            )

            warmup_steps = self.args.warmup_steps or int(total_steps * self.args.warmup_ratio)
            from transformers import get_linear_schedule_with_warmup

            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
            logger.info("Manual Optimizer & Scheduler Initialized")

        # Refined Optimizer/Scheduler setup logic:
        # If DS init was skipped (no config), we rely on manual setup above.
        # Ideally, we should move Manual Setup to "else" block of DS check,
        # but purely wrapping provided optimizer is also valid usage (ZeRO-1/2).

        # Training loop
        running_loss = 0.0
        loss_count = 0
        # Zero grad handled by engine step or optimizer
        if self.model_engine == self.model:  # Non-DS
            self.optimizer.zero_grad()

        for epoch in range(self.args.num_train_epochs):
            self.epoch = epoch
            logger.info(f"Epoch {epoch + 1}/{self.args.num_train_epochs}")

            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")

            for step, batch in enumerate(progress_bar):
                # Move to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass
                if self.args.bf16 or self.args.fp16:
                    with torch.autocast(device_type="cuda", dtype=self.dtype):
                        loss = self.compute_loss(self.model, batch)
                        loss = loss / self.args.gradient_accumulation_steps
                else:
                    loss = self.compute_loss(self.model, batch)
                    loss = loss / self.args.gradient_accumulation_steps

                # Accumulate for logging
                running_loss += loss.item() * self.args.gradient_accumulation_steps
                loss_count += 1

                # Backward & Step
                if self.args.deepspeed and _DEEPSPEED_AVAILABLE:
                    self.model_engine.backward(loss)
                    self.model_engine.step()
                    # DS handles gradient accumulation and zero_grad internally
                else:
                    loss.backward()

                    # Gradient accumulation
                    if (step + 1) % self.args.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or self.args.deepspeed:
                    # Logging (Trigger only on update steps or periodically?)
                    # DS step() happens every micro-batch, but optimizer update happens every gas.
                    # We should align logging with updates.
                    # Simpler: Log every N global steps.

                    # Manual GS increment for non-DS is above.
                    # For DS, engine.global_steps is tracked.

                    if self.args.deepspeed:
                        curr_step = self.model_engine.global_steps
                    else:
                        if (step + 1) % self.args.gradient_accumulation_steps == 0:
                            self.global_step += 1
                        curr_step = self.global_step

                    if curr_step > self.global_step:  # Update happened
                        self.global_step = curr_step

                        # Logging logic...
                        if self.global_step % self.args.logging_steps == 0:
                            # ... (rest of logging)

                            # ... update progress bar
                            try:
                                if self.scheduler:
                                    current_lr = self.scheduler.get_last_lr()[0]
                                else:
                                    current_lr = 0.0  # DS might hide it
                            except Exception:
                                current_lr = 0.0

                            avg_loss = running_loss / max(loss_count, 1)
                            progress_bar.set_postfix(
                                {
                                    "loss": f"{avg_loss:.4f}",
                                    "lr": f"{current_lr:.2e}",
                                }
                            )
                            # ... log

                            self._log_metrics(
                                {
                                    "train/loss": avg_loss,
                                    "train/learning_rate": current_lr,
                                    "train/epoch": epoch + 1,
                                },
                                self.global_step,
                            )

                    # Evaluation
                    if self.eval_dataset and self.global_step > 0 and self.global_step % self.args.eval_steps == 0:
                        # Ensure sync before eval?
                        eval_metrics = self.evaluate()
                        self._log_metrics(eval_metrics, self.global_step)
                        # Switch back to train
                        if self.args.deepspeed:
                            self.model_engine.train()
                        else:
                            self.model.train()

                    # Save checkpoint
                    if self.global_step > 0 and self.global_step % self.args.save_steps == 0:
                        self.save_model(os.path.join(self.args.output_dir, f"checkpoint-{self.global_step}"))

            # Non-DS update logic handled inside loop above for accumulation

        # Final save
        self.save_model()

        # Cleanup
        if self.writer:
            self.writer.close()
        if self.wandb_run:
            import wandb

            wandb.finish()

        logger.info("Training complete!")
        return {"global_step": self.global_step}

    def evaluate(self, eval_dataset: Any = None) -> dict[str, float]:
        """
        Run evaluation.

        Args:
            eval_dataset: Optional evaluation dataset

        Returns:
            Evaluation metrics
        """
        self.model.eval()
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        total_loss = 0.0
        total_steps = 0

        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                if self.args.bf16 or self.args.fp16:
                    with torch.autocast(device_type="cuda", dtype=self.dtype):
                        loss = self.compute_loss(self.model, batch)
                else:
                    loss = self.compute_loss(self.model, batch)

                total_loss += loss.item()
                total_steps += 1

        avg_loss = total_loss / total_steps if total_steps > 0 else 0.0
        logger.info(f"Eval loss: {avg_loss:.4f}")

        return {"eval/loss": avg_loss}

    def save_model(self, output_dir: str | None = None):
        """
        Save the model.

        Args:
            output_dir: Directory to save model to
        """
        save_dir = output_dir or self.args.output_dir
        os.makedirs(save_dir, exist_ok=True)

        self.model.save_pretrained(save_dir)
        if self.tokenizer:
            self.tokenizer.save_pretrained(save_dir)

        logger.info(f"Model saved to {save_dir}")

    def push_to_hub(self, repo_id: str, **kwargs):
        """
        Push model to HuggingFace Hub.

        Args:
            repo_id: Repository ID on the Hub
            **kwargs: Additional arguments for push_to_hub
        """
        self.model.push_to_hub(repo_id, **kwargs)
        if self.tokenizer:
            self.tokenizer.push_to_hub(repo_id, **kwargs)
        logger.info(f"Model pushed to {repo_id}")


def create_sft_trainer(
    model: Any,
    tokenizer: Any,
    train_dataset: Any,
    config: SFTConfig | None = None,
    **kwargs,
) -> SFTTrainer:
    """
    Factory function to create an SFT Trainer.

    Args:
        model: Model to train
        tokenizer: Tokenizer
        train_dataset: Training dataset
        config: Training configuration
        **kwargs: Additional arguments

    Returns:
        SFTTrainer instance
    """
    return SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        **kwargs,
    )


__all__ = ["SFTConfig", "SFTTrainer", "create_sft_trainer"]
