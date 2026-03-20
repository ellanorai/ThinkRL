from dataclasses import dataclass, field
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

from thinkrl.algorithms.star import STaRConfig
from thinkrl.data.datasets import RLHFDataset
from thinkrl.training.star_trainer import STaRTrainer
from thinkrl.utils.logging import get_logger


logger = get_logger(__name__)


@dataclass
class ScriptArguments:
    """
    Arguments for the STaR training script.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    dataset_name: str = field(metadata={"help": "Path to dataset (e.g., 'gsm8k')"})
    output_dir: str = field(default="output/star", metadata={"help": "Output directory"})
    learning_rate: float = field(default=1e-6, metadata={"help": "Learning rate"})
    max_iterations: int = field(default=40, metadata={"help": "Number of STaR iterations"})
    base_steps: int = field(default=40, metadata={"help": "Base training steps in loop 0"})
    scaling_factor: float = field(default=1.2, metadata={"help": "Step scaling factor"})
    warmup_steps: int = field(default=100, metadata={"help": "Linear warmup steps"})


def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    logger.info("Starting STaR training example...")

    logger.info("Loading models...")
    policy_model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Custom Reward Function
    def reward_fn(prompts, completions, targets=None, **kwargs):
        """
        Dummy reward function (Simple substring match).
        """
        if not targets:
            return torch.zeros(len(completions))
        rewards = []
        for c, t in zip(completions, targets):
            # Check if target answer is in completion
            r = 1.0 if t.strip() in c else 0.0
            rewards.append(r)
        return torch.tensor(rewards)

    # 3. Load Dataset
    logger.info(f"Loading dataset: {script_args.dataset_name}")
    dataset = RLHFDataset(
        dataset_name_or_path=script_args.dataset_name,
        tokenizer=tokenizer,
        prompt_column="prompt",
        target_column="answer",
        max_length=512,
    )

    # 4. Config
    config = STaRConfig(
        learning_rate=script_args.learning_rate,
        max_iterations=script_args.max_iterations,
        base_training_steps=script_args.base_steps,
        step_scaling_factor=script_args.scaling_factor,
        warmup_steps=script_args.warmup_steps,
    )

    # 5. Initialize Trainer
    trainer = STaRTrainer(
        model=policy_model,
        tokenizer=tokenizer,
        dataset=dataset,
        reward_fn=reward_fn,
        config=config,
    )

    # 6. Start Training
    logger.info("Starting STaR loop...")
    trainer.train()

    logger.info("Training complete.")

    # Save model
    if script_args.output_dir:
        os.makedirs(script_args.output_dir, exist_ok=True)
        policy_model.save_pretrained(script_args.output_dir)
        tokenizer.save_pretrained(script_args.output_dir)
        logger.info(f"Model saved to {script_args.output_dir}")


if __name__ == "__main__":
    main()
