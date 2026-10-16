from dataclasses import dataclass, field
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

from thinkrl.algorithms.kto import KTOConfig
from thinkrl.data.datasets import RLHFDataset
from thinkrl.training.kto_trainer import KTOTrainer
from thinkrl.utils.logging import get_logger


logger = get_logger(__name__)


@dataclass
class ScriptArguments:
    """
    Arguments for the KTO training script.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained policy model or identifier from huggingface.co/models"}
    )
    dataset_name: str = field(metadata={"help": "Path to dataset (e.g., 'gsm8k')"})
    output_dir: str = field(default="output/kto", metadata={"help": "Output directory"})
    learning_rate: float = field(default=1e-6, metadata={"help": "Learning rate"})
    beta: float = field(default=0.1, metadata={"help": "Beta parameter for KL penalty equivalent"})
    lambda_d: float = field(default=1.0, metadata={"help": "Weight for desirable generations"})
    lambda_u: float = field(default=1.0, metadata={"help": "Weight for undesirable generations"})
    reward_threshold: float = field(default=0.5, metadata={"help": "Threshold for binarizing rewards"})
    use_vllm: bool = field(default=False, metadata={"help": "Use vLLM for generation"})


def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    logger.info("Starting KTO training example...")

    logger.info("Loading models...")
    # KTO requires a reference model
    policy_model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
        device_map="auto",
    )
    
    ref_model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
        device_map="auto",
    )
    # Freeze reference model
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

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

    # Load Dataset
    logger.info(f"Loading dataset: {script_args.dataset_name}")
    dataset = RLHFDataset(
        dataset_name_or_path=script_args.dataset_name,
        tokenizer=tokenizer,
        prompt_column="prompt",
        target_column="answer",
        max_length=512,
    )

    # Config
    config = KTOConfig(
        learning_rate=script_args.learning_rate,
        beta=script_args.beta,
        lambda_d=script_args.lambda_d,
        lambda_u=script_args.lambda_u,
        use_vllm=script_args.use_vllm,
    )

    # Initialize Trainer
    trainer = KTOTrainer(
        model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset,
        reward_fn=reward_fn,
        config=config,
        reward_threshold=script_args.reward_threshold,
    )

    # Start Training
    logger.info("Starting KTO loop...")
    trainer.train(steps=100) # Small number of steps for example

    logger.info("Training complete.")

    # Save model
    if script_args.output_dir:
        os.makedirs(script_args.output_dir, exist_ok=True)
        policy_model.save_pretrained(script_args.output_dir)
        tokenizer.save_pretrained(script_args.output_dir)
        logger.info(f"Model saved to {script_args.output_dir}")


if __name__ == "__main__":
    main()
