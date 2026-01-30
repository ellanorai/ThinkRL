from dataclasses import dataclass, field
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

from thinkrl.algorithms.reinforce_pp import REINFORCEPPConfig
from thinkrl.data.datasets import RLHFDataset
from thinkrl.training.reinforce_pp_trainer import ReinforcePPTrainer
from thinkrl.utils.logging import get_logger


logger = get_logger(__name__)


@dataclass
class ScriptArguments:
    """
    Arguments for the training script.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    ref_model_name_or_path: str = field(metadata={"help": "Path to reference model. Required for REINFORCE++."})
    dataset_name: str = field(metadata={"help": "Path to dataset"})
    output_dir: str = field(default="output/reinforce_pp", metadata={"help": "Output directory"})
    learning_rate: float = field(default=1e-6, metadata={"help": "Learning rate"})
    batch_size: int = field(default=4, metadata={"help": "Per-device batch size"})
    max_steps: int = field(default=100, metadata={"help": "Maximum training steps"})
    mode: str = field(default="baseline", metadata={"help": "Mode: 'general' (k=1) or 'baseline' (k>1)"})
    group_size: int = field(default=4, metadata={"help": "Group size for baseline mode"})
    kl_coeff: float = field(default=0.1, metadata={"help": "KL coefficient for baseline mode"})
    entropy_coeff: float = field(default=0.01, metadata={"help": "Entropy coefficient"})
    # VLLM
    use_vllm: bool = field(default=False, metadata={"help": "Use VLLM for generation"})
    vllm_group_port: int = field(default=51216, metadata={"help": "NCCL group port for VLLM sync"})
    # Logging
    logging_backend: str = field(
        default="tensorboard", metadata={"help": "Logging backend: 'tensorboard', 'wandb', or 'none'."}
    )
    wandb_project: str = field(
        default="thinkrl-reinforce-pp", metadata={"help": "WandB project name (if using wandb)."}
    )


def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    logger.info(f"Starting REINFORCE++ training in {script_args.mode} mode...")

    # 1. Load Models
    logger.info("Loading models...")
    policy_model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
        device_map="auto",
    )

    ref_model = AutoModelForCausalLM.from_pretrained(
        script_args.ref_model_name_or_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Implemented Custom Reward Function
    # Here we define a simple reward function: Length reward (longer is better, up to a point)
    # In practice, this would be a verifier or a learned reward model.
    def reward_fn(prompts, completions):
        """
        Reward function.
        Args:
            prompts: List of prompt strings.
            completions: List of completion strings.
        Returns:
            torch.Tensor: Rewards of shape [B]
        """
        rewards = []
        for c in completions:
            # Dummy reward: length of response
            # Normalized to be somewhat standard normal-ish
            r = len(c) / 100.0
            rewards.append(r)
        return torch.tensor(rewards)

    # 3. Load Dataset
    logger.info(f"Loading dataset: {script_args.dataset_name}")
    dataset = RLHFDataset(
        dataset_name_or_path=script_args.dataset_name,
        tokenizer=tokenizer,
        prompt_column="prompt",  # Adjust based on your dataset
        max_length=512,
    )

    # 4. Config
    config = REINFORCEPPConfig(
        learning_rate=script_args.learning_rate,
        batch_size=script_args.batch_size,
        mode=script_args.mode,
        group_size=script_args.group_size,
        kl_loss_coeff=script_args.kl_coeff,
        entropy_coeff=script_args.entropy_coeff,
    )

    # 5. Initialize Trainer
    trainer = ReinforcePPTrainer(
        model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset,
        reward_fn=reward_fn,
        config=config,
        use_vllm=script_args.use_vllm,
        vllm_group_port=script_args.vllm_group_port,
    )

    # 6. Start Training
    logger.info("Starting training loop...")
    trainer.train(steps=script_args.max_steps, batch_size=script_args.batch_size)

    logger.info("Training complete.")

    # Save model
    if script_args.output_dir:
        os.makedirs(script_args.output_dir, exist_ok=True)
        policy_model.save_pretrained(script_args.output_dir)
        tokenizer.save_pretrained(script_args.output_dir)
        logger.info(f"Model saved to {script_args.output_dir}")


if __name__ == "__main__":
    main()
