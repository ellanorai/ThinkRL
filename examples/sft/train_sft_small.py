"""
ThinkRL SFT Training Example
============================

Example script demonstrating how to use the SFTTrainer for instruction tuning.

Usage:
    python examples/sft/train_sft.py \
        --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --dataset_name yahma/alpaca-cleaned \
        --output_dir ./output/sft \
        --logging_backend wandb

Recommended small models for 4GB VRAM:
- TinyLlama/TinyLlama-1.1B-Chat-v1.0 (~2.2 GB FP16)
- facebook/opt-350m (~700 MB)
- facebook/opt-125m (~250 MB, good for testing)
"""

from dataclasses import dataclass, field

from datasets import load_dataset
from transformers import AutoTokenizer, HfArgumentParser

from thinkrl.models.loader import get_model
from thinkrl.training.sft_trainer import SFTConfig, SFTTrainer
from thinkrl.utils.logging import get_logger


logger = get_logger(__name__)


@dataclass
class ScriptArgs:
    """Script arguments."""

    model_name_or_path: str = field(
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", metadata={"help": "Model name or path"}
    )
    dataset_name: str = field(default="yahma/alpaca-cleaned", metadata={"help": "HuggingFace dataset name"})
    max_samples: int | None = field(default=1000, metadata={"help": "Max samples to use. None for full dataset."})
    output_dir: str = field(default="./output/sft", metadata={"help": "Output directory"})

    # Training
    learning_rate: float = field(default=2e-5)
    batch_size: int = field(default=2)
    gradient_accumulation_steps: int = field(default=8)
    num_epochs: int = field(default=1)
    max_length: int = field(default=512)

    # Precision
    bf16: bool = field(default=False)
    fp16: bool = field(default=False)
    gradient_checkpointing: bool = field(default=True)

    # Logging
    logging_backend: str = field(
        default="tensorboard", metadata={"help": "Logging backend: 'tensorboard', 'wandb', or 'none'"}
    )
    wandb_project: str = field(default="thinkrl-sft")
    logging_steps: int = field(default=10)
    save_steps: int = field(default=500)


def format_alpaca(example: dict) -> str:
    """Format Alpaca-style example."""
    if example.get("input", ""):
        return (
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Input:\n{example['input']}\n\n"
            f"### Response:\n{example['output']}"
        )
    return f"### Instruction:\n{example['instruction']}\n\n" f"### Response:\n{example['output']}"


def main():
    parser = HfArgumentParser(ScriptArgs)
    args = parser.parse_args_into_dataclasses()[0]

    logger.info("=" * 60)
    logger.info("ThinkRL SFT Training")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model_name_or_path}")
    logger.info(f"Dataset: {args.dataset_name}")
    logger.info(f"Logging: {args.logging_backend}")
    logger.info("=" * 60)

    # bf16 flag controls model precision via get_model
    # Load model
    logger.info("Loading model...")
    model = get_model(
        args.model_name_or_path,
        model_type="actor",
        bf16=args.bf16,
        trust_remote_code=True,
        # SFT typically doesn't need to return output dict in forward, but Actor handles it
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load and preprocess dataset
    logger.info(f"Loading dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name, split="train")

    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    logger.info(f"Dataset size: {len(dataset)}")

    # Tokenize
    def tokenize(example):
        text = format_alpaca(example)
        tokens = tokenizer(
            text,
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized = dataset.map(tokenize, remove_columns=dataset.column_names)

    # Create config
    config = SFTConfig(
        model_name_or_path=args.model_name_or_path,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        max_seq_length=args.max_length,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        output_dir=args.output_dir,
        logging_backend=args.logging_backend,
        wandb_project=args.wandb_project,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=tokenized,
        tokenizer=tokenizer,
    )

    # Train
    trainer.train()

    logger.info("Done!")


if __name__ == "__main__":
    main()
