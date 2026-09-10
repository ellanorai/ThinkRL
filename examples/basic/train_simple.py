"""
ThinkRL Minimal GRPO Example
============================

The smallest end-to-end GRPO run: a small model, a toy prompt set, and a reward function
that scores completions on a property you can check without a reward model. Meant to prove
the pipeline works on your machine before committing to a real run.

Usage:
    python examples/basic/train_simple.py --steps 20

Runs on CPU with the default model (~250 MB), so it needs no GPU.
"""

import argparse
import json
import tempfile
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from thinkrl.algorithms.grpo import GRPOConfig
from thinkrl.data.datasets import RLHFDataset
from thinkrl.training.grpo_trainer import GRPOTrainer
from thinkrl.utils.logging import get_logger


logger = get_logger(__name__)

PROMPTS = [
    "Count from one to five:",
    "Name three primary colours:",
    "What is two plus two?",
    "List two planets:",
]


def length_reward(prompts: list[str], completions: list[str], **kwargs) -> torch.Tensor:
    """Reward brevity, scaled into [0, 1].

    A stand-in for a real reward model: it is cheap, deterministic and moves in a direction
    you can see in the logs, which is what a smoke test needs.
    """
    target = 40
    scores = [max(0.0, 1.0 - abs(len(c) - target) / target) for c in completions]
    return torch.tensor(scores, dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="facebook/opt-125m")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=4)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # float32 explicitly: many checkpoints (opt-125m among them) declare float16 in their
    # config, and AdamW with eps=1e-8 cannot train fp16 weights, eps underflows to zero.
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    ref_model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)

    # RLHFDataset loads from a path, so the toy prompts are written out first.
    prompt_file = Path(tempfile.mkdtemp()) / "prompts.jsonl"
    prompt_file.write_text("\n".join(json.dumps({"prompt": p}) for p in PROMPTS))

    dataset = RLHFDataset(dataset_name_or_path=str(prompt_file), tokenizer=tokenizer)

    trainer = GRPOTrainer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset,
        reward_fn=length_reward,
        config=GRPOConfig(group_size=args.group_size),
    )

    logger.info(f"Training {args.model} for {args.steps} steps on {len(PROMPTS)} prompts")
    trainer.train(steps=args.steps, batch_size=args.batch_size)
    logger.info("Done")


if __name__ == "__main__":
    main()
