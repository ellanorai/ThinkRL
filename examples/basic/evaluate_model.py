"""
ThinkRL Evaluation Example
==========================

Score a model, or a checkpoint produced by one of the trainers, with the Evaluator.

Usage:
    python examples/basic/evaluate_model.py --model facebook/opt-125m

Runs on CPU with the default model.
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from thinkrl.evaluation import Evaluator
from thinkrl.utils.logging import get_logger


logger = get_logger(__name__)

PROMPTS = ["What is 2 + 2?", "Name a primary colour:", "Count to three:"]
TARGETS = ["4", "red", "1 2 3"]


def length_reward(prompts: list[str], completions: list[str]) -> torch.Tensor:
    """Same brevity reward as the training example, so the two are comparable."""
    target = 40
    return torch.tensor([max(0.0, 1.0 - abs(len(c) - target) / target) for c in completions])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="facebook/opt-125m", help="Model id or checkpoint directory")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)

    evaluator = Evaluator(model=model, tokenizer=tokenizer, reward_fn=length_reward)
    result = evaluator.evaluate(PROMPTS, targets=TARGETS, max_new_tokens=args.max_new_tokens)

    print(result)
    for prompt, completion in zip(PROMPTS, result.completions):
        print(f"\n> {prompt}\n{completion.strip()}")


if __name__ == "__main__":
    main()
