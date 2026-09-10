"""
ThinkRL Inference Example
=========================

Load a model, or a checkpoint produced by one of the trainers, and generate completions.

Usage:
    python examples/basic/inference.py --model facebook/opt-125m --prompt "What is 2 + 2?"
    python examples/basic/inference.py --model ./outputs/checkpoint_epoch0_step100

Runs on CPU with the default model.
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from thinkrl.utils.logging import get_logger


logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="facebook/opt-125m", help="Model id or checkpoint directory")
    parser.add_argument("--tokenizer", default=None, help="Defaults to --model")
    parser.add_argument("--prompt", action="append", help="Repeatable; a default set is used if omitted")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    prompts = args.prompt or ["What is 2 + 2?", "Name a colour:"]

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Decoder-only generation needs left padding, or short prompts are padded on the side
    # the model continues from.
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.eval()

    encoded = tokenizer(prompts, return_tensors="pt", padding=True)
    generated = model.generate(
        **encoded,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.temperature > 0,
        temperature=args.temperature,
        pad_token_id=tokenizer.pad_token_id,
    )

    completions = tokenizer.batch_decode(generated[:, encoded["input_ids"].shape[1] :], skip_special_tokens=True)
    for prompt, completion in zip(prompts, completions):
        print(f"\n> {prompt}\n{completion.strip()}")


if __name__ == "__main__":
    main()
