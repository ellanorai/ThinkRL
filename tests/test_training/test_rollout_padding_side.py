"""Regression test: rollout batches must be left-padded.

These prompts go straight into model.generate. A decoder-only model continues from the last
position, so right padding makes every prompt shorter than the batch maximum continue from
pad tokens, and the reward function then scores completions the policy never produces.
"""

import inspect

import torch
from transformers import GPT2Config, GPT2LMHeadModel

import thinkrl.training.grpo_trainer as grpo_trainer
import thinkrl.training.reinforce_pp_trainer as reinforce_pp_trainer
from thinkrl.data.loaders import create_rlhf_collate_fn


class _StubTokenizer:
    pad_token_id = 0
    padding_side = "right"  # the library default, and the thing being overridden


def _batch():
    return [
        {"input_ids": torch.tensor([5, 6, 7]), "attention_mask": torch.tensor([1, 1, 1])},
        {"input_ids": torch.tensor([8]), "attention_mask": torch.tensor([1])},
    ]


def test_collator_left_pads_when_asked():
    collated = create_rlhf_collate_fn(_StubTokenizer(), padding_side="left")(_batch())

    # The short sequence keeps its real token last, where generation continues from.
    assert collated["input_ids"][1].tolist() == [0, 0, 8]
    assert collated["attention_mask"][1].tolist() == [0, 0, 1]


def test_collator_still_right_pads_by_default():
    """Loss computation wants right padding, so the default must not move."""
    collated = create_rlhf_collate_fn(_StubTokenizer())(_batch())

    assert collated["input_ids"][1].tolist() == [8, 0, 0]


def test_both_rollout_trainers_request_left_padding():
    for module in (grpo_trainer, reinforce_pp_trainer):
        source = inspect.getsource(module)
        assert 'padding_side="left"' in source, f"{module.__name__} builds its rollout loader without left padding"


def test_left_padding_makes_batched_generation_match_unbatched():
    """The property that actually matters, checked on a real generate() call."""
    torch.manual_seed(0)
    model = GPT2LMHeadModel(GPT2Config(n_layer=2, n_head=2, n_embd=16, n_positions=32, vocab_size=40)).eval()

    long_prompt = torch.tensor([5, 6, 7, 8])
    short_prompt = torch.tensor([9])

    def generate(input_ids, attention_mask):
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=0,
        )
        return out[:, input_ids.shape[1] :]

    alone = generate(short_prompt.unsqueeze(0), torch.ones(1, 1, dtype=torch.long))

    collate = create_rlhf_collate_fn(_StubTokenizer(), padding_side="left")
    left = collate(
        [
            {"input_ids": long_prompt, "attention_mask": torch.ones(4, dtype=torch.long)},
            {"input_ids": short_prompt, "attention_mask": torch.ones(1, dtype=torch.long)},
        ]
    )
    batched_left = generate(left["input_ids"], left["attention_mask"])

    right = create_rlhf_collate_fn(_StubTokenizer())(
        [
            {"input_ids": long_prompt, "attention_mask": torch.ones(4, dtype=torch.long)},
            {"input_ids": short_prompt, "attention_mask": torch.ones(1, dtype=torch.long)},
        ]
    )
    batched_right = generate(right["input_ids"], right["attention_mask"])

    assert torch.equal(batched_left[1], alone[0]), "left-padded batch diverged from unbatched generation"
    assert not torch.equal(batched_right[1], alone[0]), "right padding no longer diverges; the fix can go"
