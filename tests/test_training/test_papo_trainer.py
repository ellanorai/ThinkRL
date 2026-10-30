"""PAPOTrainer (#194): PAPOAlgorithm.compute_loss hard-requires `masked_input_ids`
in the rollout batch, and nothing in GRPOTrainer's generic rollout loop produces
one. ThinkRL has no multimodal pipeline for the paper's vision occlusion, so this
trainer masks prompt *tokens* instead -- these tests pin that the masking only
ever touches the prompt region (never the completion) and respects mask_prob.
"""

import torch

from tests.test_algorithms.harness import TinyPolicy
import thinkrl.algorithms as algorithms
from thinkrl.training.papo_trainer import PAPOTrainer


class FakeTokenizer:
    mask_token_id = 99
    pad_token_id = 0
    eos_token_id = 1


def _trainer(mask_prob: float = 0.15) -> PAPOTrainer:
    return PAPOTrainer(model=TinyPolicy(), tokenizer=FakeTokenizer(), mask_prob=mask_prob)


def test_papo_registered_as_a_real_non_stub_algorithm():
    assert not algorithms.is_stub(algorithms.PAPOAlgorithm)


def test_mask_prob_out_of_range_is_rejected():
    import pytest

    with pytest.raises(ValueError, match="mask_prob"):
        PAPOTrainer(model=TinyPolicy(), tokenizer=FakeTokenizer(), mask_prob=1.5)


def test_masking_never_touches_completion_tokens():
    trainer = _trainer(mask_prob=1.0)  # mask everything possible, to make the boundary obvious
    input_ids = torch.arange(1, 13).reshape(2, 6)
    labels = input_ids.clone()
    labels[:, :3] = -100  # first 3 tokens are prompt, last 3 are completion

    masked = trainer._mask_prompt_tokens(input_ids, labels)

    assert torch.equal(masked[:, 3:], input_ids[:, 3:])  # completion untouched
    assert torch.all(masked[:, :3] == FakeTokenizer.mask_token_id)  # prompt fully masked at prob=1.0


def test_zero_mask_prob_changes_nothing():
    trainer = _trainer(mask_prob=0.0)
    input_ids = torch.arange(1, 13).reshape(2, 6)
    labels = input_ids.clone()
    labels[:, :3] = -100

    masked = trainer._mask_prompt_tokens(input_ids, labels)

    assert torch.equal(masked, input_ids)


def test_make_experience_adds_masked_input_ids(monkeypatch):
    trainer = _trainer(mask_prob=0.5)
    base_rollout = {
        "input_ids": torch.arange(1, 13).reshape(2, 6),
        "labels": torch.tensor([[-100, -100, -100, 7, 8, 9], [-100, -100, -100, 10, 11, 12]]),
        "attention_mask": torch.ones(2, 6),
    }
    monkeypatch.setattr(
        "thinkrl.training.grpo_trainer.GRPOTrainer.make_experience",
        lambda self, batch_prompts: dict(base_rollout),
    )

    rollout = trainer.make_experience({})

    assert "masked_input_ids" in rollout
    assert rollout["masked_input_ids"].shape == rollout["input_ids"].shape
