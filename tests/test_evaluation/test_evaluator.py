"""thinkrl.evaluation was four empty files; these cover the loop that replaced them."""

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from thinkrl.evaluation import EvalResult, Evaluator, contains_match, exact_match, mean


class _StubTokenizer:
    """Character-level stand-in, so the tests need no downloaded tokenizer."""

    pad_token_id = 0

    def __call__(self, texts, return_tensors=None, padding=True, truncation=True):
        ids = [[min(ord(c) % 30 + 1, 30) for c in t] for t in texts]
        width = max(len(row) for row in ids)
        padded = [row + [0] * (width - len(row)) for row in ids]
        mask = [[1] * len(row) + [0] * (width - len(row)) for row in ids]
        return {"input_ids": torch.tensor(padded), "attention_mask": torch.tensor(mask)}

    def batch_decode(self, sequences, skip_special_tokens=True):
        return ["".join(chr(int(i) % 26 + 97) for i in row if int(i) != 0) for row in sequences]


def _model():
    return GPT2LMHeadModel(GPT2Config(n_layer=1, n_head=1, n_embd=8, n_positions=64, vocab_size=32))


def _evaluator(reward_fn=None):
    return Evaluator(model=_model(), tokenizer=_StubTokenizer(), reward_fn=reward_fn, device="cpu")


def test_metrics_helpers():
    assert exact_match(["a", "b"], ["a", "c"]) == 0.5
    assert exact_match([" a "], ["a"]) == 1.0
    assert contains_match(["the answer is 42"], ["42"]) == 1.0
    assert mean([]) == 0.0
    assert mean([1.0, 3.0]) == 2.0


def test_metrics_helpers_reject_mismatched_lengths():
    with pytest.raises(ValueError, match="differ in length"):
        exact_match(["a"], ["a", "b"])


def test_empty_input_returns_an_empty_result():
    result = _evaluator().evaluate([])

    assert isinstance(result, EvalResult)
    assert result.num_samples == 0
    assert result.metrics == {}


def test_generation_produces_one_completion_per_prompt():
    completions = _evaluator().generate(["ab", "cd", "ef"], batch_size=2, max_new_tokens=4)

    assert len(completions) == 3


def test_reward_function_is_summarised():
    def reward_fn(prompts, completions):
        return torch.tensor([1.0, 3.0])

    result = _evaluator(reward_fn).evaluate(["ab", "cd"], max_new_tokens=2)

    assert result.metrics["reward_mean"] == 2.0
    assert result.metrics["reward_std"] == pytest.approx(1.0)


def test_wrong_length_reward_is_rejected_rather_than_broadcast():
    def result_fn(prompts, completions):
        return torch.tensor([1.0])

    with pytest.raises(ValueError, match="reward_fn returned"):
        _evaluator(result_fn).evaluate(["ab", "cd"], max_new_tokens=2)


def test_targets_produce_match_metrics():
    result = _evaluator().evaluate(["ab", "cd"], targets=["zz", "yy"], max_new_tokens=2)

    assert "exact_match" in result.metrics
    assert "contains_match" in result.metrics


def test_mismatched_targets_are_rejected():
    with pytest.raises(ValueError, match="differ in length"):
        _evaluator().evaluate(["ab"], targets=["a", "b"])


def test_model_training_mode_is_restored():
    evaluator = _evaluator()
    evaluator.model.train()

    evaluator.evaluate(["ab"], max_new_tokens=2)

    assert evaluator.model.training, "evaluate left the model in eval mode"
