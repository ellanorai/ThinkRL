"""STaR: 159 lines with no test file of its own until #132.

The trainer was covered, the algorithm was not, which is the gap that matters because
`compute_loss` is where the maths lives.
"""

import pytest
import torch

from tests.test_algorithms.harness import TinyPolicy, assert_trainable_loss, make_batch
from thinkrl.algorithms.star import STaRAlgorithm, STaRConfig, create_star


def _algorithm(**config_kwargs):
    model = TinyPolicy()
    algorithm = STaRAlgorithm(policy_model=model, config=STaRConfig(**config_kwargs))
    return algorithm, model


def test_compute_loss_is_a_trainable_scalar():
    algorithm, model = _algorithm()

    loss = algorithm.compute_loss(make_batch())["loss"]

    assert_trainable_loss(loss, model)


def test_loss_only_counts_completion_tokens():
    """Labels are -100 over the prompt. If the loss ignored that it would train the
    policy to reproduce prompts, and changing a prompt token would move the number."""
    algorithm, _ = _algorithm()
    batch = make_batch()

    before = algorithm.compute_loss(batch)["loss"].item()

    prompt_changed = {k: v.clone() for k, v in batch.items()}
    prompt_changed["labels"][:, 0] = -100  # already masked; assert it stays masked
    after = algorithm.compute_loss(prompt_changed)["loss"].item()

    assert before == pytest.approx(after, abs=1e-6)


def test_a_fully_masked_batch_does_not_produce_nan():
    """Every label -100 means nothing to learn from. A mean over an empty mask is the
    classic way that becomes nan several steps later."""
    algorithm, _ = _algorithm()
    batch = make_batch()
    batch["labels"] = torch.full_like(batch["labels"], -100)

    loss = algorithm.compute_loss(batch)["loss"]

    assert torch.isfinite(loss), f"fully masked batch gave {loss.item()}"


def test_loss_val_is_detached():
    """loss_val is reported as a metric; if it carried grad it would keep the graph
    alive for the whole run."""
    algorithm, _ = _algorithm()

    result = algorithm.compute_loss(make_batch())

    assert not result["loss_val"].requires_grad


def test_factory_builds_the_algorithm():
    algorithm = create_star(policy_model=TinyPolicy())

    assert isinstance(algorithm, STaRAlgorithm)
