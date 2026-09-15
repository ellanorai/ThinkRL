"""Dr.GRPO: 307 lines with no test file of its own until #132.

The defining claim of the algorithm is that it drops the standard-deviation
normalization GRPO applies, to keep the policy-gradient estimator unbiased. That claim
is one subtraction in `compute_advantages`, and nothing checked it.
"""

import pytest
import torch

from tests.test_algorithms.harness import TinyPolicy, assert_trainable_loss, make_batch
from thinkrl.algorithms.dr_grpo import DrGRPOAlgorithm, DrGRPOConfig, create_dr_grpo


def _algorithm(**config_kwargs):
    config_kwargs.setdefault("group_size", 2)
    model = TinyPolicy()
    algorithm = DrGRPOAlgorithm(policy_model=model, config=DrGRPOConfig(**config_kwargs))
    return algorithm, model


def test_compute_loss_is_a_trainable_scalar():
    algorithm, model = _algorithm()

    loss = algorithm.compute_loss(make_batch(batch_size=4))["loss"]

    assert_trainable_loss(loss, model)


def test_advantages_are_centred_but_not_scaled():
    """The whole point of Dr.GRPO: subtract the group mean, do not divide by the group
    standard deviation. Dividing would make these unit-variance and the test fail."""
    algorithm, _ = _algorithm(group_size=4)
    batch = make_batch(batch_size=4)
    batch["rewards"] = torch.tensor([1.0, 2.0, 3.0, 4.0])

    advantages = algorithm.compute_advantages(batch)

    assert advantages.tolist() == pytest.approx([-1.5, -0.5, 0.5, 1.5])
    assert advantages.std(unbiased=False).item() != pytest.approx(1.0, abs=1e-3)


def test_a_batch_that_does_not_divide_into_groups_is_rejected():
    """Silently reshaping would mix rewards across group boundaries, which changes the
    baseline every sample is measured against."""
    algorithm, _ = _algorithm(group_size=4)
    batch = make_batch(batch_size=2)

    with pytest.raises(ValueError, match="not divisible by group_size"):
        algorithm.compute_advantages(batch)


def test_identical_rewards_give_zero_advantage_rather_than_nan():
    """A group where every completion scored the same has no signal. It should produce
    no gradient, not a division by a zero standard deviation."""
    algorithm, _ = _algorithm(group_size=4)
    batch = make_batch(batch_size=4)
    batch["rewards"] = torch.full((4,), 2.5)

    advantages = algorithm.compute_advantages(batch)

    assert torch.isfinite(advantages).all()
    assert advantages.abs().sum().item() == pytest.approx(0.0, abs=1e-6)


def test_factory_builds_the_algorithm():
    algorithm = create_dr_grpo(policy_model=TinyPolicy())

    assert isinstance(algorithm, DrGRPOAlgorithm)
