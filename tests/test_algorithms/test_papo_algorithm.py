"""PAPO: 178 lines with no test file of its own until #132.

PAPO is the module that was unreachable entirely until #90 exported it, and it had no
`create_papo` factory until that PR, so nothing in the package had ever constructed it.
"""

import pytest
import torch

from tests.test_algorithms.harness import TinyPolicy, assert_trainable_loss, make_batch
from thinkrl.algorithms.papo import PAPOAlgorithm, PAPOConfig, create_papo


def _algorithm(**config_kwargs):
    config_kwargs.setdefault("group_size", 2)
    model = TinyPolicy()
    algorithm = PAPOAlgorithm(policy_model=model, config=PAPOConfig(**config_kwargs))
    return algorithm, model


def _papo_batch(batch_size: int = 4):
    """PAPO needs a second, corrupted view of the same inputs."""
    batch = make_batch(batch_size=batch_size)
    masked = batch["input_ids"].clone()
    masked[:, 2:] = 0  # blank the completion region, the "perception" ablation
    batch["masked_input_ids"] = masked
    return batch


def test_compute_loss_is_a_trainable_scalar():
    algorithm, model = _algorithm()

    loss = algorithm.compute_loss(_papo_batch())["loss"]

    assert_trainable_loss(loss, model)


def test_missing_masked_inputs_fails_loudly():
    """Implicit Perception Loss is the entire contribution of PAPO. Running without the
    masked view would quietly reduce it to GRPO."""
    algorithm, _ = _algorithm()
    batch = make_batch(batch_size=4)

    with pytest.raises(ValueError, match="requires 'masked_input_ids'"):
        algorithm.compute_loss(batch)


def test_masked_attention_mask_defaults_to_the_unmasked_one():
    """Documented as optional, so omitting it must not change the shape of the run."""
    algorithm, _ = _algorithm()
    batch = _papo_batch()

    without = algorithm.compute_loss(batch)["loss"]

    batch_with = dict(batch)
    batch_with["masked_attention_mask"] = batch["attention_mask"].clone()
    with_explicit = algorithm.compute_loss(batch_with)["loss"]

    assert without.item() == pytest.approx(with_explicit.item(), abs=1e-6)


def test_the_perception_term_actually_depends_on_the_masked_view():
    """If the masked forward pass were ignored, changing it could not move the loss,
    and PAPO would be GRPO wearing a different name."""
    algorithm, _ = _algorithm(gamma=1.0)
    batch = _papo_batch()

    baseline = algorithm.compute_loss(batch)["loss"].item()

    perturbed = dict(batch)
    perturbed["masked_input_ids"] = torch.full_like(batch["masked_input_ids"], 3)
    changed = algorithm.compute_loss(perturbed)["loss"].item()

    assert baseline != pytest.approx(changed, abs=1e-6)


def test_factory_builds_the_algorithm():
    algorithm = create_papo(policy_model=TinyPolicy())

    assert isinstance(algorithm, PAPOAlgorithm)
