"""IPO's dedicated trainer.

#124: IPO shipped a loss, a config, a `create_ipo` factory and a registry
entry, and the only way to train it was to hand-assemble
`DPOTrainer(algorithm=IPOAlgorithm(...))`. Per the shape agreed on #124,
IPOTrainer builds the algorithm itself, so no caller chooses an `algorithm=`.

The TinyLM/StubTokenizer/PreferencePairDataset fixtures are reused from
test_dpo_trainer.py rather than copied: the point of IPOTrainer subclassing
DPOTrainer is that the batch shape is the same one, so the test should be
driving the same shape.
"""

import pytest
import torch

from thinkrl.algorithms.ipo import IPOAlgorithm, IPOConfig
from thinkrl.training.dpo_trainer import DPOTrainer, DPOTrainerConfig
from thinkrl.training.ipo_trainer import IPOTrainer

from .test_dpo_trainer import PreferencePairDataset, StubTokenizer, TinyLM


def _trainer(**config_kwargs):
    config_kwargs.setdefault("learning_rate", 1e-2)
    return IPOTrainer(
        model=TinyLM(),
        ref_model=TinyLM(),
        config=IPOConfig(**config_kwargs),
        tokenizer=StubTokenizer(),
        train_dataset=PreferencePairDataset(),
        args=DPOTrainerConfig(per_device_train_batch_size=4, num_train_epochs=1, logging_steps=1),
    )


def test_ipo_trainer_is_a_dpo_trainer_that_owns_its_algorithm():
    trainer = _trainer()

    assert isinstance(trainer, DPOTrainer)
    assert isinstance(trainer.algorithm, IPOAlgorithm)


def test_config_reaches_the_algorithm_and_the_loss():
    trainer = _trainer(tau=0.25, loss_type="ipo_hinge", length_normalization=True)

    assert trainer.algorithm.config.tau == 0.25
    assert trainer.algorithm.config.length_normalization is True
    # tau is only meaningful if it reached the loss module, where the 1/(2*tau)
    # target actually lives.
    assert trainer.algorithm.loss_fn.tau == 0.25
    assert trainer.algorithm.loss_fn.loss_type == "ipo_hinge"


def test_training_runs_and_steps_the_optimizer():
    torch.manual_seed(0)
    trainer = _trainer()
    metrics = trainer.train()

    assert trainer.global_step > 0
    assert torch.isfinite(torch.tensor(metrics["loss"]))


def test_hinge_loss_trains_too():
    torch.manual_seed(0)
    trainer = _trainer(loss_type="ipo_hinge")
    metrics = trainer.train()

    assert trainer.global_step > 0
    assert torch.isfinite(torch.tensor(metrics["loss"]))


def test_length_normalization_changes_the_loss():
    """The flag is only worth exposing on the CLI if it moves the objective."""
    losses = {}
    for normalize in (False, True):
        torch.manual_seed(0)
        trainer = _trainer(length_normalization=normalize)
        batch = trainer._default_collator([trainer.train_dataset[0], trainer.train_dataset[1]])
        losses[normalize] = trainer.algorithm.compute_loss(batch)["loss"].item()

    assert losses[False] != pytest.approx(losses[True])


def test_reference_model_is_required():
    """IPO's loss is a function of the policy/reference log-ratio, so there is no
    sensible no-reference fallback to pick silently."""
    with pytest.raises(ValueError, match="reference model"):
        IPOTrainer(
            model=TinyLM(),
            ref_model=None,
            tokenizer=StubTokenizer(),
            train_dataset=PreferencePairDataset(),
        )


def test_save_model_writes_a_checkpoint(tmp_path):
    trainer = _trainer()
    out = trainer.save_model(str(tmp_path / "ipo"))

    assert (tmp_path / "ipo" / "model.pt").exists()
    assert out.endswith("ipo")
