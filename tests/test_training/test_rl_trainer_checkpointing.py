"""Regression test: the RL trainers must persist progress, not only the final model."""

import inspect

import torch
import torch.nn as nn

import thinkrl.training.grpo_trainer as grpo_trainer
import thinkrl.training.reinforce_pp_trainer as reinforce_pp_trainer
import thinkrl.training.star_trainer as star_trainer
from thinkrl.utils.checkpoint import CheckpointManager, save_training_checkpoint


class _TinyLM(nn.Module):
    def __init__(self, vocab: int = 16, dim: int = 8):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.head = nn.Linear(dim, vocab)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        return {"logits": self.head(self.emb(input_ids))}


TRAINERS = (grpo_trainer.GRPOTrainer, reinforce_pp_trainer.ReinforcePPTrainer, star_trainer.STaRTrainer)


def test_every_rl_trainer_accepts_a_checkpoint_directory():
    """The regression: none of these took a checkpoint argument at all."""
    for trainer in TRAINERS:
        params = inspect.signature(trainer.train).parameters
        assert "checkpoint_dir" in params, f"{trainer.__name__}.train has no checkpoint_dir"
        assert "save_every" in params, f"{trainer.__name__}.train has no save_every"


def test_every_rl_trainer_reaches_the_checkpoint_manager():
    for module in (grpo_trainer, reinforce_pp_trainer, star_trainer):
        source = inspect.getsource(module)
        assert "CheckpointManager" in source, f"{module.__name__} never constructs a CheckpointManager"
        assert "save_training_checkpoint" in source, f"{module.__name__} never saves"


def test_helper_is_a_no_op_without_a_manager():
    assert save_training_checkpoint(None, model=_TinyLM(), step=1) is None


def test_helper_writes_a_loadable_checkpoint(tmp_path):
    manager = CheckpointManager(tmp_path, max_checkpoints=5)
    model = _TinyLM()

    path = save_training_checkpoint(manager, model=model, epoch=0, step=7, metrics={"loss": 1.5})

    assert path is not None and path.exists()
    assert manager.checkpoints, "checkpoint was not registered with the manager"


def test_helper_coerces_tensor_metrics(tmp_path):
    """Trainers collect metrics as a mix of floats and 0-dim tensors."""
    manager = CheckpointManager(tmp_path, max_checkpoints=5)

    path = save_training_checkpoint(
        manager,
        model=_TinyLM(),
        step=1,
        metrics={"loss": torch.tensor(0.5), "reward": 2.0, "grad": torch.ones(3), "name": "ignored"},
    )

    assert path is not None
    saved = manager.checkpoints[-1]["metrics"]
    assert saved == {"loss": 0.5, "reward": 2.0}


def test_rotation_keeps_max_checkpoints(tmp_path):
    manager = CheckpointManager(tmp_path, max_checkpoints=2)
    model = _TinyLM()

    for step in range(4):
        save_training_checkpoint(manager, model=model, step=step)

    assert len(manager.checkpoints) <= 2
