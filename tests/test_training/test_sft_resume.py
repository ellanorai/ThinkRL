"""Regression test: resume_from_checkpoint must actually restore the run."""

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from thinkrl.training.sft_trainer import TRAINER_STATE_FILE, SFTConfig, SFTTrainer


def _model():
    return GPT2LMHeadModel(GPT2Config(n_layer=1, n_head=1, n_embd=8, n_positions=16, vocab_size=32))


def _trainer(tmp_path):
    trainer = SFTTrainer(
        model=_model(),
        args=SFTConfig(output_dir=str(tmp_path), num_train_epochs=3),
        train_dataset=[{"input_ids": [1, 2, 3], "labels": [1, 2, 3]}],
    )
    trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=1e-4)
    trainer.scheduler = torch.optim.lr_scheduler.StepLR(trainer.optimizer, step_size=1)
    return trainer


def test_save_model_writes_the_trainer_state(tmp_path):
    trainer = _trainer(tmp_path)
    trainer.global_step = 42
    trainer.epoch = 1

    trainer.save_model(str(tmp_path / "checkpoint-42"))

    assert (tmp_path / "checkpoint-42" / TRAINER_STATE_FILE).exists()


def test_resume_restores_step_epoch_and_optimizer(tmp_path):
    saved = _trainer(tmp_path)
    saved.global_step = 42
    saved.epoch = 1
    # Take one real optimizer step so the saved state is non-trivial.
    loss = saved.model(input_ids=torch.tensor([[1, 2, 3]]), labels=torch.tensor([[1, 2, 3]])).loss
    loss.backward()
    saved.optimizer.step()
    saved.save_model(str(tmp_path / "checkpoint-42"))

    resumed = _trainer(tmp_path)
    assert resumed.global_step == 0

    resumed._load_trainer_state(str(tmp_path / "checkpoint-42"))

    assert resumed.global_step == 42
    assert resumed.start_epoch == 1
    assert resumed.optimizer.state_dict()["state"], "optimizer moments were not restored"


def test_resume_restores_the_weights(tmp_path):
    saved = _trainer(tmp_path)
    with torch.no_grad():
        saved.model.transformer.wte.weight.fill_(0.25)
    saved.save_model(str(tmp_path / "checkpoint-1"))

    resumed = _trainer(tmp_path)
    assert not torch.allclose(resumed.model.transformer.wte.weight, torch.full_like(resumed.model.transformer.wte.weight, 0.25))

    resumed._load_trainer_state(str(tmp_path / "checkpoint-1"))

    torch.testing.assert_close(
        resumed.model.transformer.wte.weight,
        torch.full_like(resumed.model.transformer.wte.weight, 0.25),
    )


def test_missing_directory_raises_rather_than_restarting(tmp_path):
    trainer = _trainer(tmp_path)

    with pytest.raises(FileNotFoundError, match="does not exist"):
        trainer._load_trainer_state(str(tmp_path / "nope"))


def test_checkpoint_without_trainer_state_raises(tmp_path):
    """A bare save_pretrained directory cannot restore a run, so it must not pretend to."""
    trainer = _trainer(tmp_path)
    bare = tmp_path / "bare"
    trainer.model.save_pretrained(str(bare))

    with pytest.raises(FileNotFoundError, match=TRAINER_STATE_FILE):
        trainer._load_trainer_state(str(bare))


def test_train_signature_argument_is_read():
    """The regression: the parameter used to appear only in the signature and docstring."""
    import inspect

    source = inspect.getsource(SFTTrainer.train)
    body = source.split('"""')[2]
    assert "resume_from_checkpoint" in body
