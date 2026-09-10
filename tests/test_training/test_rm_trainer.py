"""Reward model training.

Step 2 of the RLHF pipeline had no implementation: models/reward_model.py was
architecture only and PRMTrainer is a stub, so the reward checkpoint every RL
example needs could not be produced by this library.
"""

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from thinkrl.training.rm_trainer import RMConfig, RMTrainer, create_rm_trainer


class TinyRewardModel(nn.Module):
    """Scores a sequence with one number, like RewardModel does."""

    def __init__(self, vocab_size=50, hidden=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden)
        self.head = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        hidden = self.embedding(input_ids)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        else:
            pooled = hidden.mean(1)
        return self.head(pooled).squeeze(-1)


class StubTokenizer:
    pad_token_id = 0
    eos_token_id = 1


class PairDataset(Dataset):
    """Chosen sequences are longer than rejected, so padding is exercised."""

    def __init__(self, n=8):
        self.samples = [
            {
                "chosen_input_ids": torch.tensor([2, 3, 4, 5, 6], dtype=torch.long),
                "chosen_attention_mask": torch.ones(5, dtype=torch.long),
                "rejected_input_ids": torch.tensor([7, 8, 9], dtype=torch.long),
                "rejected_attention_mask": torch.ones(3, dtype=torch.long),
            }
            for _ in range(n)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def _trainer(**kwargs):
    kwargs.setdefault("per_device_train_batch_size", 4)
    kwargs.setdefault("num_train_epochs", 1)
    kwargs.setdefault("logging_steps", 1)
    args = RMConfig(**kwargs)
    return RMTrainer(
        model=TinyRewardModel(),
        tokenizer=StubTokenizer(),
        train_dataset=PairDataset(),
        args=args,
    )


def test_collator_pads_both_halves_to_one_length():
    trainer = _trainer()
    batch = trainer._default_collator([PairDataset()[0], PairDataset()[1]])
    assert batch["input_ids"].shape == (4, 5)
    assert batch["attention_mask"].shape == (4, 5)
    # rejected sequences are 3 long, so their tail is masked
    assert batch["attention_mask"][2].tolist() == [1, 1, 1, 0, 0]
    assert batch["input_ids"][2].tolist()[3:] == [0, 0]


def test_chosen_and_rejected_stay_in_their_halves():
    trainer = _trainer()
    batch = trainer._default_collator([PairDataset()[0], PairDataset()[1]])
    first_half = batch["input_ids"][:2]
    second_half = batch["input_ids"][2:]
    assert (first_half[:, 0] == 2).all(), "chosen sequences are not the first half"
    assert (second_half[:, 0] == 7).all(), "rejected sequences are not the second half"


def test_loss_and_metrics_are_produced():
    trainer = _trainer()
    batch = next(iter(trainer.get_train_dataloader()))
    loss, metrics = trainer.compute_loss(batch)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert loss.requires_grad
    assert set(metrics) >= {"rm_loss", "rm_accuracy", "reward_diff"}
    assert 0.0 <= float(metrics["rm_accuracy"]) <= 1.0


def test_training_makes_the_model_score_chosen_above_rejected():
    """Asserts on the model itself, not on the metric the trainer reports.

    Checking reward_diff alone passes even if chosen and rejected are swapped
    inside compute_loss, because that metric is computed from the same swapped
    variables. Scoring the two sequences directly is what pins the direction.
    """
    torch.manual_seed(0)
    trainer = _trainer(learning_rate=0.1, num_train_epochs=30)
    sample = PairDataset()[0]
    chosen = sample["chosen_input_ids"].unsqueeze(0)
    rejected = sample["rejected_input_ids"].unsqueeze(0)

    trainer.train()

    trainer.model.eval()
    with torch.no_grad():
        chosen_score = trainer.model(chosen, torch.ones_like(chosen))
        rejected_score = trainer.model(rejected, torch.ones_like(rejected))

    assert float(chosen_score) > float(rejected_score), "training did not rank chosen above rejected"
    assert trainer.global_step > 0


def test_missing_preference_columns_is_a_clear_error():
    trainer = _trainer()
    with pytest.raises(KeyError, match="chosen_input_ids"):
        trainer._default_collator([{"input_ids": torch.tensor([1, 2])}])


def test_tokenizer_is_required():
    with pytest.raises(ValueError, match="tokenizer"):
        RMTrainer(model=TinyRewardModel(), tokenizer=None, train_dataset=PairDataset())


def test_save_model_writes_a_checkpoint(tmp_path):
    trainer = _trainer()
    out = trainer.save_model(str(tmp_path / "rm"))
    assert (tmp_path / "rm" / "model.pt").exists()
    assert out.endswith("rm")


def test_factory_splits_config_arguments():
    trainer = create_rm_trainer(
        model=TinyRewardModel(),
        tokenizer=StubTokenizer(),
        train_dataset=PairDataset(),
        learning_rate=3e-5,
        margin=0.5,
    )
    assert trainer.args.learning_rate == 3e-5
    assert trainer.loss_fn.margin == 0.5
