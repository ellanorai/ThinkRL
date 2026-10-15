"""Offline preference training (DPO/IPO).

#124: PPO, DPO, DAPO, VAPO, IPO, PRIME, COPO and Dr.GRPO shipped a loss, a
config dataclass and a factory, but no code path stepped an optimizer with
them. The rollout family (DAPO/VAPO/PRIME/Dr.GRPO) got a generic trainer by
generalizing GRPOTrainer; DPOTrainer is the equivalent for the offline
preference family, since DPO and IPO share the same chosen/rejected batch
shape and training_step contract and differ only in their loss math.
"""

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from thinkrl.algorithms.dpo import DPOAlgorithm, DPOConfig
from thinkrl.algorithms.ipo import IPOAlgorithm, IPOConfig
from thinkrl.training.dpo_trainer import DPOTrainer, DPOTrainerConfig


VOCAB = 50


class TinyLM(nn.Module):
    """Causal LM stand-in: embeds tokens, returns logits over VOCAB."""

    def __init__(self, vocab_size=VOCAB, hidden=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden)
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        hidden = self.embedding(input_ids)
        return {"logits": self.head(hidden)}


class StubTokenizer:
    pad_token_id = 0
    eos_token_id = 1


class PreferencePairDataset(Dataset):
    """Mimics PreferenceDataset's output shape, including prompt_length (#193's
    prompt-masking fix, applied here to the chosen/rejected side)."""

    def __init__(self, n=8):
        self.samples = [
            {
                # First 2 tokens are the shared prompt; prompt_length marks that.
                "chosen_input_ids": torch.tensor([2, 3, 4, 5, 6], dtype=torch.long),
                "chosen_attention_mask": torch.ones(5, dtype=torch.long),
                "rejected_input_ids": torch.tensor([2, 3, 7, 8], dtype=torch.long),
                "rejected_attention_mask": torch.ones(4, dtype=torch.long),
                "prompt_length": 2,
            }
            for _ in range(n)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def _dpo_algorithm(**config_kwargs):
    config_kwargs.setdefault("learning_rate", 1e-2)
    return DPOAlgorithm(
        policy_model=TinyLM(),
        ref_model=TinyLM(),
        config=DPOConfig(**config_kwargs),
    )


def _trainer(algorithm=None, **kwargs):
    kwargs.setdefault("per_device_train_batch_size", 4)
    kwargs.setdefault("num_train_epochs", 1)
    kwargs.setdefault("logging_steps", 1)
    return DPOTrainer(
        algorithm=algorithm or _dpo_algorithm(),
        tokenizer=StubTokenizer(),
        train_dataset=PreferencePairDataset(),
        args=DPOTrainerConfig(**kwargs),
    )


def test_collator_pads_both_sides_to_one_shared_length():
    """compute_loss torch.cat's chosen and rejected along the batch dim, so both
    sides must land at the same sequence length, not each at its own max."""
    trainer = _trainer()
    batch = trainer._default_collator([PreferencePairDataset()[0], PreferencePairDataset()[1]])

    assert batch["chosen_input_ids"].shape == (2, 5)
    assert batch["rejected_input_ids"].shape == (2, 5)
    # prompt_length=2: first two label positions are masked, the rest hold the real tokens
    assert batch["chosen_labels"][0].tolist() == [-100, -100, 4, 5, 6]
    assert batch["rejected_labels"][0].tolist() == [-100, -100, 7, 8, -100]


def test_collator_masks_padding_out_of_labels_too():
    trainer = _trainer()
    short = PreferencePairDataset()[0]
    long = dict(short)
    long["chosen_input_ids"] = torch.tensor([2, 3, 4, 5, 6, 9, 10], dtype=torch.long)
    long["chosen_attention_mask"] = torch.ones(7, dtype=torch.long)

    batch = trainer._default_collator([short, long])
    # short sequence's padded tail (positions 5, 6) must not look like real labels
    assert batch["chosen_labels"][0, 5:].tolist() == [-100, -100]


def test_missing_preference_columns_is_a_clear_error():
    trainer = _trainer()
    with pytest.raises(KeyError, match="chosen_input_ids"):
        trainer._default_collator([{"input_ids": torch.tensor([1, 2])}])


def test_tokenizer_is_required():
    with pytest.raises(ValueError, match="tokenizer"):
        DPOTrainer(algorithm=_dpo_algorithm(), tokenizer=None, train_dataset=PreferencePairDataset())


def test_algorithm_without_training_step_is_rejected():
    with pytest.raises(TypeError, match="training_step"):
        DPOTrainer(algorithm=object(), tokenizer=StubTokenizer(), train_dataset=PreferencePairDataset())


def test_training_runs_and_steps_the_optimizer():
    torch.manual_seed(0)
    trainer = _trainer()
    metrics = trainer.train()

    assert trainer.global_step > 0
    assert torch.isfinite(torch.tensor(metrics["loss"]))


def test_ipo_algorithm_runs_through_the_same_trainer():
    """DPOTrainer isn't DPO-specific: IPO shares the batch shape (#124)."""
    torch.manual_seed(0)
    algorithm = IPOAlgorithm(
        policy_model=TinyLM(),
        ref_model=TinyLM(),
        config=IPOConfig(learning_rate=1e-2),
    )
    trainer = _trainer(algorithm=algorithm)
    metrics = trainer.train()

    assert trainer.global_step > 0
    assert torch.isfinite(torch.tensor(metrics["loss"]))


def test_save_model_writes_a_checkpoint(tmp_path):
    trainer = _trainer()
    out = trainer.save_model(str(tmp_path / "dpo"))
    assert (tmp_path / "dpo" / "model.pt").exists()
    assert out.endswith("dpo")
