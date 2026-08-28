"""SFTTrainer(packing=True) must actually pack.

`self.packing` was assigned in __init__ and never read again, so the flag was
inert: data/packing.py was fully implemented and imported by nothing but its
own tests.
"""

import pytest
import torch
from torch.utils.data import Dataset

from thinkrl.training.sft_trainer import SFTConfig, SFTTrainer


class TinyDataset(Dataset):
    """Ten four-token samples, well under the pack length."""

    def __init__(self, n=10, length=4):
        self.samples = [
            {
                "input_ids": torch.arange(i * length, (i + 1) * length, dtype=torch.long),
                "attention_mask": torch.ones(length, dtype=torch.long),
            }
            for i in range(n)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class StubTokenizer:
    pad_token = "<pad>"
    pad_token_id = 0
    eos_token = "<eos>"
    eos_token_id = 2


def _trainer(**kwargs):
    kwargs.setdefault("tokenizer", StubTokenizer())
    return SFTTrainer(
        args=SFTConfig(per_device_train_batch_size=2, max_seq_length=16),
        train_dataset=TinyDataset(),
        max_seq_length=16,
        **kwargs,
    )


def test_packing_reduces_the_number_of_samples():
    unpacked = _trainer(packing=False).get_train_dataloader()
    packed = _trainer(packing=True).get_train_dataloader()
    assert len(packed.dataset) < len(unpacked.dataset)


def test_packed_samples_fill_the_context():
    loader = _trainer(packing=True).get_train_dataloader()
    batch = next(iter(loader))
    assert batch["input_ids"].shape[1] == 16


def test_position_ids_reach_the_batch():
    loader = _trainer(packing=True).get_train_dataloader()
    batch = next(iter(loader))
    assert "position_ids" in batch, "position_ids were dropped by the collator"


def test_position_ids_restart_per_sub_sequence():
    """A pack holding several samples must not look like one long sequence."""
    packed = _trainer(packing=True)._pack_dataset(TinyDataset())
    positions = packed[0]["position_ids"].tolist()
    assert positions[0] == 0
    assert 0 in positions[1:], "position ids never restart within the pack"


def test_padding_is_not_trained_on():
    """The tail pack is padded out to max_seq_length; that padding must be masked."""
    loader = _trainer(packing=True).get_train_dataloader()
    seen_padding = False
    for batch in loader:
        padded = batch["attention_mask"] == 0
        if padded.any():
            seen_padding = True
            assert (batch["labels"][padded] == -100).all()
    assert seen_padding, "no padded pack in the fixture, test would pass vacuously"


def test_unpacked_path_is_unchanged():
    loader = _trainer(packing=False).get_train_dataloader()
    batch = next(iter(loader))
    assert batch["input_ids"].shape == (2, 4)
    assert "position_ids" not in batch
    assert torch.equal(batch["labels"], batch["input_ids"])


def test_packing_without_tokenizer_raises():
    trainer = SFTTrainer(
        args=SFTConfig(),
        train_dataset=TinyDataset(),
        packing=True,
        tokenizer=None,
    )
    with pytest.raises(ValueError, match="tokenizer"):
        trainer.get_train_dataloader()
