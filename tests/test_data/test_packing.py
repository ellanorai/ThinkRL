from unittest.mock import MagicMock

import pytest
import torch

from thinkrl.data.packing import (
    PackedDataset,
    PackingConfig,
    StreamingPackedDataset,
    compute_packing_efficiency,
    pack_sequences,
    unpack_sequences,
)


class TestPacking:
    def test_pack_sequences_basic(self):
        """Test basic packing functionality."""
        seqs = [{"input_ids": [1, 2], "attention_mask": [1, 1]}, {"input_ids": [3, 4], "attention_mask": [1, 1]}]
        # Max length 10, eos_token_id=0. Default add_eos_between=True
        # Sequence 1: 1, 2 (len 2). No EOS at start.
        # Sequence 2: 0, 3, 4 (len 3). EOS added between.
        # Packed: 1, 2, 0, 3, 4 + padding (5 zeros) -> len 10
        packed = pack_sequences(seqs, max_seq_length=10, pad_token_id=0, eos_token_id=0, add_eos_between=True)

        assert len(packed) == 1
        assert packed[0]["input_ids"].tolist() == [1, 2, 0, 3, 4, 0, 0, 0, 0, 0]
        # Mask: 1, 1 (seq1), 1 (EOS), 1, 1 (seq2) -> 1, 1, 1, 1, 1. Padding 0s.
        assert packed[0]["attention_mask"].tolist() == [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

    def test_pack_sequences_with_labels(self):
        """Test packing with labels."""
        seqs = [{"input_ids": [1], "labels": [1]}, {"input_ids": [2], "labels": [2]}]
        # eos=99. pad=0
        # seq1: 1. labels: 1
        # seq2: 99, 2. labels: -100, 2
        packed = pack_sequences(seqs, max_seq_length=6, pad_token_id=0, eos_token_id=99, add_eos_between=True)

        assert len(packed) == 1
        assert packed[0]["input_ids"].tolist() == [1, 99, 2, 0, 0, 0]
        assert packed[0]["labels"].tolist() == [1, -100, 2, -100, -100, -100]

    def test_pack_sequences_overflow(self):
        """Test sequences that exceed max length are split into new packs."""
        seqs = [{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5, 6]}]
        # max_len=4.
        # seq1 (with eos if expected): lets say add_eos=False for simplicity
        packed = pack_sequences(seqs, max_seq_length=4, add_eos_between=False)

        # seq1: [1, 2, 3] (len 3). Fits.
        # seq2: [4, 5, 6] (len 3). 3+3=6 > 4. New pack.
        assert len(packed) == 2
        assert packed[0]["input_ids"].tolist() == [1, 2, 3, 0]
        assert packed[1]["input_ids"].tolist() == [4, 5, 6, 0]

    def test_packed_dataset(self):
        """Test PackedDataset class."""
        seqs = [{"input_ids": torch.tensor([1, 2])}, {"input_ids": torch.tensor([3, 4])}]
        ds = PackedDataset(seqs, max_seq_length=10, pre_pack=True)

        assert len(ds) == 1
        item = ds[0]
        assert "input_ids" in item
        assert isinstance(item["input_ids"], torch.Tensor)

    def test_streaming_packed_dataset(self):
        """Test StreamingPackedDataset."""
        data = ["hello", "world"]
        mock_tokenizer = MagicMock()
        mock_tokenizer.side_effect = lambda x, **kwargs: {
            "input_ids": torch.tensor([1, 2]),
            "attention_mask": torch.tensor([1, 1]),
        }
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 2

        ds = StreamingPackedDataset(data, mock_tokenizer, max_seq_length=10, buffer_size=2)

        # Iterate
        packed_items = list(ds)
        assert len(packed_items) > 0  # Should yield at least one pack

    def test_compute_packing_efficiency(self):
        """Test efficiency calculation."""
        seqs = [{"input_ids": torch.tensor([1]), "attention_mask": torch.tensor([1])}]
        packed = [{"input_ids": torch.tensor([1, 0]), "attention_mask": torch.tensor([1, 0])}]  # 50% eff

        metrics = compute_packing_efficiency(seqs, packed)
        assert metrics["token_efficiency"] == 0.5
        assert metrics["compression_ratio"] == 1.0

    def test_unpack_sequences(self):
        """Test unpacking."""
        packed = {"input_ids": torch.tensor([2, 1, 2, 3, 0]), "attention_mask": torch.tensor([1, 1, 1, 1, 0])}
        # eos=2.
        # First: 2 (eos)
        # Second: 1
        # Third: 2 (eos)
        # wait, current logic:
        # if token==eos and current_ids: end seq.
        # if mask==0 break.

        # [2]: eos. current_ids empty. append? Unpack logic depends on impl.
        # The impl:
        # loop:
        #  if mask==0 break
        #  if token==eos and current_ids: append sequence, clear.
        #  else: append to current_ids.

        # Trace:
        # 1. token=2 (eos). current_ids=[] -> append 2 to current_ids=[2], mask=[1]
        # 2. token=1. curr=[2, 1]
        # 3. token=2. curr=[2, 1, 2]. if token==eos and current? No wait.
        # "if token == eos_token_id and current_ids:"
        #  Yes, current_ids is [2, 1]. So append sequence [2, 1], clear.
        # 4. token=3. curr=[3]
        # 5. token=0. mask=0. break.
        # Final: append [3].

        unpacked = unpack_sequences(packed, eos_token_id=2)
        assert len(unpacked) == 2
        # Note: logic seems to strip the trailing EOS from the sequence if it triggers the split?
        # Actually logic says: "else: current_ids.append". So EOS is NOT appended to current_ids if it triggers split.
        # But the first EOS (index 0) was appended because current_ids was empty.

    def test_pack_sequences_empty(self):
        assert pack_sequences([]) == []
