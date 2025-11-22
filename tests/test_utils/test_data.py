"""
Test Suite for ThinkRL Dataset Utilities
========================================

Tests for:
- thinkrl.utils.datasets
"""

import pytest
import torch


try:
    import cupy as cp

    _CUPY_AVAILABLE = True
except (ImportError, OSError):
    _CUPY_AVAILABLE = False

from unittest.mock import MagicMock, patch

from thinkrl.utils.datasets import (
    BatchEncoding,
    collate_batch,
    compute_sequence_lengths,
    create_attention_mask,
    create_causal_mask,
    create_dataloader,
    create_labels_for_clm,
    create_position_ids,
    mask_padding_in_loss,
    pad_sequences,
    prepare_batch_for_training,
    preprocess_text,
    shuffle_batch,
    split_batch,
    to_device,
    truncate_sequence,
)


@pytest.fixture
def sample_batch_tensors():
    return {
        "input_ids": torch.tensor([[1, 2], [3, 4]]),
        "attention_mask": torch.tensor([[1, 1], [1, 0]]),
    }


@pytest.fixture
def sample_list_sequences():
    return [
        torch.tensor([1, 2, 3]),
        torch.tensor([4, 5]),
    ]


class TestBatchEncoding:
    def test_init(self):
        data = {"input_ids": torch.tensor([1, 2])}
        be = BatchEncoding(data, encoding="dummy", tensor_type="pt")
        assert be["input_ids"] is data["input_ids"]
        assert be.encoding == "dummy"
        assert be.tensor_type == "pt"

    def test_to_device(self):
        data = {"input_ids": torch.tensor([1, 2]), "meta": "metadata"}
        be = BatchEncoding(data)
        device = torch.device("cpu")
        be_moved = be.to(device)

        assert be_moved is be
        assert isinstance(be["input_ids"], torch.Tensor)
        assert be["meta"] == "metadata"


def test_pad_sequences(sample_list_sequences):
    padded_right = pad_sequences(sample_list_sequences, padding_value=0, padding_side="right")
    expected_right = torch.tensor([[1, 2, 3], [4, 5, 0]])
    assert torch.equal(padded_right, expected_right)

    padded_left = pad_sequences(sample_list_sequences, padding_value=0, padding_side="left")
    expected_left = torch.tensor([[1, 2, 3], [0, 4, 5]])
    assert torch.equal(padded_left, expected_left)

    assert torch.equal(pad_sequences([]), torch.tensor([]))


def test_create_attention_mask():
    input_ids = torch.tensor([[1, 2, 0], [0, 3, 4]])
    mask = create_attention_mask(input_ids, padding_value=0)
    expected = torch.tensor([[1, 1, 0], [0, 1, 1]])
    assert torch.equal(mask, expected)


def test_create_position_ids():
    mask = torch.tensor(
        [
            [1, 1, 0],  # 1, 2, 0
            [0, 1, 1],  # 0, 1, 2
        ]
    )
    pos_ids = create_position_ids(mask)
    expected = torch.tensor([[1, 2, 0], [0, 1, 2]])
    assert torch.equal(pos_ids, expected)


def test_create_causal_mask():
    seq_len = 3
    mask = create_causal_mask(seq_len)
    expected = torch.tensor([[1, 0, 0], [1, 1, 0], [1, 1, 1]]).view(1, 1, 3, 3)
    assert torch.equal(mask, expected.float())


def test_collate_batch():
    batch = [
        {"id": 1, "val": torch.tensor([1, 2]), "label": 0, "score": 1.5},
        {"id": 2, "val": torch.tensor([3, 4]), "label": 1, "score": 2.5},
    ]
    collated = collate_batch(batch)
    assert torch.equal(collated["val"], torch.tensor([[1, 2], [3, 4]]))
    assert torch.equal(collated["label"], torch.tensor([0, 1]))
    assert torch.equal(collated["id"], torch.tensor([1, 2]))
    assert torch.equal(collated["score"], torch.tensor([1.5, 2.5]))

    batch_var = [
        {"val": torch.tensor([1, 2, 3])},
        {"val": torch.tensor([4, 5])},
    ]
    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token_id = 99

    collated_var = collate_batch(batch_var, tokenizer=mock_tokenizer)
    expected = torch.tensor([[1, 2, 3], [4, 5, 99]])
    assert torch.equal(collated_var["val"], expected)

    assert collate_batch([]) == {}

    with patch("thinkrl.utils.datasets.to_device") as mock_to_device:
        mock_device = MagicMock()
        collate_batch(batch, device=mock_device)
        mock_to_device.assert_called()

    batch_2d = [{"img": torch.randn(3, 3)}, {"img": torch.randn(3, 3)}]
    collated_2d = collate_batch(batch_2d)
    assert collated_2d["img"].shape == (2, 3, 3)

    batch_mismatch = [{"img": torch.randn(3, 3)}, {"img": torch.randn(2, 2)}]
    collated_mismatch = collate_batch(batch_mismatch)
    assert isinstance(collated_mismatch["img"], list)
    assert len(collated_mismatch["img"]) == 2

    batch_str = [{"txt": "hello"}, {"txt": "world"}]
    collated_str = collate_batch(batch_str)
    assert collated_str["txt"] == ["hello", "world"]


def test_create_dataloader():
    dataset = [1, 2, 3]
    dl = create_dataloader(dataset, batch_size=1)
    assert len(dl) == 3


def test_preprocess_text():
    assert preprocess_text("  hello  ") == "hello"
    assert preprocess_text(123) == "123"


def test_truncate_sequence():
    seq = [1, 2, 3, 4, 5]
    assert truncate_sequence(seq, 10) == seq
    assert truncate_sequence(seq, 3, side="right") == [1, 2, 3]
    assert truncate_sequence(seq, 3, side="left") == [3, 4, 5]


def test_create_labels_for_clm():
    input_ids = torch.tensor([1, 2, 3])
    labels = create_labels_for_clm(input_ids)
    assert torch.equal(labels, input_ids)
    assert labels is not input_ids


def test_mask_padding_in_loss():
    labels = torch.tensor([1, 2, 3, 0])
    mask = torch.tensor([1, 1, 1, 0])
    masked_labels = mask_padding_in_loss(labels, mask, ignore_index=-100)

    expected = torch.tensor([1, 2, 3, -100])
    assert torch.equal(masked_labels, expected)


def test_split_batch():
    batch = {
        "input_ids": torch.tensor([[1, 1], [2, 2], [3, 3], [4, 4]]),
        "labels": [1, 2, 3, 4],
    }

    micro_batches = split_batch(batch, 2)
    assert len(micro_batches) == 2
    assert torch.equal(micro_batches[0]["input_ids"], torch.tensor([[1, 1], [2, 2]]))
    assert micro_batches[1]["labels"] == [3, 4]

    assert split_batch({}, 2) == [{}]

    batch_meta = {"meta": "data"}
    assert split_batch(batch_meta, 2) == [batch_meta]


def test_compute_sequence_lengths():
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    lengths = compute_sequence_lengths(mask)
    assert torch.equal(lengths, torch.tensor([2, 3]))


def test_shuffle_batch():
    batch = {"a": torch.tensor([1, 2, 3]), "b": ["x", "y", "z"], "c": "constant"}

    torch.manual_seed(42)
    shuffled = shuffle_batch(batch)

    assert len(shuffled["a"]) == 3
    assert len(shuffled["b"]) == 3
    assert shuffled["c"] == "constant"

    assert shuffle_batch({}) == {}


def test_to_device_recursive():
    batch = {
        "tensor": torch.tensor([1]),
        "nested": {"tensor": torch.tensor([2])},
        "list": [1, 2],
    }

    device = torch.device("cpu")
    moved = to_device(batch, device)

    assert isinstance(moved["tensor"], torch.Tensor)
    assert isinstance(moved["nested"]["tensor"], torch.Tensor)
    assert moved["list"] == [1, 2]


def test_prepare_batch_for_training():
    with patch("thinkrl.utils.datasets.to_device") as mock_to:
        prepare_batch_for_training({}, "cpu")
        mock_to.assert_called_once()
