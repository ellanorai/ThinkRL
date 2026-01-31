"""
Test Suite for ThinkRL Data Loaders
===================================

Tests for:
- thinkrl.data.loaders.create_rlhf_collate_fn
- thinkrl.data.loaders.RLHFDataLoader

"""

import pytest
import torch
from torch.utils.data import Dataset

from thinkrl.data.loaders import RLHFDataLoader, create_rlhf_collate_fn


# --- Mocks and Fixtures ---


class MockTokenizer:
    """A simple mock tokenizer for testing padding."""

    def __init__(self, padding_side="right", pad_token_id=0):
        self.padding_side = padding_side
        self.pad_token_id = pad_token_id


@pytest.fixture
def mock_tokenizer_right():
    return MockTokenizer(padding_side="right")


@pytest.fixture
def mock_tokenizer_left():
    return MockTokenizer(padding_side="left")


class SimpleDataset(Dataset):
    """A simple dataset returning dicts of tensors and strings."""

    def __init__(self):
        self.data = [
            {
                "input_ids": torch.tensor([1, 2, 3]),
                "attention_mask": torch.tensor([1, 1, 1]),
                "prompt_text": "short",
            },
            {
                "input_ids": torch.tensor([1, 2, 3, 4, 5]),
                "attention_mask": torch.tensor([1, 1, 1, 1, 1]),
                "prompt_text": "long",
            },
            {
                "input_ids": torch.tensor([1, 2]),
                "attention_mask": torch.tensor([1, 1]),
                "prompt_text": "tiny",
            },
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# --- Test Collate Function ---


def test_collate_fn_right_padding(mock_tokenizer_right):
    """Test collation with right padding (default)."""
    dataset = SimpleDataset()
    batch = [dataset[0], dataset[1], dataset[2]]  # Lengths: 3, 5, 2

    collate_fn = create_rlhf_collate_fn(mock_tokenizer_right, padding_side="right")
    collated = collate_fn(batch)

    # Check keys
    assert "input_ids" in collated
    assert "attention_mask" in collated
    assert "prompt_text" in collated

    # Check dimensions (Batch=3, MaxLen=5)
    assert collated["input_ids"].shape == (3, 5)
    assert collated["attention_mask"].shape == (3, 5)

    # Check padding (0 is pad_token_id)
    # Item 0 (len 3): [1, 2, 3, 0, 0]
    assert torch.equal(collated["input_ids"][0], torch.tensor([1, 2, 3, 0, 0]))
    # Item 2 (len 2): [1, 2, 0, 0, 0]
    assert torch.equal(collated["input_ids"][2], torch.tensor([1, 2, 0, 0, 0]))

    # Check string pass-through
    assert collated["prompt_text"] == ["short", "long", "tiny"]


def test_collate_fn_left_padding(mock_tokenizer_left):
    """Test collation with left padding (generation mode)."""
    dataset = SimpleDataset()
    batch = [dataset[0], dataset[1], dataset[2]]  # Lengths: 3, 5, 2

    collate_fn = create_rlhf_collate_fn(mock_tokenizer_left, padding_side="left")
    collated = collate_fn(batch)

    # Check dimensions
    assert collated["input_ids"].shape == (3, 5)

    # Check padding
    # Item 0 (len 3): [0, 0, 1, 2, 3]
    assert torch.equal(collated["input_ids"][0], torch.tensor([0, 0, 1, 2, 3]))
    # Item 1 (len 5): [1, 2, 3, 4, 5]
    assert torch.equal(collated["input_ids"][1], torch.tensor([1, 2, 3, 4, 5]))
    # Item 2 (len 2): [0, 0, 0, 1, 2]
    assert torch.equal(collated["input_ids"][2], torch.tensor([0, 0, 0, 1, 2]))


def test_collate_fn_empty_batch(mock_tokenizer_right):
    """Test collation with an empty batch."""
    collate_fn = create_rlhf_collate_fn(mock_tokenizer_right)
    collated = collate_fn([])
    assert collated == {}


# --- Test DataLoader ---


def test_dataloader_basic(mock_tokenizer_right):
    """Test basic initialization and iteration of RLHFDataLoader."""
    dataset = SimpleDataset()

    loader = RLHFDataLoader(dataset, mock_tokenizer_right, batch_size=2, shuffle=False, drop_last=False)

    batches = list(loader)
    assert len(batches) == 2  # 3 items / 2 batch_size -> 2 batches

    # First batch (size 2, max len 5)
    b1 = batches[0]
    assert b1["input_ids"].shape == (2, 5)
    assert b1["prompt_text"] == ["short", "long"]

    # Second batch (size 1, max len 2)
    b2 = batches[1]
    assert b2["input_ids"].shape == (1, 2)
    assert b2["prompt_text"] == ["tiny"]


def test_dataloader_drop_last(mock_tokenizer_right):
    """Test drop_last functionality."""
    dataset = SimpleDataset()  # 3 items

    loader = RLHFDataLoader(dataset, mock_tokenizer_right, batch_size=2, drop_last=True)

    batches = list(loader)
    assert len(batches) == 1  # 3 items / 2 batch_size -> 1 batch, 1 dropped
    assert batches[0]["input_ids"].shape[0] == 2


# --- Test Bug Fix: Correct Padding Values ---


class DatasetWithLabels(Dataset):
    """Dataset with labels for testing correct padding values."""

    def __init__(self):
        self.data = [
            {
                "input_ids": torch.tensor([1, 2, 3]),
                "attention_mask": torch.tensor([1, 1, 1]),
                "labels": torch.tensor([10, 20, 30]),
                "prompt_text": "short",
            },
            {
                "input_ids": torch.tensor([4, 5, 6, 7, 8]),
                "attention_mask": torch.tensor([1, 1, 1, 1, 1]),
                "labels": torch.tensor([40, 50, 60, 70, 80]),
                "prompt_text": "long",
            },
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def test_collate_fn_attention_mask_padded_with_zero():
    """Test that attention_mask is padded with 0, not pad_token_id."""
    # Use a non-zero pad_token_id to verify the fix
    tokenizer = MockTokenizer(padding_side="right", pad_token_id=99)
    dataset = DatasetWithLabels()
    batch = [dataset[0], dataset[1]]  # Lengths: 3, 5

    collate_fn = create_rlhf_collate_fn(tokenizer, padding_side="right")
    collated = collate_fn(batch)

    # Attention mask should be padded with 0, not 99
    expected_attn_0 = torch.tensor([1, 1, 1, 0, 0])
    assert torch.equal(
        collated["attention_mask"][0], expected_attn_0
    ), f"Expected {expected_attn_0}, got {collated['attention_mask'][0]}"

    # Verify no 99s in attention mask
    assert (collated["attention_mask"] == 99).sum() == 0, "attention_mask should NOT contain pad_token_id (99)"


def test_collate_fn_labels_padded_with_minus_100():
    """Test that labels are padded with -100, not pad_token_id."""
    # Use a non-zero pad_token_id to verify the fix
    tokenizer = MockTokenizer(padding_side="right", pad_token_id=99)
    dataset = DatasetWithLabels()
    batch = [dataset[0], dataset[1]]  # Lengths: 3, 5

    collate_fn = create_rlhf_collate_fn(tokenizer, padding_side="right")
    collated = collate_fn(batch)

    # Labels should be padded with -100, not 99
    expected_labels_0 = torch.tensor([10, 20, 30, -100, -100])
    assert torch.equal(
        collated["labels"][0], expected_labels_0
    ), f"Expected {expected_labels_0}, got {collated['labels'][0]}"

    # Verify no 99s in labels
    assert (collated["labels"] == 99).sum() == 0, "labels should NOT contain pad_token_id (99)"


def test_collate_fn_input_ids_padded_with_pad_token_id():
    """Test that input_ids IS padded with pad_token_id."""
    tokenizer = MockTokenizer(padding_side="right", pad_token_id=99)
    dataset = DatasetWithLabels()
    batch = [dataset[0], dataset[1]]  # Lengths: 3, 5

    collate_fn = create_rlhf_collate_fn(tokenizer, padding_side="right")
    collated = collate_fn(batch)

    # input_ids should be padded with pad_token_id (99)
    expected_input_0 = torch.tensor([1, 2, 3, 99, 99])
    assert torch.equal(
        collated["input_ids"][0], expected_input_0
    ), f"Expected {expected_input_0}, got {collated['input_ids'][0]}"
