from unittest.mock import MagicMock

import pytest

from thinkrl.data.datasets import PreferenceDataset, RLHFDataset


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.return_value = {"input_ids": MagicMock(), "attention_mask": MagicMock()}
    return tokenizer


def test_rlhf_dataset_max_samples(mock_tokenizer):
    """Test max_samples limit."""
    # Mock dataset with 10 samples
    mock_data = [{"prompt": f"p{i}"} for i in range(10)]

    # Init with limit 5
    dataset = RLHFDataset(dataset_name_or_path=mock_data, tokenizer=mock_tokenizer, max_samples=5)

    assert len(dataset) == 5
    assert len(dataset.data) == 5
    assert dataset.data[-1]["prompt"] == "p4"


def test_preference_dataset_max_samples(mock_tokenizer):
    """Test max_samples limit on preference dataset."""
    mock_data = [{"prompt": f"p{i}", "chosen": "c", "rejected": "r"} for i in range(10)]

    dataset = PreferenceDataset(dataset_name_or_path=mock_data, tokenizer=mock_tokenizer, max_samples=3)

    assert len(dataset) == 3
