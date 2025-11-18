"""
Test Suite for ThinkRL Datasets
===============================

Tests for:
- thinkrl.data.datasets.RLHFDataset
- thinkrl.data.datasets.PreferenceDataset

"""

import pytest
import torch
import tempfile
import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Check availability for skip logic
try:
    import datasets
    _DATASETS_AVAILABLE = True
except ImportError:
    _DATASETS_AVAILABLE = False

from thinkrl.data.datasets import RLHFDataset, PreferenceDataset

# Mock Tokenizer
class MockTokenizer:
    """A mock tokenizer for testing."""
    def __init__(self, padding_side="right", pad_token_id=0, eos_token="<EOS>", eos_token_id=1):
        self.padding_side = padding_side
        self.pad_token_id = pad_token_id
        self.eos_token = eos_token
        self.eos_token_id = eos_token_id
    
    def __call__(
        self,
        text: str,
        max_length: int = None,
        padding: bool = False,
        truncation: bool = False,
        return_tensors: str = None,
        **kwargs
    ):
        """Mock tokenization."""
        # Simple whitespace tokenizer
        tokens = text.split()
        token_ids = [len(token) for token in tokens] # Use length as token ID
        
        if truncation and max_length and len(token_ids) > max_length:
            if self.padding_side == 'right':
                token_ids = token_ids[:max_length]
            else:
                token_ids = token_ids[-max_length:]
            
        attention_mask = [1] * len(token_ids)
        
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor([token_ids]),
                "attention_mask": torch.tensor([attention_mask]),
            }
        return {"input_ids": [token_ids], "attention_mask": [attention_mask]}

    def decode(self, token_ids, **kwargs):
        return " ".join([str(tid) for tid in token_ids])

@pytest.fixture
def mock_tokenizer():
    """Provides a mock tokenizer instance."""
    return MockTokenizer(padding_side="right")

@pytest.fixture
def temp_jsonl_file():
    """Creates a temporary JSONL file with dummy data."""
    # Use delete=False to manage closure manually for Windows compatibility
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd) # Close the file descriptor immediately
    path = Path(path)
    
    data = [
        {"prompt": "this is prompt 1.", "chosen": "this is chosen 1.", "rejected": "this is rejected 1."},
        {"prompt": "this is prompt 2.", "chosen": "this is chosen 2.", "rejected": "this is rejected 2."},
        {"prompt": "this is prompt 3.", "chosen": "this is chosen 3.", "rejected": "this is rejected 3."},
        {"prompt": "  needs stripping.  ", "chosen": " chosen ", "rejected": " rejected "},
        {"prompt": None, "chosen": "no prompt", "rejected": "no prompt"}, # Invalid
        {}, # Invalid
    ]
    
    try:
        with open(path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        yield path
    finally:
        if path.exists():
            try:
                path.unlink()
            except PermissionError:
                pass # Ignore if still locked on Windows in edge cases

@pytest.mark.skipif(not _DATASETS_AVAILABLE, reason="datasets library not installed")
class TestRLHFDataset:
    """Tests for the RLHFDataset class."""

    @patch('thinkrl.data.datasets.load_dataset')
    def test_init_from_hf(self, mock_load_dataset, mock_tokenizer):
        """Test loading data from HuggingFace datasets."""
        # Note: We expect 2 valid rows
        hf_data = [
            {"prompt": "hf prompt 1."},
            {"prompt": "hf prompt 2."},
            {"prompt": None}, # Invalid
        ]
        # Mock a Dataset object that supports filtering via list comprehension in __init__
        mock_load_dataset.return_value = hf_data
        
        dataset = RLHFDataset(
            dataset_name_or_path="hf/dummy-dataset",
            tokenizer=mock_tokenizer,
            prompt_column="prompt",
        )
        
        mock_load_dataset.assert_called_with("hf/dummy-dataset", split="train")
        assert len(dataset) == 2
        assert dataset.data[0]["prompt"] == "hf prompt 1."

    def test_init_from_jsonl(self, temp_jsonl_file, mock_tokenizer):
        """Test loading data from a JSONL file."""
        dataset = RLHFDataset(
            dataset_name_or_path=str(temp_jsonl_file),
            tokenizer=mock_tokenizer,
            prompt_column="prompt",
        )
        assert len(dataset) == 4 # 3 valid, 1 needs stripping
        assert dataset.data[0]["prompt"] == "this is prompt 1."
        assert dataset.data[3]["prompt"] == "needs stripping." # Stripped and preprocessed

    def test_getitem(self, temp_jsonl_file, mock_tokenizer):
        """Test the __getitem__ method for tokenization."""
        dataset = RLHFDataset(
            dataset_name_or_path=str(temp_jsonl_file),
            tokenizer=mock_tokenizer,
            prompt_column="prompt",
            max_length=10,
        )
        
        sample = dataset[0]
        
        # "this is prompt 1." -> split ["this", "is", "prompt", "1."] -> lens [4, 2, 6, 2]
        expected_ids = torch.tensor([4, 2, 6, 2])
        
        assert "input_ids" in sample
        assert "attention_mask" in sample
        assert "prompt_text" in sample
        assert sample["prompt_text"] == "this is prompt 1."
        assert torch.allclose(sample["input_ids"], expected_ids)
        assert torch.allclose(sample["attention_mask"], torch.ones_like(expected_ids))

    def test_preprocess_fn(self, temp_jsonl_file, mock_tokenizer):
        """Test that the preprocessing function is applied."""
        def add_prefix(sample):
            sample["prompt"] = "PREFIX: " + sample["prompt"]
            return sample

        dataset = RLHFDataset(
            dataset_name_or_path=str(temp_jsonl_file),
            tokenizer=mock_tokenizer,
            prompt_column="prompt",
            preprocess_fn=add_prefix,
        )
        
        sample = dataset[0]
        # "PREFIX: this is prompt 1." -> split ["PREFIX:", "this", "is", "prompt", "1."] -> lens [7, 4, 2, 6, 2]
        expected_ids = torch.tensor([7, 4, 2, 6, 2])
        assert sample["prompt_text"] == "PREFIX: this is prompt 1."
        assert torch.allclose(sample["input_ids"], expected_ids)

    def test_out_of_bounds(self, temp_jsonl_file, mock_tokenizer):
        """Test index out of bounds raises IndexError."""
        dataset = RLHFDataset(
            dataset_name_or_path=str(temp_jsonl_file),
            tokenizer=mock_tokenizer,
        )
        assert len(dataset) == 4
        with pytest.raises(IndexError):
            _ = dataset[10]
        with pytest.raises(IndexError):
            _ = dataset[-5]

@pytest.mark.skipif(not _DATASETS_AVAILABLE, reason="datasets library not installed")
class TestPreferenceDataset:
    """Tests for the PreferenceDataset class."""

    @patch('thinkrl.data.datasets.load_dataset')
    def test_init_from_hf(self, mock_load_dataset, mock_tokenizer):
        """Test loading preference pairs from HuggingFace."""
        hf_data = [
            {"prompt": "hf prompt 1.", "chosen": "good", "rejected": "bad"},
            {"prompt": "hf prompt 2.", "chosen": "better", "rejected": "worse"},
            {"prompt": "hf prompt 3.", "chosen": "best", "rejected": None}, # Invalid
        ]
        mock_load_dataset.return_value = hf_data

        dataset = PreferenceDataset(
            dataset_name_or_path="hf/dummy-pref-dataset",
            tokenizer=mock_tokenizer,
        )
        assert len(dataset) == 2
        assert dataset.data[0]["prompt"] == "hf prompt 1."
        assert dataset.data[0]["chosen"] == "good"
        assert dataset.data[0]["rejected"] == "bad"

    def test_init_from_jsonl(self, temp_jsonl_file, mock_tokenizer):
        """Test loading preference pairs from a JSONL file."""
        dataset = PreferenceDataset(
            dataset_name_or_path=str(temp_jsonl_file),
            tokenizer=mock_tokenizer,
        )
        assert len(dataset) == 4 # 3 valid, 1 needs stripping
        assert dataset.data[0]["prompt"] == "this is prompt 1."
        assert dataset.data[0]["chosen"] == "this is chosen 1."
        assert dataset.data[0]["rejected"] == "this is rejected 1."
        assert dataset.data[3]["chosen"] == "chosen" # Stripped

    def test_getitem_tokenization(self, temp_jsonl_file, mock_tokenizer):
        """Test tokenization of chosen and rejected pairs."""
        dataset = PreferenceDataset(
            dataset_name_or_path=str(temp_jsonl_file),
            tokenizer=mock_tokenizer,
            max_length=20, # Ensure truncation
        )
        
        sample = dataset[0]
        
        # Input: "this is prompt 1.this is chosen 1.<EOS>"
        # The Dataset implementation joins with NO space: f"{prompt}{chosen}{eos}"
        # Prompt: "this is prompt 1."
        # Chosen: "this is chosen 1."
        # Full: "this is prompt 1.this is chosen 1.<EOS>"
        # Tokens (space split in MockTokenizer): "this", "is", "prompt", "1.this", "is", "chosen", "1.<EOS>"
        # IDs: [4, 2, 6, 6, 2, 6, 7]
        
        expected_chosen_ids = torch.tensor([4, 2, 6, 6, 2, 6, 7])
        
        # Rejected: "this is prompt 1.this is rejected 1.<EOS>"
        # Tokens: "this", "is", "prompt", "1.this", "is", "rejected", "1.<EOS>"
        # IDs: [4, 2, 6, 6, 2, 8, 7]
        expected_rejected_ids = torch.tensor([4, 2, 6, 6, 2, 8, 7])
        
        assert "chosen_input_ids" in sample
        assert "chosen_attention_mask" in sample
        assert "rejected_input_ids" in sample
        assert "rejected_attention_mask" in sample
        
        assert torch.allclose(sample["chosen_input_ids"], expected_chosen_ids)
        assert torch.allclose(sample["rejected_input_ids"], expected_rejected_ids)
        assert sample["chosen_attention_mask"].shape == expected_chosen_ids.shape
        assert sample["rejected_attention_mask"].shape == expected_rejected_ids.shape