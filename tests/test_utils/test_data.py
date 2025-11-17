"""
Test Suite for ThinkRL Data Package
===================================

Tests for:
- thinkrl.data.datasets (RLHFDataset, PreferenceDataset)
- thinkrl.data.loaders (RLHFDataLoader, collate_fn)
- thinkrl.data.processors (process_image, process_audio)

Author: Archit Sood
"""

import pytest
import torch
import tempfile
import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Mock optional dependencies
try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

try:
    import librosa
    _LIBROSA_AVAILABLE = True
except ImportError:
    _LIBROSA_AVAILABLE = False

# Modules under test
from thinkrl.data.datasets import RLHFDataset, PreferenceDataset, BaseRLHFDataset
from thinkrl.data.loaders import RLHFDataLoader, create_rlhf_collate_fn
from thinkrl.data.processors import process_image, process_audio

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

# Fixtures

@pytest.fixture
def mock_tokenizer():
    """Provides a mock tokenizer instance."""
    return MockTokenizer(padding_side="left") # Default to left padding for generation

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

@pytest.fixture
def temp_image_file():
    """Creates a temporary dummy image file."""
    if not _PIL_AVAILABLE:
        pytest.skip("Pillow (PIL) is not installed.")
        
    fd, path_str = tempfile.mkstemp(suffix=".png")
    os.close(fd) # Close the file descriptor
    path = Path(path_str)
    try:
        img = Image.new('RGB', (60, 30), color = 'red')
        img.save(path)
        yield path
    finally:
        if path.exists():
            try:
                path.unlink()
            except PermissionError:
                pass

@pytest.fixture
def temp_audio_file(tmp_path):
    """Creates a temporary dummy audio file."""
    if not _LIBROSA_AVAILABLE:
        pytest.skip("librosa is not installed.")
        
    pytest.importorskip("soundfile") # Need soundfile to write
    import soundfile as sf
    import numpy as np
    
    path = tmp_path / "test.wav"
    samplerate = 22050
    data = np.random.uniform(-0.5, 0.5, size=samplerate) # 1 second of noise
    try:
        sf.write(path, data, samplerate)
        yield path
    finally:
        if path.exists():
            try:
                path.unlink()
            except PermissionError:
                pass

# --- Test Classes ---

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
        # Split: ["this", "is", "prompt", "1.this", "is", "chosen", "1.<EOS>"] (Space tokenizer logic)
        # Lengths: [4, 2, 6, 6, 2, 6, 7]
        # Wait, the Dataset implementation joins with NO space: f"{prompt}{chosen}{eos}"
        # Prompt: "this is prompt 1."
        # Chosen: "this is chosen 1."
        # Full: "this is prompt 1.this is chosen 1.<EOS>"
        # Tokens (space split): "this", "is", "prompt", "1.this", "is", "chosen", "1.<EOS>"
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


class TestRLHFDataLoader:
    """Tests for the RLHFDataLoader and its collate function."""

    @pytest.fixture
    def rlhf_dataset(self, mock_tokenizer):
        """Creates a dummy RLHFDataset for loader tests."""
        # Create a dummy class to avoid file/HF dependency
        class DummyRLHFDataset(BaseRLHFDataset):
            def __init__(self, tokenizer):
                # Pass None as dataset_name_or_path to bypass load_dataset
                super().__init__(tokenizer, None, 20)
                self.data = [
                    {"prompt": "short prompt"},
                    {"prompt": "a much longer prompt here"},
                    {"prompt": "prompt three"},
                ]
                self.prompt_column = "prompt"
                # Manually set dataset to list since BaseRLHFDataset init skipped it
                self.dataset = self.data 
            
            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self._tokenize_sample(self.data[idx])

            def _tokenize_sample(self, sample):
                text = sample["prompt"]
                tokenized = self.tokenizer(
                    text,
                    max_length=self.max_length,
                    truncation=True,
                    return_tensors="pt"
                )
                return {
                    "prompt_text": text,
                    "input_ids": tokenized["input_ids"].squeeze(0),
                    "attention_mask": tokenized["attention_mask"].squeeze(0),
                }
        
        return DummyRLHFDataset(mock_tokenizer)

    def test_collate_fn_left_padding(self, rlhf_dataset, mock_tokenizer):
        """Test that the collate function performs left padding correctly."""
        collate_fn = create_rlhf_collate_fn(mock_tokenizer, padding_side="left")
        
        batch_samples = [rlhf_dataset[0], rlhf_dataset[1], rlhf_dataset[2]]
        
        # Manually get expected token IDs using MockTokenizer logic (whitespace split length)
        # "short prompt" -> [5, 6]
        # "a much longer prompt here" -> [1, 4, 6, 6, 4] (longest, len 5)
        # "prompt three" -> [6, 5]
        
        collated_batch = collate_fn(batch_samples)
        
        expected_input_ids = torch.tensor([
            [0, 0, 0, 5, 6],  # [pad, pad, pad, 5, 6]
            [1, 4, 6, 6, 4],  # [1, 4, 6, 6, 4]
            [0, 0, 0, 6, 5],  # [pad, pad, pad, 6, 5]
        ])
        
        expected_attn_mask = torch.tensor([
            [0, 0, 0, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1],
        ])
        
        assert "input_ids" in collated_batch
        assert "attention_mask" in collated_batch
        assert "prompt_text" in collated_batch # Changed from 'prompt_texts' to 'prompt_text'
        assert collated_batch["prompt_text"] == [ # Changed from 'prompt_texts' to 'prompt_text'
            "short prompt",
            "a much longer prompt here",
            "prompt three"
        ]
        
        assert torch.allclose(collated_batch["input_ids"], expected_input_ids)
        assert torch.allclose(collated_batch["attention_mask"], expected_attn_mask)

    def test_dataloader_wrapper(self, rlhf_dataset, mock_tokenizer):
        """Test the RLHFDataLoader wrapper with drop_last=True (default)."""
        loader = RLHFDataLoader(
            dataset=rlhf_dataset,
            tokenizer=mock_tokenizer,
            batch_size=2,
            shuffle=False,
            drop_last=True # Explicitly setting default
        )
        
        assert len(loader) == 1 # 3 samples, batch_size=2, drop_last=True
        
        batch_iter = iter(loader)
        first_batch = next(batch_iter)
        
        # First batch has "short prompt" and "a much longer prompt here"
        # Longest is [1, 4, 6, 6, 4] (len 5)
        # Default padding side is "right" in DataLoader unless specified
        # RLHFDataLoader uses "right" by default in my implementation
        expected_input_ids_b1 = torch.tensor([
            [5, 6, 0, 0, 0],
            [1, 4, 6, 6, 4],
        ])
        
        assert torch.allclose(first_batch["input_ids"], expected_input_ids_b1)
        
        with pytest.raises(StopIteration):
            next(batch_iter) # No second batch

    def test_dataloader_wrapper_no_drop_last(self, rlhf_dataset, mock_tokenizer):
        """Test the RLHFDataLoader wrapper with drop_last=False."""
        loader = RLHFDataLoader(
            dataset=rlhf_dataset,
            tokenizer=mock_tokenizer,
            batch_size=2,
            shuffle=False,
            drop_last=False, # Override default
        )
        
        assert len(loader) == 2 # 3 samples, batch_size=2, drop_last=False
        
        batch_iter = iter(loader)
        _ = next(batch_iter)
        second_batch = next(batch_iter)

        # Second batch has "prompt three" -> [6, 5]
        # Padded to its own max length (which is 2)
        expected_input_ids_b2 = torch.tensor([
            [6, 5],
        ])
        assert torch.allclose(second_batch["input_ids"], expected_input_ids_b2)
        assert torch.allclose(second_batch["attention_mask"], torch.tensor([[1, 1]]))


class TestProcessors:
    """Tests for multimodal data processors."""

    @pytest.mark.skipif(not _PIL_AVAILABLE, reason="Pillow (PIL) is not installed.")
    def test_process_image(self, temp_image_file):
        """Test basic image loading."""
        img = process_image(str(temp_image_file))
        assert img is not None
        assert isinstance(img, Image.Image)
        assert img.size == (60, 30)

    @pytest.mark.skipif(not _PIL_AVAILABLE, reason="Pillow (PIL) is not installed.")
    def test_process_image_with_transform(self, temp_image_file):
        """Test image loading with a mock transform."""
        mock_transform = MagicMock(return_value="transformed_image")
        img = process_image(str(temp_image_file), transform=mock_transform)
        
        assert img == "transformed_image"
        mock_transform.assert_called_once()
        assert isinstance(mock_transform.call_args[0][0], Image.Image)

    def test_process_image_fail(self, tmp_path):
        """Test graceful failure on non-existent or corrupt image."""
        non_existent_file = tmp_path / "fake.png"
        img = process_image(str(non_existent_file))
        assert img is None

    @pytest.mark.skipif(not _LIBROSA_AVAILABLE, reason="librosa is not installed.")
    def test_process_audio(self, temp_audio_file):
        """Test basic audio loading and resampling."""
        # Mock librosa.load and librosa.resample
        with patch('thinkrl.data.processors.librosa') as mock_librosa:
            mock_librosa.load.return_value = ("fake_waveform_22k", 22050)
            mock_librosa.resample.return_value = "fake_waveform_16k"
            
            audio = process_audio(str(temp_audio_file), sr=16000)
            
            mock_librosa.load.assert_called_with(str(temp_audio_file), sr=None)
            mock_librosa.resample.assert_called_with("fake_waveform_22k", orig_sr=22050, target_sr=16000)
            assert audio == "fake_waveform_16k"

    @pytest.mark.skipif(not _LIBROSA_AVAILABLE, reason="librosa is not installed.")
    def test_process_audio_with_transform(self, temp_audio_file):
        """Test audio loading with a mock transform."""
        mock_transform = MagicMock(return_value="transformed_audio")
        
        with patch('thinkrl.data.processors.librosa') as mock_librosa:
            mock_librosa.load.return_value = ("fake_waveform_16k", 16000)
            
            audio = process_audio(str(temp_audio_file), transform=mock_transform, sr=16000)
            
            mock_librosa.load.assert_called_with(str(temp_audio_file), sr=None)
            mock_transform.assert_called_with("fake_waveform_16k", sampling_rate=16000, return_tensors="pt")
            assert audio == "transformed_audio"

    def test_process_audio_fail(self, tmp_path):
        """Test graceful failure on non-existent audio."""
        non_existent_file = tmp_path / "fake.wav"
        audio = process_audio(str(non_existent_file))
        assert audio is None