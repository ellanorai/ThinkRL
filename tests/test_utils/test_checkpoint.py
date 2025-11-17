"""
Test Suite for ThinkRL Tokenizer Utilities
==========================================

Tests for:
- get_tokenizer
- tokenize_text, tokenize_batch
- decode_tokens
- get_special_tokens, add_special_tokens
- tokenize_conversation
- prepare_input_for_generation
- count_tokens
- truncate_to_token_limit
- get_tokenizer_info
- save_tokenizer, load_tokenizer

Author: Gemini
"""

import pytest
import torch
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

# Check for optional dependencies
try:
    from transformers import (
        AutoTokenizer,
        PreTrainedTokenizer,
        PreTrainedTokenizerFast,
    )
    from thinkrl.utils.tokenizer import (
        TokenizerConfig,
        get_tokenizer,
        tokenize_text,
        tokenize_batch,
        decode_tokens,
        get_special_tokens,
        add_special_tokens,
        tokenize_conversation,
        prepare_input_for_generation,
        count_tokens,
        truncate_to_token_limit,
        get_tokenizer_info,
        save_tokenizer,
        load_tokenizer,
    )
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False

# Skip all tests in this file if transformers is not installed
pytestmark = pytest.mark.skipif(
    not _TRANSFORMERS_AVAILABLE, reason="transformers not installed"
)

# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def gpt2_tokenizer():
    """Provides a real GPT-2 tokenizer for testing."""
    return get_tokenizer("gpt2", padding_side="right")

@pytest.fixture
def temp_dir():
    """Create a temporary directory for saving/loading."""
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d)

# ============================================================================
# Tokenizer Tests
# ============================================================================

class TestTokenizers:
    """Test tokenizer utilities (requires transformers)."""

    def test_get_tokenizer(self, gpt2_tokenizer):
        """Test tokenizer loading and default pad token setting."""
        assert gpt2_tokenizer is not None
        assert gpt2_tokenizer.vocab_size == 50257
        # gpt2 doesn't have a pad token by default, check it was set to eos
        assert gpt2_tokenizer.pad_token == gpt2_tokenizer.eos_token
        assert gpt2_tokenizer.padding_side == "right"

    def test_get_tokenizer_left_padding(self):
        """Test setting padding_side."""
        tokenizer = get_tokenizer("gpt2", padding_side="left")
        assert tokenizer.padding_side == "left"

    def test_tokenize_text(self, gpt2_tokenizer):
        """Test text tokenization for single string and batch."""
        text = "Hello, world!"
        encoded = tokenize_text(
            text, gpt2_tokenizer, max_length=10, padding="max_length", truncation=True
        )
        
        assert "input_ids" in encoded
        assert "attention_mask" in encoded
        assert encoded["input_ids"].shape == (1, 10)
        assert encoded["attention_mask"].shape == (1, 10)

        # Test batch
        texts = ["Hello!", "How are you?"]
        encoded_batch = tokenize_text(
            texts, gpt2_tokenizer, padding=True, return_tensors="pt"
        )
        assert encoded_batch["input_ids"].shape[0] == 2
        assert encoded_batch["input_ids"].shape[1] >= 3 # "How are you?" is 3 tokens

    def test_tokenize_batch(self, gpt2_tokenizer):
        """Test efficient batch tokenization."""
        texts = ["Text 1", "Text 2", "A slightly longer text"] * 10
        
        # Test with batch_size
        encoded = tokenize_batch(
            texts, 
            gpt2_tokenizer, 
            max_length=10, 
            padding="max_length", 
            batch_size=8
        )
        
        assert "input_ids" in encoded
        assert "attention_mask" in encoded
        assert encoded["input_ids"].shape == (30, 10)
        assert encoded["attention_mask"].shape == (30, 10)

    def test_decode_tokens(self, gpt2_tokenizer):
        """Test token decoding for single and batch."""
        text = "Hello, world!"
        token_ids = gpt2_tokenizer.encode(text)

        # Test single decode
        decoded_text = decode_tokens(token_ids, gpt2_tokenizer, skip_special_tokens=True)
        assert decoded_text == text

        # Test batch decode
        texts = ["Hello!", "How are you?"]
        batch_ids = [gpt2_tokenizer.encode(t) for t in texts]
        decoded_batch = decode_tokens(batch_ids, gpt2_tokenizer, skip_special_tokens=True)
        
        assert isinstance(decoded_batch, list)
        assert len(decoded_batch) == 2
        assert decoded_batch[0] == "Hello!"
        assert decoded_batch[1] == "How are you?"

    def test_get_special_tokens(self, gpt2_tokenizer):
        """Test special tokens extraction."""
        special_tokens = get_special_tokens(gpt2_tokenizer)

        assert "pad_token" in special_tokens
        assert "eos_token" in special_tokens
        assert "bos_token" in special_tokens
        assert special_tokens["pad_token_id"] is not None
        assert special_tokens["pad_token_id"] == 50256

    def test_add_special_tokens(self):
        """Test adding new special tokens."""
        # Load a fresh tokenizer
        tokenizer = get_tokenizer("gpt2")
        original_vocab_size = len(tokenizer)
        
        num_added = add_special_tokens(
            tokenizer,
            {"additional_special_tokens": ["<|user|>", "<|assistant|>"]}
        )
        
        assert num_added == 2
        assert len(tokenizer) == original_vocab_size + 2
        assert tokenizer.convert_tokens_to_ids("<|user|>") == original_vocab_size
        assert tokenizer.convert_tokens_to_ids("<|assistant|>") == original_vocab_size + 1

    def test_tokenize_conversation_manual(self, gpt2_tokenizer):
        """Test conversation tokenization using manual formatting."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi!"},
        ]

        # Test without generation prompt
        encoded = tokenize_conversation(
            messages,
            gpt2_tokenizer,
            system_prefix="SYS: ",
            user_prefix="USER: ",
            assistant_prefix="ASSIST: ",
            separator="\n",
            add_generation_prompt=False
        )
        
        decoded = decode_tokens(encoded["input_ids"].squeeze(0), gpt2_tokenizer)
        assert decoded == "SYS: You are helpful.\nUSER: Hello!\nASSIST: Hi!"

        # Test with generation prompt
        encoded_gen = tokenize_conversation(
            messages,
            gpt2_tokenizer,
            system_prefix="SYS: ",
            user_prefix="USER: ",
            assistant_prefix="ASSIST: ",
            separator="\n",
            add_generation_prompt=True
        )
        
        decoded_gen = decode_tokens(encoded_gen["input_ids"].squeeze(0), gpt2_tokenizer)
        assert decoded_gen.endswith("\nASSIST: ")

    def test_tokenize_conversation_chat_template(self):
        """Test that apply_chat_template is used if available."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = "fake_template"
        mock_tokenizer.apply_chat_template.return_value = "Template output"
        
        # Mock tokenize_text to check the input
        with patch("thinkrl.utils.tokenizer.tokenize_text") as mock_tokenize_text:
            messages = [{"role": "user", "content": "Hello"}]
            
            tokenize_conversation(
                messages,
                mock_tokenizer,
                add_generation_prompt=True
            )
            
            # Check that apply_chat_template was called correctly
            mock_tokenizer.apply_chat_template.assert_called_with(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Check that tokenize_text was called with the template's output
            mock_tokenize_text.assert_called_with(
                "Template output",
                mock_tokenizer
            )

    def test_prepare_input_for_generation(self, gpt2_tokenizer):
        """Test preparing inputs for model.generate()."""
        prompt = "Once upon a time"
        inputs = prepare_input_for_generation(
            prompt, 
            gpt2_tokenizer, 
            device="cpu"
        )
        
        assert "input_ids" in inputs
        assert "attention_mask" in inputs
        assert isinstance(inputs["input_ids"], torch.Tensor)
        assert inputs["input_ids"].device.type == "cpu"
        assert inputs["input_ids"].shape[0] == 1 # Batch size of 1

    def test_count_tokens(self, gpt2_tokenizer):
        """Test token counting."""
        text = "Hello, world!"
        count = count_tokens(text, gpt2_tokenizer)
        assert count == 4 # "Hello", ",", " world", "!"

        texts = ["Hello!", "How are you?"]
        counts = count_tokens(texts, gpt2_tokenizer)
        assert counts == [2, 3]

    def test_truncate_to_token_limit(self, gpt2_tokenizer):
        """Test truncating text based on token count."""
        long_text = "This is a very long text that will surely be truncated."
        # Tokens: "This", " is", " a", " very", " long", " text", " that", " will", " surely", " be", " tr", "uncated", "." (13 tokens)
        
        # Right truncation (default)
        truncated_right = truncate_to_token_limit(
            long_text, gpt2_tokenizer, max_tokens=7
        )
        assert truncated_right == "This is a very long text that"

        # Left truncation
        truncated_left = truncate_to_token_limit(
            long_text, gpt2_tokenizer, max_tokens=7, side="left"
        )
        assert truncated_left == " text that will surely be truncated."

    def test_get_tokenizer_info(self, gpt2_tokenizer):
        """Test getting tokenizer information."""
        info = get_tokenizer_info(gpt2_tokenizer)
        
        assert "vocab_size" in info
        assert "model_max_length" in info
        assert "padding_side" in info
        assert "special_tokens" in info
        
        assert info["vocab_size"] == 50257
        assert info["padding_side"] == "right"
        assert info["special_tokens"]["pad_token_id"] == 50256

    def test_save_and_load_tokenizer(self, temp_dir):
        """Test saving and loading a modified tokenizer."""
        # Load, modify, and save
        tokenizer_to_save = get_tokenizer("gpt2")
        add_special_tokens(tokenizer_to_save, {"additional_special_tokens": ["<|my_token|>"]})
        
        assert tokenizer_to_save.convert_tokens_to_ids("<|my_token|>") == 50257
        
        save_tokenizer(tokenizer_to_save, temp_dir)
        
        assert (temp_dir / "tokenizer.json").exists()
        assert (temp_dir / "special_tokens_map.json").exists()
        
        # Load back
        loaded_tokenizer = load_tokenizer(temp_dir)
        
        assert isinstance(loaded_tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast))
        assert len(loaded_tokenizer) == len(tokenizer_to_save)
        assert loaded_tokenizer.convert_tokens_to_ids("<|my_token|>") == 50257
        assert loaded_tokenizer.pad_token_id == 50256