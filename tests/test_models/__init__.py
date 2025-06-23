"""
Test module for ThinkRL models.

This module contains tests for all model implementations in ThinkRL,
including base classes, specific model architectures, and integration tests.

Test Modules:
    test_base: Tests for base model classes and protocols
    test_gpt: Tests for GPT-style model implementations
    test_llama: Tests for LLaMA model implementations
    test_qwen: Tests for Qwen model implementations
    test_multimodal: Tests for multimodal model implementations

Test Utilities:
    MockModel: Mock model for testing algorithm interfaces
    create_dummy_batch: Helper to create test batches
    assert_model_output: Helper to validate model outputs

Example:
    >>> from tests.test_models import MockModel, create_dummy_batch
    >>> model = MockModel(vocab_size=1000, hidden_size=512)
    >>> batch = create_dummy_batch(batch_size=4, seq_len=32)
    >>> outputs = model(**batch)
"""

import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Tuple, Union
from dataclasses import dataclass

# Test configuration
TEST_CONFIG = {
    "vocab_size": 1000,
    "hidden_size": 512,
    "num_layers": 4,
    "num_heads": 8,
    "max_seq_length": 512,
    "dropout": 0.1,
}

# Common test devices
TEST_DEVICES = ["cpu"]
if torch.cuda.is_available():
    TEST_DEVICES.append("cuda")


@dataclass
class ModelTestConfig:
    """Configuration for model tests."""
    
    vocab_size: int = 1000
    hidden_size: int = 512
    num_layers: int = 4
    num_heads: int = 8
    max_seq_length: int = 512
    dropout: float = 0.1
    batch_size: int = 4
    seq_length: int = 32


class MockModel(nn.Module):
    """
    Mock model for testing algorithm interfaces.
    
    This class provides a minimal model implementation that satisfies
    the ModelProtocol interface for testing algorithms without requiring
    actual large language models.
    
    Args:
        vocab_size: Vocabulary size
        hidden_size: Hidden dimension size
        num_layers: Number of transformer layers
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        vocab_size: int = 1000,
        hidden_size: int = 512,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Simple embedding layer
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # Simple transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=hidden_size * 4,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        self.training = True
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the mock model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Target labels for loss computation [batch_size, seq_len]
            **kwargs: Additional arguments (ignored)
            
        Returns:
            Dictionary containing logits, loss, and hidden states
        """
        batch_size, seq_len = input_ids.shape
        
        # Embedding
        hidden_states = self.embedding(input_ids)
        
        # Apply transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        
        # Layer norm
        hidden_states = self.layer_norm(hidden_states)
        
        # Get logits
        logits = self.lm_head(hidden_states)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, self.vocab_size),
                labels.view(-1)
            )
        
        return {
            "logits": logits,
            "loss": loss,
            "hidden_states": hidden_states,
            "last_hidden_state": hidden_states,
        }
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 50,
        temperature: float = 1.0,
        **kwargs
    ) -> torch.Tensor:
        """
        Simple generation method for testing.
        
        Args:
            input_ids: Input token IDs
            max_length: Maximum generation length
            temperature: Sampling temperature
            **kwargs: Additional arguments
            
        Returns:
            Generated token IDs
        """
        self.eval()
        
        generated = input_ids.clone()
        
        for _ in range(max_length - input_ids.size(1)):
            with torch.no_grad():
                outputs = self.forward(generated)
                logits = outputs["logits"][:, -1, :] / temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                generated = torch.cat([generated, next_token], dim=1)
        
        return generated


class MockValueModel(nn.Module):
    """Mock value model for testing value-based algorithms."""
    
    def __init__(self, hidden_size: int = 512):
        super().__init__()
        self.hidden_size = hidden_size
        self.value_head = nn.Linear(hidden_size, 1)
        self.training = True
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through value model."""
        values = self.value_head(hidden_states).squeeze(-1)
        return {"values": values}


def create_dummy_batch(
    batch_size: int = 4,
    seq_len: int = 32,
    vocab_size: int = 1000,
    device: str = "cpu",
    include_labels: bool = False,
    include_rewards: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Create a dummy batch for testing.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        vocab_size: Vocabulary size
        device: Device to create tensors on
        include_labels: Whether to include labels
        include_rewards: Whether to include rewards
        
    Returns:
        Dictionary containing batch tensors
    """
    batch = {
        "input_ids": torch.randint(
            0, vocab_size, (batch_size, seq_len), device=device
        ),
        "attention_mask": torch.ones(
            (batch_size, seq_len), device=device, dtype=torch.long
        ),
    }
    
    if include_labels:
        batch["labels"] = torch.randint(
            0, vocab_size, (batch_size, seq_len), device=device
        )
    
    if include_rewards:
        batch["rewards"] = torch.randn(
            (batch_size, seq_len), device=device
        )
    
    return batch


def assert_model_output(
    output: Dict[str, torch.Tensor],
    expected_batch_size: int,
    expected_seq_len: int,
    expected_vocab_size: int,
    should_have_loss: bool = False,
) -> None:
    """
    Assert that model output has expected shape and contents.
    
    Args:
        output: Model output dictionary
        expected_batch_size: Expected batch size
        expected_seq_len: Expected sequence length
        expected_vocab_size: Expected vocabulary size
        should_have_loss: Whether output should contain loss
    """
    # Check required keys
    assert "logits" in output, "Output should contain 'logits'"
    assert "hidden_states" in output, "Output should contain 'hidden_states'"
    
    # Check logits shape
    logits = output["logits"]
    assert logits.shape == (expected_batch_size, expected_seq_len, expected_vocab_size), \
        f"Logits shape {logits.shape} doesn't match expected {(expected_batch_size, expected_seq_len, expected_vocab_size)}"
    
    # Check hidden states shape
    hidden_states = output["hidden_states"]
    assert len(hidden_states.shape) == 3, "Hidden states should be 3D tensor"
    assert hidden_states.shape[0] == expected_batch_size, "Batch size mismatch in hidden states"
    assert hidden_states.shape[1] == expected_seq_len, "Sequence length mismatch in hidden states"
    
    # Check loss if expected
    if should_have_loss:
        assert "loss" in output, "Output should contain 'loss'"
        assert output["loss"] is not None, "Loss should not be None"
        assert output["loss"].numel() == 1, "Loss should be scalar"
    
    # Check tensor types
    assert logits.dtype in [torch.float32, torch.float16], "Logits should be float tensor"
    assert hidden_states.dtype in [torch.float32, torch.float16], "Hidden states should be float tensor"


def create_mock_tokenizer():
    """Create a mock tokenizer for testing."""
    class MockTokenizer:
        def __init__(self, vocab_size: int = 1000):
            self.vocab_size = vocab_size
            self.pad_token_id = 0
            self.eos_token_id = 1
            self.bos_token_id = 2
        
        def encode(self, text: str, **kwargs) -> list:
            # Simple mock encoding
            return list(range(min(len(text), 10)))
        
        def decode(self, token_ids: list, **kwargs) -> str:
            # Simple mock decoding
            return "".join(["a"] * len(token_ids))
        
        def __call__(self, text, **kwargs):
            if isinstance(text, str):
                text = [text]
            
            max_length = kwargs.get("max_length", 512)
            
            input_ids = []
            attention_mask = []
            
            for t in text:
                ids = self.encode(t)[:max_length]
                mask = [1] * len(ids)
                
                # Pad to max_length if needed
                if len(ids) < max_length:
                    ids.extend([self.pad_token_id] * (max_length - len(ids)))
                    mask.extend([0] * (max_length - len(mask)))
                
                input_ids.append(ids)
                attention_mask.append(mask)
            
            return {
                "input_ids": torch.tensor(input_ids),
                "attention_mask": torch.tensor(attention_mask),
            }
    
    return MockTokenizer()


# Export commonly used test utilities
__all__ = [
    "MockModel",
    "MockValueModel", 
    "ModelTestConfig",
    "TEST_CONFIG",
    "TEST_DEVICES",
    "create_dummy_batch",
    "assert_model_output",
    "create_mock_tokenizer",
]