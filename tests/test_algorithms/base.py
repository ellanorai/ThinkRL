"""
Base classes and utilities for algorithm testing.

This module provides foundational classes, mock implementations, and testing utilities
that are shared across all algorithm tests in ThinkRL. It includes base algorithm
interfaces, configuration classes, and helper functions for creating test data.

Classes:
    AlgorithmConfig: Base configuration for all algorithms
    AlgorithmOutput: Standardized output format
    BaseAlgorithm: Abstract base class for algorithms
    AlgorithmRegistry: Registry for algorithm discovery
    MockModel: Simple model for testing
    TestDataGenerator: Utility for generating test data

Usage:
    >>> from tests.test_algorithms.base import AlgorithmConfig, MockModel
    >>> config = AlgorithmConfig(learning_rate=1e-4)
    >>> model = MockModel(vocab_size=1000)
"""

import abc
import logging
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# Set up logger for testing
logger = logging.getLogger(__name__)


# ===== BASE ALGORITHM INTERFACES =====

@dataclass
class AlgorithmConfig:
    """
    Base configuration class for all RLHF algorithms.
    
    This class serves as the foundation for algorithm-specific configurations
    used in testing. It provides common parameters and validation methods.
    
    Args:
        learning_rate: Learning rate for optimization
        batch_size: Training batch size
        max_steps: Maximum training steps (None for unlimited)
        gradient_clip_norm: Maximum gradient norm for clipping
        seed: Random seed for reproducibility
        device: Compute device ('auto', 'cpu', 'cuda')
        mixed_precision: Enable mixed precision training
        log_level: Logging verbosity level
    """
    
    # Core training hyperparameters
    learning_rate: float = 3e-4
    batch_size: int = 32
    max_steps: Optional[int] = None
    gradient_clip_norm: float = 1.0
    
    # System configuration
    seed: int = 42
    device: str = "auto"
    mixed_precision: bool = False
    
    # Logging and debugging
    log_level: str = "INFO"
    log_interval: int = 10
    
    # Advanced settings
    gradient_accumulation_steps: int = 1
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        self._validate_config()
        self._resolve_device()
    
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        
        if self.gradient_clip_norm <= 0:
            raise ValueError(f"gradient_clip_norm must be positive, got {self.gradient_clip_norm}")
        
        if self.gradient_accumulation_steps <= 0:
            raise ValueError(f"gradient_accumulation_steps must be positive, got {self.gradient_accumulation_steps}")
        
        if not 0 <= self.warmup_ratio <= 1:
            raise ValueError(f"warmup_ratio must be between 0 and 1, got {self.warmup_ratio}")
    
    def _resolve_device(self) -> None:
        """Resolve device specification to actual device."""
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Validate device availability
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            warnings.warn(f"CUDA device '{self.device}' requested but not available. Using CPU.")
            self.device = "cpu"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AlgorithmConfig":
        """Create configuration from dictionary."""
        # Filter out unknown parameters
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
        
        if len(filtered_dict) != len(config_dict):
            unknown = set(config_dict.keys()) - valid_fields
            warnings.warn(f"Unknown configuration parameters ignored: {unknown}")
        
        return cls(**filtered_dict)
    
    def update(self, **kwargs) -> "AlgorithmConfig":
        """Create a new config with updated parameters."""
        config_dict = self.to_dict()
        config_dict.update(kwargs)
        return self.__class__.from_dict(config_dict)


@dataclass
class AlgorithmOutput:
    """
    Standardized output format for algorithm operations.
    
    This class provides a consistent interface for algorithm outputs,
    making it easier to handle results in tests and training loops.
    
    Args:
        loss: Primary loss value
        metrics: Dictionary of additional metrics
        logs: Dictionary of loggable values
        predictions: Model predictions (optional)
        hidden_states: Model hidden states (optional)
    """
    
    loss: Optional[torch.Tensor] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    logs: Dict[str, Any] = field(default_factory=dict)
    predictions: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None
    
    def __post_init__(self):
        """Ensure metrics and logs are mutable."""
        if not isinstance(self.metrics, dict):
            self.metrics = dict(self.metrics) if self.metrics else {}
        if not isinstance(self.logs, dict):
            self.logs = dict(self.logs) if self.logs else {}
    
    def update_metrics(self, **kwargs) -> None:
        """Update metrics with new values."""
        self.metrics.update(kwargs)
    
    def update_logs(self, **kwargs) -> None:
        """Update logs with new values."""
        self.logs.update(kwargs)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert output to dictionary for serialization."""
        result = {}
        if self.loss is not None:
            result["loss"] = self.loss.item() if hasattr(self.loss, "item") else self.loss
        result.update(self.metrics)
        result.update(self.logs)
        return result


class ModelProtocol(Protocol):
    """Protocol defining the interface for models used with algorithms."""
    
    def forward(self, *args, **kwargs) -> Any:
        """Forward pass through the model."""
        ...
    
    def parameters(self) -> Any:
        """Return model parameters."""
        ...
    
    def train(self, mode: bool = True) -> Any:
        """Set training mode."""
        ...
    
    def eval(self) -> Any:
        """Set evaluation mode."""
        ...


class BaseAlgorithm(abc.ABC):
    """
    Abstract base class for all RLHF algorithms.
    
    This class defines the core interface that all ThinkRL algorithms must implement.
    It provides common functionality while allowing algorithms to implement their
    specific training logic.
    
    Args:
        config: Algorithm configuration
    """
    
    def __init__(self, config: AlgorithmConfig):
        if not isinstance(config, AlgorithmConfig):
            raise TypeError(f"config must be an instance of AlgorithmConfig, got {type(config)}")
        
        self.config = config
        self._model: Optional[ModelProtocol] = None
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._is_setup = False
        
        # Set random seed
        self._set_seed(config.seed)
    
    @property
    def model(self) -> Optional[ModelProtocol]:
        """Get the current model."""
        return self._model
    
    @property
    def optimizer(self) -> Optional[torch.optim.Optimizer]:
        """Get the current optimizer."""
        return self._optimizer
    
    @property
    def is_setup(self) -> bool:
        """Check if algorithm is properly setup."""
        return self._is_setup
    
    def setup(self, model: ModelProtocol, optimizer: Optional[torch.optim.Optimizer] = None, **kwargs) -> None:
        """Setup the algorithm with model and optimizer."""
        self._model = model
        
        if optimizer is None:
            self._optimizer = self._create_default_optimizer()
        else:
            self._optimizer = optimizer
        
        # Move model to device
        if hasattr(self._model, "to"):
            self._model.to(self.config.device)
        
        self._is_setup = True
    
    @abc.abstractmethod
    def step(self, batch: Dict[str, Any], **kwargs) -> AlgorithmOutput:
        """Perform one algorithm step (training or inference)."""
        pass
    
    @abc.abstractmethod
    def compute_loss(self, batch: Dict[str, Any], model_outputs: Any, **kwargs) -> AlgorithmOutput:
        """Compute algorithm-specific loss."""
        pass
    
    def forward(self, batch: Dict[str, Any], **kwargs) -> Any:
        """Forward pass through the model."""
        if not self._is_setup:
            raise RuntimeError("Algorithm must be setup before calling forward()")
        return self._model(**batch, **kwargs)
    
    def _create_default_optimizer(self) -> torch.optim.Optimizer:
        """Create default optimizer for the model."""
        if self._model is None:
            raise RuntimeError("Model must be set before creating optimizer")
        
        return torch.optim.AdamW(
            self._model.parameters(),
            lr=self.config.learning_rate,
            eps=self.config.adam_epsilon,
            weight_decay=self.config.weight_decay,
        )
    
    def _set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def _validate_batch(self, batch: Dict[str, Any]) -> None:
        """Validate input batch format."""
        if not isinstance(batch, dict):
            raise TypeError(f"batch must be a dictionary, got {type(batch)}")
        
        # Move tensors to device
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.config.device)
    
    def save_checkpoint(self, path: str, **kwargs) -> None:
        """Save algorithm checkpoint."""
        if not self._is_setup:
            raise RuntimeError("Algorithm must be setup before saving checkpoint")
        
        checkpoint = {
            "algorithm_class": self.__class__.__name__,
            "config": self.config.to_dict(),
            "model_state_dict": self._model.state_dict() if hasattr(self._model, "state_dict") else None,
            "optimizer_state_dict": self._optimizer.state_dict() if self._optimizer else None,
            **kwargs,
        }
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str, **kwargs) -> Dict[str, Any]:
        """Load algorithm checkpoint."""
        checkpoint = torch.load(path, map_location=self.config.device)
        
        if "model_state_dict" in checkpoint and checkpoint["model_state_dict"] is not None:
            if self._model and hasattr(self._model, "load_state_dict"):
                self._model.load_state_dict(checkpoint["model_state_dict"])
        
        if "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"] is not None:
            if self._optimizer:
                self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        return checkpoint
    
    def get_info(self) -> Dict[str, Any]:
        """Get algorithm information."""
        info = {
            "algorithm": self.__class__.__name__,
            "config": self.config.to_dict(),
            "is_setup": self._is_setup,
            "device": self.config.device,
        }
        
        if self._model and hasattr(self._model, "parameters"):
            total_params = sum(p.numel() for p in self._model.parameters())
            trainable_params = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
            info.update({
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
            })
        
        return info


class AlgorithmRegistry:
    """
    Registry for algorithm discovery and instantiation.
    
    This class allows for dynamic registration and creation of algorithms,
    making it easy to extend the library with custom algorithms.
    """
    
    _algorithms: Dict[str, Callable[..., BaseAlgorithm]] = {}
    _configs: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str, algorithm_class: type, config_class: type, override: bool = False) -> None:
        """Register an algorithm."""
        if name in cls._algorithms and not override:
            raise ValueError(f"Algorithm '{name}' already registered. Use override=True to replace.")
        
        cls._algorithms[name] = algorithm_class
        cls._configs[name] = config_class
    
    @classmethod
    def get(cls, name: str) -> Tuple[type, type]:
        """Get algorithm and config classes by name."""
        if name not in cls._algorithms:
            available = ", ".join(cls._algorithms.keys())
            raise ValueError(f"Unknown algorithm '{name}'. Available: {available}")
        
        return cls._algorithms[name], cls._configs[name]
    
    @classmethod
    def list_algorithms(cls) -> List[str]:
        """List all registered algorithms."""
        return list(cls._algorithms.keys())
    
    @classmethod
    def create(cls, name: str, config: Optional[AlgorithmConfig] = None, **kwargs) -> BaseAlgorithm:
        """Create algorithm instance by name."""
        algorithm_class, config_class = cls.get(name)
        
        if config is None:
            config = config_class(**kwargs)
        elif isinstance(config, dict):
            config = config_class(**{**config, **kwargs})
        
        return algorithm_class(config)


# ===== MOCK IMPLEMENTATIONS FOR TESTING =====

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
    
    def __init__(self, vocab_size: int = 1000, hidden_size: int = 512, num_layers: int = 4, dropout: float = 0.1):
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
                batch_first=True,
            )
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, 
                labels: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass through the mock model."""
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
            loss = loss_fct(logits.view(-1, self.vocab_size), labels.view(-1))
        
        return {
            "logits": logits,
            "loss": loss,
            "hidden_states": hidden_states,
            "last_hidden_state": hidden_states,
        }
    
    def generate(self, input_ids: torch.Tensor, max_length: int = 50, temperature: float = 1.0, **kwargs) -> torch.Tensor:
        """Simple generation method for testing."""
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
    
    def forward(self, hidden_states: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass through value model."""
        values = self.value_head(hidden_states).squeeze(-1)
        return {"values": values}


# ===== TEST DATA UTILITIES =====

class TestDataGenerator:
    """Utility class for generating test data for algorithm testing."""
    
    @staticmethod
    def create_dummy_batch(batch_size: int = 4, seq_len: int = 32, vocab_size: int = 1000, 
                          device: str = "cpu", include_labels: bool = False, 
                          include_rewards: bool = False) -> Dict[str, torch.Tensor]:
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
            "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len), device=device),
            "attention_mask": torch.ones((batch_size, seq_len), device=device, dtype=torch.long),
        }
        
        if include_labels:
            batch["labels"] = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        
        if include_rewards:
            batch["rewards"] = torch.randn((batch_size, seq_len), device=device)
        
        return batch
    
    @staticmethod
    def create_mock_tokenizer(vocab_size: int = 1000):
        """Create a mock tokenizer for testing."""
        
        class MockTokenizer:
            def __init__(self, vocab_size: int):
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
        
        return MockTokenizer(vocab_size)


def assert_model_output(output: Dict[str, torch.Tensor], expected_batch_size: int, 
                       expected_seq_len: int, expected_vocab_size: int, 
                       should_have_loss: bool = False) -> None:
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
    expected_shape = (expected_batch_size, expected_seq_len, expected_vocab_size)
    assert logits.shape == expected_shape, f"Logits shape {logits.shape} != expected {expected_shape}"
    
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


# ===== TEST CONFIGURATION =====

# Common test devices
TEST_DEVICES = ["cpu"]
if torch.cuda.is_available():
    TEST_DEVICES.append("cuda")

# Test configuration
TEST_CONFIG = {
    "vocab_size": 1000,
    "hidden_size": 512,
    "num_layers": 4,
    "num_heads": 8,
    "max_seq_length": 512,
    "dropout": 0.1,
}


# ===== EXPORTS =====

__all__ = [
    # Base classes
    "AlgorithmConfig",
    "AlgorithmOutput", 
    "BaseAlgorithm",
    "AlgorithmRegistry",
    "ModelProtocol",
    
    # Mock implementations
    "MockModel",
    "MockValueModel",
    
    # Test utilities
    "TestDataGenerator",
    "assert_model_output",
    
    # Constants
    "TEST_DEVICES",
    "TEST_CONFIG",
]


# ===== CONVENIENCE FUNCTIONS =====

# Direct access to commonly used functions
create_dummy_batch = TestDataGenerator.create_dummy_batch
create_mock_tokenizer = TestDataGenerator.create_mock_tokenizer