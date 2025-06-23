"""
Base classes for ThinkRL algorithms.

This module provides the foundational abstract base classes and configuration
classes that all RLHF algorithms inherit from. It defines the common interface
and shared functionality across all algorithm implementations.

Classes:
    AlgorithmConfig: Base configuration class for algorithm parameters
    BaseAlgorithm: Abstract base class for all RLHF algorithms
    AlgorithmOutput: Standardized output format for algorithm operations
    AlgorithmRegistry: Registry for dynamic algorithm discovery
"""

import abc
import logging
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union

import torch
import torch.nn as nn

# Set up logger
logger = logging.getLogger(__name__)


@dataclass
class AlgorithmConfig:
    """
    Base configuration class for all RLHF algorithms.
    
    This class serves as the foundation for algorithm-specific configurations.
    Library users should use algorithm-specific config classes that inherit
    from this base class.
    
    Args:
        learning_rate: Learning rate for optimization
        batch_size: Training batch size
        max_steps: Maximum training steps (None for unlimited)
        gradient_clip_norm: Maximum gradient norm for clipping
        seed: Random seed for reproducibility
        device: Compute device ('auto', 'cpu', 'cuda', specific GPU)
        mixed_precision: Enable mixed precision training
        log_level: Logging verbosity level
        
    Example:
        >>> from thinkrl.algorithms import PPOConfig
        >>> config = PPOConfig(
        ...     learning_rate=3e-4,
        ...     batch_size=64,
        ...     max_steps=10000
        ... )
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
        self._setup_logging()
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
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        numeric_level = getattr(logging, self.log_level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {self.log_level}")
        logger.setLevel(numeric_level)
    
    def _resolve_device(self) -> None:
        """Resolve device specification to actual device."""
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Auto-detected device: {self.device}")
        
        # Validate device availability
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            warnings.warn(f"CUDA device '{self.device}' requested but CUDA not available. Falling back to CPU.")
            self.device = "cpu"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of the configuration
        """
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AlgorithmConfig":
        """Create configuration from dictionary.
        
        Args:
            config_dict: Dictionary containing configuration parameters
            
        Returns:
            Configuration instance
        """
        # Filter out unknown parameters
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
        
        if len(filtered_dict) != len(config_dict):
            unknown = set(config_dict.keys()) - valid_fields
            warnings.warn(f"Unknown configuration parameters ignored: {unknown}")
        
        return cls(**filtered_dict)
    
    def update(self, **kwargs) -> "AlgorithmConfig":
        """Create a new config with updated parameters.
        
        Args:
            **kwargs: Parameters to update
            
        Returns:
            New configuration instance with updated parameters
        """
        config_dict = self.to_dict()
        config_dict.update(kwargs)
        return self.__class__.from_dict(config_dict)


@dataclass
class AlgorithmOutput:
    """
    Standardized output format for algorithm operations.
    
    This class provides a consistent interface for algorithm outputs,
    making it easier for library users to handle results.
    
    Args:
        loss: Primary loss value
        metrics: Dictionary of additional metrics
        logs: Dictionary of loggable values
        predictions: Model predictions (optional)
        hidden_states: Model hidden states (optional)
        
    Example:
        >>> output = AlgorithmOutput(
        ...     loss=0.5,
        ...     metrics={"accuracy": 0.85},
        ...     logs={"step": 100}
        ... )
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
            result["loss"] = self.loss.item() if hasattr(self.loss, 'item') else self.loss
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
        
    Example:
        >>> from thinkrl.algorithms import PPO, PPOConfig
        >>> config = PPOConfig(learning_rate=1e-4)
        >>> algorithm = PPO(config)
        >>> algorithm.setup(model=my_model, optimizer=my_optimizer)
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
        
        logger.info(f"Initialized {self.__class__.__name__} with config: {config}")
    
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
    
    def setup(
        self,
        model: ModelProtocol,
        optimizer: Optional[torch.optim.Optimizer] = None,
        **kwargs
    ) -> None:
        """
        Setup the algorithm with model and optimizer.
        
        Args:
            model: The model to train
            optimizer: Optimizer (will create default if None)
            **kwargs: Additional setup parameters
        """
        self._model = model
        
        self._model = model
        
        if optimizer is None:
            self._optimizer = self._create_default_optimizer()
        else:
            self._optimizer = optimizer
        
        # Move model to device
        if hasattr(self._model, 'to'):
            self._model.to(self.config.device)
        
        self._is_setup = True
        logger.info(f"Algorithm setup complete. Model on device: {self.config.device}")
    
    @abc.abstractmethod
    def step(
        self,
        batch: Dict[str, Any],
        **kwargs
    ) -> AlgorithmOutput:
        """
        Perform one algorithm step (training or inference).
        
        Args:
            batch: Input batch data
            **kwargs: Additional step parameters
            
        Returns:
            AlgorithmOutput containing loss, metrics, and logs
        """
        pass
    
    @abc.abstractmethod
    def compute_loss(
        self,
        batch: Dict[str, Any],
        model_outputs: Any,
        **kwargs
    ) -> AlgorithmOutput:
        """
        Compute algorithm-specific loss.
        
        Args:
            batch: Input batch data
            model_outputs: Model forward pass outputs
            **kwargs: Additional parameters
            
        Returns:
            AlgorithmOutput containing loss and metrics
        """
        pass
    
    def forward(
        self,
        batch: Dict[str, Any],
        **kwargs
    ) -> Any:
        """
        Forward pass through the model.
        
        Args:
            batch: Input batch data
            **kwargs: Additional forward pass parameters
            
        Returns:
            Model outputs
        """
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
            weight_decay=self.config.weight_decay
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
        """
        Save algorithm checkpoint.
        
        Args:
            path: Path to save checkpoint
            **kwargs: Additional data to save
        """
        if not self._is_setup:
            raise RuntimeError("Algorithm must be setup before saving checkpoint")
        
        checkpoint = {
            "algorithm_class": self.__class__.__name__,
            "config": self.config.to_dict(),
            "model_state_dict": self._model.state_dict() if hasattr(self._model, 'state_dict') else None,
            "optimizer_state_dict": self._optimizer.state_dict() if self._optimizer else None,
            **kwargs
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str, **kwargs) -> Dict[str, Any]:
        """
        Load algorithm checkpoint.
        
        Args:
            path: Path to checkpoint file
            **kwargs: Additional load parameters
            
        Returns:
            Dictionary containing loaded checkpoint data
        """
        checkpoint = torch.load(path, map_location=self.config.device)
        
        if "model_state_dict" in checkpoint and checkpoint["model_state_dict"] is not None:
            if self._model and hasattr(self._model, 'load_state_dict'):
                self._model.load_state_dict(checkpoint["model_state_dict"])
        
        if "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"] is not None:
            if self._optimizer:
                self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        logger.info(f"Checkpoint loaded from {path}")
        return checkpoint
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get algorithm information.
        
        Returns:
            Dictionary containing algorithm information
        """
        info = {
            "algorithm": self.__class__.__name__,
            "config": self.config.to_dict(),
            "is_setup": self._is_setup,
            "device": self.config.device,
        }
        
        if self._model and hasattr(self._model, 'parameters'):
            total_params = sum(p.numel() for p in self._model.parameters())
            trainable_params = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
            info.update({
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
            })
        
        return info
    
    def __repr__(self) -> str:
        """String representation of the algorithm."""
        return (
            f"{self.__class__.__name__}("
            f"config={self.config.__class__.__name__}, "
            f"setup={self._is_setup}, "
            f"device={self.config.device})"
        )


class AlgorithmRegistry:
    """
    Registry for algorithm discovery and instantiation.
    
    This class allows for dynamic registration and creation of algorithms,
    making it easy for users to extend the library with custom algorithms.
    """
    
    _algorithms: Dict[str, Callable[..., BaseAlgorithm]] = {}
    _configs: Dict[str, type] = {}
    
    @classmethod
    def register(
        cls,
        name: str,
        algorithm_class: type,
        config_class: type,
        override: bool = False
    ) -> None:
        """
        Register an algorithm.
        
        Args:
            name: Algorithm name
            algorithm_class: Algorithm class
            config_class: Configuration class
            override: Whether to override existing registration
        """
        if name in cls._algorithms and not override:
            raise ValueError(f"Algorithm '{name}' already registered. Use override=True to replace.")
        
        cls._algorithms[name] = algorithm_class
        cls._configs[name] = config_class
        logger.info(f"Registered algorithm: {name}")
    
    @classmethod
    def get(cls, name: str) -> Tuple[type, type]:
        """
        Get algorithm and config classes by name.
        
        Args:
            name: Algorithm name
            
        Returns:
            Tuple of (algorithm_class, config_class)
        """
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
        """
        Create algorithm instance by name.
        
        Args:
            name: Algorithm name
            config: Algorithm configuration
            **kwargs: Additional configuration parameters
            
        Returns:
            Algorithm instance
        """
        algorithm_class, config_class = cls.get(name)
        
        if config is None:
            config = config_class(**kwargs)
        elif isinstance(config, dict):
            config = config_class(**{**config, **kwargs})
        
        return algorithm_class(config)