"""
Configuration Dataclasses
=========================

Type-safe configuration system with YAML support.

Author: EllanorAI
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import logging
from pathlib import Path
from typing import Any

import torch


# Optional YAML support
try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    yaml = None  # type: ignore
    YAML_AVAILABLE = False


logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model configuration."""

    # Model source
    name_or_path: str = "meta-llama/Llama-3.1-8B"

    # Precision
    dtype: str = "bfloat16"  # "float32", "float16", "bfloat16"

    # Optimization
    use_flash_attention: bool = True
    gradient_checkpointing: bool = False

    # Quantization
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"

    # Loading
    trust_remote_code: bool = True
    device_map: str = "auto"

    # Reference model
    ref_model_name_or_path: str | None = None
    share_ref_weights: bool = True

    def get_dtype(self) -> torch.dtype:
        """Get torch dtype from string."""
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map.get(self.dtype, torch.bfloat16)


@dataclass
class PeftConfig:
    """PEFT/LoRA configuration."""

    enabled: bool = True
    type: str = "lora"  # "lora", "qlora"

    # LoRA parameters
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: list[str] | None = None
    modules_to_save: list[str] | None = None
    bias: str = "none"

    # Auto-detect architecture
    auto_target_modules: bool = True


@dataclass
class DistributedConfig:
    """Distributed training configuration."""

    # Strategy
    strategy: str = "zero2"  # zero1, zero2, zero3, zero2_offload, zero3_offload
    backend: str = "nccl"

    # Batch sizes
    micro_batch_size: int = 4
    gradient_accumulation_steps: int = 4

    # Offloading
    offload_optimizer: bool = False
    offload_param: bool = False

    # Multi-node
    master_addr: str = "localhost"
    master_port: int = 29500

    # Precision
    bf16: bool = True
    fp16: bool = False


@dataclass
class DataConfig:
    """Data configuration."""

    # Dataset
    dataset: str | None = None
    dataset_split: str = "train"
    eval_dataset: str | None = None
    eval_split: str = "test"

    # Processing
    max_length: int = 2048
    max_prompt_length: int = 1024
    max_response_length: int = 1024

    # Batching
    batch_size: int = 4
    eval_batch_size: int = 8
    num_workers: int = 4

    # Columns
    prompt_column: str = "prompt"
    response_column: str = "response"
    chosen_column: str = "chosen"
    rejected_column: str = "rejected"

    # Chat template
    chat_template: str | None = None
    apply_chat_template: bool = True

    # Streaming
    streaming: bool = False
    buffer_size: int = 10000


@dataclass
class LoggingConfig:
    """Logging configuration."""

    # Backends
    backends: list[str] = field(default_factory=lambda: ["console"])

    # W&B
    wandb_project: str = "thinkrl"
    wandb_entity: str | None = None
    wandb_name: str | None = None
    wandb_tags: list[str] = field(default_factory=list)

    # TensorBoard
    tensorboard_dir: str = "./logs/tensorboard"

    # Logging frequency
    log_every_n_steps: int = 10
    eval_every_n_steps: int = 100
    save_every_n_steps: int = 500

    # Output
    output_dir: str = "./outputs"
    save_total_limit: int = 3

    # Verbosity
    log_level: str = "INFO"


@dataclass
class ThinkRLConfig:
    """Root configuration object."""

    model: ModelConfig = field(default_factory=ModelConfig)
    algorithm: Any = None  # Dynamically loaded config (e.g., PPOConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    peft: PeftConfig | None = None

    # Training
    max_steps: int = 1000
    eval_steps: int = 100
    save_steps: int = 500
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path) -> ThinkRLConfig:
        """Load configuration from YAML file."""
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML is required. Install with: pip install pyyaml")

        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data)

    @classmethod
    def from_json(cls, path: str | Path) -> ThinkRLConfig:
        """Load configuration from JSON file."""
        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ThinkRLConfig:
        """Create config from dictionary."""
        # Parse nested configs
        model = ModelConfig(**data.get("model", {}))

        # Dynamic Algorithm Config Loading
        algo_data = data.get("algorithm", {})
        # If algorithm is already an object, assume it's valid
        if not isinstance(algo_data, dict):
            algorithm = algo_data
        else:
            # Determine algorithm name
            algo_name = algo_data.get("name", "ppo")

            try:
                # Lazy import to avoid circular dependency
                from thinkrl.algorithms import get_config

                ConfigClass = get_config(algo_name)
                # Filter args that are valid for this config
                # (Optional: specialized configs might accept **kwargs, but we can be safe)
                # For now, we assume ConfigClass accepts relevant kwargs or we pass all
                # if it uses **kwargs (most do via dataclass or explicit init)
                # However, clean dataclasses don't take extra args by default unless __init__ is custom.
                # All ours are dataclasses.
                # Best practice: Filter keys against __annotations__
                valid_keys = ConfigClass.__annotations__.keys()
                # Also include fields if defined via dataclass field() (handled by annotations usually)

                # Check if the class allows extra args (unlikely for pure dataclass)
                # We will just pass what matches
                filtered_args = {k: v for k, v in algo_data.items() if k in valid_keys}

                # Special handling: some args might be passed that aren't in fields but handled by custom init?
                # Our configs are pure dataclasses generally.
                # Let's instantiate
                algorithm = ConfigClass(**filtered_args)

            except Exception as e:
                logger.warning(f"Failed to load algorithm config for '{algo_name}': {e}. Falling back to dict.")
                algorithm = algo_data
        distributed = DistributedConfig(**data.get("distributed", {}))
        data_config = DataConfig(**data.get("data", {}))
        logging_config = LoggingConfig(**data.get("logging", {}))

        peft = None
        if "peft" in data and data["peft"]:
            peft = PeftConfig(**data["peft"])

        # Get top-level fields
        top_level = {
            k: v for k, v in data.items() if k not in ["model", "algorithm", "distributed", "data", "logging", "peft"]
        }

        return cls(
            model=model,
            algorithm=algorithm,
            distributed=distributed,
            data=data_config,
            logging=logging_config,
            peft=peft,
            **top_level,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        # Handle algorithm being either a dataclass or a plain dict
        if self.algorithm is None:
            algo_dict = None
        elif isinstance(self.algorithm, dict):
            algo_dict = self.algorithm
        else:
            # It's a dataclass, use asdict
            algo_dict = asdict(self.algorithm)

        result = {
            "model": asdict(self.model),
            "algorithm": algo_dict,
            "distributed": asdict(self.distributed),
            "data": asdict(self.data),
            "logging": asdict(self.logging),
            "peft": asdict(self.peft) if self.peft else None,
            "max_steps": self.max_steps,
            "eval_steps": self.eval_steps,
            "save_steps": self.save_steps,
            "seed": self.seed,
        }
        return result

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML is required. Install with: pip install pyyaml")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def to_json(self, path: str | Path) -> None:
        """Save configuration to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def validate(self) -> list[str]:
        """
        Validate configuration consistency.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check algorithm compatibility
        if self.algorithm is not None:
            # Handle both object and dict (fallback) configurations
            is_dict = isinstance(self.algorithm, dict)
            algo_name = self.algorithm.get("name", "ppo") if is_dict else getattr(self.algorithm, "name", "ppo")

            if algo_name in ["ppo", "vapo"] and self.peft:
                # Check value_coeff
                value_coeff = (
                    self.algorithm.get("value_coeff", 0.5) if is_dict else getattr(self.algorithm, "value_coeff", 0.5)
                )

                if self.peft.enabled and value_coeff > 0:
                    # Value model with LoRA needs special handling
                    pass

        # Check distributed settings
        if self.distributed.strategy.startswith("zero3"):
            if not self.distributed.offload_optimizer and not self.distributed.offload_param:
                logger.warning("ZeRO-3 without offloading may have high memory usage")

        # Check data settings
        if self.data.max_length < self.data.max_prompt_length + self.data.max_response_length:
            errors.append(
                f"max_length ({self.data.max_length}) should be >= "
                f"max_prompt_length + max_response_length "
                f"({self.data.max_prompt_length + self.data.max_response_length})"
            )

        return errors


def load_config(path: str | Path) -> ThinkRLConfig:
    """
    Load configuration from file.

    Supports YAML and JSON formats.

    Args:
        path: Path to config file

    Returns:
        ThinkRLConfig instance
    """
    path = Path(path)

    if path.suffix in [".yaml", ".yml"]:
        return ThinkRLConfig.from_yaml(path)
    elif path.suffix == ".json":
        return ThinkRLConfig.from_json(path)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")


def save_config(config: ThinkRLConfig, path: str | Path) -> None:
    """
    Save configuration to file.

    Args:
        config: Configuration to save
        path: Output path (format determined by extension)
    """
    path = Path(path)

    if path.suffix in [".yaml", ".yml"]:
        config.to_yaml(path)
    elif path.suffix == ".json":
        config.to_json(path)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")

    logger.info(f"Saved config to: {path}")


def merge_configs(base: ThinkRLConfig, overrides: dict[str, Any]) -> ThinkRLConfig:
    """
    Merge override values into base config.

    Args:
        base: Base configuration
        overrides: Dictionary of overrides (supports dot notation)

    Returns:
        New config with overrides applied

    Example:
        >>> config = merge_configs(base, {"algorithm.learning_rate": 1e-4})
    """
    base_dict = base.to_dict()

    for key, value in overrides.items():
        # Handle dot notation (e.g., "algorithm.learning_rate")
        parts = key.split(".")
        target = base_dict

        for part in parts[:-1]:
            if part not in target or target[part] is None:
                target[part] = {}
            target = target[part]

        target[parts[-1]] = value

    return ThinkRLConfig.from_dict(base_dict)


__all__ = [
    "ThinkRLConfig",
    "ModelConfig",
    "DistributedConfig",
    "DataConfig",
    "LoggingConfig",
    "PeftConfig",
    "load_config",
    "save_config",
    "merge_configs",
]
