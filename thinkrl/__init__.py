"""
ThinkRL: Universal RLHF Training Library
=========================================

A powerful, open-source library for Reinforcement Learning from Human Feedback (RLHF)
with state-of-the-art algorithms, reasoning capabilities, and multimodal support.

Core Features:
-------------
- **Algorithms**: VAPO, DAPO, GRPO, PPO, REINFORCE
- **Reasoning**: Chain-of-Thought (CoT), Tree-of-Thought (ToT)
- **Multimodal**: Vision-language and audio-text models
- **Distributed**: DeepSpeed and FSDP support
- **PEFT**: LoRA and QLoRA integration

Quick Start:
-----------
```python
from thinkrl import RLHFTrainer, ModelConfig

config = ModelConfig(
    model_name_or_path="microsoft/DialoGPT-small",
    algorithm="vapo"
)

trainer = RLHFTrainer(config)
trainer.train()
```

Author: Archit Sood @ EllanorAI
License: Apache 2.0
Repository: https://github.com/Archit03/ThinkRL
"""

__version__ = "0.1.0"
__author__ = "Archit Sood"
__email__ = "archit@ellanorai.org"
__license__ = "Apache-2.0"
__copyright__ = "Copyright 2025 Archit Sood @ EllanorAI"

# Package metadata
__all__ = [
    # Version and metadata
    "__version__",
    "__author__",
    "__email__",
    
    # Core algorithms
    "BaseAlgorithm",
    "AlgorithmConfig",
    "AlgorithmOutput",
    "VAPO",
    "VAPOConfig",
    "DAPO",
    "DAPOConfig",
    "GRPO",
    "GRPOConfig",
    "PPO",
    "PPOConfig",
    "REINFORCE",
    "REINFORCEConfig",
    
    # Models
    "BaseModel",
    "ModelConfig",
    "GPTModel",
    "GPTConfig",
    "LlamaModel",
    "LlamaConfig",
    "QwenModel",
    "QwenConfig",
    "MultimodalModel",
    "MultimodalConfig",
    
    # Training
    "RLHFTrainer",
    "TrainerConfig",
    "DistributedTrainer",
    "CoTTrainer",
    "ToTTrainer",
    "MultimodalTrainer",
    
    # Reasoning
    "ReasoningConfig",
    "ChainOfThought",
    "CoTConfig",
    "TreeOfThought",
    "ToTConfig",
    
    # Data
    "RLHFDataset",
    "PreferenceDataset",
    "RLHFDataLoader",
    
    # Evaluation
    "RLHFEvaluator",
    "BenchmarkSuite",
    
    # Utilities
    "get_logger",
    "setup_logger",
    "compute_metrics",
    "save_checkpoint",
    "load_checkpoint",
    
    # Registry functions
    "get_algorithm",
    "list_algorithms",
    "create_algorithm",
    "register_algorithm",
    "get_model",
    "register_model",
]

# Import guard for optional dependencies
import warnings
import sys
from typing import TYPE_CHECKING

# Check for torch availability (required)
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    warnings.warn(
        "PyTorch is not installed. Please install it with: pip install torch>=2.0.0",
        ImportWarning
    )

# Check for optional dependencies
try:
    import transformers
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False

try:
    import peft
    _PEFT_AVAILABLE = True
except ImportError:
    _PEFT_AVAILABLE = False

try:
    import deepspeed
    _DEEPSPEED_AVAILABLE = True
except ImportError:
    _DEEPSPEED_AVAILABLE = False

try:
    import cupy
    _CUPY_AVAILABLE = True
except ImportError:
    _CUPY_AVAILABLE = False

# Lazy imports to avoid circular dependencies and improve startup time
# Core components are imported only when accessed
_LAZY_IMPORTS = {
    # Algorithms
    "BaseAlgorithm": "thinkrl.algorithms.base",
    "AlgorithmConfig": "thinkrl.algorithms.base",
    "AlgorithmOutput": "thinkrl.algorithms.base",
    "VAPO": "thinkrl.algorithms.vapo",
    "VAPOConfig": "thinkrl.algorithms.vapo",
    "DAPO": "thinkrl.algorithms.dapo",
    "DAPOConfig": "thinkrl.algorithms.dapo",
    "GRPO": "thinkrl.algorithms.grpo",
    "GRPOConfig": "thinkrl.algorithms.grpo",
    "PPO": "thinkrl.algorithms.ppo",
    "PPOConfig": "thinkrl.algorithms.ppo",
    "REINFORCE": "thinkrl.algorithms.reinforce",
    "REINFORCEConfig": "thinkrl.algorithms.reinforce",
    
    # Models
    "BaseModel": "thinkrl.models.base",
    "ModelConfig": "thinkrl.models.base",
    "GPTModel": "thinkrl.models.gpt",
    "GPTConfig": "thinkrl.models.gpt",
    "LlamaModel": "thinkrl.models.llama",
    "LlamaConfig": "thinkrl.models.llama",
    "QwenModel": "thinkrl.models.qwen",
    "QwenConfig": "thinkrl.models.qwen",
    "MultimodalModel": "thinkrl.models.multimodal",
    "MultimodalConfig": "thinkrl.models.multimodal",
    
    # Training
    "RLHFTrainer": "thinkrl.training.trainer",
    "TrainerConfig": "thinkrl.training.trainer",
    "DistributedTrainer": "thinkrl.training.distributed",
    "CoTTrainer": "thinkrl.training.cot_trainer",
    "ToTTrainer": "thinkrl.training.tot_trainer",
    "MultimodalTrainer": "thinkrl.training.multimodal_trainer",
    
    # Reasoning
    "ReasoningConfig": "thinkrl.reasoning.config",
    "ChainOfThought": "thinkrl.reasoning.cot.cot",
    "CoTConfig": "thinkrl.reasoning.cot.cot",
    "TreeOfThought": "thinkrl.reasoning.tot.tot",
    "ToTConfig": "thinkrl.reasoning.tot.tot",
    
    # Data
    "RLHFDataset": "thinkrl.data.datasets",
    "PreferenceDataset": "thinkrl.data.datasets",
    "RLHFDataLoader": "thinkrl.data.loaders",
    
    # Evaluation
    "RLHFEvaluator": "thinkrl.evaluation.evaluators",
    "BenchmarkSuite": "thinkrl.evaluation.benchmarks",
    
    # Utilities
    "get_logger": "thinkrl.utils.logging",
    "setup_logger": "thinkrl.utils.logging",
    "compute_metrics": "thinkrl.utils.metrics",
    "save_checkpoint": "thinkrl.utils.checkpoint",
    "load_checkpoint": "thinkrl.utils.checkpoint",
    
    # Registry
    "get_algorithm": "thinkrl.algorithms",
    "list_algorithms": "thinkrl.algorithms",
    "create_algorithm": "thinkrl.algorithms",
    "register_algorithm": "thinkrl.registry.algorithms",
    "get_model": "thinkrl.models",
    "register_model": "thinkrl.registry.models",
}


def __getattr__(name):
    """
    Lazy import mechanism for package components.
    
    This allows for faster package import times and deferred loading
    of heavy dependencies until they're actually needed.
    
    Args:
        name: Name of the attribute to import
        
    Returns:
        The requested module or attribute
        
    Raises:
        AttributeError: If the attribute is not found
    """
    if name in _LAZY_IMPORTS:
        module_path = _LAZY_IMPORTS[name]
        
        # Import the module
        try:
            from importlib import import_module
            module = import_module(module_path)
            
            # Get the specific attribute if it's a submodule import
            if "." in module_path:
                # For imports like "thinkrl.algorithms.base.BaseAlgorithm"
                attr = getattr(module, name)
            else:
                # For module-level imports
                attr = module
                
            # Cache the imported attribute in the module namespace
            globals()[name] = attr
            return attr
            
        except ImportError as e:
            # Provide helpful error messages for missing optional dependencies
            error_msg = f"Cannot import '{name}' from thinkrl. "
            
            if "transformers" in str(e) and not _TRANSFORMERS_AVAILABLE:
                error_msg += "This requires the 'transformers' package. Install with: pip install thinkrl[transformers]"
            elif "peft" in str(e) and not _PEFT_AVAILABLE:
                error_msg += "This requires the 'peft' package. Install with: pip install thinkrl[peft]"
            elif "deepspeed" in str(e) and not _DEEPSPEED_AVAILABLE:
                error_msg += "This requires the 'deepspeed' package. Install with: pip install thinkrl[deepspeed]"
            elif "cupy" in str(e) and not _CUPY_AVAILABLE:
                error_msg += "This requires the 'cupy' package for GPU acceleration. Install with: pip install thinkrl[cuda]"
            else:
                error_msg += f"Original error: {str(e)}"
                
            raise ImportError(error_msg) from e
            
    raise AttributeError(f"module 'thinkrl' has no attribute '{name}'")


def __dir__():
    """
    Define which attributes are available when dir() is called on the package.
    
    Returns:
        List of available attribute names
    """
    return sorted(__all__)


# Convenience functions for checking optional dependencies
def is_torch_available() -> bool:
    """Check if PyTorch is available."""
    return _TORCH_AVAILABLE


def is_transformers_available() -> bool:
    """Check if HuggingFace Transformers is available."""
    return _TRANSFORMERS_AVAILABLE


def is_peft_available() -> bool:
    """Check if PEFT is available."""
    return _PEFT_AVAILABLE


def is_deepspeed_available() -> bool:
    """Check if DeepSpeed is available."""
    return _DEEPSPEED_AVAILABLE


def is_cupy_available() -> bool:
    """Check if CuPy (GPU acceleration) is available."""
    return _CUPY_AVAILABLE


def get_version() -> str:
    """
    Get the ThinkRL version string.
    
    Returns:
        Version string (e.g., "0.1.0")
    """
    return __version__


def get_device_info() -> dict:
    """
    Get information about available compute devices.
    
    Returns:
        Dictionary containing device information
    """
    info = {
        "torch_available": _TORCH_AVAILABLE,
        "cuda_available": False,
        "cuda_version": None,
        "cuda_devices": 0,
        "cupy_available": _CUPY_AVAILABLE,
    }
    
    if _TORCH_AVAILABLE:
        info["cuda_available"] = torch.cuda.is_available()
        if info["cuda_available"]:
            info["cuda_version"] = torch.version.cuda
            info["cuda_devices"] = torch.cuda.device_count()
    
    return info


# Print initialization message (can be disabled by setting THINKRL_QUIET=1)
import os
if not os.environ.get("THINKRL_QUIET"):
    if not _TORCH_AVAILABLE:
        warnings.warn(
            "ThinkRL requires PyTorch but it's not installed. "
            "Please install it with: pip install torch>=2.0.0",
            ImportWarning
        )