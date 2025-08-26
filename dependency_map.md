# Complete ThinkRL Dependency Map

## 📦 External Dependencies

### Core Required Dependencies
```
torch>=2.0.0,<3.0.0
├── Used by: ALL algorithm implementations
├── Used by: ALL model implementations
├── Used by: ALL training modules
├── Used by: utils/checkpoint.py
└── Used by: tests/*

numpy>=1.24.0,<2.0.0
├── Used by: algorithms/* (numerical computations)
├── Used by: data/processors.py
├── Used by: evaluation/metrics.py
└── Used by: utils/data.py
└── Used by: tests/test_sample.py

pyyaml>=6.0,<7.0
├── Used by: utils/checkpoint.py
├── Used by: scripts/train.py
├── Used by: scripts/evaluate.py
└── Used by: configs/*.yaml parsing

tqdm>=4.65.0
├── Used by: training/trainer.py
├── Used by: data/loaders.py
├── Used by: evaluation/evaluators.py
└── Used by: scripts/*

accelerate>=0.21.0,<1.0.0
├── Used by: training/distributed.py
├── Used by: training/trainer.py
├── Used by: utils/checkpoint.py
└── Device management and distributed training
```

### Optional GPU Dependencies
```
cupy-cuda12x>=12.0.0,<13.0.0 (OR cupy-cuda11x>=11.0.0,<12.0.0)
├── Used by: algorithms/dapo.py (GPU advantage computation)
├── Used by: algorithms/vapo.py (GPU value estimation)
├── Used by: utils/data.py (GPU data processing)
└── Fallback to NumPy if not available
```

### ML Framework Dependencies
```
transformers>=4.30.0,<5.0.0
├── Used by: models/gpt.py
├── Used by: models/llama.py
├── Used by: models/qwen.py
├── Used by: training/trainer.py
├── Used by: utils/tokenizers.py
└── Dependencies: tokenizers, safetensors, datasets

peft>=0.4.0,<1.0.0
├── Used by: models/base.py (PEFT integration)
├── Used by: training/trainer.py (LoRA/QLoRA)
└── Dependencies: bitsandbytes>=0.41.0

deepspeed>=0.9.0,<1.0.0
├── Used by: training/distributed.py
├── Used by: scripts/train.py (--deepspeed flag)
└── ZeRO optimization stages

datasets>=2.14.0,<3.0.0
├── Used by: data/datasets.py
├── Used by: data/loaders.py
└── HuggingFace dataset integration

safetensors>=0.3.0
├── Used by: utils/checkpoint.py
└── Safe model serialization

tokenizers>=0.15.0,<1.0.0
├── Used by: utils/tokenizers.py
├── Used by: models/*
└── Fast tokenization
```

### Multimodal Dependencies
```
Vision:
  pillow>=9.0.0,<11.0.0
  ├── Used by: models/multimodal.py
  └── Used by: data/processors.py

  torchvision>=0.15.0,<1.0.0
  ├── Used by: models/multimodal.py
  └── Used by: training/multimodal_trainer.py

  opencv-python>=4.5.0
  ├── Used by: data/processors.py
  └── Used by: evaluation/benchmarks.py

Audio:
  torchaudio>=2.0.0,<3.0.0
  ├── Used by: models/multimodal.py
  └── Used by: data/processors.py

  librosa>=0.10.0
  ├── Used by: data/processors.py
  └── Audio feature extraction

  soundfile>=0.12.0
  └── Used by: data/loaders.py
```

### Reasoning Dependencies
```
networkx>=3.1,<4.0
├── Used by: reasoning/tot/tree.py
└── Tree structure management

graphviz>=0.20.0
├── Used by: reasoning/tot/tree.py
└── Tree visualization

matplotlib>=3.5.0
├── Used by: evaluation/benchmarks.py
├── Used by: utils/metrics.py
└── Plotting and visualization

sympy>=1.12.0
├── Used by: reasoning/cot/cot.py
└── Symbolic math for reasoning

scipy>=1.10.0
├── Used by: evaluation/metrics.py
└── Statistical computations
```

### Experiment Tracking Dependencies
```
wandb>=0.15.0,<1.0.0
├── Used by: training/trainer.py
├── Used by: utils/logging.py
└── Weights & Biases integration

tensorboard>=2.13.0
├── Used by: training/trainer.py
├── Used by: utils/logging.py
└── TensorBoard logging

mlflow>=2.5.0,<3.0.0
├── Used by: training/trainer.py
└── MLflow experiment tracking
```

### Development Dependencies
```
Testing:
  pytest>=7.0.0,<8.0.0
  pytest-cov>=4.1.0,<5.0.0
  pytest-xdist>=3.3.0
  pytest-mock>=3.11.0

Code Quality:
  black>=23.9.0,<24.0.0
  isort>=5.12.0,<6.0.0
  flake8>=6.0.0,<7.0.0
  mypy>=1.5.0,<2.0.0
  pre-commit>=3.0.0,<4.0.0
```

## 🏗️ Internal Dependencies

### Level 0: Foundation (No Internal Dependencies)

```python
# Base Classes and Protocols
thinkrl/algorithms/base.py
├── Classes: AlgorithmConfig, BaseAlgorithm, AlgorithmOutput, AlgorithmRegistry
├── External: torch, logging, warnings, dataclasses, typing
└── Internal: None

thinkrl/models/base.py
├── Classes: BaseModel, ModelConfig, ModelProtocol
├── External: torch, torch.nn, typing
└── Internal: None

thinkrl/reasoning/config.py
├── Classes: ReasoningConfig
├── External: dataclasses, typing
└── Internal: None

# Utilities (Independent)
thinkrl/utils/logging.py
├── Functions: setup_logger, get_logger
├── External: logging, sys
└── Internal: None

thinkrl/utils/metrics.py
├── Functions: compute_metrics, aggregate_metrics
├── External: numpy, torch, typing
└── Internal: None

thinkrl/utils/data.py
├── Functions: create_dataloader, preprocess_data
├── External: torch, numpy
└── Internal: None

thinkrl/utils/tokenizers.py
├── Functions: get_tokenizer, tokenize_batch
├── External: transformers (optional)
└── Internal: None

thinkrl/utils/checkpoint.py
├── Functions: save_checkpoint, load_checkpoint
├── External: torch, pathlib, safetensors (optional)
└── Internal: None
```

### Level 1: Core Implementations

```python
# Algorithm Implementations
thinkrl/algorithms/dapo.py
├── Classes: DAPO, DAPOConfig, DAPOAdvantageEstimator, DAPOLoss, DAPOSampler
├── External: torch, torch.nn.functional, logging, math
├── Internal: from .base import AlgorithmConfig, AlgorithmOutput, BaseAlgorithm
└── Exports: DAPO, DAPOConfig, create_dapo_algorithm, create_dapo_config

thinkrl/algorithms/grpo.py
├── Classes: GRPO, GRPOConfig, GRPORewardNormalizer, GRPOLoss, GRPOBatcher
├── External: torch, torch.nn.functional, collections.defaultdict
├── Internal: from .base import AlgorithmConfig, AlgorithmOutput, BaseAlgorithm
└── Exports: GRPO, GRPOConfig

thinkrl/algorithms/ppo.py
├── Classes: PPO, PPOConfig, PPOAdvantageEstimator, PPOValueFunction, PPOLoss
├── External: torch, torch.nn, torch.nn.functional, random
├── Internal: from .base import AlgorithmConfig, AlgorithmOutput, BaseAlgorithm
└── Exports: PPO, PPOConfig, create_ppo_algorithm, create_ppo_config

thinkrl/algorithms/reinforce.py
├── Classes: REINFORCE, REINFORCEConfig, REINFORCEReturns, REINFORCEBaseline, REINFORCELoss
├── External: torch, torch.nn, torch.nn.functional
├── Internal: from .base import AlgorithmConfig, AlgorithmOutput, BaseAlgorithm
└── Exports: REINFORCE, REINFORCEConfig, create_reinforce_algorithm

thinkrl/algorithms/vapo.py
├── Classes: VAPO, VAPOConfig (placeholder)
├── External: torch
├── Internal: from .base import AlgorithmConfig, AlgorithmOutput, BaseAlgorithm
└── Exports: VAPO, VAPOConfig

# Model Implementations
thinkrl/models/gpt.py
├── Classes: GPTModel, GPTConfig
├── External: torch, torch.nn, transformers (optional)
├── Internal: from .base import BaseModel, ModelConfig
└── Exports: GPTModel, GPTConfig

thinkrl/models/llama.py
├── Classes: LlamaModel, LlamaConfig
├── External: torch, torch.nn, transformers (optional)
├── Internal: from .base import BaseModel, ModelConfig
└── Exports: LlamaModel, LlamaConfig

thinkrl/models/qwen.py
├── Classes: QwenModel, QwenConfig
├── External: torch, torch.nn, transformers (optional)
├── Internal: from .base import BaseModel, ModelConfig
└── Exports: QwenModel, QwenConfig

thinkrl/models/multimodal.py
├── Classes: MultimodalModel, MultimodalConfig
├── External: torch, torch.nn, torchvision, torchaudio (optional)
├── Internal: from .base import BaseModel, ModelConfig
└── Exports: MultimodalModel, MultimodalConfig

# Data Layer
thinkrl/data/datasets.py
├── Classes: RLHFDataset, PreferenceDataset
├── External: torch.utils.data, datasets (optional)
├── Internal: from ..utils.data import preprocess_data
└── Exports: RLHFDataset, PreferenceDataset

thinkrl/data/processors.py
├── Functions: process_text, process_image, process_audio
├── External: numpy, pillow, librosa (optional)
├── Internal: from ..utils.data import *
└── Exports: process_text, process_image, process_audio

# Evaluation Layer
thinkrl/evaluation/metrics.py
├── Functions: compute_reward, compute_kl_divergence, compute_accuracy
├── External: torch, numpy, scipy (optional)
├── Internal: from ..utils.metrics import *
└── Exports: compute_reward, compute_kl_divergence

# Reasoning Components
thinkrl/reasoning/cot/prompts.py
├── Constants: COT_PROMPTS, COT_TEMPLATES
├── External: None
├── Internal: None
└── Exports: COT_PROMPTS, COT_TEMPLATES

thinkrl/reasoning/cot/cot.py
├── Classes: ChainOfThought, CoTConfig
├── External: torch, sympy (optional)
├── Internal: 
│   from ..config import ReasoningConfig
│   from .prompts import COT_PROMPTS
└── Exports: ChainOfThought, CoTConfig

thinkrl/reasoning/tot/tree.py
├── Classes: ThoughtTree, TreeNode
├── External: networkx, graphviz (optional)
├── Internal: None
└── Exports: ThoughtTree, TreeNode

thinkrl/reasoning/tot/evaluator.py
├── Classes: ThoughtEvaluator
├── External: torch
├── Internal: from .tree import TreeNode
└── Exports: ThoughtEvaluator

thinkrl/reasoning/tot/tot.py
├── Classes: TreeOfThought, ToTConfig
├── External: torch
├── Internal:
│   from ..config import ReasoningConfig
│   from .tree import ThoughtTree
│   from .evaluator import ThoughtEvaluator
└── Exports: TreeOfThought, ToTConfig
```

### Level 2: Aggregation and Orchestration

```python
# Algorithm Module Init
thinkrl/algorithms/__init__.py
├── External: typing
├── Internal:
│   from .base import AlgorithmConfig, BaseAlgorithm
│   from .dapo import DAPO, DAPOConfig
│   from .grpo import GRPO, GRPOConfig
│   from .ppo import PPO, PPOConfig
│   from .reinforce import REINFORCE, REINFORCEConfig
│   from .vapo import VAPO, VAPOConfig
├── Functions: get_algorithm(), list_algorithms(), create_algorithm()
└── Exports: All algorithm classes and configs

# Data Module
thinkrl/data/loaders.py
├── Classes: RLHFDataLoader
├── External: torch.utils.data
├── Internal:
│   from .datasets import RLHFDataset, PreferenceDataset
│   from .processors import process_text
│   from ..utils.data import create_dataloader
└── Exports: RLHFDataLoader

# Evaluation Module
thinkrl/evaluation/evaluators.py
├── Classes: RLHFEvaluator
├── External: torch, tqdm
├── Internal:
│   from .metrics import compute_reward, compute_accuracy
│   from ..utils.metrics import aggregate_metrics
└── Exports: RLHFEvaluator

# Registry System
thinkrl/registry/algorithms.py
├── Functions: register_algorithm, get_registered_algorithms
├── External: typing
├── Internal:
│   from ..algorithms.base import BaseAlgorithm
│   from ..algorithms import *
└── Manages dynamic algorithm registration

thinkrl/registry/models.py
├── Functions: register_model, get_registered_models
├── External: typing
├── Internal:
│   from ..models.base import BaseModel
│   from ..models import *
└── Manages dynamic model registration
```

### Level 3: Training and Integration

```python
# Core Trainer
thinkrl/training/trainer.py
├── Classes: RLHFTrainer, TrainerConfig
├── External: torch, tqdm, wandb/tensorboard (optional)
├── Internal:
│   from ..algorithms import get_algorithm
│   from ..models.base import ModelProtocol
│   from ..data.loaders import RLHFDataLoader
│   from ..evaluation.evaluators import RLHFEvaluator
│   from ..utils.logging import get_logger
│   from ..utils.checkpoint import save_checkpoint, load_checkpoint
│   from ..utils.metrics import aggregate_metrics
└── Exports: RLHFTrainer, TrainerConfig

# Distributed Training
thinkrl/training/distributed.py
├── Classes: DistributedTrainer
├── External: torch.distributed, accelerate, deepspeed (optional)
├── Internal:
│   from .trainer import RLHFTrainer, TrainerConfig
│   from ..utils.logging import get_logger
└── Exports: DistributedTrainer

# Specialized Trainers
thinkrl/training/cot_trainer.py
├── Classes: CoTTrainer
├── External: torch
├── Internal:
│   from .trainer import RLHFTrainer
│   from ..reasoning.cot import ChainOfThought, CoTConfig
│   from ..algorithms import get_algorithm
└── Exports: CoTTrainer

thinkrl/training/tot_trainer.py
├── Classes: ToTTrainer
├── External: torch
├── Internal:
│   from .trainer import RLHFTrainer
│   from ..reasoning.tot import TreeOfThought, ToTConfig
│   from ..algorithms import get_algorithm
└── Exports: ToTTrainer

thinkrl/training/multimodal_trainer.py
├── Classes: MultimodalTrainer
├── External: torch, torchvision
├── Internal:
│   from .trainer import RLHFTrainer
│   from ..models.multimodal import MultimodalModel
│   from ..data.processors import process_image, process_audio
└── Exports: MultimodalTrainer

# Evaluation Integration
thinkrl/evaluation/benchmarks.py
├── Classes: BenchmarkSuite, AIFEBenchmark
├── External: torch, matplotlib (optional)
├── Internal:
│   from .evaluators import RLHFEvaluator
│   from .metrics import compute_reward, compute_accuracy
│   from ..utils.metrics import aggregate_metrics
└── Exports: BenchmarkSuite, AIFEBenchmark
```

### Level 4: Entry Points and Scripts

```python
# Main Training Script
thinkrl/scripts/train.py
├── External: argparse, yaml, torch
├── Internal:
│   from ..training.trainer import RLHFTrainer, TrainerConfig
│   from ..training.distributed import DistributedTrainer
│   from ..algorithms import create_algorithm, get_algorithm_config
│   from ..models import get_model  # Would be in models/__init__.py
│   from ..data.loaders import RLHFDataLoader
│   from ..utils.logging import setup_logger
│   from ..utils.checkpoint import load_checkpoint
├── Entry point: main()
└── CLI: thinkrl-train

# Evaluation Script
thinkrl/scripts/evaluate.py
├── External: argparse, torch
├── Internal:
│   from ..evaluation.evaluators import RLHFEvaluator
│   from ..evaluation.benchmarks import BenchmarkSuite
│   from ..models import get_model
│   from ..utils.checkpoint import load_checkpoint
│   from ..utils.logging import setup_logger
├── Entry point: main()
└── CLI: thinkrl-eval

# Chain of Thought Script
thinkrl/scripts/chain_of_thought.py
├── External: argparse, torch
├── Internal:
│   from ..training.cot_trainer import CoTTrainer
│   from ..reasoning.cot import ChainOfThought, CoTConfig
│   from ..models import get_model
│   from ..utils.logging import setup_logger
├── Entry point: main()
└── CLI: thinkrl-cot

# Tree of Thought Script
thinkrl/scripts/tree_of_thought.py
├── External: argparse, torch
├── Internal:
│   from ..training.tot_trainer import ToTTrainer
│   from ..reasoning.tot import TreeOfThought, ToTConfig
│   from ..models import get_model
│   from ..utils.logging import setup_logger
├── Entry point: main()
└── CLI: thinkrl-tot

# Multimodal Training Script
thinkrl/scripts/multimodal_train.py
├── External: argparse, torch
├── Internal:
│   from ..training.multimodal_trainer import MultimodalTrainer
│   from ..models.multimodal import MultimodalModel
│   from ..data.processors import process_image, process_audio
│   from ..utils.logging import setup_logger
├── Entry point: main()
└── CLI: thinkrl-multimodal
```

### Test Dependencies

```python
# Test Infrastructure
tests/__init__.py
├── External: pytest
└── Internal: None

tests/test_sample.py
├── External: pytest, torch, numpy
├── Internal: from thinkrl import __version__
└── Tests basic imports and operations

# Base Test Utilities
tests/test_algorithms/base.py
├── Classes: AlgorithmConfig, BaseAlgorithm, MockModel
├── External: pytest, torch, mock
├── Internal: None (defines mocks)
└── Exports: Test utilities for algorithm testing

tests/test_models/__init__.py
├── Classes: MockModel, MockValueModel
├── External: torch, torch.nn
├── Internal: None (defines mocks)
└── Exports: Test utilities for model testing

# Algorithm Tests
tests/test_algorithms/test_dapo.py
├── External: pytest, torch, mock
├── Internal:
│   from tests.test_models import MockModel, create_dummy_batch
│   from thinkrl.algorithms.base import AlgorithmOutput
│   from thinkrl.algorithms.dapo import *
└── Tests DAPO implementation

tests/test_algorithms/test_grpo.py
├── External: pytest, torch
├── Internal:
│   from tests.test_models import MockModel, create_dummy_batch
│   from thinkrl.algorithms.base import AlgorithmOutput
│   from thinkrl.algorithms.grpo import *
└── Tests GRPO implementation

tests/test_algorithms/test_ppo.py
├── External: pytest, torch
├── Internal:
│   from tests.test_models import MockModel, create_dummy_batch
│   from thinkrl.algorithms.base import AlgorithmOutput
│   from thinkrl.algorithms.ppo import *
└── Tests PPO implementation

tests/test_algorithms/test_reinforce.py
├── External: pytest, torch
├── Internal:
│   from tests.test_models import MockModel, create_dummy_batch
│   from thinkrl.algorithms.reinforce import *
└── Tests REINFORCE implementation

# Model Tests
tests/test_models/test_base.py
├── External: pytest, torch, mock
├── Internal:
│   from tests.test_models import MockModel, ModelTestConfig
│   from thinkrl.models.base import BaseModel, ModelProtocol (if exists)
└── Tests base model functionality
```

## 📊 Dependency Statistics

### External Dependencies Count:
- **Core Required**: 5 packages
- **Optional GPU**: 2 packages (cupy variants)
- **ML Frameworks**: 7 packages
- **Multimodal**: 6 packages
- **Reasoning**: 5 packages
- **Monitoring**: 4 packages
- **Development**: 9 packages
- **Total Unique**: ~38 packages

### Internal Module Dependencies:
- **Level 0 (Foundation)**: 7 modules
- **Level 1 (Core)**: 24 modules
- **Level 2 (Aggregation)**: 8 modules
- **Level 3 (Training)**: 6 modules
- **Level 4 (Scripts)**: 5 modules
- **Test Modules**: 12 modules
- **Total Internal Modules**: ~62 modules

### Dependency Depth:
- **Maximum depth**: 4 levels
- **Average depth**: 2.3 levels
- **Circular dependencies**: 0

### Most Depended Upon (Internal):
1. `algorithms/base.py` - Used by 6 algorithm implementations
2. `utils/*` modules - Used throughout the codebase
3. `models/base.py` - Used by 4 model implementations
4. `training/trainer.py` - Used by 4 specialized trainers
5. `evaluation/metrics.py` - Used by evaluators and benchmarks

### Least Dependencies (Most Independent):
1. All `utils/*` modules - No internal dependencies
2. `reasoning/config.py` - No internal dependencies
3. `algorithms/base.py` - No internal dependencies
4. `models/base.py` - No internal dependencies
5. Prompt templates and constants

### Critical Path Dependencies:
```
scripts/train.py
└── training/trainer.py
    ├── algorithms/__init__.py
    │   └── algorithms/*.py
    │       └── algorithms/base.py
    ├── models/*.py
    │   └── models/base.py
    └── utils/*.py
```

This architecture ensures:
- **Modularity**: Each component can be developed and tested independently
- **Extensibility**: New algorithms/models can be added without modifying existing code
- **Maintainability**: Clear separation of concerns with no circular dependencies
- **Testability**: Mock implementations allow isolated testing
- **Performance**: Optional dependencies allow lightweight installations