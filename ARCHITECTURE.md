# ThinkRL Architecture

A production-grade RLHF framework for reasoning-centric language models, inspired by TRL, OpenRLHF, and veRL.

## Design Principles

1. **Simplicity over complexity**: Minimal abstractions, explicit over implicit
2. **Composability**: Small, focused modules that combine cleanly
3. **DeepSpeed-first**: Native distributed training with ZeRO optimization
4. **PEFT as primitive**: LoRA/adapter support at the core, not bolted on
5. **No trainers yet**: Clean infrastructure without training orchestration

---

## Directory Structure

```
thinkrl/
├── __init__.py
├── algorithms/           # RLHF algorithms (PPO, GRPO, DPO, DAPO, VAPO, REINFORCE)
│   ├── __init__.py
│   ├── base.py          # BaseRLHFAlgorithm
│   ├── ppo.py
│   ├── grpo.py
│   ├── dpo.py
│   ├── dapo.py
│   ├── vapo.py
│   └── reinforce.py
├── models/              # Model wrappers and loss functions
│   ├── __init__.py
│   ├── actor.py         # Policy model
│   ├── critic.py        # Value model
│   ├── reward_model.py  # Reward scoring
│   ├── model.py         # Factory functions
│   └── loss.py          # All loss functions
├── distributed/         # DeepSpeed integration
│   ├── __init__.py
│   ├── deepspeed.py     # DeepSpeed engine wrapper
│   ├── strategies.py    # ZeRO-2, ZeRO-3 strategies
│   └── utils.py         # Distributed utilities
├── data/                # Dataset handling
│   ├── __init__.py
│   ├── datasets.py      # Dataset classes
│   ├── loaders.py       # DataLoader utilities
│   ├── collators.py     # Collate functions
│   └── templates.py     # Chat template handling
├── peft/                # PEFT/LoRA integration
│   ├── __init__.py
│   ├── lora.py          # LoRA configuration and utilities
│   └── adapters.py      # Adapter injection/merging
├── config/              # Configuration system
│   ├── __init__.py
│   ├── base.py          # Base config classes
│   ├── algorithm.py     # Algorithm configs
│   ├── model.py         # Model configs
│   ├── training.py      # Training configs
│   └── schema.py        # YAML schema validation
├── logging/             # Experiment tracking
│   ├── __init__.py
│   ├── loggers.py       # Logger implementations
│   ├── wandb.py         # W&B integration
│   └── tensorboard.py   # TensorBoard integration
├── cli/                 # Command-line interface
│   ├── __init__.py
│   └── main.py          # CLI entrypoints
├── utils/               # Core utilities (existing, enhanced)
└── generation/          # Generation engines
    ├── __init__.py
    └── vllm_engine.py   # vLLM integration
```

---

## 1. Codebase Audit & Pruning

### Components to KEEP (Already Well-Implemented)
- `algorithms/`: PPO, GRPO, DPO, DAPO, VAPO - clean implementations
- `utils/metrics.py`: Comprehensive metrics with GPU acceleration
- `utils/checkpoint.py`: CheckpointManager with SafeTensors
- `utils/logging.py`: Good logging foundation
- `utils/distributed_util.py`: Distributed primitives
- `utils/seqlen_balancing.py`: Sequence balancing algorithms
- `models/`: Actor, Critic, RewardModel, loss functions

### Components to REMOVE/SIMPLIFY
- `reasoning/`: Over-engineered CoT/ToT - reasoning is better handled in prompting
- `evaluation/`: Stub files with no implementation
- `registry/`: Unnecessary abstraction for this scale
- `scripts/`: Empty training scripts

### Components to IMPLEMENT
- `distributed/deepspeed.py`: DeepSpeed engine wrapper
- `peft/lora.py`: First-class LoRA support
- `config/`: YAML-based configuration
- `cli/main.py`: Command-line interface
- `algorithms/reinforce.py`: REINFORCE baseline

---

## 2. Distributed Training Infrastructure (DeepSpeed)

### DeepSpeed Strategy Hierarchy

```python
# thinkrl/distributed/strategies.py

class DeepSpeedStrategy:
    """Base strategy for DeepSpeed configuration."""

    def get_config(self) -> dict:
        raise NotImplementedError

class ZeRO2Strategy(DeepSpeedStrategy):
    """ZeRO Stage 2: Optimizer state + gradient partitioning."""

    def __init__(
        self,
        offload_optimizer: bool = False,
        offload_param: bool = False,
        overlap_comm: bool = True,
        reduce_bucket_size: int = 500_000_000,
        allgather_bucket_size: int = 500_000_000,
    ): ...

class ZeRO3Strategy(DeepSpeedStrategy):
    """ZeRO Stage 3: Full sharding across devices."""

    def __init__(
        self,
        offload_optimizer: bool = True,
        offload_param: bool = True,
        sub_group_size: int = 1_000_000_000,
        stage3_max_live_parameters: int = 1_000_000_000,
        stage3_prefetch_bucket_size: int = 500_000_000,
    ): ...
```

### DeepSpeed Engine Wrapper

```python
# thinkrl/distributed/deepspeed.py

class DeepSpeedEngine:
    """Unified DeepSpeed engine wrapper."""

    def __init__(
        self,
        model: nn.Module,
        strategy: DeepSpeedStrategy,
        optimizer: Optimizer | None = None,
        scheduler: LRScheduler | None = None,
        gradient_accumulation_steps: int = 1,
        gradient_clipping: float = 1.0,
    ): ...

    def forward(self, *args, **kwargs) -> Any: ...
    def backward(self, loss: Tensor) -> None: ...
    def step(self) -> None: ...
    def save_checkpoint(self, path: str) -> None: ...
    def load_checkpoint(self, path: str) -> None: ...
```

---

## 3. Dataset System

### Dataset Hierarchy

```python
# thinkrl/data/datasets.py

class BaseDataset(Dataset):
    """Base class for all datasets."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
        chat_template: str | None = None,
    ): ...

class PromptDataset(BaseDataset):
    """Dataset for prompt-only data (PPO/GRPO rollouts)."""

    @classmethod
    def from_huggingface(cls, name: str, split: str = "train", **kwargs): ...

    @classmethod
    def from_jsonl(cls, path: str, **kwargs): ...

class PreferenceDataset(BaseDataset):
    """Dataset for preference pairs (DPO/reward modeling)."""

    # Supports: chosen/rejected pairs, rankings, trajectory-based

class SFTDataset(BaseDataset):
    """Dataset for supervised fine-tuning."""

class StreamingDataset(BaseDataset):
    """Streaming dataset for large-scale data."""

    def __init__(self, data_files: list[str], buffer_size: int = 10000): ...
```

### Chat Template System

```python
# thinkrl/data/templates.py

TEMPLATES = {
    "chatml": "<|im_start|>{role}\n{content}<|im_end|>",
    "llama2": "[INST] {content} [/INST]",
    "mistral": "[INST] {content} [/INST]",
    "qwen": "<|im_start|>{role}\n{content}<|im_end|>",
}

def apply_chat_template(
    messages: list[dict],
    tokenizer: PreTrainedTokenizer,
    template: str | None = None,
    add_generation_prompt: bool = True,
) -> str: ...
```

---

## 4. Model Loading & Initialization

### Model Factory

```python
# thinkrl/models/model.py

def load_model(
    model_name_or_path: str,
    model_type: Literal["actor", "critic", "reward"] = "actor",
    # Precision
    torch_dtype: torch.dtype = torch.bfloat16,
    use_flash_attention: bool = True,
    # Quantization
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    bnb_4bit_compute_dtype: torch.dtype = torch.bfloat16,
    bnb_4bit_quant_type: str = "nf4",
    # PEFT
    peft_config: PeftConfig | None = None,
    # Trust
    trust_remote_code: bool = True,
    # Device
    device_map: str | dict = "auto",
) -> nn.Module: ...

def create_reference_model(
    model: nn.Module,
    share_weights: bool = True,
) -> nn.Module: ...
```

### Key Loading Features
- Automatic FlashAttention 2 detection and enablement
- BitsAndBytes 4-bit/8-bit quantization
- PEFT injection at load time
- DeepSpeed-compatible initialization

---

## 5. PEFT/LoRA Integration

### First-Class LoRA Support

```python
# thinkrl/peft/lora.py

@dataclass
class LoRAConfig:
    """LoRA configuration - aligned with peft library."""

    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: list[str] | None = None
    modules_to_save: list[str] | None = None
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    # Default targets for common architectures
    @classmethod
    def for_llama(cls, r: int = 8) -> "LoRAConfig":
        return cls(
            r=r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        )

    @classmethod
    def for_qwen(cls, r: int = 8) -> "LoRAConfig":
        return cls(
            r=r,
            target_modules=["c_attn", "c_proj", "w1", "w2"],
        )

def inject_lora(
    model: nn.Module,
    config: LoRAConfig,
) -> PeftModel: ...

def merge_lora(model: PeftModel) -> nn.Module: ...

def get_trainable_parameters(model: nn.Module) -> dict[str, int]: ...
```

### DeepSpeed + LoRA Compatibility

```python
# Handles the complexity of LoRA + ZeRO-3
def prepare_model_for_deepspeed(
    model: nn.Module,
    lora_config: LoRAConfig | None = None,
    zero_stage: int = 2,
) -> nn.Module: ...
```

---

## 6. Logging & Experiment Tracking

### Unified Logger Interface

```python
# thinkrl/logging/loggers.py

class Logger(ABC):
    """Abstract base logger."""

    @abstractmethod
    def log(self, metrics: dict[str, float], step: int) -> None: ...

    @abstractmethod
    def log_hyperparams(self, params: dict) -> None: ...

    @abstractmethod
    def finish(self) -> None: ...

class CompositeLogger(Logger):
    """Combines multiple loggers."""

    def __init__(self, loggers: list[Logger]): ...

class ConsoleLogger(Logger):
    """Logs to stdout with formatting."""

class WandBLogger(Logger):
    """Weights & Biases integration."""

    def __init__(
        self,
        project: str,
        name: str | None = None,
        config: dict | None = None,
        tags: list[str] | None = None,
    ): ...

class TensorBoardLogger(Logger):
    """TensorBoard integration."""

    def __init__(self, log_dir: str): ...
```

### Metric Naming Convention

```
{phase}/{metric_name}

Examples:
  train/loss
  train/policy_loss
  train/kl_divergence
  eval/reward_mean
  rollout/tokens_per_second
```

### Distributed-Safe Logging

```python
def log_only_main_process(logger: Logger, metrics: dict, step: int) -> None:
    """Only log on rank 0 to avoid duplicate entries."""
    if is_main_process():
        logger.log(metrics, step)
```

---

## 7. Configuration System

### YAML-Based Configuration

```yaml
# configs/ppo_llama.yaml
model:
  name_or_path: "meta-llama/Llama-3.1-8B"
  torch_dtype: "bfloat16"
  use_flash_attention: true

peft:
  type: "lora"
  r: 8
  lora_alpha: 16
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]

algorithm:
  name: "ppo"
  learning_rate: 1e-5
  kl_coeff: 0.1
  gamma: 0.99
  gae_lambda: 0.95
  clip_epsilon: 0.2
  n_epochs: 4

distributed:
  strategy: "zero2"
  gradient_accumulation_steps: 4
  offload_optimizer: false

data:
  dataset: "Anthropic/hh-rlhf"
  max_length: 2048
  batch_size: 4

logging:
  backends: ["wandb", "tensorboard"]
  project: "thinkrl-experiments"
  log_every_n_steps: 10
```

### Config Dataclasses

```python
# thinkrl/config/base.py

@dataclass
class ThinkRLConfig:
    """Root configuration object."""

    model: ModelConfig
    algorithm: AlgorithmConfig
    distributed: DistributedConfig
    data: DataConfig
    logging: LoggingConfig
    peft: PeftConfig | None = None

    @classmethod
    def from_yaml(cls, path: str) -> "ThinkRLConfig": ...

    def to_yaml(self, path: str) -> None: ...

    def validate(self) -> None:
        """Validate configuration consistency."""
        ...
```

---

## 8. CLI Design

### Command Structure

```bash
# Train with config file
thinkrl train --config configs/ppo_llama.yaml

# Train with CLI overrides
thinkrl train --config configs/base.yaml \
    --algorithm.learning_rate 1e-5 \
    --distributed.strategy zero3

# Generate rollouts
thinkrl generate --model path/to/model --prompts data/prompts.jsonl

# Evaluate
thinkrl eval --model path/to/model --dataset Anthropic/hh-rlhf

# Merge LoRA adapters
thinkrl merge --base-model meta-llama/Llama-3.1-8B --adapter path/to/adapter

# Launch distributed (wrapper for torchrun/deepspeed)
thinkrl launch --nnodes 2 --nproc_per_node 8 train --config configs/ppo.yaml
```

### CLI Implementation

```python
# thinkrl/cli/main.py

import typer

app = typer.Typer(name="thinkrl", help="ThinkRL RLHF Training Framework")

@app.command()
def train(
    config: Path = typer.Option(..., "--config", "-c"),
    overrides: list[str] = typer.Option(None, "--override", "-o"),
): ...

@app.command()
def generate(
    model: str = typer.Option(..., "--model", "-m"),
    prompts: Path = typer.Option(..., "--prompts", "-p"),
    output: Path = typer.Option("outputs/generations.jsonl", "--output", "-o"),
): ...

@app.command()
def merge(
    base_model: str = typer.Option(..., "--base-model"),
    adapter: Path = typer.Option(..., "--adapter"),
    output: Path = typer.Option(..., "--output"),
): ...
```

---

## 9. Additional Patterns from TRL/OpenRLHF/veRL

### Memory-Efficient Rollout Generation
- Use vLLM for high-throughput generation
- Gradient checkpointing during training
- Offload reference model to CPU during generation

### Reference Policy Handling
```python
# Share weights with policy, only clone for inference
ref_model = create_reference_model(policy_model, share_weights=True)

# During training, detach for KL computation
with torch.no_grad():
    ref_logits = ref_model(input_ids)
```

### Reward Normalization
```python
def normalize_rewards(
    rewards: Tensor,
    method: str = "running",  # "batch", "running", "none"
    running_mean: float | None = None,
    running_std: float | None = None,
) -> tuple[Tensor, float, float]: ...
```

### KL Control Mechanisms
```python
class KLController:
    """Adaptive KL coefficient controller."""

    def __init__(
        self,
        init_kl_coef: float = 0.1,
        target_kl: float = 0.01,
        horizon: int = 10000,
    ): ...

    def update(self, current_kl: float) -> float: ...
```

### Sequence-Length-Aware Optimization
```python
# Already in utils/seqlen_balancing.py
# Use karmarkar_karp for balanced micro-batches
partitions = karmarkar_karp(sequence_lengths, num_partitions)
```

---

## 10. What NOT to Include

### Explicitly Excluded
1. **Trainer classes** - Build infrastructure only, users compose training loops
2. **Ray integration** - Focus on DeepSpeed for simplicity
3. **Model parallelism** - ZeRO-3 covers most use cases
4. **Custom CUDA kernels** - Use FlashAttention and existing optimizations
5. **Evaluation harness** - Use lm-eval-harness externally
6. **Registry patterns** - Direct imports are clearer

### Rationale
- Keep the codebase < 10k lines
- Every abstraction must earn its place
- Prefer composition over inheritance
- Users should understand the code, not fight it

---

## Implementation Priority

1. **Phase 1: Core Infrastructure**
   - [x] Algorithms (PPO, GRPO, DPO, DAPO, VAPO)
   - [x] Models (Actor, Critic, RewardModel, losses)
   - [x] Utils (metrics, checkpoint, logging, distributed)
   - [ ] REINFORCE algorithm
   - [ ] DeepSpeed engine wrapper
   - [ ] PEFT/LoRA integration

2. **Phase 2: Configuration & CLI**
   - [ ] YAML config system
   - [ ] CLI with typer
   - [ ] Config validation

3. **Phase 3: Logging & Data**
   - [ ] W&B/TensorBoard loggers
   - [ ] Enhanced dataset classes
   - [ ] Chat template support

4. **Phase 4: Polish**
   - [ ] Clean up stubs
   - [ ] Documentation
   - [ ] Tests
