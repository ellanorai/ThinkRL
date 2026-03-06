# REINFORCE++ Training Guide

REINFORCE++ is a critic-free reinforcement learning algorithm with variance reduction techniques and global advantage normalization. It's designed for efficient policy optimization without requiring a separate value network.

## Quick Start

```bash
# Activate your environment
source .venv/bin/activate

# Basic training command
reinforce-pp \
  -m Qwen/Qwen2.5-0.5B-Instruct \
  -r Qwen/Qwen2.5-0.5B-Instruct \
  -d openai/gsm8k \
  --dataset-config main \
  --prompt-column question \
  --target-column answer \
  --reward-fn reward_config.py:reward_fn \
  -o ./outputs/gsm8k
```

## Installation

```bash
# Clone the repository
git clone https://github.com/ellanorai/ThinkRL.git
cd ThinkRL

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install from source with all dependencies
pip install -e ".[sota]"
```

## CLI Reference

### Required Arguments

| Argument | Short | Description |
|----------|-------|-------------|
| `--model` | `-m` | Model name or HuggingFace path |
| `--ref-model` | `-r` | Reference model for KL divergence |
| `--dataset` | `-d` | Dataset name or path |

### Dataset Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset-config` | `None` | Config name (e.g., `main` for gsm8k) |
| `--dataset-split` | `train` | Dataset split to use |
| `--source` | `hf` | Source: `hf`, `local`, `json`, `csv` |
| `--prompt-column` | `prompt` | Column containing prompts |
| `--target-column` | `answer` | Column containing ground truth |
| `--max-samples` | `None` | Limit number of samples |

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--output-dir`, `-o` | `./reinforce_pp_output` | Output directory |
| `--mode` | `baseline` | `general` (k=1) or `baseline` (k>1) |
| `--group-size`, `-g` | `4` | Samples per prompt (baseline mode) |
| `--learning-rate`, `--lr` | `1e-6` | Learning rate |
| `--batch-size`, `-b` | `4` | Per-device batch size |
| `--grad-accum`, `-ga` | `1` | Gradient accumulation steps |
| `--epochs` | `1` | Number of training epochs |
| `--max-length` | `512` | Maximum sequence length |

### LoRA Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--lora-r` | `None` | LoRA rank (enables LoRA if set) |
| `--lora-init` | `default` | Init type: `default`, `pissa`, `garbage` |

### Optimization Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--entropy-coeff` | `0.01` | Entropy bonus coefficient |
| `--kl-coeff` | `0.1` | KL penalty coefficient |
| `--bf16/--no-bf16` | `True` | Use bfloat16 precision |
| `--fp16/--no-fp16` | `False` | Use float16 precision |
| `--flash-attn` | `False` | Enable Flash Attention 2 |
| `--gradient-checkpointing` | `False` | Save memory with checkpointing |
| `--deepspeed` | `None` | Path to DeepSpeed config |

### Logging Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--logging-backend` | `tensorboard` | `tensorboard`, `wandb`, or `none` |
| `--wandb-project` | `thinkrl-reinforce-pp` | W&B project name |

### Advanced Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--use-vllm` | `False` | Use vLLM for generation |
| `--vllm-group-port` | `51216` | NCCL port for vLLM sync |
| `--reward-fn` | `None` | Custom reward function path |

---

## Reward Functions

### Using the Universal Reward

ThinkRL includes a built-in `UniversalReward` that handles:
- **Math**: Numerical equivalence checking
- **Code**: Markdown block extraction and comparison
- **Text**: Normalized string matching
- **Structure**: `<think>...</think><answer>...</answer>` format validation

```bash
reinforce-pp ... --reward-fn reward_config.py:reward_fn
```

### Custom Reward Functions

Create a Python file with a `reward_fn` function:

```python
# my_reward.py
import torch

def reward_fn(prompts: list[str], completions: list[str], **kwargs) -> torch.Tensor:
    """
    Args:
        prompts: List of input prompts
        completions: List of model completions
        **kwargs: Contains 'targets' if --target-column is set
    
    Returns:
        Tensor of reward values (one per completion)
    """
    targets = kwargs.get("targets", None)
    rewards = []
    
    for i, completion in enumerate(completions):
        reward = 0.0
        
        # Your reward logic here
        if targets and i < len(targets):
            if targets[i].lower() in completion.lower():
                reward += 1.0
        
        rewards.append(reward)
    
    return torch.tensor(rewards, dtype=torch.float)
```

Use it:
```bash
reinforce-pp ... --reward-fn my_reward.py:reward_fn
```

---

## Training Modes

### Baseline Mode (Default)

Generates multiple completions per prompt and uses group-relative advantages:

```bash
reinforce-pp ... --mode baseline --group-size 4
```

- Computes advantage as: `A = r - mean(r_group)`
- Better variance reduction
- Recommended for most use cases

### General Mode

Standard REINFORCE with single sample per prompt:

```bash
reinforce-pp ... --mode general
```

- Simpler but higher variance
- Faster per step (fewer generations)

---

## Example Configurations

### Math (GSM8K)

```bash
reinforce-pp \
  -m Qwen/Qwen2.5-0.5B-Instruct \
  -r Qwen/Qwen2.5-0.5B-Instruct \
  -d openai/gsm8k \
  --dataset-config main \
  --prompt-column question \
  --target-column answer \
  --reward-fn reward_config.py:reward_fn \
  -o ./outputs/gsm8k \
  --mode baseline \
  -g 4 \
  -b 4 \
  --grad-accum 4 \
  --epochs 3 \
  --lr 5e-6 \
  --bf16 \
  --gradient-checkpointing \
  --logging-backend wandb
```

### Code (MBPP)

```bash
reinforce-pp \
  -m Qwen/Qwen2.5-0.5B-Instruct \
  -r Qwen/Qwen2.5-0.5B-Instruct \
  -d google-research-datasets/mbpp \
  --dataset-config sanitized \
  --prompt-column text \
  --target-column code \
  --reward-fn reward_config.py:reward_fn \
  -o ./outputs/mbpp \
  --logging-backend wandb
```

### With LoRA

```bash
reinforce-pp \
  -m meta-llama/Llama-3.1-8B-Instruct \
  -r meta-llama/Llama-3.1-8B-Instruct \
  -d openai/gsm8k \
  --dataset-config main \
  --lora-r 16 \
  --lora-init pissa \
  -o ./outputs/llama_lora
```

### With DeepSpeed

```bash
reinforce-pp \
  -m Qwen/Qwen2.5-7B-Instruct \
  -r Qwen/Qwen2.5-7B-Instruct \
  -d openai/gsm8k \
  --dataset-config main \
  --deepspeed configs/ds_zero2.json \
  -o ./outputs/qwen7b
```

---

## Monitoring Training

### W&B Metrics

| Metric | Description | Expected Behavior |
|--------|-------------|-------------------|
| `reward_mean` | Average reward per batch | Should increase over time |
| `reward_std` | Reward standard deviation | Normal variance |
| `ratio_mean` | Policy/Reference probability ratio | Near 1.0, slight movement |
| `policy_loss` | Policy gradient loss | Small, oscillating |
| `loss` | Total loss | Can be negative (OK) |
| `kl_k2_val` | KL divergence from reference | Slowly increasing |

### Interpreting Rewards

With `UniversalReward`:

| Score | Meaning |
|-------|---------|
| +1.2 | Perfect: correct structure + correct answer |
| +0.5 | Correct answer, broken structure |
| -0.5 | Wrong answer, broken structure |
| -1.5 | Missing answer block + structure penalty |

---

## Troubleshooting

### Dataset Config Error

```
ValueError: Config name is missing.
```

**Solution**: Add `--dataset-config <name>`:
```bash
reinforce-pp -d openai/gsm8k --dataset-config main ...
```

### Flash Attention Error

```
ImportError: flash_attn seems to be not installed
```

**Solution**: Install flash-attn or remove `--flash-attn`:
```bash
pip install flash-attn --no-build-isolation
```

### Out of Memory

**Solutions**:
1. Reduce `--batch-size`
2. Increase `--grad-accum`
3. Enable `--gradient-checkpointing`
4. Use `--lora-r` for parameter-efficient training
5. Use DeepSpeed: `--deepspeed configs/ds_zero2.json`

### Reward Not Improving

**Possible causes**:
1. Model too small — try larger model
2. Learning rate too low — try `--lr 5e-6` or higher
3. Not enough steps — train for more epochs
4. Reward function issue — check custom reward logic

---

## API Reference

### Python API

```python
from thinkrl.algorithms.reinforce_pp import REINFORCEPPConfig
from thinkrl.training.reinforce_pp_trainer import ReinforcePPTrainer
from thinkrl.models.loader import get_model
from thinkrl.data.datasets import RLHFDataset

# Load model
model = get_model("Qwen/Qwen2.5-0.5B-Instruct", model_type="actor")
ref_model = get_model("Qwen/Qwen2.5-0.5B-Instruct", model_type="ref")

# Load dataset
dataset = RLHFDataset(
    dataset_name_or_path="openai/gsm8k",
    tokenizer=tokenizer,
    dataset_config="main",
    prompt_column="question",
    target_column="answer",
)

# Configure trainer
config = REINFORCEPPConfig(
    learning_rate=1e-6,
    batch_size=4,
    mode="baseline",
    group_size=4,
)

trainer = ReinforcePPTrainer(
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    dataset=dataset,
    reward_fn=my_reward_fn,
    config=config,
)

# Train
trainer.train(steps=1000, batch_size=4)
```

---

## References

- [REINFORCE++ Paper](https://arxiv.org/abs/2501.03262)
- [ThinkRL GitHub](https://github.com/ellanorai/ThinkRL)
- [Qwen Models](https://huggingface.co/Qwen)
