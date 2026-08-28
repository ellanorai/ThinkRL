# GRPO Training Guide

Group Relative Policy Optimization, as described in
[DeepSeekMath](https://arxiv.org/abs/2402.03300), Appendix A.1.6.

GRPO removes the critic used by PPO. For each prompt it samples a group of `G` completions,
normalizes their rewards within the group to obtain advantages, and optimizes a PPO-style
clipped surrogate with a direct KL penalty against a frozen reference model.

> **Status**: the CLI trains end to end, but GRPO has not been validated against accuracy
> benchmarks. Treat results as unverified. REINFORCE++ is the validated algorithm.

## Quick Start

```bash
# Activate your environment
source .venv/bin/activate

# Basic training command
thinkrl grpo \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --ref-model Qwen/Qwen2.5-0.5B-Instruct \
    --dataset openai/gsm8k \
    --dataset-config main \
    --prompt-column question \
    --target-column answer \
    --group-size 8 \
    --batch-size 2 \
    --reward-fn my_reward.py:reward_fn
```

A reference model is required. GRPO's KL penalty is computed against it, and the command
exits with an error if `--ref-model` is omitted.

## Installation

```bash
git clone https://github.com/ellanorai/ThinkRL.git
cd ThinkRL

python -m venv .venv
source .venv/bin/activate

pip install -e .
```

A CUDA GPU is required. The dependency set includes CUDA-only packages and the training
loop holds a policy model, a reference model, and a full group of rollouts at once.

## CLI Reference

### Required Arguments

| Flag | Description |
| --- | --- |
| `--model`, `-m` | Policy model name or path |
| `--ref-model`, `-r` | Reference model for the KL penalty |
| `--dataset`, `-d` | Prompt dataset name or path |

### Dataset Arguments

| Flag | Default | Description |
| --- | --- | --- |
| `--source`, `-s` | `hf` | One of `hf`, `local`, `json`, `csv` |
| `--dataset-split` | `train` | Split to load |
| `--dataset-config` | `None` | Config name, for example `main` for gsm8k |
| `--prompt-column`, `-pc` | `prompt` | Column holding the prompt |
| `--target-column` | `answer` | Column holding the reference answer |
| `--max-samples` | all | Cap on samples loaded |
| `--max-length` | `512` | Maximum prompt length in tokens |
| `--system-prompt` | reasoning preset | Prepended to every prompt |

`--max-length` truncates the prompt only. Completion length is currently fixed at 256 new
tokens and is not exposed on the command line.

### Training Arguments

| Flag | Default | Description |
| --- | --- | --- |
| `--group-size`, `-g` | `64` | Completions sampled per prompt (`G`) |
| `--batch-size`, `-b` | `4` | Prompts per device per step |
| `--epochs` | `1` | Passes over the dataset |
| `--learning-rate`, `--lr` | `1e-6` | Optimizer learning rate |
| `--kl-coeff` | `0.04` | KL penalty coefficient (`beta`) |
| `--grad-accum`, `-ga` | `1` | Gradient accumulation steps |

Each step generates `batch_size * group_size` sequences. The defaults produce 256
sequences per step, which will not fit on a single consumer GPU. Start with
`--group-size 8 --batch-size 2` and increase from there.

The step count is `epochs * dataset_size // batch_size`. If that evaluates to zero the
command exits with an error rather than reporting a completed run.

### LoRA Arguments

| Flag | Default | Description |
| --- | --- | --- |
| `--lora-r` | `None` | LoRA rank; LoRA is enabled when set |
| `--lora-init` | `default` | One of `default`, `garbage`, `pissa`, `pissa_niter_[n]` |

### Optimization Arguments

| Flag | Default | Description |
| --- | --- | --- |
| `--bf16` / `--no-bf16` | `--bf16` | bfloat16 precision |
| `--fp16` / `--no-fp16` | `--no-fp16` | float16 precision; takes priority over bf16 |
| `--flash-attn` / `--no-flash-attn` | `--no-flash-attn` | Flash Attention 2 |
| `--gradient-checkpointing` | off | Trade compute for memory |
| `--deepspeed` | `None` | Path to a DeepSpeed config, for example `configs/ds_zero2.json` |

### Logging Arguments

| Flag | Default | Description |
| --- | --- | --- |
| `--logging-backend` | `tensorboard` | One of `tensorboard`, `wandb`, `none` |
| `--wandb-project` | `thinkrl-grpo` | W&B project name |
| `--output-dir`, `-o` | `./grpo_output` | Where the trained model is written |

With `--logging-backend wandb` the following are recorded each step: `train/loss`,
`train/reward_mean`, `train/reward_std`, `train/kl_mean`, `train/advantage_mean`,
`train/clip_fraction`, `train/grad_norm`, `train/epoch`.

### Advanced Arguments

| Flag | Default | Description |
| --- | --- | --- |
| `--use-vllm` | `false` | Use a vLLM server for generation; accepts `true` or `false` |
| `--vllm-group-port` | `51216` | NCCL group port for weight synchronization |
| `--dry-run` | off | Build models, dataset and trainer, then exit before training |

The vLLM server address is currently fixed at `http://localhost:8000` and the weight
synchronization group is fixed at one trainer rank plus one server.

## Reward Functions

### Using the Universal Reward

`UniversalReward` handles numerical equivalence for math, markdown block extraction for
code, normalized string matching for text, and `<think>...</think><answer>...</answer>`
structure validation. The repository root contains a ready-made configuration:

```bash
thinkrl grpo ... --reward-fn my_reward.py:reward_fn
```

### Custom Reward Functions

```python
# my_reward.py
import torch

def reward_fn(prompts: list[str], completions: list[str], **kwargs) -> torch.Tensor:
    """
    Args:
        prompts: One entry per completion, repeated group_size times per prompt.
        completions: The generated text.
        **kwargs: Contains 'targets' when --target-column is set, aligned with completions.

    Returns:
        One reward per completion, in the same order as `completions`.
    """
    targets = kwargs.get("targets") or [None] * len(completions)
    rewards = [
        1.0 if target and target.strip() in completion else 0.0
        for completion, target in zip(completions, targets)
    ]
    return torch.tensor(rewards, dtype=torch.float)
```

The returned tensor must have exactly `batch_size * group_size` entries. A mismatch raises
a `ValueError` naming both counts.

If `--reward-fn` is omitted, a placeholder reward equal to the character length of the
completion is used. It exists to let the loop run and will train the model to produce
longer output. Do not use it for real training.

### Zero-variance groups

Advantages are normalized within each group. When every completion for a prompt receives
the same reward the group's standard deviation is zero, its advantages are zero, and it
contributes no gradient. This is expected early in a run, and the trainer logs a warning
with the number of affected groups. Persistent warnings mean the reward function is not
discriminating between completions.

## Example Configurations

### Math (GSM8K), single 24 GB GPU

```bash
thinkrl grpo \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --ref-model Qwen/Qwen2.5-0.5B-Instruct \
    --dataset openai/gsm8k --dataset-config main \
    --prompt-column question --target-column answer \
    --group-size 8 --batch-size 2 --max-length 512 \
    --lora-r 16 --bf16 --gradient-checkpointing \
    --reward-fn my_reward.py:reward_fn \
    --logging-backend wandb
```

### Local JSONL

Each line needs a prompt field and, if the reward function uses targets, an answer field:

```json
{"prompt": "What is 12 plus 30?", "answer": "42"}
```

```bash
thinkrl grpo \
    --model ./my-model --ref-model ./my-model \
    --dataset ./data.jsonl --source json \
    --group-size 8 --batch-size 2 \
    --reward-fn my_reward.py:reward_fn
```

### Validating a configuration without training

```bash
thinkrl grpo ... --dry-run
```

Loads both models, the tokenizer, the dataset and the trainer, then exits. Use it to catch
argument and memory problems before committing to a run.
