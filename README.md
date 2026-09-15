<div align="center">
    <img alt="ThinkRL logo" src="assets/logo.png" style="height: 140px;" />
</div>
<div align="center">
<p align="center">
      <a href="https://github.com/ellanorai/ThinkRL/graphs/contributors">
        <img alt="GitHub Contributors" src="https://img.shields.io/github/contributors/ellanorai/ThinkRL" />
      </a>
      <a href="https://github.com/ellanorai/ThinkRL/issues">
        <img alt="Issues" src="https://img.shields.io/github/issues/ellanorai/ThinkRL?color=0088ff" />
      </a>
      <a href="https://github.com/ellanorai/ThinkRL/discussions">
        <img alt="Discussions" src="https://img.shields.io/github/discussions/ellanorai/ThinkRL?color=0088ff" />
      </a>
      <a href="https://github.com/ellanorai/ThinkRL/stargazers">
        <img alt="GitHub stars" src="https://img.shields.io/github/stars/ellanorai/ThinkRL?color=ccf" />
      </a>
      <a href="LICENSE">
        <img alt="License" src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" />
      </a>
      <br>
      <img alt="Development Status" src="https://img.shields.io/badge/status-alpha%20%2F%20work--in--progress-yellow" />
      <br>
      <em>Innovate / Optimize / Scale / Reasoning-Centric</em>
    </p>
</div>

<hr>

> **⚠️ Alpha / Work in Progress**: ThinkRL is under active development. Core infrastructure is production-ready, but some training loops and features are still being implemented. See [Implementation Status](#implementation-status) below.

ThinkRL is a **modular, high-performance, and reasoning-centric** open-source library for Reinforcement Learning from Human and AI Feedback (RLHF & RLAIF). It integrates **vLLM-based generation** with advanced policy optimization to enable scalable training of reasoning models (System 2) and standard LLMs.

📚 **Learn More**: [Documentation](https://thinkrl.readthedocs.io/) | [Technical Report](https://arxiv.org/abs/2507.06448)

## 📖 Table of Contents

- [🗞️ News](#news)
- [📊 Implementation Status](#implementation-status) - What's Ready vs In Progress
- [🏗️ Architecture](#architecture-foundation-vllm--pytorch) - vLLM + PyTorch Infrastructure
- [🎯 Reasoning Paradigm](#design-paradigm-reasoning-centric-execution) - Unified Reasoning Pipelines
- [🚀 Algorithms](#state-of-the-art-algorithms) - VAPO, DAPO, COPO, PAPO, GRPO
- [📋 Features](#comprehensive-features) - Full RLHF & RLAIF Pipeline
- [🎬 Quick Start](#quick-start) - Installation & Workflow
- [🎓 Training Guides](#training-guides) - SFT, CoT, RLHF
- [🔧 Advanced](#advanced-topics) - Process Rewards, LoRA

---

<a id="news"></a>
## 🗞️ News

<details>
<summary>View Latest Updates</summary>

- **[2026/02]** **Alpha Release**: ThinkRL now publicly available. Core infrastructure production-ready, training loops in active development.
- **[2026/01]** **ThinkRL 1.0**: Full support for **STaR (Self-Taught Reasoner)** and **Process Reward Models (PRM)**.
- **[2026/01]** Integrated **PAPO (Perception-Aware Policy Optimization)** for multimodal reasoning.
- **[2025/12]** Added **COPO (Count-based Online Preference Optimization)** for exploration-heavy tasks.
- **[2025/12]** Added **Dr. GRPO** (GRPO Done Right), which drops the standard-deviation normalization to keep the policy-gradient estimator unbiased.
- **[2025/11]** **VAPO** and **DAPO** algorithms merged into core.
- **[2025/10]** Complete **vLLM Integration** for 10x generation speedup during RLHF.

</details>

---

<a id="architecture-foundation-vllm--pytorch"></a>
## 🏗️ Architecture Foundation: vLLM + PyTorch

ThinkRL is built on a high-performance stack designed for scale:

<div align="center">
  <!-- Placeholder for architecture diagram -->
  <br>
  <b>vLLM Generation ⟺ PyTorch Training Loop ⟺ Distributed Strategy (DeepSpeed)</b>
  <br><br>
</div>

### Core Components

**vLLM - High-Throughput Inference**
RLHF depends heavily on generation speed, so ThinkRL can offload rollout generation to
[vLLM](https://github.com/vllm-project/vllm) and its PagedAttention and continuous batching.

This runs out of process: `thinkrl/integration/vllm_worker.py` is a standalone FastAPI
server holding the vLLM engine, and the trainers talk to it through
`thinkrl/integration/vllm_client.py`, pushing updated weights over NCCL between rollouts.
Pass `use_vllm=True` to `GRPOTrainer` or `ReinforcePPTrainer` to use it. See
[the vLLM worker guide](./docs/vllm_worker.md) for how to start one, and read the security
note there before exposing it: the worker binds every interface by default and has no
authentication.

The in-process `thinkrl.generation` engine is not implemented yet, and the "80% faster"
figure that used to sit here was never accompanied by a benchmark in this repository, so it
has been removed rather than restated.

**DeepSpeed - Memory-Efficient Training**
Native integration with [DeepSpeed](https://github.com/microsoft/DeepSpeed) (ZeRO-2/3) enables training **70B+ parameter** models on commodity hardware.

**Unified Loss Module**
All loss functions (DPO, PPO, VAPO, etc.) are centralized in a highly optimized `nn.Module` library, ensuring numerical stability and ease of extension.

---

<a id="design-paradigm-reasoning-centric-execution"></a>
## 🎯 Design Paradigm: Reasoning-Centric Execution

Unlike standard RLHF libraries, ThinkRL focuses on **Reasoning (System 2)** capabilities.

### Token-in-Token-out Agents
The design direction is to treat every model as an agent that consumes tokens
(observations/prompts) and produces tokens (thoughts/actions). This is a direction rather
than a shipped abstraction today, so it is worth being precise about what exists:
- **Agent executor**: `thinkrl.utils.agent` implements `AgentState`, `AgentInstanceBase`
  and `AgentExecutorBase`. *(experimental: no trainer, CLI or example runs it, and
  training on multi-turn rollouts needs a loss mask that covers only model-generated
  tokens and not tool output; see #130)*
- **Chain-of-Thought (CoT)**: Linear reasoning traces. *(planned, not implemented)*
- **Tree-of-Thought (ToT)**: Branching exploration. *(planned, not implemented)*
- **Multimodal Inputs**: Visual and textual context via PAPO. *(the algorithm is
  implemented and exported as of #90, and has no trainer; see #124)*

---

<a id="state-of-the-art-algorithms"></a>
## 🚀 State-of-the-Art Algorithms

ThinkRL implements standard baselines and cutting-edge **Reasoning-Aware** algorithms.

| Algorithm | Key Feature | Best Use Case |
|-----------|-------------|---------------|
| **PPO** | Proximal Policy Optimization | General purpose, stable alignment |
| **DPO / IPO** | Direct/Identity Preference Opt. | Offline preference learning |
| **GRPO** | Group Relative Policy Opt. | Reasoning with group baselines |
| **REINFORCE++** | Variance-reduced Policy Gradient | Efficient, low-memory RL |
| **VAPO** | **Value-Aware Policy Opt.** | Explicit value guidance for complex tasks |
| **DAPO** | **Dynamic Asymmetric Policy Opt.** | Long-horizon reasoning stability |
| **COPO** | **Count-based Online Pref. Opt.** | Exploration-heavy environments |
| **PAPO** | **Perception-Aware Policy Opt.** | Multimodal reasoning & grounding |
| **STaR** | **Self-Taught Reasoner** | Bootstrapping reasoning with hints |

---

<a id="comprehensive-features"></a>
## 📋 Comprehensive Features

ThinkRL provides a full-stack solution for modern alignment:

### 🧠 Reasoning & Verification
- **Process Reward Models (PRM)**: Step-by-step verification training.
- **STaR**: Self-Taught Reasoner bootstrapping loops.
- **Dual-System Training**: Joint training of System 1 (Intuition) and System 2 (Reasoning).

### ⚡ Optimization
- **Packing**: Sequence packing for 2x faster training.
- **LoRA / QLoRA**: Parameter-efficient fine-tuning.
- **Gradient Checkpointing**: Memory optimization for long contexts.

### 🔌 Integrations
- **Hugging Face**: Native `transformers` and `datasets` support.
- **WandB**: Experiment tracking and visualization.

---

<a id="quick-start"></a>
## 🎬 Quick Start

### Installation

> **Note**: ThinkRL is currently not available on PyPI. Please install from source.

```bash
# Clone the repository
git clone https://github.com/ellanorai/ThinkRL.git
cd ThinkRL

# Install from source
pip install -e .

# With vLLM and DeepSpeed support (Recommended)
pip install -e .[all]
```

### Typical Workflow

**1. Supervised Fine-Tuning (SFT)**

```python
from thinkrl.training import SFTConfig, SFTTrainer

config = SFTConfig(output_dir="./sft_output", num_train_epochs=1)

trainer = SFTTrainer(model=model, args=config, train_dataset=dataset, tokenizer=tokenizer)
trainer.train()
```

> **CoT / ToT trainers are not implemented yet.** `thinkrl.training.CoTTrainer` and
> `CoTConfig` do not exist and importing them raises `ImportError`; the modules under
> `thinkrl/reasoning/` are empty placeholders. See the In Development section below.

**2. RLHF with GRPO**

```bash
# GRPO is the training loop that is implemented end to end today.
thinkrl grpo \
    --model HuggingFaceTB/SmolLM2-135M \
    --dataset gsm8k \
    --steps 1000
```

> The generic `--algo` entry point does not exist yet: `thinkrl.cli.train_rl` is not a
> module, and the `train`, `sft`, `dpo` and `ppo` subcommands are stubs. GRPO and STaR are
> the two implemented CLIs.

---

<a id="training-guides"></a>
## 🎓 Training Guides

See `examples/` for runnable scripts. These four run on CPU with a small model:
- [Minimal GRPO run](./examples/basic/train_simple.py)
- [Inference from a model or checkpoint](./examples/basic/inference.py)
- [Evaluating a policy](./examples/basic/evaluate_model.py)
- [Supervised fine-tuning](./examples/sft/train_sft_small.py)

And these need a GPU:
- [GRPO on a reasoning dataset](./examples/reasoning/train_grpo.py)
- [REINFORCE++](./examples/reasoning/train_reinforce_pp.py)
- [STaR](./examples/reasoning/train_star.py)
- [Custom reward functions](./examples/example_universal_reward.py)

DPO, PRM and multimodal PAPO examples are not written yet.

---

<a id="implementation-status"></a>
## 📊 Implementation Status

### ✅ Production-Ready
- **Algorithm implementations**: PPO, GRPO, DPO, IPO, VAPO, DAPO, COPO, REINFORCE++
  (implemented and unit tested; see Training Loops below for which have a working CLI)
- **Models**: Actor, Critic, Reward Model, Process Reward Model (PRM)
- **Loss Functions**: Comprehensive loss module with 15+ implementations
- **Distributed Training**: DeepSpeed integration (ZeRO-2/3), distributed utilities
- **Data Pipeline**: Datasets, loaders, packing, processors
- **Utilities**: Metrics (GPU-accelerated), checkpointing, logging, KL controller
- **PEFT**: LoRA/QLoRA integration with multiple initialization strategies
- **vLLM Integration**: High-throughput generation client/worker

### 🚧 In Development
- **Training Loops**: REINFORCE++ is the only algorithm validated end to end against
  accuracy benchmarks. GRPO and STaR have working CLI commands that train, but have not
  been benchmark validated. SFT, DPO, PPO, ORPO, KTO and reward model training are not
  yet implemented and their CLI commands exit with an error.
- **CoT/ToT Trainers**: Chain-of-Thought and Tree-of-Thought training modules
- **Multimodal Training**: PAPO implementation for vision-language models
- **Complete Examples**: End-to-end training scripts

### 📋 Planned
- Evaluation harness integration
- Model serving infrastructure
- Additional reasoning algorithms (STaR fully integrated)

---

<a id="advanced-topics"></a>
## 🔧 Advanced

### Custom Reward Functions
ThinkRL supports plug-and-play reward functions for specialized domains (coding, math):

A reward function receives every completion in the batch and returns one reward per
completion, in prompt-major order:

```python
# my_reward.py
import torch

def reward_fn(prompts: list[str], completions: list[str], **kwargs) -> torch.Tensor:
    """kwargs contains 'targets' when --target-column is set."""
    targets = kwargs.get("targets") or [None] * len(completions)
    return torch.tensor(
        [1.0 if t and t in c else 0.0 for c, t in zip(completions, targets)],
        dtype=torch.float,
    )
```

```bash
thinkrl grpo ... --reward-fn my_reward.py:reward_fn
```

The bundled `UniversalReward` covers math, code and `<think>`/`<answer>` structure
checking; see `my_reward.py` in the repository root for a ready-made configuration.

### LoRA Merging
```bash
thinkrl merge \
    --base-model meta-llama/Llama-3-8b \
    --adapter ./checkpoints/final_lora \
    --output ./exported_model
```

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTION.md](CONTRIBUTION.md).

## 📄 Citation

```bibtex
@software{thinkrl2025,
  author = {Sood, Archit and EllanorAI Team},
  title = {ThinkRL: A Modular Library for Reasoning-Centric Reinforcement Learning},
  year = {2025},
  url = {https://github.com/ellanorai/ThinkRL}
}
```

## 📜 License

Apache License 2.0

---

<div align="center">
   Crafted by <a href="https://ellanorai.org">EllanorAI</a>
</div>
