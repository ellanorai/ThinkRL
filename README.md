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
- **[2025/12]** Released benchmarks for **REINFORCE++** and **Dr. GRPO** (Distributionally Robust GRPO).
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
RLHF depends heavily on generation speed. ThinkRL uses [vLLM](https://github.com/vllm-project/vllm) for **80% faster experience collection**, leveraging PagedAttention and continuous batching.

**DeepSpeed - Memory-Efficient Training**
Native integration with [DeepSpeed](https://github.com/microsoft/DeepSpeed) (ZeRO-2/3) enables training **70B+ parameter** models on commodity hardware.

**Unified Loss Module**
All loss functions (DPO, PPO, VAPO, etc.) are centralized in a highly optimized `nn.Module` library, ensuring numerical stability and ease of extension.

---

<a id="design-paradigm-reasoning-centric-execution"></a>
## 🎯 Design Paradigm: Reasoning-Centric Execution

Unlike standard RLHF libraries, ThinkRL focuses on **Reasoning (System 2)** capabilities.

### Token-in-Token-out Agents
We treat every model as an agent that consumes tokens (observations/prompts) and produces tokens (thoughts/actions). This unified interface supports:
- **Chain-of-Thought (CoT)**: Linear reasoning traces.
- **Tree-of-Thought (ToT)**: Branching exploration.
- **Multimodal Inputs**: Visual and textual context (via PAPO).

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

**1. Supervised Fine-Tuning (SFT) / CoT**

```python
from thinkrl.training import CoTTrainer, CoTConfig

config = CoTConfig(
    model_name_or_path="Qwen/Qwen2.5-7B",
    reasoning_type="cot",
    max_reasoning_steps=10
)

trainer = CoTTrainer(config)
trainer.train()
```

**2. RLHF with PPO/GRPO/VAPO**

```bash
# Launch generic RL training via CLI
python -m thinkrl.cli.train_rl \
    --algo vapo \
    --model_name_or_path meta-llama/Llama-3-8b \
    --reward_model_path ./rm_checkpoint \
    --use_vllm True
```

---

<a id="training-guides"></a>
## 🎓 Training Guides

See `examples/` for detailed scripts:
- [SFT & CoT Training](./examples/scripts/train_sft_cot.sh)
- [DPO / Online DPO](./examples/scripts/train_dpo.sh)
- [PRM Training](./examples/scripts/train_prm.sh)
- [Multimodal PAPO](./examples/scripts/train_papo.sh)

---

<a id="implementation-status"></a>
## 📊 Implementation Status

### ✅ Production-Ready
- **Core Algorithms**: PPO, GRPO, DPO, IPO, VAPO, DAPO, COPO, REINFORCE++
- **Models**: Actor, Critic, Reward Model, Process Reward Model (PRM)
- **Loss Functions**: Comprehensive loss module with 15+ implementations
- **Distributed Training**: DeepSpeed integration (ZeRO-2/3), distributed utilities
- **Data Pipeline**: Datasets, loaders, packing, processors
- **Utilities**: Metrics (GPU-accelerated), checkpointing, logging, KL controller
- **PEFT**: LoRA/QLoRA integration with multiple initialization strategies
- **vLLM Integration**: High-throughput generation client/worker

### 🚧 In Development
- **Training Loops**: GRPO CLI commands and Standalone scripts are fully Production-Ready. SFT, DPO, PPO are in progress.
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

```python
def math_reward(completion, answer):
    # Custom logic
    return 1.0 if verify_math(completion, answer) else 0.0
```

### LoRA Merging
```bash
python -m thinkrl.cli.merge_lora \
    --base_model meta-llama/Llama-3-8b \
    --lora_path ./checkpoints/final_lora \
    --output_path ./exported_model
```

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md).

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
