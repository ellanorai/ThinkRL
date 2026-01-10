---

<div align="center">
  <img src="assets/logo.png" alt="ThinkRL Logo" width="200"/>
  <h3>Innovate. Optimize. Scale.</h3>
  <p>An open-source library for Reinforcement Learning from Human and AI Feedback (RLHF & RLAIF)</p>
  <p>
    By <a href="https://github.com/Archit03">Archit Sood</a> ·
    <a href="https://ellanorai.org">EllanorAI</a>
  </p>
  <a href="https://github.com/ellanorai/ThinkRL">
    <img src="https://img.shields.io/github/stars/ellanorai/ThinkRL?style=social" alt="GitHub Stars">
  </a>
  <a href="https://pypi.org/project/thinkrl/">
    <img src="https://img.shields.io/pypi/v/thinkrl" alt="PyPI Version">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License">
  </a>
</div>

---

## Overview

ThinkRL is a modular, high-performance open-source library for large-scale reinforcement learning with human and AI feedback. It focuses on **reasoning-centric alignment**, providing production-grade implementations of modern policy optimization algorithms alongside structured reasoning techniques such as Chain-of-Thought and Tree-of-Thought. The library is designed to scale from single-GPU experimentation to distributed and multimodal training environments.

---

## 🎉 Latest Updates

### January 2026

* **Reasoning & Verification**: Added **Self-Taught Reasoner (STaR)** for bootstrapping reasoning and **Process Reward Models (PRM)** for step-by-step verification.
* **Preference Suite**: Full support for **Online DPO**, **KTO**, **IPO**, and **ORPO**, expanding alignment capabilities beyond standard RLHF.
* **Advanced Policy Optimization**: Introduced **PRIME**, **RLOO**, **REINFORCE++**, and **DR-GRPO** for robust and efficient training.

### December 2025

* **Algorithms**: Added **Direct Preference Optimization (DPO)** and **Decoupled Clip and Dynamic Sampling Policy Optimization (DAPO)** for preference alignment and long-horizon reasoning.

### November 2025

* **Performance**: Integrated **CuPy** for GPU-accelerated metric computation with a robust CPU fallback.
* **Core Infrastructure**: Completed and tested logging, checkpointing, metrics, and data pipelines.

### July 2025

* **Initial Release**: Core support for VAPO, DAPO, GRPO, PPO, and REINFORCE with reasoning-aware training loops.

---

## 🚀 Features

### 🧠 Algorithms

**Policy Optimization**
* **VAPO** — Value-model-based Augmented PPO with length-adaptive GAE
* **DAPO** — Decoupled Clip and Dynamic Sampling Policy Optimization
* **GRPO / DR-GRPO** — Group Relative Policy Optimization (Standard & Distributionally Robust)
* **PPO** — Enhanced Proximal Policy Optimization
* **REINFORCE / REINFORCE++** — Advanced policy gradient with variance reduction
* **PRIME** — Policy optimization for reasoning tasks
* **RLOO** — REINFORCE Leave-One-Out estimator

**Preference Optimization**
* **DPO / Online DPO** — Direct Preference Optimization
* **KTO** — Kahneman-Tversky Optimization
* **ORPO** — Odds Ratio Preference Optimization
* **IPO** — Identity Preference Optimization

### 🤔 Reasoning

* **STaR** — Self-Taught Reasoner (Bootstrapping)
* Chain-of-Thought (CoT) & Long-CoT
* Tree-of-Thought (ToT)
* Self-verification & Step-by-step Reward Modeling

### 🌐 Model Support

* **Process Reward Models (PRM)**
* GPT-style autoregressive models
* LLaMA 3 / 4 and Code LLaMA
* Qwen 2.5+
* T5 / BART
* Multimodal models (CLIP, BLIP and ViT)

### ⚡ Training & Optimization

* **LoRA / QLoRA (PEFT)**
* Singular Value Decomposition (SVD) for finetuning
* DeepSpeed (ZeRO-2 / ZeRO-3)
* Hugging Face integration
* Mixed precision (FP16 / BF16)

---

## 📦 Installation

### Core

```bash
pip install thinkrl

```

### Optional Features

```bash
pip install thinkrl[transformers]
pip install thinkrl[multimodal]
pip install thinkrl[peft]
pip install thinkrl[deepspeed]
pip install thinkrl[reasoning]
pip install thinkrl[wandb]
pip install thinkrl[all]

```

---

## 🎯 Quick Start

### Supervised Fine-Tuning (SFT)

```python
from thinkrl.training import SFTTrainer, SFTConfig

config = SFTConfig(
    model_name_or_path="facebook/opt-125m",
    output_dir="./outputs",
    train_batch_size=4
)

trainer = SFTTrainer(config)
# trainer.train()  # Start training
```

### Chain-of-Thought (CoT) Training

```python
from thinkrl.training.cot_trainer import CoTTrainer, CoTConfig

config = CoTConfig(
    model_name_or_path="Qwen/Qwen2.5-7B",
    reasoning_type="cot",
    max_reasoning_steps=10
)

# trainer = CoTTrainer(config)
# trainer.train()
```

---

## 🏗️ Project Structure

```text
ThinkRL/
├── algorithms/       # PPO, DPO, GRPO, STaR, PRIME, etc.
├── models/           # Actor, Critic, PRM, Reward Models
├── reasoning/        # CoT, ToT, Verification logic
├── training/         # Trainers (SFT, RLHF) and Loops
├── data/             # Datasets, Processors, Packing
├── peft/             # LoRA/QLoRA integration
├── distributed/      # DeepSpeed Strategies
├── utils/            # Metrics, Logging, Checkpointing
└── configs/          # YAML Configuration
```

---

## 🤝 Contributing

We welcome contributions! Please see our [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to get started, our code of conduct, and the process for submitting pull requests.

## 📄 Citation

If you use ThinkRL in your research, please cite:

```bibtex
@software{thinkrl2025,
  author = {Sood, Archit and EllanorAI Team},
  title = {ThinkRL: A Modular Library for Reasoning-Centric Reinforcement Learning},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/ellanorai/ThinkRL}}
}
```

## 📜 License

Apache License 2.0

---

<div align="center">
⭐ <strong>Star us on <a href="https://github.com/ellanorai/ThinkRL">GitHub</a> to support ThinkRL</strong>

Crafted by <a href="https://ellanorai.org">EllanorAI</a>
</div>
