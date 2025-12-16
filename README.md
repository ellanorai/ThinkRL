
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

* **VAPO** — Value-model-based Augmented PPO with length-adaptive GAE
* **DAPO** — Decoupled Clip and Dynamic Sampling Policy Optimization
* **GRPO** — Group Relative Policy Optimization
* **PPO** — Enhanced Proximal Policy Optimization
* **REINFORCE** — Policy gradient with variance reduction

### 🤔 Reasoning

* Chain-of-Thought (CoT)
* Tree-of-Thought (ToT)
* Long-CoT for extended reasoning
* Self-verification for consistency

### 🌐 Model Support

* GPT-style autoregressive models
* LLaMA 3 / 4 and Code LLaMA
* Qwen 2.5+
* T5 / BART
* Multimodal models (CLIP, BLIP)

### ⚡ Training & Optimization

* LoRA / QLoRA (PEFT)
* DeepSpeed (ZeRO)
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

### Basic RLHF Training

```python
from thinkrl import RLHFTrainer, ModelConfig

config = ModelConfig(
    model_name_or_path="microsoft/DialoGPT-small",
    model_type="gpt",
    algorithm="vapo"
)

trainer = RLHFTrainer(config)
trainer.train()
```

### Chain-of-Thought Training

```python
from thinkrl import CoTTrainer, ReasoningConfig

config = ReasoningConfig(
    model_name_or_path="Qwen/Qwen2.5-8B",
    reasoning_type="cot",
    max_reasoning_steps=10
)

trainer = CoTTrainer(config)
trainer.train()
```

---

## 🏗️ Project Structure

```text
ThinkRL/
├── algorithms/
├── models/
├── reasoning/
├── training/
├── data/
├── peft/
├── utils/
├── configs/
└── logs/
```

---

## 📜 License

Apache License 2.0

---

<div align="center">
  ⭐ <strong>Star us on <a href="https://github.com/ellanorai/ThinkRL">GitHub</a> to support ThinkRL</strong><br/>
  Crafted by <a href="https://ellanorai.org">EllanorAI</a>
</div>

---
