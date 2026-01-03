# Changelog

All notable changes to the **ThinkRL** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [January 2026] - The Reasoning & Preference Expansion

### Added
- **Advanced Reasoning Algorithms**:
  - **STaR** (`star.py`): Implemented Self-Taught Reasoner for bootstrapping reasoning capabilities.
  - **PRIME** (`prime.py`): Added PRIME algorithm for advanced policy optimization.
- **Preference Optimization Suite**:
  - **Online DPO** (`online_dpo.py`): Support for online Direct Preference Optimization.
  - **ORPO** (`orpo.py`): Added Odds Ratio Preference Optimization.
  - **KTO** (`kto.py`): Implemented Kahneman-Tversky Optimization.
  - **IPO** (`ipo.py`): Added Identity Preference Optimization.
- **Policy Gradient Enhancements**:
  - **RLOO** (`rloo.py`): Added REINFORCE Leave-One-Out estimator.
  - **REINFORCE++** (`reinforce_pp.py`): Enhanced version of the standard REINFORCE algorithm.
  - **DR-GRPO** (`dr_grpo.py`): Added Distributionally Robust Group Relative Policy Optimization.
- **Model Architecture**:
  - **PRM** (`prm.py`): Introduced Process Reward Model support for step-by-step reasoning verification.

## [December 2025] - Core Algorithms & Infrastructure

### Added
- **Core Algorithms**:
  - **DPO** (`dpo.py`): Direct Preference Optimization.
  - **DAPO** (`dapo.py`): Decoupled Clip and Dynamic Sampling Policy Optimization.
- **Infrastructure**:
  - **SFT Trainer**: Dedicated trainer for Supervised Fine-Tuning.
  - **PEFT Integration**: Native LoRA configuration and injection.
  - **Distributed**: DeepSpeed ZeRO-2 and ZeRO-3 strategy wrappers.

## [November 2025] - Performance Optimization

### Added
- **Optimization**:
  - **Mixed Precision**: FP16/BF16 training support.
  - **Metrics**: GPU-accelerated metric computation using CuPy.
- **Data Pipeline**:
  - **Packing**: Efficient sequence packing for training throughput.
- **Logging**:
  - Integrated Weights & Biases and TensorBoard support.

## [July 2025] - Initial Release

### Added
- **Foundation**:
  - Initial implementations of **PPO**, **VAPO**, **GRPO**, and **REINFORCE**.
  - Basic Actor-Critic model wrappers.
  - Core configuration and utility systems.
