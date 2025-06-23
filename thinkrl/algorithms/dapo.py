"""
Decoupled Clip and Dynamic sAmpling Policy Optimization (DAPO) algorithm implementation.

DAPO is a state-of-the-art reinforcement learning algorithm for training large language models
with long Chain-of-Thought reasoning. It achieves 50 points on AIME 2024 using Qwen2.5-32B,
outperforming previous methods with 50% fewer training steps.

Key Features:
- Clip-Higher: Decoupled clipping ranges to prevent entropy collapse
- Dynamic Sampling: Filters samples with zero gradients for efficiency
- Token-Level Policy Gradient Loss: Better handling of long sequences
- Overlong Reward Shaping: Reduces reward noise from truncated samples
- Group Relative Advantage Estimation: No value function required

References:
    - "DAPO: An Open-Source LLM Reinforcement Learning System at Scale"
    - ByteDance Seed & Tsinghua University, 2025
    - https://dapo-sia.github.io/
"""
