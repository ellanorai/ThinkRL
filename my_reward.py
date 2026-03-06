from thinkrl.rewards.universal import UniversalReward


# Instantiate the UniversalReward with optimal defaults for math and reasoning
reward_fn = UniversalReward(
    format_reward=0.1,  # Reward for correct <think>...</think><answer>...</answer> tags
    answer_reward=1.0,  # Reward for matching the ground truth answer
    structure_penalty=-0.5,  # Penalty if the XML tags are completely missing
    math_tolerance=1e-4,  # Numerical tolerance for floating point comparisons
)
