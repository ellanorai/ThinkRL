from thinkrl.rewards.universal import UniversalReward


# Initialize the UniversalReward with production-grade settings
# Adjust these coefficients based on your specific goal
universal_reward = UniversalReward(
    format_reward=0.1,      # Slight reward for correct <think><answer> tags
    answer_reward=1.0,      # Main reward for correct answers
    structure_penalty=-0.5, # Penalty for missing or malformed tags
    length_penalty=0.001,   # Word-level penalty to prevent verbosity (adjust as needed)
    math_tolerance=1e-5     # Tolerance for numerical math answers
)

def reward_fn(prompts, completions, **kwargs):
    """
    Reward function entry point for the ThinkRL reinforce_pp CLI.
    
    Args:
        prompts (list[str]): List of prompt strings.
        completions (list[str]): List of completion strings.
        **kwargs: Dictionary containing 'targets' if provided by the dataset.
        
    Returns:
        torch.Tensor: Tensor of rewards for the batch.
    """
    # The UniversalReward class handles domain routing (Math, Code, Text)
    # and structural validation automatically.
    return universal_reward(prompts, completions, **kwargs)
