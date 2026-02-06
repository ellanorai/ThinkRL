
import torch
from thinkrl.rewards.universal import UniversalReward

def main():
    # Initialize the Universal Reward
    # Adjust weights as needed
    reward_fn = UniversalReward(
        format_reward=0.1,      # For <think>...</think><answer>...</answer>
        answer_reward=1.0,      # For correctness
        structure_penalty=-0.5, # For missing tags
        math_tolerance=1e-4,    # For numerical comparison
        length_penalty=0.0      # Optional length penalty
    )
    
    # Example 1: Math
    prompts = ["What is 2 + 2?"]
    completions = ["<think>It is 4.</think><answer>4.0</answer>"]
    targets = ["4"]
    
    rewards = reward_fn(prompts, completions, targets=targets)
    print(f"Math Reward: {rewards[0].item()}") 
    # Expect: 0.1 + 0.1 + 1.0 = 1.2
    
    # Example 2: Code
    prompts_code = ["Write a function to add two numbers."]
    # Note: Whitespace in code is normalized for comparison
    comp_code = """<think>Easy.</think><answer>
    ```python
    def add(a, b):
        return a + b
    ```
    </answer>"""
    target_code = "def add(a, b): return a + b"
    
    rewards_code = reward_fn(prompts_code, [comp_code], targets=[target_code])
    print(f"Code Reward: {rewards_code[0].item()}")
    # Expect: 1.2
    
    # Example 3: Text (Case Insensitive)
    prompts_text = ["Capital of France?"]
    comp_text = "<think>..</think><answer>paris</answer>"
    target_text = "Paris"
    
    rewards_text = reward_fn(prompts_text, [comp_text], targets=[target_text])
    print(f"Text Reward: {rewards_text[0].item()}")
    # Expect: 1.2
    
    # Example 4: Penalties
    # Missing answer block
    comp_missing = "<think>I don't know.</think>"
    rewards_missing = reward_fn(["Q"], [comp_missing], targets=["A"])
    print(f"Missing Answer Reward: {rewards_missing[0].item()}")
    # Expect: -0.5 (structure) - 1.0 (missing answer penalty) = -1.5
    
    # Broken structure
    comp_broken = "<answer>A</answer>"
    rewards_broken = reward_fn(["Q"], [comp_broken], targets=["A"])
    print(f"Broken Structure Reward: {rewards_broken[0].item()}")
    # Expect: -0.5 (structure) + 1.0 (correct) = 0.5

    print("Success! Universal Reward works across domains.")

if __name__ == "__main__":
    main()
