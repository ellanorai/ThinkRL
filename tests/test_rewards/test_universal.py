
import pytest
import torch

from thinkrl.rewards.universal import UniversalReward


@pytest.fixture
def universal_reward():
    return UniversalReward(
        format_reward=0.1, 
        answer_reward=1.0, 
        structure_penalty=-0.5,
        math_tolerance=1e-4,
        length_penalty=0.0 # Simplify testing
    )

def test_structure_reward_perfect(universal_reward):
    prompts = ["Q"]
    completions = ["<think>...</think><answer>A</answer>"]
    rewards = universal_reward(prompts, completions)
    # 2 * 0.1 (format) = 0.2
    assert torch.isclose(rewards[0], torch.tensor(0.2))

def test_structure_penalty(universal_reward):
    prompts = ["Q"]
    # Missing think -> Penalty
    completions = ["<answer>A</answer>"]
    rewards = universal_reward(prompts, completions, targets=["A"])
    # -0.5 (struct penalty) + 1.0 (correct answer) = 0.5
    assert torch.isclose(rewards[0], torch.tensor(0.5))
    
def test_missing_answer_penalty(universal_reward):
    prompts = ["2+2?"]
    completions = ["<think>4</think>"]
    targets = ["4"] # Target exists
    rewards = universal_reward(prompts, completions, targets=targets)
    # -0.5 (struct penalty, missing answer tag) - 1.0 (missing answer penalty) = -1.5
    assert torch.isclose(rewards[0], torch.tensor(-1.5))

def test_math_correctness_int(universal_reward):
    prompts = ["2+2?"]
    completions = ["<think>..</think><answer>4</answer>"]
    targets = ["4"]
    rewards = universal_reward(prompts, completions, targets=targets)
    # 0.2 (struct) + 1.0 (correct) = 1.2
    assert torch.isclose(rewards[0], torch.tensor(1.2))

def test_code_correctness_containment(universal_reward):
    prompts = ["def?"]
    # Code with extra noise/comments but containing target
    code_pred = "<think>..</think><answer>```python\n# Helper\ndef foo():\n  pass\n```</answer>"
    code_target = "def foo(): pass"
    rewards = universal_reward(prompts, [code_pred], targets=[code_target])
    # Should match via containment
    assert torch.isclose(rewards[0], torch.tensor(1.2))

def test_text_correctness_gaming(universal_reward):
    prompts = ["Explain?"]
    # Target is "Paris". Model pastes random junk.
    target = "Paris"
    junk = "a" * 100 
    comp = f"<think>..</think><answer>{target} {junk}</answer>"
    
    rewards = universal_reward(prompts, [comp], targets=[target])
    # 0.2 (struct) + 0.0 (wrong because ratio check fails) = 0.2
    assert torch.isclose(rewards[0], torch.tensor(0.2))

def test_length_penalty():
    scorer = UniversalReward(length_penalty=0.01)
    prompts = ["Q"]
    comp = "<think>a b c</think><answer>d</answer>" # 4 words appx
    # Base: 0.2 (format). Penalty: 0.01 * 4 = 0.04. Total 0.16. (approx word count check)
    rewards = scorer(prompts, [comp])
    # Implementation uses split() on whole completion string
    # "<think>a b c</think><answer>d</answer>" split might be complex depending on spaces
    # Let's trust logic: 0.2 - 0.01 * len
    expected = 0.2 - 0.01 * len(comp.split())
    assert torch.isclose(rewards[0], torch.tensor(expected, dtype=torch.float))
