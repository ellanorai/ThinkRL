import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from thinkrl.algorithms.kto import KTOConfig
from thinkrl.training.kto_trainer import KTOTrainer
from thinkrl.data.datasets import RLHFDataset
import tempfile
import json
import os
import logging

logging.basicConfig(level=logging.ERROR) # Suppress massive logs

def evaluate(model, tokenizer, prompts, device="cuda"):
    model.eval()
    total_reward = 0
    
    print("\n--- Evaluation ---")
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=15, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        completion = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Reward: 1.0 if it outputs at least 4 'e's.
        reward = 1.0 if completion.lower().count('e') >= 4 else 0.0
        total_reward += reward
        print(f"Completion: {repr(completion)} | Reward: {reward}")
    
    avg_reward = total_reward / len(prompts)
    print(f"Average Reward: {avg_reward}\n")
    model.train()
    return avg_reward

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "EleutherAI/pythia-14m"
    
    print("Loading models...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    policy_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    ref_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    
    for param in ref_model.parameters():
        param.requires_grad = False
    ref_model.eval()

    # Create dataset
    dummy_data = [
        {"prompt": "Tell me about the history of", "answer": ""},
        {"prompt": "Why is the ocean", "answer": ""},
        {"prompt": "How do you build a", "answer": ""},
        {"prompt": "What are the benefits of", "answer": ""},
        {"prompt": "Can you explain quantum", "answer": ""}
    ] * 4 # 20 training prompts
    
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        for d in dummy_data:
            f.write(json.dumps(d) + "\n")
        tmp_path = f.name
        
    dataset = RLHFDataset(
        dataset_name_or_path=tmp_path,
        tokenizer=tokenizer,
        source="json",
        prompt_column="prompt",
        target_column="answer"
    )

    # Reward function encouraging the letter 'e'
    def reward_fn(prompts, completions, **kwargs):
        return torch.tensor([1.0 if c.lower().count("e") >= 4 else 0.0 for c in completions])

    eval_prompts = [
        "Tell me about the history of",
        "Why is the ocean",
        "How do you build a",
        "What are the benefits of",
        "Can you explain quantum"
    ]
    
    print("=== BEFORE TRAINING ===")
    before_reward = evaluate(policy_model, tokenizer, eval_prompts, device)
    
    config = KTOConfig(
        learning_rate=5e-5, # Moderate LR to avoid collapse but allow learning
        beta=0.1, 
        lambda_d=1.0,
        lambda_u=1.0,
        n_epochs=1,
    )
    
    trainer = KTOTrainer(
        model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset,
        reward_fn=reward_fn,
        config=config,
        reward_threshold=0.5,
        device=device
    )
    
    print("=== TRAINING (30 steps) ===")
    trainer.train(steps=30, batch_size=2, log_interval=10)
    
    print("=== AFTER TRAINING ===")
    after_reward = evaluate(policy_model, tokenizer, eval_prompts, device)
    
    print("=== SUMMARY ===")
    print(f"Before KTO Average Reward: {before_reward}")
    print(f"After KTO Average Reward:  {after_reward}")
    
    if after_reward > before_reward:
        print("Success! The KTO algorithm successfully improved the policy.")
    else:
        print("No improvement observed.")
        
    os.remove(tmp_path)

if __name__ == "__main__":
    main()
