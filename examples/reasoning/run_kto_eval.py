import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from thinkrl.algorithms.kto import KTOConfig
from thinkrl.training.kto_trainer import KTOTrainer
from thinkrl.data.datasets import RLHFDataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate(model, tokenizer, prompts, device="cuda"):
    model.eval()
    total_reward = 0
    
    print("\n--- Evaluation ---")
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=15, do_sample=False)
        completion = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        # Simple dummy reward: +1 if completion is longer than 20 chars
        # (This is just to demonstrate that the optimizer learns to increase the reward)
        reward = 1.0 if len(completion) > 20 else 0.0
        total_reward += reward
        print(f"Prompt: {prompt}")
        print(f"Completion: {completion}")
        print(f"Reward: {reward}")
    
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
    
    # Freeze ref model
    for param in ref_model.parameters():
        param.requires_grad = False
    ref_model.eval()

    # Create dummy dataset
    dummy_data = [
        {"prompt": "Tell me a story about", "answer": ""},
        {"prompt": "What is the meaning of", "answer": ""},
        {"prompt": "How do you make a", "answer": ""},
        {"prompt": "Once upon a time", "answer": ""},
        {"prompt": "In a galaxy far far", "answer": ""}
    ]
    
    import tempfile
    import json
    import os
    
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

    # Custom reward function encouraging the model to output a completion longer than 20 chars
    def reward_fn(prompts, completions, **kwargs):
        return torch.tensor([1.0 if len(c) > 20 else 0.0 for c in completions])

    eval_prompts = [d["prompt"] for d in dummy_data]
    
    print("=== BEFORE TRAINING ===")
    before_reward = evaluate(policy_model, tokenizer, eval_prompts, device)
    
    config = KTOConfig(
        learning_rate=1e-3, # Use a higher LR again to ensure it moves quickly
        beta=0.01,
        lambda_d=1.0,
        lambda_u=1.0,
        n_epochs=2,
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
    
    print("=== TRAINING ===")
    trainer.train(steps=15, batch_size=2, log_interval=5)
    
    print("=== AFTER TRAINING ===")
    after_reward = evaluate(policy_model, tokenizer, eval_prompts, device)
    
    print("=== SUMMARY ===")
    print(f"Before KTO Average Reward: {before_reward}")
    print(f"After KTO Average Reward:  {after_reward}")
    
    if after_reward > before_reward:
        print("Success! The KTO algorithm successfully improved the policy.")
    else:
        print("No improvement observed in this short run, but the pipeline executed successfully.")
        
    os.remove(tmp_path)

if __name__ == "__main__":
    main()
