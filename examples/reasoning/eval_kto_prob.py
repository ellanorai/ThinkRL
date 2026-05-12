import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from thinkrl.algorithms.kto import KTOConfig
from thinkrl.training.kto_trainer import KTOTrainer
from thinkrl.data.datasets import RLHFDataset
import tempfile
import json
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_prob(model, tokenizer, prompt, target_word, device="cuda"):
    model.eval()
    
    # Calculate the probability of the target_word following the prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    target_ids = tokenizer.encode(target_word, add_special_tokens=False, return_tensors="pt").to(device)
    
    with torch.no_grad():
        # Get logits for the next tokens
        outputs = model(input_ids)
        next_token_logits = outputs.logits[0, -1, :]
        probs = F.softmax(next_token_logits, dim=-1)
        
        # We look at the probability of the first token of the target word
        first_target_token = target_ids[0, 0].item()
        prob_target = probs[first_target_token].item()
        
    model.train()
    return prob_target

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

    prompt = "The color of the sky is"
    target_word = " blue"
    
    dummy_data = [
        {"prompt": prompt, "answer": target_word}
    ] * 10 # 10 identical examples to train it to say "blue"
    
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

    # Custom reward function encouraging the model to output the word "blue"
    def reward_fn(prompts, completions, **kwargs):
        return torch.tensor([1.0 if "blue" in c.lower() else 0.0 for c in completions])

    print("\n=== BEFORE TRAINING ===")
    prob_before = evaluate_prob(policy_model, tokenizer, prompt, target_word, device)
    print(f"Probability of '{target_word}' after '{prompt}': {prob_before:.4f}")
    
    config = KTOConfig(
        learning_rate=1e-5, 
        beta=0.1, 
        lambda_d=1.0,
        lambda_u=1.0, 
        n_epochs=5,
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
    
    # Force a hack: During training in this short script, we will force the rollouts to actually contain "blue" 
    # so KTO gets desirable examples. Otherwise random generation never produces "blue".
    original_make_exp = trainer.make_experience
    def hack_make_exp(batch):
        exp = original_make_exp(batch)
        # We need both desirable and undesirable examples in the batch for KTO's KL baseline to work!
        # If all examples are identical, logratio - kl_div = 0, and loss stays at 0.693!
        target_ids_blue = tokenizer.encode(" blue", add_special_tokens=False, return_tensors="pt").to(device)
        target_ids_red = tokenizer.encode(" red", add_special_tokens=False, return_tensors="pt").to(device)
        
        for i in range(len(exp["generated_ids"])):
            if i % 2 == 0:
                exp["generated_ids"][i][0] = target_ids_blue[0][0]
            else:
                exp["generated_ids"][i][0] = target_ids_red[0][0]
            exp["generated_ids"][i][1:] = tokenizer.eos_token_id
            
        exp["completions_text"] = tokenizer.batch_decode(exp["generated_ids"], skip_special_tokens=True)
        return exp
    trainer.make_experience = hack_make_exp

    print("\n=== TRAINING ===")
    trainer.train(steps=30, batch_size=2, log_interval=5)
    
    print("\n=== AFTER TRAINING ===")
    prob_after = evaluate_prob(policy_model, tokenizer, prompt, target_word, device)
    print(f"Probability of '{target_word}' after '{prompt}': {prob_after:.4f}")
    
    print("\n=== SUMMARY ===")
    print(f"Before KTO Prob: {prob_before:.4f}")
    print(f"After KTO Prob:  {prob_after:.4f}")
    
    if prob_after > prob_before:
        print("Success! The KTO algorithm successfully increased the target probability.")
    
    os.remove(tmp_path)

if __name__ == "__main__":
    main()
