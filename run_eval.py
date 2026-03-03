import lm_eval
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_eval.models.huggingface import HFLM
import json

print("Loading model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

print("Initializing lm_eval wrapper...")
lm_obj = HFLM(pretrained=model, tokenizer=tokenizer, backend='causal')

print("Starting evaluation...")
results = lm_eval.simple_evaluate(
    model=lm_obj,
    tasks=["hellaswag"],
    limit=5,
    device="cpu"
)

print("\n\n=== RESULTS ===")
print(json.dumps(results["results"], indent=2))
