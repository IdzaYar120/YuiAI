import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F

MODEL_PATH = "models/yui_bloom"
DEVICE = "cpu"

def test_model(path, name):
    print(f"\n--- Testing {name} ({path}) on {DEVICE} ---")
    try:
        tokenizer = AutoTokenizer.from_pretrained(path)
        # Force float32 to rule out precision issues
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).to(DEVICE)
        model.eval()
    except Exception as e:
        print(f"Failed to load {name}: {e}")
        return

    prompt = "User: Привіт\nYui:"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_ids = inputs.input_ids

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
        next_token_logits = logits[0, -1, :]
        
        # Check for NaNs/Infs
        has_nan = torch.isnan(next_token_logits).any().item()
        has_inf = torch.isinf(next_token_logits).any().item()
        
        print(f"Logits shape: {next_token_logits.shape}")
        print(f"Min logit: {next_token_logits.min().item()}")
        print(f"Max logit: {next_token_logits.max().item()}")
        print(f"NaNs in logits: {has_nan}")
        print(f"Infs in logits: {has_inf}")

        if has_nan or has_inf:
            print(f"❌ {name} is CORRUPTED (contains NaNs/Infs).")
        else:
            probs = F.softmax(next_token_logits, dim=-1)
            top_prob, top_id = torch.max(probs, dim=-1)
            print(f"✅ {name} seems OK. Top prediction: '{tokenizer.decode([top_id.item()])}' ({top_prob.item():.4f})")

def check():
    # 1. Test Base Model (Reference)
    test_model("bigscience/bloom-560m", "BASE MODEL")
    
    # 2. Test Fine-Tuned Model (Target)
    test_model("models/yui_bloom", "FINE-TUNED MODEL")

if __name__ == "__main__":
    check()
