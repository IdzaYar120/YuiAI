import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "bigscience/bloom-560m"

def test_backward():
    print("Testing standard loading...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME) # How it was loaded in train.py

    inputs = tokenizer("User: привіт як справи\nYui: все чудово, Майстре!", return_tensors="pt")
    labels = inputs.input_ids.clone()
    
    outputs = model(**inputs, labels=labels)
    loss = outputs.loss
    print(f"Loss: {loss.item()}")
    
    loss.backward()
    
    has_nans = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                print(f"NaN grad detected in {name}")
                has_nans = True
                break
    
    if not has_nans:
        print("No NaN gradients found with standard loading in a single step.")
    
    print("\nTesting with torch.float32 and use_cache=False...")
    model2 = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    model2.config.use_cache = False
    
    outputs2 = model2(**inputs, labels=labels)
    loss2 = outputs2.loss
    print(f"Loss2: {loss2.item()}")
    
    loss2.backward()
    
    has_nans2 = False
    for name, param in model2.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                print(f"NaN grad detected in {name} with float32")
                has_nans2 = True
                break
                
    if not has_nans2:
        print("No NaN gradients found with float32/no_cache in a single step.")

if __name__ == "__main__":
    test_backward()
