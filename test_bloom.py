import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

MODEL_PATH = "models/yui_bloom"
BASE_MODEL = "bigscience/bloom-560m"

def test():
    path = MODEL_PATH if os.path.exists(MODEL_PATH) else BASE_MODEL
    print(f"Loading model from {path} on CPU...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path).to("cpu") 
        print(f"Model loaded on CPU.")
        
        print(f"Vocab size (tokenizer): {tokenizer.vocab_size}")
        print(f"Vocab size (model): {model.config.vocab_size}")
        print(f"Embeddings size: {model.get_input_embeddings().weight.shape}")
        
        prompt = "User: привіт\nYui:"
        print(f"\nTesting generation with prompt: {repr(prompt)}")
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
        print(f"Input IDs: {inputs.input_ids}")
        
        outputs = model.generate(
            **inputs, 
            max_new_tokens=60, 
            do_sample=False,       # Greedy decoding match
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        print(f"\nOutput IDs: {outputs[0]}")
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated text: {text}")
        print("\n✅ Success! CPU generation works.")
        
    except Exception as e:
        print(f"\n❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test()
