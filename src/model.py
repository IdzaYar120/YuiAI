import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Гіперпараметри
MODEL_NAME = "distilgpt2" 

class YuiGPT(nn.Module):
    """Обгортка навколо Pre-trained DistilGPT2"""
    def __init__(self):
        super().__init__()
        print(f"Завантаження {MODEL_NAME} з HuggingFace...")
        self.model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
        self.tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
        
        # Додаємо паддинг-токен, бо GPT-2 його не має
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

    def save_pretrained(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    @classmethod
    def from_pretrained(cls, path):
        instance = cls()
        instance.model = GPT2LMHeadModel.from_pretrained(path)
        instance.tokenizer = GPT2Tokenizer.from_pretrained(path)
        return instance