import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

# Налаштування
# bigscience/bloom-560m - мультимовна модель, яка знає багато мов
MODEL_NAME = "bigscience/bloom-560m" 
TRAIN_FILE = "data/input.txt"
OUTPUT_DIR = "models/yui_bloom"

import shutil
import os

if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)

class LocalTextDataset(Dataset):
    def __init__(self, tokenizer, file_path, block_size):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Tokenize (add eos token)
        self.examples = []
        tokenized_text = tokenizer.encode(text) + [tokenizer.eos_token_id]
        
        # Split into blocks of block_size
        for i in range(0, len(tokenized_text) - block_size + 1, block_size):
            self.examples.append(tokenized_text[i : i + block_size])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)

def main():
    print(f"🚀 Завантаження токенізатора та моделі {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Завантажуємо в чистому float32 і вимикаємо кеш для кращої стабільності під час навчання
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    model.config.use_cache = False

    # Bloom іноді не має pad_token встановленим
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        # model.config.pad_token_id = model.config.eos_token_id # Bloom це зазвичай має

    print("📚 Підготовка даних...")
    # Використовуємо наш власний клас Dataset
    train_dataset = LocalTextDataset(
        tokenizer=tokenizer,
        file_path=TRAIN_FILE,
        block_size=128
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    print("⚙️ Налаштування тренування (Fine-tuning)...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2, # Зменшуємо батч, щоб згладити навантаження
        gradient_accumulation_steps=2, # Емулюємо більший батч (2 * 2 = 4)
        save_steps=200,
        save_total_limit=2,
        prediction_loss_only=True,
        learning_rate=5e-6,           # ДУЖЕ мала швидкість, щоб не злетіти в NaN
        max_grad_norm=0.3,            # Жорстке обрізання градієнтів
        weight_decay=0.01,            # Зв'язує надмірне зростання ваг
        adam_epsilon=1e-6,            # Запобігає діленню на нуль оптимізатора
        warmup_steps=50,              # Плавний старт навчання
        report_to="none",
        fp16=False,                   # ВИМКНУТИ mixed precision на CPU
        bf16=False                    # ВИМКНУТИ bf16
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    print("🏁 Починаємо навчання Brain 3.0!")
    trainer.train()

    print("💾 Збереження моделі...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("✅ Готово! Юї стала розумнішою.")

if __name__ == "__main__":
    main()
