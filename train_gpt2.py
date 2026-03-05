import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

# Налаштування
MODEL_NAME = "distilgpt2"
TRAIN_FILE = "data/input.txt"
OUTPUT_DIR = "models/yui_distilgpt2"

def main():
    print(f"🚀 Завантаження токенізатора та моделі {MODEL_NAME}...")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

    # GPT-2 не має padding token, додаємо його
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    print("📚 Підготовка даних...")
    # TextDataset автоматично ріже текст на блоки
    train_dataset = TextDataset(
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
        overwrite_output_dir=True,
        num_train_epochs=5,           # Кількість проходів по даним
        per_device_train_batch_size=8, 
        save_steps=500,
        save_total_limit=2,
        prediction_loss_only=True,
        learning_rate=5e-5,           # Маленька швидкість для точного налаштування
        report_to="none"
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
