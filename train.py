import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Trainer, TrainingArguments

from src.data import load_dialogs

# Налаштування
# bigscience/bloom-560m - мультимовна модель, яка знає багато мов
MODEL_NAME = "bigscience/bloom-560m"
TRAIN_FILE = "data/input.txt"
OUTPUT_DIR = "models/yui_bloom"
MAX_LENGTH = 256

class DialogDataset(Dataset):
    """Кожен діалог - окремий приклад, що закінчується EOS, щоб модель вчилась зупинятися."""
    def __init__(self, tokenizer, file_path, max_length):
        self.examples = []
        for dialog in load_dialogs(file_path):
            ids = tokenizer.encode(dialog)[:max_length - 1] + [tokenizer.eos_token_id]
            self.examples.append(ids)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]

class DialogCollator:
    """Паддинг батчу. Лосс рахується по всіх реальних токенах (включно з EOS), але не по паддингу."""
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        max_len = max(len(ids) for ids in batch)
        input_ids = torch.full((len(batch), max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, ids in enumerate(batch):
            input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, :len(ids)] = 1
        labels = input_ids.masked_fill(attention_mask == 0, -100)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def main():
    print(f"🚀 Завантаження токенізатора та моделі {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Завантажуємо в чистому float32 і вимикаємо кеш для кращої стабільності під час навчання
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    model.config.use_cache = False

    # Bloom іноді не має pad_token встановленим
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("📚 Підготовка даних...")
    train_dataset = DialogDataset(
        tokenizer=tokenizer,
        file_path=TRAIN_FILE,
        max_length=MAX_LENGTH
    )
    print(f"   Діалогів: {len(train_dataset)}")

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
        bf16=False,                   # ВИМКНУТИ bf16
        remove_unused_columns=False   # Датасет віддає сирі списки токенів
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=DialogCollator(tokenizer.pad_token_id),
        train_dataset=train_dataset,
    )

    print("🏁 Починаємо навчання Brain 3.0!")
    trainer.train()

    # Зберігаємо поверх старої моделі лише після успішного навчання
    print("💾 Збереження моделі...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("✅ Готово! Юї стала розумнішою.")

if __name__ == "__main__":
    main()
