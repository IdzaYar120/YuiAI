import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

from src.data import load_dialogs, dialog_to_messages, YUI_SYSTEM_PROMPT

# Налаштування
# Qwen3-1.7B - мультимовна модель з підтримкою української та готовим чат-форматом
MODEL_NAME = "Qwen/Qwen3-1.7B"
TRAIN_FILE = "data/input.txt"
OUTPUT_DIR = "models/yui_qwen_lora"   # Зберігаються лише LoRA-адаптери (десятки МБ)
MAX_LENGTH = 512

# Порожній блок роздумів - так Qwen3 відповідає в режимі enable_thinking=False
THINK_PREFIX = "<think>\n\n</think>\n\n"

class ChatDataset(Dataset):
    """Діалоги в форматі ChatML Qwen. Лосс рахується лише по відповідях Юї (і <|im_end|>)."""
    def __init__(self, tokenizer, file_path, max_length):
        self.examples = []
        for dialog in load_dialogs(file_path):
            ids, labels = self._encode(tokenizer, dialog_to_messages(dialog))
            self.examples.append((ids[:max_length], labels[:max_length]))

    @staticmethod
    def _encode(tokenizer, messages):
        def tok(text):
            return tokenizer(text, add_special_tokens=False).input_ids

        ids = tok(f"<|im_start|>system\n{YUI_SYSTEM_PROMPT}<|im_end|>\n")
        labels = [-100] * len(ids)
        for msg in messages:
            if msg["role"] == "user":
                part = tok(f"<|im_start|>user\n{msg['content']}<|im_end|>\n")
                ids += part
                labels += [-100] * len(part)
            else:
                prompt = tok(f"<|im_start|>assistant\n{THINK_PREFIX}")
                answer = tok(f"{msg['content']}<|im_end|>")
                ids += prompt + answer
                labels += [-100] * len(prompt) + answer
                newline = tok("\n")
                ids += newline
                labels += [-100] * len(newline)
        return ids, labels

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]

class ChatCollator:
    """Паддинг батчу; паддинг не впливає на лосс."""
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        max_len = max(len(ids) for ids, _ in batch)
        input_ids = torch.full((len(batch), max_len), self.pad_token_id, dtype=torch.long)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, (ids, lbl) in enumerate(batch):
            input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            labels[i, :len(lbl)] = torch.tensor(lbl, dtype=torch.long)
            attention_mask[i, :len(ids)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def main():
    print(f"🚀 Завантаження токенізатора та моделі {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    model.config.use_cache = False

    # LoRA: навчаємо лише невеликі адаптери (~1% параметрів), базова модель заморожена
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()  # потрібно для gradient checkpointing з LoRA
    model.print_trainable_parameters()

    print("📚 Підготовка даних...")
    train_dataset = ChatDataset(
        tokenizer=tokenizer,
        file_path=TRAIN_FILE,
        max_length=MAX_LENGTH
    )
    print(f"   Діалогів: {len(train_dataset)}")

    print("⚙️ Налаштування тренування (LoRA)...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,  # Ефективний батч 2 * 4 = 8
        gradient_checkpointing=True,    # Економія пам'яті ціною швидкості
        learning_rate=2e-4,             # Для LoRA типова швидкість вища, ніж для повного донавчання
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_grad_norm=1.0,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        report_to="none",
        use_cpu=not torch.cuda.is_available(),
        remove_unused_columns=False     # Датасет віддає сирі списки токенів
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=ChatCollator(tokenizer.pad_token_id),
        train_dataset=train_dataset,
    )

    print("🏁 Починаємо навчання Brain 4.0!")
    trainer.train()

    # Зберігаємо поверх старих адаптерів лише після успішного навчання
    print("💾 Збереження LoRA-адаптерів...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("✅ Готово! Юї стала розумнішою.")

if __name__ == "__main__":
    main()
