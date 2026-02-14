import torch
import os
import tiktoken
import time
import math
from src.model import YuiGPT, BATCH_SIZE, BLOCK_SIZE

# --- ГІПЕРПАРАМЕТРИ ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LEARNING_RATE = 3e-4
MAX_ITERS = 3000        # Більше ітерацій, бо є scheduler
EVAL_INTERVAL = 100     # Як часто перевіряти on validation set
EVAL_ITERS = 50         # Скільки кроків усереднювати для оцінки

def main():
    print(f"🚀 Запуск ПРОФЕСІЙНОГО навчання на {DEVICE}...")
    
    # 1. Читаємо дані
    if not os.path.exists('data/input.txt'):
        print("❌ Немає data/input.txt! Запусти спочатку setup_data.py")
        return

    with open('data/input.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    # 2. Токенізація
    print("🧠 Кодуємо текст (Tiktoken BPE)...")
    enc = tiktoken.get_encoding("cl100k_base")
    vocab_size = enc.n_vocab
    
    data_ids = enc.encode_ordinary(text)
    data = torch.tensor(data_ids, dtype=torch.long)
    n = int(0.9 * len(data)) # 90% навчання, 10% тест
    train_data = data[:n]
    val_data = data[n:]
    
    print(f"📚 Всього токенів: {len(data)}")
    print(f"🎓 Train set: {len(train_data)} | 🧪 Val set: {len(val_data)}")

    # 3. Батчінг
    def get_batch(split):
        data_source = train_data if split == 'train' else val_data
        ix = torch.randint(len(data_source) - BLOCK_SIZE, (BATCH_SIZE,))
        x = torch.stack([data_source[i:i+BLOCK_SIZE] for i in ix])
        y = torch.stack([data_source[i+1:i+BLOCK_SIZE+1] for i in ix])
        return x.to(DEVICE), y.to(DEVICE)

    # 4. Функція оцінки (без навчання, тільки перевірка)
    @torch.no_grad()
    def estimate_loss(model):
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(EVAL_ITERS)
            for k in range(EVAL_ITERS):
                X, Y = get_batch(split)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    # 5. Ініціалізація
    model = YuiGPT(vocab_size).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Scheduler: плавно зменшує LR до 10% від початкового
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_ITERS, eta_min=LEARNING_RATE/10)
    
    # Спробуємо завантажити чекпоінт, якщо є
    best_val_loss = float('inf')
    if os.path.exists('models/yui_best.pth'):
        print("📥 Знайдено попередній найкращий чекпоінт. Завантажую...")
        try:
            model.load_state_dict(torch.load('models/yui_best.pth', map_location=DEVICE))
            # Оцінимо його
            losses = estimate_loss(model)
            best_val_loss = losses['val']
            print(f"   Поточний best_val_loss: {best_val_loss:.4f}")
        except:
            print("   ⚠️ Помилка завантаження, вчимо з нуля.")

    # 6. Цикл навчання
    print("\n🏁 Поїхали!")
    start_time = time.time()
    
    for iter in range(MAX_ITERS):
        # Оцінка
        if iter % EVAL_INTERVAL == 0:
            losses = estimate_loss(model)
            dt = time.time() - start_time
            print(f"Step {iter}: Train loss {losses['train']:.4f}, Val loss {losses['val']:.4f} [Time: {dt:.1f}s]")
            
            # Збереження найкращої моделі
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                if not os.path.exists('models'): os.makedirs('models')
                torch.save(model.state_dict(), 'models/yui_best.pth')
                print(f"   💾 Збережено нову найкращу модель! (Loss: {best_val_loss:.4f})")

        # Навчальний крок
        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

    # Фінальне збереження
    torch.save(model.state_dict(), 'models/yui_final.pth')
    print("\n🎉 Навчання завершено!")
    print(f"Найкращий Val Loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()
