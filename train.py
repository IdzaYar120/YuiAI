import torch
import pickle
import os
import tiktoken # <--- Нова бібліотека для токенів
from src.model import YuiGPT, BATCH_SIZE, BLOCK_SIZE

# Налаштування
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LEARNING_RATE = 3e-4
MAX_ITERS = 2000 

def main():
    print(f"Запуск навчання на {DEVICE} (BPE Mode)...")
    
    # 1. Читаємо дані
    if not os.path.exists('data/input.txt'):
        print("❌ Немає файлу data/input.txt! Запусти спочатку setup_data.py")
        return

    with open('data/input.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    # 2. Токенізація через Tiktoken (BPE)
    print("Кодуємо текст через tiktoken (cl100k_base)...")
    enc = tiktoken.get_encoding("cl100k_base")
    vocab_size = enc.n_vocab
    print(f"Розмір словника: {vocab_size} токенів (раніше було ~150)")

    # 3. Готуємо тензори
    # encode_ordinary швидше, але ігнорує спец-токени (нам ок для простого тексту)
    data_ids = enc.encode_ordinary(text)
    data = torch.tensor(data_ids, dtype=torch.long)
    print(f"Всього токенів у датасеті: {len(data)}")
    
    def get_batch():
        ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
        x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
        y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
        return x.to(DEVICE), y.to(DEVICE)

    # 4. Ініціалізація
    model = YuiGPT(vocab_size).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # 5. Цикл навчання
    print("Поїхали!")
    for iter in range(MAX_ITERS):
        xb, yb = get_batch()
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if iter % 100 == 0:
            print(f"Крок {iter}: втрати {loss.item():.4f}")

    # 6. Збереження
    if not os.path.exists('models'): os.makedirs('models')
    torch.save(model.state_dict(), 'models/yui_v1.pth')
    print("YuiAI (BPE версія) успішно збережена!")

if __name__ == "__main__":
    main()
