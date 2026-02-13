import os
try:
    from src.memory import MemorySystem
except ImportError:
    print("❌ Бібліотеки ще не встановлені.")
    exit()

def ingest_data():
    if not os.path.exists('data/input.txt'):
        print("❌ data/input.txt не знайдено.")
        return

    print("🧠 Ініціалізація пам'яті...")
    mem = MemorySystem()
    
    print("📖 Читаємо data/input.txt...")
    with open('data/input.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    # Розбиваємо на діалоги (припускаємо, що вони розділені порожніми рядками або якось логічно)
    # У нашому випадку це суцільний текст, тому поб'ємо на шматки по 200-300 символів
    chunks = [text[i:i+300] for i in range(0, len(text), 250)]
    
    print(f"📥 Завантажуємо {len(chunks)} спогадів у базу...")
    for i, chunk in enumerate(chunks):
        mem.add(chunk, metadata={"source": "training_data", "chunk_id": i})
        if i % 10 == 0:
            print(f"  Processed {i}/{len(chunks)}")

    print(f"✅ Готово! Всього спогадів у базі: {mem.collection.count()}")

if __name__ == "__main__":
    ingest_data()
