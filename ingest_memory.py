import os
try:
    from src.memory import MemorySystem
    from src.data import load_dialogs
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
    # Кожен спогад - один цілий діалог, а не випадковий шматок тексту
    chunks = load_dialogs('data/input.txt')
    
    print(f"📥 Завантажуємо {len(chunks)} спогадів у базу...")
    for i, chunk in enumerate(chunks):
        mem.add(chunk, metadata={"source": "training_data", "chunk_id": i})
        if i % 10 == 0:
            print(f"  Processed {i}/{len(chunks)}")

    print(f"✅ Готово! Всього спогадів у базі: {mem.collection.count()}")

if __name__ == "__main__":
    ingest_data()
