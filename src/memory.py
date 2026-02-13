import chromadb
from sentence_transformers import SentenceTransformer
import os
import uuid

class MemorySystem:
    def __init__(self, db_path="memory_db"):
        print("🧠 Ініціалізація довгострокової пам'яті (RAG)...")
        # Ініціалізація бази даних (зберігається в папці memory_db)
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Завантаження моделі для ембеддінгів (перетворення тексту в цифри)
        # 'all-MiniLM-L6-v2' - швидка і легка модель
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Створення або завантаження колекції
        self.collection = self.client.get_or_create_collection(name="yui_memory")
        print(f"📚 Пам'ять завантажено. Кількість спогадів: {self.collection.count()}")

    def add(self, text, metadata=None):
        """Зберігає текст у пам'ять"""
        if not text or len(text.strip()) < 3: return

        # Генеруємо вектор
        vector = self.embedder.encode(text).tolist()
        
        # Зберігаємо
        self.collection.add(
            documents=[text],
            embeddings=[vector],
            metadatas=[metadata] if metadata else None,
            ids=[str(uuid.uuid4())]
        )

    def search(self, query, n_results=3):
        """Шукає схожі тексти в пам'яті"""
        if self.collection.count() == 0:
            return []

        vector = self.embedder.encode(query).tolist()
        
        results = self.collection.query(
            query_embeddings=[vector],
            n_results=n_results
        )
        
        # results['documents'] це список списків, тому беремо [0]
        return results['documents'][0] if results['documents'] else []
