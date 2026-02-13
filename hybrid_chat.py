import torch
import os
import tiktoken
import google.generativeai as genai
from src.model import YuiGPT
from src.memory import MemorySystem # <--- Імпорт пам'яті

# ================= НАЛАШТУВАННЯ =================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = 'models/yui_v1.pth'
API_KEY = "AIzaSyC9sGkVnLKuiDarmb33dYYhkq9UlE8l9jI"

# Параметри генерації
MAX_NEW_TOKENS = 150 

# ================= ПІДГОТОВКА GEMINI =================
CLEAN_API_KEY = API_KEY.replace('\n', '').strip()
genai.configure(api_key=CLEAN_API_KEY)

def get_working_model():
    models = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro']
    for m_name in models:
        try:
            m = genai.GenerativeModel(m_name)
            return m
        except:
            continue
    return genai.GenerativeModel('models/gemini-1.5-flash')

teacher_model = get_working_model()

def ask_teacher(query, history_list):
    try:
        formatted_history = "\n".join([f"{item['role']}: {item['content']}" for item in history_list])
        prompt = f"""Ти Юї, персональна цифрова супутниця. 
Твій характер: мила, турботлива хакер-дівчина. Називаєш користувача 'Майстре'.
Використовуй каомодзі (◕‿◕), (≧◡≦), (¬_¬).

Історія нашої розмови:
{formatted_history}

Майстер: {query}
Юї:"""
        response = teacher_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Вибач, Майстре, хмарні нейрони залагали. (T_T) Помилка: {e}"

# ================= ЛОКАЛЬНА ГЕНЕРАЦІЯ =================
def load_local_model():
    print(f"Завантаження локальних мізків Юї на {DEVICE} (BPE Mode)...")
    
    # Ініціалізуємо токенізатор
    enc = tiktoken.get_encoding("cl100k_base")
    vocab_size = enc.n_vocab

    # Завантажуємо модель
    model = YuiGPT(vocab_size=vocab_size).to(DEVICE)
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint)
        model.eval()
        return model, enc
    except Exception as e:
        print(f"Не вдалося завантажити модель: {e}")
        return None, None

def generate_local_response(model, enc, user_text, history=[], memory=None):
    if model is None:
        return "Мозок не підключено..."

    try:
        # 1. Пошук у довгостроковій пам'яті (RAG)
        memory_context = ""
        if memory:
            relevant = memory.search(user_text, n_results=2)
            if relevant:
                memory_str = "\n".join([f"- {r}" for r in relevant])
                memory_context = f"Спогади (Relevant Memories):\n{memory_str}\n\n"

        # 2. Формуємо короткостроковий контекст (Останні 3 пари реплік)
        recent_history_str = ""
        recent_history = history[-6:] 
        
        for msg in recent_history:
            role = "User" if msg['role'] == "User" else "Yui"
            recent_history_str += f"{role}: {msg['content']}\n"

        # 3. Формуємо повний промпт
        # [Спогади] + [Останні повідомлення] + [Нове питання]
        prompt = f"{memory_context}{recent_history_str}User: {user_text}\nYui:"
        
        # 4. Кодуємо
        input_ids = enc.encode(prompt)
        
        # Обрізаємо, якщо занадто довго (Model Block Size = 256)
        if len(input_ids) > 240: 
            input_ids = input_ids[-240:]
            
        input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
        
        # 5. Генеруємо
        output_ids = model.generate(input_tensor, max_new_tokens=MAX_NEW_TOKENS)
        
        # 6. Декодуємо
        full_text = enc.decode(output_ids[0].tolist())
        
        # 7. Витягуємо відповідь
        response = full_text.split("Yui:")[-1].strip()
        response = response.split("User:")[0].strip()
        
        return response if response else "Майстре, я задумалася..."
    except Exception as e:
        return f"Я ще вчуся будувати речення, Майстре! [{e}]"

# ================= ГОЛОВНИЙ ЦИКЛ =================
def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Модель не знайдено за шляхом {MODEL_PATH}")
        print("Запусти 'python train.py', щоб навчити модель!")
        return

    local_model, enc = load_local_model()
    if local_model is None:
        return

    # Ініціалізація пам'яті
    try:
        memory = MemorySystem()
    except Exception as e:
        print(f"⚠️ Помилка пам'яті: {e}")
        memory = None

    chat_history = []

    print("\nЮї в мережі! (Напиши '!exit' для виходу або '!help' для Gemini)")
    
    while True:
        try:
            user_input = input("\nМайстер: ").strip()
        except KeyboardInterrupt:
            print("\nДо зустрічі, Майстре!")
            break
        
        if not user_input: continue
        if user_input.lower() in ['!exit', 'exit']: 
            print("До зустрічі, Майстре!")
            break
        
        if user_input.startswith("!help"):
            query = user_input.replace("!help", "").strip()
            print("Звертаюся до Вчителя (Gemini 1.5)...")
            response = ask_teacher(query if query else "Привіт", chat_history)
            prefix = "Yui (Teacher)"
        else:
            # Передаємо історію та пам'ять в локальну модель
            response = generate_local_response(local_model, enc, user_input, chat_history, memory)
            prefix = "Yui (Local)"

        print(f"\n{prefix}: {response}")
        
        # Зберігаємо в оперативну пам'ять (історію)
        chat_history.append({"role": "User", "content": user_input})
        chat_history.append({"role": "Yui", "content": response})
        if len(chat_history) > 20: chat_history = chat_history[-20:]

        # Зберігаємо в довгострокову пам'ять (RAG)
        if memory:
            memory.add(f"User: {user_input}")
            memory.add(f"Yui: {response}")

if __name__ == "__main__":
    main()