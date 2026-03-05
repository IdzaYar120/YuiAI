import torch
import os
import tiktoken
import google.generativeai as genai
from src.model import YuiGPT
# from src.memory import MemorySystem # <--- Імпорт пам'яті

# ================= НАЛАШТУВАННЯ =================
DEVICE = 'cpu' # Примусово використовуємо CPU для стабільності
MODEL_PATH = 'models/yui_best.pth'
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

# ================= КЕШУВАННЯ =================
import json

CACHE_FILE = "teacher_cache.json"

def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

teacher_cache = load_cache()

def ask_teacher(query, history_list):
    # 1. Перевірка кешу
    if query in teacher_cache:
        print(f"⚡ (Знайдено в кеші) Економія API!")
        return teacher_cache[query]

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
        text_response = response.text.strip()
        
        # 2. Збереження в кеш
        teacher_cache[query] = text_response
        save_cache(teacher_cache)
        
        return text_response
    except Exception as e:
        return f"Вибач, Майстре, хмарні нейрони залагали. (T_T) Помилка: {e}"

# ================= ЛОКАЛЬНА ГЕНЕРАЦІЯ (Bloom) =================
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = "models/yui_bloom" 
BASE_MODEL = "bigscience/bloom-560m"

def load_local_model():
    path = MODEL_DIR if os.path.exists(MODEL_DIR) else BASE_MODEL
    print(f"Завантаження мізків ({path})...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(path)
        # Примусово float32 для стабільності на CPU
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).to(DEVICE)
        
        return model, tokenizer
    except Exception as e:
        print(f"Помилка завантаження моделі: {e}")
        return None, None

def generate_local_response(model, tokenizer, user_text, history=[], memory=None):
    if model is None:
        return "Мозок не підключено..."

    try:
        # 1. Пошук у довгостроковій пам'яті (RAG)
        memory_context = ""
        if memory:
            relevant = memory.search(user_text, n_results=1) # Менше контексту для Bloom, щоб не плуталась
            if relevant:
                memory_str = "\n".join([f"- {r}" for r in relevant])
                memory_context = f"Context:\n{memory_str}\n\n"

        # 2. Формуємо контекст
        history_str = ""
        for msg in history[-2:]: # Тільки останні 2 повідомлення (Bloom чутлива до довжини)
            role = "User" if msg['role'] == "User" else "Yui"
            history_str += f"{role}: {msg['content']}\n"

        # Bloom любить чисті промпти
        prompt = f"{memory_context}{history_str}User: {user_text}\nYui:"
        print(f"DEBUG PROMPT: {repr(prompt)}")

        # 3. Кодуємо
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        print(f"DEBUG TOKENS: {inputs.input_ids}")
        
        # 4. Генеруємо
        outputs = model.generate(
            **inputs, 
            max_new_tokens=80, 
            do_sample=True,        # Вмикаємо креативність
            temperature=0.6,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        # 5. Декодуємо
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Вирізаємо тільки нову відповідь
        response = full_text.replace(prompt, "").strip()
        response = response.split("User:")[0].split("\n")[0].strip()
        
        if not response: return "..."
        return response

    except Exception as e:
        return f"Помилка нейромережі: {e}"

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
    # Ініціалізація пам'яті
    # try:
    #     memory = MemorySystem()
    # except Exception as e:
    #     print(f"⚠️ Помилка пам'яті: {e}")
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