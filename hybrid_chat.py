import torch
import os
import json
import hashlib

# ================= НАЛАШТУВАННЯ =================
DEVICE = 'cpu' # Примусово використовуємо CPU для стабільності
DEBUG = os.environ.get("YUI_DEBUG") == "1"
USE_MEMORY = os.environ.get("YUI_MEMORY") == "1"

# Параметри генерації
MAX_NEW_TOKENS = 256
STOP_MARKERS = ["<END>", "\nUser:"]

# ================= ВЧИТЕЛЬ (Z.ai GLM-Flash -> OpenRouter) =================
from src import teacher
from src.data import YUI_SYSTEM_PROMPT

# ================= КЕШУВАННЯ =================
CACHE_FILE = "teacher_cache.json"

def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}
    return {}

def save_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

teacher_cache = load_cache()

def ask_teacher(query, history_list):
    if not teacher.is_configured():
        return "Майстре, у мене немає ключа до Вчителя. (¬_¬) Задай ZAI_API_KEY або OPENROUTER_API_KEY."

    messages = [{"role": "system", "content": YUI_SYSTEM_PROMPT}]
    for item in history_list:
        role = "user" if item['role'] == "User" else "assistant"
        messages.append({"role": role, "content": item['content']})
    messages.append({"role": "user", "content": query})

    # 1. Перевірка кешу (ключ враховує і запит, і історію, бо відповідь залежить від контексту)
    cache_key = hashlib.sha256(json.dumps(messages, ensure_ascii=False).encode("utf-8")).hexdigest()
    if cache_key in teacher_cache:
        print(f"⚡ (Знайдено в кеші) Економія API!")
        return teacher_cache[cache_key]

    try:
        text_response, source = teacher.generate(messages)
        if DEBUG:
            print(f"DEBUG TEACHER: {source}")
        
        # 2. Збереження в кеш
        teacher_cache[cache_key] = text_response
        save_cache(teacher_cache)
        
        return text_response
    except Exception as e:
        return f"Вибач, Майстре, хмарні нейрони залагали. (T_T) Помилка: {e}"

# ================= ЛОКАЛЬНА ГЕНЕРАЦІЯ (Qwen3 + LoRA) =================
from transformers import AutoTokenizer, AutoModelForCausalLM

ADAPTER_DIR = "models/yui_qwen_lora"
BASE_MODEL = "Qwen/Qwen3-1.7B"
HISTORY_MESSAGES = 6  # Скільки останніх повідомлень передавати моделі

def load_local_model():
    print(f"Завантаження мізків ({BASE_MODEL})...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        # Примусово float32 для стабільності на CPU
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32).to(DEVICE)

        if os.path.exists(ADAPTER_DIR):
            from peft import PeftModel
            print(f"Підключаю LoRA-адаптери Юї ({ADAPTER_DIR})...")
            # Вливаємо адаптери у ваги - генерація швидша, ніж з окремим LoRA-шаром
            model = PeftModel.from_pretrained(model, ADAPTER_DIR).merge_and_unload()
        else:
            print(f"⚠️ Адаптери не знайдено в {ADAPTER_DIR} - Юї без донавчання. Запусти 'python train.py'.")

        model.eval()
        return model, tokenizer
    except Exception as e:
        print(f"Помилка завантаження моделі: {e}")
        return None, None

def extract_reply(generated):
    """Обрізає згенерований текст на кінці репліки Юї, зберігаючи багаторядкові відповіді."""
    generated = generated.replace("<think>", "").replace("</think>", "")
    for marker in STOP_MARKERS:
        generated = generated.split(marker)[0]
    return generated.strip()

def build_messages(user_text, history=(), memory_context=""):
    system = YUI_SYSTEM_PROMPT
    if memory_context:
        system += f"\n\nЩо ти пам'ятаєш з минулих розмов:\n{memory_context}"
    messages = [{"role": "system", "content": system}]
    for msg in history[-HISTORY_MESSAGES:]:
        role = "user" if msg['role'] == "User" else "assistant"
        messages.append({"role": role, "content": msg['content']})
    messages.append({"role": "user", "content": user_text})
    return messages

def generate_local_response(model, tokenizer, user_text, history=(), memory=None):
    if model is None:
        return "Мозок не підключено..."

    try:
        # 1. Пошук у довгостроковій пам'яті (RAG)
        memory_context = ""
        if memory:
            relevant = memory.search(user_text, n_results=2)
            if relevant:
                memory_context = "\n".join([f"- {r}" for r in relevant])

        # 2. Формуємо чат-промпт (режим роздумів Qwen3 вимкнено, як і під час навчання)
        prompt = tokenizer.apply_chat_template(
            build_messages(user_text, history, memory_context),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if DEBUG:
            print(f"DEBUG PROMPT: {repr(prompt)}")

        # 3. Кодуємо
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(DEVICE)

        # 4. Генеруємо (рекомендовані Qwen параметри для режиму без роздумів)
        eos_ids = [i for i in (tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.eos_token_id)
                   if isinstance(i, int) and i != tokenizer.unk_token_id]
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,        # Вмикаємо креативність
                temperature=0.7,
                top_k=20,
                top_p=0.8,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_ids[0],
                eos_token_id=eos_ids
            )

        # 5. Декодуємо лише нові токени (без промпту)
        new_tokens = outputs[0][inputs.input_ids.shape[1]:]
        response = extract_reply(tokenizer.decode(new_tokens, skip_special_tokens=True))

        if not response: return "..."
        return response

    except Exception as e:
        return f"Помилка нейромережі: {e}"

# ================= ГОЛОВНИЙ ЦИКЛ =================
def main():
    local_model, enc = load_local_model()
    if local_model is None:
        return

    # Ініціалізація пам'яті (вмикається через YUI_MEMORY=1)
    memory = None
    if USE_MEMORY:
        try:
            from src.memory import MemorySystem
            memory = MemorySystem()
        except Exception as e:
            print(f"⚠️ Помилка пам'яті: {e}")

    chat_history = []

    print("\nЮї в мережі! (Напиши '!exit' для виходу або '!help' для Вчителя)")

    while True:
        try:
            user_input = input("\nМайстер: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nДо зустрічі, Майстре!")
            break

        if not user_input: continue
        if user_input.lower() in ['!exit', 'exit']:
            print("До зустрічі, Майстре!")
            break

        if user_input.startswith("!help"):
            query = user_input.replace("!help", "", 1).strip() or "Привіт"
            print("Звертаюся до Вчителя (GLM)...")
            response = ask_teacher(query, chat_history)
            prefix = "Yui (Teacher)"
            user_input = query
        else:
            # Передаємо історію та пам'ять в локальну модель
            response = generate_local_response(local_model, enc, user_input, chat_history, memory)
            prefix = "Yui (Local)"

        print(f"\n{prefix}: {response}")

        # Зберігаємо в оперативну пам'ять (історію)
        chat_history.append({"role": "User", "content": user_input})
        chat_history.append({"role": "Yui", "content": response})
        if len(chat_history) > 20: chat_history = chat_history[-20:]

        # Зберігаємо в довгострокову пам'ять (RAG) - репліку разом з відповіддю
        if memory:
            memory.add(f"User: {user_input}\nYui: {response}", metadata={"source": "chat"})

if __name__ == "__main__":
    main()
