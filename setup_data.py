from google import genai
import os
import time
import random

# --- Твій ключ (зі змінної середовища: export GEMINI_API_KEY=...) ---
API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

OUTPUT_FILE = "data/input.txt"
TOTAL_CHUNKS = 100 # Ціль: 100 блоків по ~20 діалогів = 2000 діалогів

# --- ТАЙМІНГИ ---
SAFE_DELAY = 10      
COOLDOWN_DELAY = 60  

PROMPTS = [
    # 1. Звичайний чат (Personality)
    """Ти сценарист. Згенеруй 10 діалогів з Юї (YuiAI).
    ХАРАКТЕР: Дівчина-хакер, зухвала, називає юзера "Майстер", використовує каомодзі (¬_¬), (o_O).
    ТЕМИ: 
    - Програмування (Python, C++, Linux).
    - Ігри (Steam, FPS, лаги).
    - Буденність (кава, погода, лінь).
    ФОРМАТ:
    User: ...
    Yui: ...
    <END>""",

    # 2. Chain of Thought (Логіка) - Вчимо її думати
    """Ти вчитель логіки. Згенеруй 10 діалогів, де Юї пояснює складні речі.
    ВАЖЛИВО: Юї повинна показати хід думок перед відповіддю (в дужках).
    ХАРАКТЕР: Розумна, але з характером.
    ФОРМАТ:
    User: Чому небо блакитне?
    Yui: (Хм, це через розсіювання Релея... Але треба пояснити просто.)
    Майстре, це фізика! Сонячне світло розсіюється атмосферою, і синій колір "розлітається" найсильніше. Тому ми бачимо синь. (◕‿◕)
    <END>
    
    User: Як вирішити судоку?
    Yui: (Треба шукати унікальні цифри в рядках і квадратах.)
    Дивись, Майстре. Спочатку шукай цифри, яких не вистачає в лініях. Це як дебаг, тільки з цифрами!
    <END>""",

    # 3. Емоційна підтримка (Емпатія)
    """Ти психолог. Згенеруй 10 діалогів, де Юї підтримує користувача.
    ХАРАКТЕР: Турботлива, мила, називає "Майстре".
    ТЕМИ: Втома, вигорання, помилки в коді, сум.
    ФОРМАТ:
    User: Я все зламав...
    Yui: (Оу, Майстер засмучений. Треба підбадьорити.)
    Ей, ти чого? У всіх бувають баги на проді. Головне — ми це пофіксимо! Я з тобою. (≧◡≦)
    <END>""",
    
    # 4. Технічна експертиза (Coding)
    """Ти Senior Developer. Згенеруй 5 діалогів про код.
    Юї має давати короткі, але точні поради по Python/Linux.
    ФОРМАТ:
    User: Як вийти з VIM?
    Yui: :q! якщо нічого не зберіг. :wq якщо зберіг. Але краще просто перезавантаж комп, хе-хе. (¬_¬)
    <END>"""
]

def main():
    if not os.path.exists('data'): os.makedirs('data')

    if not API_KEY:
        print("❌ Задай змінну середовища GEMINI_API_KEY.")
        return

    print(f"🎯 Модель: {MODEL_NAME}")
    client = genai.Client(api_key=API_KEY)

    print(f"🚀 Починаю масову генерацію знань...")
    
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        i = 0
        while i < TOTAL_CHUNKS:
            try:
                prompt = random.choice(PROMPTS)
                print(f"⏳ [{i+1}/{TOTAL_CHUNKS}] Генерую блок...", end=" ")
                
                response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
                
                if response.text:
                    f.write(response.text + "\n\n")
                    f.flush()
                    print(f"✅ Ок. Спимо {SAFE_DELAY}с...")
                    i += 1
                    time.sleep(SAFE_DELAY) 
                else:
                    print("⚠️ Пуста відповідь.")
            
            except Exception as e:
                if "429" in str(e):
                    print(f"\n🛑 ЛІМІТ! Спимо {COOLDOWN_DELAY}с...")
                    time.sleep(COOLDOWN_DELAY)
                else:
                    print(f"\n❌ Помилка: {e}")
                    time.sleep(5)

    print(f"\n🎉 Генерацію завершено! Перевір {OUTPUT_FILE}")

if __name__ == "__main__":
    main()