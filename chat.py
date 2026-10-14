import os
import sys

from src import llm
from src.data import YUI_SYSTEM_PROMPT

# ================= НАЛАШТУВАННЯ =================
DEBUG = os.environ.get("YUI_DEBUG") == "1"
USE_MEMORY = os.environ.get("YUI_MEMORY") == "1"
HISTORY_MESSAGES = 20  # Скільки останніх повідомлень пам'ятати в розмові

HELP_TEXT = """Команди:
  !think <питання>  - відповідь з роздумами (повільніше, точніше для логіки, математики, коду)
  !think on / off   - увімкнути / вимкнути роздуми для всіх відповідей
  !clear            - забути поточну розмову
  !exit             - вихід"""

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

def generate_response(user_text, history=(), memory=None, think=False):
    # 1. Пошук у довгостроковій пам'яті (RAG)
    memory_context = ""
    if memory:
        relevant = memory.search(user_text, n_results=3)
        if relevant:
            memory_context = "\n".join([f"- {r}" for r in relevant])

    messages = build_messages(user_text, history, memory_context)
    if DEBUG:
        print(f"DEBUG MESSAGES: {messages}")

    # 2. Стрімінг відповіді в термінал
    state = {"thinking_shown": False}
    def on_thinking(_):
        if not state["thinking_shown"]:
            state["thinking_shown"] = True
            print("💭 думаю...", flush=True)
            print("Yui: ", end="", flush=True)
    def on_token(text):
        print(text, end="", flush=True)

    print("\nYui: ", end="", flush=True)
    try:
        response = llm.chat(messages, think=think, on_token=on_token, on_thinking=on_thinking)
    except llm.LLMError as e:
        response = f"Ой, Майстре, мої нейрони не відповідають. (T_T) {e}"
        print(response, end="")
    print()
    return response or "..."

# ================= ГОЛОВНИЙ ЦИКЛ =================
def main():
    try:
        llm.check()
    except llm.LLMError as e:
        print(f"❌ {e}")
        sys.exit(1)

    # Ініціалізація пам'яті (вмикається через YUI_MEMORY=1)
    memory = None
    if USE_MEMORY:
        try:
            from src.memory import MemorySystem
            memory = MemorySystem()
        except Exception as e:
            print(f"⚠️ Помилка пам'яті: {e}")

    chat_history = []
    think_always = False

    print(f"\nЮї в мережі! (модель: {llm.MODEL}, '!help' - команди)")

    while True:
        try:
            user_input = input("\nМайстер: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nДо зустрічі, Майстре!")
            break

        if not user_input: continue
        command = user_input.lower()
        if command in ['!exit', 'exit']:
            print("До зустрічі, Майстре!")
            break
        if command == '!help':
            print(HELP_TEXT)
            continue
        if command == '!clear':
            chat_history = []
            print("🧹 Розмову очищено.")
            continue
        if command in ['!think on', '!think off']:
            think_always = command.endswith("on")
            print(f"💭 Роздуми {'увімкнено' if think_always else 'вимкнено'}.")
            continue

        think = think_always
        if command.startswith("!think"):
            user_input = user_input[len("!think"):].strip()
            if not user_input: continue
            think = True

        response = generate_response(user_input, chat_history, memory, think=think)

        # Зберігаємо в оперативну пам'ять (історію)
        chat_history.append({"role": "User", "content": user_input})
        chat_history.append({"role": "Yui", "content": response})
        chat_history = chat_history[-HISTORY_MESSAGES:]

        # Зберігаємо в довгострокову пам'ять (RAG) - репліку разом з відповіддю
        if memory:
            memory.add(f"User: {user_input}\nYui: {response}", metadata={"source": "chat"})

if __name__ == "__main__":
    main()
