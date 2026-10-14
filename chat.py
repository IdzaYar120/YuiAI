import argparse
import os
import sys

from src import llm
from src.data import YUI_SYSTEM_PROMPT
from src.devices import DeviceManager, DeviceError
from src.tools import DEVICE_TOOLS, make_executor, device_prompt

# ================= НАЛАШТУВАННЯ =================
DEBUG = os.environ.get("YUI_DEBUG") == "1"
USE_MEMORY = os.environ.get("YUI_MEMORY") == "1"
HISTORY_MESSAGES = 20  # Скільки останніх повідомлень пам'ятати в розмові
DEVICES_CONFIG = "config/devices.toml"

HELP_TEXT = """Команди:
  !think <питання>  - відповідь з роздумами (повільніше, точніше для логіки, математики, коду)
  !think on / off   - увімкнути / вимкнути роздуми для всіх відповідей
  !devices          - стан пристроїв
  !clear            - забути поточну розмову
  !exit             - вихід"""

def build_messages(user_text, history=(), memory_context="", devices=None, room=None):
    system = YUI_SYSTEM_PROMPT
    if devices:
        system += device_prompt(devices, room)
    if memory_context:
        system += f"\n\nЩо ти пам'ятаєш з минулих розмов:\n{memory_context}"
    messages = [{"role": "system", "content": system}]
    for msg in history[-HISTORY_MESSAGES:]:
        role = "user" if msg['role'] == "User" else "assistant"
        messages.append({"role": role, "content": msg['content']})
    messages.append({"role": "user", "content": user_text})
    return messages

def generate_response(user_text, history=(), memory=None, think=False, devices=None, room=None):
    # 1. Пошук у довгостроковій пам'яті (RAG)
    memory_context = ""
    if memory:
        relevant = memory.search(user_text, n_results=3)
        if relevant:
            memory_context = "\n".join([f"- {r}" for r in relevant])

    messages = build_messages(user_text, history, memory_context, devices, room)
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
    def on_tool(name, arguments, result):
        print(f"🔌 {result}\nYui: ", end="", flush=True)

    print("\nYui: ", end="", flush=True)
    try:
        if devices:
            response = llm.chat_with_tools(messages, DEVICE_TOOLS, make_executor(devices), think=think,
                                           on_token=on_token, on_thinking=on_thinking, on_tool=on_tool)
        else:
            response = llm.chat(messages, think=think, on_token=on_token, on_thinking=on_thinking)
    except llm.LLMError as e:
        response = f"Ой, Майстре, мої нейрони не відповідають. (T_T) {e}"
        print(response, end="")
    print()
    return response or "..."

# ================= ГОЛОВНИЙ ЦИКЛ =================
def load_devices(path, simulate):
    if simulate and not os.path.exists(path):
        path = "config/devices.example.toml"  # Для симуляції достатньо прикладу
    if not os.path.exists(path):
        print(f"ℹ️ {path} не знайдено - розумний будинок вимкнено (приклад: config/devices.example.toml)")
        return None
    try:
        manager = DeviceManager.from_config(path, simulate=simulate)
        manager.start()
    except (DeviceError, OSError, ValueError, KeyError) as e:
        print(f"⚠️ Пристрої недоступні: {e}")
        return None
    mode = "симуляція" if simulate else "MQTT"
    print(f"🏠 Пристроїв: {len(manager.devices)} ({mode})")
    return manager

def main():
    parser = argparse.ArgumentParser(description="Юї - локальна AI-асистентка")
    parser.add_argument("--devices", default=DEVICES_CONFIG, help="конфіг пристроїв (TOML)")
    parser.add_argument("--simulate", action="store_true", help="імітувати пристрої без MQTT і заліза")
    parser.add_argument("--room", help="кімната, в якій ви зараз (для команд без назви кімнати)")
    args = parser.parse_args()

    try:
        llm.check()
    except llm.LLMError as e:
        print(f"❌ {e}")
        sys.exit(1)

    devices = load_devices(args.devices, args.simulate)

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
        if command == '!devices':
            print(devices.describe() if devices else "Пристрої не підключені.")
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

        response = generate_response(user_input, chat_history, memory, think=think, devices=devices, room=args.room)

        # Зберігаємо в оперативну пам'ять (історію)
        chat_history.append({"role": "User", "content": user_input})
        chat_history.append({"role": "Yui", "content": response})
        chat_history = chat_history[-HISTORY_MESSAGES:]

        # Зберігаємо в довгострокову пам'ять (RAG) - репліку разом з відповіддю
        if memory:
            memory.add(f"User: {user_input}\nYui: {response}", metadata={"source": "chat"})

if __name__ == "__main__":
    main()
