"""Інструменти, які Юї може викликати (tool calling)."""
from src.devices import DeviceError

DEVICE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "control_devices",
            "description": "Увімкнути, вимкнути або перемкнути пристрої розумного будинку. "
                           "Можна передати кілька пристроїв одразу (напр. все світло в кімнаті).",
            "parameters": {
                "type": "object",
                "properties": {
                    "device_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "id пристроїв зі списку пристроїв",
                    },
                    "action": {"type": "string", "enum": ["on", "off", "toggle"]},
                },
                "required": ["device_ids", "action"],
            },
        },
    },
]

STATE_NAMES = {"on": "увімкнено", "off": "вимкнено"}


def make_executor(manager):
    """Повертає execute(name, arguments) -> str для llm.chat_with_tools."""
    def execute(name, arguments):
        if name != "control_devices":
            return f"Помилка: невідомий інструмент '{name}'"
        device_ids = arguments.get("device_ids") or []
        if isinstance(device_ids, str):
            device_ids = [device_ids]
        action = arguments.get("action", "")
        if not device_ids:
            return "Помилка: не вказано жодного пристрою"

        results = []
        for device_id in device_ids:
            try:
                state = manager.set_state(device_id, action)
                results.append(f"{manager.devices[device_id].name}: {STATE_NAMES[state]}")
            except DeviceError as e:
                results.append(f"{device_id}: помилка - {e}")
        return "; ".join(results)
    return execute


def device_prompt(manager, room=None):
    """Частина системного промпту про будинок - з актуальними станами пристроїв."""
    text = (
        "\n\nТи керуєш розумним будинком Майстра. Пристрої:\n"
        f"{manager.describe()}\n\n"
        "Коли Майстер просить щось увімкнути чи вимкнути - виклич інструмент control_devices. "
        "Не кажи, що щось увімкнула, доки інструмент не підтвердив це. "
        "Якщо незрозуміло, який саме пристрій мається на увазі - перепитай. "
        "Після виконання відповідай коротко, одним реченням."
    )
    if room:
        text += f"\nМайстер зараз у кімнаті: {room}. Якщо кімната не названа - мається на увазі ця."
    return text
