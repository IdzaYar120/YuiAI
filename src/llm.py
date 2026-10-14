"""Локальна модель через Ollama (https://ollama.com) - без зовнішніх AI-сервісів.

Встановлення: ollama pull qwen3:8b
Налаштування через змінні середовища:
    YUI_MODEL    - модель Ollama, за замовчуванням qwen3:8b (Q4, ~5.2 ГБ, влазить у 8 ГБ VRAM)
    OLLAMA_HOST  - адреса сервера Ollama, за замовчуванням http://localhost:11434
    YUI_CTX      - розмір контексту в токенах, за замовчуванням 8192
"""
import json
import os
import urllib.error
import urllib.request

MODEL = os.environ.get("YUI_MODEL", "qwen3:8b")
HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
if not HOST.startswith("http"):
    HOST = "http://" + HOST
NUM_CTX = int(os.environ.get("YUI_CTX", "8192"))

# Рекомендовані Qwen параметри семплінгу для кожного режиму
OPTIONS_FAST = {"temperature": 0.7, "top_p": 0.8, "top_k": 20}
OPTIONS_THINK = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}


class LLMError(RuntimeError):
    pass


def _post(path, payload, timeout=600):
    request = urllib.request.Request(
        f"{HOST}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        return urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")
        if e.code == 404 and "not found" in detail:
            raise LLMError(f"Модель {MODEL} не завантажена. Виконай: ollama pull {MODEL}") from e
        raise LLMError(f"Ollama повернула {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise LLMError(f"Ollama недоступна за адресою {HOST}. Запусти її: ollama serve") from e


def check():
    """Перевіряє, що Ollama запущена і потрібна модель завантажена."""
    try:
        with urllib.request.urlopen(f"{HOST}/api/tags", timeout=5) as response:
            names = [m["name"] for m in json.load(response).get("models", [])]
    except (urllib.error.URLError, OSError) as e:
        raise LLMError(f"Ollama недоступна за адресою {HOST}. Запусти її: ollama serve") from e
    wanted = MODEL if ":" in MODEL else f"{MODEL}:latest"
    if wanted not in names:
        raise LLMError(f"Модель {MODEL} не завантажена. Виконай: ollama pull {MODEL}")


def chat(messages, think=False, on_token=None, on_thinking=None, max_tokens=None):
    """Генерує відповідь. on_token(text) викликається для кожного шматка відповіді (стрімінг),
    on_thinking(text) - для шматків роздумів. Потрібна Ollama 0.9+ (параметр think).

    think=True вмикає режим роздумів Qwen3: повільніше, але точніше для логіки, математики та коду.
    Роздуми не повертаються - лише фінальна відповідь.
    """
    options = dict(OPTIONS_THINK if think else OPTIONS_FAST, num_ctx=NUM_CTX)
    if max_tokens:
        options["num_predict"] = max_tokens
    payload = {"model": MODEL, "messages": messages, "stream": True, "think": think, "options": options}

    parts = []
    with _post("/api/chat", payload) as response:
        for line in response:
            if not line.strip():
                continue
            chunk = json.loads(line)
            if "error" in chunk:
                raise LLMError(chunk["error"])
            message = chunk.get("message", {})
            if message.get("thinking") and on_thinking:
                on_thinking(message["thinking"])
            text = message.get("content", "")
            if text:
                parts.append(text)
                if on_token:
                    on_token(text)
            if chunk.get("done"):
                break
    return "".join(parts).strip()
