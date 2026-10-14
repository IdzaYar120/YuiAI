"""Хмарний "Вчитель": Z.ai GLM-Flash (безкоштовно), а при помилці або ліміті - OpenRouter.

Обидва провайдери мають OpenAI-сумісний API, тому працюють через один клієнт `openai`.
Налаштування через змінні середовища:
    ZAI_API_KEY          - ключ з https://z.ai (або open.bigmodel.cn)
    ZAI_BASE_URL         - за замовчуванням https://api.z.ai/api/paas/v4/
                           (для китайського акаунта: https://open.bigmodel.cn/api/paas/v4/)
    ZAI_MODELS           - список через кому, за замовчуванням glm-4.7-flash,glm-4.5-flash
    OPENROUTER_API_KEY   - ключ з https://openrouter.ai (запасний варіант)
    OPENROUTER_MODELS    - список безкоштовних моделей (":free") через кому
"""
import os


def _env_list(name, default):
    return [m.strip() for m in os.environ.get(name, default).split(",") if m.strip()]


PROVIDERS = [
    {
        "name": "Z.ai",
        "api_key": os.environ.get("ZAI_API_KEY", "").strip(),
        "base_url": os.environ.get("ZAI_BASE_URL", "https://api.z.ai/api/paas/v4/"),
        "models": _env_list("ZAI_MODELS", "glm-4.7-flash,glm-4.5-flash"),
        # Вимикаємо режим "роздумів" GLM - швидше і не витрачає ліміт
        "extra_body": {"thinking": {"type": "disabled"}},
    },
    {
        "name": "OpenRouter",
        "api_key": os.environ.get("OPENROUTER_API_KEY", "").strip(),
        "base_url": os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        # Безкоштовні моделі OpenRouter часто змінюються - актуальні див. openrouter.ai/models?q=free
        "models": _env_list("OPENROUTER_MODELS", "z-ai/glm-4.5-air:free,qwen/qwen3-coder:free"),
        "extra_body": {},
    },
]

_clients = {}


def is_configured():
    return any(p["api_key"] for p in PROVIDERS)


def _client(provider):
    if provider["name"] not in _clients:
        from openai import OpenAI
        _clients[provider["name"]] = OpenAI(
            api_key=provider["api_key"], base_url=provider["base_url"], max_retries=0, timeout=60
        )
    return _clients[provider["name"]]


def generate(messages, max_tokens=1024, temperature=0.8):
    """Повертає (текст, "Провайдер/модель"). Пробує всі моделі всіх провайдерів по черзі."""
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    errors = []
    for provider in PROVIDERS:
        if not provider["api_key"]:
            continue
        for model in provider["models"]:
            try:
                response = _client(provider).chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    extra_body=provider["extra_body"] or None,
                )
                text = (response.choices[0].message.content or "").strip()
                if text:
                    return text, f"{provider['name']}/{model}"
                errors.append(f"{provider['name']}/{model}: порожня відповідь")
            except Exception as e:
                errors.append(f"{provider['name']}/{model}: {e}")

    if not errors:
        raise RuntimeError("Не задано жодного ключа: ZAI_API_KEY або OPENROUTER_API_KEY")
    raise RuntimeError("; ".join(errors))
