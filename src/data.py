import re

DIALOG_SEPARATOR = "<END>"

# "7.  User: ..." / "**User:** ..." / "Юї: ..." -> "User: ..." / "Yui: ..."
_SPEAKER_RE = re.compile(r"^\s*(?:\d+\.\s+)?\**\s*(User|Yui|Юї)\s*\**\s*:\s*\**\s*", re.IGNORECASE)
# Заголовки та розділювачі, які Gemini додає між діалогами: "**Діалог 3**", "5.  **Тема:** ...", "---"
_HEADING_RE = re.compile(r"^\s*(?:-{3,}|#+\s.*|(?:\d+\.\s+)?\*\*.*)\s*$")


def _normalize_speaker(line):
    m = _SPEAKER_RE.match(line)
    if not m:
        return None
    speaker = "User" if m.group(1).lower() == "user" else "Yui"
    content = line[m.end():].strip()
    if content.endswith("**"):
        content = content[:-2].rstrip()
    return f"{speaker}: {content}"


def parse_dialogs(text):
    """Розбирає сирий датасет на чисті діалоги без markdown-заголовків і дублікатів."""
    dialogs = []
    seen = set()
    for block in text.split(DIALOG_SEPARATOR):
        lines = []
        for raw in block.splitlines():
            if not raw.strip():
                continue
            line = _normalize_speaker(raw)
            if line is None:
                # Все до першої репліки User — це преамбула Gemini ("Ось 20 діалогів...")
                if not lines or _HEADING_RE.match(raw):
                    continue
                line = raw.strip()  # продовження попередньої репліки
            elif not lines and not line.startswith("User:"):
                continue
            lines.append(line)

        dialog = "\n".join(lines)
        if "\nYui:" not in dialog or dialog in seen:
            continue
        seen.add(dialog)
        dialogs.append(dialog)
    return dialogs


def load_dialogs(path):
    with open(path, "r", encoding="utf-8") as f:
        return parse_dialogs(f.read())


YUI_SYSTEM_PROMPT = """Ти Юї, персональна цифрова супутниця. 
Твій характер: мила, турботлива хакер-дівчина. Називаєш користувача 'Майстре'.
Використовуй каомодзі (◕‿◕), (≧◡≦), (¬_¬). Відповідай українською."""


def dialog_to_messages(dialog):
    """'User: ...\\nYui: ...' -> [{"role": "user", ...}, {"role": "assistant", ...}] для chat template."""
    messages = []
    for line in dialog.splitlines():
        if line.startswith("User: "):
            messages.append({"role": "user", "content": line[len("User: "):]})
        elif line.startswith("Yui: "):
            messages.append({"role": "assistant", "content": line[len("Yui: "):]})
        elif messages:
            messages[-1]["content"] += "\n" + line  # продовження репліки (напр. думка + відповідь)
    return messages
