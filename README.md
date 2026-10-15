# YuiAI

Юї - локальна голосова AI-асистентка для розумного будинку. Працює повністю на домашньому сервері,
без зовнішніх AI-сервісів: мозок - Qwen3-8B через Ollama, пристрої - через MQTT (ESPHome, Shelly).

```
 Майстер: Юї, включи чайник
 🔌 Чайник: увімкнено
 Yui: Вже гріється, Майстре! (◕‿◕)
```

## Статус

- [x] Етап 1 - керування пристроями текстом (MQTT, Shelly, ESPHome, автовимкнення)
- [ ] Етап 2 - голос на сервері (wake word «Юї», Whisper, Piper)
- [ ] Етап 3 - сервер як сервіс (systemd, API)
- [ ] Етап 4 - голосові ESP32-S3-сателіти по кімнатах

## Сервер (Linux)

### 1. Ollama і модель
```bash
# Спершу драйвер NVIDIA (для GTX 1070 підходить будь-який актуальний), перевірка: nvidia-smi
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3:8b          # ~5.2 ГБ, повністю влазить у 8 ГБ VRAM
```

### 2. MQTT-брокер Mosquitto
```bash
sudo apt install mosquitto mosquitto-clients
sudo mosquitto_passwd -c /etc/mosquitto/passwd yui
sudo tee /etc/mosquitto/conf.d/yui.conf <<'EOF'
listener 1883
allow_anonymous false
password_file /etc/mosquitto/passwd
EOF
sudo systemctl restart mosquitto

# Подивитися весь трафік Юї:
mosquitto_sub -h localhost -u yui -P <пароль> -t '#' -v
```

### 3. Юї
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install paho-mqtt
cp config/devices.example.toml config/devices.toml   # і заповнити під свою квартиру
python chat.py --room Кухня
```

Без заліза можна спробувати в режимі симуляції: `python chat.py --simulate`.

Команди в чаті: `!devices` - стан пристроїв, `!think <питання>` - відповідь з роздумами,
`!clear` - очистити розмову, `!help` - усі команди.

## Пристрої

Усі пристрої описуються в `config/devices.toml` (приклад - `config/devices.example.toml`).
Юї бачить список пристроїв з кімнатами та станами і сама вирішує, що увімкнути.

### Розетки Shelly (найпростіше і найбезпечніше)
Shelly Plus Plug S / Gen3 мають MQTT вбудований. У веб-інтерфейсі Shelly:
**Settings → MQTT** - увімкнути, вказати IP сервера, логін/пароль Mosquitto,
увімкнути *RPC status notifications* і *Generic status update over MQTT*.
Для чайника обов'язково задай **Auto off** у налаштуваннях розетки - це спрацює навіть без сервера.

### ESP32 + реле (ESPHome)
```bash
pip install esphome
cd firmware/esphome
cp secrets.example.yaml secrets.yaml      # Wi-Fi, IP сервера, паролі
# У relay.yaml змінити substitutions: device_id, friendly_name, relay_pin, auto_off_ms
esphome run relay.yaml                    # Перший раз - через USB, далі по Wi-Fi
```
⚠️ Чайник споживає ~9 А - беріть реле на 16 А+ або готову розумну розетку.
Робота з 230 В небезпечна, якщо немає досвіду - використовуйте Shelly.

## Тести
```bash
python -m unittest discover -s tests -t .
```
Інтеграційний тест зі справжнім MQTT-брокером запускається, якщо встановлені `paho-mqtt` і `amqtt`.

## Налаштування (змінні середовища)

| Змінна | За замовчуванням | Що робить |
|---|---|---|
| `YUI_MODEL` | `qwen3:8b` | Модель Ollama |
| `OLLAMA_HOST` | `http://localhost:11434` | Адреса Ollama |
| `YUI_CTX` | `8192` | Розмір контексту |
| `YUI_MEMORY` | - | `1` вмикає довгострокову пам'ять (ChromaDB) |
| `YUI_DEBUG` | - | `1` показує повний промпт |
