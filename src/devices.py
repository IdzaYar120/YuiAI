"""Пристрої розумного будинку через MQTT.

Опис пристроїв - у config/devices.toml (приклад: config/devices.example.toml).
Підтримувані драйвери:
    shelly - розетки/реле Shelly Gen2+ (MQTT вбудований, перепрошивка не потрібна)
    mqtt   - будь-яка прошивка з командним топіком і ON/OFF (напр. ESPHome, Tasmota)
"""
import json
import threading
import time
import tomllib

ON, OFF = "on", "off"
RPC_REPLY_TOPIC = "yui/rpc"  # Сюди Shelly відповідає на RPC-запити з "src": "yui"


class DeviceError(Exception):
    pass


class Device:
    def __init__(self, cfg):
        try:
            self.id = cfg["id"]
            self.name = cfg["name"]
        except KeyError as e:
            raise DeviceError(f"У пристрою немає обов'язкового поля {e}") from e
        self.room = cfg.get("room", "")
        self.aliases = cfg.get("aliases", [])
        self.driver = cfg.get("driver", "mqtt")
        self.auto_off_minutes = cfg.get("auto_off_minutes")
        self.state = None  # None - стан ще невідомий (пристрій не відповідав)
        self.last_seen = None

        if self.driver == "shelly":
            # Shelly Gen2+: RPC через <prefix>/rpc, стан приходить у <prefix>/status/switch:<n>
            self.shelly_prefix = cfg["topic_prefix"]
            self.shelly_switch = cfg.get("switch_id", 0)
            # ID пристрою Shelly (напр. shellyplugsg3-a0b1c2) - за замовчуванням збігається з префіксом
            self.shelly_device_id = cfg.get("shelly_id", self.shelly_prefix)
            self.state_topic = f"{self.shelly_prefix}/status/switch:{self.shelly_switch}"
        elif self.driver == "mqtt":
            self.command_topic = cfg["command_topic"]
            self.state_topic = cfg.get("state_topic")
            self.payload_on = cfg.get("payload_on", "ON")
            self.payload_off = cfg.get("payload_off", "OFF")
        else:
            raise DeviceError(f"{self.id}: невідомий драйвер '{self.driver}'")

    def command(self, state):
        """Повертає (топік, payload) для перемикання в state."""
        if self.driver == "shelly":
            payload = {
                "id": int(time.time()), "src": "yui", "method": "Switch.Set",
                "params": {"id": self.shelly_switch, "on": state == ON},
            }
            return f"{self.shelly_prefix}/rpc", json.dumps(payload)
        return self.command_topic, self.payload_on if state == ON else self.payload_off

    def parse_state(self, payload):
        """Розбирає повідомлення зі state_topic. Повертає ON/OFF або None."""
        text = payload.decode("utf-8", "replace").strip() if isinstance(payload, bytes) else str(payload)
        if self.driver == "shelly":
            try:
                output = json.loads(text).get("output")
            except (ValueError, AttributeError):
                return None
            return None if output is None else (ON if output else OFF)
        if text == self.payload_on:
            return ON
        if text == self.payload_off:
            return OFF
        return None


class SimulatedBus:
    """Імітація MQTT без брокера і заліза: пристрої миттєво 'підтверджують' команди."""
    def __init__(self):
        self.on_message = None
        self.published = []

    def start(self):
        pass

    def subscribe(self, topic):
        pass

    def publish(self, topic, payload):
        self.published.append((topic, payload))
        print(f"\n   [симуляція] {topic} <- {payload}")

    def stop(self):
        pass


class MqttBus:
    def __init__(self, host="localhost", port=1883, username=None, password=None):
        import paho.mqtt.client as mqtt
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"yui-{int(time.time())}")
        if username:
            self.client.username_pw_set(username, password)
        self.host, self.port = host, port
        self.topics = set()
        self.on_message = None
        self.connected = threading.Event()
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = lambda *a: self.connected.clear()
        self.client.on_message = lambda client, userdata, msg: self.on_message and self.on_message(msg.topic, msg.payload)

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code == 0:
            for topic in self.topics:  # Повторна підписка після перепідключення
                client.subscribe(topic)
            self.connected.set()

    def start(self):
        try:
            self.client.connect(self.host, self.port)
        except OSError as e:
            raise DeviceError(f"MQTT-брокер недоступний ({self.host}:{self.port}): {e}") from e
        self.client.loop_start()
        if not self.connected.wait(5):
            raise DeviceError(f"Не вдалося підключитися до MQTT ({self.host}:{self.port}) - перевір логін/пароль")

    def subscribe(self, topic):
        self.topics.add(topic)
        if self.connected.is_set():
            self.client.subscribe(topic)

    def publish(self, topic, payload):
        if not self.connected.is_set():
            raise DeviceError("Немає зв'язку з MQTT-брокером")
        self.client.publish(topic, payload, qos=1)

    def stop(self):
        self.client.loop_stop()
        self.client.disconnect()


class DeviceManager:
    def __init__(self, devices, bus):
        self.devices = {d.id: d for d in devices}
        self.bus = bus
        self.lock = threading.Lock()
        self.timers = {}
        self.simulated = isinstance(bus, SimulatedBus)
        if self.simulated:
            for d in devices:
                d.state = OFF
        self.by_state_topic = {d.state_topic: d for d in devices if d.state_topic}
        bus.on_message = self._on_message

    @classmethod
    def from_config(cls, path, simulate=False):
        with open(path, "rb") as f:
            cfg = tomllib.load(f)
        devices = [Device(d) for d in cfg.get("device", [])]
        ids = [d.id for d in devices]
        if len(ids) != len(set(ids)):
            raise DeviceError("Ідентифікатори пристроїв (id) мають бути унікальними")
        if simulate:
            bus = SimulatedBus()
        else:
            mqtt_cfg = cfg.get("mqtt", {})
            bus = MqttBus(mqtt_cfg.get("host", "localhost"), mqtt_cfg.get("port", 1883),
                          mqtt_cfg.get("username"), mqtt_cfg.get("password"))
        return cls(devices, bus)

    def start(self):
        self.bus.start()
        for topic in self.by_state_topic:
            self.bus.subscribe(topic)
        if any(d.driver == "shelly" for d in self.devices.values()):
            self.bus.subscribe(RPC_REPLY_TOPIC)
        if not self.simulated:
            # Shelly не шле стан сам по собі одразу - просимо його
            for d in self.devices.values():
                if d.driver == "shelly":
                    self.bus.publish(f"{d.shelly_prefix}/rpc", json.dumps(
                        {"id": 0, "src": "yui", "method": "Switch.GetStatus", "params": {"id": d.shelly_switch}}))

    def stop(self):
        for timer in self.timers.values():
            timer.cancel()
        self.bus.stop()

    def _parse_rpc_reply(self, payload):
        """Відповідь Shelly на Switch.GetStatus: {"src": "<shelly_id>", "result": {"id": 0, "output": true}}."""
        try:
            reply = json.loads(payload)
            result = reply.get("result") or {}
            src = reply.get("src")
        except (ValueError, AttributeError):
            return None, None
        if "output" not in result:
            return None, None
        for d in self.devices.values():
            if d.driver == "shelly" and d.shelly_device_id == src and d.shelly_switch == result.get("id", 0):
                return d, ON if result["output"] else OFF
        return None, None

    def _on_message(self, topic, payload):
        if topic == RPC_REPLY_TOPIC:
            device, state = self._parse_rpc_reply(payload)
        else:
            device = self.by_state_topic.get(topic)
            state = device.parse_state(payload) if device else None
        if device is None:
            return
        if state is not None:
            with self.lock:
                device.state = state
                device.last_seen = time.time()
            if state == OFF:
                self._cancel_timer(device.id)

    def _cancel_timer(self, device_id):
        timer = self.timers.pop(device_id, None)
        if timer:
            timer.cancel()

    def set_state(self, device_id, action):
        """action: on / off / toggle. Повертає новий очікуваний стан."""
        device = self.devices.get(device_id)
        if device is None:
            raise DeviceError(f"Невідомий пристрій '{device_id}'")
        if action == "toggle":
            action = OFF if device.state == ON else ON
        if action not in (ON, OFF):
            raise DeviceError(f"Невідома дія '{action}'")

        topic, payload = device.command(action)
        self.bus.publish(topic, payload)
        if self.simulated or not device.state_topic:
            device.state = action  # Підтвердження не буде - вважаємо, що спрацювало

        # Страховка: автовимкнення на сервері (на самому пристрої теж варто налаштувати)
        self._cancel_timer(device.id)
        if action == ON and device.auto_off_minutes:
            timer = threading.Timer(device.auto_off_minutes * 60, self._auto_off, args=[device.id])
            timer.daemon = True
            self.timers[device.id] = timer
            timer.start()
        return action

    def _auto_off(self, device_id):
        self.timers.pop(device_id, None)
        try:
            self.set_state(device_id, OFF)
            print(f"\n⏱️ Автовимкнення: {self.devices[device_id].name}")
        except DeviceError as e:
            print(f"\n⚠️ Автовимкнення {device_id} не вдалося: {e}")

    def describe(self):
        """Список пристроїв для системного промпту."""
        names = {ON: "увімкнено", OFF: "вимкнено", None: "стан невідомий"}
        lines = []
        for d in self.devices.values():
            extra = f", інші назви: {', '.join(d.aliases)}" if d.aliases else ""
            room = f", кімната: {d.room}" if d.room else ""
            lines.append(f"- id={d.id}: {d.name}{room}{extra} - {names[d.state]}")
        return "\n".join(lines)
