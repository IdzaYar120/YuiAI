"""Інтеграційний тест зі справжнім MQTT-брокером (amqtt) і імітацією ESPHome-реле та Shelly."""
import asyncio
import json
import threading
import time
import unittest

try:
    import paho.mqtt.client as mqtt
    from amqtt.broker import Broker
except ImportError:
    mqtt = None

from src.devices import Device, DeviceManager, MqttBus

PORT = 18830


def wait_for(predicate, timeout=5):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


@unittest.skipIf(mqtt is None, "потрібні paho-mqtt і amqtt")
class MqttIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loop = asyncio.new_event_loop()
        started = threading.Event()

        async def start_broker():
            cls.broker = Broker({"listeners": {"default": {"type": "tcp", "bind": f"127.0.0.1:{PORT}"}},
                                 "sys_interval": 0, "auth": {"allow-anonymous": True}})
            await cls.broker.start()

        def run():
            asyncio.set_event_loop(cls.loop)
            cls.loop.run_until_complete(start_broker())
            started.set()
            cls.loop.run_forever()

        threading.Thread(target=run, daemon=True).start()
        assert started.wait(10)

        # Імітація заліза: ESPHome-реле і Shelly відповідають станом на команди
        cls.hw = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="fake-hw")
        def on_message(client, userdata, msg):
            if msg.topic == "yui/light/set":
                client.publish("yui/light/state", msg.payload, retain=True)
            elif msg.topic == "shelly-1/rpc":
                req = json.loads(msg.payload)
                if req["method"] == "Switch.Set":
                    cls.shelly_on = req["params"]["on"]
                    client.publish("shelly-1/status/switch:0", json.dumps({"id": 0, "output": cls.shelly_on}))
                client.publish(f"{req['src']}/rpc", json.dumps(
                    {"id": req["id"], "src": "shelly-1", "dst": req["src"],
                     "result": {"id": 0, "output": cls.shelly_on} if req["method"] == "Switch.GetStatus" else {}}))
        cls.shelly_on = True
        cls.hw.on_message = on_message
        cls.hw.connect("127.0.0.1", PORT)
        cls.hw.subscribe("yui/light/set")
        cls.hw.subscribe("shelly-1/rpc")
        cls.hw.loop_start()
        time.sleep(0.5)

    @classmethod
    def tearDownClass(cls):
        cls.hw.loop_stop()
        asyncio.run_coroutine_threadsafe(cls.broker.shutdown(), cls.loop).result(10)
        cls.loop.call_soon_threadsafe(cls.loop.stop)

    def test_roundtrip(self):
        devices = [
            Device({"id": "light", "name": "Світло", "driver": "mqtt",
                    "command_topic": "yui/light/set", "state_topic": "yui/light/state"}),
            Device({"id": "kettle", "name": "Чайник", "driver": "shelly", "topic_prefix": "shelly-1"}),
        ]
        manager = DeviceManager(devices, MqttBus("127.0.0.1", PORT))
        manager.start()
        try:
            # Початковий стан Shelly (увімкнена) приходить у відповідь на Switch.GetStatus
            self.assertTrue(wait_for(lambda: manager.devices["kettle"].state == "on"))

            manager.set_state("light", "on")
            self.assertTrue(wait_for(lambda: manager.devices["light"].state == "on"))
            manager.set_state("kettle", "off")
            self.assertTrue(wait_for(lambda: manager.devices["kettle"].state == "off"))
            self.assertFalse(self.shelly_on)
        finally:
            manager.stop()


if __name__ == "__main__":
    unittest.main()
