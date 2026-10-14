import json
import time
import unittest

from src.devices import Device, DeviceManager, DeviceError, SimulatedBus
from src.tools import make_executor

SHELLY = {"id": "kettle", "name": "Чайник", "room": "Кухня", "driver": "shelly", "topic_prefix": "shelly-1"}
RELAY = {"id": "light", "name": "Світло", "room": "Кухня", "driver": "mqtt",
         "command_topic": "yui/light/set", "state_topic": "yui/light/state"}


class DeviceTest(unittest.TestCase):
    def test_shelly_command_and_state(self):
        d = Device(SHELLY)
        topic, payload = d.command("on")
        self.assertEqual(topic, "shelly-1/rpc")
        self.assertEqual(json.loads(payload)["params"], {"id": 0, "on": True})
        self.assertEqual(d.state_topic, "shelly-1/status/switch:0")
        self.assertEqual(d.parse_state(b'{"id":0,"output":false,"apower":0}'), "off")
        self.assertIsNone(d.parse_state(b"garbage"))

    def test_mqtt_command_and_state(self):
        d = Device(RELAY)
        self.assertEqual(d.command("off"), ("yui/light/set", "OFF"))
        self.assertEqual(d.parse_state(b"ON"), "on")
        self.assertIsNone(d.parse_state(b"maybe"))

    def test_bad_config(self):
        with self.assertRaises(DeviceError):
            Device({"id": "x"})
        with self.assertRaises(DeviceError):
            Device({"id": "x", "name": "X", "driver": "zigbee"})


class ManagerTest(unittest.TestCase):
    def setUp(self):
        self.manager = DeviceManager([Device(SHELLY), Device(RELAY)], SimulatedBus())
        self.manager.start()

    def tearDown(self):
        self.manager.stop()

    def test_set_and_toggle(self):
        self.assertEqual(self.manager.set_state("light", "on"), "on")
        self.assertEqual(self.manager.set_state("light", "toggle"), "off")
        self.assertEqual(self.manager.bus.published[-1], ("yui/light/set", "OFF"))
        with self.assertRaises(DeviceError):
            self.manager.set_state("toaster", "on")
        with self.assertRaises(DeviceError):
            self.manager.set_state("light", "explode")

    def test_state_messages(self):
        self.manager._on_message("yui/light/state", b"ON")
        self.assertEqual(self.manager.devices["light"].state, "on")
        # Відповідь Shelly на Switch.GetStatus
        self.manager._on_message("yui/rpc", json.dumps({"src": "shelly-1", "result": {"id": 0, "output": True}}).encode())
        self.assertEqual(self.manager.devices["kettle"].state, "on")
        self.assertIn("Світло, кімната: Кухня - увімкнено", self.manager.describe())

    def test_auto_off(self):
        self.manager.devices["kettle"].auto_off_minutes = 0.002  # ~0.1 с
        self.manager.set_state("kettle", "on")
        time.sleep(0.4)
        self.assertEqual(self.manager.devices["kettle"].state, "off")

    def test_executor(self):
        execute = make_executor(self.manager)
        self.assertEqual(execute("control_devices", {"device_ids": ["kettle", "light"], "action": "on"}),
                         "Чайник: увімкнено; Світло: увімкнено")
        self.assertIn("помилка", execute("control_devices", {"device_ids": ["nope"], "action": "on"}))
        self.assertIn("невідомий інструмент", execute("rm_rf", {}))


if __name__ == "__main__":
    unittest.main()
