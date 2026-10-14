"""Цикл виклику інструментів через фейковий сервер Ollama."""
import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

os.environ["OLLAMA_HOST"] = "http://127.0.0.1:18766"
os.environ.setdefault("NO_PROXY", "127.0.0.1")
os.environ.setdefault("no_proxy", "127.0.0.1")

from src import llm  # noqa: E402

REQUESTS = []


class FakeOllama(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        REQUESTS.append(body)
        if body["messages"][-1]["role"] == "user":
            chunks = [{"message": {"content": "", "tool_calls": [{"function": {
                "name": "control_devices", "arguments": {"device_ids": ["kettle"], "action": "on"}}}]}},
                {"done": True}]
        else:
            chunks = [{"message": {"content": "Чайник увімкнено, Майстре! (◕‿◕)"}}, {"done": True}]
        self.send_response(200)
        self.end_headers()
        for chunk in chunks:
            self.wfile.write((json.dumps(chunk, ensure_ascii=False) + "\n").encode())


class ToolLoopTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 18766), FakeOllama)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def test_tool_call_then_answer(self):
        calls = []
        def execute(name, arguments):
            calls.append((name, arguments))
            return "Чайник: увімкнено"

        messages = [{"role": "user", "content": "Юї, включи чайник"}]
        text = llm.chat_with_tools(messages, [{"type": "function"}], execute)

        self.assertEqual(text, "Чайник увімкнено, Майстре! (◕‿◕)")
        self.assertEqual(calls, [("control_devices", {"device_ids": ["kettle"], "action": "on"})])
        self.assertEqual(len(messages), 1)  # Вхідний список не змінюється
        second = REQUESTS[-1]["messages"]
        self.assertEqual(second[-2]["role"], "assistant")
        self.assertIn("tool_calls", second[-2])
        self.assertEqual(second[-1], {"role": "tool", "tool_name": "control_devices", "content": "Чайник: увімкнено"})


if __name__ == "__main__":
    unittest.main()
