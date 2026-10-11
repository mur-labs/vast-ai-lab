import json
import socket
import threading
import time
import unittest
import urllib.request
from http.server import HTTPServer, BaseHTTPRequestHandler
from unittest.mock import MagicMock

from orchestrator.config import OrchestratorConfig
from orchestrator.instance_manager import InstanceManager, InstanceState
from orchestrator.proxy_server import create_proxy_server


class MockLlamaServerHandler(BaseHTTPRequestHandler):
    """Mocks remote llama-server endpoints."""
    received_auth_header = None
    received_payload = None

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        MockLlamaServerHandler.received_auth_header = self.headers.get("Authorization")
        content_len = int(self.headers.get("Content-Length", 0))
        MockLlamaServerHandler.received_payload = self.rfile.read(content_len)

        if self.path == "/v1/chat/completions":
            resp_body = json.dumps({
                "id": "chatcmpl-test-123",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello from Vast.ai GPU!"},
                    "finish_reason": "stop"
                }]
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp_body)))
            self.end_headers()
            self.wfile.write(resp_body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


class TestProxyServer(unittest.TestCase):
    def setUp(self):
        MockLlamaServerHandler.received_auth_header = None
        MockLlamaServerHandler.received_payload = None

        # Start mock remote llama server
        self.remote_server = HTTPServer(("127.0.0.1", 0), MockLlamaServerHandler)
        self.remote_port = self.remote_server.server_port
        self.remote_thread = threading.Thread(target=self.remote_server.serve_forever, daemon=True)
        self.remote_thread.start()

        # Mock VastClient
        self.mock_client = MagicMock()
        self.mock_client.search_offers.return_value = [{"id": 11, "dph_total": 0.15}]
        self.mock_client.launch_instance.return_value = 777
        self.mock_client.resolve_dynamic_endpoint.return_value = ("127.0.0.1", self.remote_port)
        self.mock_client.destroy_instance.return_value = True

        self.config = OrchestratorConfig(
            listen_host="127.0.0.1",
            listen_port=0,
            poll_interval_seconds=0.01,
            boot_timeout_seconds=2,
            container_port=8000
        )
        self.manager = InstanceManager(self.config, vast_client=self.mock_client)

        # Start proxy server on ephemeral port
        self.proxy_server = create_proxy_server(self.manager, "127.0.0.1", 0)
        self.proxy_port = self.proxy_server.server_address[1]
        self.proxy_thread = threading.Thread(target=self.proxy_server.serve_forever, daemon=True)
        self.proxy_thread.start()

    def tearDown(self):
        self.manager.stop_reaper()
        self.manager.destroy()
        self.proxy_server.shutdown()
        self.proxy_server.server_close()
        self.remote_server.shutdown()
        self.remote_server.server_close()

    def test_status_endpoint(self):
        """Verify /status endpoint returns valid json state."""
        url = f"http://127.0.0.1:{self.proxy_port}/status"
        with urllib.request.urlopen(url, timeout=2) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertIn("status", data)

    def test_lazy_provision_and_forwarding(self):
        """
        Verify incoming request from LiteLLM triggers provisioning,
        forwards with ephemeral token, and returns completion response.
        """
        url = f"http://127.0.0.1:{self.proxy_port}/v1/chat/completions"
        payload = json.dumps({
            "model": "qwen2.5-14b-instruct",
            "messages": [{"role": "user", "content": "Hi!"}]
        }).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        with urllib.request.urlopen(req, timeout=5) as resp:
            self.assertEqual(resp.status, 200)
            resp_body = json.loads(resp.read().decode("utf-8"))
            content = resp_body["choices"][0]["message"]["content"]
            self.assertEqual(content, "Hello from Vast.ai GPU!")

        # Verify remote server received Authorization header with ephemeral token
        self.assertIsNotNone(MockLlamaServerHandler.received_auth_header)
        self.assertTrue(MockLlamaServerHandler.received_auth_header.startswith("Bearer "))
        self.assertEqual(self.manager.get_state(), InstanceState.READY)


if __name__ == "__main__":
    unittest.main()
