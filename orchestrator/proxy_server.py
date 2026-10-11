import http.server
import logging
import socketserver
import urllib.request
import urllib.error
from typing import Optional

from .instance_manager import InstanceManager

logger = logging.getLogger("orchestrator.proxy_server")


class ProxyHTTPRequestHandler(http.server.BaseHTTPRequestHandler):
    """
    HTTP Request Handler that lazily boots a Vast.ai instance on arrival of
    OpenAI-compatible requests from LiteLLM and proxies the payload.
    """
    manager: InstanceManager

    def log_message(self, format, *args):
        logger.info("%s - %s", self.address_string(), format % args)

    def _handle_forward(self):
        # 1. Health/Status endpoint for local monitoring
        if self.path in ("/status", "/health"):
            state = self.manager.get_state()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(f'{{"status": "{state.value}"}}'.encode("utf-8"))
            return

        # 2. Lazy provisioning / readiness check
        try:
            endpoint, api_token = self.manager.ensure_ready()
        except Exception as e:
            logger.error("Failed to ensure instance readiness: %s", e)
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            err_payload = f'{{"error": {{"message": "Failed to provision GPU instance: {str(e)}", "type": "orchestrator_error"}}}}'
            self.wfile.write(err_payload.encode("utf-8"))
            return

        # 3. Read incoming request body
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else None

        # 4. Construct remote URL
        host_ip, host_port = endpoint
        target_url = f"http://{host_ip}:{host_port}{self.path}"

        # 5. Build forwarded headers
        forward_headers = {}
        for key, value in self.headers.items():
            if key.lower() not in ("host", "authorization", "content-length"):
                forward_headers[key] = value

        # Inject ephemeral security token
        forward_headers["Authorization"] = f"Bearer {api_token}"
        forward_headers["Host"] = f"{host_ip}:{host_port}"

        req = urllib.request.Request(
            target_url,
            data=body,
            headers=forward_headers,
            method=self.command
        )

        try:
            self.manager.touch()
            with urllib.request.urlopen(req, timeout=120) as resp:
                self.send_response(resp.status)
                for key, val in resp.getheaders():
                    if key.lower() not in ("transfer-encoding", "content-length"):
                        self.send_header(key, val)
                
                # Stream response bytes
                data = resp.read()
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                self.manager.touch()
        except urllib.error.HTTPError as e:
            err_data = e.read()
            self.send_response(e.code)
            for key, val in e.headers.items():
                if key.lower() not in ("transfer-encoding", "content-length"):
                    self.send_header(key, val)
            self.send_header("Content-Length", str(len(err_data)))
            self.end_headers()
            self.wfile.write(err_data)
        except Exception as e:
            logger.error("Error proxying request to %s: %s", target_url, e)
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            err_payload = f'{{"error": {{"message": "Proxy communication error: {str(e)}", "type": "bad_gateway"}}}}'
            self.wfile.write(err_payload.encode("utf-8"))

    def do_GET(self):
        self._handle_forward()

    def do_POST(self):
        self._handle_forward()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()


class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


def create_proxy_server(manager: InstanceManager, host: str, port: int) -> ThreadedTCPServer:
    """Create a multithreaded proxy server wired to the given InstanceManager."""
    handler_class = type("CustomProxyHandler", (ProxyHTTPRequestHandler,), {"manager": manager})
    server = ThreadedTCPServer((host, port), handler_class)
    return server
