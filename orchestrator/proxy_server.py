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

        # 2. Read incoming request body immediately to drain client socket
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else None

        # 3. Lazy provisioning / readiness check
        try:
            endpoint, api_token = self.manager.ensure_ready()
        except Exception as e:
            logger.error("Failed to ensure instance readiness: %s", e)
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            err_bytes = f'{{"error": {{"message": "Failed to provision GPU instance: {str(e)}", "type": "orchestrator_error"}}}}'.encode("utf-8")
            self.send_header("Content-Length", str(len(err_bytes)))
            self.end_headers()
            self.wfile.write(err_bytes)
            return

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
            with urllib.request.urlopen(req, timeout=600) as resp:
                self.send_response(resp.status)
                for key, val in resp.getheaders():
                    if key.lower() not in ("transfer-encoding", "connection", "keep-alive"):
                        self.send_header(key, val)
                self.end_headers()

                # Stream response bytes progressively (crucial for SSE tokens and large outputs)
                try:
                    while True:
                        chunk = resp.read(4096)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    logger.debug("Client disconnected while reading response stream")
                self.manager.touch()
        except urllib.error.HTTPError as e:
            try:
                err_data = e.read()
                self.send_response(e.code)
                for key, val in e.headers.items():
                    if key.lower() not in ("transfer-encoding", "connection", "keep-alive", "content-length"):
                        self.send_header(key, val)
                self.send_header("Content-Length", str(len(err_data)))
                self.end_headers()
                self.wfile.write(err_data)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
        except (BrokenPipeError, ConnectionResetError):
            logger.debug("Client disconnected before response could be sent")
        except Exception as e:
            logger.error("Error proxying request to %s: %s", target_url, e)
            if isinstance(e, urllib.error.URLError) or "refused" in str(e).lower() or "reset" in str(e).lower():
                self.manager.mark_failed()
            try:
                self.send_response(502)
                self.send_header("Content-Type", "application/json")
                err_bytes = f'{{"error": {{"message": "Proxy communication error: {str(e)}", "type": "bad_gateway"}}}}'.encode("utf-8")
                self.send_header("Content-Length", str(len(err_bytes)))
                self.end_headers()
                self.wfile.write(err_bytes)
            except (BrokenPipeError, ConnectionResetError):
                pass

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
