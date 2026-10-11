import os
import sys
import stat
import json
import socket
import tempfile
import threading
import subprocess
import unittest
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

ENTRYPOINT_PATH = Path(__file__).resolve().parent.parent / "entrypoint.sh"


class MockModelHandler(BaseHTTPRequestHandler):
    """HTTP handler that serves mock model bytes and logs headers."""
    model_data = b"GGUF_MOCK_MODEL_DATA_FOR_TESTING_1234567890\n"
    received_headers = []

    def do_GET(self):
        MockModelHandler.received_headers.append(dict(self.headers))
        if self.path == "/model.gguf":
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(self.model_data)))
            self.end_headers()
            self.wfile.write(self.model_data)
        elif self.path == "/notfound.gguf":
            self.send_response(404)
            self.end_headers()
        else:
            self.send_response(400)
            self.end_headers()

    def log_message(self, format, *args):
        # Silence standard HTTP server output in test logs
        pass


class TestEntrypoint(unittest.TestCase):
    def setUp(self):
        self.assertTrue(ENTRYPOINT_PATH.exists(), f"entrypoint.sh not found at {ENTRYPOINT_PATH}")
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        MockModelHandler.received_headers.clear()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_mock_llama_server(self, record_path: Path) -> Path:
        """Create an executable mock llama-server script that dumps received argv to JSON."""
        mock_server = self.temp_path / "mock_llama_server.sh"
        content = f"""#!/usr/bin/env bash
python3 -c "import sys, json; json.dump(sys.argv[1:], open('{record_path}', 'w'))" "$@"
exit 0
"""
        mock_server.write_text(content)
        mock_server.chmod(mock_server.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        return mock_server

    def test_missing_model_url(self):
        """Verifies error and non-zero exit when MODEL_URL is missing."""
        env = {
            "PATH": os.environ["PATH"],
            "API_TOKEN": "test-token-123",
        }
        proc = subprocess.run(
            [str(ENTRYPOINT_PATH)],
            env=env,
            capture_output=True,
            text=True
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("MODEL_URL environment variable is required", proc.stderr)

    def test_missing_api_token(self):
        """Verifies error and non-zero exit when API_TOKEN is missing."""
        env = {
            "PATH": os.environ["PATH"],
            "MODEL_URL": "http://example.com/model.gguf",
        }
        proc = subprocess.run(
            [str(ENTRYPOINT_PATH)],
            env=env,
            capture_output=True,
            text=True
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("API_TOKEN environment variable is required", proc.stderr)

    def test_missing_llama_server_binary(self):
        """Verifies error and non-zero exit when llama-server binary is not found."""
        # Provide PATH with standard bin utilities so bash/curl exist, but without llama-server
        env = {
            "PATH": "/bin:/usr/bin",
            "MODEL_URL": "http://example.com/model.gguf",
            "API_TOKEN": "test-token-123",
            "LLAMA_BIN": "/nonexistent/path/llama-server",
        }
        proc = subprocess.run(
            [str(ENTRYPOINT_PATH)],
            env=env,
            capture_output=True,
            text=True
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("llama-server executable not found", proc.stderr)

    def test_successful_download_and_secure_start(self):
        """Verifies complete flow: downloads model, validates token, and starts server with correct flags."""
        server = HTTPServer(("127.0.0.1", 0), MockModelHandler)
        port = server.server_port
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()

        args_record_file = self.temp_path / "server_args.json"
        mock_server = self._create_mock_llama_server(args_record_file)
        model_download_dir = self.temp_path / "models"

        test_token = "secret-vast-ephemeral-token-98765"
        env = {
            "PATH": os.environ["PATH"],
            "MODEL_URL": f"http://127.0.0.1:{port}/model.gguf",
            "API_TOKEN": test_token,
            "MODEL_DIR": str(model_download_dir),
            "LLAMA_BIN": str(mock_server),
            "HOST": "0.0.0.0",
            "PORT": "8080",
            "N_GPU_LAYERS": "33",
            "CTX_SIZE": "8192",
        }

        extra_args = ["--cont-batching", "--parallel", "2"]
        proc = subprocess.run(
            [str(ENTRYPOINT_PATH)] + extra_args,
            env=env,
            capture_output=True,
            text=True
        )

        server.shutdown()
        server.server_close()

        self.assertEqual(proc.returncode, 0, f"Script failed with stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}")

        # Check downloaded file
        downloaded_model = model_download_dir / "model.gguf"
        self.assertTrue(downloaded_model.exists())
        self.assertEqual(downloaded_model.read_bytes(), MockModelHandler.model_data)

        # Check mock server invocation arguments
        self.assertTrue(args_record_file.exists())
        invoked_args = json.loads(args_record_file.read_text())

        self.assertIn("--model", invoked_args)
        model_arg_idx = invoked_args.index("--model")
        self.assertEqual(invoked_args[model_arg_idx + 1], str(downloaded_model))

        self.assertIn("--api-key", invoked_args)
        api_key_idx = invoked_args.index("--api-key")
        self.assertEqual(invoked_args[api_key_idx + 1], test_token)

        self.assertIn("--host", invoked_args)
        self.assertEqual(invoked_args[invoked_args.index("--host") + 1], "0.0.0.0")

        self.assertIn("--port", invoked_args)
        self.assertEqual(invoked_args[invoked_args.index("--port") + 1], "8080")

        self.assertIn("-ngl", invoked_args)
        self.assertEqual(invoked_args[invoked_args.index("-ngl") + 1], "33")

        self.assertIn("-c", invoked_args)
        self.assertEqual(invoked_args[invoked_args.index("-c") + 1], "8192")

        # Verify extra forwarded args
        self.assertIn("--cont-batching", invoked_args)
        self.assertIn("--parallel", invoked_args)

    def test_hf_token_authorization_header(self):
        """Verifies that HF_TOKEN is correctly passed in Authorization header during download."""
        server = HTTPServer(("127.0.0.1", 0), MockModelHandler)
        port = server.server_port
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()

        args_record_file = self.temp_path / "server_args_hf.json"
        mock_server = self._create_mock_llama_server(args_record_file)
        model_download_dir = self.temp_path / "models_hf"

        hf_secret = "hf_dummy_token_abcdef123"
        env = {
            "PATH": os.environ["PATH"],
            "MODEL_URL": f"http://127.0.0.1:{port}/model.gguf",
            "API_TOKEN": "ephemeral-api-token",
            "HF_TOKEN": hf_secret,
            "MODEL_DIR": str(model_download_dir),
            "LLAMA_BIN": str(mock_server),
        }

        proc = subprocess.run(
            [str(ENTRYPOINT_PATH)],
            env=env,
            capture_output=True,
            text=True
        )

        server.shutdown()
        server.server_close()

        self.assertEqual(proc.returncode, 0, f"Script failed with stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}")
        self.assertTrue(len(MockModelHandler.received_headers) > 0)
        auth_header = MockModelHandler.received_headers[0].get("Authorization")
        self.assertEqual(auth_header, f"Bearer {hf_secret}")

    def test_download_failure_exits_with_error(self):
        """Verifies that a 404 download failure terminates without running the server."""
        server = HTTPServer(("127.0.0.1", 0), MockModelHandler)
        port = server.server_port
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()

        args_record_file = self.temp_path / "server_args_fail.json"
        mock_server = self._create_mock_llama_server(args_record_file)

        env = {
            "PATH": os.environ["PATH"],
            "MODEL_URL": f"http://127.0.0.1:{port}/notfound.gguf",
            "API_TOKEN": "ephemeral-api-token",
            "MODEL_DIR": str(self.temp_path / "fail_models"),
            "LLAMA_BIN": str(mock_server),
        }

        proc = subprocess.run(
            [str(ENTRYPOINT_PATH)],
            env=env,
            capture_output=True,
            text=True
        )

        server.shutdown()
        server.server_close()

        self.assertNotEqual(proc.returncode, 0)
        # Server must not have been invoked
        self.assertFalse(args_record_file.exists())


if __name__ == "__main__":
    unittest.main()
