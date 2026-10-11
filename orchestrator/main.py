import argparse
import logging
import os
import signal
import sys
from pathlib import Path

# Ensure project root is in sys.path regardless of execution directory
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from .config import HostCriteria, ModelConfig, OrchestratorConfig
    from .instance_manager import InstanceManager
    from .proxy_server import create_proxy_server
except (ImportError, ValueError):
    from orchestrator.config import HostCriteria, ModelConfig, OrchestratorConfig
    from orchestrator.instance_manager import InstanceManager
    from orchestrator.proxy_server import create_proxy_server

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s"
)
logger = logging.getLogger("orchestrator")


def main():
    parser = argparse.ArgumentParser(description="Rental LLM Vast.ai Orchestrator")
    parser.add_argument("--port", type=int, default=8088, help="Local listening port for LiteLLM (default: 8088)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Local host binding (default: 127.0.0.1)")
    parser.add_argument("--idle-timeout", type=int, default=600, help="Idle timeout in seconds before destruction (default: 600)")
    parser.add_argument("--image", type=str, default=os.getenv("RENTAL_DOCKER_IMAGE", "ghcr.io/mur-labs/rental-llm:latest"), help="Docker image name")
    parser.add_argument("--model-url", type=str, default=None, help="Direct GGUF model download URL")

    args = parser.parse_args()

    config = OrchestratorConfig(
        listen_host=args.host,
        listen_port=args.port,
        idle_timeout_seconds=args.idle_timeout,
        docker_image=args.image,
    )

    if args.model_url:
        config.active_model.model_url = args.model_url

    if not config.vast_api_key:
        logger.warning("VAST_API_KEY environment variable is not set. Run with VAST_API_KEY for live provisioning.")

    logger.info("Initializing Instance Manager...")
    manager = InstanceManager(config)

    server = create_proxy_server(manager, config.listen_host, config.listen_port)
    logger.info("Orchestrator proxy server listening on http://%s:%s", config.listen_host, config.listen_port)
    logger.info("Ready to receive requests from LiteLLM.")

    def handle_shutdown(signum, frame):
        logger.info("Shutting down orchestrator server...")
        manager.stop_reaper()
        manager.destroy()
        server.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        handle_shutdown(None, None)


if __name__ == "__main__":
    main()
