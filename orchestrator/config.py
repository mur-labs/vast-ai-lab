import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


def _load_dotenv():
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if env_file.exists():
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        if k and k not in os.environ:
                            os.environ[k] = v
        except Exception:
            pass


_load_dotenv()


@dataclass
class HostCriteria:
    """Hardware and network quality minimums for Vast.ai host selection."""
    min_vram_gb: float = 20.0
    min_disk_gb: float = 40.0
    min_inet_down_mbps: float = 500.0
    min_reliability: float = 0.97
    spot_instances: bool = field(
        default_factory=lambda: os.getenv("VAST_SPOT_INSTANCES", "true").lower() in ("true", "1", "yes")
    )
    bid_buffer_ratio: float = field(
        default_factory=lambda: float(os.getenv("VAST_BID_BUFFER", "0.20"))
    )  # 20% bid buffer above min_bid to prevent preemption


@dataclass
class ModelConfig:
    """Configuration for inference models."""
    name: str = "qwen2.5-14b-instruct"
    model_url: str = field(
        default_factory=lambda: os.getenv(
            "MODEL_URL",
            "https://huggingface.co/bartowski/Qwen2.5-14B-Instruct-GGUF/resolve/main/Qwen2.5-14B-Instruct-Q4_K_M.gguf"
        )
    )
    context_size: int = field(default_factory=lambda: int(os.getenv("CTX_SIZE", "32768")))
    gpu_layers: int = 99
    hf_token: Optional[str] = field(default_factory=lambda: os.getenv("HF_TOKEN"))


@dataclass
class OrchestratorConfig:
    """Main orchestrator configuration."""
    listen_host: str = "127.0.0.1"
    listen_port: int = 8088
    idle_timeout_seconds: int = 600  # 10 minutes idle timeout
    poll_interval_seconds: int = 5
    boot_timeout_seconds: int = 600
    container_port: int = field(default_factory=lambda: int(os.getenv("CONTAINER_PORT", "8080")))
    docker_image: str = os.getenv("RENTAL_DOCKER_IMAGE", "localai/localai:v4.11.0-gpu-nvidia-cuda-12")
    vast_api_key: Optional[str] = field(default_factory=lambda: os.getenv("VAST_API_KEY"))
    idle_action: str = field(default_factory=lambda: os.getenv("VAST_IDLE_ACTION", "destroy"))  # "destroy" for ephemeral, "stop" to preserve weights
    host_criteria: HostCriteria = field(default_factory=HostCriteria)
    active_model: ModelConfig = field(default_factory=ModelConfig)
