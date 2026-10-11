import os
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class HostCriteria:
    """Hardware and network quality minimums for Vast.ai host selection."""
    min_vram_gb: float = 20.0
    min_disk_gb: float = 40.0
    min_inet_down_mbps: float = 500.0
    min_reliability: float = 0.97
    spot_instances: bool = True
    bid_buffer_ratio: float = 0.10  # 10% bid buffer above min_bid to prevent preemption


@dataclass
class ModelConfig:
    """Configuration for inference models."""
    name: str = "qwen2.5-14b-instruct"
    model_url: str = (
        "https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-GGUF/resolve/main/"
        "qwen2.5-14b-instruct-q4_k_m.gguf"
    )
    context_size: int = 4096
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
    container_port: int = 8000
    docker_image: str = os.getenv("RENTAL_DOCKER_IMAGE", "ghcr.io/mur-labs/rental-llm:latest")
    vast_api_key: Optional[str] = field(default_factory=lambda: os.getenv("VAST_API_KEY"))
    host_criteria: HostCriteria = field(default_factory=HostCriteria)
    active_model: ModelConfig = field(default_factory=ModelConfig)
