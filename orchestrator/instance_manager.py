import enum
import logging
import secrets
import threading
import time
import urllib.request
from typing import Optional, Tuple

from .config import OrchestratorConfig
from .vast_client import VastClient, VastClientError

logger = logging.getLogger("orchestrator.instance_manager")


class InstanceState(str, enum.Enum):
    STOPPED = "STOPPED"
    PROVISIONING = "PROVISIONING"
    READY = "READY"
    BUSY = "BUSY"
    DESTROYING = "DESTROYING"


class InstanceManager:
    """
    Manages the lifecycle of on-demand Vast.ai GPU instances:
    - Zero-trust ephemeral API token generation per instance
    - State tracking and deduplication to prevent LiteLLM retry storms
    - Dynamic port resolution and readiness health checks
    - Idle timeout monitoring (10 minutes) with auto-destruction
    """

    def __init__(self, config: OrchestratorConfig, vast_client: Optional[VastClient] = None):
        self.config = config
        self.client = vast_client or VastClient(api_key=config.vast_api_key)

        self.state: InstanceState = InstanceState.STOPPED
        self.instance_id: Optional[int] = None
        self.endpoint: Optional[Tuple[str, int]] = None
        self.api_token: Optional[str] = None
        self.last_active_time: float = 0.0

        self._lock = threading.RLock()
        self._state_condition = threading.Condition(self._lock)
        self._reaper_stop_event = threading.Event()
        self._reaper_thread = threading.Thread(target=self._reaper_loop, daemon=True)
        self._reaper_thread.start()

    def get_state(self) -> InstanceState:
        with self._lock:
            return self.state

    def touch(self):
        """Reset the idle timer upon receiving an incoming request."""
        with self._lock:
            self.last_active_time = time.time()

    def ensure_ready(self, timeout: Optional[float] = None) -> Tuple[Tuple[str, int], str]:
        """
        Ensures a GPU instance is provisioned and ready to handle traffic.
        Thread-safe deduplication: if multiple requests arrive simultaneously,
        only one instance is launched while others wait.
        Returns: ((host_ip, host_port), ephemeral_api_token)
        """
        effective_timeout = timeout or self.config.boot_timeout_seconds
        start_time = time.time()

        with self._state_condition:
            while True:
                if self.state == InstanceState.READY and self.endpoint and self.api_token:
                    self.touch()
                    return self.endpoint, self.api_token

                if self.state == InstanceState.STOPPED:
                    # Claim the provisioning lock
                    self.state = InstanceState.PROVISIONING
                    self._state_condition.notify_all()
                    break

                if self.state == InstanceState.PROVISIONING:
                    # Another thread is already provisioning; wait for it to finish
                    elapsed = time.time() - start_time
                    remaining = effective_timeout - elapsed
                    if remaining <= 0:
                        raise TimeoutError("Timed out waiting for instance provisioning.")
                    self._state_condition.wait(timeout=min(2.0, remaining))
                    continue

                if self.state == InstanceState.DESTROYING:
                    # Wait for destruction to complete before re-provisioning
                    self._state_condition.wait(timeout=1.0)
                    continue

        # If we broke out of the loop above, this thread is responsible for provisioning
        try:
            endpoint, token = self._provision_instance(timeout=effective_timeout)
            with self._state_condition:
                self.state = InstanceState.READY
                self.endpoint = endpoint
                self.api_token = token
                self.last_active_time = time.time()
                self._state_condition.notify_all()
            return endpoint, token
        except Exception as e:
            logger.error("Provisioning failed: %s", e)
            with self._state_condition:
                # If an instance contract was partially created, schedule destruction
                if self.instance_id:
                    try:
                        self.client.destroy_instance(self.instance_id)
                    except Exception as destroy_err:
                        logger.error("Failed to cleanup instance after provision error: %s", destroy_err)
                self.state = InstanceState.STOPPED
                self.instance_id = None
                self.endpoint = None
                self.api_token = None
                self._state_condition.notify_all()
            raise

    def _provision_instance(self, timeout: float) -> Tuple[Tuple[str, int], str]:
        """Execute offer search, launch, polling, and port resolution."""
        logger.info("Searching Vast.ai spot offers matching criteria...")
        offers = self.client.search_offers(self.config.host_criteria)
        if not offers:
            raise VastClientError("No Vast.ai hosts met the criteria (>=20GB VRAM, >=40GB Disk, >=500Mbps, >=97%).")

        best_offer = offers[0]
        offer_id = best_offer["id"]
        logger.info(
            "Selected offer ID %s (GPU: %s, VRAM: %s MB, $/hr: %s)",
            offer_id,
            best_offer.get("gpu_name"),
            best_offer.get("gpu_ram"),
            best_offer.get("dph_total") or best_offer.get("min_bid")
        )

        # Generate zero-trust ephemeral API token
        ephemeral_token = secrets.token_urlsafe(32)

        env_vars = {
            "MODEL_URL": self.config.active_model.model_url,
            "API_TOKEN": ephemeral_token,
            "PORT": str(self.config.container_port),
            "CTX_SIZE": str(self.config.active_model.context_size),
            "N_GPU_LAYERS": str(self.config.active_model.gpu_layers),
        }
        if self.config.active_model.hf_token:
            env_vars["HF_TOKEN"] = self.config.active_model.hf_token

        logger.info("Launching container image %s on offer %s...", self.config.docker_image, offer_id)
        instance_id = self.client.launch_instance(
            offer_id=offer_id,
            image=self.config.docker_image,
            disk_gb=self.config.host_criteria.min_disk_gb,
            env=env_vars,
            container_port=self.config.container_port,
        )

        with self._lock:
            self.instance_id = instance_id

        # Poll until instance is running, port resolved, and server responds
        start_poll = time.time()
        logger.info("Awaiting container boot and dynamic port mapping for instance %s...", instance_id)

        while time.time() - start_poll < timeout:
            endpoint = self.client.resolve_dynamic_endpoint(
                instance_id=instance_id,
                container_port=self.config.container_port
            )
            if endpoint:
                host_ip, host_port = endpoint
                logger.info("Dynamic port resolved: %s:%s. Verifying health...", host_ip, host_port)
                if self._check_health(host_ip, host_port, ephemeral_token):
                    logger.info("Inference endpoint is live and verified healthy!")
                    return (host_ip, host_port), ephemeral_token

            time.sleep(self.config.poll_interval_seconds)

        raise TimeoutError(f"Instance {instance_id} failed to become healthy within {timeout} seconds.")

    def _check_health(self, host_ip: str, host_port: int, token: str) -> bool:
        """Pings the llama.cpp server /health endpoint with the ephemeral bearer token."""
        url = f"http://{host_ip}:{host_port}/health"
        req = urllib.request.Request(
            url,
            headers={"Authorization": f"Bearer {token}"},
            method="GET"
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status in (200, 204)
        except Exception:
            return False

    def destroy(self):
        """Explicitly destroy the running instance and reset state."""
        with self._state_condition:
            if self.state in (InstanceState.STOPPED, InstanceState.DESTROYING):
                return
            self.state = InstanceState.DESTROYING
            inst_to_destroy = self.instance_id
            self._state_condition.notify_all()

        if inst_to_destroy:
            try:
                self.client.destroy_instance(inst_to_destroy)
            except Exception as e:
                logger.error("Error destroying instance %s: %s", inst_to_destroy, e)

        with self._state_condition:
            self.state = InstanceState.STOPPED
            self.instance_id = None
            self.endpoint = None
            self.api_token = None
            self.last_active_time = 0.0
            self._state_condition.notify_all()

    def _reaper_loop(self):
        """Background thread checking for idle inactivity."""
        interval = min(2.0, max(0.05, self.config.idle_timeout_seconds / 2.0))
        while not self._reaper_stop_event.wait(timeout=interval):
            with self._lock:
                if self.state == InstanceState.READY and self.last_active_time > 0:
                    idle_duration = time.time() - self.last_active_time
                    if idle_duration >= self.config.idle_timeout_seconds:
                        logger.warning(
                            "Instance idle for %.1f seconds (threshold %s s). Triggering destruction.",
                            idle_duration,
                            self.config.idle_timeout_seconds
                        )
                        # Trigger destroy in separate thread to avoid deadlock
                        threading.Thread(target=self.destroy, daemon=True).start()

    def stop_reaper(self):
        """Stop the background reaper."""
        self._reaper_stop_event.set()
