import enum
import json
import logging
import secrets
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .config import OrchestratorConfig
from .vast_client import VastClient, VastClientError

logger = logging.getLogger("orchestrator.instance_manager")

STATE_FILE = Path.home() / ".vast_rental_llm_state.json"


def _load_persisted_state() -> Dict[str, Any]:
    try:
        if STATE_FILE.is_file():
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save_persisted_state(instance_id: int, api_token: str):
    try:
        STATE_FILE.write_text(
            json.dumps({"instance_id": instance_id, "api_token": api_token}),
            encoding="utf-8"
        )
    except Exception:
        pass


def _clear_persisted_state():
    try:
        if STATE_FILE.is_file():
            STATE_FILE.unlink()
    except Exception:
        pass


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
                    host_ip, host_port = self.endpoint
                    if self._check_health(host_ip, host_port, self.api_token):
                        self.touch()
                        return self.endpoint, self.api_token
                    logger.warning("Existing READY instance %s:%s is no longer healthy/responding. Re-provisioning...", host_ip, host_port)
                    self.state = InstanceState.STOPPED
                    self.endpoint = None
                    self.api_token = None
                    if self.instance_id:
                        try:
                            self.client.destroy_instance(self.instance_id)
                        except Exception:
                            pass
                        self.instance_id = None

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
        """
        Detects any already installed/running instance to reuse it without downloading
        weights again, or launches a new cost-optimal instance from Vast marketplace.
        """
        persisted = _load_persisted_state()
        saved_id = persisted.get("instance_id")
        saved_token = persisted.get("api_token")

        # 1. Check if an instance with our image or saved ID already exists on account
        reusable_inst = None
        try:
            instances = self.client.list_instances()
            logger.info("Found %d existing instance(s) on Vast account.", len(instances))
            for inst in instances:
                inst_id = inst.get("id")
                inst_dump = json.dumps(inst).lower()
                raw_status = inst.get("actual_status") or inst.get("cur_state")
                status = str(raw_status).lower() if isinstance(raw_status, str) else ""
                raw_msg = inst.get("status_msg")
                status_msg = str(raw_msg).lower() if isinstance(raw_msg, str) else ""

                logger.info(
                    "Inspecting instance ID %s: status=%s, status_msg=%s",
                    inst_id,
                    status,
                    raw_msg
                )

                # Match by image name, saved instance ID, or sole instance on account
                is_our_image = ("rental-llm" in inst_dump) or (self.config.docker_image.lower() in inst_dump)
                is_saved = (saved_id is not None and inst_id == saved_id)
                is_only = (len(instances) == 1)

                if (is_our_image or is_saved or is_only) and "outbid" not in status_msg and status != "outbid":
                    reusable_inst = inst
                    logger.info("Selected instance ID %s for reuse!", inst_id)
                    break
        except Exception as check_err:
            logger.debug("Failed listing instances during reuse check: %s", check_err)

        if reusable_inst:
            inst_id = int(reusable_inst["id"])
            raw_status = reusable_inst.get("actual_status") or reusable_inst.get("cur_state")
            status = str(raw_status).lower() if isinstance(raw_status, str) else ""
            logger.info("Reusing existing instance ID %s on Vast.ai (current status: %s).", inst_id, status)

            env_dict = reusable_inst.get("env") if isinstance(reusable_inst.get("env"), dict) else {}
            token = saved_token or env_dict.get("API_TOKEN") or secrets.token_urlsafe(32)

            # If stopped or exited, start it up to preserve downloaded weights
            if status in ("stopped", "exited", "offline"):
                logger.info("Starting stopped instance %s to reuse existing downloaded weights...", inst_id)
                self.client.start_instance(inst_id)

            with self._lock:
                self.instance_id = inst_id

            try:
                endpoint, token = self._await_instance_ready(inst_id, token, timeout)
                _save_persisted_state(inst_id, token)
                logger.info("Successfully reused existing instance %s! Model download skipped.", inst_id)
                return endpoint, token
            except Exception as reuse_err:
                logger.warning(
                    "Failed to reactivate existing instance %s (%s). Falling back to marketplace...",
                    inst_id,
                    reuse_err
                )
                try:
                    self.client.destroy_instance(inst_id)
                except Exception:
                    pass
                _clear_persisted_state()
                with self._lock:
                    self.instance_id = None

        # 2. Clean up any unknown lingering active instances
        try:
            for inst in self.client.list_instances():
                raw_st = inst.get("actual_status") or inst.get("cur_state")
                status = str(raw_st).lower() if isinstance(raw_st, str) else ""
                if status in ("running", "loading", "created", "scheduling", "pending"):
                    self.client.destroy_instance(inst.get("id"))
        except Exception:
            pass

        # 3. Search offers and rent fresh GPU
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
        if self.config.host_criteria.spot_instances:
            min_bid = float(best_offer.get("min_bid") or best_offer.get("dph_total") or 0.15)
            bid_price = round(min_bid * (1.0 + self.config.host_criteria.bid_buffer_ratio), 4)
            logger.info("Bidding $%.4f/hr on spot instance (min_bid: $%.4f/hr)", bid_price, min_bid)
        else:
            bid_price = None
            logger.info("Launching on-demand instance at $%.4f/hr", float(best_offer.get("dph_total") or 0.20))

        instance_id = self.client.launch_instance(
            offer_id=offer_id,
            image=self.config.docker_image,
            disk_gb=self.config.host_criteria.min_disk_gb,
            env=env_vars,
            container_port=self.config.container_port,
            bid_price=bid_price,
        )

        with self._lock:
            self.instance_id = instance_id

        endpoint, ephemeral_token = self._await_instance_ready(instance_id, ephemeral_token, timeout)
        _save_persisted_state(instance_id, ephemeral_token)
        return endpoint, ephemeral_token

    def _await_instance_ready(
        self,
        instance_id: int,
        token: str,
        timeout: float
    ) -> Tuple[Tuple[str, int], str]:
        """Poll until instance is running, dynamic port is mapped, and health check passes."""
        start_poll = time.time()
        last_progress_log = time.time()
        stopped_since: Optional[float] = None
        logger.info("Awaiting container boot and dynamic port mapping for instance %s...", instance_id)

        while time.time() - start_poll < timeout:
            inst = self.client.get_instance(instance_id)
            if inst and isinstance(inst, dict):
                raw_status = inst.get("actual_status") or inst.get("cur_state")
                status = str(raw_status).lower() if isinstance(raw_status, str) else ""
                raw_msg = inst.get("status_msg")
                status_msg = str(raw_msg).lower() if isinstance(raw_msg, str) else ""
                raw_intended = inst.get("intended_status")
                intended_status = str(raw_intended).lower() if isinstance(raw_intended, str) else ""

                if "outbid" in status_msg or status == "outbid":
                    raise VastClientError(
                        f"Instance {instance_id} was outbid on the marketplace ({raw_msg or status})."
                    )

                if status == "unloaded" or (intended_status == "stopped" and time.time() - start_poll > 30):
                    raise VastClientError(
                        f"Instance {instance_id} was terminated or unloaded by host ({raw_msg or status})."
                    )

                if status in ("stopped", "offline"):
                    if stopped_since is None:
                        stopped_since = time.time()
                    elif time.time() - stopped_since > 60 and time.time() - start_poll > 60:
                        raise VastClientError(
                            f"Instance {instance_id} failed to boot (stuck in {status} for >60s, msg: {raw_msg})."
                        )
                else:
                    stopped_since = None

            endpoint = self.client.resolve_dynamic_endpoint(
                instance_id=instance_id,
                container_port=self.config.container_port
            )
            if endpoint:
                host_ip, host_port = endpoint
                if self._check_health(host_ip, host_port, token):
                    logger.info("Inference endpoint is live and verified healthy!")
                    return (host_ip, host_port), token

                if time.time() - last_progress_log >= 15:
                    elapsed = int(time.time() - start_poll)
                    logger.info(
                        "Dynamic port is %s:%s. Container initializing (downloading weights / loading model, elapsed: %ds)...",
                        host_ip,
                        host_port,
                        elapsed
                    )
                    last_progress_log = time.time()
            else:
                if time.time() - last_progress_log >= 15:
                    elapsed = int(time.time() - start_poll)
                    logger.info(
                        "Instance %s is starting up on Vast host (elapsed: %ds)...",
                        instance_id,
                        elapsed
                    )
                    last_progress_log = time.time()

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
        """Clean up the active instance on idle timeout or shutdown."""
        with self._state_condition:
            if self.state in (InstanceState.STOPPED, InstanceState.DESTROYING):
                return
            self.state = InstanceState.DESTROYING
            inst_to_manage = self.instance_id
            self._state_condition.notify_all()

        if inst_to_manage:
            if self.config.idle_action == "stop":
                logger.info("Stopping Vast.ai instance %s (preserving weights on disk)...", inst_to_manage)
                try:
                    self.client.stop_instance(inst_to_manage)
                except Exception as e:
                    logger.error("Error stopping instance %s: %s", inst_to_manage, e)
            else:
                logger.info("Destroying Vast.ai instance %s...", inst_to_manage)
                try:
                    self.client.destroy_instance(inst_to_manage)
                    _clear_persisted_state()
                except Exception as e:
                    logger.error("Error destroying instance %s: %s", inst_to_manage, e)

        with self._state_condition:
            self.state = InstanceState.STOPPED
            self.instance_id = None
            self.endpoint = None
            self.api_token = None
            self.last_active_time = 0.0
            self._state_condition.notify_all()

    def mark_failed(self):
        """Mark current instance as stopped/failed so subsequent requests re-provision."""
        with self._state_condition:
            if self.state in (InstanceState.READY, InstanceState.PROVISIONING):
                logger.warning("Marking instance as failed and triggering re-provision.")
                inst_to_destroy = self.instance_id
                self.state = InstanceState.STOPPED
                self.instance_id = None
                self.endpoint = None
                self.api_token = None
                self._state_condition.notify_all()
                if inst_to_destroy:
                    try:
                        self.client.destroy_instance(inst_to_destroy)
                    except Exception:
                        pass
                _clear_persisted_state()

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
