import json
import logging
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from .config import HostCriteria

logger = logging.getLogger("orchestrator.vast_client")


class VastClientError(Exception):
    """Base exception for Vast.ai API communication failures."""
    pass


class VastClient:
    """REST API Client for Vast.ai GPU Marketplace (API v0)."""

    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://console.vast.ai/api/v0"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Perform an HTTP request against the Vast.ai REST API."""
        if endpoint.startswith("http://") or endpoint.startswith("https://"):
            url = endpoint
        elif endpoint.startswith("/api/"):
            parsed = urllib.parse.urlparse(self.base_url)
            url = f"{parsed.scheme}://{parsed.netloc}{endpoint}"
        else:
            url = f"{self.base_url}/{endpoint.lstrip('/')}"

        query_params = dict(params or {})
        # Vast.ai supports api_key parameter or Authorization header
        query_params["api_key"] = self.api_key
        url = f"{url}?{urllib.parse.urlencode(query_params)}"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "RentalLLM-Orchestrator/1.0"
        }

        body_bytes = None
        if data is not None:
            body_bytes = json.dumps(data).encode("utf-8")

        req = urllib.request.Request(url, data=body_bytes, headers=headers, method=method)

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                resp_bytes = resp.read()
                if not resp_bytes:
                    return {}
                return json.loads(resp_bytes.decode("utf-8"))
        except urllib.error.HTTPError as e:
            err_msg = e.read().decode("utf-8", errors="replace")
            logger.error("Vast API error %s on %s: %s", e.code, endpoint, err_msg)
            raise VastClientError(f"Vast API HTTP {e.code} error: {err_msg}") from e
        except Exception as e:
            logger.error("Failed request to %s: %s", endpoint, e)
            raise VastClientError(f"Network error connecting to Vast API: {e}") from e

    def search_offers(self, criteria: HostCriteria) -> List[Dict[str, Any]]:
        """
        Search offers meeting strict requirements:
        - VRAM >= min_vram_gb (converted to MB: min_vram_gb * 1024)
        - Disk >= min_disk_gb
        - Download speed >= min_inet_down_mbps
        - Reliability >= min_reliability
        - Spot/interruptible offers prioritized for lowest task cost
        """
        query_payload: Dict[str, Any] = {
            "rentable": {"eq": True},
            "rented": {"eq": False},
            "verified": {"eq": True},
            "gpu_ram": {"gte": int(criteria.min_vram_gb * 1024)},
            "disk_space": {"gte": float(criteria.min_disk_gb)},
            "inet_down": {"gte": float(criteria.min_inet_down_mbps)},
            "reliability2": {"gte": float(criteria.min_reliability)},
            "type": "bid" if criteria.spot_instances else "on-demand",
        }

        res = self._request("POST", "bundles/", data=query_payload)
        offers = res.get("offers", []) if isinstance(res, dict) else []

        # If no spot/bid offers match, fallback to on-demand
        if not offers and criteria.spot_instances:
            logger.info("No spot/bid offers found. Trying on-demand offers...")
            query_payload["type"] = "on-demand"
            res = self._request("POST", "bundles/", data=query_payload)
            offers = res.get("offers", []) if isinstance(res, dict) else []

        # Sort offers prioritizing lowest total cost per hour (dph_total or dph_base)
        def cost_key(offer: Dict[str, Any]) -> float:
            return float(offer.get("dph_total") or offer.get("min_bid") or offer.get("dph_base") or 999.0)

        offers.sort(key=cost_key)
        return offers

    def launch_instance(
        self,
        offer_id: int,
        image: str,
        disk_gb: float,
        env: Dict[str, str],
        container_port: int = 8000,
        bid_price: Optional[float] = None,
        onstart: Optional[str] = None
    ) -> int:
        """
        Rent an offer and start a container. Returns the new instance ID.
        """
        env_copy = dict(env)
        # Port mapping in Vast.ai API is passed via env dictionary or Dockerfile EXPOSE
        env_copy[f"-p {container_port}:{container_port}"] = "1"

        payload: Dict[str, Any] = {
            "client_id": "me",
            "image": image,
            "disk": float(disk_gb),
            "env": env_copy,
            "runtype": "args",
        }
        if bid_price is not None:
            payload["price"] = float(bid_price)
        if onstart:
            payload["onstart"] = onstart

        res = self._request("PUT", f"asks/{offer_id}/", data=payload)
        new_contract = res.get("new_contract")

        # If Vast returned new_contract, the instance contract was created!
        if not new_contract and not res.get("success"):
            raise VastClientError(f"Failed to launch instance on offer {offer_id}: {res}")

        if not new_contract:
            raise VastClientError(f"No new_contract ID returned by Vast.ai: {res}")

        logger.info("Successfully requested instance creation: Contract ID %s", new_contract)
        return int(new_contract)

    def list_instances(self) -> List[Dict[str, Any]]:
        """List all active and pending instances for account using API v1."""
        res = self._request("GET", "/api/v1/instances/")
        if isinstance(res, dict) and "instances" in res:
            return res["instances"]
        if isinstance(res, list):
            return res
        return []

    def get_instance(self, instance_id: int) -> Optional[Dict[str, Any]]:
        """Get details for a specific instance ID."""
        instances = self.list_instances()
        for inst in instances:
            if inst.get("id") == instance_id:
                return inst
        return None

    def resolve_dynamic_endpoint(
        self,
        instance_id: int,
        container_port: int = 8000
    ) -> Optional[Tuple[str, int]]:
        """
        Resolve the external dynamic port mapping for an internal container port.
        Returns (host_ip, host_port) when mapped, or None if still pending.
        """
        inst = self.get_instance(instance_id)
        if not inst:
            return None

        # Check instance running state
        status = inst.get("actual_status") or inst.get("cur_state")
        if status != "running":
            return None

        host_ip = inst.get("public_ipaddr") or inst.get("ssh_host")
        if not host_ip:
            return None

        ports_dict = inst.get("ports", {})
        port_key = f"{container_port}/tcp"
        if port_key in ports_dict and ports_dict[port_key]:
            host_port_info = ports_dict[port_key][0]
            host_port = host_port_info.get("HostPort")
            if host_port:
                return str(host_ip), int(host_port)

        return None

    def start_instance(self, instance_id: int) -> bool:
        """Start an existing stopped Vast.ai instance."""
        logger.info("Starting existing Vast.ai instance ID %s", instance_id)
        try:
            res = self._request("PUT", f"instances/{instance_id}/", data={"state": "running"})
            return bool(res.get("success", True))
        except Exception as e:
            logger.error("Failed to start instance %s: %s", instance_id, e)
            return False

    def stop_instance(self, instance_id: int) -> bool:
        """Stop an instance without destroying container disk storage."""
        logger.info("Stopping Vast.ai instance ID %s", instance_id)
        try:
            res = self._request("PUT", f"instances/{instance_id}/", data={"state": "stopped"})
            return bool(res.get("success", True))
        except Exception as e:
            logger.error("Failed to stop instance %s: %s", instance_id, e)
            return False

    def destroy_instance(self, instance_id: int) -> bool:
        """Thoroughly destroy an instance to prevent lingering costs."""
        logger.info("Destroying Vast.ai instance ID %s", instance_id)
        try:
            res = self._request("DELETE", f"/api/v1/instances/{instance_id}/")
            return bool(res.get("success", True))
        except VastClientError:
            res = self._request("DELETE", f"instances/{instance_id}/")
            return bool(res.get("success", True))
