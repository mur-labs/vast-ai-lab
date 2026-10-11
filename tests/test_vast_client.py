import json
import unittest
from unittest.mock import MagicMock, patch

from orchestrator.config import HostCriteria
from orchestrator.vast_client import VastClient, VastClientError


class TestVastClient(unittest.TestCase):
    def setUp(self):
        self.client = VastClient(api_key="test-vast-api-key")

    @patch("orchestrator.vast_client.urllib.request.urlopen")
    def test_search_offers_criteria_and_sorting(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "offers": [
                {
                    "id": 101,
                    "gpu_name": "RTX 3090",
                    "gpu_ram": 24576,
                    "dph_total": 0.25,
                    "min_bid": 0.25,
                },
                {
                    "id": 102,
                    "gpu_name": "RTX 4090",
                    "gpu_ram": 24576,
                    "dph_total": 0.18,
                    "min_bid": 0.18,
                },
                {
                    "id": 103,
                    "gpu_name": "A5000",
                    "gpu_ram": 24576,
                    "dph_total": 0.35,
                    "min_bid": 0.35,
                }
            ]
        }).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        criteria = HostCriteria(
            min_vram_gb=20.0,
            min_disk_gb=40.0,
            min_inet_down_mbps=500.0,
            min_reliability=0.97,
            spot_instances=True
        )

        offers = self.client.search_offers(criteria)

        # Check that request sent criteria
        self.assertEqual(len(offers), 3)
        # Check sorted by minimum total cost (102 @ 0.18 should be first)
        self.assertEqual(offers[0]["id"], 102)
        self.assertEqual(offers[1]["id"], 101)
        self.assertEqual(offers[2]["id"], 103)

    @patch("orchestrator.vast_client.urllib.request.urlopen")
    def test_launch_instance(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "success": True,
            "new_contract": 556677
        }).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        instance_id = self.client.launch_instance(
            offer_id=102,
            image="rental-llm/llama-server:latest",
            disk_gb=40,
            env={"MODEL_URL": "http://model", "API_TOKEN": "tok123"},
            container_port=8000
        )

        self.assertEqual(instance_id, 556677)

    @patch("orchestrator.vast_client.urllib.request.urlopen")
    def test_launch_instance_with_success_false_and_new_contract(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "success": False,
            "new_contract": 55420477,
            "instance_api_key": "some_key"
        }).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        instance_id = self.client.launch_instance(
            offer_id=36757236,
            image="ghcr.io/mur-labs/rental-llm:latest",
            disk_gb=40,
            env={"MODEL_URL": "http://model", "API_TOKEN": "tok123"},
            container_port=8000
        )

        self.assertEqual(instance_id, 55420477)

    @patch("orchestrator.vast_client.urllib.request.urlopen")
    def test_resolve_dynamic_endpoint(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "instances": [
                {
                    "id": 556677,
                    "actual_status": "running",
                    "public_ipaddr": "198.51.100.42",
                    "ports": {
                        "8000/tcp": [{"HostIp": "0.0.0.0", "HostPort": "32145"}]
                    }
                }
            ]
        }).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        endpoint = self.client.resolve_dynamic_endpoint(556677, container_port=8000)
        self.assertIsNotNone(endpoint)
        self.assertEqual(endpoint, ("198.51.100.42", 32145))

    @patch("orchestrator.vast_client.urllib.request.urlopen")
    def test_resolve_dynamic_endpoint_pending(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "instances": [
                {
                    "id": 556677,
                    "actual_status": "loading",
                    "public_ipaddr": "198.51.100.42",
                    "ports": {}
                }
            ]
        }).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        endpoint = self.client.resolve_dynamic_endpoint(556677, container_port=8000)
        self.assertIsNone(endpoint)

    @patch("orchestrator.vast_client.urllib.request.urlopen")
    def test_destroy_instance(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"success": True}).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        success = self.client.destroy_instance(556677)
        self.assertTrue(success)


if __name__ == "__main__":
    unittest.main()
