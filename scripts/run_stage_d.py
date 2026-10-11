#!/usr/bin/env python3
"""
Stage D Execution Script:
Provisions an approved Vast.ai GPU instance, waits for remote model download & LocalAI boot,
validates the endpoint independently, runs the academic benchmark suite, verifies local router integration,
and records all performance and cost metrics.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from orchestrator.config import OrchestratorConfig
from orchestrator.instance_manager import InstanceManager
from tests.benchmark_academic import AcademicBenchmark

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s"
)
logger = logging.getLogger("stage_d_runner")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Stage D Runner")
    parser.add_argument("--mode", choices=["on-demand", "spot"], default="on-demand", help="Rental mode (default: on-demand to prevent preemption)")
    args = parser.parse_args()

    logger.info("==================================================")
    logger.info(" Starting Stage D: Remote Deployment & Validation")
    logger.info("==================================================")

    config = OrchestratorConfig()
    if args.mode == "on-demand":
        config.host_criteria.spot_instances = False
    else:
        config.host_criteria.spot_instances = True

    if not config.vast_api_key:
        logger.error("VAST_API_KEY is not configured! Check .env file.")
        sys.exit(1)

    logger.info("Configuration:")
    logger.info("  Rental Mode:  %s", "ON-DEMAND (Guaranteed, no outbidding)" if not config.host_criteria.spot_instances else "SPOT")
    logger.info("  Image:        %s", config.docker_image)
    logger.info("  Model:        %s", config.active_model.name)
    logger.info("  Context:      %d tokens", config.active_model.context_size)
    logger.info("  Min VRAM:     %.1f GB", config.host_criteria.min_vram_gb)
    logger.info("  Local Port:   %d", config.listen_port)

    manager = InstanceManager(config)

    start_time = time.time()
    try:
        logger.info("\n[STEP 1] Provisioning Vast.ai instance and waiting for readiness...")
        endpoint, api_token = manager.ensure_ready(timeout=600)
        boot_duration = time.time() - start_time

        logger.info("[STEP 1 SUCCESS] Instance is LIVE and ready!")
        logger.info("  Instance ID:    %s", manager.instance_id)
        logger.info("  Endpoint:       %s:%s", endpoint[0], endpoint[1])
        logger.info("  Boot Time:      %.1f seconds", boot_duration)

        # Get instance details for cost tracking
        inst_details = manager.client.get_instance(manager.instance_id) if manager.instance_id else {}
        dph = float(inst_details.get("dph_total") or inst_details.get("min_bid") or 0.25)
        gpu_name = inst_details.get("gpu_name", "Unknown GPU")
        logger.info("  Assigned GPU:   %s", gpu_name)
        logger.info("  Hourly Rate:    $%.4f/hr", dph)

        # STEP 2: Independent Endpoint Test
        logger.info("\n[STEP 2] Running Independent Endpoint Validation...")
        base_url = f"http://127.0.0.1:{config.listen_port}/v1"
        benchmark = AcademicBenchmark(
            base_url=base_url,
            model=config.active_model.name,
            api_key=api_token
        )

        logger.info("Testing endpoint directly on %s...", base_url)
        summary = benchmark.run_all()

        # STEP 3: Router Integration Test
        logger.info("\n[STEP 3] Testing Integration with Local AI Router (Port 5000)...")
        router_benchmark = AcademicBenchmark(
            base_url="http://127.0.0.1:5000/v1",
            model="vast-qwen",
            api_key="anything"
        )
        try:
            sample_msg = [
                {"role": "user", "content": "Briefly state what epistemic uncertainty is in one sentence."}
            ]
            resp_text, router_metrics = router_benchmark._send_request(sample_msg, stream=False, max_tokens=60)
            logger.info("Router response: %s", resp_text.strip())
            logger.info("Router request duration: %.2fs", router_metrics.get("duration_s", 0))
            router_success = bool(resp_text)
        except Exception as router_err:
            logger.warning("Router test encountered error: %s (Is ai-router running on :5000?)", router_err)
            router_success = False

        # Calculate session cost
        total_session_seconds = time.time() - start_time
        total_cost = (total_session_seconds / 3600.0) * dph

        logger.info("\n==================================================")
        logger.info(" STAGE D VALIDATION SUMMARY")
        logger.info("==================================================")
        logger.info("Instance ID:       %s", manager.instance_id)
        logger.info("GPU Model:         %s", gpu_name)
        logger.info("Hourly Rate:       $%.4f/hr", dph)
        logger.info("Session Duration:  %.1f seconds (%.2f minutes)", total_session_seconds, total_session_seconds / 60.0)
        logger.info("Total Cost:        $%.4f", total_cost)
        logger.info("Benchmark Result:  %d/%d tests passed (%s)", summary["passed_tests"], summary["total_tests"], summary["success_rate"])
        logger.info("Router Route:      %s", "PASS" if router_success else "SKIPPED/UNREACHABLE")

        # Save results to JSON
        output_report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "instance_id": manager.instance_id,
            "gpu_name": gpu_name,
            "hourly_rate": dph,
            "session_duration_s": round(total_session_seconds, 1),
            "session_cost_usd": round(total_cost, 4),
            "benchmark_summary": summary,
            "router_integration_passed": router_success
        }
        with open("stage_d_results.json", "w", encoding="utf-8") as f:
            json.dump(output_report, f, indent=2)
        logger.info("Report saved to stage_d_results.json")

    except KeyboardInterrupt:
        logger.info("\nUser requested interrupt. Cleaning up...")
    except Exception as e:
        logger.error("Stage D execution failed: %s", e, exc_info=True)
    finally:
        logger.info("\n[CLEANUP] Tearing down ephemeral instance to ensure $0 lingering cost...")
        manager.stop_reaper()
        manager.destroy()
        logger.info("[CLEANUP COMPLETE] Instance destroyed. Active billing terminated.")


if __name__ == "__main__":
    main()
