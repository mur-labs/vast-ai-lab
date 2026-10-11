# Project Rental LLM's

## 1. Project Objective
Build a personal, on-demand AI inference infrastructure for academic research using Vast.ai GPU spot instances and custom Docker images. The system provisions infrastructure only when requests arrive, prioritizes minimum total task cost over lowest hourly rate, and ensures instances are thoroughly destroyed after an idle period.

## 2. Architecture & Tech Stack
*   **Local Host (Linux Mint, Docker-free):** OpenCode $\rightarrow$ Local Router (Port 5000) $\rightarrow$ LiteLLM (Port 4000) $\rightarrow$ Python Orchestrator.
*   **Remote Environment (Vast.ai):** Custom Docker container running `llama.cpp` as the inference engine.
*   **Initial Model:** Qwen2.5-14B-Instruct (GGUF Q4_K_M).

## 3. Strict Development Constraints (Audit Findings)
*   **Dynamic Port Resolution:** Vast.ai does not provide standard port 80/443 mappings. It assigns random external ports to internal Docker ports. The Python Orchestrator must query the Vast API after container readiness to retrieve the mapping and dynamically update the local proxy path.
*   **No Persistent Storage:** Persistent volumes tie availability to a single unvetted host and incur hourly costs while stopped. Use ephemeral disk space and download the weights on container boot.
*   **Zero-Trust Security:** Do not pass permanent Vast API keys, SSH keys, or cloud credentials into the remote Docker environment. Secure the inference endpoint by passing a randomly generated ephemeral `API_TOKEN` as an environment variable at container launch.
*   **Host Selection Criteria:** The orchestrator must dynamically search for GPUs meeting these minimums: $\ge$ 20 GB VRAM, $\ge$ 40 GB disk, $\ge$ 500 Mbps download speed, and a host reliability score of $\ge$ 97%.
*   **Lifecycle Management:** The idle timeout before destruction must be 10 minutes to support academic reading workflows. The Orchestrator must track active instance states to prevent LiteLLM retries from spawning duplicate GPU instances.

## 4. MVP Roadmap & Current Status
*   **Phase 1 — Local Image Validation:** [COMPLETE]
    *   `Dockerfile` and `entrypoint.sh` created.
    *   Validated locally using standard Python subprocess tests.
*   **Phase 2 — Python Orchestrator & Dynamic Proxy:** [COMPLETE]
    *   `orchestrator/vast_client.py`: API v0 client with hardware filtering ($\ge$ 20GB VRAM, $\ge$ 40GB disk, $\ge$ 500 Mbps, $\ge$ 97% reliability), cost-optimal spot selection, and dynamic port resolution.
    *   `orchestrator/instance_manager.py`: Ephemeral token injection, state machine with retry deduplication, and 10-minute idle reaper.
    *   `orchestrator/proxy_server.py`: Local reverse proxy for LiteLLM with on-demand lazy provisioning.
    *   Full test suite in `tests/` passes (15 tests).
*   **Phase 3 — Image Delivery & End-to-End Integration:** [READY]
    *   Push custom container image to Docker Hub / GHCR registry.
    *   Connect LiteLLM route (`ai-router`) to the Orchestrator endpoint (`http://127.0.0.1:8088`).
