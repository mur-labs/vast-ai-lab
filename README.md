# Rental LLM's — Vast.ai Lab

Custom on-demand `llama.cpp` inference container and intelligent Python Orchestrator for academic research using Vast.ai GPU spot instances.

## Architecture

$$\text{OpenCode} \longrightarrow \text{Local Router (Port 5000)} \longrightarrow \text{LiteLLM (Port 4000)} \longrightarrow \text{Python Orchestrator (Port 8088)} \overset{\text{dynamic port}}{\longrightarrow} \text{Vast.ai GPU Instance}$$

---

## Components

### 1. Custom Inference Container
* **[`Dockerfile`](file:///home/carol/vast-ai-lab/Dockerfile)**: Base image `ghcr.io/ggml-org/llama.cpp:server-cuda` with `aria2`, `curl`, and `ca-certificates`.
* **[`entrypoint.sh`](file:///home/carol/vast-ai-lab/entrypoint.sh)**:
  * Enforces `MODEL_URL` and `API_TOKEN` environment variables.
  * Downloads model weights on boot into `/tmp` (ephemeral disk) using `aria2c` (16 concurrent connections).
  * Launches `llama-server` binding to `0.0.0.0:8000` with Bearer token authentication via `--api-key`.

### 2. Python Orchestrator (`orchestrator/`)
* **[`orchestrator/config.py`](file:///home/carol/vast-ai-lab/orchestrator/config.py)**: Defines hardware criteria ($\ge 20$ GB VRAM, $\ge 40$ GB disk, $\ge 500$ Mbps download, $\ge 97\%$ reliability), initial model (`Qwen2.5-14B-Instruct` Q4_K_M), and 10-minute idle destruction timeout.
* **[`orchestrator/vast_client.py`](file:///home/carol/vast-ai-lab/orchestrator/vast_client.py)**: REST API client for Vast.ai API v0. Filters offers by lowest total task cost, rents spot instances with ephemeral credentials, queries status, resolves dynamic host ports, and destroys instances.
* **[`orchestrator/instance_manager.py`](file:///home/carol/vast-ai-lab/orchestrator/instance_manager.py)**: Thread-safe state machine (`STOPPED`, `PROVISIONING`, `READY`, `DESTROYING`). Implements concurrency deduplication against LiteLLM retry storms, generates zero-trust ephemeral tokens, conducts readiness health pings, and operates a background idle reaper.
* **[`orchestrator/proxy_server.py`](file:///home/carol/vast-ai-lab/orchestrator/proxy_server.py)**: OpenAI-compatible reverse proxy that intercepts LiteLLM requests, lazily triggers GPU provisioning, and forwards requests to the remote dynamic endpoint.
* **[`orchestrator/main.py`](file:///home/carol/vast-ai-lab/orchestrator/main.py)**: CLI entrypoint.

---

## Test Suite

All components are tested locally using Python standard library without requiring a local Docker daemon or active Vast.ai billing.

```bash
python3 -m unittest discover -s tests -v
```

### Test Coverage
* `tests/test_entrypoint.py`: Subprocess testing of environment validation, download handling, Hugging Face token passing, and `llama-server` CLI forwarding.
* `tests/test_vast_client.py`: API request structure, minimum hardware filtering, cost sorting, dynamic port resolution, and destruction calls.
* `tests/test_instance_manager.py`: Concurrent request deduplication (preventing duplicate GPU instances) and 10-minute idle auto-destruction.
* `tests/test_proxy_server.py`: Lazy provisioning, proxy routing, Authorization token injection, and `/status` monitoring.
