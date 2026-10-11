#!/usr/bin/env bash
# ==============================================================================
# Remote Bootstrap Script for LocalAI on Vast.ai GPU Instances
# ==============================================================================
set -euo pipefail

echo "=================================================="
echo " Starting Remote-First LocalAI Academic LLM Boot"
echo "=================================================="

# 1. Environment & Path Defaults
export MODELS_PATH="${MODELS_PATH:-/models}"
mkdir -p "$MODELS_PATH"

MODEL_NAME="${MODEL_NAME:-qwen2.5-14b-instruct}"
MODEL_FILENAME="${MODEL_FILENAME:-qwen2.5-14b-instruct-q4_k_m.gguf}"
MODEL_URL="${MODEL_URL:-https://huggingface.co/bartowski/Qwen2.5-14B-Instruct-GGUF/resolve/main/Qwen2.5-14B-Instruct-Q4_K_M.gguf}"
MODEL_FILE="${MODELS_PATH}/${MODEL_FILENAME}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
export ADDRESS="${HOST}:${PORT}"

# Ephemeral bearer token configuration
API_TOKEN="${API_TOKEN:-${LOCALAI_API_KEY:-}}"
if [ -n "$API_TOKEN" ]; then
    export LOCALAI_API_KEY="$API_TOKEN"
    echo "[SECURITY] Ephemeral API key configured for LocalAI."
else
    echo "[WARNING] No API_TOKEN or LOCALAI_API_KEY provided."
fi

# 2. Check NVIDIA GPU Visibility & Drivers
echo "[HARDWARE] Inspecting GPU..."
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo "[WARNING] nvidia-smi not found. CUDA may not be available!"
fi

# 3. Model Weights Download (Remote Instance Direct Download)
if [ -s "$MODEL_FILE" ]; then
    echo "[CACHE] Model file exists at ${MODEL_FILE} ($(du -h "$MODEL_FILE" | cut -f1)). Skipping download."
else
    echo "[DOWNLOAD] Fetching model weights directly from repository..."
    echo "  URL:  ${MODEL_URL}"
    echo "  Dest: ${MODEL_FILE}"

    START_TIME=$(date +%s)

    if command -v aria2c >/dev/null 2>&1; then
        echo "[DOWNLOAD] Using aria2c with 16 connections..."
        aria2c -x 16 -s 16 -k 1M --continue=true \
            --dir="$MODELS_PATH" \
            --out="$MODEL_FILENAME" \
            "$MODEL_URL"
    elif command -v curl >/dev/null 2>&1; then
        echo "[DOWNLOAD] Using curl..."
        curl -fSL --retry 3 --retry-delay 2 \
            -o "$MODEL_FILE" \
            "$MODEL_URL"
    elif command -v wget >/dev/null 2>&1; then
        echo "[DOWNLOAD] Using wget..."
        wget -c -O "$MODEL_FILE" "$MODEL_URL"
    else
        echo "[ERROR] Neither aria2c, curl, nor wget available for model download." >&2
        exit 1
    fi

    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    FILE_SIZE=$(du -h "$MODEL_FILE" | cut -f1)
    echo "[DOWNLOAD] Completed in ${DURATION}s. Total size: ${FILE_SIZE}."
fi

# Validate downloaded file size (> 1GB)
ACTUAL_BYTES=$(stat -c%s "$MODEL_FILE" 2>/dev/null || stat -f%z "$MODEL_FILE")
if [ "$ACTUAL_BYTES" -lt 1000000000 ]; then
    echo "[ERROR] Downloaded model is suspiciously small (${ACTUAL_BYTES} bytes). Aborting!" >&2
    exit 1
fi

# 4. Generate LocalAI Model YAML Configuration
CONFIG_FILE="${MODELS_PATH}/${MODEL_NAME}.yaml"
echo "[CONFIG] Writing model configuration to ${CONFIG_FILE}..."
cat <<EOF > "$CONFIG_FILE"
name: ${MODEL_NAME}
backend: llama-cpp
parameters:
  model: ${MODEL_FILE}
  context_size: ${CTX_SIZE:-32768}
  gpu_layers: ${N_GPU_LAYERS:-99}
  threads: ${THREADS:-8}
  f16: true
  mmap: true
  batch: 512

template:
  chat_message: |
    <|im_start|>{{.Role}}
    {{.Content}}<|im_end|>
  chat: |
    {{.Input}}
    <|im_start|>assistant
  completion: |
    {{.Input}}
EOF

# 5. Ensure llama-cpp backend is staged if local-ai command exists
if command -v local-ai >/dev/null 2>&1; then
    echo "[LOCALAI] Checking backend availability..."
    local-ai backends install llama-cpp || true
fi

# 6. Start LocalAI Server
echo "=================================================="
echo " Launching LocalAI on ${ADDRESS}"
echo "=================================================="

# Locate local-ai executable
LOCALAI_BIN=""
if command -v local-ai >/dev/null 2>&1; then
    LOCALAI_BIN="local-ai"
elif [ -x /usr/bin/local-ai ]; then
    LOCALAI_BIN="/usr/bin/local-ai"
elif [ -x /build/local-ai ]; then
    LOCALAI_BIN="/build/local-ai"
fi

if [ -n "$LOCALAI_BIN" ]; then
    exec "$LOCALAI_BIN" run --models-path "$MODELS_PATH" --address "$ADDRESS" "$@"
else
    echo "[ERROR] local-ai binary not found in PATH, /usr/bin, or /build." >&2
    exit 1
fi
