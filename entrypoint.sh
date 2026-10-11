#!/usr/bin/env bash
set -euo pipefail

echo "=================================================="
echo " Starting Rental LLM Inference Container"
echo "=================================================="

# 1. Validate required environment variables
if [ -z "${MODEL_URL:-}" ]; then
    echo "ERROR: MODEL_URL environment variable is required." >&2
    exit 1
fi

if [ -z "${API_TOKEN:-}" ]; then
    echo "ERROR: API_TOKEN environment variable is required." >&2
    exit 1
fi

# 2. Setup paths and default configurations
MODEL_DIR="${MODEL_DIR:-/tmp}"
mkdir -p "$MODEL_DIR"
MODEL_FILE="${MODEL_DIR}/${MODEL_FILENAME:-model.gguf}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
N_GPU_LAYERS="${N_GPU_LAYERS:-99}"
CTX_SIZE="${CTX_SIZE:-4096}"

# 3. Locate llama-server binary
SERVER_BIN=""
if [ -n "${LLAMA_BIN:-}" ] && command -v "$LLAMA_BIN" >/dev/null 2>&1; then
    SERVER_BIN="$LLAMA_BIN"
elif command -v llama-server >/dev/null 2>&1; then
    SERVER_BIN="llama-server"
elif [ -x /app/llama-server ]; then
    SERVER_BIN="/app/llama-server"
elif [ -x /usr/local/bin/llama-server ]; then
    SERVER_BIN="/usr/local/bin/llama-server"
else
    echo "ERROR: llama-server executable not found in PATH, /app, or /usr/local/bin." >&2
    exit 1
fi

# 4. Download model weights
echo "Downloading model weights..."
echo "  Source URL: ${MODEL_URL}"
echo "  Destination: ${MODEL_FILE}"

DOWNLOAD_HEADER=()
if [ -n "${HF_TOKEN:-}" ]; then
    DOWNLOAD_HEADER=(-H "Authorization: Bearer ${HF_TOKEN}")
fi

if command -v aria2c >/dev/null 2>&1; then
    echo "Using aria2c for fast multi-connection download..."
    ARIA2_HEADER_ARG=()
    if [ -n "${HF_TOKEN:-}" ]; then
        ARIA2_HEADER_ARG=(--header="Authorization: Bearer ${HF_TOKEN}")
    fi
    aria2c -x 16 -s 16 -k 1M --continue=true \
        "${ARIA2_HEADER_ARG[@]}" \
        --dir="$MODEL_DIR" \
        --out="$(basename "$MODEL_FILE")" \
        "$MODEL_URL"
elif command -v curl >/dev/null 2>&1; then
    echo "Using curl for download..."
    curl -fSL --retry 3 --retry-delay 2 \
        "${DOWNLOAD_HEADER[@]}" \
        -o "$MODEL_FILE" \
        "$MODEL_URL"
elif command -v wget >/dev/null 2>&1; then
    echo "Using wget for download..."
    WGET_HEADER_ARG=()
    if [ -n "${HF_TOKEN:-}" ]; then
        WGET_HEADER_ARG=(--header="Authorization: Bearer ${HF_TOKEN}")
    fi
    wget -c "${WGET_HEADER_ARG[@]}" -O "$MODEL_FILE" "$MODEL_URL"
else
    echo "ERROR: Neither aria2c, curl, nor wget is available for downloading." >&2
    exit 1
fi

if [ ! -s "$MODEL_FILE" ]; then
    echo "ERROR: Downloaded model file is missing or empty at ${MODEL_FILE}." >&2
    exit 1
fi

echo "Model weights successfully verified at ${MODEL_FILE}."

# 5. Launch llama-server with security token and parameters
echo "Starting inference server on ${HOST}:${PORT}..."
echo "Security token is configured for authorization."

exec "$SERVER_BIN" \
    --model "$MODEL_FILE" \
    --api-key "$API_TOKEN" \
    --host "$HOST" \
    --port "$PORT" \
    -ngl "$N_GPU_LAYERS" \
    -c "$CTX_SIZE" \
    "$@"
