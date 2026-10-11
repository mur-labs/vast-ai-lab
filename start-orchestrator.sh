#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

# Terminate any previous orchestrator process holding port 8088
fuser -k 8088/tcp 2>/dev/null || true
sleep 1

echo "Starting Rental LLM Orchestrator from: ${SCRIPT_DIR}"
exec python3 "${SCRIPT_DIR}/orchestrator/main.py" "$@"
