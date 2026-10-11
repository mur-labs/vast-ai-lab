#!/usr/bin/env bash
# ==============================================================================
# SSH Tunnel Manager for Vast.ai Remote LLM Access
# Securely forwards local port 8088 to remote LocalAI port 8080 over SSH.
# ==============================================================================
set -euo pipefail

PID_FILE="/tmp/vast_ssh_tunnel.pid"
DEFAULT_LOCAL_PORT="8088"
DEFAULT_REMOTE_PORT="8080"

usage() {
    echo "Usage:"
    echo "  $0 start <ssh_host> <ssh_port> [local_port] [identity_file]"
    echo "  $0 stop"
    echo "  $0 status"
    echo ""
    echo "Examples:"
    echo "  $0 start ssh4.vast.ai 34125"
    echo "  $0 start ssh4.vast.ai 34125 8088 ~/.ssh/id_ed25519"
    exit 1
}

start_tunnel() {
    local host="${1:-}"
    local port="${2:-}"
    local local_port="${3:-$DEFAULT_LOCAL_PORT}"
    local id_file="${4:-}"

    if [ -z "$host" ] || [ -z "$port" ]; then
        usage
    fi

    if [ -f "$PID_FILE" ]; then
        local existing_pid
        existing_pid=$(cat "$PID_FILE" 2>/dev/null || true)
        if [ -n "$existing_pid" ] && kill -0 "$existing_pid" 2>/dev/null; then
            echo "[TUNNEL] Tunnel already running with PID ${existing_pid}."
            return 0
        fi
    fi

    local ssh_opts=(
        -N
        -f
        -L "${local_port}:localhost:${DEFAULT_REMOTE_PORT}"
        -p "${port}"
        -o "ExitOnForwardFailure=yes"
        -o "ServerAliveInterval=30"
        -o "ServerAliveCountMax=3"
        -o "StrictHostKeyChecking=no"
        -o "UserKnownHostsFile=/dev/null"
    )

    if [ -n "$id_file" ] && [ -f "$id_file" ]; then
        ssh_opts+=(-i "$id_file")
    fi

    echo "[TUNNEL] Establishing secure SSH tunnel..."
    echo "  Local endpoint:  http://127.0.0.1:${local_port}"
    echo "  Remote instance: root@${host}:${port}"

    ssh "${ssh_opts[@]}" "root@${host}"

    # Find the launched SSH PID
    local pid
    pid=$(pgrep -f "ssh.*-L ${local_port}:localhost:${DEFAULT_REMOTE_PORT}" | head -n 1 || true)
    if [ -n "$pid" ]; then
        echo "$pid" > "$PID_FILE"
        echo "[TUNNEL] Connected successfully (PID: ${pid})."
    else
        echo "[TUNNEL] Started, but PID detection fell back to port check."
    fi

    # Verify connectivity
    echo "[TUNNEL] Waiting for remote API readiness on http://127.0.0.1:${local_port}..."
    local ready=0
    for _ in {1..10}; do
        if curl -s -f -m 2 "http://127.0.0.1:${local_port}/v1/models" >/dev/null 2>&1; then
            ready=1
            break
        fi
        sleep 1
    done

    if [ "$ready" -eq 1 ]; then
        echo "[TUNNEL] Connection verified! LiteLLM can now reach http://127.0.0.1:${local_port}/v1"
    else
        echo "[TUNNEL] Tunnel active, but remote LocalAI server is still loading or requires authorization token."
    fi
}

stop_tunnel() {
    if [ -f "$PID_FILE" ]; then
        local pid
        pid=$(cat "$PID_FILE" 2>/dev/null || true)
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            echo "[TUNNEL] Stopping SSH tunnel PID ${pid}..."
            kill "$pid" 2>/dev/null || true
            rm -f "$PID_FILE"
            echo "[TUNNEL] Tunnel stopped."
            return 0
        fi
        rm -f "$PID_FILE"
    fi

    # Fallback to pkill
    pkill -f "ssh.*-L ${DEFAULT_LOCAL_PORT}:localhost" 2>/dev/null && echo "[TUNNEL] Stopped existing tunnels." || echo "[TUNNEL] No active tunnel found."
}

status_tunnel() {
    if [ -f "$PID_FILE" ]; then
        local pid
        pid=$(cat "$PID_FILE" 2>/dev/null || true)
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            echo "[TUNNEL] Active (PID: ${pid})"
            return 0
        fi
    fi
    echo "[TUNNEL] Inactive."
}

ACTION="${1:-}"
shift || usage

case "$ACTION" in
    start)
        start_tunnel "$@"
        ;;
    stop)
        stop_tunnel
        ;;
    status)
        status_tunnel
        ;;
    *)
        usage
        ;;
esac
