#!/usr/bin/env bash
# Usage: ./run_tf_opacus.sh [start|status|tail|stop]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

PID_FILE="$SCRIPT_DIR/tf_train_opacus.pid"
LATEST_LOG="$LOG_DIR/tf_train_opacus.latest.log"

cmd="${1:-start}"

is_running() {
    [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

case "$cmd" in
    start)
        if is_running; then
            echo "Already running (PID $(cat "$PID_FILE")). Log: $LATEST_LOG"
            exit 1
        fi
        ts="$(date +%Y%m%d_%H%M%S)"
        log="$LOG_DIR/tf_train_opacus_${ts}.log"
        nohup python3 -u tf_train_opacus.py >"$log" 2>&1 &
        pid=$!
        echo "$pid" >"$PID_FILE"
        ln -sf "$log" "$LATEST_LOG"
        echo "Started tf_train_opacus.py (PID $pid)"
        echo "Log: $log"
        echo "Tail with: $0 tail"
        ;;
    status)
        if is_running; then
            echo "Running (PID $(cat "$PID_FILE")). Log: $LATEST_LOG"
        else
            echo "Not running."
            [[ -f "$PID_FILE" ]] && rm -f "$PID_FILE"
        fi
        ;;
    tail)
        [[ -f "$LATEST_LOG" ]] || { echo "No log yet."; exit 1; }
        tail -f "$LATEST_LOG"
        ;;
    stop)
        if is_running; then
            pid="$(cat "$PID_FILE")"
            kill "$pid"
            rm -f "$PID_FILE"
            echo "Stopped PID $pid."
        else
            echo "Not running."
            [[ -f "$PID_FILE" ]] && rm -f "$PID_FILE"
        fi
        ;;
    *)
        echo "Usage: $0 {start|status|tail|stop}" >&2
        exit 2
        ;;
esac
