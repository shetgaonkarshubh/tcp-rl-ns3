#!/usr/bin/env bash
# Start the comparison dashboard at http://localhost:8000
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$DIR")"
PORT="${PORT:-8000}"
PY="$REPO/venv/bin/python"
[ -x "$PY" ] || PY="python3"

( sleep 1.5; open "http://localhost:$PORT" 2>/dev/null || xdg-open "http://localhost:$PORT" 2>/dev/null || true ) &
cd "$DIR"
exec "$PY" -m uvicorn server:app --port "$PORT"
