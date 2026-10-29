#!/usr/bin/env bash
# Utility launcher for real-time demonstration
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

if [ -f "$DIR/../venv/bin/activate" ]; then
    source "$DIR/../venv/bin/activate"
elif [ -f "$HOME/ns-allinone-3.40/ns-3.40/venv/bin/activate" ]; then
    source "$HOME/ns-allinone-3.40/ns-3.40/venv/bin/activate"
elif [ -f "$DIR/venv/bin/activate" ]; then
    source "$DIR/venv/bin/activate"
fi

python3 "$DIR/realtime_demo.py" "$@"
