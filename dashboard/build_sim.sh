#!/usr/bin/env bash
# Rebuild the ns-3 rl-tcp binary from this repo's rl-tcp.cc (macOS / Homebrew setup).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NS3_DIR="${NS3_DIR:-$REPO/../ns-allinone-3.40/ns-3.40}"
TARGET="$NS3_DIR/contrib/opengym/examples/rl-tcp/rl-tcp.cc"

# Keep ns-3 pointed at the repo copy so edits here are what gets built
if [ ! -L "$TARGET" ]; then
  ln -sf "$REPO/rl-tcp.cc" "$TARGET"
fi

make -C "$NS3_DIR/cmake-cache" -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc)" rl-tcp
