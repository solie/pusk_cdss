#!/usr/bin/env bash
# Poncol System-1 Fully Static Web Demo Launcher
# Zero Flask / Zero PyTorch required.
set -euo pipefail

PORT="${1:-8085}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================================="
echo "  Poncol System-1: Fully Static Client-Side Web Demo"
echo "  Architecture: ELECTRA-Small 13.5M INT8 ONNX (WASM)"
echo "  Zero Flask / Zero Server-Side Python Dependencies"
echo "=========================================================="
echo "Serving static files from: $DIR"
echo "URL: http://127.0.0.1:${PORT}/index.html"
echo ""
echo "Press Ctrl+C to stop."

# Use Python built-in http.server or Bun
if command -v python3 >/dev/null 2>&1; then
  exec python3 -m http.server "$PORT" --directory "$DIR"
elif command -v bun >/dev/null 2>&1; then
  exec bun x serve "$DIR" -p "$PORT"
else
  echo "Error: Need python3 or bun to serve static files."
  exit 1
fi
